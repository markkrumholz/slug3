/**
 * @file TrackUtils.cpp
 * @author Mark Krumholz
 * @brief Implementations for TrackUtils.hpp
 * @date 2024-07-10
 */

#include "TrackUtils.hpp"
#include "../tomlplusplus/toml.hpp"
#include "../utils/MiscUtils.hpp"
#include "hdf5.h"   // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tracks
{
    static auto getTrackSetsFromRegistry(toml::table& registry)
    {
        std::vector<std::string> trackSets;
        if (toml::array* arr = registry["track_sets"].as_array())
        {
            arr->for_each([&trackSets](auto&& el) {
                if constexpr (toml::is_string<decltype(el)>)
                {
                    trackSets.push_back(std::string(el));
                }
            });
        }
        return std::move(trackSets);
    }

    /**
     * @brief Read a scalar double attribute from an HDF5 group, if present
     * @param grp Handle to the group
     * @param name Name of the attribute
     * @returns The attribute's value, or an empty optional if the group
     *   has no attribute of that name
     */
    static auto readScalarAttrIfPresent(const hid_t grp,
        const std::string& name) -> std::optional<double>
    {
        if (H5Aexists(grp, name.c_str()) <= 0) { return std::nullopt; }
        const hid_t attr = H5Aopen(grp, name.c_str(), H5P_DEFAULT);
        if (attr < 0) { return std::nullopt; }
        double value = 0.0;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
        return value;
    }

    auto parseRegistry(const std::string& registryName)
    -> std::pair<toml::table, std::filesystem::path>
    {
        // Validate that registry file exists
        auto registryPath = utils::getFilePath(registryName);
        if (registryPath.empty())
        {
            throw std::runtime_error(
                "parseRegistry: registry " + registryName + 
                " not found");
        }

        // Validate that registry file parses successfully
        toml::table registry;
        try
        {
            registry = toml::parse_file(registryPath.string());
        }
        catch(const std::exception& e)
        {
            throw std::runtime_error(
                "parseRegistry: registry " + registryPath.string() + 
                " is not a valid toml file");
        }

        // Extract list of track sets
        auto trackSets = getTrackSetsFromRegistry(registry);
        if (trackSets.empty())        
        {
            throw std::runtime_error(
                "parseRegistry: registry " + registryPath.string() +
                " does not contain a valid [track_sets] field"
            );
        }

        // For each entry in trackSets, make sure we have a
        // table entry that has the required "file" and "Fe_H" fields
        for (const auto& ts : trackSets)
        {
            if (!registry.contains(ts))
            {
                throw std::runtime_error(
                    "parseRegistry: registry " + registryPath.string() +
                    " is missing entry for tracks " + ts
                );
            }
            auto tsEntry = registry[ts];
            if (!tsEntry["file"] || !tsEntry["Fe_H"])
            {
                throw std::runtime_error(
                    "parseRegistry: registry " + registryPath.string() +
                    ", entry for tracks " + ts + " is missing required "
                    "'file' or 'Fe_H' fields"
                );
            }
        }

        // If we are here, parse was successful
        return std::make_pair(registry, registryPath);
    }

    auto findMatchingTracks(
        const std::string& registryName,
        const std::string& trackName,
        const double fehMin,
        const double fehMax,
        const double vvcrit,
        const double afe)
    -> std::pair<std::vector<double>, std::vector<std::string>>
    {
        // First parse the registry
        auto [registry, registryPath] = parseRegistry(registryName);

        // Now check the registry for tracks matching the given track name
        auto trackSets = getTrackSetsFromRegistry(registry);
        auto it = std::ranges::find(trackSets.begin(), trackSets.end(), trackName);
        if (it == trackSets.end())
        {
            throw std::runtime_error("findMatchingTracks: no trackset named " +
                trackName + " found in track registry " + trackName);
        }

        // Get the h5file name for this track set from the registry
        const auto h5name =
            registry[trackName]["file"].value_or(std::string{});

        // The h5 file name is given relative to the directory
        // containing the registry file itself
        const auto h5path = registryPath.parent_path() / h5name;

        // Suppress clang-tidy warnings iun this namespace caused by just including
        // hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
        // that HDF5 wants
        // NOLINTBEGIN(misc-include-cleaner)

        // Open the HDF5 file
        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error("findMatchingTracks: unable to open "
                "HDF5 file " + h5path.string());
        }

        // Find the number of top-level links (groups) in the file
        H5G_info_t ginfo{};
        H5Gget_info(file, &ginfo);

        // Loop over every group in the file except "masses", checking
        // each against the input feh/vvcrit/afe criteria; groups that
        // don't have a vvcrit or afe attribute are treated as matching
        // regardless of the input vvcrit and afe values
        std::vector<std::pair<double, std::string>> matches;
        for (hsize_t i = 0; i < ginfo.nlinks; ++i)
        {
            const auto nameLen = H5Lget_name_by_idx(file, ".",
                H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
            if (nameLen < 0) { continue; }
            std::vector<char> nameBuf(static_cast<size_t>(nameLen) + 1);
            H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                nameBuf.data(), nameBuf.size(), H5P_DEFAULT);
            const std::string groupName(nameBuf.data());

            if (groupName == "masses") { continue; }

            const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
            if (grp < 0) { continue; }

            const auto fehVal = readScalarAttrIfPresent(grp, "feh");
            bool isMatch = fehVal.has_value() &&
                *fehVal >= fehMin && *fehVal <= fehMax;

            if (isMatch)
            {
                const auto vvcritVal = readScalarAttrIfPresent(grp, "vvcrit");
                if (vvcritVal && !utils::approxEqual(*vvcritVal, vvcrit))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto afeVal = readScalarAttrIfPresent(grp, "afe");
                if (afeVal && !utils::approxEqual(*afeVal, afe))
                {
                    isMatch = false;
                }
            }

            if (isMatch) { matches.emplace_back(*fehVal, groupName); }

            H5Gclose(grp);
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)
        
        // Sort the matches by feh, from lowest to highest, then split
        // them back out into separate vectors to return
        std::ranges::sort(matches, {},
            &std::pair<double, std::string>::first);
        std::vector<double> fehOut;
        std::vector<std::string> nameOut;
        fehOut.reserve(matches.size());
        nameOut.reserve(matches.size());
        for (auto& [feh, name] : matches)
        {
            fehOut.push_back(feh);
            nameOut.push_back(std::move(name));
        }

        return { std::move(fehOut), std::move(nameOut) };
    }


} // namespace tracks