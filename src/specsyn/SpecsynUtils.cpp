/**
 * @file SpecsynUtils.cpp
 * @author Mark Krumholz
 * @brief Implementations for SpecsynUtils.hpp
 * @date 2026-07-20
 */

#include "SpecsynUtils.hpp"
#include "../utils/MiscUtils.hpp"
#include "hdf5.h"   // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <exception>
#include <filesystem>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace specsyn
{
    static auto getSpectraSetsFromRegistry(toml::table& registry)
    {
        std::vector<std::string> spectraSets;
        if (toml::array* arr = registry.at_path("spectra_sets").as_array())
        {
            arr->for_each([&spectraSets](auto&& el) -> void {
                if constexpr (toml::is_string<decltype(el)>)
                {
                    spectraSets.push_back(std::string(el));
                }
            });
        }
        return std::move(spectraSets);
    }

    // Suppress clang-tidy warnings iun this namespace caused by just including
    // hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
    // that HDF5 wants
    // NOLINTBEGIN(misc-include-cleaner)

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

    // NOLINTEND(misc-include-cleaner)

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

        // Extract list of spectra sets
        auto spectraSets = getSpectraSetsFromRegistry(registry);
        if (spectraSets.empty())
        {
            throw std::runtime_error(
                "parseRegistry: registry " + registryPath.string() +
                " does not contain a valid [spectra_sets] field"
            );
        }

        // For each entry in spectraSets, make sure we have a
        // table entry that has the required "file" and "Fe_H" fields
        for (const auto& ss : spectraSets)
        {
            if (!registry.contains(ss))
            {
                throw std::runtime_error(
                    "parseRegistry: registry " + registryPath.string() +
                    " is missing entry for spectra " + ss
                );
            }
            auto ssEntry = registry.at_path(ss);
            if (!ssEntry.at_path("file") || !ssEntry.at_path("Fe_H"))
            {
                throw std::runtime_error(
                    "parseRegistry: registry " + registryPath.string() +
                    ", entry for spectra " + ss + " is missing required "
                    "'file' or 'Fe_H' fields"
                );
            }
        }

        // If we are here, parse was successful
        return std::make_pair(registry, registryPath);
    }

    auto findMatchingSpectra( // NOLINT(readability-function-cognitive-complexity)
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const double microTurb,
        const double r,
        const std::string& registryName)
    -> std::pair<std::vector<double>, std::vector<std::string>>
    {
        // First parse the registry
        auto [registry, registryPath] = parseRegistry(registryName);

        // Now check the registry for spectra matching the given spectra name
        auto spectraSets = getSpectraSetsFromRegistry(registry);
        auto it = std::ranges::find(spectraSets.begin(), spectraSets.end(), spectraName);
        if (it == spectraSets.end())
        {
            throw std::runtime_error("findMatchingSpectra: no spectra set named " +
                spectraName + " found in spectra registry " + registryName);
        }

        // Get the h5file name for this spectra set from the registry
        const auto h5name =
            registry.at_path(spectraName).at_path("file").value_or(std::string{});

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
            throw std::runtime_error("findMatchingSpectra: unable to open "
                "HDF5 file " + h5path.string());
        }

        // Find the number of top-level links (groups) in the file
        H5G_info_t ginfo{};
        H5Gget_info(file, &ginfo);

        // Loop over every group in the file, collecting the feh value
        // and name of each one whose feh attribute lies in
        // [fehMin, fehMax] and whose afe, cfe, micro, and r attributes
        // all match the input values; a group missing one of these
        // attributes is treated as matching regardless of the
        // corresponding input value. Groups with no feh attribute at
        // all (e.g. "wavelengths", "logg_Teff_grid") are simply
        // skipped.
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

            const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
            if (grp < 0) { continue; }

            const auto fehVal = readScalarAttrIfPresent(grp, "feh");
            bool isMatch = fehVal.has_value() &&
                (*fehVal >= fehMin) && (*fehVal <= fehMax);

            if (isMatch)
            {
                const auto afeVal = readScalarAttrIfPresent(grp, "afe");
                if (afeVal && !utils::approxEqual(*afeVal, afe))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto cfeVal = readScalarAttrIfPresent(grp, "cfe");
                if (cfeVal && !utils::approxEqual(*cfeVal, cfe))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto microVal = readScalarAttrIfPresent(grp, "micro");
                if (microVal && !utils::approxEqual(*microVal, microTurb))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto rVal = readScalarAttrIfPresent(grp, "r");
                if (rVal && !utils::approxEqual(*rVal, r))
                {
                    isMatch = false;
                }
            }

            if (isMatch) { matches.emplace_back(*fehVal, groupName); }

            H5Gclose(grp);
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        // Sort all the matches by feh, from lowest to highest
        std::ranges::sort(matches, {},
            &std::pair<double, std::string>::first);

        // Extract the matches into the output vectors
        std::vector<double> fehOut;
        std::vector<std::string> nameOut;
        fehOut.reserve(matches.size());
        nameOut.reserve(matches.size());
        for (auto& match : matches)
        {
            fehOut.push_back(match.first);
            nameOut.push_back(std::move(match.second));
        }

        return { std::move(fehOut), std::move(nameOut) };
    }

    auto getMicroDefault(
        const std::string& spectraName, const std::string& registryName) -> double
    {
        auto [registry, registryPath] = parseRegistry(registryName);

        auto spectraSets = getSpectraSetsFromRegistry(registry);
        if (std::ranges::find(spectraSets, spectraName) == spectraSets.end())
        {
            throw std::runtime_error("getMicroDefault: no spectra set named " +
                spectraName + " found in spectra registry " + registryName);
        }

        const auto microDefault =
            registry.at_path(spectraName).at_path("micro_default").value<double>();
        if (!microDefault)
        {
            throw std::runtime_error("getMicroDefault: spectra set " + spectraName +
                " in registry " + registryPath.string() + " has no micro_default field");
        }
        return *microDefault;
    }

} // namespace specsyn
