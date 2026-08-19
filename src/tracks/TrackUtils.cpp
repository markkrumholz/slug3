/**
 * @file TrackUtils.cpp
 * @author Mark Krumholz
 * @brief Implementations for TrackUtils.hpp
 * @date 2024-07-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "TrackUtils.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h"   // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace tracks
{
    auto parseRegistry(const std::string& registryName)
    -> std::pair<toml::table, std::filesystem::path>
    {
        [[maybe_unused]] auto [registry, registryPath, trackSets] =
            utils::parseSetRegistry(registryName, "track_sets", "tracks", "parseRegistry");
        return { std::move(registry), std::move(registryPath) };
    }

    auto findMatchingTracks( // NOLINT(readability-function-cognitive-complexity)
        const std::string& trackName,
        const double fehMin,
        const double fehMax,
        const double vvcrit,
        const double afe,
        const unsigned int nExpand,
        const std::string& registryName)
    -> std::pair<std::vector<double>, std::vector<std::string>>
    {
        // First parse the registry
        auto [registry, registryPath] = parseRegistry(registryName);

        // Now check the registry for tracks matching the given track name
        auto trackSets = utils::getStringArrayField(registry, "track_sets");
        auto it = std::ranges::find(trackSets.begin(), trackSets.end(), trackName);
        if (it == trackSets.end())
        {
            throw std::runtime_error("findMatchingTracks: no trackset named " +
                trackName + " found in track registry " + registryName);
        }

        // Get the h5file name for this track set from the registry
        const auto h5name =
            registry.at_path(trackName).at_path("file").value_or(std::string{});

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

        // Loop over every group in the file except "masses", collecting
        // the feh value and name of each one that matches the input
        // vvcrit and afe criteria; groups that don't have a vvcrit or
        // afe attribute are treated as matching regardless of the input
        // vvcrit and afe values. The feh range is deliberately not
        // applied here -- every vvcrit/afe match is collected, and the
        // feh range is instead used below to select a bracketing subset
        // of them.
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

            const auto fehVal = utils::readScalarAttrIfPresent(grp, "feh");
            bool isMatch = fehVal.has_value();

            if (isMatch)
            {
                const auto vvcritVal = utils::readScalarAttrIfPresent(grp, "vvcrit");
                if (vvcritVal && !utils::approxEqual(*vvcritVal, vvcrit))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto afeVal = utils::readScalarAttrIfPresent(grp, "afe");
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

        // Sort all the vvcrit/afe-matching tracks by feh, from lowest
        // to highest
        std::ranges::sort(matches, {},
            &std::pair<double, std::string>::first);

        if (matches.empty())
        {
            return { {}, {} };
        }

        // Find the minimal bracketing range of indices into matches
        // whose feh values encompass [fehMin, fehMax]: from the last
        // index with feh <= fehMin (or index 0, if no feh is that low),
        // through the first index with feh >= fehMax (or the last
        // index, if no feh is that high). std::ranges::upper_bound and
        // lower_bound trip up misc-include-cleaner on some libc++
        // versions (it can't find a header to attribute them to, even
        // with <algorithm> already included), hence the NOLINTs below.
        const auto loBound = std::ranges::upper_bound(matches, fehMin, //NOLINT(misc-include-cleaner)
            {}, &std::pair<double, std::string>::first);
        const size_t loIdx = (loBound == matches.begin()) ? 0 :
            static_cast<size_t>(loBound - matches.begin()) - 1;
        const auto hiBound = std::ranges::lower_bound(matches, fehMax, //NOLINT(misc-include-cleaner)
            {}, &std::pair<double, std::string>::first);
        const size_t hiIdx = (hiBound == matches.end()) ?
            matches.size() - 1 : static_cast<size_t>(hiBound - matches.begin());

        // Expand the bracketing range by nExpand tracks on each side,
        // silently limiting the expansion to the available range
        const auto nExpandSz = static_cast<size_t>(nExpand);
        const size_t loIdxExpanded = (nExpandSz >= loIdx) ? 0 : loIdx - nExpandSz;
        const size_t hiIdxExpanded = std::min(matches.size() - 1, hiIdx + nExpandSz);

        // Extract the selected range of matches into the output vectors
        std::vector<double> fehOut;
        std::vector<std::string> nameOut;
        const size_t nOut = hiIdxExpanded - loIdxExpanded + 1;
        fehOut.reserve(nOut);
        nameOut.reserve(nOut);
        for (size_t idx = loIdxExpanded; idx <= hiIdxExpanded; ++idx)
        {
            fehOut.push_back(matches.at(idx).first);
            nameOut.push_back(std::move(matches.at(idx).second));
        }

        return { std::move(fehOut), std::move(nameOut) };
    }

    auto findTrack( // NOLINT(readability-function-cognitive-complexity); complexity is 26, not refactoring to get rid of an over-by-one!
        const std::string& trackName,
        const double feh,
        const double vvcrit,
        const double afe,
        const std::string& registryName)
    -> std::string
    {
        // First parse the registry
        auto [registry, registryPath] = parseRegistry(registryName);

        // Now check the registry for tracks matching the given track name
        auto trackSets = utils::getStringArrayField(registry, "track_sets");
        auto it = std::ranges::find(trackSets.begin(), trackSets.end(), trackName);
        if (it == trackSets.end())
        {
            throw std::runtime_error("findTrack: no trackset named " +
                trackName + " found in track registry " + registryName);
        }

        // Get the h5file name for this track set from the registry
        const auto h5name =
            registry.at_path(trackName).at_path("file").value_or(std::string{});

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
            throw std::runtime_error("findTrack: unable to open "
                "HDF5 file " + h5path.string());
        }

        // Find the number of top-level links (groups) in the file
        H5G_info_t ginfo{};
        H5Gget_info(file, &ginfo);

        // Loop over every group in the file except "masses", looking
        // for the (unique) one whose feh, vvcrit, and afe attributes
        // all match the input values; a group missing one of these
        // attributes is treated as matching regardless of the
        // corresponding input value.
        std::string result;
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

            bool isMatch = true;

            const auto fehVal = utils::readScalarAttrIfPresent(grp, "feh");
            if (fehVal && !utils::approxEqual(*fehVal, feh))
            {
                isMatch = false;
            }
            if (isMatch)
            {
                const auto vvcritVal = utils::readScalarAttrIfPresent(grp, "vvcrit");
                if (vvcritVal && !utils::approxEqual(*vvcritVal, vvcrit))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto afeVal = utils::readScalarAttrIfPresent(grp, "afe");
                if (afeVal && !utils::approxEqual(*afeVal, afe))
                {
                    isMatch = false;
                }
            }

            H5Gclose(grp);

            if (isMatch)
            {
                result = groupName;
                break;
            }
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        return result;
    }

    auto getTrackSize(
        const std::filesystem::path& h5Name,
        const std::string& groupName)
    -> std::pair<size_t, size_t>
    {
        // Suppress clang-tidy warnings iun this namespace caused by just
        // including hdf5.h, instead of the individual HDF5 headers, since
        // this is the paradigm that HDF5 wants
        // NOLINTBEGIN(misc-include-cleaner)

        const hid_t file = H5Fopen(h5Name.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "getTrackSize: unable to open HDF5 file " + h5Name.string());
        }

        const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
        if (grp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                "getTrackSize: unable to open group " + groupName +
                " in file " + h5Name.string());
        }

        // The number of masses is the length of the masses dataset
        const auto massData = utils::readDataset1D(grp, "masses", "getTrackSize");
        const size_t nmass = massData.size();

        // The number of times is the maximum number of time points
        // among all the tracks in this group, since tracks for
        // different masses need not have the same number of time points
        size_t ntime = 0;
        for (const double m : massData)
        {
            const auto name = std::format("track_m{:.3f}", m);
            ntime = std::max(ntime, utils::dataset2DShape(grp, name, "getTrackSize").first);
        }

        H5Gclose(grp);
        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        return { nmass, ntime };
    }

} // namespace tracks