/**
 * @file TrackUtils.hpp
 * @author Mark Krumholz
 * @brief Utility methods for dealing with track files
 * @date 2024-07-10
 */

#ifndef TRACKUTILS_HPP
#define TRACKUTILS_HPP

#include "../extern/tomlplusplus/toml.hpp"
#include "TrackCommons.hpp"
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace tracks
{
    /**
     * @brief Read a track registry file
     * @param registryName Name of the registry file
     * @returns A pair consisting of a toml table containing the
     *          contents of the registry a the path from which the
     *          table was read.
     * @details
     * This routine checks that registryName exists, is a valid toml
     * file, and contains the minimum set of required entries. It
     * throws a runtime error if any of these conditions are not met.
     */
    auto parseRegistry(const std::string& registryName = defaultRegistry)
    -> std::pair<toml::table, std::filesystem::path>;

    /**
     * @brief Get list of HDF5 groups containing tracks matching given inputs
     * @param trackName Name of track set
     * @param fehMin Minimum [Fe/H] value
     * @param fehMax Maximum [Fe/H] value
     * @param vvcrit Rotation rate v/vcrit
     * @param afe Value of [alpha/Fe]
     * @param nExpand Number of extra tracks to include on each side of
     *                the [fehMin, fehMax] range
     * @param registryName Name of the track registry file
     * @returns A pair consisting of a vector of [Fe/H] values and corresponding group names
     * @details
     * Among the tracks matching vvcrit and afe (see findMatchingTracks
     * for how those are matched), this returns the minimal set of
     * [Fe/H] values whose range encompasses [fehMin, fehMax] -- that
     * is, the largest available [Fe/H] <= fehMin (or the smallest
     * available value, if none is <= fehMin) through the smallest
     * available [Fe/H] >= fehMax (or the largest available value, if
     * none is >= fehMax), inclusive. If nExpand is nonzero, this range
     * is further expanded by nExpand additional tracks on each side,
     * silently limited to the available range if there are not enough
     * tracks to expand by the full amount requested.
     */
    auto findMatchingTracks(
        const std::string& trackName,
        double fehMin,
        double fehMax,
        double vvcrit = 0.0,
        double afe = 0.0,
        unsigned int nExpand = 0,
        const std::string& registryName = defaultRegistry)
    -> std::pair<std::vector<double>, std::vector<std::string>>;

    /**
     * @brief Get the dimensions of a set of tracks
     * @param h5Name Path to HDF5 file containing tracks
     * @param groupName Name of the group within the HDF file
     * @returns A pair containing the number of masses and number of times
     *          in the track set
     */
    auto getTrackSize(
        const std::filesystem::path& h5Name,
        const std::string& groupName)
    -> std::pair<size_t, size_t>;

} // namespace tracks

#endif // TRACKUTILS_HPP