/**
 * @file TrackUtils.hpp
 * @author Mark Krumholz
 * @brief Utility methods for dealing with track files
 * @date 2024-07-10
 */

#ifndef TRACKUTILS_HPP
#define TRACKUTILS_HPP

#include "../tomlplusplus/toml.hpp"
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
    auto parseRegistry(const std::string& registryName)
    -> std::pair<toml::table, std::filesystem::path>;

    /**
     * @brief Get list of HDF5 groups containing tracks matching given inputs
     * @param registryName Name of the track registry file
     * @param trackName Name of track set
     * @param fehMin Minimum [Fe/H] value
     * @param fehMax Maximum [Fe/H] value
     * @param vvcrit Rotation rate v/vcrit
     * @param afe Value of [alpha/Fe] 
     * @returns A pair consisting of a vector of [Fe/H] values and corresponding group names
     */
    auto findMatchingTracks(
        const std::string& registryName,
        const std::string& trackName,
        double fehMin, 
        double fehMax,
        double vvcrit = 0.0,
        double afe = 0.0)
    -> std::pair<std::vector<double>, std::vector<std::string>>;

} // namespace tracks

#endif // TRACKUTILS_HPP