/**
 * @file TrackUtils.hpp
 * @author Mark Krumholz
 * @brief Utility methods for dealing with track files
 * @date 2024-07-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TRACKUTILS_HPP
#define TRACKUTILS_HPP

#include "TrackCommons.hpp"
#include <cstddef>
#include <filesystem>
#include <string>
#include <toml.hpp>
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
     * @brief Find the HDF5 group holding the unique track matching given inputs
     * @param trackName Name of track set
     * @param feh [Fe/H] value
     * @param vvcrit Rotation rate v/vcrit
     * @param afe Value of [alpha/Fe]
     * @param registryName Name of the track registry file
     * @returns The name of the matching group, or an empty string if no
     *          group matches
     * @details
     * Among the groups in the track set's HDF5 file (excluding
     * "masses"), this searches for the one whose feh, vvcrit, and afe
     * attributes all match the input values; a group missing one of
     * these attributes is treated as matching regardless of the
     * corresponding input value. The registry and track set are still
     * located and searched as in findMatchingTracks; only the absence
     * of a matching group results in an empty string being returned.
     */
    auto findTrack(
        const std::string& trackName,
        double feh = 0.0,
        double vvcrit = 0.0,
        double afe = 0.0,
        const std::string& registryName = defaultRegistry)
    -> std::string;

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

    /**
     * @brief Largest amount by which a tabulated star's surface mass fractions may sum above 1
     * @details
     * See checkSurfaceAbundances(): roundoff in published track tables
     * can push X + Y + C + N + O fractionally above 1 (the largest such
     * excess in any track set slug ships is ~6e-5, in PARSEC), but a
     * larger excess at a tabulated point indicates corrupt data.
     */
    inline constexpr double surfaceAbundanceTol = 1e-3;

    /**
     * @brief Check that every tabulated point of a track has physical surface mass fractions
     * @param data A track's data, as read by utils::readDataset2D:
     *   nrow rows (one per time point) of ncol fields, row-major
     * @param nrow Number of rows in data
     * @param ncol Number of fields per row in data
     * @param qtyIdx Column of data holding each FieldIdx quantity, as
     *   built by Tracks2D/Tracks3D (qtyIdx[k] is the column holding
     *   quantity k)
     * @param context Description of the track being checked (e.g. its
     *   file, group, and dataset), for the error message
     * @throws std::runtime_error if any row's X, Y, C, N, or O is not
     *   finite or is negative (no tabulated value in any track set slug
     *   ships is), or their sum exceeds 1 + surfaceAbundanceTol
     * @details
     * Tracks2D and Tracks3D call this on every track they read, so that
     * corrupt track data is reported where it is loaded, rather than
     * silently absorbed by renormalizeSurfaceAbundances() -- which, by
     * design, corrects any excess above 1 in an interpolated value,
     * since interpolating each surface mass fraction independently
     * legitimately overshoots by up to several percent even when every
     * tabulated point is valid.
     */
    void checkSurfaceAbundances(const std::vector<double>& data, std::size_t nrow,
        std::size_t ncol, const std::vector<std::size_t>& qtyIdx, const std::string& context);


} // namespace tracks

#endif // TRACKUTILS_HPP