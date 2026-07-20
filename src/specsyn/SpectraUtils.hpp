/**
 * @file SpectraUtils.hpp
 * @author Mark Krumholz
 * @brief Utility methods for dealing with spectral library files
 * @date 2026-07-20
 */

#ifndef SPECTRAUTILS_HPP
#define SPECTRAUTILS_HPP

#include "../tracks/TrackCommons.hpp"
#include "SpectraCommons.hpp"
#include <filesystem>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace specsyn
{
    /**
     * @brief Read a spectral library registry file
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
     * @brief Get list of HDF5 groups containing spectra matching given inputs
     * @param spectraName Name of the spectral model
     * @param fehMin Minimum [Fe/H] value
     * @param fehMax Maximum [Fe/H] value
     * @param afe Value of [alpha/Fe]
     * @param cfe Value of [C/Fe]
     * @param microTurb Microturbulent velocity, in km/s
     * @param r Spectral resolution
     * @param registryName Name of the spectral library registry file
     * @returns A pair consisting of a vector of [Fe/H] values and corresponding group names
     * @details
     * Among the groups in the spectral model's HDF5 file, this
     * returns those whose feh attribute lies in [fehMin, fehMax] and
     * whose afe, cfe, microTurb (stored as "micro" in the file), and
     * r attributes all match the corresponding input values; a group
     * missing one of the afe, cfe, micro, or r attributes is treated
     * as matching regardless of the corresponding input value. The
     * returned vectors are sorted by increasing [Fe/H].
     */
    auto findMatchingSpectra(
        const std::string& spectraName,
        double fehMin,
        double fehMax,
        double afe = tracks::defaultAFe,
        double cfe = defaultCFe,
        double microTurb = defaultMicroTurb,
        double r = defaultR,
        const std::string& registryName = defaultRegistry)
    -> std::pair<std::vector<double>, std::vector<std::string>>;

} // namespace specsyn

#endif // SPECTRAUTILS_HPP
