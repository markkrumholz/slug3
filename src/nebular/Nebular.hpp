/**
 * @file Nebular.hpp
 * @author Mark Krumholz
 * @brief A class to provide nebular emission per ionizing photon from a pre-computed cloudy grid
 * @date 2026-08-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef NEBULAR_HPP
#define NEBULAR_HPP

#include "../tracks/TrackCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <string>
#include <utility>
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

/**
 * @brief A namespace to hold machinery for nebular emission from a pre-computed cloudy grid
 */
namespace nebular
{

    /**
     * @class Nebular
     * @brief Nebular emission line and continuum luminosity per ionizing photon, from a pre-computed cloudy grid
     * @details
     * This class looks up the "quick and dirty" nebular emission grid
     * built by data/tools/cloudy's own three-stage pipeline (see that
     * directory's own scripts) for one track set, and provides the
     * line and continuum luminosity per ionizing photon it holds, as
     * a function of [Fe/H] and (for a cluster) age -- so that slug can
     * add nebular emission to a stellar spectrum without running
     * cloudy itself.
     */
    class Nebular
    {
    public:

        // Shorten mdspan types
        using Grid2D = std::mdspan<double, std::dextents<std::size_t, 2>>; // NOLINT(misc-include-cleaner)
        using Grid3D = std::mdspan<double, std::dextents<std::size_t, 3>>;

        /**
         * @brief Construct a Nebular object
         * @param trackName Name of the track set this nebular grid is for
         * @param wl Simulation wavelength grid the continuum luminosity
         *   grids should ultimately be resampled onto
         * @param simControls Simulation controls (physics and control-flow settings)
         * @param vvcrit Rotation rate v/vcrit of the track set
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this Nebular. trackName and wl are each stored
         * as an owned copy (taken by value and moved from, rather than
         * by const reference, per this project's own enforced
         * modernize-pass-by-value style). This constructor only stores
         * its own arguments; it does not yet load or populate any of
         * this class's own nebular emission grids (lineLumPerQCluster_
         * etc), which are left default-constructed (empty) for now.
         */
        Nebular(
            std::string trackName,
            std::vector<double> wl,
            const io::SimControls& simControls,
            double vvcrit = tracks::defaultVVcrit) :
            trackName_(std::move(trackName)), wl_(std::move(wl)),
            simControls_(simControls), vvcrit_(vvcrit)
        { }

        ~Nebular() = default;

        // The mdspan members below are non-owning views into this
        // object's own backing vectors (lineLumPerQClusterData_ etc);
        // a naive copy or move would leave the copy's/moved-from's own
        // views pointing at the wrong object's data, so both are
        // disallowed for now -- mirrors utils::ThreadVec's own
        // identical reasoning for its own vector-backed member.
        Nebular(const Nebular&) = delete;
        Nebular(Nebular&&) = delete;
        auto operator=(const Nebular&) -> Nebular& = delete;
        auto operator=(Nebular&&) -> Nebular& = delete;

    private:

        std::string trackName_;              /**< Name of the track set this nebular grid is for */
        std::vector<double> wl_;             /**< Simulation wavelength grid */
        const io::SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */
        double vvcrit_;                      /**< Rotation rate v/vcrit of the track set */

        std::vector<std::string> lineLabel_; /**< Label of each nebular emission line */
        std::vector<double> lineWl_;         /**< Wavelength of each nebular emission line */

        /** @brief Cluster line luminosity per ionizing photon, shaped ([Fe/H], age, line) */
        Grid3D lineLumPerQCluster_;
        std::vector<double> lineLumPerQClusterData_; /**< Data holder for lineLumPerQCluster_ */

        /** @brief Galaxy line luminosity per ionizing photon, shaped ([Fe/H], line) */
        Grid2D lineLumPerQGalaxy_;
        std::vector<double> lineLumPerQGalaxyData_; /**< Data holder for lineLumPerQGalaxy_ */

        /** @brief Cluster continuum luminosity per ionizing photon, shaped ([Fe/H], age, wavelength) */
        Grid3D ctmLumPerQCluster_;
        std::vector<double> ctmLumPerQClusterData_; /**< Data holder for ctmLumPerQCluster_ */

        /** @brief Galaxy continuum luminosity per ionizing photon, shaped ([Fe/H], wavelength) */
        Grid2D ctmLumPerQGalaxy_;
        std::vector<double> ctmLumPerQGalaxyData_; /**< Data holder for ctmLumPerQGalaxy_ */
    };

} // namespace nebular

#endif // NEBULAR_HPP
