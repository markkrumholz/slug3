/**
 * @file Specsyn.hpp
 * @author Mark Krumholz
 * @brief Defines the common interface for stellar spectral synthesis
 * @date 2026-07-18
 */

#ifndef SPECSYN_HPP
#define SPECSYN_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../pdfs/PDF.hpp"
#include "../tracks/TrackCommons.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace specsyn
{

    /**
     * @class Specsyn
     * @brief A base class defining a common interface for spectral synthesis
     * @details
     * This class defines a common interface to spectral synthesis
     * classes, whose purpose is to take as input a set of stellar
     * parameters produced by interpolation on the tracks, and return
     * as output stellar spectra on a pre-defined wavelength grid.
     */
    class Specsyn
    {
    public:

        // Shorten type name: the isochrone type returned by
        // Tracks2D::getIsochrone
        using Isochrone = std::vector<std::unique_ptr<
            interp::Interpolator1D<static_cast<size_t>(tracks::FieldIdx::nTrackQty)>>>;

        /**
         * @brief Construct a Specsyn
         * @param z The redshift; defaults to zero
         */
        explicit Specsyn(double z = 0.0) : z_(z) { }

        virtual ~Specsyn() = default;
        Specsyn(const Specsyn&) = default;
        Specsyn(Specsyn&&) = default;
        auto operator=(const Specsyn&) -> Specsyn& = default;
        auto operator=(Specsyn&&) -> Specsyn& = default;

        /**
         * @brief Return the rest-frame wavelength grid
         * @return A const reference to the wavelength grid, in Angstrom
         */
        [[nodiscard]] auto wl() const -> const std::vector<double>& { return wl_; }

        /**
         * @brief Return the observed-frame wavelength grid
         * @return The wavelength grid, in Angstrom, redshifted by (1 + z)
         */
        [[nodiscard]] auto wlObs() const -> std::vector<double>
        {
            std::vector<double> wlObs(wl_.size());
            std::ranges::transform(wl_, wlObs.begin(),
                [this](const double wl) -> double { return wl * (1.0 + z_); });
            return wlObs;
        }

        /**
         * @brief Compute the spectrum of a single star
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom
         */
        [[nodiscard]] virtual auto spec(
            const std::array<double, static_cast<size_t>(tracks::FieldIdx::nTrackQty)>& props
        ) const -> std::vector<double> = 0;

        /**
         * @brief Compute the spectrum of a continuously-sampled stellar population
         * @param isochrone The isochrone for the population, as
         *   returned by Tracks2D::getIsochrone
         * @param imf The initial mass function of the population
         * @param mTot Total mass of the population, in Msun
         * @param mMin Minimum stellar mass in the population, in Msun
         * @param mMax Maximum stellar mass in the population, in Msun
         * @return The specific luminosity of the population, evaluated
         *   on the wavelength grid returned by wl(), in units of
         *   erg/s/Angstrom
         * @details
         * Combines the star-by-star spectral synthesis provided by
         * spec() with the population's IMF, continuously sampled over
         * [mMin, mMax] rather than drawn as a discrete set of stars.
         */
        [[nodiscard]] virtual auto specCts(
            const Isochrone& /*isochrone*/,
            const pdfs::PDF& /*imf*/,
            double /*mTot*/,
            double /*mMin*/,
            double /*mMax*/
        ) const -> std::vector<double>
        {
            return {};
        }

    protected:

        double z_;                   /**< Redshift */
        std::vector<double> wl_;     /**< Wavelength grid for the spectral synthesizer, in Angstrom */
    };

} // namespace specsyn

#endif // SPECSYN_HPP
