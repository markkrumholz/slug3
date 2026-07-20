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
#include "../utils/PDFIntegrator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_const_cgsm.h> // NOLINT(misc-include-cleaner)
#include <memory>
#include <numbers>
#include <utility>
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

        // Shorten type names: Segment is a single isochrone segment
        // (an element of the Isochrone returned by
        // Tracks2D::getIsochrone -- an isochrone can have more than
        // one disjoint segment for non-monotonic tracks), and
        // StarData is the type returned by evaluating a Segment at a
        // given mass
        using Segment = interp::Interpolator1D<static_cast<size_t>(tracks::FieldIdx::nTrackQty)>;
        using Isochrone = std::vector<std::unique_ptr<Segment>>;
        using StarData = std::array<double,
            static_cast<size_t>(tracks::FieldIdx::nTrackQty)>;

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
         * @param feh [Fe/H] value of the star, needed because it is
         *   not carried by props itself (e.g. by a SpecsynLib, to
         *   locate props in its spectral library's [Fe/H] direction)
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom
         */
        [[nodiscard]] virtual auto spec(const StarData& props, double feh) const
        -> std::vector<double> = 0;

        /**
         * @brief Compute the spectrum of a single star, given its mass and isochrone segment
         * @param m Stellar mass, in Msun; must lie within segment's
         *   valid domain (segment.xMin() <= m <= segment.xMax())
         * @param segment A single isochrone segment (one element of
         *   the Isochrone returned by Tracks2D::getIsochrone) to
         *   evaluate at mass m
         * @param feh [Fe/H] value of the segment's isochrone, passed
         *   through to spec() unchanged
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom
         * @details
         * Evaluates segment at m to get m's stellar properties, and
         * returns spec() of those properties. This overload exists
         * mainly so spec() can be handed to PDFIntegrator, which
         * expects a callable taking the integration variable (here,
         * mass) as its first argument. It takes a single segment,
         * rather than a full Isochrone, so that specCts() can
         * integrate each segment separately over its own valid
         * domain -- an Isochrone as a whole may have gaps where no
         * segment is defined, and pcubature has no way to know to
         * avoid evaluating there.
         */
        [[nodiscard]] auto spec(double m, const Segment& segment, double feh) const -> std::vector<double>
        {
            return spec(segment(m), feh);
        }

        /**
         * @brief Compute the spectrum of a continuously-sampled stellar population
         * @param isochrone The isochrone for the population, as
         *   returned by Tracks2D::getIsochrone
         * @param imf The initial mass function of the population
         * @param mTot Total mass of the population, in Msun
         * @param mMin Minimum stellar mass in the population, in Msun
         * @param mMax Maximum stellar mass in the population, in Msun
         * @param feh [Fe/H] value of the population, passed through
         *   to spec() unchanged
         * @return The specific luminosity of the population, evaluated
         *   on the wavelength grid returned by wl(), in units of
         *   erg/s/Angstrom
         * @details
         * Combines the star-by-star spectral synthesis provided by
         * spec() with the population's IMF, continuously sampled over
         * [mMin, mMax] rather than drawn as a discrete set of stars.
         *
         * isochrone may consist of several disjoint segments (see
         * Tracks2D::getIsochrone), each valid only over its own
         * [segment.xMin(), segment.xMax()] range, with gaps possibly
         * in between where no segment is defined. pcubature has no
         * way to know to avoid evaluating in those gaps, so this
         * integrates each segment separately, over the intersection
         * of its own domain with [mMin, mMax], skipping segments
         * whose domain does not overlap [mMin, mMax] at all, and sums
         * the per-segment results -- this guarantees spec() is only
         * ever evaluated at masses where some segment is actually
         * defined. The summed result is then scaled by mTot.
         */
        [[nodiscard]] virtual auto specCts(
            const Isochrone& isochrone,
            const pdfs::PDF& imf,
            double mTot,
            double mMin,
            double mMax,
            double feh
        ) const -> std::vector<double>
        {
            using SpecSegFn = std::vector<double> (Specsyn::*)(double, const Segment&, double) const;
            const utils::PDFIntegrator integrator(
                imf, static_cast<SpecSegFn>(&Specsyn::spec), static_cast<unsigned>(wl_.size()));

            std::vector<double> result(wl_.size(), 0.0);
            for (const auto& seg : isochrone)
            {
                const double a = std::max(mMin, seg->xMin());
                const double b = std::min(mMax, seg->xMax());
                if (a >= b) { continue; } // empty intersection with [mMin, mMax]

                const auto segResult = integrator.integrate(a, b, this, *seg, feh);
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    result.at(i) += segResult.at(i);
                }
            }
            for (auto& r : result) { r *= mTot; }
            return result;
        }

    protected:

        /**
         * @brief Compute a star's surface area and log(g) from its stellar properties
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @return A pair containing the star's surface area, in cm^2,
         *   and log10(g), with g in cgs units (cm/s^2)
         * @details
         * The stellar radius (and hence surface area) is derived from
         * the star's luminosity and effective temperature via the
         * Stefan-Boltzmann law, L = 4 pi R^2 sigma T^4. log(g) is then
         * computed as log10(G M / R^2), with the stellar mass (given
         * in Msun) converted to grams and G taken from GSL's cgs unit
         * system, so that g comes out in cgs units.
         */
        [[nodiscard]] static auto getSAandLogg(const StarData& props) -> std::pair<double, double>
        {
            constexpr double pi = std::numbers::pi_v<double>;
            constexpr double stefanBoltzmann = GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT;
            constexpr double gravConst = GSL_CONST_CGSM_GRAVITATIONAL_CONSTANT;
            constexpr double solarMass = GSL_CONST_CGSM_SOLAR_MASS;
            constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value

            const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
            const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            const double mass = props[static_cast<size_t>(tracks::FieldIdx::mass)]; // Msun // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

            const double temperature = std::pow(10.0, logTeff);               // K
            const double luminosity = std::pow(10.0, logL) * solarLuminosity; // erg/s
            const double temperature4 = temperature * temperature * temperature * temperature;
            const double radius = std::sqrt(luminosity / (4.0 * pi * stefanBoltzmann * temperature4)); // cm
            const double area = 4.0 * pi * radius * radius; // cm^2

            const double g = gravConst * mass * solarMass / (radius * radius); // cm/s^2
            const double logg = std::log10(g);

            return { area, logg };
        }

        double z_;                   /**< Redshift */
        std::vector<double> wl_;     /**< Wavelength grid for the spectral synthesizer, in Angstrom */
    };

} // namespace specsyn

#endif // SPECSYN_HPP
