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
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom
         */
        [[nodiscard]] virtual auto spec(const StarData& props) const
        -> std::vector<double> = 0;

        /**
         * @brief Compute the spectrum of a single star, given its mass and isochrone segment
         * @param m Stellar mass, in Msun; must lie within segment's
         *   valid domain (segment.xMin() <= m <= segment.xMax())
         * @param segment A single isochrone segment (one element of
         *   the Isochrone returned by Tracks2D::getIsochrone) to
         *   evaluate at mass m
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
        [[nodiscard]] auto spec(double m, const Segment& segment) const -> std::vector<double>
        {
            return spec(segment(m));
        }

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
            double mMax
        ) const -> std::vector<double>
        {
            using SpecSegFn = std::vector<double> (Specsyn::*)(double, const Segment&) const;
            const utils::PDFIntegrator integrator(
                imf, static_cast<SpecSegFn>(&Specsyn::spec), static_cast<unsigned>(wl_.size()));

            std::vector<double> result(wl_.size(), 0.0);
            for (const auto& seg : isochrone)
            {
                const double a = std::max(mMin, seg->xMin());
                const double b = std::min(mMax, seg->xMax());
                if (a >= b) { continue; } // empty intersection with [mMin, mMax]

                const auto segResult = integrator.integrate(a, b, this, *seg);
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    result.at(i) += segResult.at(i);
                }
            }
            for (auto& r : result) { r *= mTot; }
            return result;
        }

    protected:

        double z_;                   /**< Redshift */
        std::vector<double> wl_;     /**< Wavelength grid for the spectral synthesizer, in Angstrom */
    };

} // namespace specsyn

#endif // SPECSYN_HPP
