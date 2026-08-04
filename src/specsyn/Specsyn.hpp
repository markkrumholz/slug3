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
#include "../utils/Constants.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

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
         * @param controls Simulation controls this synthesizer reads
         *   its integrator tolerances (intRelTol(), intAbsTol(),
         *   intMaxIter()) and redshift (see wlObs()) from, live, for
         *   the rest of its lifetime -- see controls_'s own comment.
         *   Must outlive this Specsyn.
         */
        explicit Specsyn(const io::SimControls& controls) :
            controls_(controls) { }

        virtual ~Specsyn() = default;

        // Copyable and movable (matching the C++ reference semantics
        // of controls_, which just gets rebound to the same referent
        // on copy), but never actually copied/assigned in practice --
        // every Specsyn is held via unique_ptr (see SimControls's own
        // specsyn_ and SpecsynLibChained's libs_)
        Specsyn(const Specsyn&) = default;
        Specsyn(Specsyn&&) = default;
        auto operator=(const Specsyn&) -> Specsyn& = delete;
        auto operator=(Specsyn&&) -> Specsyn& = delete;

        /**
         * @brief Return the rest-frame wavelength grid
         * @return A const reference to the wavelength grid, in Angstrom
         */
        [[nodiscard]] auto wl() const -> const std::vector<double>& { return wl_; }

        /**
         * @brief Return the observed-frame wavelength grid
         * @return The wavelength grid, in Angstrom, redshifted by
         *   (1 + z), with z read live from controls_
         * @details
         * Defined out-of-line, in Specsyn.cpp -- see that file's own
         * comment for why (same reason as intRelTol()/intAbsTol()/
         * intMaxIter(): controls_.z() needs io::SimControls's complete
         * type, which this header can only forward-declare).
         */
        [[nodiscard]] auto wlObs() const -> std::vector<double>;

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
         * @brief Compute the wavelength-weighted spectrum of a single star, given its mass and isochrone segment
         * @param m Stellar mass, in Msun; must lie within segment's
         *   valid domain (segment.xMin() <= m <= segment.xMax())
         * @param segment A single isochrone segment (one element of
         *   the Isochrone returned by Tracks2D::getIsochrone) to
         *   evaluate at mass m
         * @param feh [Fe/H] value of the segment's isochrone, passed
         *   through to spec() unchanged
         * @return spec(m, segment, feh), multiplied elementwise by
         *   wl(): lambda * dL/dlambda rather than dL/dlambda, in
         *   units of erg/s
         * @details
         * Exists so specCts() can integrate lambda * dL/dlambda
         * instead of dL/dlambda: near the peak of a stellar
         * population's SED, lambda * dL/dlambda is order-unity in
         * reasonable physical units (comparable to the population's
         * total luminosity), whereas dL/dlambda itself spans many
         * orders of magnitude across a wide wavelength grid, making a
         * single absolute tolerance on it effectively meaningless.
         */
        [[nodiscard]] auto specWl(double m, const Segment& segment, double feh) const -> std::vector<double>
        {
            auto result = spec(m, segment, feh);
            for (std::size_t i = 0; i < result.size(); ++i)
            {
                result[i] *= wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has the same size as wl_ (one entry per wavelength), and i is bounded by result.size()
            }
            return result;
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
        ) const -> std::vector<double>;

        /**
         * @brief Return the relative tolerance for PDF integration
         * @return Relative tolerance passed to PDFIntegrator, read live from controls_
         */
        [[nodiscard]] auto intRelTol() const -> double;

        /**
         * @brief Return the absolute tolerance for PDF integration
         * @return Absolute tolerance passed to PDFIntegrator, read live from controls_
         */
        [[nodiscard]] auto intAbsTol() const -> double;

        /**
         * @brief Return the maximum number of evaluations for PDF integration
         * @return Max evaluations passed to PDFIntegrator (0 = unlimited), read live from controls_
         */
        [[nodiscard]] auto intMaxIter() const -> std::size_t;

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
         * in Msun) converted to grams via utils::Msun and G taken from
         * utils::G, so that g comes out in cgs units.
         */
        [[nodiscard]] static auto getSAandLogg(const StarData& props) -> std::pair<double, double>
        {
            constexpr double pi = std::numbers::pi_v<double>;

            const double logL = props[static_cast<size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
            const double logTeff = props[static_cast<size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
            const double mass = props[static_cast<size_t>(tracks::FieldIdx::mass)]; // Msun // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

            const double temperature = std::pow(10.0, logTeff);          // K
            const double luminosity = std::pow(10.0, logL) * utils::Lsun; // erg/s
            const double temperature4 = temperature * temperature * temperature * temperature;
            const double radius = std::sqrt(luminosity / (4.0 * pi * utils::sigmaSB * temperature4)); // cm
            const double area = 4.0 * pi * radius * radius; // cm^2

            const double g = utils::G * mass * utils::Msun / (radius * radius); // cm/s^2
            const double logg = std::log10(g);

            return { area, logg };
        }

        /**
         * @brief Simulation controls this synthesizer reads its integrator tolerances and redshift from
         * @details
         * Read live, not snapshotted, every time specCts() integrates
         * or wlObs() is called (see intRelTol()/intAbsTol()/
         * intMaxIter()/wlObs(), which just forward to the same-named
         * methods on this reference) -- changing controls_'s own
         * tolerances or redshift after this Specsyn is built takes
         * effect immediately, with no need to rebuild the synthesizer.
         * Bound once, at construction, from whichever SimControls
         * actually built this synthesizer (see
         * SimControls::readSpectra()); never reseated afterward, so
         * that SimControls must outlive this Specsyn.
         */
        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy: see this member's own comment for why (controls_'s own tolerances/redshift must be readable live, not snapshotted). This class is only ever used through the same non-copyable, non-movable ownership pattern (unique_ptr in SimControls's own specsyn_) as every other class with a reference member in this codebase (e.g. SpecsynLibNoWind's FeH_/logg_/logTeff_), so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.

        std::vector<double> wl_;     /**< Wavelength grid for the spectral synthesizer, in Angstrom */
    };

} // namespace specsyn

#endif // SPECSYN_HPP
