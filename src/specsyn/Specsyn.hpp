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
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
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
         * @brief Return the minimum [Fe/H] this synthesizer has real spectral data for
         * @return -infinity by default (no restriction) -- overridden
         *   by SpecsynLibNoWind/SpecsynLibWR (their own FeH_.front())
         * @details
         * Unlike getMin() on a pdfs::PDF, this describes a hard data
         * boundary, not a probability domain: a synthesizer with no
         * [Fe/H] axis at all (SpecsynBlackbody, SpecsynLibWD) has
         * nothing to restrict, so the base class default of -infinity
         * (paired with fehMax()'s own +infinity) leaves any feh
         * unclamped. Exists for SpecsynLibChained's own benefit, which
         * scans every library it chains together via this (see its own
         * fehMin_ member) to build a per-GridType [Fe/H] range,
         * ultimately used by its own specForIntegration() override --
         * see that function's own comment for why a single library's
         * own fehMin()/fehMax() aren't clamped against directly by
         * continuousSpecIntegrand() (which doesn't, in general, know
         * which concrete Specsyn it's talking to).
         */
        [[nodiscard]] virtual auto fehMin() const -> double
        { return -std::numeric_limits<double>::infinity(); }

        /**
         * @brief Return the maximum [Fe/H] this synthesizer has real spectral data for
         * @return +infinity by default (no restriction) -- see
         *   fehMin()'s own comment for the overriding classes and full
         *   rationale, which applies here symmetrically
         */
        [[nodiscard]] virtual auto fehMax() const -> double
        { return std::numeric_limits<double>::infinity(); }

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
         * @brief Compute a star's spectrum, forcing a result even if it falls outside this Specsyn's own domain
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star, needed because it is
         *   not carried by props itself
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom --
         *   unlike spec(), never a size-0 vector
         * @throws std::runtime_error if no spectrum can be produced
         *   for this star even by forcing
         * @details
         * Unlike spec(), which may return an empty vector (depending
         * on OOBPolicy) for a star outside this Specsyn's own domain,
         * specForce() always either returns a genuine spectrum or
         * throws. Used by SpecsynLibChained as a last resort on the
         * last library in a star's own type-specific chain, once
         * every library in that chain (tried via ordinary spec())
         * has failed to produce one.
         *
         * The default implementation here just calls spec() and
         * throws if it comes back empty -- appropriate for a Specsyn
         * (like SpecsynLibWR) whose own spec() already forces a
         * match internally (e.g. by moving a star to the nearest
         * populated grid point along one axis) rather than ever
         * returning empty for a star within its own domain.
         * SpecsynLibNoWind and SpecsynLibWD override this with a
         * real forcing mechanism of their own (moving log(g) to the
         * nearest populated grid value), since their own spec() can
         * genuinely return empty for a star whose log(g) falls in a
         * gap ordinary interpolation can't bridge.
         */
        [[nodiscard]] virtual auto specForce(const StarData& props, double feh) const
        -> std::vector<double>
        {
            auto result = spec(props, feh);
            if (result.empty())
            {
                throw std::runtime_error(
                    "Specsyn::specForce: unable to force a spectrum for this star");
            }
            return result;
        }

        /**
         * @brief Compute a star's spectrum for Specsyn::continuousSpecIntegrand()'s own benefit
         * @param props See spec()'s own props parameter
         * @param feh [Fe/H] value of the star -- unlike spec()'s/
         *   specForce()'s own feh, not necessarily guaranteed to fall
         *   within this synthesizer's own real data coverage; see this
         *   function's own @details
         * @return The star's spectrum -- see spec()'s own return value
         * @details
         * The default implementation here just calls spec(props, feh)
         * unchanged -- correct for every Specsyn with a single, uniform
         * notion of its own [Fe/H] coverage (SpecsynBlackbody,
         * SpecsynLibWD, and a standalone SpecsynLibNoWind/SpecsynLibWR
         * alike), which is exactly what spec()/specForce() already
         * handle via their own existing OOBPolicy. Overridden only by
         * SpecsynLibChained, whose own [Fe/H] coverage is not uniform
         * (it varies by which chained library, of possibly several
         * different real coverages, a given star's own GridType
         * dispatches to) -- see its own override's comment for why
         * that needs a real clamp, and why this exists as a separate
         * function from spec()/specForce() rather than changing their
         * own behavior: [Fe/H] deliberately has no forcing/rescue
         * mechanism there (unlike log(Teff)/log(g)), since a genuinely
         * out-of-range [Fe/H] request reflects a real physical
         * difference no nearest-neighbor substitution can paper over
         * -- a distinction this function exists specifically to
         * preserve for every caller except
         * Specsyn::continuousSpecIntegrand() (via specWlForIntegration()
         * below), which alone needs the tracks' own [Fe/H] grid
         * (padded beyond any single library's own real coverage, for
         * interpolation purposes -- see Specsyn::specCtsHelper()'s own
         * comment) to still produce a usable spectrum.
         */
        [[nodiscard]] virtual auto specForIntegration(const StarData& props, double feh) const
        -> std::vector<double>
        { return spec(props, feh); }

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
         * @brief Compute the wavelength-weighted spectrum of a single star, for Specsyn::continuousSpecIntegrand()'s own benefit
         * @param props See spec()'s own props parameter
         * @param feh See specForIntegration()'s own feh parameter --
         *   unlike specWl()'s own feh, not necessarily within this
         *   synthesizer's own real data coverage
         * @return specForIntegration(props, feh), multiplied
         *   elementwise by wl() -- see specWl()'s own return value
         * @details
         * Identical to specWl(), except taking a star's properties
         * directly (rather than a mass and an isochrone segment to
         * evaluate) and calling specForIntegration() in place of
         * spec() -- see specForIntegration()'s own comment for why
         * Specsyn::continuousSpecIntegrand() needs this, rather than
         * specWl() itself.
         */
        [[nodiscard]] auto specWlForIntegration(const StarData& props, double feh) const -> std::vector<double>
        {
            auto result = specForIntegration(props, feh);
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
         * @brief Compute the integrated spectrum of a continuously-formed stellar population
         * @param sfr Star formation rate, in Msun/yr, as a function of
         *   time (see io::SimControls::sfr()) -- unlike a strictly
         *   normalized probability density, integrating this over a
         *   time range gives the actual stellar mass (in Msun) formed
         *   over that range, matching how io::SimControls::sfr() is
         *   already used elsewhere (e.g. Galaxy::advance())
         * @param imf The initial mass function of the population
         * @param fehDist The [Fe/H] distribution of the population
         * @param curTime The current time, in yr; star formation is
         *   integrated from 0 to curTime
         * @param fCluster The fraction of star-forming mass treated
         *   stochastically, in individual Cluster objects (see
         *   io::SimControls::fCluster()); the result is scaled by
         *   (1 - fCluster), the remaining continuously-treated share
         * @return The specific luminosity of the continuously-formed
         *   population, evaluated on the wavelength grid returned by
         *   wl(), in units of erg/s/Angstrom
         * @details
         * Computes the integrated spectrum from a stellar population
         * that is continuously distributed in age from 0 to curTime,
         * in mass over imf, and in metallicity over fehDist -- unlike
         * the other specCts() overload, which integrates a single
         * fixed-age isochrone over mass alone, this one integrates
         * over age (equivalently, formation time) and metallicity as
         * well, so a different isochrone applies at different points
         * in the integration domain, computed on demand -- see
         * continuousSpecIntegrand()'s own comment for how.
         *
         * Called by Galaxy::computeSpec() when fCluster() < 1, to
         * cover the star-forming mass Galaxy's own individual Cluster
         * objects don't -- see io::SimControls::fCluster()'s own
         * comment.
         *
         * Structured as a pair of *nested* 1D integrals, not a single
         * joint 2D (age, mass) one: a 1D utils::PDFIntegrator over age
         * alone (CubatureMethod::hAdaptive, PDFIntegrator's own default),
         * whose own integrand -- continuousSpecIntegrand() -- builds
         * the isochrone for its own age on demand and immediately hands
         * it to specCtsImpl() to perform a complete inner 1D integral
         * over mass, reusing the same proven mass integrator the
         * isochrone specCts() overload itself uses. An earlier design
         * instead integrated (age, mass) jointly as a single 2D domain;
         * that domain has an internal, curved live/dead-mass boundary
         * (wherever the shortest-lived stars at a given age reach the
         * end of their own lives) that 2D adaptive cubature -- h- or
         * p-adaptive alike -- resolves very poorly once curTime exceeds
         * those stars' own lifetimes, producing a sharp, empirically
         * observed cost cliff (some real, permanent-test output times
         * converging in seconds, others not converging within minutes).
         * Nesting the integrals instead removes that boundary from the
         * integration domain entirely: each age's own isochrone already
         * encodes exactly which masses are alive, as ordinary segment
         * edges specCtsImpl() naturally integrates up to and no
         * further -- see continuousSpecIntegrand()'s own comment.
         *
         * If fehDist is degenerate (fehDist.getMin() == fehDist.getMax(),
         * a single metallicity shared by every star), runs this nested
         * integral once. Otherwise, rather than folding [Fe/H] into
         * either integral as a further continuous dimension, integrates
         * over [Fe/H] via a separate mechanism entirely -- see
         * specCtsHelper()'s own comment for why -- running the same
         * nested (age, mass) integral once at every [Fe/H] grid point
         * the tracks are actually defined at (controls_.tracks().feH()),
         * then interpolating and integrating those discrete results
         * over [Fe/H] with interp::Interpolator1D.
         *
         * The time dimension is integrated as age instead, via a
         * pdfs::PDFReflect view of sfr pivoted at curTime / 2 (so
         * reflect(x) = curTime - x, i.e. this coordinate already is
         * age), and log-transformed whenever curTime exceeds
         * min(1e4 yr, 1e-3 * curTime) -- see specCtsHelper()'s own
         * comment for why: the luminosity this integral is dominated by
         * (young, hot, blue stars) is concentrated in a narrow sliver
         * of age near 0, which a linear axis resolves increasingly
         * poorly as curTime grows.
         *
         * As with the other specCts() overload, the actual quantity
         * integrated is lambda * dL/dlambda (via continuousSpecIntegrand(),
         * which mirrors specWl()'s own multiplication), undone by
         * dividing back out by wl_ elementwise once the integral
         * itself is complete, before scaling by (1 - fCluster).
         *
         * A thin wrapper over specCtsHelper() (computeLbol = false) --
         * see specAndLbolCts() for the sibling that also returns the
         * same population's bolometric luminosity, computed alongside
         * the spectrum in the same integral rather than a second one.
         */
        [[nodiscard]] auto specCts(
            const pdfs::PDF& sfr,
            const pdfs::PDF& imf,
            const pdfs::PDF& fehDist,
            double curTime,
            double fCluster
        ) const -> std::vector<double>;

        /**
         * @brief Compute the integrated spectrum and bolometric luminosity of a continuously-formed stellar population, together
         * @param sfr See specCts()'s own sfr parameter
         * @param imf See specCts()'s own imf parameter
         * @param fehDist See specCts()'s own fehDist parameter
         * @param curTime See specCts()'s own curTime parameter
         * @param fCluster See specCts()'s own fCluster parameter
         * @return A pair of (1) the same specific luminosity specCts()
         *   itself returns, and (2) the population's own bolometric
         *   luminosity, in Lsun (matching Cluster::lbol()'s own units)
         * @details
         * Identical to specCts() in every respect but what it returns:
         * calls specCtsHelper() with computeLbol = true, so the same
         * PDFIntegratorND evaluation -- the expensive part of which is
         * building an isochrone at every distinct (age, feh) the
         * integral visits, not the handful of extra scalar quantities
         * integrated alongside the spectrum at each of those isochrones
         * -- also integrates the population's bolometric luminosity,
         * rather than requiring a second, separate integral (sharing
         * none of the same isochrones) to get it. See
         * continuousSpecIntegrand()'s own comment for exactly how the
         * extra quantity is folded into the integrand itself.
         *
         * Used by Galaxy::computeSpec() in place of specCts() itself
         * when io::SimControls::computeLbol() is true, so that a
         * caller requesting both a spectrum and Lbol together (the
         * common case) never pays for isochrone construction twice.
         */
        [[nodiscard]] auto specAndLbolCts(
            const pdfs::PDF& sfr,
            const pdfs::PDF& imf,
            const pdfs::PDF& fehDist,
            double curTime,
            double fCluster
        ) const -> std::pair<std::vector<double>, double>;

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

    private:

        /**
         * @brief Shared implementation behind both specCts() overloads and continuousSpecIntegrand()
         * @param isochrone See the isochrone specCts() overload's own isochrone parameter
         * @param imf See specCts()'s own imf parameter
         * @param mTot See the isochrone specCts() overload's own mTot
         *   parameter -- ignored (no scaling applied) when
         *   forIntegration is true, since continuousSpecIntegrand()'s
         *   own per-age-point result is weighted entirely by
         *   specCtsHelper()'s outer sfrAge PDF, not by any mass total
         *   of its own
         * @param mMin See the isochrone specCts() overload's own mMin parameter
         * @param mMax See the isochrone specCts() overload's own mMax parameter
         * @param feh See the isochrone specCts() overload's own feh parameter
         * @param forIntegration If false, evaluates each star via
         *   specWl() (ordinary spec(), with no [Fe/H] rescue) and, once
         *   every segment's own contribution is summed, undoes specWl()'s
         *   lambda weighting and scales by mTot -- exactly the isochrone
         *   specCts() overload's own long-standing behavior, unchanged.
         *   If true, evaluates each star via specWlForIntegration()
         *   instead (see its own comment for why: the [Fe/H] this is
         *   called at may fall outside a real spectral library's own
         *   tabulated coverage, at the tracks' padded grid points --
         *   see specCtsHelper()'s own comment), and skips the final
         *   mTot/wl_ step entirely, leaving the result in the same
         *   lambda * dL/dlambda form specCtsHelper()'s own final
         *   division loop expects to undo itself, once, after its own
         *   age integral is complete.
         * @param computeLbol Whether to also integrate each segment's
         *   own bolometric luminosity contribution alongside its
         *   spectrum, appending it as one further element at index
         *   wl_.size() of the returned vector -- see
         *   continuousSpecIntegrand()'s own comment for the exact
         *   units and rationale (erg/s, not Lsun). Only ever true
         *   together with forIntegration: the isochrone specCts()
         *   overload never requests it, relying instead on Cluster's
         *   own separate lbolStar-based integral for its own,
         *   unrelated Lbol needs.
         * @return If computeLbol is false, a vector of wl_.size()
         *   values (lambda * dL/dlambda if forIntegration, dL/dlambda
         *   scaled by mTot otherwise -- see forIntegration's own
         *   comment). If computeLbol is true, that same vector with
         *   one further element appended: the integrated bolometric
         *   luminosity, in erg/s.
         * @details
         * As with the isochrone specCts() overload this generalizes,
         * integrates each isochrone segment separately, over the
         * intersection of its own domain with [mMin, mMax], skipping
         * segments whose domain does not overlap [mMin, mMax] at all,
         * and sums the per-segment results -- see that overload's own
         * comment for why (an isochrone may have gaps pcubature has no
         * way to know to avoid). Uses CubatureMethod::pAdaptive in
         * linear mass space, exactly as that overload's own benchmarked
         * choice -- see its comment -- since the per-star evaluation
         * cost this integrates over is the same either way, only which
         * function (specWl() vs specWlForIntegration()) performs it
         * differs.
         */
        [[nodiscard]] auto specCtsImpl(
            const Isochrone& isochrone,
            const pdfs::PDF& imf,
            double mTot,
            double mMin,
            double mMax,
            double feh,
            bool forIntegration,
            bool computeLbol
        ) const -> std::vector<double>;

        /**
         * @brief Shared implementation behind specCts()/specAndLbolCts()'s continuous-population overloads
         * @param sfr See specCts()'s own sfr parameter
         * @param imf See specCts()'s own imf parameter
         * @param fehDist See specCts()'s own fehDist parameter
         * @param curTime See specCts()'s own curTime parameter
         * @param fCluster See specCts()'s own fCluster parameter
         * @param computeLbol Whether to also integrate the population's
         *   bolometric luminosity alongside its spectrum
         * @return If computeLbol is false, exactly what specCts()
         *   itself returns. If computeLbol is true, that same result
         *   with one extra element appended, at index wl_.size() (see
         *   continuousSpecIntegrand()'s own comment for how it gets
         *   there): the population's own bolometric luminosity, in
         *   Lsun -- specAndLbolCts() is the one that splits this back
         *   out into its own pair.
         * @details
         * Everything specCts()'s own comment describes -- the
         * degenerate-vs-general [Fe/H] handling, the nested-1D (age
         * then mass) integral structure, the reflected-and-log-transformed
         * age coordinate, the lambda * dL/dlambda convention -- applies
         * here unchanged; computeLbol only changes nInt (wl_.size(),
         * or wl_.size() + 1) and whether continuousSpecIntegrand() is
         * asked to also fill in that extra element. The trailing Lbol
         * element, when present, is a genuine luminosity, not a
         * lambda-weighted flux, so it is excluded from this function's
         * own dL/dlambda-recovering division by wl_ -- only scaled by
         * (1 - fCluster), same as every other element. It does,
         * however, get one further unit conversion of its own:
         * continuousSpecIntegrand() reports it in erg/s, to stay on the
         * same absolute scale as the spectral elements this function's
         * own reqAbsError applies to (see its own comment for why), so
         * this function divides it by utils::Lsun before returning,
         * converting it to the Lsun Cluster::lbol() itself uses.
         *
         * In the general (non-degenerate fehDist) case, [Fe/H] is
         * integrated by running the same nested (age, mass) integral
         * once at every grid point in controls_.tracks().feH() -- rather than
         * folding it into the cubature as a third continuous
         * dimension. This was a deliberate change from an earlier,
         * simpler design that did use a joint 3D (feh, age, mass)
         * pAdaptive integral: that design was found, on the real,
         * permanent slow test (7 output times, a real non-degenerate
         * [Fe/H] distribution, the full nWl-element spectrum as the
         * integrand's vector-valued output), to grow memory without
         * bound at most output times, in some cases exceeding 6 GB in
         * under a minute. The cause is intrinsic to cubature's own
         * pAdaptive algorithm (see src/extern/cubature/pcubature.c):
         * unlike hAdaptive, which subdivides the domain into
         * independent boxes of bounded cost, pAdaptive is a single
         * global tensor-product Clenshaw-Curtis rule that increases
         * per-dimension resolution and permanently caches every
         * function value it has ever computed, at every resolution
         * level, until the whole integral converges. The size of one
         * "add another resolution level" step scales as fdim (here,
         * wl_.size(), often ~1e3) times the product of the *other*
         * dimensions' current resolutions -- with three genuinely
         * non-trivial dimensions (feh, age, mass) all needing
         * refinement simultaneously, a single such step can require a
         * multi-GB allocation. This risk was actually already latent
         * with only 2 dimensions log-transformed correctly, but with
         * only 2 dimensions ever refining simultaneously it stayed
         * small enough in practice never to be observed.
         *
         * Decomposing the [Fe/H] direction into a discrete grid instead
         * turns that single fragile 3D integral into nFeh independent
         * nested (age, mass) integrals -- each identical in character
         * and cost to the already-robust degenerate-fehDist case above --
         * plus cheap post-hoc 1D interpolation over [Fe/H]. Total cost
         * is therefore nFeh times the cost of one such integral, plus
         * interp::Interpolator1D's own small overhead; nFeh is
         * typically modest (e.g. 7, for the real permanent slow test's
         * own [Fe/H] distribution). controls_.tracks() is itself always
         * constructed with fehMin/fehMax = fehDist.getMin()/getMax()
         * (see io::SimControls::readTracks()), so controls_.tracks().feH()
         * is guaranteed to bracket fehDist's own domain -- safe to use
         * directly as the Interpolator1D x-grid, without further
         * clamping, whenever fehDist is non-degenerate.
         */
        [[nodiscard]] auto specCtsHelper(
            const pdfs::PDF& sfr,
            const pdfs::PDF& imf,
            const pdfs::PDF& fehDist,
            double curTime,
            double fCluster,
            bool computeLbol
        ) const -> std::vector<double>;

        /**
         * @brief The integrand for the continuous-population specCts() overload's outer (age) integral
         * @param age The stellar age, in yr, to evaluate at (not time --
         *   specCtsHelper() integrates this dimension via a
         *   pdfs::PDFReflect view of sfr, pivoted so that the
         *   coordinate PDFIntegrator hands back already is age; see its
         *   own comment)
         * @param imf The initial mass function of the population -- see
         *   specCtsHelper()'s own imf parameter; used both as the
         *   inner mass integral's own weighting PDF and, via
         *   getMin()/getMax(), its own integration bounds
         * @param computeLbol Whether to also compute, and append, this
         *   age's own integrated bolometric luminosity -- see
         *   specCtsImpl()'s own computeLbol parameter
         * @param feh The single [Fe/H] value this call is evaluated
         *   at -- specCtsHelper() calls this once per grid point in
         *   controls_.tracks().feH() when fehDist is non-degenerate
         *   (see its own comment for why), or once, at fehDist.getMin()
         *   == fehDist.getMax(), when it is degenerate. Either way,
         *   [Fe/H] is fixed for the whole call.
         * @return wl_.size() values (lambda * dL/dlambda, see specWl()),
         *   plus -- only if computeLbol -- one further value, this
         *   age's own integrated bolometric luminosity, in erg/s (see
         *   specCtsImpl()'s own comment for why erg/s rather than
         *   Cluster::lbol()'s own Lsun).
         * @details
         * Floors log10(age) at controls_.tracks().logTMin() (as
         * Cluster::advance() already does, to avoid taking log(0) for
         * an age of exactly 0), builds the isochrone at that (log age,
         * feh) via controls_.tracks().getIsochrone(), and then hands
         * that isochrone straight to specCtsImpl() (forIntegration =
         * true) to perform the inner integral over mass, bounded to
         * [imf.getMin(), imf.getMax()] -- reusing the same proven,
         * benchmarked pAdaptive-in-linear-mass-space mass integrator
         * the isochrone specCts() overload itself uses, rather than a
         * second, independent implementation of the same integral. No
         * explicit live/dead mass check is needed here: the isochrone's
         * own segments, each only valid over their own
         * [xMin(), xMax()], already encode exactly which masses are
         * alive at this age -- specCtsImpl() only ever evaluates a
         * segment within its own domain (see its own comment), so a
         * dead mass is simply never visited, rather than visited and
         * then zeroed out as an earlier, jointly-2D design over (age,
         * mass) did (see this function's own git history for that
         * version). That earlier design's real cost wasn't the extra
         * zero-fill work itself, but a curved live/dead boundary
         * embedded *inside* its own 2D (age, mass) integration domain,
         * which 2D adaptive cubature resolves very poorly at ages
         * beyond the shortest-lived stars' own lifetimes (an
         * empirically observed, sharp cost cliff); recasting the
         * problem as nested 1D integrals -- this function providing the
         * outer age integrand, specCtsImpl() the inner mass one --
         * removes that boundary from the integration domain entirely,
         * since each age's own isochrone already has it built in as an
         * ordinary segment edge.
         */
        [[nodiscard]] auto continuousSpecIntegrand(
            double age, const pdfs::PDF& imf, bool computeLbol, double feh) const -> std::vector<double>;
    };

} // namespace specsyn

#endif // SPECSYN_HPP
