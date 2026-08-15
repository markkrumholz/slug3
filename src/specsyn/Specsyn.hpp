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
#include <mdspan> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLibWR.hpp's own <mdspan> include
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
         * @param m See specWl()'s own m parameter
         * @param segment See specWl()'s own segment parameter
         * @param feh See specForIntegration()'s own feh parameter --
         *   unlike specWl()'s own feh, not necessarily within this
         *   synthesizer's own real data coverage
         * @return specForIntegration(segment(m), feh), multiplied
         *   elementwise by wl() -- see specWl()'s own return value
         * @details
         * Identical to specWl(), except calling specForIntegration()
         * in place of spec() -- see its own comment for why
         * Specsyn::continuousSpecIntegrand() needs this, rather than
         * specWl() itself.
         */
        [[nodiscard]] auto specWlForIntegration(double m, const Segment& segment, double feh) const -> std::vector<double>
        {
            auto result = specForIntegration(segment(m), feh);
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
         * If fehDist is degenerate (fehDist.getMin() == fehDist.getMax(),
         * a single metallicity shared by every star), uses a single 2D
         * PDFIntegratorND evaluation (time, mass), weighted by
         * (sfr, imf). Otherwise, rather than folding [Fe/H] into the
         * cubature as a third continuous dimension, integrates over
         * [Fe/H] via a separate mechanism entirely -- see
         * specCtsHelper()'s own comment for why (in short: a genuinely
         * 3D, vector-valued -- fdim == wl_.size() -- pAdaptive integral
         * was found to have unbounded memory growth, a property of
         * cubature's own tensor-product algorithm, not of this
         * integral's own complexity) -- running the same 2D integral
         * once at every [Fe/H] grid point the tracks are actually
         * defined at (controls_.tracks().feH()), then interpolating
         * and integrating those discrete results over [Fe/H] with
         * interp::Interpolator1D.
         *
         * Uses CubatureMethod::pAdaptive specifically because the
         * integrand's own dominant cost -- building an isochrone at a
         * given age -- is shareable across every mass evaluated at
         * that same age: pAdaptive's nested-quadrature point cache
         * guarantees every batch of points continuousSpecIntegrand()
         * receives shares the same age coordinate within a run (see its
         * own comment), letting a single cached isochrone be reused
         * across an entire run rather than rebuilt per point. Benchmarked head
         * to head against CubatureMethod::hAdaptive on this same
         * integral (real MIST tracks, curTime up to 1e10 yr, capped at
         * a range of max_iter values): with the age dimension correctly
         * log-transformed, the two methods converge to the same answer
         * (agreeing to ~0.03% median / ~4% max relative difference by
         * max_iter = 2^19 each) and both stay memory-safe, but
         * pAdaptive is dramatically faster -- every max_iter tried, up
         * to 2^19, finished in a few seconds, versus tens of seconds to
         * a few minutes for hAdaptive at the same max_iter.
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
         * degenerate-vs-general [Fe/H] handling, why
         * CubatureMethod::pAdaptive, the reflected-and-log-transformed
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
         * integrated by running the 2D (age, mass) integral once at
         * every grid point in controls_.tracks().feH() -- rather than
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
         * 2D integrals -- each identical in character and cost to the
         * already-robust, benchmarked degenerate-fehDist case above --
         * plus cheap post-hoc 1D interpolation over [Fe/H]. Total cost
         * is therefore nFeh times the cost of one 2D integral, plus
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
         * @brief The vectorized integrand for the continuous-population specCts() overload's 2D (age, mass) integral
         * @param points An (npts, 2) view of the points to evaluate, as
         *   handed by PDFIntegratorND's own vectorized (pAdaptive)
         *   interface: points[i, 0] is the i-th point's own age
         *   coordinate (not time -- specCtsHelper() integrates this
         *   dimension via a pdfs::PDFReflect view of sfr, pivoted so
         *   that the coordinate PDFIntegratorND hands back already is
         *   age; see its own comment), and points[i, 1] is its own mass
         *   coordinate.
         * @param computeLbol Whether to also compute, and append, each
         *   point's own bolometric luminosity -- see specCtsHelper()'s
         *   own comment for the outward effect of this
         * @param feh The single [Fe/H] value shared by every point in
         *   this call -- specCtsHelper() calls this once per grid point
         *   in controls_.tracks().feH() when fehDist is non-degenerate
         *   (see its own comment for why), or once, at fehDist.getMin()
         *   == fehDist.getMax(), when it is degenerate. Either way,
         *   [Fe/H] is fixed for the whole call, not read from points
         *   itself. Used as-is both to locate the isochrone (via
         *   controls_.tracks(), whose own [Fe/H] coverage this value is
         *   always guaranteed to fall within) and, via
         *   specWlForIntegration() rather than specWl() itself, to
         *   evaluate the star's own spectrum -- see
         *   Specsyn::specForIntegration()'s own comment for why that,
         *   not specWl()'s ordinary spec() call, is what belongs here:
         *   the tracks' own [Fe/H] coverage (padded for interpolation
         *   purposes -- see specCtsHelper()'s own comment) can extend
         *   slightly beyond a real spectral synthesizer's own tabulated
         *   data, which specWl() itself has no way to accommodate
         *   without changing spec()/specForce()'s own, deliberately
         *   un-rescued, behavior for every other caller.
         * @return npts * (wl_.size() + computeLbol) values: for each
         *   point, lambda * dL/dlambda (see specWl()) at every one of
         *   wl_.size() wavelengths, then -- only if computeLbol -- one
         *   further value, that point's own bolometric luminosity, in
         *   erg/s (see this function's own @details for why erg/s
         *   rather than Cluster::lbol()'s own Lsun). Laid out as
         *   result[i * (wl_.size() + computeLbol) + k] for the k-th
         *   such value of the i-th point -- the layout
         *   PDFIntegratorND's own vectorized interface requires.
         * @details
         * Builds at most one isochrone at a time, in a single-slot
         * local cache (an (age, isochrone) pair, isochrone starting
         * null): for each point, if its own age doesn't match what's
         * currently in that slot, replaces the slot's isochrone with a
         * freshly built one via controls_.tracks().getIsochrone()
         * (log(age) floored at controls_.tracks().logTMin() first, as
         * Cluster::advance() already does, to avoid taking log(0) for a
         * point landing exactly at age == 0), then uses whatever is in
         * the slot. PDFIntegratorND's pAdaptive interface hands points
         * grouped into runs sharing the same age (time is deliberately
         * the first/outermost integration dimension precisely so
         * batches arrive already sorted this way -- see
         * specCtsHelper()'s own comment), so consecutive points
         * typically hit the same slot and reuse its isochrone rather
         * than rebuilding one -- but never more than a single isochrone
         * is resident at once, regardless of how many distinct ages a
         * batch spans. An earlier design instead built every isochrone
         * a whole batch needed up front, keyed in a persistent map:
         * memory-safe only as long as a batch's own distinct-age count
         * stayed modest, which measurably failed (tens of GB) once the
         * age dimension's own log transform (see specCtsHelper()'s own
         * comment) let a single batch span a very large number of
         * distinct ages at large curTime.
         *
         * Once a point's own isochrone (freshly built or reused from
         * the slot) is in hand, finds whichever of its segments (if
         * any) contains that point's mass coordinate -- if none does,
         * that point represents a star already dead at this age, so
         * its own row is left at zero (including its own Lbol element,
         * if present) -- and evaluates specWl() at that point's own
         * (mass, segment, feh) otherwise, storing the result in that
         * point's own row. If computeLbol, also evaluates that same
         * segment at that same mass directly (a second, cheap
         * Interpolator1D call -- specWl() doesn't expose the StarData
         * it computes internally) to read off log(L/Lsun), and stores
         * 10^that value times utils::Lsun -- the star's own bolometric
         * luminosity, in erg/s, not the Lsun Cluster::lbol() itself
         * returns -- as the row's own final element. erg/s, matching
         * the scale of the spectral elements alongside it, because
         * specCtsHelper()'s own reqAbsError is itself in erg/s (see its
         * own comment): a Lbol value of order unity (Lsun) would look
         * converged to that tolerance immediately regardless of its
         * actual accuracy, so specCtsHelper() instead divides this back
         * down to Lsun only after the integral itself is done.
         */
        [[nodiscard]] auto continuousSpecIntegrand(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>> points, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
            bool computeLbol, double feh) const -> std::vector<double>;
    };

} // namespace specsyn

#endif // SPECSYN_HPP
