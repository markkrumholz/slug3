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
#include <map>
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
         * in the integration domain, computed on demand and cached in
         * isochroneCache_ (cleared again once this integral finishes).
         *
         * Called by Galaxy::computeSpec() when fCluster() < 1, to
         * cover the star-forming mass Galaxy's own individual Cluster
         * objects don't -- see io::SimControls::fCluster()'s own
         * comment.
         *
         * Uses a single PDFIntegratorND evaluation, either 2D (time,
         * mass), weighted by (sfr, imf), if fehDist is degenerate
         * (fehDist.getMin() == fehDist.getMax(), a single metallicity
         * shared by every star), or 3D (time, feh, mass), weighted by
         * (sfr, fehDist, imf), otherwise -- a genuine multi-dimensional
         * integral, rather than nested 1D integrals, so the underlying
         * cubature routine can refine wherever in the full joint domain
         * it needs to in order to hit its own target accuracy, instead
         * of a fixed, uniform refinement in each dimension separately.
         * Uses CubatureMethod::pAdaptive specifically because the
         * integrand's own dominant cost -- building an isochrone at a
         * given (age, feh) -- is shareable across every mass evaluated
         * at that same (age, feh): pAdaptive's nested-quadrature point
         * cache guarantees every batch of points continuousSpecIntegrand()
         * receives shares the same time coordinate (see its own
         * comment), letting isochroneCache_ hit on every reuse.
         *
         * The time dimension is deliberately not log-transformed, even
         * though the mass dimension is: the integral's own lower time
         * bound is always exactly 0 (star formation begins at the
         * start of the simulation), and sfr's own domain -- for the
         * common constant-SFR case (see io::SimControls's own
         * buildConstantSFR()) -- already starts at 0 too, so after
         * PDFIntegratorND::integrate()'s own clamp to sfr's support,
         * the log-transformed lower bound would be exactly log(0),
         * which PDFIntegratorND::integrate() rejects outright.
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
         * Everything specCts()'s own comment describes -- the 2D-vs-3D
         * dimensionality choice, why CubatureMethod::pAdaptive, why the
         * time dimension isn't log-transformed, the lambda * dL/dlambda
         * convention -- applies here unchanged; computeLbol only
         * changes nInt (wl_.size(), or wl_.size() + 1) and whether
         * continuousSpecIntegrand() is asked to also fill in that extra
         * element. The trailing Lbol element, when present, is a
         * genuine luminosity, not a lambda-weighted flux, so it is
         * excluded from this function's own dL/dlambda-recovering
         * division by wl_ -- only scaled by (1 - fCluster), same as
         * every other element. It does, however, get one further
         * unit conversion of its own: continuousSpecIntegrand() reports
         * it in erg/s, to stay on the same absolute scale as the
         * spectral elements this function's own reqAbsError applies to
         * (see its own comment for why), so this function divides it
         * by utils::Lsun before returning, converting it to the Lsun
         * Cluster::lbol() itself uses.
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
         * @brief The vectorized integrand for the continuous-population specCts() overload
         * @tparam Ndim Dimensionality of the integral this is being
         *   used for -- 2 (time, mass) or 3 (time, feh, mass); see
         *   specCts()'s own comment for when each applies
         * @param points An (npts, Ndim) view of the points to evaluate,
         *   as handed by PDFIntegratorND's own vectorized (pAdaptive)
         *   interface: points[i, 0] is the i-th point's time
         *   coordinate, points[i, Ndim - 1] its mass coordinate, and
         *   (if Ndim == 3) points[i, 1] its [Fe/H] coordinate
         * @param cache The calling specCts() call's own isochroneCache_,
         *   passed by reference (rather than read via *this directly)
         *   so this stays a plain function of its own explicit
         *   arguments, matching PDFIntegratorND's own vectorized F
         *   contract
         * @param curTime The current time, in yr, passed through from
         *   specCts() unchanged -- needed to convert each point's own
         *   time coordinate into an age (curTime - time) for isochrone
         *   lookup, since PDFIntegratorND has no notion of this
         *   function's own caller-side context beyond its arguments
         * @param computeLbol Whether to also compute, and append, each
         *   point's own bolometric luminosity -- see specCtsHelper()'s
         *   own comment for the outward effect of this
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
         * For each of the (up to npts) distinct (age, feh) pairs
         * represented in points (age computed from each point's own
         * time coordinate; feh taken directly from points if Ndim == 3,
         * or from controls_.fehDist().getMin() if Ndim == 2, where
         * fehDist is necessarily degenerate -- see specCts()'s own
         * comment), builds (or reuses, from cache) the isochrone at
         * that (age, feh) via controls_.tracks().getIsochrone(),
         * flooring log(age) at controls_.tracks().logTMin() first (as
         * Cluster::advance() already does) to avoid taking log(0) for
         * a point landing exactly at age == 0. Then, for each point,
         * finds whichever of that point's own cached isochrone's
         * segments (if any) contains its mass coordinate -- if none
         * does, that point represents a star already dead at this age,
         * so its own row is left at zero (including its own Lbol
         * element, if present) -- and evaluates specWl() at that
         * point's own (mass, segment, feh) otherwise, storing the
         * result in that point's own row. If computeLbol, also
         * evaluates that same segment at that same mass directly (a
         * second, cheap Interpolator1D call -- specWl() doesn't expose
         * the StarData it computes internally) to read off log(L/Lsun),
         * and stores 10^that value times utils::Lsun -- the star's own
         * bolometric luminosity, in erg/s, not the Lsun
         * Cluster::lbol() itself returns -- as the row's own final
         * element. erg/s, matching the scale of the spectral elements
         * alongside it, because specCtsHelper()'s own reqAbsError is
         * itself in erg/s (see its own comment): a Lbol value of order
         * unity (Lsun) would look converged to that tolerance
         * immediately regardless of its actual accuracy, so
         * specCtsHelper() instead divides this back down to Lsun only
         * after the integral itself is done.
         *
         * Deliberately does not clear cache itself: a single specCts()
         * integral typically calls this function many times (once per
         * cubature refinement step), and cache is meant to accumulate
         * isochrones across all of them, only cleared by specCtsHelper()
         * itself once the whole integral is done.
         */
        template <std::size_t Ndim>
        [[nodiscard]] auto continuousSpecIntegrand(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, Ndim>> points, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
            std::map<std::pair<double, double>, Isochrone>& cache,
            double curTime,
            bool computeLbol) const -> std::vector<double>;

        /**
         * @brief Cache of isochrones built while evaluating the continuous-population specCts() overload
         * @details
         * Keyed by (age, feh) -- see continuousSpecIntegrand()'s own
         * comment for how each entry is built and reused. Always empty
         * outside of a specCtsHelper() call (i.e. one made via
         * specCts()/specAndLbolCts()): populated as needed during that
         * call, then cleared again once it returns. Never wrapped for
         * thread safety (e.g. in a ThreadVec), since a single Specsyn is
         * never evaluated from more than one thread at a time in
         * practice (a Galaxy, and the Specsyn it reads from
         * SimControls, are never shared across threads). Declared
         * mutable so specCtsHelper() -- a const method, like every
         * other Specsyn method that touches controls_ -- can still
         * populate and clear it.
         */
        mutable std::map<std::pair<double, double>, Isochrone> isochroneCache_;
    };

} // namespace specsyn

#endif // SPECSYN_HPP
