/**
 * @file Galaxy.hpp
 * @author Mark Krumholz
 * @brief A class to represent a galaxy, built from a time-evolving population of star clusters
 * @date 2026-08-10
 */

#ifndef GALAXY_HPP
#define GALAXY_HPP

#include "../io/SimControls.hpp"
#include "../pdfs/PDF.hpp"
#include "../specsyn/Specsyn.hpp"
#include "Cluster.hpp"
#include <cstddef>
#include <functional>
#include <vector>

namespace core
{

    /**
     * @brief A class to represent a galaxy
     * @details
     * Unlike Cluster, which represents a single mono-age population,
     * a Galaxy represents an entire, continuously star-forming system:
     * advance() draws new clusters from SimControls::sfr()/cmf() as
     * time passes, advances every cluster formed so far (whether still
     * bound or already disrupted), and sums their individual spectra,
     * photometry, and bolometric luminosities into this Galaxy's own
     * spec()/specExtinct()/phot()/photExtinct()/lbol().
     */
    class Galaxy
    {
    public:

        /**
         * @brief Initialize a galaxy
         * @param controls Simulation controls (physics settings and
         *   control-flow/integrator-tolerance settings together);
         *   stored by reference, so it must outlive this Galaxy --
         *   see controls_'s own comment
         * @details
         * curTime_, lbol_, targetMass_, and actualMass_ all start at 0,
         * and clusters_/disruptedClusters_/spec_/specExtinct_/phot_/
         * photExtinct_ all start empty -- no clusters exist, and no
         * spectrum/photometry has been computed, until the first call
         * to advance().
         */
        explicit Galaxy(const io::SimControls& controls);

        // Observers

        /**
         * @brief Return the galaxy's current time
         * @return Current simulation time, in yr
         */
        [[nodiscard]] auto curTime() const { return curTime_; }

        /**
         * @brief Return the galaxy's currently-alive (non-disrupted) clusters
         * @return A reference to the list of clusters formed so far
         *   that have not yet disrupted
         * @details
         * Not const: callers commonly need to call a lazily-computed
         * getter (spec()/phot()/lbol()/etc. -- see spec()'s own
         * comment) on the individual Cluster elements this returns,
         * which are themselves non-const for the same reason.
         */
        [[nodiscard]] auto clusters() -> auto& { return clusters_; }

        /**
         * @brief Return the galaxy's disrupted clusters
         * @return A reference to the list of clusters formed so far
         *   that have already disrupted
         * @details
         * Not const -- see clusters()'s own comment.
         */
        [[nodiscard]] auto disruptedClusters() -> auto& { return disruptedClusters_; }

        /**
         * @brief Return the galaxy's continuously-sampled spectrum
         * @return A const reference to the sum of spec() over every
         *   cluster in clusters() and disruptedClusters(), on the
         *   wavelength grid of the simulation's spectral synthesizer,
         *   or an empty vector if no spectral synthesizer was
         *   requested (SimControls::specsyn() is null)
         * @details
         * Computed lazily: if advance() has run since spec_/specExtinct_
         * were last computed, this triggers computeSpec() (which
         * computes both together) before returning, so the result is
         * always current as of the last advance() -- see specCurrent_'s
         * own comment. Not const, since it may need to run that
         * computation.
         */
        [[nodiscard]] auto spec() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return spec_;
        }

        /**
         * @brief Return the galaxy's extincted spectrum
         * @return A const reference to the sum of specExtinct() over
         *   every cluster in clusters() and disruptedClusters(), on
         *   the wavelength grid returned by SimControls::extinct()'s
         *   own wl(); an empty vector if no extinction curve was
         *   requested (SimControls::extinct() is null) or spec()
         *   itself is empty (no spectral synthesizer was requested)
         * @details
         * Computed lazily -- see spec()'s own comment; both share the
         * same specCurrent_ flag, since a single computeSpec() call
         * computes both together.
         */
        [[nodiscard]] auto specExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specExtinct_;
        }

        /**
         * @brief Return the galaxy's photometry
         * @return A const reference to the photometric value computed
         *   from spec() by each filter in SimControls::filters(), in
         *   the same order as FilterCollection::filterNames()/
         *   filterUnits(), or an empty vector if no filter collection
         *   was requested (SimControls::filters() is null)
         * @details
         * Computed lazily -- see spec()'s own comment; computePhot()
         * itself calls spec()/specExtinct() (rather than reading
         * spec_/specExtinct_ directly), so calling phot() alone, with
         * no prior call to spec(), still computes a spectrum current
         * as of the last advance() first -- which in turn computes
         * each individual cluster's own spectrum, but not its
         * photometry (see Cluster::computePhot()'s own comment).
         */
        [[nodiscard]] auto phot() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return phot_;
        }

        /**
         * @brief Return the galaxy's extincted photometry
         * @return A const reference to the photometric value computed
         *   from specExtinct() by each filter in
         *   SimControls::filters(), in the same order as phot(); an
         *   empty vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or no filter collection
         *   was requested (SimControls::filters() is null)
         * @details
         * Computed lazily -- see phot()'s own comment; both share the
         * same photCurrent_ flag, since a single computePhot() call
         * computes both together.
         */
        [[nodiscard]] auto photExtinct() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return photExtinct_;
        }

        /**
         * @brief Return the galaxy's bolometric luminosity
         * @return The sum of lbol() over every cluster in clusters()
         *   and disruptedClusters(), in Lsun, at the current time, or
         *   0 if SimControls::computeLbol() is false (Lbol was never
         *   requested)
         * @details
         * Computed lazily -- see spec()'s own comment. computeLbol()
         * itself is a no-op (leaving lbol_ at 0) unless
         * SimControls::computeLbol() is true, mirroring computeSpec()/
         * computePhot()'s own null-guards.
         */
        [[nodiscard]] auto lbol()
        {
            if (!lbolCurrent_) { computeLbol(); lbolCurrent_ = true; }
            return lbol_;
        }

        /**
         * @brief Return the total target stellar mass formed so far
         * @return The sum, over every advance() call so far, of the
         *   total stellar mass (sfr().integral() over that call's own
         *   (curTime(), t] step) that should have formed -- both the
         *   stochastically-treated (clustered) and continuous
         *   (non-clustered) parts together -- in Msun -- may differ
         *   from actualMass() due to stochastic sampling from
         *   SimControls::cmf() (see Galaxy::advance())
         */
        [[nodiscard]] auto targetMass() const { return targetMass_; }

        /**
         * @brief Return the total actual stellar mass formed so far
         * @return The sum, over every advance() call so far, of (1)
         *   the target mass (Cluster::targetMass()) of every cluster
         *   actually drawn from SimControls::cmf() during that call
         *   and (2) that same call's own continuous (non-clustered)
         *   population mass, (1 - SimControls::fCluster()) times that
         *   call's own total target stellar mass, in Msun
         */
        [[nodiscard]] auto actualMass() const { return actualMass_; }

        /**
         * @brief Advance the galaxy in time
         * @param t Time to which to advance, in yr; must be >= curTime()
         * @details
         * Draws and forms new clusters over (curTime(), t] from
         * SimControls::sfr()/cmf(), scaled down to the
         * stochastically-treated fraction SimControls::fCluster() of
         * the step's own total target mass (the remaining
         * (1 - fCluster()) is folded directly into actualMass(), as
         * the continuous, non-clustered population's own share, with
         * no individual Cluster object of its own), accumulating the
         * step's own target and actual mass into targetMass()/
         * actualMass(), advances every cluster formed so far (in
         * clusters() and disruptedClusters()) to t, moves any cluster
         * that disrupted during this step from clusters() to
         * disruptedClusters(), then marks spec_/specExtinct_/phot_/
         * photExtinct_/lbol_/lbolCts_ as stale (see specCurrent_/
         * photCurrent_/lbolCurrent_/lbolCtsCurrent_'s own comments)
         * rather than recomputing them itself -- they are instead
         * recomputed lazily, on demand, the next time spec()/
         * specExtinct()/phot()/photExtinct()/lbol() is actually called
         * -- before finally updating curTime() to t.
         */
        void advance(double t);

    private:

        double curTime_ = 0.0;                  /**< Current simulation time */
        std::vector<Cluster> clusters_;          /**< Currently alive (non-disrupted) clusters */
        std::vector<Cluster> disruptedClusters_; /**< Disrupted clusters */

        std::vector<double> spec_;         /**< Sum of spec() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> specExtinct_;  /**< Sum of specExtinct() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> phot_;         /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_;  /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;                /**< Sum of lbol() over every cluster in clusters_/disruptedClusters_, at the current time */
        double targetMass_ = 0.0;          /**< Cumulative total target stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */
        double actualMass_ = 0.0;          /**< Cumulative total actual stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */

        /**
         * @brief Whether spec_/specExtinct_ are current as of curTime_
         * @details
         * Mirrors Cluster::specCurrent_'s own comment: true at
         * construction, set false at the end of every advance() call,
         * and back to true by spec()/specExtinct() after recomputing
         * them.
         */
        bool specCurrent_ = true;

        /**
         * @brief Whether phot_/photExtinct_ are current as of curTime_
         * @details
         * Mirrors specCurrent_'s own comment, for phot()/photExtinct().
         */
        bool photCurrent_ = true;

        /**
         * @brief Whether lbol_ is current as of curTime_
         * @details
         * Mirrors specCurrent_'s own comment, for lbol().
         */
        bool lbolCurrent_ = true;

        /**
         * @brief The continuous population's own bolometric luminosity, in Lsun (matching lbol_'s own units)
         * @details
         * Set by computeSpec() whenever it computes a spectrum via
         * Specsyn::specAndLbolCts() rather than plain specCts() (see
         * lbolCtsCurrent_'s own comment for when that happens), as a
         * byproduct of that same integral rather than a separate one
         * -- see specAndLbolCts()'s own comment for why. Read by
         * computeLbol(), which adds it into lbol_ when
         * lbolCtsCurrent_ is true. Stays 0 whenever it was never set
         * this way (fCluster() == 1, no spectral synthesizer, or Lbol
         * was computed via the standalone computeLbolCts() path
         * instead -- see lbolCtsCurrent_'s own comment).
         */
        double lbolCts_ = 0.0;

        /**
         * @brief Whether lbolCts_ is current as of curTime_
         * @details
         * Unlike specCurrent_/photCurrent_/lbolCurrent_ (all true at
         * construction, since there is nothing yet to be stale), starts
         * false: lbolCts_ itself starts at 0, which is both the
         * "nothing computed yet" value and a value computeLbol() must
         * not blindly trust without this flag. Set true by
         * computeSpec() only when it actually took the
         * Specsyn::specAndLbolCts() path (SimControls::computeLbol()
         * true and fCluster() < 1); set false at the end of every
         * advance() call, alongside specCurrent_/photCurrent_/
         * lbolCurrent_. computeLbol() checks this before trusting
         * lbolCts_: if false when Lbol is still wanted (Lbol requested
         * but no spectrum was computed this step, e.g. only lbol() was
         * called, not spec()), it instead falls back to the standalone
         * computeLbolCts() path.
         */
        bool lbolCtsCurrent_ = false;

        /**
         * @brief Simulation controls (physics and control-flow settings) this galaxy was built from
         * @details
         * See Cluster::controls_'s own comment: read live wherever a
         * physics setting or integrator tolerance is needed, rather
         * than snapshotted at construction.
         */
        std::reference_wrapper<const io::SimControls> controls_;

        /**
         * @brief Update spec_ (and specExtinct_) from the current cluster population
         * @details
         * Mirrors Cluster::computeSpec()'s own null-guard, but sums
         * over clusters rather than stars: does nothing if
         * SimControls::specsyn() is null. Otherwise sets spec_ to the
         * sum of spec() over every cluster in clusters_ and
         * disruptedClusters_ (forcing each cluster's own spectrum to be
         * computed, if not already current); if SimControls::extinct()
         * is also non-null, also sets specExtinct_ to the sum of
         * specExtinct() over the same clusters.
         *
         * If fCluster() < 1, then adds the continuous population's own
         * contribution to both spec_ and (if extinct() is non-null)
         * specExtinct_ -- see the implementation's own comment for why
         * specExtinct_'s own share goes in unattenuated, and at an
         * offset from spec_'s own. Gets this contribution via
         * Specsyn::specAndLbolCts() rather than plain specCts() -- also
         * setting lbolCts_/lbolCtsCurrent_ from its own second return
         * value -- whenever SimControls::computeLbol() is true, so Lbol
         * comes along for free from the same integral; via plain
         * specCts() otherwise, leaving lbolCts_/lbolCtsCurrent_
         * untouched.
         */
        void computeSpec();

        /**
         * @brief Update phot_ (and photExtinct_) from the current spec_ (and specExtinct_)
         * @details
         * Identical to Cluster::computePhot(), but reading this
         * Galaxy's own spec()/specExtinct() (the lazy getters, not
         * spec_/specExtinct_ directly) rather than a Cluster's -- note
         * that this reads the Galaxy's own already-summed spectrum,
         * not a sum of each cluster's own phot(), so requesting
         * Galaxy::phot() alone forces every cluster's own spectrum to
         * be computed, but not any cluster's own photometry.
         */
        void computePhot();

        /**
         * @brief Update lbol_ from the current cluster population (and, if current, the continuous population)
         * @details
         * Does nothing if SimControls::computeLbol() is false (Lbol was
         * never requested), mirroring Cluster::computeLbol()'s own
         * null-guard. Otherwise sets lbol_ to the sum of lbol() over
         * every cluster in clusters_ and disruptedClusters_ (forcing
         * each cluster's own Lbol to be computed, if not already
         * current), plus lbolCts_ if lbolCtsCurrent_ is true (i.e.
         * computeSpec() already computed it this step, as a byproduct
         * of computing spec() -- see lbolCtsCurrent_'s own comment).
         *
         * If lbolCtsCurrent_ is instead false -- Lbol was requested,
         * fCluster() < 1, but computeSpec() hasn't run since the last
         * advance() (e.g. a caller asked for lbol() without ever
         * asking for spec()) -- the continuous population's own Lbol
         * still needs computing, but not by paying for a full spectrum
         * it was never asked for; calls computeLbolCts() for exactly
         * that.
         */
        void computeLbol();

        /**
         * @brief The integrand for computeLbolCts()'s own standalone Lbol integral's outer (age) integral
         * @param age The stellar age, in yr, to evaluate at -- see
         *   Specsyn::continuousSpecIntegrand()'s own age parameter for
         *   the identical convention (not time -- computeLbolCts()
         *   integrates this dimension via a pdfs::PDFReflect view of
         *   sfr, pivoted so that the coordinate PDFIntegratorGK hands
         *   back already is age)
         * @param imf The initial mass function of the population --
         *   see computeLbolCts()'s own imf parameter; used both as the
         *   inner mass integral's own weighting PDF and, via
         *   getMin()/getMax(), its own integration bounds
         * @param feh The single [Fe/H] value this call is evaluated
         *   at -- see Specsyn::continuousSpecIntegrand()'s own feh
         *   parameter for the identical convention
         * @return A single-element vector holding this age's own
         *   integrated bolometric luminosity, in Lsun -- unlike
         *   Specsyn::continuousSpecIntegrand()'s own Lbol element, no
         *   further unit conversion is needed afterward, since this
         *   integral's own absolute tolerance is already specified
         *   directly in Lsun (see computeLbolCts()'s own comment for
         *   why that differs from the shared-with-a-spectrum case)
         * @details
         * Mirrors Specsyn::continuousSpecIntegrand()'s own structure
         * exactly (see its own comment): floors log10(age), builds the
         * isochrone at that (log age, feh), then integrates each of its
         * segments' own bolometric luminosity against imf via a nested
         * PDFIntegratorGK (GKOrder::GK15) -- a local, capture-free
         * lambda mirroring Cluster::lbolStar()'s own role, rather than
         * calling into any Specsyn (which may not exist at all when
         * this runs -- see computeLbolCts()'s own comment), reading
         * each star's own log(L/Lsun) directly off its isochrone
         * segment and summing 10^that over the segments a star of that
         * mass could belong to. No explicit live/dead mass check is
         * needed: exactly as in continuousSpecIntegrand(), a dead mass
         * is simply never visited, since it falls in none of the
         * isochrone's own segment domains.
         */
        [[nodiscard]] auto lbolCtsIntegrand(double age, const pdfs::PDF& imf, double feh) const -> std::vector<double>;

        /**
         * @brief Compute lbolCts_ directly, without going through a full spectrum
         * @details
         * Called by computeLbol() when Lbol is wanted but computeSpec()
         * hasn't already computed lbolCts_ as a byproduct of computing
         * a spectrum (lbolCtsCurrent_ is false) -- most notably when
         * SimControls::specsyn() is null (no spectral synthesizer
         * configured at all, so Galaxy::computeSpec() is a no-op and
         * Specsyn::specAndLbolCts() never runs), but also whenever
         * lbol() is requested without spec() ever having been
         * requested first this step.
         *
         * Mirrors Specsyn::specCtsHelper()'s own nested-1D structure
         * exactly (the reflected-and-log-transformed age coordinate, a
         * PDFIntegratorGK over age alone whose own integrand --
         * lbolCtsIntegrand() -- performs a complete inner 1D integral
         * over mass via a fresh isochrone built at its own age, and --
         * when fehDist is non-degenerate -- running that same nested
         * integral once per grid point in tracks().feH() and
         * integrating those discrete results over [Fe/H] via
         * interp::Interpolator1D; see its own comment for the full
         * rationale, which applies here unchanged), just integrating a
         * single quantity (Lbol alone, via lbolCtsIntegrand()) instead
         * of a spectrum plus Lbol together, and directly in Lsun
         * throughout: unlike specCtsHelper(), there is no spectral
         * absolute tolerance for an erg/s-scale intermediate to share,
         * so reqAbsError is simply intAbsTol() * sfr().integral(0,
         * curTime()), with no utils::Lsun factor. Its own
         * lbolCtsIntegrand() builds isochrones directly (see its own
         * comment), independent of any Specsyn, since a Specsyn may not
         * exist at all here.
         *
         * Sets lbolCts_ to the integral's own result, scaled by
         * (1 - fCluster()) for the continuously-treated share of the
         * population, matching Specsyn::specAndLbolCts()'s own
         * identical scaling -- necessary for the two to agree when
         * both are exercised for the same population. Sets
         * lbolCtsCurrent_ to true afterward.
         */
        void computeLbolCts();

    };

} // namespace core

#endif // GALAXY_HPP
