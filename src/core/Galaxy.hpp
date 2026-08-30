/**
 * @file Galaxy.hpp
 * @author Mark Krumholz
 * @brief A class to represent a galaxy, built from a time-evolving population of star clusters
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef GALAXY_HPP
#define GALAXY_HPP

#include "../io/SimControls.hpp"
#include "../pdfs/PDF.hpp"
#include "../specsyn/Specsyn.hpp"
#include "Cluster.hpp"
#include <cstddef>
#include <functional>
#include <variant>
#include <vector>

namespace extinct
{
    class Extinct;
} // namespace extinct

namespace nebular
{
    class Nebular;
} // namespace nebular

namespace core
{

    /**
     * @brief A class to represent a galaxy
     * @details
     * Unlike Cluster, which represents a single mono-age population,
     * a Galaxy represents an entire, continuously star-forming system:
     * advance() draws new clusters from sfr()/SimControls::cmf() as
     * time passes, advances every cluster formed so far (whether still
     * bound or already disrupted), and sums their individual spectra,
     * photometry, and bolometric luminosities into this Galaxy's own
     * spec()/specExtinct()/phot()/photExtinct()/lbol().
     */
    class Galaxy
    {
    public:

        /**
         * @brief Holds this galaxy's own resolved star formation rate
         * @details
         * Either a reference to SimControls::sfr() itself (when
         * galaxy.sfr was given, so every Galaxy built from the same
         * SimControls shares the identical rate) or a PDF this Galaxy
         * owns outright (when galaxy.sfr_dist was given instead, so
         * each Galaxy draws its own, independent constant rate -- see
         * the constructor's own comment). Mirrors Cluster::Track2DVar's
         * identical shared-vs-owned pattern exactly -- see its own
         * comment for the rationale.
         */
        using SFRVar = std::variant<pdfs::PDF, std::reference_wrapper<const pdfs::PDF>>;

        /**
         * @brief A single non-clustered, individually-tracked (field) star
         * @details
         * Represents one star drawn from the stochastically-treated
         * share of the non-clustered population -- mass at or above
         * SimControls::minStochMass(), but never folded into any
         * individual Cluster object, since it formed outside
         * SimControls::fCluster()'s own clustered fraction -- see
         * Galaxy::advance()'s own comment for how these are drawn,
         * alongside the purely continuous (below minStochMass) share
         * Specsyn::specCts()/specAndLbolCts() integrate directly, with
         * no individual FieldStar of its own.
         */
        struct FieldStar
        {
            double mass_;      /**< Initial (birth) mass, in Msun */
            double feh_;       /**< [Fe/H] */
            double formTime_;  /**< Formation time, in yr */
            double deathTime_; /**< Time this star dies, in yr (formTime_ + SimControls::tracks()'s own starLifetime(mass_, feh_)) */
            double aV_;        /**< V-band extinction, in magnitudes, drawn from SimControls::avDistField() at formation (see Galaxy::advance()'s own comment) -- unlike a bound cluster (Cluster::aV_, shared by every star in it), each field star draws its own, independent value, since field stars are not physically clustered together; not currently read for spec()/specExtinct(), which instead attenuate every field star by the same *expected* extinction as the purely continuous population (see addContinuousSpec()'s own comment) */
        };

        /**
         * @brief Initialize a galaxy
         * @param controls Simulation controls (physics settings and
         *   control-flow/integrator-tolerance settings together);
         *   stored by reference, so it must outlive this Galaxy --
         *   see controls_'s own comment
         * @details
         * curTime_, lbol_, targetMass_, and actualMass_ all start at 0,
         * and clusters_/disruptedClusters_/fieldStars_/deadFieldStars_/
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/phot_/
         * photExtinct_/photNeb_/photNebExtinct_ all start empty -- no
         * clusters or field stars exist, and no spectrum/photometry has
         * been computed, until the first call to advance().
         *
         * Also resolves sfr_ here, once, for the rest of this Galaxy's
         * own lifetime: if controls.sfr() is valid (galaxy.sfr was
         * given), sfr_ becomes a reference to that same PDF, read live
         * exactly as every other SimControls-derived quantity is
         * elsewhere in this class. Otherwise, if controls.sfrDist() is
         * valid instead (galaxy.sfr_dist was given -- SimControls's own
         * constructor guarantees exactly one of the two for a
         * galaxy-type simulation), sfr_ instead becomes a PDF this
         * Galaxy owns outright, built by SimControls::buildConstantSFR()
         * from a single value drawn from controls.sfrDist() -- a
         * constant rate, but an independent one per Galaxy, so
         * multiple Galaxy instances built from the same SimControls
         * (e.g. separate trials) each get their own draw. If neither is
         * valid (a Galaxy built from a non-galaxy-type SimControls,
         * e.g. the pybind default, sim_type = "cluster", which never
         * sets either), sfr_ is simply left invalid, exactly as
         * sc.sfr() itself would have been -- any code that actually
         * needs a valid sfr() fails the same way it always did.
         */
        explicit Galaxy(const io::SimControls& controls);

        // Observers

        /**
         * @brief Return this galaxy's own resolved star formation rate
         * @return A const reference to the PDF sfr_ holds -- see its
         *   own comment for how this galaxy resolved it, once, at
         *   construction
         */
        [[nodiscard]] auto sfr() const -> const pdfs::PDF&
        {
            if (const auto* owned = std::get_if<pdfs::PDF>(&sfr_)) { return *owned; }
            return std::get<std::reference_wrapper<const pdfs::PDF>>(sfr_).get();
        }

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
         * @brief Return the galaxy's currently-alive field stars
         * @return A const reference to the list of individually-tracked
         *   (field) stars formed so far -- see FieldStar's own comment
         *   -- that have not yet died, sorted by formTime_ (see
         *   advance()'s own comment for why that ordering is
         *   guaranteed)
         * @details
         * Const, unlike clusters()/disruptedClusters(): a FieldStar is
         * a plain data struct with no lazily-computed getter of its
         * own to trigger.
         */
        [[nodiscard]] auto fieldStars() const -> const auto& { return fieldStars_; }

        /**
         * @brief Return the galaxy's dead field stars
         * @return A const reference to the list of individually-tracked
         *   (field) stars that have died as of curTime() -- see
         *   advance()'s own comment for how a field star moves from
         *   fieldStars() to here
         */
        [[nodiscard]] auto deadFieldStars() const -> const auto& { return deadFieldStars_; }

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
         * computes spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_
         * together) before returning, so the result is always current
         * as of the last advance() -- see specCurrent_'s own comment.
         * Not const, since it may need to run that computation.
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
         * Computed lazily -- see spec()'s own comment; all share the
         * same specCurrent_ flag, since a single computeSpec() call
         * computes them together.
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
         * Computed lazily -- see phot()'s own comment; all share the
         * same photCurrent_ flag, since a single computePhot() call
         * computes them together.
         */
        [[nodiscard]] auto photExtinct() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return photExtinct_;
        }

        /**
         * @brief Return the galaxy's stellar + nebular spectrum
         * @return A const reference to the sum of specNeb() over every
         *   cluster in clusters() and disruptedClusters(), plus the
         *   continuous population's own nebular-reprocessed
         *   contribution (see addContinuousSpec()'s own comment), on
         *   the wavelength grid of the simulation's spectral
         *   synthesizer; an empty vector if no nebular emission grid
         *   was requested (SimControls::nebular() is null) or spec()
         *   itself is empty (no spectral synthesizer was requested)
         * @details
         * Computed lazily -- see spec()'s own comment; shares the same
         * specCurrent_ flag, since a single computeSpec() call computes
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_ together.
         */
        [[nodiscard]] auto specNeb() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specNeb_;
        }

        /**
         * @brief Return the galaxy's extincted stellar + nebular spectrum
         * @return A const reference to the sum of specNebExtinct() over
         *   every cluster in clusters() and disruptedClusters(), plus
         *   the continuous population's own extincted, nebular-
         *   reprocessed contribution, on the wavelength grid returned
         *   by SimControls::extinct()'s own wl(); an empty vector if no
         *   extinction curve was requested (SimControls::extinct() is
         *   null), no nebular emission grid was requested
         *   (SimControls::nebular() is null), or spec() itself is empty
         * @details
         * Computed lazily -- see specNeb()'s own comment.
         */
        [[nodiscard]] auto specNebExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specNebExtinct_;
        }

        /**
         * @brief Return the galaxy's nebular emission line luminosities
         * @return A const reference to the sum of lineLum() over every
         *   cluster in clusters() and disruptedClusters(), plus the
         *   continuous population's own contribution, in erg/s, in the
         *   same order as SimControls::nebular()'s own lineWl(); an
         *   empty vector if no nebular emission grid was requested
         *   (SimControls::nebular() is null) or spec() itself is empty
         * @details
         * Computed lazily -- see specNeb()'s own comment.
         */
        [[nodiscard]] auto lineLum() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return lineLum_;
        }

        /**
         * @brief Return the galaxy's stellar + nebular photometry
         * @return A const reference to the photometric value computed
         *   from specNeb() by each filter in SimControls::filters(), in
         *   the same order as phot(); an empty vector if no nebular
         *   emission grid was requested (SimControls::nebular() is
         *   null) or no filter collection was requested
         *   (SimControls::filters() is null)
         * @details
         * Computed lazily -- see phot()'s own comment; shares the same
         * photCurrent_ flag, since a single computePhot() call computes
         * phot_/photExtinct_/photNeb_/photNebExtinct_ together.
         */
        [[nodiscard]] auto photNeb() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return photNeb_;
        }

        /**
         * @brief Return the galaxy's extincted stellar + nebular photometry
         * @return A const reference to the photometric value computed
         *   from specNebExtinct() by each filter in
         *   SimControls::filters(), in the same order as phot(); an
         *   empty vector if no extinction curve was requested
         *   (SimControls::extinct() is null), no nebular emission grid
         *   was requested (SimControls::nebular() is null), or no
         *   filter collection was requested (SimControls::filters() is
         *   null)
         * @details
         * Computed lazily -- see photNeb()'s own comment.
         */
        [[nodiscard]] auto photNebExtinct() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return photNebExtinct_;
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
         *   actually drawn from SimControls::cmf() during that call,
         *   (2) that same call's own individually-drawn field-star
         *   mass (see FieldStar's own comment), and (3) that same
         *   call's own purely continuous (non-clustered, below
         *   SimControls::minStochMass()) population mass,
         *   (1 - SimControls::fCluster()) * (1 - SimControls::
         *   fracStochMass()) times that call's own total target
         *   stellar mass, in Msun
         */
        [[nodiscard]] auto actualMass() const { return actualMass_; }

        /**
         * @brief Advance the galaxy in time
         * @param t Time to which to advance, in yr; must be >= curTime()
         * @details
         * Draws and forms new clusters over (curTime(), t] from
         * sfr()/SimControls::cmf(), scaled down to the
         * stochastically-treated fraction SimControls::fCluster() of
         * the step's own total target mass. Of the remaining
         * (1 - fCluster()), the share at or above
         * SimControls::minStochMass() -- SimControls::fracStochMass()
         * of it -- is drawn individually too, from SimControls::imf()
         * over [minStochMass(), imf().getMax()], as new FieldStar
         * entries (mass_, feh_ from fehDist(), formTime_, deathTime_,
         * and aV_ from SimControls::avDistField() -- an independent
         * draw per star, unlike a bound cluster's own single, shared
         * aV_) appended to fieldStars() (see FieldStar's own comment);
         * the rest is folded directly into actualMass(), as the purely
         * continuous, non-clustered population's own share, with no
         * individual Cluster or FieldStar of its own.
         * Accumulates the step's own target and actual mass into
         * targetMass()/actualMass(), advances every cluster formed so
         * far (in clusters() and disruptedClusters()) to t, moves any
         * cluster that disrupted during this step from clusters() to
         * disruptedClusters(), moves any field star that has died as
         * of t from fieldStars() to deadFieldStars(), then marks
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/phot_/
         * photExtinct_/photNeb_/photNebExtinct_/lbol_/lbolCts_ as stale
         * (see specCurrent_/photCurrent_/lbolCurrent_/
         * lbolCtsCurrent_'s own comments) rather than recomputing them
         * itself -- they are instead recomputed lazily, on demand, the
         * next time spec()/specExtinct()/specNeb()/specNebExtinct()/
         * lineLum()/phot()/photExtinct()/photNeb()/photNebExtinct()/
         * lbol() is actually called -- before finally updating
         * curTime() to t.
         */
        void advance(double t);

    private:

        double curTime_ = 0.0;                  /**< Current simulation time */
        std::vector<Cluster> clusters_;          /**< Currently alive (non-disrupted) clusters */
        std::vector<Cluster> disruptedClusters_; /**< Disrupted clusters */
        std::vector<FieldStar> fieldStars_;      /**< Currently alive field stars, sorted by formTime_ -- see advance()'s own comment */
        std::vector<FieldStar> deadFieldStars_;  /**< Field stars that have died as of curTime_ */

        /**
         * @brief This galaxy's own resolved star formation rate
         * @details
         * Set once, at construction, and never reseated afterward --
         * use sfr() rather than this member directly. See SFRVar's own
         * comment for what the two alternatives mean, and the
         * constructor's own comment for exactly how this is resolved.
         */
        SFRVar sfr_;

        std::vector<double> spec_;         /**< Sum of spec() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> specExtinct_;  /**< Sum of specExtinct() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> specNeb_;      /**< Sum of specNeb() over every cluster in clusters_/disruptedClusters_, plus the continuous population's own nebular-reprocessed contribution, at the current time */
        std::vector<double> specNebExtinct_; /**< Sum of specNebExtinct() over every cluster in clusters_/disruptedClusters_, plus the continuous population's own extincted, nebular-reprocessed contribution, at the current time */
        std::vector<double> lineLum_;      /**< Sum of lineLum() over every cluster in clusters_/disruptedClusters_, plus the continuous population's own contribution, at the current time */
        std::vector<double> phot_;         /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_;  /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNeb_;      /**< Photometry of specNeb_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNebExtinct_; /**< Photometry of specNebExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;                /**< Sum of lbol() over every cluster in clusters_/disruptedClusters_, at the current time */
        double targetMass_ = 0.0;          /**< Cumulative total target stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */
        double actualMass_ = 0.0;          /**< Cumulative total actual stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */

        /**
         * @brief Whether spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_ are current as of curTime_
         * @details
         * Mirrors Cluster::specCurrent_'s own comment: true at
         * construction, set false at the end of every advance() call,
         * and back to true by spec()/specExtinct()/specNeb()/
         * specNebExtinct()/lineLum() after recomputing them.
         */
        bool specCurrent_ = true;

        /**
         * @brief Whether phot_/photExtinct_/photNeb_/photNebExtinct_ are current as of curTime_
         * @details
         * Mirrors specCurrent_'s own comment, for phot()/photExtinct()/
         * photNeb()/photNebExtinct().
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
         * @brief Update spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_ from the current cluster population
         * @details
         * Mirrors Cluster::computeSpec()'s own null-guard, but sums
         * over clusters rather than stars: does nothing if
         * SimControls::specsyn() is null. Otherwise sets spec_ to the
         * sum of spec() over every cluster in clusters_ and
         * disruptedClusters_ (forcing each cluster's own spectrum to be
         * computed, if not already current); if SimControls::extinct()
         * is also non-null, also sets specExtinct_ to the sum of
         * specExtinct() over the same clusters. If SimControls::nebular()
         * is also non-null, hands off to addClusterSpecNeb() to set
         * specNeb_/specNebExtinct_/lineLum_ the same way.
         *
         * If fCluster() < 1, hands off to addContinuousSpec() to add
         * the purely continuous population's own contribution together
         * with every currently-alive field star's own contribution --
         * see addContinuousSpec()'s own comment for why the two are
         * combined there rather than handled separately. Both this and
         * addClusterSpecNeb() are factored out into their own methods
         * purely to keep this one's own cognitive complexity down, not
         * for any reuse elsewhere.
         */
        void computeSpec();

        /**
         * @brief Set specNeb_/specNebExtinct_/lineLum_ from the current cluster population
         * @param ext SimControls::extinct(), passed through from
         *   computeSpec() rather than re-read here, since computeSpec()
         *   already has it in hand
         * @param neb SimControls::nebular(), passed through from
         *   computeSpec() likewise; must be non-null (computeSpec()
         *   only calls this when it is)
         * @details
         * Mirrors computeSpec()'s own spec_/specExtinct_ summing
         * exactly, but for specNeb_/lineLum_ (unconditionally) and
         * specNebExtinct_ (if ext is non-null), from specNeb()/
         * lineLum()/specNebExtinct() over every cluster in clusters_
         * and disruptedClusters_. Split out of computeSpec() itself
         * purely to keep that function's own cognitive complexity
         * down, not for any reuse elsewhere.
         */
        void addClusterSpecNeb(const extinct::Extinct* ext, const nebular::Nebular* neb);

        /**
         * @brief Add the purely continuous population's and every field star's own contribution to spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_
         * @param ext SimControls::extinct(), passed through from
         *   computeSpec() rather than re-read here, since computeSpec()
         *   already has it in hand
         * @param neb SimControls::nebular(), passed through from
         *   computeSpec() likewise
         * @details
         * Builds contSpec from Specsyn::specCts()'s/specAndLbolCts()'s
         * own [imf().getMin(), minStochMass()] mass range -- the purely
         * continuous, below-minStochMass() share of the non-clustered
         * population -- if SimControls::fracStochMass() < 1 (otherwise
         * contSpec starts all-zero: there is no purely continuous share
         * at all, but fieldStars_ may still be non-empty and need
         * processing below). Gets this share via
         * Specsyn::specAndLbolCts() rather than plain specCts() -- also
         * setting lbolCts_/lbolCtsCurrent_ from its own second return
         * value -- whenever SimControls::computeLbol() is true, so Lbol
         * comes along for free from the same integral; via plain
         * specCts() otherwise, leaving lbolCts_/lbolCtsCurrent_
         * untouched.
         *
         * Then adds every entry of getFieldStarProps() (evaluated via
         * Specsyn::spec() star by star, at that star's own feh_)
         * directly into this same contSpec, before extinction or
         * nebular emission is applied to it -- rather than handling
         * field stars separately (an earlier version of this class
         * did), so that both are computed on the combined (purely
         * continuous + field star) spectrum in a single pass. One
         * consequence: field stars are now attenuated by the same
         * *expected* extinction below (via Extinct::applyExtinctionCts(),
         * over SimControls::avDistField()) as the purely continuous
         * share, rather than each by its own individually-drawn
         * FieldStar::aV_ (which this function no longer reads for that
         * purpose) -- a deliberate simplification, since it is what
         * also lets field stars share the purely continuous share's own
         * nebular reprocessing below at no extra cost, rather than
         * needing their own separate getGalaxy() call.
         *
         * Adds the combined contSpec to spec_ unconditionally, and (if
         * ext is non-null) its own expected attenuation to specExtinct_.
         * If neb is non-null, also passes this same combined
         * contribution (before extinction) through neb->getGalaxy(), at
         * SimControls::fehDist()'s own expectationValue() -- getGalaxy()
         * takes a single scalar [Fe/H], not a distribution, so a
         * non-degenerate fehDist() (a genuine spread, not a fixed
         * value) is represented by its own mean here, rather than
         * attempting a full nebular emission integral over [Fe/H] --
         * adding the resulting stellar + nebular spectrum to specNeb_
         * and its own line luminosities to lineLum_; if ext is also
         * non-null, extinguishes that same nebular-reprocessed
         * contribution via applyExtinctionCts(), exactly as the plain
         * (non-nebular) contribution is, and adds it to
         * specNebExtinct_.
         *
         * Split out of computeSpec() itself purely to keep that
         * function's own cognitive complexity down.
         */
        void addContinuousSpec(const extinct::Extinct* ext, const nebular::Nebular* neb);

        /**
         * @brief Update phot_/photExtinct_/photNeb_/photNebExtinct_ from the current spec_/specExtinct_/specNeb_/specNebExtinct_
         * @details
         * Identical to Cluster::computePhot(), but reading this
         * Galaxy's own spec()/specExtinct()/specNeb()/specNebExtinct()
         * (the lazy getters, not spec_/specExtinct_/specNeb_/
         * specNebExtinct_ directly) rather than a Cluster's -- note
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
         *
         * Finally, adds every currently-alive field star's own
         * contribution (10^logL, read directly off getFieldStarProps())
         * -- independent of lbolCtsCurrent_/lbolCts_, since this is
         * cheap and direct either way, unlike the continuous
         * population's own Specsyn-mediated integral.
         */
        void computeLbol();

        /**
         * @brief The integrand for computeLbolCts()'s own standalone Lbol integral's outer (age) integral
         * @param age The stellar age, in yr, to evaluate at -- see
         *   Specsyn::continuousSpecIntegrand()'s own age parameter for
         *   the identical convention (not time -- computeLbolCts()
         *   integrates this dimension via a pdfs::PDFReflect view of
         *   sfr, pivoted so that the coordinate PDFIntegrator hands
         *   back already is age)
         * @param imf The initial mass function of the population --
         *   see computeLbolCts()'s own imf parameter; used both as the
         *   inner mass integral's own weighting PDF and, via
         *   getMin(), its own lower integration bound (see this
         *   function's own comment for its own upper bound)
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
         * segments' own bolometric luminosity against imf, over
         * [imf.getMin(), controls_.minStochMass()] (the purely
         * continuous share of the population -- see Specsyn::
         * specCts()'s own comment for why controls_.minStochMass()
         * rather than imf.getMax()), via a nested PDFIntegrator
         * (GKOrder::GK15) --
         * a local, capture-free lambda mirroring Cluster::lbolStar()'s
         * own role, rather than calling into any Specsyn (which may
         * not exist at all when this runs -- see computeLbolCts()'s
         * own comment), reading each star's own log(L/Lsun) directly
         * off its isochrone segment and summing 10^that over the
         * segments a star of that mass could belong to. No explicit
         * live/dead mass check is needed: exactly as in
         * continuousSpecIntegrand(), a dead mass is simply never
         * visited, since it falls in none of the isochrone's own
         * segment domains.
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
         * PDFIntegrator over age alone whose own integrand --
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
         * (1 - fCluster()) * (1 - fracStochMass()) for the purely
         * continuously-treated share of the population, matching
         * Specsyn::specAndLbolCts()'s own identical scaling --
         * necessary for the two to agree when both are exercised for
         * the same population (see testContinuousPopLbolStandaloneMatchesSpec
         * in tests/core/testGalaxy.cpp). Sets lbolCtsCurrent_ to true
         * afterward.
         */
        void computeLbolCts();

        /**
         * @brief Evaluate every currently-alive field star's stellar properties
         * @return A vector, one element per entry in fieldStars() (same
         *   order), of that star's properties at curTime() -- see
         *   tracks::Tracks2D::getStar()/tracks::Tracks3D::getStar()'s
         *   own comment for what a StarData holds
         * @details
         * If the simulation has a fixed [Fe/H] (SimControls::constFeH()),
         * loops over fieldStars() directly, calling
         * SimControls::tracks2D()'s own getStar(mass_, log10(curTime()
         * - formTime_)) once per star -- cheap, since tracks2D() is a
         * single, already-built Tracks2D slice shared across every
         * call.
         *
         * Otherwise, evaluating SimControls::tracks()'s own
         * getStar(mass_, logT, feh_) at each star's own (in general,
         * distinct) feh_ directly would rebuild a fresh
         * tracks::Tracks3D::sliceConstZ() slice on nearly every call
         * (its own single-entry cache only helps when consecutive
         * calls share the same feh -- see its own comment), which is
         * prohibitively expensive for a field-star population of any
         * size. Instead, rounds each star's own feh_ to the nearest
         * multiple of 0.25, sorts the stars by that rounded value
         * (mapping back to the original order once every star has
         * been evaluated), and evaluates every star at its own rounded
         * feh_ (not its raw, drawn value) -- consecutive stars sharing
         * the same rounded feh_ then hit sliceConstZ()'s own cache, so
         * the number of slices actually built is bounded by the number
         * of distinct rounded feh_ values present, not the number of
         * stars. This evaluates every field star at a [Fe/H] snapped
         * to the nearest 0.25 dex grid point rather than its own exact
         * drawn value -- a deliberate, small approximation traded for
         * tractable cost.
         *
         * Either way, logT (log10 of the star's own age,
         * curTime() - formTime_) is floored at
         * SimControls::tracks()'s own logTMin(), mirroring
         * Cluster::advance()'s and Specsyn::continuousSpecIntegrand()'s
         * own identical floor, to avoid taking log10(0) for a field
         * star whose formTime_ is exactly curTime_ (formed during the
         * very advance() call that produced this evaluation).
         */
        [[nodiscard]] auto getFieldStarProps() const -> std::vector<specsyn::Specsyn::StarData>;

    };

} // namespace core

#endif // GALAXY_HPP
