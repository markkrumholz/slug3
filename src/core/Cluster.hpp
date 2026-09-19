/**
 * @file Cluster.hpp
 * @author Mark Krumholz
 * @brief A class to represent a mono-age star cluster
 * @date 2026-07-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/RngThread.hpp"
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace yields
{
    class Yields;
} // namespace yields

namespace core
{

    /**
     * @brief A class to represent a mono-age stellar cluster
     */
    class Cluster
    {
    public:

        // Shorten type names
        using Segment = interp::Interpolator1D<
            static_cast<size_t>(tracks::FieldIdx::nTrackQty)>;
        using Interp1dPtr = std::vector<std::unique_ptr<Segment>>;
        using Track2DVar = std::variant<tracks::Tracks2D,
            std::shared_ptr<const tracks::Tracks2D>>;

        /**
         * @brief Initialize a cluster
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param controls Simulation controls (physics settings and
         *   control-flow/integrator-tolerance settings together);
         *   stored by reference, so it must outlive this Cluster --
         *   see controls_'s own comment
         */
        Cluster(unsigned long uid,
            double mass,
            double time,
            const io::SimControls& controls);

        /**
         * @brief Reconstruct a cluster's stellar masses from a previously recorded rng state
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param controls Simulation controls; see the primary
         *   constructor's own comment
         * @param rngState The rng state to draw star masses from --
         *   typically one previously returned by rngState() (e.g. read
         *   back from an output file), so that the resulting
         *   starMasses()/birthMass() come out bitwise identical to the
         *   original cluster's
         * @details
         * Sets every member the same way the primary constructor does,
         * except for feH_, aV_, m_, and birthMass_: rngState_ is set to
         * rngState directly, since that is this cluster's own birth
         * state by construction, rather than one captured live.
         * feH_, aV_ (if SimControls::avDist() is valid), and m_ are
         * then all drawn together, in the same order the primary
         * constructor draws them, from rngState rather than the live
         * rng stream -- saving the live state first and restoring it
         * immediately afterward, so this constructor has no lasting
         * effect on the ambient rng stream. All of these draws, not
         * just m_'s, must be replayed from rngState for the result
         * to come out bitwise identical to the original cluster's:
         * PDF::draw() consumes rng state (via a
         * std::discrete_distribution used to pick a segment) even for
         * a single-segment/delta [Fe/H] distribution, so drawing feH_
         * (and aV_) live first would already have advanced the stream
         * before m_'s draw ever saw rngState, decoupling the draws
         * from the sequence that originally produced them. birthMass_
         * (which starts at 0.0, overwritten once m_ is drawn) is then
         * reduced from m_ before m_ is sorted, matching the primary
         * constructor's own order (there, forced by computing
         * birthMass_ in the member initializer list, which runs
         * before the constructor body's own sort) -- reducing in a
         * different order would reproduce the same set of masses but
         * not necessarily the same birthMass_ bit-for-bit, since
         * floating-point addition is not associative.
         */
        Cluster(unsigned long uid,
            double mass,
            double time,
            const io::SimControls& controls,
            const utils::RngState& rngState);

        // Observers

        /**
         * @brief Return the state of the rng at this cluster's birth
         * @return A const reference to the serialized rng state (see
         *   utils::RngThread::getState()) in effect immediately before
         *   this cluster's own stochastic draws, sufficient to exactly
         *   reproduce its starMasses()/birthMass() later via the
         *   rngState-accepting constructor overload
         */
        [[nodiscard]] auto rngState() const -> const utils::RngState& { return rngState_; }

        /**
         * @brief Return the cluster's unique identifier
         * @return Unique ID of cluster
         */
        [[nodiscard]] auto uid() const { return uid_; }

        /**
         * @brief Return the cluster's target mass
         * @return Target cluster mass in Msun
         */
        [[nodiscard]] auto targetMass() const { return targetMass_; }

        /**
         * @brief Return the cluster's actual mass at birth
         * @return Actual cluster mass at birth in Msun
         */
        [[nodiscard]] auto birthMass() const { return birthMass_; }

        /**
         * @brief Return the cluster's formation time
         * @return Cluster formation time
         */
        [[nodiscard]] auto formTime() const { return formTime_; }

        /**
         * @brief Return the cluster's [Fe/H]
         * @return [Fe/H] of cluster
         */
        [[nodiscard]] auto feH() const { return feH_; }

        /**
         * @brief Return the cluster's V-band extinction
         * @return A_V, in magnitudes, drawn from
         *   SimControls::avDist() at construction, or 0 if
         *   SimControls::avDist() is not valid (extinct.AV was not
         *   given in the input deck)
         */
        [[nodiscard]] auto aV() const { return aV_; }

        /**
         * @brief Return the current list of living stellar masses
         * @return Masses of currently alive stars in Msun
         */
        [[nodiscard]] auto starMasses() const -> const auto& { return m_; }

        /**
         * @brief Return the list of dead stellar masses
         * @return Masses of dead stars in Msun
         */
        [[nodiscard]] auto deadStarMasses() const -> const auto& { return mDead_; }

        /**
         * @brief Return the death time of every currently-alive star
         * @return Death time (formTime() + the star's own lifetime at
         *   this cluster's [Fe/H]) of every entry of starMasses(), in
         *   the same order -- see the constructor's own comment for
         *   why that order is not, in general, starMasses()'s own
         *   ascending-mass order
         */
        [[nodiscard]] auto starDeathTimes() const -> const auto& { return tDeath_; }

        /**
         * @brief Return the death times of stars that died in the most recent advance() call
         * @return Death time of every entry of deadStarMasses(), in
         *   the same order -- both are cleared and repopulated
         *   together every advance() call, see updateLivingStars()'s
         *   own comment
         */
        [[nodiscard]] auto deadStarDeathTimes() const -> const auto& { return tDied_; }

        /**
         * @brief Get the stellar tracks at this cluster's [Fe/H]
         * @return A const reference to a Tracks2D object at this
         *         cluster's [Fe/H]
         * @details
         * If the simulation has a fixed [Fe/H], this returns a
         * reference to the slice shared by SimControls (and thus by
         * every Cluster in the simulation). Otherwise it returns a
         * reference to the slice computed for, and owned by, this
         * Cluster alone. Either way, callers can use the returned
         * Tracks2D uniformly without needing to know which case
         * applies.
         */
        [[nodiscard]] auto tracks() const -> const tracks::Tracks2D&;

        /**
         * @brief Return the cluster's continuously-sampled spectrum
         * @return A const reference to the spectrum of the
         *   non-stochastically-sampled part of the population, on
         *   the wavelength grid of the simulation's spectral
         *   synthesizer, or an empty vector if no spectral
         *   synthesizer was requested (SimControls::specsyn() is
         *   null)
         * @details
         * Computed lazily: if advance() has run since spec_/specExtinct_
         * were last computed, this triggers computeSpec() (which
         * computes spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_
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
         * @brief Return the cluster's extincted spectrum
         * @return A const reference to spec(), attenuated by aV()
         *   through SimControls::extinct(), on the wavelength grid
         *   returned by SimControls::extinct()'s own wl(); an empty
         *   vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or spec() itself is
         *   empty (no spectral synthesizer was requested)
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
         * @brief Return the cluster's photometry
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
         * as of the last advance() first.
         */
        [[nodiscard]] auto phot() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return phot_;
        }

        /**
         * @brief Return the cluster's extincted photometry
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
         * @brief Return the cluster's stellar + nebular spectrum
         * @return A const reference to spec(), with nebular continuum
         *   and line emission added via SimControls::nebular()'s own
         *   getCluster(), on the same wavelength grid as spec(); an
         *   empty vector if no nebular emission grid was requested
         *   (SimControls::nebular() is null) or spec() itself is empty
         *   (no spectral synthesizer was requested)
         * @details
         * Computed lazily -- see spec()'s own comment; shares the same
         * specCurrent_ flag, since a single computeSpec() call computes
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ together.
         */
        [[nodiscard]] auto specNeb() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specNeb_;
        }

        /**
         * @brief Return the cluster's extincted stellar + nebular spectrum
         * @return A const reference to specNeb(), attenuated by aV()
         *   through SimControls::extinct(), on the wavelength grid
         *   returned by SimControls::extinct()'s own wl(); an empty
         *   vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or specNeb() itself is
         *   empty (no nebular emission grid or no spectral synthesizer
         *   was requested)
         * @details
         * Computed lazily -- see specNeb()'s own comment.
         */
        [[nodiscard]] auto specNebExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specNebExtinct_;
        }

        /**
         * @brief Return the cluster's nebular emission line luminosities
         * @return A const reference to the luminosity of each of
         *   SimControls::nebular()'s own lineWl() lines, in erg/s, in
         *   the same order; an empty vector if no nebular emission
         *   grid was requested (SimControls::nebular() is null) or
         *   spec() itself is empty (no spectral synthesizer was
         *   requested)
         * @details
         * Computed lazily -- see specNeb()'s own comment.
         */
        [[nodiscard]] auto lineLum() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return lineLum_;
        }

        /**
         * @brief Return the cluster's extincted nebular emission line luminosities
         * @return A const reference to lineLum(), attenuated by aV()
         *   through SimControls::extinct(); an empty vector if no
         *   extinction curve was requested (SimControls::extinct() is
         *   null) or lineLum() itself is empty (no nebular emission
         *   grid or no spectral synthesizer was requested)
         * @details
         * Computed lazily -- see lineLum()'s own comment.
         */
        [[nodiscard]] auto lineLumExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return lineLumExtinct_;
        }

        /**
         * @brief Return the cluster's stellar + nebular photometry
         * @return A const reference to the photometric value computed
         *   from specNeb() by each filter in SimControls::filters(),
         *   in the same order as phot(); an empty vector if no nebular
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
         * @brief Return the cluster's extincted stellar + nebular photometry
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
         * @brief Return the cluster's bolometric luminosity
         * @return The population's total bolometric luminosity, in
         *   Lsun, at the current time, or 0 if SimControls::computeLbol()
         *   is false (Lbol was never requested)
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
         * @brief Return this cluster's total nucleosynthetic yield of each isotope
         * @return A const reference to yields_: intended to be the
         *   total yield (Msun) of each isotope in controls().yields()'s
         *   own isotopes(), summed over this cluster's stars -- laid
         *   out as one channel-major row (of isotopes().size() entries
         *   each) per entry in controls().yields()->yieldChannels() if
         *   controls().yieldsChannelDecomposed() is true, or just
         *   isotopes().size() combined totals (summed over every
         *   channel) if false; an empty vector if no yield channels
         *   were requested (controls().yields() is null)
         * @details
         * Unlike spec()/phot()/lbol() (see spec()'s own comment), not
         * computed lazily here: advance() itself calls computeYields()
         * eagerly at the end of every call (see its own comment for
         * why), so yields_ is already current by the time this is
         * called -- rather than being recomputed from scratch, yields_
         * accumulates the contribution of every star that has died
         * over this cluster's whole lifetime so far, and stays at the
         * all-zero vector it was sized to at construction (see its own
         * comment) until advance() has run at least once.
         */
        [[nodiscard]] auto yields() const -> const auto&
        {
            return yields_;
        }

        /**
         * @brief Return whether the cluster has disrupted
         * @return True if the cluster has disrupted
         */
        [[nodiscard]] auto isDisrupted() const { return isDisrupted_; }

        /**
         * @brief Advance the cluster in time
         * @param t Time to which to advance, in yr
         * @details
         * Updates the living/dead star lists and the isochrone for the
         * new time, then marks spec_/specExtinct_/specNeb_/
         * specNebExtinct_/lineLum_/lineLumExtinct_/phot_/photExtinct_/photNeb_/
         * photNebExtinct_/lbol_ as stale (see specCurrent_/
         * photCurrent_/lbolCurrent_'s own comments) rather than
         * recomputing them itself -- they are instead recomputed
         * lazily, on demand, the next time spec()/specExtinct()/
         * specNeb()/specNebExtinct()/lineLum()/lineLumExtinct()/phot()/
         * photExtinct()/photNeb()/photNebExtinct()/lbol() is actually
         * called.
         */
        void advance(double t);
        
    private:

        // Data set on creation
        utils::RngState rngState_;  /**< State of the rng at birth */
        unsigned long uid_;         /**< Unique identifier */
        double targetMass_;         /**< Target mass */
        double formTime_;           /**< Formation time */
        double feH_;                /**< [Fe/H] of cluster */
        double aV_;                 /**< V-band extinction, in magnitudes, drawn from SimControls::avDist() (0 if that PDF is not valid) */

        /**
         * @brief Simulation controls (physics and control-flow settings) this cluster was built from
         * @details
         * Read live wherever this cluster needs a physics setting or
         * an integrator tolerance (e.g. computeLbol()'s own
         * utils::PDFIntegrator), rather than snapshotted at
         * construction -- so a change to controls_'s own tolerances
         * after this Cluster is built takes effect the next time it
         * integrates, with no need to rebuild the Cluster.
         */
        std::reference_wrapper<const io::SimControls> controls_;

        // Masses
        std::vector<double> m_;     /**< Stellar masses */
        std::vector<double> mDead_; /**< Mass of dead stars */
        double birthNonStochMass_;  /**< Mass in non-stochastic part of IMF at birth */
        double birthMass_;          /**< Actual mass at birth */

        /**
         * @brief Death time of every currently-alive star in m_, in the same order
         * @details
         * Set at construction to formTime_ + this cluster's own
         * starLifetime() for every entry of m_, then m_ and tDeath_ are
         * jointly sorted by tDeath_, descending, so that the stars
         * that will die soonest (smallest tDeath_) end up at the back
         * of both lists -- letting updateLivingStars() find and remove
         * every newly-dead star with a simple backward scan, rather
         * than the mass-range-based search this class used before this
         * member existed. Every entry popped off the back of both m_
         * and tDeath_ by updateLivingStars() is moved to mDead_/tDied_
         * respectively, keeping those two also in the same order as
         * each other.
         */
        std::vector<double> tDeath_;

        /**
         * @brief Death time of every star in mDead_, in the same order
         * @details
         * Empty at construction (nothing has died yet); cleared and
         * repopulated by every updateLivingStars() call exactly as
         * mDead_ itself is -- see tDeath_'s and updateLivingStars()'s
         * own comments.
         */
        std::vector<double> tDied_;

        // Times
        double disruptTime_;        /**< Time when this cluster will disrupt */
        double curTime_;            /**< Current time */

        // Other state information
        bool isDisrupted_ = false;  /**< Has this cluster disrupted */
        bool advanced_ = false;     /**< Has advance() ever run its body (as opposed to a same-time no-op)? */
        Interp1dPtr isochrone_;     /**< Isochrone for the current time */
        std::vector<double> spec_;  /**< Spectrum of the continuously-sampled part of the population at the current time */
        std::vector<double> specExtinct_; /**< spec_ attenuated by aV_ through SimControls::extinct(), at the current time */
        std::vector<double> specNeb_; /**< spec_ plus nebular continuum and line emission, via SimControls::nebular()'s own getCluster(), at the current time */
        std::vector<double> specNebExtinct_; /**< specNeb_ attenuated by aV_ through SimControls::extinct(), at the current time */
        std::vector<double> lineLum_; /**< Luminosity of each of SimControls::nebular()'s own lineWl() lines, in erg/s, at the current time */
        std::vector<double> lineLumExtinct_; /**< lineLum_ attenuated by aV_ through SimControls::extinct(), at the current time */
        std::vector<double> phot_;  /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_; /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNeb_; /**< Photometry of specNeb_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNebExtinct_; /**< Photometry of specNebExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;         /**< Bolometric luminosity of the population, in Lsun, at the current time */
        std::vector<double> yields_; /**< Total nucleosynthetic yield of each isotope, in Msun, accumulated over every star that has died so far in this cluster's lifetime, laid out per yields()'s own comment -- sized to all zeros at construction if controls().yields() is non-null, empty otherwise; see computeYields()'s own comment for how it accumulates */

        /**
         * @brief Whether spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ are current as of curTime_
         * @details
         * True at construction (spec_/specExtinct_/specNeb_/
         * specNebExtinct_/lineLum_/lineLumExtinct_'s own empty in-class defaults are
         * already the correct, current value before advance() has ever
         * run), and after every spec()/specExtinct()/specNeb()/
         * specNebExtinct()/lineLum()/lineLumExtinct() call recomputes
         * them; set back to
         * false at the end of every advance() call, so the next such
         * call recomputes them lazily -- see advance()'s and spec()'s
         * own comments. Letting spec()/phot()/lbol() compute only when
         * actually requested, rather than unconditionally inside
         * advance() itself, avoids e.g. computing this cluster's own
         * photometry (an unwanted expense) when a galaxy this cluster
         * belongs to needs only this cluster's spectrum, to build the
         * galaxy's own summed spectrum, but per-cluster photometry
         * output was disabled (output.write_cluster_phot = false).
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
         * @brief Simulation time through which yields_ has been updated
         * @details
         * Initialized to 0 rather than curTime_ (formTime_'s own
         * value); computeYields() itself treats any lastYieldTime_
         * before formTime_ as formTime_ (see its own comment), so this
         * has no effect beyond documenting that nothing has died yet.
         *
         * advance() itself sets this to curTime_ right after every
         * call to computeYields() (see both of their own comments for
         * why this can't be left to a lazy yields() call the way
         * spec_/phot_/lbol_'s own analogous staleness flags are):
         * computeYields() does not recompute yields_ from scratch when
         * it runs, it only adds the contribution of whatever died
         * between lastYieldTime_ and curTime_ -- for the stochastic
         * population, this relies on mDead_, which only ever holds the
         * deaths from the single most recently advance() call (see
         * updateLivingStars()'s own comment), so computeYields() must
         * run before mDead_'s own contents are overwritten by the next
         * advance() call, not merely before yields_ is next read.
         */
        double lastYieldTime_ = 0.0;

        /**
         * Tracks for this cluster's [Fe/H]: either owned outright (when
         * the simulation has a variable [Fe/H], so each cluster needs
         * its own slice) or a shared_ptr to the slice SimControls
         * itself owns (when [Fe/H] is fixed for the whole simulation)
         * -- a shared_ptr, not a reference, so this Cluster keeps that
         * slice alive on its own even if SimControls later recomputes
         * its own copy (see SimControls::tracks2D()'s own comment).
         * Use tracks() rather than this member directly.
         */
        Track2DVar tracks_;         /**< 2d track holder */

        /**
         * @brief Update the lists of living and dead stars
         * @details
         * Clears mDead_/tDied_, then pops stars off the back of m_/
         * tDeath_ (moving each into mDead_/tDied_) for as long as the
         * back of tDeath_ is less than curTime_ -- since m_/tDeath_
         * are kept jointly sorted by tDeath_, descending (see
         * tDeath_'s own comment), every remaining entry is guaranteed
         * to have a tDeath_ still >= curTime_ once this stops, with no
         * need to scan the rest of the list.
         */
        void updateLivingStars();

        /**
         * @brief Update spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ from the current isochrone and star lists
         * @details
         * Does nothing if SimControls::specsyn() is null (no spectral
         * synthesizer was requested). Otherwise sets spec_ to the sum
         * of the continuously-sampled (non-stochastic) part of the
         * population, if any, and each individually-sampled
         * (stochastic) star in m_. If SimControls::nebular() is
         * non-null, then sets specNeb_/lineLum_ from spec_ via
         * SimControls::nebular()'s own getCluster(), passing this
         * cluster's own feH_ and age (curTime_ - formTime_). If
         * SimControls::extinct() is also non-null, sets specExtinct_
         * from spec_, and -- if a nebular emission grid was requested
         * -- specNebExtinct_ from specNeb_ via
         * SimControls::extinct()'s own applyExtinction(), and
         * lineLumExtinct_ from lineLum_ via its own
         * applyExtinctionLines().
         */
        void computeSpec();

        /**
         * @brief Update phot_/photExtinct_/photNeb_/photNebExtinct_ from the current spec_/specExtinct_/specNeb_/specNebExtinct_
         * @details
         * Does nothing if SimControls::filters() is null (no filter
         * collection was requested). Otherwise sets phot_ from spec()
         * (the lazy getter, not spec_ directly, so that calling phot()
         * alone -- with no prior call to spec() -- still computes a
         * current spectrum first, regardless of call order); if
         * SimControls::extinct() is also non-null, also sets
         * photExtinct_ from specExtinct() likewise. If
         * SimControls::nebular() is non-null, likewise sets photNeb_
         * from specNeb(), and -- if SimControls::extinct() is also
         * non-null -- photNebExtinct_ from specNebExtinct().
         */
        void computePhot();

        /**
         * @brief Update lbol_ from the current isochrone and star lists
         * @details
         * Does nothing if SimControls::computeLbol() is false (Lbol was
         * never requested), mirroring computeSpec()/computePhot()'s own
         * null-guards. Otherwise mirrors computeSpec()'s own two-part
         * structure: sums 10^logL (logL read directly off the isochrone
         * via Segment's single-quantity operator()(x, idx), rather than
         * interpolating every quantity) over each individually-sampled
         * (stochastic) star in m_ with a valid isochrone segment, then
         * adds the continuously-sampled (non-stochastic) part of the
         * population's own contribution, integrated against the IMF
         * over each isochrone segment via lbolStar() and
         * utils::PDFIntegrator -- the same per-segment integration
         * Specsyn::specCts() uses, and for the same reason (an
         * isochrone may have gaps between segments that the quadrature
         * routine has no way to know to avoid).
         */
        void computeLbol();

        /**
         * @brief Add the yield of every star that died since lastYieldTime_ into yields_
         * @details
         * Unlike computeSpec()/computePhot()/computeLbol() (each
         * called lazily from its own accessor), called eagerly from
         * advance() itself, at the end of every call -- see
         * lastYieldTime_'s own comment for why. Does nothing if
         * controls().yields() is null (no yield channels were
         * requested), mirroring computeSpec()/computePhot()/
         * computeLbol()'s own null-guards. Otherwise handles the
         * stochastic and non-stochastic parts of the population
         * separately, adding both into yields_ -- see
         * lastYieldTime_'s own comment for why this accumulates onto
         * yields_ rather than overwriting it, unlike computeSpec()/
         * computePhot()/computeLbol().
         *
         * Stochastic (individually-sampled) stars: loops over mDead_/
         * tDied_ together (the stars that died during the most recent
         * advance() call, and their own death times -- see
         * updateLivingStars()'s own comment) and, for each, adds
         * controls().yields()'s own yield(mass, feH_, dtDecay) (if
         * controls().yieldsChannelDecomposed()) or yieldSum(mass,
         * feH_, dtDecay) (otherwise) into yields_, where dtDecay is
         * curTime_ minus that star's own death time if
         * controls().noDecay() is false, or 0 if it is true (matching
         * yieldStar()'s own identical dtDecay convention for the
         * non-stochastic population below).
         *
         * Continuously-sampled (non-stochastic) stars: does nothing if
         * birthNonStochMass_ is 0 (no continuously-sampled population
         * at all), mirroring computeLbol()'s own identical guard.
         * Otherwise finds the live mass range at lastYieldTime_ (via
         * tracks().liveMassRange(), clamping lastYieldTime_ up to
         * formTime_ first, so the very first call -- lastYieldTime_
         * still at its initial 0 -- reads the live range at this
         * cluster's own birth, not simulation time 0) and subtracts
         * from it the live mass range now (read directly off
         * isochrone_'s own segments, rather than calling
         * tracks().liveMassRange() a second time). The result is the
         * set of mass ranges that were alive at lastYieldTime_ but are
         * dead now -- possibly several disjoint ranges -- each then
         * clipped to lie below controls().minStochMass() (the
         * non-stochastic population's own upper mass limit; the part
         * above it, if any, belongs to the stochastic stars already
         * handled above, via mDead_). For each surviving, non-empty
         * range [m0, m1], integrates yieldStar() (see its own comment)
         * against controls().imf() over [m0, m1] via PDFIntegrator,
         * exactly as computeLbol() integrates lbolStar() over the
         * analogous non-stochastic mass range, and adds
         * birthNonStochMass_ times that integral into yields_ -- the
         * same "integrate a per-unit-mass quantity against the
         * normalized IMF, then scale by the population's actual total
         * mass" pattern computeLbol()/Specsyn::specCts() both use.
         */
        void computeYields();

        /**
         * @brief Bolometric luminosity of a single star, given its mass and isochrone segment
         * @param m Stellar mass, in Msun; must lie within segment's
         *   valid domain (segment.xMin() <= m <= segment.xMax())
         * @param segment A single isochrone segment (one element of
         *   the Isochrone returned by Tracks2D::getIsochrone) to
         *   evaluate at mass m
         * @return 10^logL at mass m, in Lsun, wrapped in a
         *   single-element array
         * @details
         * Exists so computeLbol() can hand it to utils::PDFIntegrator,
         * which expects a callable taking the integration variable
         * (here, mass) as its first argument and returning a fixed- or
         * dynamically-sized container of doubles -- mirroring
         * Specsyn::specWl()'s own role for specCts(), but wrapped in a
         * std::array<double, 1> rather than a per-wavelength vector,
         * since there is only one quantity (Lbol) here, not one per
         * wavelength. Static since it needs no instance state, which
         * lets computeLbol() hand PDFIntegrator a plain function
         * pointer rather than a pointer to member function.
         */
        [[nodiscard]] static auto lbolStar(double m, const Segment& segment) -> std::array<double, 1>;

        /**
         * @brief Nucleosynthetic yield of a single star, given its mass and [Fe/H]
         * @param m Stellar mass, in Msun
         * @param feH [Fe/H] to evaluate the yield at
         * @param yields The Yields to evaluate -- controls().yields(),
         *   passed explicitly rather than read from controls_ directly,
         *   for the same reason segment is passed explicitly to
         *   lbolStar() rather than read off isochrone_
         * @param decomposed Whether to keep every channel's own
         *   contribution separate (yields.yield()) or combine them all
         *   into one per-isotope total (yields.yieldSum()) -- see
         *   controls().yieldsChannelDecomposed()'s own comment
         * @param controls This cluster's own controls_, passed
         *   explicitly for the same reason yields is -- used only for
         *   noDecay(), to decide whether dtDecay below is computed at
         *   all
         * @param curTime This cluster's own curTime_ at the time of
         *   the call, passed explicitly for the same reason
         * @param formTime This cluster's own formTime_, passed
         *   explicitly for the same reason
         * @param tracks2D This cluster's own tracks() -- passed
         *   explicitly (rather than read via controls.tracks(),
         *   Yields's own SimControls-owned Tracks3D) because
         *   Tracks3D::starLifetime() reads through
         *   Mesh3DInterpolator::sliceConstZ()'s single, mutable,
         *   not-thread-safe cache, which this method -- called from
         *   inside SimCluster::runTrial()'s own OpenMP-parallelized
         *   loop -- cannot safely touch; tracks2D, already sliced once
         *   via the thread-safe sliceConstFeH()/sliceConstZCopy() this
         *   cluster's own tracks_ is built from, has no such cache to
         *   race on
         * @return yields.yield(m, feH, dtDecay).second if decomposed,
         *   or yields.yieldSum(m, feH, dtDecay) otherwise -- either
         *   way, a vector the same length as yields_ itself
         * @details
         * Exists so computeYields() can hand it to utils::PDFIntegrator,
         * mirroring lbolStar()'s own identical role for computeLbol()
         * -- see its own comment. Static for the same reason lbolStar()
         * is: it needs no instance state, so computeYields() can hand
         * PDFIntegrator a plain function pointer.
         *
         * dtDecay is 0 if controls.noDecay() is true. Otherwise it is
         * curTime - formTime - tracks2D.starLifetime(m) -- the time
         * elapsed since this particular mass m (part of the
         * continuously-sampled population, so it has no tDeath_ entry
         * of its own the way a stochastic star does) died, mirroring
         * computeYields()'s own identical dtDecay convention for the
         * stochastic population, which reads it directly out of
         * tDied_ instead.
         */
        [[nodiscard]] static auto yieldStar(double m, double feH,
            const yields::Yields& yields, bool decomposed,
            const io::SimControls& controls, double curTime, double formTime,
            const tracks::Tracks2D& tracks2D) -> std::vector<double>;

    };

} // namespace core

#endif // CLUSTER_HPP
