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
#include "../pdfs/PDFReflect.hpp"
#include "../pdfs/PDFSegmentPowerlaw.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../utils/PDFIntegrator.hpp"
#include "Cluster.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numbers>
#include <optional>
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
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_/phot_/
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
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ together.
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
         * @brief Return the galaxy's extincted nebular emission line luminosities
         * @return A const reference to lineLum(), attenuated by
         *   extinction; an empty vector if no extinction curve was
         *   requested (SimControls::extinct() is null) or lineLum()
         *   itself is empty (no nebular emission grid or no spectral
         *   synthesizer was requested)
         * @details
         * Computed lazily -- see lineLum()'s own comment.
         */
        [[nodiscard]] auto lineLumExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return lineLumExtinct_;
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
         * @brief Return this galaxy's total nucleosynthetic yield of each isotope
         * @return A const reference to yields_ -- see
         *   Cluster::yields()'s own comment for the general shape/
         *   layout this mirrors
         * @details
         * Unlike spec()/phot()/lbol(), not computed lazily here --
         * mirrors Cluster::yields() exactly (see its own comment):
         * advance() itself calls computeYields() eagerly at the end of
         * every call, so yields_ is already current by the time this
         * is called, and stays at the all-zero vector it was sized to
         * at construction (see its own comment) until advance() has
         * run at least once.
         */
        [[nodiscard]] auto yields() const -> const auto&
        {
            return yields_;
        }

        /**
         * @brief Return the cumulative number of supernovae in this galaxy up to the current time
         * @return Number of supernovae that have occurred in this
         *   galaxy from its formation up to the current time: the sum
         *   of Cluster::cumSNe() over every cluster in clusters() and
         *   disruptedClusters(), plus fieldStarStochSNe_ (from the
         *   individually-sampled field stars) and fieldStarNonStochSNe_
         *   (from the continuously-sampled field population)
         * @details
         * Like Cluster::cumSNe() (see its own comment), a double
         * rather than an integer type, since the non-stochastic
         * contributions are in general non-integer, and not computed
         * lazily here: advance() itself updates every cluster's own
         * count, and calls computeFeedback() to update
         * fieldStarStochSNe_ and fieldStarNonStochSNe_, eagerly at
         * every call.
         */
        [[nodiscard]] auto cumSNe() const -> double
        {
            double nSN = static_cast<double>(fieldStarStochSNe_) + fieldStarNonStochSNe_;
            for (const auto& cluster : clusters_) { nSN += cluster.cumSNe(); }
            for (const auto& cluster : disruptedClusters_) { nSN += cluster.cumSNe(); }
            return nSN;
        }

        /**
         * @brief Return the galaxy's total stellar wind mass flux
         * @return The summed mass loss rate, in Msun/yr, at the current
         *   time, of every cluster in clusters() and
         *   disruptedClusters() (via Cluster::mDotWind()) and every
         *   currently-alive field star; 0 until advance() has run
         * @details
         * Updated by computeFeedback() -- see its own comment.
         * advance() already calls computeFeedback() eagerly at the end
         * of every call, so this is normally current already; if
         * lastFeedbackTime_ is somehow not curTime_, it calls
         * computeFeedback() first, mirroring Cluster::mDotWind(). Does
         * not yet include the purely continuous (non-clustered,
         * non-stochastic) population.
         */
        [[nodiscard]] auto mDotWind() -> double
        {
            updateFeedback();
            return mDotWind_;
        }

        /**
         * @brief Return the galaxy's total stellar wind momentum flux
         * @return The summed wind momentum flux, in g cm s^-2, at the
         *   current time, over the same stars as mDotWind()
         * @details
         * See mDotWind()'s own comment.
         */
        [[nodiscard]] auto pDotWind() -> double
        {
            updateFeedback();
            return pDotWind_;
        }

        /**
         * @brief Return the galaxy's total stellar wind energy flux
         * @return The summed wind kinetic energy flux, in erg s^-1, at
         *   the current time, over the same stars as mDotWind()
         * @details
         * See mDotWind()'s own comment.
         */
        [[nodiscard]] auto eDotWind() -> double
        {
            updateFeedback();
            return eDotWind_;
        }

        /**
         * @brief The instantaneous rate at which the continuous population returns each isotope, at a given time and [Fe/H]
         * @param t Simulation time, in yr, since this galaxy's own
         *   formation (time 0) -- not necessarily curTime(); this is a
         *   standalone calculation, independent of advance()/curTime()
         * @param feh [Fe/H] to evaluate at
         * @return The total rate (Msun/yr) at which the continuously
         *   sampled population -- over this galaxy's own star formation
         *   history sfr() from 0 to t, and all at [Fe/H] feh -- returns
         *   each isotope to the ISM at t, laid out as yields() (one
         *   channel-major row per isotope if
         *   controls().yieldsChannelDecomposed(), or isotopes().size()
         *   combined entries otherwise), decayed forward to curTime_
         *   unless controls().noDecay(). Excludes the stochastically
         *   sampled (cluster and field star) population, and the
         *   1 - fCluster() share applied by computeYields(). An empty
         *   vector if controls().yields() is null.
         * @details
         * deathRate() with each dying star's own yields as its
         * quantity, radioactive decay forward to curTime_ as its hook,
         * and a fixed [Fe/H] -- see continuousDeathQuantity()'s own
         * comment. The decay uses dtDecay = curTime_ - t, so assumes
         * t <= curTime_: on a Galaxy not yet advance()d past t it would
         * run decay backward in time, which Yields::applyDecay() does
         * not reject but which is not physically meaningful.
         */
        [[nodiscard]] auto yieldsRate(double t, double feh) const -> std::vector<double>;

        /**
         * @brief The instantaneous rate at which the continuous population returns each isotope, at a given time, averaged over [Fe/H]
         * @param t Simulation time, in yr -- see the (t, feh)
         *   overload's own comment
         * @return As the (t, feh) overload, but averaged over
         *   SimControls::fehDist() rather than at a single [Fe/H]
         * @details
         * deathRate() as in the (t, feh) overload, but with its [Fe/H]
         * integral innermost (fehAveragedDeathIntegrand()), so that
         * decay is applied once, to the [Fe/H]-averaged rate -- see
         * continuousDeathQuantity()'s own comment. For a degenerate
         * fehDist(), identical to the (t, feh) overload at its value.
         */
        [[nodiscard]] auto yieldsRate(double t) const -> std::vector<double>;

        /**
         * @brief A per-time hook for continuousDeathQuantity() that leaves each instantaneous rate unchanged
         * @details
         * The default for continuousDeathQuantity()'s own G -- for a
         * quantity (e.g. a supernova count) that needs no processing
         * between its release and the end of the integration, unlike
         * radioactive yields, whose hook applies decay.
         */
        struct NoDeathRateHook
        {
            /**
             * @brief Leave rate unchanged
             * @param t Simulation time, in yr, at which rate applies
             * @param rate The instantaneous rate at t
             */
            void operator()(double /*t*/, std::vector<double>& /*rate*/) const {}
        };

        /**
         * @brief The cumulative amount of a quantity released at stellar death by the continuous population between two times
         * @tparam F Callable as f(m, feh), m a stellar mass in Msun and
         *   feh its [Fe/H], returning a std::vector<double> of n values:
         *   the amount of each quantity one star of that mass and [Fe/H]
         *   releases when it dies (e.g. its yields, or 1 or 0 for
         *   whether it produces a supernova)
         * @tparam G Callable as g(t, rate), t a simulation time in yr
         *   and rate a std::vector<double>& holding the instantaneous
         *   rate of release at t, modifying rate in place before it is
         *   integrated over time -- e.g. radioactive decay forward to
         *   tNow; NoDeathRateHook (the default) for none
         * @param f The quantity released per star
         * @param n The number of values f returns
         * @param tLast Start of the time interval, in yr since this
         *   galaxy's own formation
         * @param tNow End of the time interval, in yr
         * @param g The per-time hook
         * @return The total of each quantity released by the continuous
         *   population's own stars that died during (tLast, tNow], for
         *   this galaxy's own star formation history; all zero if
         *   tNow <= tLast, or if there is no continuous population
         *   (SimControls::minStochMass() == 0 or fCluster() == 1)
         * @details
         * Integrates, from the outermost integral in:
         * - over simulation time t in (tLast, tNow], of the rate at
         *   which the quantity is released at t (deathRate()), after
         *   g(t, rate) -- weighted by a flat distribution over the
         *   step, so that utils::integrateScaled() applies unchanged --
         *   then scales by the step's own length and by
         *   1 - SimControls::fCluster(), the continuous population's
         *   own share of the star formation rate;
         * - over stellar age, weighted by the star formation rate at
         *   t minus that age (deathRate());
         * - over [Fe/H], weighted by SimControls::fehDist(), unless
         *   that is degenerate (fehAveragedDeathIntegrand());
         * - innermost, the rate at which stars of that age and [Fe/H]
         *   die, each releasing f(m, feh) (deathIntegrand()).
         *
         * Integrating time outermost and [Fe/H] innermost means g is
         * applied once per time point, after the [Fe/H] integral
         * (the time and [Fe/H] integrals commute), rather than at every
         * [Fe/H] point -- the expensive step for radioactive decay.
         * Every integral makes its integrand dimensionless first -- see
         * utils::integrateScaled()'s own comment.
         */
        template <class F, class G = NoDeathRateHook>
        [[nodiscard]] auto continuousDeathQuantity(const F& f, const std::size_t n, const double tLast,
            const double tNow, const G& g = G{}) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            std::vector<double> result(n, 0.0);
            if (tNow <= tLast || sc.minStochMass() <= 0.0 || sc.fCluster() >= 1.0) { return result; }

            const double dt = tNow - tLast;
            const pdfs::PDF step(std::make_unique<pdfs::PDFSegmentPowerlaw>(tLast, tNow, 0.0));
            const auto rateAt = [&](const double t) -> std::vector<double>
            { return deathRate(t, f, n, g, std::nullopt); };
            result = utils::integrateScaled(step, rateAt, n, tLast, tNow,
                { tLast + (0.25 * dt), tLast + (0.5 * dt), tLast + (0.75 * dt), tNow },
                sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());
            const double factor = (1.0 - sc.fCluster()) * dt / step.integral(tLast, tNow);
            for (double& r : result) { r *= factor; }
            return result;
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
         * of t from fieldStars() to deadFieldStars(), updates
         * curTime() to t and caches every surviving field star's own
         * properties in fieldStarProps_ (see its own comment), then marks
         * spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_/phot_/
         * photExtinct_/photNeb_/photNebExtinct_/lbol_/lbolCts_ as stale
         * (see specCurrent_/photCurrent_/lbolCurrent_/
         * lbolCtsCurrent_'s own comments) rather than recomputing them
         * itself -- they are instead recomputed lazily, on demand, the
         * next time spec()/specExtinct()/specNeb()/specNebExtinct()/
         * lineLum()/lineLumExtinct()/phot()/photExtinct()/photNeb()/
         * photNebExtinct()/lbol() is actually called. Finally updates
         * the yields and feedback quantities eagerly, via
         * computeYields() and computeFeedback().
         */
        void advance(double t);

    private:

        /**
         * @brief The inner (mass) layer of integrateCts(): a per-star quantity integrated over the continuous population of a single age and [Fe/H]
         * @tparam F A callable f(m, segment, feh), with m a stellar mass
         *   in Msun, segment a specsyn::Specsyn::Segment (one segment
         *   of an isochrone) whose domain contains m, and feh the
         *   [Fe/H], returning a container (e.g. std::array or
         *   std::vector) of nInt doubles: the per-star quantities to
         *   integrate
         * @param age Stellar age, in yr -- see integrateCts()'s own
         *   comment for why this is age, not time
         * @param feh The single [Fe/H] value this call is evaluated at
         * @param f The per-star quantity, as described above
         * @param nInt The number of values f returns
         * @return The integral, over masses in
         *   [imf().getMin(), minStochMass()] alive at this age, of
         *   imf()(m) f(m, segment, feh) -- per star formed, since imf()
         *   is normalized by number
         * @details
         * Floors log10(age) at the tracks' own logTMin(), then builds
         * the isochrone at that (log age, feh) -- from the shared
         * fixed-[Fe/H] tracks2D() slice when SimControls::constFeH()
         * is true, since this is called many times per integral, and
         * from tracks() at feh otherwise -- and integrates f against
         * imf() separately over each of its segments' own intersection
         * with [imf().getMin(), minStochMass()] (the purely continuous
         * share of the population -- see Specsyn::specCts()'s own
         * comment), summing the results. Per-segment integration
         * mirrors Specsyn::continuousSpecIntegrand()'s: an isochrone
         * may have gaps between segments that the quadrature has no
         * way to know to avoid, and a dead mass is simply never
         * visited, since it falls in no segment's own domain.
         *
         * Each segment's integral uses utils::integrateScaled(), with
         * f's scale taken at the segment's own two ends and its
         * geometric midpoint, so that SimControls::intAbsTol() is a
         * meaningful absolute tolerance whatever f's own units and
         * however many orders of magnitude its elements span -- see
         * integrateScaled()'s own comment.
         */
        template <class F>
        [[nodiscard]] auto integrateCtsIntegrand(const double age, const double feh, const F& f,
            const std::size_t nInt) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            const auto& imf = sc.imf();
            const double logAge = std::max(std::log10(age), sc.tracks()->logTMin());
            const auto isochrone = sc.constFeH() ?
                sc.tracks2D()->getIsochrone(logAge) : sc.tracks()->getIsochrone(logAge, feh);

            std::vector<double> result(nInt, 0.0);
            for (const auto& seg : isochrone)
            {
                const double a = std::max(imf.getMin(), seg->xMin());
                const double b = std::min(sc.minStochMass(), seg->xMax());
                if (a >= b) { continue; } // empty intersection with [imf.getMin(), minStochMass()]

                const auto fAt = [&f, &seg, feh](const double m) -> std::vector<double>
                {
                    const auto v = f(m, *seg, feh);
                    std::vector<double> out(std::begin(v), std::end(v));
                    return out;
                };
                const auto segResult = utils::integrateScaled(imf, fAt, nInt, a, b,
                    { a, std::sqrt(a * b), b }, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());
                for (std::size_t k = 0; k < nInt; ++k) { result.at(k) += segResult.at(k); }
            }
            return result;
        }

        /**
         * @brief Integrate a per-star quantity over the whole purely continuous (non-clustered, non-stochastic) population, at curTime_
         * @tparam F See integrateCtsIntegrand()'s own F
         * @param f The per-star quantity -- see integrateCtsIntegrand()
         * @param nInt The number of values f returns
         * @return The total of each of f's values over every star in
         *   the purely continuous population alive at curTime_: the
         *   integral, over stellar age in [0, curTime_] weighted by the
         *   star formation rate at curTime_ minus that age, of
         *   integrateCtsIntegrand() (averaged over
         *   SimControls::fehDist() if it is non-degenerate), scaled to
         *   this population's own share of the stellar mass formed --
         *   (1 - fCluster()) (1 - fracStochMass()) / nonStochIMFMass()
         *   -- or all zeros if nonStochIMFMass() is not positive or
         *   curTime_ is 0
         * @details
         * Used by computeLbolCts() (with f the per-star bolometric
         * luminosity). Mirrors Specsyn::specCtsHelper()'s own nested
         * structure: the age integral runs over a pdfs::PDFReflect view
         * of sfr() pivoted at curTime_ / 2, so its own coordinate is
         * age directly, from ageMin = min(1e4 yr, 1e-3 curTime_) to
         * curTime_, log-transformed (see specCtsHelper()'s own comment
         * for both choices); and, if fehDist() is non-degenerate, that
         * whole age integral is itself integrated over fehDist() and
         * divided by fehDist()'s own integral, mirroring
         * fehAveragedDeathIntegrand().
         *
         * Every layer uses utils::integrateScaled() rather than a bare
         * PDFIntegrator, so that SimControls::intAbsTol() is a
         * meaningful absolute tolerance whatever f's own units, and
         * even when f's elements span many orders of magnitude or are
         * zero (e.g. wind mass, momentum, and energy fluxes, with some
         * wind models off) -- see integrateScaled()'s own comment. The
         * age layer takes its scale at five log-spaced ages from
         * ageMin to curTime_, and the [Fe/H] layer at fehDist()'s own
         * expectation value.
         */
        template <class F>
        [[nodiscard]] auto integrateCts(const F& f, const std::size_t nInt) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            const double massInt = sc.nonStochIMFMass();
            if (!(curTime_ > 0.0) || !(massInt > 0.0)) { return std::vector<double>(nInt, 0.0); } // NOLINT(modernize-return-braced-init-list) -- braces would select vector's initializer_list constructor, giving {double(nInt), 0.0}

            // See Specsyn::specCtsHelper()'s own comment for why this
            // reflects sfr() about curTime_ / 2 (so this layer's own
            // coordinate is age directly), why ageMin is
            // min(1e4 yr, 1e-3 curTime_), and why it is log-transformed
            const pdfs::PDFReflect sfrAge(sfr(), 0.5 * curTime_);
            const double ageMin = std::min(1e4, 1e-3 * curTime_);
            constexpr int nAgeScale = 5;
            std::vector<double> ageScalePoints;
            ageScalePoints.reserve(nAgeScale);
            for (int i = 0; i < nAgeScale; ++i)
            {
                ageScalePoints.push_back(ageMin * std::pow(curTime_ / ageMin,
                    static_cast<double>(i) / (nAgeScale - 1)));
            }

            const auto ageIntegral = [&](const double feh) -> std::vector<double>
            {
                const auto atAge = [&, feh](const double age) -> std::vector<double>
                { return integrateCtsIntegrand(age, feh, f, nInt); };
                return utils::integrateScaled(sfrAge, atAge, nInt, ageMin, curTime_, ageScalePoints,
                    sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol(), true);
            };

            const auto& fehDist = sc.fehDist();
            std::vector<double> result;
            if (fehDist.getMin() == fehDist.getMax())
            {
                result = ageIntegral(fehDist.getMin());
            }
            else
            {
                result = utils::integrateScaled(fehDist, ageIntegral, nInt, fehDist.getMin(),
                    fehDist.getMax(), { fehDist.expectationValue() }, sc.intMaxIter(),
                    sc.intAbsTol(), sc.intRelTol());
                const double fehNorm = fehDist.integral(fehDist.getMin(), fehDist.getMax());
                for (double& r : result) { r /= fehNorm; }
            }

            // imf() is normalized by number, so result is per star
            // formed; convert to this population's own total -- see
            // SimControls::nonStochIMFMass()'s own comment
            const double scale = (1.0 - sc.fCluster()) * (1.0 - sc.fracStochMass()) / massInt;
            for (double& r : result) { r *= scale; }
            return result;
        }

        /**
         * @brief The rate, per unit stellar mass formed, at which stars of a given age and [Fe/H] release a quantity at death
         * @tparam F See continuousDeathQuantity()'s own F
         * @param age Stellar age, in yr
         * @param feh [Fe/H] of the stars
         * @param f See continuousDeathQuantity()'s own f
         * @param n The number of values f returns
         * @return The sum, over every mass m below
         *   SimControls::minStochMass() dying at this age (from the
         *   tracks' own massAndDerivFromLifetime()), of
         *   |dm/dt| * imf(m) / <m> * f(m, feh), <m> being imf()'s own
         *   mean mass: the rate, per yr and per unit stellar mass
         *   formed, at which the continuous population releases each
         *   quantity
         * @details
         * The innermost layer of continuousDeathQuantity(). Uses the
         * shared fixed-[Fe/H] tracks2D() slice when
         * SimControls::constFeH() is true, and tracks() at feh
         * otherwise -- see integrateCtsIntegrand()'s own identical choice.
         */
        template <class F>
        [[nodiscard]] auto deathIntegrand(const double age, const double feh, const F& f,
            const std::size_t n) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            std::vector<double> result(n, 0.0);
            const auto massDeriv = sc.constFeH() ?
                sc.tracks2D()->massAndDerivFromLifetime(std::log10(age)) :
                sc.tracks()->massAndDerivFromLifetime(std::log10(age), feh);
            for (const auto& [m, dmDlogT] : massDeriv)
            {
                if (m > sc.minStochMass()) { continue; } // stochastically sampled, handled separately
                const double dmDt = dmDlogT / (age * std::numbers::ln10); // chain rule
                const double weight = std::abs(dmDt) * sc.imf()(m) / sc.imf().expectationValue();
                const std::vector<double> y = f(m, feh);
                for (std::size_t k = 0; k < result.size(); ++k)
                {
                    result[k] += y[k] * weight; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f returns n values by its own contract, and result has size n
                }
            }
            return result;
        }

        /**
         * @brief deathIntegrand(), averaged over SimControls::fehDist()
         * @tparam F See continuousDeathQuantity()'s own F
         * @param age Stellar age, in yr
         * @param f See continuousDeathQuantity()'s own f
         * @param n The number of values f returns
         * @return deathIntegrand(age, feh, f, n) at fehDist()'s own
         *   single value if it is degenerate; otherwise its integral
         *   over fehDist() (via utils::integrateScaled(), scaled at
         *   fehDist()'s own mean), divided by fehDist()'s own integral
         * @details
         * The [Fe/H] layer of continuousDeathQuantity() -- innermost of
         * the integrals, so that its hook is applied only after this
         * integral has converged; see continuousDeathQuantity()'s own
         * comment.
         */
        template <class F>
        [[nodiscard]] auto fehAveragedDeathIntegrand(const double age, const F& f,
            const std::size_t n) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            const auto& fehDist = sc.fehDist();
            if (fehDist.getMin() == fehDist.getMax()) { return deathIntegrand(age, fehDist.getMin(), f, n); }
            const auto integrandAt = [&](const double feh) -> std::vector<double>
            { return deathIntegrand(age, feh, f, n); };
            auto result = utils::integrateScaled(fehDist, integrandAt, n, fehDist.getMin(),
                fehDist.getMax(), { fehDist.expectationValue() }, sc.intMaxIter(), sc.intAbsTol(),
                sc.intRelTol());
            const double fehNorm = fehDist.integral(fehDist.getMin(), fehDist.getMax());
            for (double& r : result) { r /= fehNorm; }
            return result;
        }

        /**
         * @brief The instantaneous rate at which the continuous population releases a quantity at stellar death, at a given time
         * @tparam F See continuousDeathQuantity()'s own F
         * @tparam G See continuousDeathQuantity()'s own G
         * @param t Simulation time, in yr
         * @param f See continuousDeathQuantity()'s own f
         * @param n The number of values f returns
         * @param g See continuousDeathQuantity()'s own g -- applied to
         *   the result before it is returned
         * @param fehFixed If set, evaluate at this single [Fe/H]
         *   (deathIntegrand()) rather than averaging over
         *   SimControls::fehDist() (fehAveragedDeathIntegrand())
         * @return The integral, over stellar age in [0, t], of the
         *   [Fe/H]-averaged (or fixed-[Fe/H]) deathIntegrand() weighted
         *   by the star formation rate at t minus that age -- the rate,
         *   per yr, at which the whole population (clustered and not;
         *   continuousDeathQuantity() applies the 1 - fCluster() share)
         *   releases each quantity at t -- after g(t, rate)
         * @details
         * The age layer of continuousDeathQuantity(). Integrates over
         * age via a pdfs::PDFReflect view of sfr() pivoted at t / 2 --
         * see integrateCts()'s own comment -- with
         * utils::integrateScaled(), scaled at a few log-spaced ages
         * (deaths only begin a few Myr in).
         */
        template <class F, class G>
        [[nodiscard]] auto deathRate(const double t, const F& f, const std::size_t n, const G& g,
            const std::optional<double> fehFixed) const -> std::vector<double>
        {
            const auto& sc = controls_.get();
            const pdfs::PDFReflect sfrAge(sfr(), 0.5 * t);
            const auto integrandAt = [&](const double age) -> std::vector<double>
            {
                return fehFixed.has_value() ? deathIntegrand(age, *fehFixed, f, n) :
                    fehAveragedDeathIntegrand(age, f, n);
            };
            auto result = utils::integrateScaled(sfrAge, integrandAt, n, 0.0, t,
                { t, 0.3 * t, 0.1 * t, 0.03 * t, 0.01 * t }, sc.intMaxIter(), sc.intAbsTol(),
                sc.intRelTol());
            g(t, result);
            return result;
        }

        double curTime_ = 0.0;                  /**< Current simulation time */
        std::vector<Cluster> clusters_;          /**< Currently alive (non-disrupted) clusters */
        std::vector<Cluster> disruptedClusters_; /**< Disrupted clusters */
        std::vector<FieldStar> fieldStars_;      /**< Currently alive field stars, sorted by formTime_ -- see advance()'s own comment */

        /**
         * @brief Cached stellar properties of every currently-alive field star, at curTime_
         * @details
         * One entry per element of fieldStars_ (same order), as
         * returned by getFieldStarProps() -- see its own comment,
         * including for why an entry may be empty. Recomputed eagerly
         * by advance(), right after it updates curTime_, and then
         * read by addContinuousSpec(), computeLbol(), and
         * computeFeedback(), so the track lookups are paid for once
         * per advance() rather than once per use. Consequently, if
         * SimControls::setTracks() replaces the tracks between one
         * advance() and the next, these properties still reflect the
         * tracks in place at that advance().
         */
        std::vector<std::optional<specsyn::Specsyn::StarData>> fieldStarProps_;
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
        std::vector<double> lineLumExtinct_; /**< Sum of lineLumExtinct() over every cluster in clusters_/disruptedClusters_, plus the continuous population's own extincted contribution, at the current time */
        std::vector<double> phot_;         /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_;  /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNeb_;      /**< Photometry of specNeb_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photNebExtinct_; /**< Photometry of specNebExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;                /**< Sum of lbol() over every cluster in clusters_/disruptedClusters_, at the current time */
        double targetMass_ = 0.0;          /**< Cumulative total target stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */
        double actualMass_ = 0.0;          /**< Cumulative total actual stellar mass formed so far (clustered and continuous together), over every advance() call, in Msun */

        /**
         * @brief Whether spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ are current as of curTime_
         * @details
         * Mirrors Cluster::specCurrent_'s own comment: true at
         * construction, set false at the end of every advance() call,
         * and back to true by spec()/specExtinct()/specNeb()/
         * specNebExtinct()/lineLum()/lineLumExtinct() after recomputing
         * them.
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

        std::vector<double> yields_; /**< Total nucleosynthetic yield of each isotope, in Msun, accumulated over every star that has died so far across the whole galaxy (clusters, field stars, and the purely continuous population together), laid out per Cluster::yields()'s own comment -- sized to all zeros at construction if controls().yields() is non-null, empty otherwise; see Cluster::yields_'s own comment for how it accumulates */
        std::vector<double> fieldYields_; /**< Like yields_, but restricted to the non-clustered population -- both the individually-tracked field stars (fieldStars_/deadFieldStars_) and the purely continuous (below minStochMass()) population together, see computeYields()'s own comment for how each is added in -- excluding only clusters_/disruptedClusters_; initialized the same way as yields_. Tracked separately so computeYields() can combine per-population contributions without double-counting */

        /**
         * @brief Simulation time through which yields_/fieldYields_ have been updated
         * @details
         * Mirrors Cluster::lastYieldTime_'s own comment exactly: starts
         * at 0 (rather than curTime_'s own initial value, though here
         * the two happen to coincide already), documenting that
         * nothing has died yet. advance() itself sets this to curTime_
         * right after every call to computeYields() -- see both of
         * their own comments for why this can't be left to a lazy
         * yields() call the way spec_/phot_/lbol_'s own analogous
         * staleness flags are: computeYields() only adds the
         * contribution of whatever died between lastYieldTime_ and
         * curTime_, and for the field-star share of that relies on
         * deadFieldStars_, which only ever holds the deaths from the
         * single most recently advance() call (see its own comment),
         * so computeYields() must run before deadFieldStars_'s own
         * contents are overwritten by the next advance() call, not
         * merely before yields_/fieldYields_ are next read.
         */
        double lastYieldTime_ = 0.0;

        unsigned long fieldStarStochSNe_ = 0;  /**< Cumulative number of supernovae from the individually-sampled field stars; see computeFeedback() */
        double fieldStarNonStochSNe_ = 0.0;    /**< Cumulative (in general non-integer) number of supernovae from the continuously-sampled field population; see computeFeedback() */
        double mDotWind_ = 0.0;                /**< Total stellar wind mass flux at the current time, in Msun/yr; see mDotWind() */
        double pDotWind_ = 0.0;                /**< Total stellar wind momentum flux at the current time, in g cm s^-2; see pDotWind() */
        double eDotWind_ = 0.0;                /**< Total stellar wind energy flux at the current time, in erg s^-1; see eDotWind() */

        /**
         * @brief Simulation time through which fieldStarNonStochSNe_ has been updated
         * @details
         * Mirrors lastYieldTime_'s own role (see its own comment) for
         * computeFeedback(): starts at 0, and advance() sets it to
         * curTime_ right after every call to computeFeedback(), which
         * adds the continuous population's supernovae between it and
         * curTime_.
         */
        double lastFeedbackTime_ = 0.0;

        /**
         * @brief Simulation controls (physics and control-flow settings) this galaxy was built from
         * @details
         * See Cluster::controls_'s own comment: read live wherever a
         * physics setting or integrator tolerance is needed, rather
         * than snapshotted at construction.
         */
        std::reference_wrapper<const io::SimControls> controls_;

        /**
         * @brief Update spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ from the current cluster population
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
         * specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ the same way.
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
         * @brief Set specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ from the current cluster population
         * @param ext SimControls::extinct(), passed through from
         *   computeSpec() rather than re-read here, since computeSpec()
         *   already has it in hand
         * @param neb SimControls::nebular(), passed through from
         *   computeSpec() likewise; must be non-null (computeSpec()
         *   only calls this when it is)
         * @details
         * Mirrors computeSpec()'s own spec_/specExtinct_ summing
         * exactly, but for specNeb_/lineLum_ (unconditionally) and
         * specNebExtinct_/lineLumExtinct_ (if ext is non-null), from
         * specNeb()/lineLum()/specNebExtinct()/lineLumExtinct() over
         * every cluster in clusters_ and disruptedClusters_. Split out
         * of computeSpec() itself
         * purely to keep that function's own cognitive complexity
         * down, not for any reuse elsewhere.
         */
        void addClusterSpecNeb(const extinct::Extinct* ext, const nebular::Nebular* neb);

        /**
         * @brief Add the purely continuous population's and every field star's own contribution to spec_/specExtinct_/specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_
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
         * Then adds every entry of fieldStarProps_ (evaluated via
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
         * contribution (before extinction) through neb->getGalaxy(),
         * averaged over SimControls::fehDist() (see
         * addContinuousNebSpec()'s own comment), adding the resulting
         * stellar + nebular spectrum to specNeb_
         * and its own line luminosities to lineLum_; if ext is also
         * non-null, extinguishes that same nebular-reprocessed spectrum
         * via applyExtinctionCts(), exactly as the plain (non-nebular)
         * contribution is, and adds it to specNebExtinct_, and
         * likewise extinguishes those same line luminosities via
         * applyExtinctionCtsLines(), adding the result to
         * lineLumExtinct_.
         *
         * Split out of computeSpec() itself purely to keep that
         * function's own cognitive complexity down.
         */
        void addContinuousSpec(const extinct::Extinct* ext, const nebular::Nebular* neb);

        /**
         * @brief Add contSpec's own nebular-reprocessed contribution to specNeb_/lineLum_/specNebExtinct_/lineLumExtinct_
         * @param ext SimControls::extinct(), passed through from
         *   addContinuousSpec() rather than re-read here
         * @param neb SimControls::nebular(), passed through from
         *   addContinuousSpec() likewise; must be non-null
         *   (addContinuousSpec() only calls this when it is)
         * @param contSpec The purely continuous population's and every
         *   field star's own combined, unattenuated spectrum, as built
         *   by addContinuousSpec() itself
         * @details
         * See addContinuousSpec()'s own comment for exactly what this
         * does -- split out purely to keep that function's own
         * cognitive complexity down, not for any reuse elsewhere.
         *
         * neb->getGalaxy() takes a single [Fe/H]. For a fixed
         * fehDist(), it is called once, there; otherwise its spectrum
         * and line luminosities are integrated over fehDist() with a
         * PDFIntegrator, and divided by fehDist()'s own integral --
         * the same treatment of [Fe/H] as the continuous population's
         * own spectrum, Lbol and yields (see
         * Specsyn::specCtsHelper()'s own comment). contSpec itself is
         * already integrated over [Fe/H], so only the nebular tables'
         * own [Fe/H] dependence is integrated here. The spectrum and
         * line luminosities are each divided by the smallest nonzero
         * magnitude they have at fehDist()'s own mean before
         * integrating, so that both are dimensionless and on a common
         * scale for intAbsTol() -- see the implementation's own
         * comment for why a relative tolerance alone is not enough.
         */
        void addContinuousNebSpec(const extinct::Extinct* ext, const nebular::Nebular* neb,
            const std::vector<double>& contSpec);

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
         * contribution (10^logL, read directly off fieldStarProps_)
         * -- independent of lbolCtsCurrent_/lbolCts_, since this is
         * cheap and direct either way, unlike the continuous
         * population's own Specsyn-mediated integral.
         */
        void computeLbol();

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
         * Sets lbolCts_ to integrateCts() of the per-star bolometric
         * luminosity, 10^logL in Lsun, read directly off each
         * isochrone segment by a local, capture-free lambda mirroring
         * Cluster::lbolStar()'s own role -- rather than calling into
         * any Specsyn, which may not exist at all here.
         * integrateCts() already applies the
         * (1 - fCluster()) * (1 - fracStochMass()) / nonStochIMFMass()
         * scaling to the purely continuously-treated share of the
         * population, matching Specsyn::specAndLbolCts()'s own -- so
         * the two agree, to integrator tolerance, when both are
         * exercised for the same population (see
         * testContinuousPopLbolStandaloneMatchesSpec and its
         * multi-[Fe/H] counterpart in tests/core/testGalaxy.cpp). Sets
         * lbolCtsCurrent_ to true afterward.
         */
        void computeLbolCts();

        /**
         * @brief Evaluate every currently-alive field star's stellar properties
         * @return A vector, one element per entry in fieldStars() (same
         *   order), of that star's properties at curTime() -- see
         *   tracks::Tracks2D::getStar()/tracks::Tracks3D::getStar()'s
         *   own comment for what a StarData holds -- or empty for a
         *   star whose mass lies outside the tracks' own mass range, or
         *   (with a non-degenerate [Fe/H]) whose feh_ lies outside the
         *   tracks' own [Fe/H] grid, so has no properties to look up
         *   (such a star is treated as contributing no light, as in
         *   Cluster::computeSpec())
         * @details
         * If the simulation has a fixed [Fe/H] (SimControls::constFeH()),
         * loops over fieldStars() directly, calling
         * SimControls::tracks2D()'s own getStar(mass_, log10(curTime()
         * - formTime_)) once per star -- cheap, since tracks2D() is a
         * single, already-built Tracks2D slice shared across every
         * call.
         *
         * Otherwise, calls SimControls::tracks()'s own getStar(mass_,
         * logT, feh_) at each star's own (in general, distinct) feh_ --
         * which evaluates a lazily-evaluated slice of the tracks at
         * that feh_, touching only the few mesh points around the star
         * itself (see tracks::Tracks3D's own class comment), so this is
         * cheap too, and safe to call from multiple threads.
         *
         * Either way, logT (log10 of the star's own age,
         * curTime() - formTime_) is floored at
         * SimControls::tracks()'s own logTMin(), mirroring
         * Cluster::advance()'s and Specsyn::continuousSpecIntegrand()'s
         * own identical floor, to avoid taking log10(0) for a field
         * star whose formTime_ is exactly curTime_ (formed during the
         * very advance() call that produced this evaluation).
         *
         * Called once per advance(), to fill fieldStarProps_; every
         * other use reads that cache instead.
         */
        [[nodiscard]] auto getFieldStarProps() const -> std::vector<std::optional<specsyn::Specsyn::StarData>>;

        /**
         * @brief Update yields_/fieldYields_ from the stars that died since lastYieldTime_
         * @details
         * Called eagerly from advance() itself, at the end of every
         * call -- not lazily from yields() the way spec_/phot_/lbol_
         * are computed -- see lastYieldTime_'s own comment for why.
         * A no-op if controls().yields() is null. Otherwise, mirrors
         * Cluster::computeYields()'s own null-guard pattern one level
         * up, summing/integrating over three populations in turn:
         *
         * - Clustered stars: yields_ is rebuilt from scratch (like
         *   lbol_, unlike fieldYields_ below) by summing
         *   Cluster::yields()'s own cumulative total over every
         *   cluster in clusters_ and disruptedClusters_, since each
         *   cluster keeps growing that total on its own rather than
         *   reporting only what died this step.
         * - Before either of the two field-star shares below is added
         *   in, if controls().noDecay() is false, fieldYields_'s own
         *   already-accumulated total (from every earlier call) is
         *   aged forward via Yields::applyDecay(), with dtDecay =
         *   curTime_ - lastYieldTime_ -- mirrors Cluster::yields_'s own
         *   identical aging step, see its own comment for why this is
         *   exact, not an approximation, by the decay operator's own
         *   compositional (semigroup) property.
         * - Individually-tracked field stars: every star in
         *   deadFieldStars_ (which, like Cluster::mDead_, only ever
         *   holds the stars that died during the most recent
         *   advance() call) has controls().yields()'s own yield() (if
         *   controls().yieldsChannelDecomposed()) or yieldSum()
         *   (otherwise) evaluated at its own mass_/feh_, with dtDecay =
         *   curTime_ - deathTime_ (or 0 if controls().noDecay() is
         *   true), added onto fieldYields_.
         * - The purely continuous (non-clustered, below
         *   minStochMass()) population: continuousDeathQuantity(),
         *   with each dying star's own yields as its quantity and
         *   radioactive decay forward to curTime_ as its per-time hook,
         *   over (lastYieldTime_, curTime_], added onto fieldYields_
         *   -- zero if there is no such population (minStochMass() ==
         *   0 or fCluster() == 1); see its own comment.
         *
         * fieldYields_ accumulates across calls (never zeroed) since,
         * unlike the clustered total, earlier steps' own dead field
         * stars are not available to re-sum from scratch; yields_'s
         * own final value is the clustered total plus fieldYields_.
         */
        void computeYields();

        /**
         * @brief Add the feedback from every field star that died during the most recent advance() call into the cumulative feedback quantities
         * @details
         * Called eagerly from advance() itself, at the end of every
         * call, right after computeYields() and for the same reason
         * (see lastYieldTime_'s own comment): it relies on
         * deadFieldStars_, which the next advance() call overwrites.
         * Currently the only feedback quantity is the number of
         * supernovae. From the individually-sampled field stars: adds
         * 1 to fieldStarStochSNe_ for each star in deadFieldStars_ for
         * which controls().hasSN(mass_, feh_) is true. From the
         * continuously-sampled field population: adds
         * continuousDeathQuantity() of the indicator hasSN(m, feh)
         * over (lastFeedbackTime_, curTime_] to fieldStarNonStochSNe_;
         * advance() then sets lastFeedbackTime_ to curTime_. Each
         * cluster's own supernovae are counted by that cluster's own
         * Cluster::computeFeedback(), run from its own advance().
         *
         * Also recomputes the instantaneous stellar wind fluxes
         * mDotWind_, pDotWind_, and eDotWind_ from scratch: zeroes all
         * three, adds Cluster::mDotWind()/pDotWind()/eDotWind() for
         * every cluster in clusters_ and disruptedClusters_, then, for
         * every currently-alive field star with properties in
         * fieldStarProps_, adds mDot, mDot v_wind, and
         * (1/2) mDot v_wind^2 (mDot from the star's own properties,
         * v_wind from controls().winds()->vWind() at the star's own
         * feh_, or 0 if winds() is null; pDot and eDot in cgs, as in
         * Cluster::windStar()). The purely continuous (non-clustered,
         * non-stochastic) population is not yet included.
         */
        void computeFeedback();

        /**
         * @brief Bring the feedback quantities up to date, if they are not already
         * @details
         * Calls computeFeedback() and sets lastFeedbackTime_ to
         * curTime_ if lastFeedbackTime_ != curTime_; otherwise does
         * nothing. Mirrors Cluster::updateFeedback(), except that no
         * separate has-advanced check is needed: curTime_ and
         * lastFeedbackTime_ both start at 0, so they only differ
         * after some advance().
         */
        void updateFeedback()
        {
            if (lastFeedbackTime_ != curTime_)
            {
                computeFeedback();
                lastFeedbackTime_ = curTime_;
            }
        }

    };

} // namespace core

#endif // GALAXY_HPP
