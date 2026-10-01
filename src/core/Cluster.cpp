/**
 * @file Cluster.cpp
 * @author Mark Krumholz
 * @brief Implementation of Cluster
 * @date 2026-07-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Cluster.hpp"
#include "../extinct/Extinct.hpp"
#include "../feedback/Winds.hpp"
#include "../io/SimControls.hpp"
#include "../nebular/Nebular.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/Constants.hpp"
#include "../utils/GKIntegratorData.hpp"
#include "../utils/PDFIntegrator.hpp"
#include "../utils/RngThread.hpp"
#include "../yields/Yields.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace
{
    /**
     * @brief Subtract one set of mass ranges from another
     * @param minuend Mass ranges to subtract from, e.g. as returned by
     *   Tracks2D::liveMassRange() -- sorted ascending by lower bound,
     *   and internally non-overlapping
     * @param subtrahend Mass ranges to subtract, in the same format
     * @return The pieces of minuend left over once every overlap with
     *   subtrahend is removed -- still sorted ascending and
     *   non-overlapping, though possibly with more entries than
     *   minuend itself (a single minuend range can be split into
     *   several pieces by one or more subtrahend ranges falling
     *   inside it)
     * @details
     * Used by Cluster::nonStochDeadMassRanges() to find which masses were alive
     * at one time but are dead at another -- see its own comment.
     */
    auto subtractMassRanges(
        const std::vector<std::pair<double, double>>& minuend,
        const std::vector<std::pair<double, double>>& subtrahend)
        -> std::vector<std::pair<double, double>>
    {
        std::vector<std::pair<double, double>> result;
        for (const auto& [aLo, aHi] : minuend)
        {
            double lo = aLo;
            for (const auto& [bLo, bHi] : subtrahend)
            {
                if (lo >= aHi) { break; } // nothing left of this range to subtract from
                if (bHi <= lo || bLo >= aHi) { continue; } // no overlap with what's left
                if (bLo > lo) { result.emplace_back(lo, bLo); }
                lo = std::max(lo, bHi);
            }
            if (lo < aHi) { result.emplace_back(lo, aHi); }
        }
        return result;
    }

    /**
     * @brief Jointly sort a cluster's stellar masses and death times by death time, descending
     * @param m Stellar masses (Cluster::m_) -- reordered in place
     * @param tDeath Death time of each entry of m, in the same order
     *   (Cluster::tDeath_) -- reordered in place, alongside m
     * @details
     * Used by both Cluster constructors right after tDeath is first
     * computed, so that m_/tDeath_ end up with the stars that will die
     * soonest (smallest tDeath_) at the back of both lists -- see
     * tDeath_'s own comment for why.
     */
    void sortByDeathTimeDescending(std::vector<double>& m, std::vector<double>& tDeath)
    {
        std::vector<std::size_t> order(m.size());
        std::iota(order.begin(), order.end(), 0UL);
        std::ranges::sort(order, std::ranges::greater{},
            [&tDeath](const std::size_t i) { return tDeath[i]; }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < tDeath.size() == m.size() by construction, for every entry of order

        std::vector<double> mSorted(m.size());
        std::vector<double> tDeathSorted(tDeath.size());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            mSorted[i] = m[order[i]]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- order[i] < m.size() by construction, i < order.size() == m.size()
            tDeathSorted[i] = tDeath[order[i]]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        }
        m = std::move(mSorted);
        tDeath = std::move(tDeathSorted);
    }

    /**
     * @brief A star's lifetime, treating masses below the tracks' own mass grid specially
     * @param tracks2D The (already feH-sliced) Tracks2D to query --
     *   this cluster's own tracks(), not controls().tracks() directly
     *   -- see @details for why
     * @param m Stellar mass, in Msun
     * @return tracks2D.starLifetime(m) if tracks2D.mMin() <= m <=
     *   tracks2D.mMax(); +infinity if m is below tracks2D.mMin() (no
     *   tabulated lifetime for so low a mass -- treated as living
     *   forever, matching computeSpec()'s own treatment of such stars
     *   as contributing zero luminosity rather than ever dying)
     * @details
     * Tracks2D::starLifetime() itself asserts its mass argument lies
     * within the tracks' own tabulated mass grid, so calling it
     * directly on every entry of m_ would crash whenever the IMF's own
     * minimum mass is below the tracks' own (which SimControls warns
     * about). m cannot exceed tracks2D.mMax(): SimControls rejects an
     * IMF extending above the tracks' maximum mass, which is the same
     * at every [Fe/H]. Used by both Cluster constructors (to build
     * tDeath_) and yieldStar().
     *
     * Takes this cluster's own Tracks2D, already sliced at its own
     * [Fe/H], rather than controls().tracks() (a Tracks3D), which
     * would need to slice the tracks afresh at feH_ on every call.
     */
    auto starLifetimeClamped(const tracks::Tracks2D& tracks2D, const double m) -> double
    {
        if (m < tracks2D.mMin()) { return std::numeric_limits<double>::infinity(); }
        return tracks2D.starLifetime(m);
    }
} // namespace

// Constructor
core::Cluster::Cluster(const unsigned long uid,
    const double mass,
    const double time,
    const io::SimControls& controls) :
    rngState_(utils::rng().getState()),
    uid_(uid),
    targetMass_(mass),
    formTime_(time),
    feH_(controls.fehDist().draw()),
    aV_(controls.avDist().valid() ? controls.avDist().draw() : 0.0),
    controls_(std::cref(controls)),
    m_(controls.imf().drawTarget(
        controls.fracStochMass() * mass,
        controls.minStochMass(),
        controls.imf().getMax())),
    birthNonStochMass_((1.0 - controls.fracStochMass()) * mass),
    birthMass_(std::accumulate(m_.begin(), m_.end(), 0.0)),
    disruptTime_(std::numeric_limits<double>::quiet_NaN()),
    curTime_(time)
{
    // Sort the generated population by mass
    std::ranges::sort(m_);

    // Set disruption time if disruption is on
    const auto& sc = controls_.get();
    if (sc.clf().valid())
    {
        disruptTime_ = formTime_ + sc.clf().draw();
    }

    // If this simulation has a variable metallicity, use a
    // lazily-evaluated slice of the tracks at this cluster's own
    // metallicity -- cheap to construct and to hold, unlike a full
    // slice (see Tracks3D::lazySliceConstFeH()). Otherwise, [Fe/H] is
    // fixed for the whole simulation, so just point at the slice
    // SimControls has already precomputed and shares across every
    // cluster.
    if (sc.constFeH())
    {
        tracks_ = sc.tracks2D();
    }
    else
    {
        tracks_ = sc.tracks()->lazySliceConstFeH(feH_);
    }

    // Death time of every star in m_, then jointly sort m_/tDeath_ by
    // tDeath_, descending, so updateLivingStars() can find newly-dead
    // stars with a simple backward scan -- see tDeath_'s own comment.
    // tDied_ starts out empty: nothing has died yet.
    tDeath_.reserve(m_.size());
    for (const double m : m_)
    {
        tDeath_.push_back(formTime_ + starLifetimeClamped(tracks(), m));
    }
    sortByDeathTimeDescending(m_, tDeath_);

    // If yield channels were requested, size yields_ to hold one
    // (currently zero) entry per isotope, times one row per channel
    // if SimControls::yieldsChannelDecomposed() is true (matching
    // Yields::yield()'s own (nchannels, nisotopes) layout), or just
    // one combined total per isotope otherwise (matching
    // Yields::yieldSum()) -- see yields_'s own comment
    if (const auto yields = sc.yields())
    {
        const std::size_t n = yields->isotopes().size() *
            (sc.yieldsChannelDecomposed() ? yields->yieldChannels().size() : 1);
        yields_.assign(n, 0.0);
    }
}

// Reconstruction constructor: replays this cluster's original draw
// sequence (feH_, then m_) from a previously recorded rng state,
// rather than drawing feH_ live and m_ from that state independently,
// so the result is bitwise identical to the original cluster's -- see
// this constructor's own header comment for why both draws, not just
// the mass one, need to be replayed: PDF::draw() consumes rng state
// (via a std::discrete_distribution used to pick a segment) even for
// a single-segment/delta [Fe/H] distribution, so feH_'s own draw
// already advances the stream before m_'s draw would otherwise see it.
core::Cluster::Cluster(const unsigned long uid,
    const double mass,
    const double time,
    const io::SimControls& controls,
    const utils::RngState& rngState) :
    rngState_(rngState),
    uid_(uid),
    targetMass_(mass),
    formTime_(time),
    feH_(0.0), // placeholder; set in the constructor body below
    aV_(0.0), // placeholder; set in the constructor body below
    controls_(std::cref(controls)),
    birthNonStochMass_((1.0 - controls.fracStochMass()) * mass),
    birthMass_(0.0), // placeholder; overwritten once m_ is drawn
    disruptTime_(std::numeric_limits<double>::quiet_NaN()),
    curTime_(time)
{
    // Draw feH_, aV_ (if valid), and m_ from the given rng state
    // rather than the live one: save the live state, temporarily
    // switch to rngState, draw each in the same order the primary
    // constructor does, then restore, so this constructor has no
    // lasting effect on the ambient rng stream
    const auto savedState = utils::rng().getState();
    utils::rng().setState(rngState);
    feH_ = controls.fehDist().draw(); //NOLINT(cppcoreguidelines-prefer-member-initializer) -- must happen after setState(rngState) above, not in the initializer list
    aV_ = controls.avDist().valid() ? controls.avDist().draw() : 0.0; //NOLINT(cppcoreguidelines-prefer-member-initializer) -- must happen after setState(rngState) above, not in the initializer list
    m_ = controls.imf().drawTarget(
        controls.fracStochMass() * mass,
        controls.minStochMass(),
        controls.imf().getMax());
    utils::rng().setState(savedState);

    // birthMass_ is summed before sorting m_, matching the primary
    // constructor's own order of operations (there, forced by summing
    // in the member initializer list, which runs before the
    // constructor body's own sort) -- floating-point addition is not
    // associative, so summing in a different order than the original
    // draw would reproduce the same set of masses but not necessarily
    // the same birthMass_ down to the last bit.
    birthMass_ = std::accumulate(m_.begin(), m_.end(), 0.0); //NOLINT(cppcoreguidelines-prefer-member-initializer) -- must happen after m_ is drawn above, not in the initializer list
    std::ranges::sort(m_);

    // Set disruption time if disruption is on
    const auto& sc = controls_.get();
    if (sc.clf().valid())
    {
        disruptTime_ = formTime_ + sc.clf().draw();
    }

    // If this simulation has a variable metallicity, use a
    // lazily-evaluated slice of the tracks at this cluster's own
    // metallicity -- cheap to construct and to hold, unlike a full
    // slice (see Tracks3D::lazySliceConstFeH()). Otherwise, [Fe/H] is
    // fixed for the whole simulation, so just point at the slice
    // SimControls has already precomputed and shares across every
    // cluster.
    if (sc.constFeH())
    {
        tracks_ = sc.tracks2D();
    }
    else
    {
        tracks_ = sc.tracks()->lazySliceConstFeH(feH_);
    }

    // Death time of every star in m_, then jointly sort m_/tDeath_ by
    // tDeath_, descending -- see the primary constructor's own
    // identical comment.
    tDeath_.reserve(m_.size());
    for (const double m : m_)
    {
        tDeath_.push_back(formTime_ + starLifetimeClamped(tracks(), m));
    }
    sortByDeathTimeDescending(m_, tDeath_);

    // If yield channels were requested, size yields_ to hold one
    // (currently zero) entry per isotope, times one row per channel
    // if SimControls::yieldsChannelDecomposed() is true (matching
    // Yields::yield()'s own (nchannels, nisotopes) layout), or just
    // one combined total per isotope otherwise (matching
    // Yields::yieldSum()) -- see yields_'s own comment
    if (const auto yields = sc.yields())
    {
        const std::size_t n = yields->isotopes().size() *
            (sc.yieldsChannelDecomposed() ? yields->yieldChannels().size() : 1);
        yields_.assign(n, 0.0);
    }
}

// Get the tracks at this cluster's [Fe/H]
auto core::Cluster::tracks() const -> const tracks::Tracks2D&
{
    if (const auto* owned = std::get_if<tracks::Tracks2D>(&tracks_))
    {
        return *owned;
    }
    return *std::get<std::shared_ptr<const tracks::Tracks2D>>(tracks_);
}

// Advance function
void core::Cluster::advance(const double t)
{
    // Make sure t >= curTime_
    if (t < curTime_)
    {
        std::stringstream ss;
        ss << "Cluster: requested advance to " << t
            << ", but curTime = " << curTime_;
        throw std::runtime_error(ss.str());
    }

    // If t == curTime_ and this isn't the very first call, do
    // nothing -- but the very first call must still run, even if
    // t == curTime_ (== formTime_, the common case of an output time
    // at t = 0), since isochrone_/spec_ have not yet been computed at
    // all before that
    if (t == curTime_ && advanced_) { return; }

    // Update time and cluster age. logAge is floored at the tracks'
    // own minimum representable log-age, since age = curTime_ -
    // formTime_ can be exactly zero (at the first call, when
    // t == formTime_), and log10(0) is -inf, which lies outside any
    // finite tracks grid; ages at or below the tracks' youngest grid
    // point are all treated as that youngest age.
    curTime_ = t;
    const auto logAge = std::max(std::log10(curTime_ - formTime_), tracks().logTMin());

    // Update list of alive and dead stars to new cluster age
    updateLivingStars();

    // Get isochrone for new time
    isochrone_ = tracks().getIsochrone(logAge);

    // Update yields_ now, eagerly -- unlike spec_/phot_/lbol_ (which
    // are recomputed lazily, from scratch, on demand -- see
    // specCurrent_/photCurrent_/lbolCurrent_'s own comments),
    // yields_ only ever accumulates the contribution of stars that
    // died since lastYieldTime_, and the stochastic half of that
    // relies on mDead_, which only ever holds the deaths from this one
    // advance() call (see updateLivingStars()'s own comment): the very
    // next advance() call's own updateLivingStars() clears it, so
    // computeYields() must consume it here, before that happens,
    // rather than waiting for some later, lazy call to yields() that
    // might never come before the next advance() -- see
    // lastYieldTime_'s own comment for the data loss that used to
    // result otherwise.
    computeYields();
    lastYieldTime_ = curTime_;

    // Update the cumulative feedback quantities eagerly too, for the
    // same reason -- see lastFeedbackTime_'s own comment
    computeFeedback();
    lastFeedbackTime_ = curTime_;

    // Mark spec_/specExtinct_/phot_/photExtinct_/lbol_ as stale; they
    // are recomputed lazily, on demand, the next time spec()/
    // specExtinct()/phot()/photExtinct()/lbol() is actually called
    // (see specCurrent_/photCurrent_/lbolCurrent_'s own comments).
    specCurrent_ = false;
    photCurrent_ = false;
    lbolCurrent_ = false;

    // Check for disruption
    if (curTime_ > disruptTime_) { isDisrupted_ = true; }

    advanced_ = true;
}

// Update lists of alive and dead stars to current age -- see this
// method's own header comment for why a simple backward scan over
// tDeath_ suffices, now that m_/tDeath_ are kept jointly sorted by
// tDeath_, descending (see tDeath_'s own comment)
void core::Cluster::updateLivingStars()
{
    // Clear the lists of dead stars/death times
    mDead_.clear();
    tDied_.clear();

    // Pop stars off the back of m_/tDeath_ (the ones that will die
    // soonest) for as long as they have already died as of curTime_;
    // every remaining entry, once this stops, is guaranteed to have a
    // tDeath_ still >= curTime_, since the list is sorted
    while (!tDeath_.empty() && tDeath_.back() < curTime_)
    {
        mDead_.push_back(m_.back());
        tDied_.push_back(tDeath_.back());
        m_.pop_back();
        tDeath_.pop_back();
    }
}

// Compute the population spectrum (and, if requested, nebular
// emission and/or extinction) at the current isochrone, if a spectral
// synthesizer was requested
void core::Cluster::computeSpec()
{
    const auto& sc = controls_.get();
    const auto synth = sc.specsyn();
    if (synth == nullptr) { return; }

    spec_.assign(synth->wl().size(), 0.0);

    // Continuously-sampled (non-stochastic) part of the population
    if (birthNonStochMass_ > 0.0)
    {
        spec_ = synth->specCts(isochrone_, sc.imf(),
            birthNonStochMass_, sc.imf().getMin(), sc.minStochMass(), feH_);
    }

    // Individually-sampled (stochastic) stars
    for (const double m : m_)
    {
        // Stars below the tracks' minimum mass have no isochrone
        // segment to evaluate spec() on (this can happen when the
        // IMF extends below the tracks' mass range); skip them,
        // treating their contribution as negligible
        const auto seg = std::ranges::find_if(isochrone_,
            [m](const auto& segment) -> bool
            { return m >= segment->xMin() && m <= segment->xMax(); });
        if (seg == isochrone_.end()) { continue; }

        const auto starSpec = synth->spec(m, **seg, feH_);
        for (std::size_t i = 0; i < spec_.size(); ++i)
        {
            spec_[i] += starSpec[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- spec_ and starSpec both have size wl().size() by construction
        }
    }

    // Add nebular continuum and line emission, if a nebular emission
    // grid was requested
    const auto neb = sc.nebular();
    if (neb != nullptr)
    {
        auto [nebSpec, nebLineLum] = neb->getCluster(spec_, feH_, curTime_ - formTime_);
        specNeb_ = std::move(nebSpec);
        lineLum_ = std::move(nebLineLum);
    }

    // Extinguish the spectrum(s) just computed, if an extinction curve
    // was requested
    const auto ext = sc.extinct();
    if (ext != nullptr)
    {
        specExtinct_ = ext->applyExtinction(aV_, spec_);
        if (neb != nullptr)
        {
            specNebExtinct_ = ext->applyExtinction(aV_, specNeb_);
            lineLumExtinct_ = ext->applyExtinctionLines(aV_, lineLum_);
        }
    }
}

// Update the population photometry (and, if an extinction curve
// and/or nebular emission grid was requested, extincted and/or
// nebular photometry) from the current spec_/specExtinct_/specNeb_/
// specNebExtinct_ -- via spec()/specExtinct()/specNeb()/
// specNebExtinct(), not the raw members, so this computes a current
// spectrum first if needed, regardless of call order (see this
// method's own header comment)
void core::Cluster::computePhot()
{
    const auto& sc = controls_.get();
    const auto filters = sc.filters();
    if (filters == nullptr) { return; }

    phot_ = filters->phot(sc.specsyn()->wlObs(), spec());

    const auto ext = sc.extinct();
    if (ext != nullptr)
    {
        photExtinct_ = filters->phot(ext->wlObs(), specExtinct());
    }

    if (sc.nebular() != nullptr)
    {
        photNeb_ = filters->phot(sc.specsyn()->wlObs(), specNeb());
        if (ext != nullptr)
        {
            photNebExtinct_ = filters->phot(ext->wlObs(), specNebExtinct());
        }
    }
}

// Compute the population's bolometric luminosity at the current
// isochrone -- see this method's own header comment for the
// null-guard and the two-part (stochastic + non-stochastic) structure
// it mirrors from computeSpec()
void core::Cluster::computeLbol()
{
    const auto& sc = controls_.get();
    if (!sc.computeLbol()) { return; }

    lbol_ = 0.0;

    // Individually-sampled (stochastic) stars
    for (const double m : m_)
    {
        // Stars below the tracks' minimum mass have no isochrone
        // segment to evaluate on (this can happen when the IMF
        // extends below the tracks' mass range); skip them, treating
        // their contribution as negligible -- mirrors computeSpec()'s
        // own identical skip
        const auto seg = std::ranges::find_if(isochrone_,
            [m](const auto& segment) -> bool
            { return m >= segment->xMin() && m <= segment->xMax(); });
        if (seg == isochrone_.end()) { continue; }

        const auto logL = (**seg)(m, static_cast<size_t>(tracks::FieldIdx::logL));
        lbol_ += std::pow(10.0, logL);
    }

    // Continuously-sampled (non-stochastic) part of the population:
    // integrate lbolStar against the IMF over each isochrone segment,
    // mirroring Specsyn::specCts's own per-segment integration (see
    // its own comment for why -- an isochrone may have gaps between
    // segments that the quadrature routine has no way to know to
    // avoid)
    if (birthNonStochMass_ > 0.0)
    {
        using LbolSegFn = std::array<double, 1> (*)(double, const Segment&);
        const utils::PDFIntegrator<LbolSegFn, utils::GKOrder::GK15> integrator(
            sc.imf(), static_cast<LbolSegFn>(&Cluster::lbolStar), 1,
            false, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());

        const double mMin = sc.imf().getMin();
        const double mMax = sc.minStochMass();
        double lbolCts = 0.0;
        for (const auto& seg : isochrone_)
        {
            const double a = std::max(mMin, seg->xMin());
            const double b = std::min(mMax, seg->xMax());
            if (a >= b) { continue; } // empty intersection with [mMin, mMax]

            const auto segResult = integrator.integrate(a, b, *seg);
            lbolCts += segResult[0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- segResult is a std::array<double, 1>, so index 0 is always valid
        }
        // imf() is normalized by number, so lbolCts is per star;
        // convert to per unit non-stochastic mass -- see
        // SimControls::nonStochIMFMass()'s own comment
        lbol_ += lbolCts * birthNonStochMass_ / sc.nonStochIMFMass();
    }
}

// Update yields_ from the stars that died since lastYieldTime_ -- see
// this method's own header comment for both halves (stochastic and
// non-stochastic) and for why this accumulates onto yields_ rather
// than recomputing it from scratch.
void core::Cluster::computeYields()
{
    const auto& sc = controls_.get();
    const auto yields = sc.yields();
    if (yields == nullptr) { return; }

    const bool decomposed = sc.yieldsChannelDecomposed();

    // Age yields_'s own already-accumulated total forward by the time
    // elapsed since it was last updated, before adding in this step's
    // new contributions below (each of which uses its own exact
    // dtDecay for the time since it was produced, since it is only now
    // being added) -- exact, not an approximation, by the decay
    // operator's own compositional (semigroup) property: applying
    // decay for dt1 then dt2 gives the same result as applying it once
    // for dt1 + dt2, for whatever abundances are present at the start
    // of each step, so there is no need to separately track when each
    // contribution was originally produced. Mirrors
    // Galaxy::computeYields()'s own identical aging step for
    // fieldYields_.
    if (!sc.noDecay())
    {
        yields->applyDecay(curTime_ - lastYieldTime_, yields_, decomposed);
    }

    // Stochastic (individually-sampled) stars that died during the
    // most recent advance() call -- mDead_/tDied_ are the same length
    // and in the same order (both filled together by
    // updateLivingStars(), see its own comment). dtDecay is the time
    // elapsed since each star actually died (curTime_ - tDied_[i]) if
    // controls().noDecay() is false, or 0 (no decay applied at all) if
    // it is true.
    for (std::size_t i = 0; i < mDead_.size(); ++i)
    {
        const double mass = mDead_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < mDead_.size() by loop bound
        const double tDied = tDied_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < mDead_.size() == tDied_.size() by construction, see updateLivingStars()'s own comment
        const double dtDecay = sc.noDecay() ? 0.0 : curTime_ - tDied;
        if (decomposed)
        {
            const auto& data = yields->yield(mass, feH_, dtDecay).second;
            for (std::size_t k = 0; k < data.size(); ++k)
            {
                yields_[k] += data[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- yields_ and data are both sized nchannels * isotopes().size() by construction, see yields_'s own comment
            }
        }
        else
        {
            const auto sum = yields->yieldSum(mass, feH_, dtDecay);
            for (std::size_t j = 0; j < sum.size(); ++j)
            {
                yields_[j] += sum[j]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- yields_ and sum are both sized isotopes().size() by construction, see yields_'s own comment
            }
        }
    }

    // Continuously-sampled (non-stochastic) stars that died between
    // lastYieldTime_ and curTime_ -- mirrors computeLbol()'s own
    // identical guard: nothing to do if there is no such population
    if (birthNonStochMass_ <= 0.0) { return; }

    // imf() is normalized by number, so each integral is per star;
    // scale to this cluster's own non-stochastic mass -- see
    // SimControls::nonStochIMFMass()'s own comment
    const double scale = birthNonStochMass_ / sc.nonStochIMFMass();

    // Masses alive at lastYieldTime_ but dead now -- see
    // nonStochDeadMassRanges()'s own comment
    const auto yieldAt = [&](const double m) -> std::vector<double>
    { return yieldStar(m, feH_, *yields, decomposed, sc, curTime_, formTime_, tracks()); };
    for (const auto& [m0, m1] : nonStochDeadMassRanges(lastYieldTime_))
    {
        // Isotope yields span many orders of magnitude, and some are
        // zero, and only a small fraction of stars die in one step, so
        // the integrand is made dimensionless first, scaled by its
        // smallest nonzero yield at either end or the middle of the
        // range -- see utils::integrateScaled()'s own comment
        const auto segResult = utils::integrateScaled(sc.imf(), yieldAt, yields_.size(),
            m0, m1, { m0, 0.5 * (m0 + m1), m1 }, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());
        for (std::size_t k = 0; k < segResult.size(); ++k)
        {
            yields_[k] += segResult[k] * scale; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- yields_ and segResult are both sized yields_.size() by construction (segResult via nInt_ above)
        }
    }
}

// Non-stochastic mass ranges alive at lastTime but dead at curTime_
// -- see this method's own header comment
auto core::Cluster::nonStochDeadMassRanges(const double lastTime) const
    -> std::vector<std::pair<double, double>>
{
    // Live mass range at lastTime, clamped up to formTime_ so a
    // lastTime still at its initial 0 reads the live range at this
    // cluster's own birth rather than simulation time 0 -- see
    // lastYieldTime_'s own comment
    const double tLast = std::max(lastTime, formTime_);
    const auto logAgeLast = std::max(std::log10(tLast - formTime_), tracks().logTMin());
    const auto liveMassRangeLast = tracks().liveMassRange(logAgeLast);

    // Live mass range now, read directly off isochrone_'s own segments
    // (already current as of curTime_, from advance()) rather than
    // calling tracks().liveMassRange() a second time
    std::vector<std::pair<double, double>> liveMassRangeNow;
    liveMassRangeNow.reserve(isochrone_.size());
    for (const auto& seg : isochrone_)
    {
        liveMassRangeNow.emplace_back(seg->xMin(), seg->xMax());
    }

    // Masses alive at lastTime but dead now, clipped to lie below
    // minStochMass() (the non-stochastic population's own upper mass
    // limit -- any part at or above it belongs to the stochastic
    // stars, handled via mDead_), dropping any range left empty
    const double mStoch = controls_.get().minStochMass();
    std::vector<std::pair<double, double>> result;
    for (const auto& [lo, hi] : subtractMassRanges(liveMassRangeLast, liveMassRangeNow))
    {
        const double m1 = std::min(hi, mStoch);
        if (lo < m1) { result.emplace_back(lo, m1); }
    }
    return result;
}

// Update the cumulative feedback quantities from the stars that died
// since lastFeedbackTime_ -- see this method's own header comment
void core::Cluster::computeFeedback()
{
    const auto& sc = controls_.get();

    // Stellar wind fluxes: instantaneous rates at curTime_, so
    // recomputed from scratch, from the living stars
    mDotWind_ = 0.0;
    pDotWind_ = 0.0;
    eDotWind_ = 0.0;
    const auto windsPtr = sc.winds();
    const feedback::Winds* const winds = windsPtr.get();

    // Individually-sampled (stochastic) stars, skipping any with no
    // isochrone segment -- mirrors computeLbol()'s own identical loop
    for (const double m : m_)
    {
        const auto seg = std::ranges::find_if(isochrone_,
            [m](const auto& segment) -> bool
            { return m >= segment->xMin() && m <= segment->xMax(); });
        if (seg == isochrone_.end()) { continue; }

        const auto [mDot, pDot, eDot] = windStar(m, **seg, winds, feH_);
        mDotWind_ += mDot;
        pDotWind_ += pDot;
        eDotWind_ += eDot;
    }

    // Continuously-sampled (non-stochastic) part of the population:
    // integrate windStar against the IMF over each isochrone segment,
    // exactly as computeLbol() integrates lbolStar
    if (birthNonStochMass_ > 0.0)
    {
        using WindSegFn = std::array<double, 3> (*)(double, const Segment&, const feedback::Winds*, double);
        const utils::PDFIntegrator<WindSegFn, utils::GKOrder::GK15> integrator(
            sc.imf(), static_cast<WindSegFn>(&Cluster::windStar), 3,
            false, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());

        const double mMin = sc.imf().getMin();
        const double mMax = sc.minStochMass();
        std::array<double, 3> windCts{};
        for (const auto& seg : isochrone_)
        {
            const double a = std::max(mMin, seg->xMin());
            const double b = std::min(mMax, seg->xMax());
            if (a >= b) { continue; } // empty intersection with [mMin, mMax]

            const auto segResult = integrator.integrate(a, b, *seg, winds, feH_);
            for (std::size_t i = 0; i < windCts.size(); ++i) { windCts.at(i) += segResult.at(i); }
        }
        // imf() is normalized by number, so windCts is per star;
        // convert to per unit non-stochastic mass -- see
        // SimControls::nonStochIMFMass()'s own comment
        const double scale = birthNonStochMass_ / sc.nonStochIMFMass();
        mDotWind_ += windCts[0] * scale; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- windCts is a std::array<double, 3>
        pDotWind_ += windCts[1] * scale; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        eDotWind_ += windCts[2] * scale; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
    }

    // Stochastic (individually-sampled) stars that died during the
    // most recent advance() call
    stochSN_ += static_cast<unsigned long>(
        std::ranges::count_if(mDead_,
            [&sc, this](const double m) -> bool { return sc.hasSN(m, feH_); }));

    // Continuously-sampled (non-stochastic) stars that died between
    // lastFeedbackTime_ and curTime_ -- mirrors computeYields()'s own
    // identical integration of yieldStar()
    if (birthNonStochMass_ <= 0.0) { return; }

    // imf() is normalized by number, so each integral is a number of
    // SNe per star; scale to this cluster's own non-stochastic mass --
    // see SimControls::nonStochIMFMass()'s own comment
    const double scale = birthNonStochMass_ / sc.nonStochIMFMass();

    // Integrated exactly as computeYields() integrates yieldStar(), via
    // utils::integrateScaled(): only a small fraction of stars die in
    // one step, so the raw per-star integral is far below intAbsTol(),
    // and an absolute tolerance in its own units would accept a first,
    // unresolved estimate of this step-function integrand (snStar() is
    // 0 or 1); integrateScaled() instead sets the absolute tolerance
    // relative to imf()'s own integral over the range
    const auto snAt = [&sc, this](const double m) -> std::vector<double> { return snStar(m, sc, feH_); };
    for (const auto& [m0, m1] : nonStochDeadMassRanges(lastFeedbackTime_))
    {
        const auto segResult = utils::integrateScaled(sc.imf(), snAt, 1, m0, m1,
            { m0, 0.5 * (m0 + m1), m1 }, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());
        nonStochSN_ += segResult.at(0) * scale;
    }
}

// Per-star supernova count, given a mass -- see this method's own
// header comment
auto core::Cluster::snStar(const double m, const io::SimControls& controls, const double feH)
    -> std::vector<double>
{
    return { controls.hasSN(m, feH) ? 1.0 : 0.0 };
}

// Per-star bolometric luminosity, given a mass and isochrone segment
// -- see this method's own header comment for why it returns a
// single-element array rather than a bare double
auto core::Cluster::lbolStar(const double m, const Segment& segment) -> std::array<double, 1>
{
    const auto logL = segment(m, static_cast<size_t>(tracks::FieldIdx::logL));
    return { std::pow(10.0, logL) };
}

// Per-star wind mass, momentum, and energy fluxes, given a mass and
// isochrone segment -- see this method's own header comment
auto core::Cluster::windStar(const double m, const Segment& segment,
    const feedback::Winds* const winds, const double feH) -> std::array<double, 3>
{
    const specsyn::Specsyn::StarData props = segment(m);
    const double mDot = props[static_cast<std::size_t>(tracks::FieldIdx::mdot)]; // Msun/yr // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and this index is compile-time-known
    const double vWind = (winds != nullptr) ? winds->vWind(props, feH) : 0.0;     // cm/s
    const double mDotCgs = mDot * utils::Msun / utils::yr;                          // g/s
    return { mDot, mDotCgs * vWind, 0.5 * mDotCgs * vWind * vWind };
}

// Per-star nucleosynthetic yield, given a mass and [Fe/H] -- see this
// method's own header comment for why m/feH/yields/decomposed/
// controls/curTime/formTime are all taken as explicit arguments rather
// than captured state, and for the dtDecay convention below
auto core::Cluster::yieldStar(
    const double m, const double feH, const yields::Yields& yields, const bool decomposed,
    const io::SimControls& controls, const double curTime, const double formTime,
    const tracks::Tracks2D& tracks2D) -> std::vector<double>
{
    const double dtDecay = controls.noDecay() ? 0.0 :
        curTime - formTime - starLifetimeClamped(tracks2D, m);
    if (decomposed) { return yields.yield(m, feH, dtDecay).second; }
    return yields.yieldSum(m, feH, dtDecay);
}
