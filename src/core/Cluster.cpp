/**
 * @file Cluster.cpp
 * @author Mark Krumholz
 * @brief Implementation of Cluster
 * @date 2026-07-12
 */

#include "Cluster.hpp"
#include "../io/SimControls.hpp"
#include "../phot/FilterCollection.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/PDFIntegrator.hpp"
#include "../utils/RngThread.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <variant>

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

    // If this simulation has a variable metallicity, generate tracks
    // for the metallicity of this cluster and save them for future
    // use. Otherwise, [Fe/H] is fixed for the whole simulation, so
    // just point at the slice SimControls has already precomputed and
    // shares across every cluster.
    if (sc.constFeH())
    {
        tracks_ = std::cref(sc.tracks2D());
    }
    else
    {
        tracks_ = sc.tracks().sliceConstFeH(feH_);
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
    controls_(std::cref(controls)),
    birthNonStochMass_((1.0 - controls.fracStochMass()) * mass),
    birthMass_(0.0), // placeholder; overwritten once m_ is drawn
    disruptTime_(std::numeric_limits<double>::quiet_NaN()),
    curTime_(time)
{
    // Draw feH_ and m_ from the given rng state rather than the live
    // one: save the live state, temporarily switch to rngState, draw
    // both in the same order the primary constructor does, then
    // restore, so this constructor has no lasting effect on the
    // ambient rng stream
    const auto savedState = utils::rng().getState();
    utils::rng().setState(rngState);
    feH_ = controls.fehDist().draw(); //NOLINT(cppcoreguidelines-prefer-member-initializer) -- must happen after setState(rngState) above, not in the initializer list
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

    // If this simulation has a variable metallicity, generate tracks
    // for the metallicity of this cluster and save them for future
    // use. Otherwise, [Fe/H] is fixed for the whole simulation, so
    // just point at the slice SimControls has already precomputed and
    // shares across every cluster.
    if (sc.constFeH())
    {
        tracks_ = std::cref(sc.tracks2D());
    }
    else
    {
        tracks_ = sc.tracks().sliceConstFeH(feH_);
    }
}

// Get the tracks at this cluster's [Fe/H]
auto core::Cluster::tracks() const -> const tracks::Tracks2D&
{
    if (const auto* owned = std::get_if<tracks::Tracks2D>(&tracks_))
    {
        return *owned;
    }
    return std::get<std::reference_wrapper<const tracks::Tracks2D>>(tracks_).get();
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
    updateLivingStars(logAge);

    // Get isochrone for new time
    isochrone_ = tracks().getIsochrone(logAge);

    // Update the population spectrum, if a spectral synthesizer was requested
    computeSpec();

    // Update the population photometry from the spectrum just
    // computed above, if a filter collection was requested
    const auto& sc = controls_.get();
    const auto& filters = sc.filters();
    if (filters != nullptr)
    {
        phot_ = filters->phot(sc.specsyn()->wl(), spec_);
    }

    // Update the population's bolometric luminosity, if "Lbol" was
    // included in phot.filters (see SimControls::computeLbol())
    if (sc.computeLbol()) { computeLbol(); }

    // Check for disruption
    if (curTime_ > disruptTime_) { isDisrupted_ = true; }

    advanced_ = true;
}

// Update lists of alive and dead stars to current age
// NOLINTBEGIN(misc-include-cleaner) -- clang-tidy < 19 lacks stdlib symbol
// mappings for std::ranges::lower_bound / upper_bound; <algorithm> is correct.
void core::Cluster::updateLivingStars(const double logAge)
{
    // Clear the list of dead stars
    mDead_.clear();

    // Get the live mass range for this time
   const auto liveMassRange = tracks().liveMassRange(logAge);

    // Use live mass range to remove dead stars, being
    // careful to handle special case when liveMassRange is
    // empty, indicating all stars are dead
    if (liveMassRange.empty())
    { 
        mDead_ = m_;
        m_.clear();
    }
    else
    {
        // If the first entry in liveMassRange is not the
        // minimum mass in the tracks, kill stars with masses
        // below this mass
        if (liveMassRange.front().first != tracks().mMin())
        {
            auto it = std::ranges::upper_bound(
                m_, liveMassRange.front().first);
            mDead_.insert(mDead_.end(), m_.begin(), it);
        }

        // Loop over entries in liveMassRange, removing stars
        // with masses larger than the second value in one entry
        // and smaller than the first value in the next. Note that
        // we write this as a range-based loop since we need to
        // access both element i and element i+1 inside the same
        // iteration.
        for (size_t i = 0; i < liveMassRange.size()-1; i++)
        {
            auto itStart = std::ranges::lower_bound(
                m_, liveMassRange[i].second); //NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            auto itEnd = std::ranges::upper_bound(
                m_, liveMassRange[i+1].first); //NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            mDead_.insert(mDead_.end(), itStart, itEnd);
            m_.erase(itStart, itEnd);
        }

        // Kill stars more massive than the highest mass in liveMassRange
        auto it = std::ranges::lower_bound(m_, 
            liveMassRange.back().second);
        mDead_.insert(mDead_.end(), it, m_.end());
        m_.erase(it, m_.end());
    }
}
// NOLINTEND(misc-include-cleaner)

// Compute the population spectrum at the current isochrone, if a
// spectral synthesizer was requested
void core::Cluster::computeSpec()
{
    const auto& sc = controls_.get();
    const auto* synth = sc.specsyn();
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
}

// Compute the population's bolometric luminosity at the current
// isochrone -- see this method's own header comment for the
// two-part (stochastic + non-stochastic) structure it mirrors from
// computeSpec()
void core::Cluster::computeLbol()
{
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
    // segments that pcubature has no way to know to avoid)
    if (birthNonStochMass_ > 0.0)
    {
        const auto& sc = controls_.get();
        using LbolSegFn = std::array<double, 1> (*)(double, const Segment&);
        const utils::PDFIntegrator integrator(
            sc.imf(), static_cast<LbolSegFn>(&Cluster::lbolStar), 1,
            sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());

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
        lbol_ += lbolCts * birthNonStochMass_;
    }
}

// Per-star bolometric luminosity, given a mass and isochrone segment
// -- see this method's own header comment for why it returns a
// single-element array rather than a bare double
auto core::Cluster::lbolStar(const double m, const Segment& segment) -> std::array<double, 1>
{
    const auto logL = segment(m, static_cast<size_t>(tracks::FieldIdx::logL));
    return { std::pow(10.0, logL) };
}
