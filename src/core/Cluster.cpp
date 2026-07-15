/**
 * @file Cluster.cpp
 * @author Mark Krumholz
 * @brief Implementation of Cluster
 * @date 2026-07-12
 */

#include "Cluster.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/RngThread.hpp"
#include "SimPhysics.hpp"
#include <algorithm>
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
    const SimPhysics& physics) :
    rngState_(utils::rng().getState()),
    uid_(uid),
    targetMass_(mass),
    formTime_(time),
    feH_(physics.fehDist().draw()),
    physics_(std::cref(physics)),
    m_(physics.imf().drawTarget(mass)),
    birthMass_(std::reduce(m_.begin(), m_.end(), 0.0)),
    disruptTime_(std::numeric_limits<double>::quiet_NaN()),
    curTime_(time),
    isDisrupted_(false)
{
    // Sort the generated population by mass
    std::ranges::sort(m_);

    // Set disruption time if disruption is on
    const auto& ph = physics_.get();
    if (ph.clf().valid())
    { 
        disruptTime_ = formTime_ + ph.clf().draw();
    }

    // If this simulation has a variable metallicity, generate tracks
    // for the metallicity of this cluster and save them for future
    // use. Otherwise, [Fe/H] is fixed for the whole simulation, so
    // just point at the slice SimPhysics has already precomputed and
    // shares across every cluster.
    if (ph.constFeH())
    {
        tracks_ = std::cref(ph.tracks2D());
    }
    else
    {
        tracks_ = ph.tracks().sliceConstFeH(feH_);
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

    // If t == curTime_, do nothing
    if (t == curTime_) { return; }

    // Update time and cluster age
    curTime_ = t;
    const auto logAge = std::log10(curTime_ - formTime_);

    // Update list of alive and dead stars to new cluster age
    updateLivingStars(logAge);

    // Get isochrone for new time
    isochrone_ = tracks().getIsochrone(logAge);

    // Check for disruption
    if (curTime_ > disruptTime_) { isDisrupted_ = true; }
}

// Update lists of alive and dead stars to current age
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
