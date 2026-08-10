/**
 * @file Galaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of Galaxy
 * @date 2026-08-10
 */

#include "Galaxy.hpp"
#include "../extinct/Extinct.hpp"
#include "../io/SimControls.hpp"
#include "../phot/FilterCollection.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "Cluster.hpp"
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// Constructor: everything but controls_ takes its in-class default
// (curTime_/lbol_ = 0, every vector empty)
core::Galaxy::Galaxy(const io::SimControls& controls) :
    controls_(std::cref(controls))
{
}

// Advance function
void core::Galaxy::advance(const double t)
{
    // Make sure t >= curTime_
    if (t < curTime_)
    {
        std::stringstream ss;
        ss << "Galaxy: requested advance to " << t
            << ", but curTime = " << curTime_;
        throw std::runtime_error(ss.str());
    }

    const auto& sc = controls_.get();

    // 1) Total mass of new clusters that should have formed between
    // curTime_ and t, and 2) the individual cluster masses drawn from
    // the CMF to reach that target
    const double mClusterNew = sc.sfr().integral(curTime_, t);
    const auto newMasses = sc.cmf().drawTarget(mClusterNew);
    targetMass_ += mClusterNew;
    actualMass_ += std::accumulate(newMasses.begin(), newMasses.end(), 0.0);

    // 3-4) For each new cluster, draw a formation time from the SFR
    // over (curTime_, t] and create it, with a unique ID from the uid
    // service, appending it to clusters_
    for (const double mass : newMasses)
    {
        const double formTime = sc.sfr().draw(curTime_, t);
        clusters_.emplace_back(utils::getID(), mass, formTime, sc);
    }

    // 5) Advance every cluster formed so far -- both still-alive ones
    // (including the brand new ones just appended above) and
    // already-disrupted ones, since a disrupted cluster's stars keep
    // evolving even after it stops being counted as a bound cluster
    for (auto& cluster : clusters_) { cluster.advance(t); }
    for (auto& cluster : disruptedClusters_) { cluster.advance(t); }

    // 6) Move any cluster that disrupted during this step from
    // clusters_ to disruptedClusters_
    std::vector<Cluster> stillAlive;
    stillAlive.reserve(clusters_.size());
    for (auto& cluster : clusters_)
    {
        if (cluster.isDisrupted())
        {
            disruptedClusters_.push_back(std::move(cluster));
        }
        else
        {
            stillAlive.push_back(std::move(cluster));
        }
    }
    clusters_ = std::move(stillAlive);

    // 7) Mark spec_/specExtinct_/phot_/photExtinct_/lbol_ as stale;
    // recomputed lazily, on demand, the next time spec()/specExtinct()/
    // phot()/photExtinct()/lbol() is actually called
    specCurrent_ = false;
    photCurrent_ = false;
    lbolCurrent_ = false;

    // 8) Update current time
    curTime_ = t;
}

// Sum spec_ (and specExtinct_) over every cluster in clusters_ and
// disruptedClusters_ -- see this method's own header comment for the
// null-guards this mirrors from Cluster::computeSpec()
void core::Galaxy::computeSpec()
{
    const auto& sc = controls_.get();
    const auto* synth = sc.specsyn();
    if (synth == nullptr) { return; }

    spec_.assign(synth->wl().size(), 0.0);
    const auto sumSpec = [this](std::vector<Cluster>& clusterList)
    {
        for (auto& cluster : clusterList)
        {
            const auto& clusterSpec = cluster.spec();
            for (std::size_t i = 0; i < spec_.size(); ++i) { spec_[i] += clusterSpec[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterSpec has size wl().size() by Cluster::computeSpec()'s own contract, matching spec_'s size set just above
        }
    };
    sumSpec(clusters_);
    sumSpec(disruptedClusters_);

    const auto* ext = sc.extinct();
    if (ext != nullptr)
    {
        specExtinct_.assign(ext->wl().size(), 0.0);
        const auto sumSpecExtinct = [this](std::vector<Cluster>& clusterList)
        {
            for (auto& cluster : clusterList)
            {
                const auto& clusterSpecExtinct = cluster.specExtinct();
                for (std::size_t i = 0; i < specExtinct_.size(); ++i) { specExtinct_[i] += clusterSpecExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterSpecExtinct has size extinct()->wl().size() by Cluster::computeSpec()'s own contract, matching specExtinct_'s size set just above
            }
        };
        sumSpecExtinct(clusters_);
        sumSpecExtinct(disruptedClusters_);
    }
}

// Update the galaxy's photometry (and, if an extinction curve was
// requested, extincted photometry) from the current spec()/specExtinct()
// -- identical to Cluster::computePhot(), just reading this Galaxy's
// own spec()/specExtinct() (the lazy getters, not spec_/specExtinct_
// directly, so this forces a current summed spectrum first if needed,
// regardless of call order)
void core::Galaxy::computePhot()
{
    const auto& sc = controls_.get();
    const auto& filters = sc.filters();
    if (filters == nullptr) { return; }

    phot_ = filters->phot(sc.specsyn()->wlObs(), spec());

    const auto* ext = sc.extinct();
    if (ext != nullptr)
    {
        photExtinct_ = filters->phot(ext->wlObs(), specExtinct());
    }
}

// Sum lbol_ over every cluster in clusters_ and disruptedClusters_,
// unless Lbol was never requested (see this method's own header comment)
void core::Galaxy::computeLbol()
{
    const auto& sc = controls_.get();
    if (!sc.computeLbol()) { return; }

    lbol_ = 0.0;
    for (auto& cluster : clusters_) { lbol_ += cluster.lbol(); }
    for (auto& cluster : disruptedClusters_) { lbol_ += cluster.lbol(); }
}
