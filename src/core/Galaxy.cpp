/**
 * @file Galaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of Galaxy
 * @date 2026-08-10
 */

#include "Galaxy.hpp"
#include "../extinct/Extinct.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimControls.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFReflect.hpp"
#include "../phot/FilterCollection.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/GKIntegrator.hpp"
#include "../utils/PDFIntegrator.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "Cluster.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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
    const double fCluster = sc.fCluster();

    // 1) Total stellar mass that should have formed between curTime_
    // and t, and 2) the individual cluster masses drawn from the CMF
    // to reach the stochastically-treated fraction (fCluster) of that
    // target -- the remaining (1 - fCluster) forms as a continuous
    // (non-clustered) population, not represented by any individual
    // Cluster object, so actualMass_ folds its own target mass in
    // directly rather than via newMasses
    const double mNew = sc.sfr().integral(curTime_, t);
    const auto newMasses = sc.cmf().drawTarget(mNew * fCluster);
    targetMass_ += mNew;
    actualMass_ += ((1.0 - fCluster) * mNew) +
        std::accumulate(newMasses.begin(), newMasses.end(), 0.0);

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

    // 7) Mark spec_/specExtinct_/phot_/photExtinct_/lbol_/lbolCts_ as
    // stale; recomputed lazily, on demand, the next time spec()/
    // specExtinct()/phot()/photExtinct()/lbol() is actually called
    specCurrent_ = false;
    photCurrent_ = false;
    lbolCurrent_ = false;
    lbolCtsCurrent_ = false;

    // 8) Update current time
    curTime_ = t;
}

// Sum spec_ (and specExtinct_) over every cluster in clusters_ and
// disruptedClusters_ -- see this method's own header comment for the
// null-guards this mirrors from Cluster::computeSpec() -- then, if
// fCluster() < 1, add the continuously-treated (non-clustered) share
// of the population's own spectrum (via Specsyn::specCts()'s
// continuous-population overload) to both, unattenuated in
// specExtinct_'s own case -- see that block's own comment for why
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

    // Add the continuously-treated (non-clustered) share of the
    // population's own spectrum, if any, to both spec_ and (if an
    // extinction curve was requested) specExtinct_ -- unlike a bound
    // cluster, which draws its own A_V, the continuous/field
    // population is assumed negligibly extincted, so its own light
    // passes into specExtinct_ unattenuated. If Lbol was also
    // requested, gets it via specAndLbolCts() (a byproduct of the same
    // integral) rather than paying for a second one via the standalone
    // Lbol path -- see lbolCtsCurrent_'s own comment.
    const double fCluster = sc.fCluster();
    if (fCluster < 1.0)
    {
        std::vector<double> contSpec;
        if (sc.computeLbol())
        {
            auto [s, l] = synth->specAndLbolCts(sc.sfr(), sc.imf(), sc.fehDist(), curTime_, fCluster);
            contSpec = std::move(s);
            lbolCts_ = l;
            lbolCtsCurrent_ = true;
        }
        else
        {
            contSpec = synth->specCts(sc.sfr(), sc.imf(), sc.fehDist(), curTime_, fCluster);
        }

        for (std::size_t i = 0; i < spec_.size(); ++i) { spec_[i] += contSpec[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- contSpec has size wl().size() by Specsyn::specCts()'s own contract, matching spec_'s size set just above
        if (ext != nullptr)
        {
            // specExtinct_ sits on ext->wl(), a possibly-narrower,
            // offset window of spec_'s own wavelength grid (see
            // Extinct::applyExtinction()'s own wlOffset_ comment) --
            // specExtinct_[i] lines up with contSpec[wlOffset + i], not
            // contSpec[i], and (per this function's own comment) gets
            // no exp(-A_V * extinct()) attenuation of its own.
            const auto wlOffset = ext->wlOffset();
            for (std::size_t i = 0; i < specExtinct_.size(); ++i) { specExtinct_[i] += contSpec[wlOffset + i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- specExtinct_.size() == wl_.size() <= contSpec.size() - wlOffset by Extinct's own constructor contract (wl_ is a sub-window of the wl originally passed to it, which matches contSpec's own grid)
        }
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
// plus the continuous population's own share if already current, unless
// Lbol was never requested (see this method's own header comment)
void core::Galaxy::computeLbol()
{
    const auto& sc = controls_.get();
    if (!sc.computeLbol()) { return; }

    lbol_ = 0.0;
    for (auto& cluster : clusters_) { lbol_ += cluster.lbol(); }
    for (auto& cluster : disruptedClusters_) { lbol_ += cluster.lbol(); }

    if (lbolCtsCurrent_)
    {
        lbol_ += lbolCts_;
    }
    else if (sc.fCluster() < 1.0)
    {
        // Lbol was requested and there is a continuous-population
        // share to account for, but computeSpec() hasn't run since the
        // last advance() to compute it as a byproduct -- e.g. a caller
        // asked for lbol() without ever asking for spec(). Get it via
        // the standalone path instead, which doesn't pay for a full
        // spectrum it wasn't asked for.
        computeLbolCts();
        lbol_ += lbolCts_;
    }
}

auto core::Galaxy::lbolCtsIntegrand(
    const double age,
    const pdfs::PDF& imf,
    const double feh
) const -> std::vector<double>
{
    const auto& sc = controls_.get();
    const double logAge = std::max(std::log10(age), sc.tracks().logTMin());
    const auto isochrone = sc.tracks().getIsochrone(logAge, feh);

    // Per-star Lbol, mirroring Cluster::lbolStar()'s own role for
    // Cluster::computeLbol()'s identical inner mass integral -- a
    // local, capture-free lambda rather than a call into any Specsyn,
    // since a Specsyn may not exist at all here (see computeLbolCts()'s
    // own comment).
    const auto lbolStar = [](const double m, const specsyn::Specsyn::Segment& segment) -> std::array<double, 1>
    {
        const auto props = segment(m);
        const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
        return { std::pow(10.0, logL) };
    };
    using LbolSegFn = decltype(lbolStar);

    const utils::PDFIntegrator<LbolSegFn, utils::GKOrder::GK15> integrator(
        imf, lbolStar, 1, false, sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());

    double lbolRaw = 0.0;
    for (const auto& seg : isochrone)
    {
        const double a = std::max(imf.getMin(), seg->xMin());
        const double b = std::min(imf.getMax(), seg->xMax());
        if (a >= b) { continue; } // empty intersection with [imf.getMin(), imf.getMax()]

        const auto segResult = integrator.integrate(a, b, *seg);
        lbolRaw += segResult[0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- segResult is a std::array<double, 1>, so index 0 is always valid
    }
    return { lbolRaw };
}

void core::Galaxy::computeLbolCts()
{
    const auto& sc = controls_.get();
    const auto& sfr = sc.sfr();
    const auto& imf = sc.imf();
    const auto& fehDist = sc.fehDist();
    const double fCluster = sc.fCluster();

    // See Specsyn::specCtsHelper()'s own comment for why this reflects
    // sfr about curTime_ / 2 (so this dimension's own coordinate
    // becomes age directly), why ageMin = min(1e4 yr, 1e-3 * curTime_)
    // rather than anything derived from tracks().logTMin(), and
    // log-transforms whenever curTime_ exceeds that floor; see
    // computeLbolCts()'s own header comment for why absTol has no
    // utils::Lsun factor here, unlike Specsyn::specCtsHelper()'s own.
    const pdfs::PDFReflect sfrAge(sfr, 0.5 * curTime_);
    const double ageMin = std::min(1e4, 1e-3 * curTime_);
    const bool logAge = curTime_ > ageMin;

    const bool singleFeh = (fehDist.getMin() == fehDist.getMax());
    const double absTol = sc.intAbsTol() * sfr.integral(0.0, curTime_);

    using IntegrandFn = std::vector<double> (Galaxy::*)(double, const pdfs::PDF&, double) const;
    const utils::PDFIntegrator<IntegrandFn, utils::GKOrder::GK15> integrator(
        sfrAge, static_cast<IntegrandFn>(&Galaxy::lbolCtsIntegrand),
        1, logAge, sc.intMaxIter(), absTol, sc.intRelTol());

    double lbolRaw = 0.0;
    if (singleFeh)
    {
        const auto result = integrator.integrate(ageMin, curTime_, this, imf, fehDist.getMin());
        lbolRaw = result[0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has exactly 1 element, by construction (nInt == 1)
    }
    else
    {
        // Integrate over [Fe/H] by running the same nested (age, mass)
        // integral once at every grid point the tracks are actually
        // defined at, then interpolating and integrating those
        // discrete results over [Fe/H] -- see
        // Specsyn::specCtsHelper()'s own comment for why, in place of
        // a joint 3D (feh, age, mass) cubature integral.
        const auto& fehGrid = sc.tracks().feH();
        const std::size_t nFeh = fehGrid.size();

        std::vector<double> lbolAtFeh(nFeh);
        std::vector<double> fehWeight(nFeh);
        for (std::size_t f = 0; f < nFeh; ++f)
        {
            const auto result = integrator.integrate(ageMin, curTime_, this, imf, fehGrid[f]);
            fehWeight[f] = fehDist(fehGrid[f]);
            lbolAtFeh[f] = result[0] * fehWeight[f]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has exactly 1 element, by construction (nInt == 1)
        }

        const interp::Interpolator1D<1> weightInterp(fehGrid, fehWeight);
        const double weightIntegral = weightInterp.integ(fehDist.getMin(), fehDist.getMax());
        const interp::Interpolator1D<1> lbolInterp(fehGrid, lbolAtFeh);
        lbolRaw = lbolInterp.integ(fehDist.getMin(), fehDist.getMax()) / weightIntegral;
    }

    lbolCts_ = lbolRaw * (1.0 - fCluster);
    lbolCtsCurrent_ = true;
}
