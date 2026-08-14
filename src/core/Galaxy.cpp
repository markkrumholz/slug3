/**
 * @file Galaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of Galaxy
 * @date 2026-08-10
 */

#include "Galaxy.hpp"
#include "../extinct/Extinct.hpp"
#include "../io/SimControls.hpp"
#include "../pdfs/PDF.hpp"
#include "../phot/FilterCollection.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/PDFIntegratorND.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "Cluster.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <mdspan> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLibWR.hpp's own <mdspan> include
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

template <std::size_t Ndim>
auto core::Galaxy::lbolCtsIntegrand(
    const std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, Ndim>> points, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
    std::map<std::pair<double, double>, specsyn::Specsyn::Isochrone>& cache,
    const double curTime
) const -> std::vector<double>
{
    static_assert(Ndim == 2 || Ndim == 3, "lbolCtsIntegrand only supports Ndim == 2 or 3");

    const auto& sc = controls_.get();
    specsyn::Specsyn::cacheIsochrones<Ndim>(points, cache, curTime, sc);

    const std::size_t npts = points.extent(0);
    std::vector<double> resultFlat(npts, 0.0);
    for (std::size_t i = 0; i < npts; ++i)
    {
        const double age = curTime - points[i, 0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < npts == points.extent(0) by the loop bound
        double feh = 0.0;
        double mass = 0.0;
        if constexpr (Ndim == 3) { feh = points[i, 1]; mass = points[i, 2]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        else { feh = sc.fehDist().getMin(); mass = points[i, 1]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

        const auto& isochrone = cache.at({ age, feh });
        const specsyn::Specsyn::Segment* seg = nullptr;
        for (const auto& s : isochrone)
        {
            if (mass >= s->xMin() && mass <= s->xMax()) { seg = s.get(); break; }
        }
        if (seg == nullptr) { continue; } // dead star: leave this point's own value at zero

        const auto props = (*seg)(mass);
        const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
        resultFlat[i] = std::pow(10.0, logL); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < npts == resultFlat.size() by construction
    }
    return resultFlat;
}

// Explicit instantiation for the two dimensionalities this is ever
// used with (2: time+mass, single feh; 3: time+feh+mass).
template auto core::Galaxy::lbolCtsIntegrand<2>(
    std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
    std::map<std::pair<double, double>, specsyn::Specsyn::Isochrone>&, double) const -> std::vector<double>;
template auto core::Galaxy::lbolCtsIntegrand<3>(
    std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 3>>, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
    std::map<std::pair<double, double>, specsyn::Specsyn::Isochrone>&, double) const -> std::vector<double>;

void core::Galaxy::computeLbolCts()
{
    const auto& sc = controls_.get();
    const auto& sfr = sc.sfr();
    const auto& imf = sc.imf();
    const auto& fehDist = sc.fehDist();
    const double fCluster = sc.fCluster();

    // See Specsyn::specCts()'s own comment for why the time dimension
    // is never log-transformed, while the mass dimension always is;
    // see computeLbolCts()'s own header comment for why absTol has no
    // utils::Lsun factor here, unlike Specsyn::specCtsHelper()'s own.
    const bool singleFeh = (fehDist.getMin() == fehDist.getMax());
    const double absTol = sc.intAbsTol() * sfr.integral(0.0, curTime_);

    std::vector<double> result;
    if (singleFeh)
    {
        using IntegrandFn = std::vector<double> (Galaxy::*)(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
            std::map<std::pair<double, double>, specsyn::Specsyn::Isochrone>&, double) const;
        const utils::PDFIntegratorND<IntegrandFn, 2, true, utils::CubatureMethod::pAdaptive> integrator(
            std::array<std::reference_wrapper<const pdfs::PDF>, 2>{ std::cref(sfr), std::cref(imf) },
            static_cast<IntegrandFn>(&Galaxy::lbolCtsIntegrand<2>),
            1,
            std::array<bool, 2>{ false, true },
            sc.intMaxIter(), absTol, sc.intRelTol());
        result = integrator.integrate(
            std::array<double, 2>{ 0.0, imf.getMin() },
            std::array<double, 2>{ curTime_, imf.getMax() },
            *this, isochroneCache_, curTime_);
    }
    else
    {
        using IntegrandFn = std::vector<double> (Galaxy::*)(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 3>>, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
            std::map<std::pair<double, double>, specsyn::Specsyn::Isochrone>&, double) const;
        const utils::PDFIntegratorND<IntegrandFn, 3, true, utils::CubatureMethod::pAdaptive> integrator(
            std::array<std::reference_wrapper<const pdfs::PDF>, 3>{
                std::cref(sfr), std::cref(fehDist), std::cref(imf) },
            static_cast<IntegrandFn>(&Galaxy::lbolCtsIntegrand<3>),
            1,
            std::array<bool, 3>{ false, false, true },
            sc.intMaxIter(), absTol, sc.intRelTol());
        result = integrator.integrate(
            std::array<double, 3>{ 0.0, fehDist.getMin(), imf.getMin() },
            std::array<double, 3>{ curTime_, fehDist.getMax(), imf.getMax() },
            *this, isochroneCache_, curTime_);
    }

    lbolCts_ = result[0] * (1.0 - fCluster); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has exactly 1 element, by construction (nInt == 1)
    lbolCtsCurrent_ = true;
    isochroneCache_.clear();
}
