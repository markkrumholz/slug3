/**
 * @file Galaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of Galaxy
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Galaxy.hpp"
#include "../extinct/Extinct.hpp"
#include "../io/SimControls.hpp"
#include "../nebular/Nebular.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFReflect.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/GKIntegrator.hpp"
#include "../utils/GKIntegratorData.hpp"
#include "../utils/PDFIntegrator.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "../yields/Yields.hpp"
#include "Cluster.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    /**
     * @brief A field star's lifetime, treating masses outside the tracks' own mass grid specially
     * @param tracks The tracks to query -- SimControls::tracks()
     * @param m Stellar mass, in Msun
     * @param feh [Fe/H] of the star
     * @return tracks.starLifetime(m, feh) if tracks.mMin() <= m <=
     *   tracks.mMax(); +infinity if m is below tracks.mMin() (treated
     *   as living forever); -infinity if m is above tracks.mMax()
     *   (treated as already dead)
     * @details
     * Mirrors Cluster's own starLifetimeClamped() (see its own
     * comment), for the same reason: Tracks3D::starLifetime() asserts
     * its mass argument lies within the tracks' own tabulated mass
     * grid (and reads outside it if assertions are disabled), but the
     * IMF's own mass range can extend beyond that grid -- e.g. an IMF
     * minimum below the tracks' own minimum mass, which a field star
     * can be drawn at whenever stars.min_stoch_mass is below the
     * tracks' own minimum.
     */
    auto fieldStarLifetime(const tracks::Tracks3D& tracks, const double m, const double feh) -> double
    {
        if (m < tracks.mMin()) { return std::numeric_limits<double>::infinity(); }
        if (m > tracks.mMax()) { return -std::numeric_limits<double>::infinity(); }
        return tracks.starLifetime(m, feh);
    }
} // namespace

// Constructor: everything but controls_ takes its in-class default
// (curTime_/lbol_ = 0, every vector empty); sfr_ is resolved in the
// body below -- see this class's own constructor comment. SimControls
// only guarantees exactly one of sfr()/sfrDist() valid for a
// galaxy-type simulation -- a Galaxy can also legitimately be
// constructed from a non-galaxy SimControls (e.g. the pybind default,
// PyDefaults.toml, which is sim_type = "cluster" and never sets
// either), so sfrDist() is only drawn from when it's actually valid;
// otherwise sfr_ is left as the default-constructed (invalid) PDF,
// exactly as sc.sfr() itself would have been before sfr_ existed --
// any code that actually needs a valid sfr() (advance(), etc.) fails
// the same way it always did in that case.
core::Galaxy::Galaxy(const io::SimControls& controls) :
    controls_(std::cref(controls))
{
    if (controls.sfr().valid())
    {
        sfr_ = std::cref(controls.sfr());
    }
    else if (controls.sfrDist().valid())
    {
        sfr_ = io::SimControls::buildConstantSFR(controls.sfrDist().draw());
    }

    // If yield channels were requested, size yields_/fieldYields_ to
    // hold one (currently zero) entry per isotope, times one row per
    // channel if SimControls::yieldsChannelDecomposed() is true, or
    // just one combined total per isotope otherwise -- mirrors
    // Cluster::Cluster()'s own identical sizing exactly
    if (const auto yields = controls.yields())
    {
        const std::size_t n = yields->isotopes().size() *
            (controls.yieldsChannelDecomposed() ? yields->yieldChannels().size() : 1);
        yields_.assign(n, 0.0);
        fieldYields_.assign(n, 0.0);
    }
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
    // and t; the individual cluster masses drawn from the CMF to
    // reach the stochastically-treated fraction (fCluster) of that
    // target; and, of the remaining (1 - fCluster), the individual
    // field-star masses drawn from the IMF (over [minStochMass(),
    // imf().getMax()]) to reach fracStochMass() of *that* -- the
    // stochastically-treated share of the non-clustered population
    // (see SimControls::minStochMass()/fracStochMass()'s own
    // comments). What remains after both -- (1 - fCluster) *
    // (1 - fracStochMass()) of mNew -- forms as a purely continuous
    // population, not represented by any individual Cluster or
    // FieldStar object, so actualMass_ folds its own share in
    // directly rather than via newMasses/newFieldMasses.
    const double mNew = sfr().integral(curTime_, t);
    const auto newMasses = sc.cmf().drawTarget(mNew * fCluster);
    const auto newFieldMasses = sc.imf().drawTarget(
        mNew * (1.0 - fCluster) * sc.fracStochMass(),
        sc.minStochMass(), sc.imf().getMax());
    targetMass_ += mNew;
    actualMass_ += ((1.0 - fCluster) * (1.0 - sc.fracStochMass()) * mNew) +
        std::accumulate(newMasses.begin(), newMasses.end(), 0.0) +
        std::accumulate(newFieldMasses.begin(), newFieldMasses.end(), 0.0);

    // 2) For each new cluster, draw a formation time from the SFR
    // over (curTime_, t] and create it, with a unique ID from the uid
    // service, appending it to clusters_
    for (const double mass : newMasses)
    {
        const double formTime = sfr().draw(curTime_, t);
        clusters_.emplace_back(utils::uniqueID().get(), mass, formTime, sc);
    }

    // 3) For each new field star, draw a formation time from the same
    // SFR distribution the clusters above draw from, a [Fe/H] from
    // fehDist(), and its own V-band extinction from avDistField() --
    // an independent draw per star (unlike a bound cluster's own
    // single, shared aV_), mirroring Cluster's own avDist().valid() ?
    // draw() : 0.0 convention for when no extinction was requested at
    // all. Its death time is then formTime + the tracks' own
    // starLifetime() at that (mass, feh), both in yr -- or +/-infinity
    // for a mass below/above the tracks' own mass range, see
    // fieldStarLifetime()'s own comment. Sorting this
    // step's own batch by formTime before appending it keeps
    // fieldStars_ sorted by formTime_ overall: every previously-
    // appended star's own formTime_ already falls at or before
    // curTime_, and every new one falls in (curTime_, t], so appending
    // a locally-sorted batch preserves the whole vector's own global
    // order.
    std::vector<FieldStar> newFieldStars;
    newFieldStars.reserve(newFieldMasses.size());
    for (const double mass : newFieldMasses)
    {
        const double formTime = sfr().draw(curTime_, t);
        const double feh = sc.fehDist().draw();
        const double deathTime = formTime + fieldStarLifetime(*sc.tracks(), mass, feh);
        const double aV = sc.avDistField().valid() ? sc.avDistField().draw() : 0.0;
        newFieldStars.push_back({ mass, feh, formTime, deathTime, aV });
    }
    std::ranges::sort(newFieldStars, {}, &FieldStar::formTime_);
    fieldStars_.insert(fieldStars_.end(), newFieldStars.begin(), newFieldStars.end());

    // 4) Advance every cluster formed so far -- both still-alive ones
    // (including the brand new ones just appended above) and
    // already-disrupted ones, since a disrupted cluster's stars keep
    // evolving even after it stops being counted as a bound cluster
    for (auto& cluster : clusters_) { cluster.advance(t); }
    for (auto& cluster : disruptedClusters_) { cluster.advance(t); }

    // 5) Move any cluster that disrupted during this step from
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

    // 6) Move any field star that has died as of t from fieldStars_ to
    // deadFieldStars_ -- deadFieldStars_ only ever holds the stars
    // that died during *this* step (mirroring Cluster::mDead_'s own
    // per-step convention), so it is cleared first. stable_partition
    // (rather than plain partition/remove_if) preserves fieldStars_'s
    // own formTime_ ordering among the survivors.
    deadFieldStars_.clear();
    const auto deadBegin = std::ranges::stable_partition(fieldStars_,
        [t](const FieldStar& fs) { return fs.deathTime_ >= t; }).begin();
    deadFieldStars_.assign(
        std::make_move_iterator(deadBegin), std::make_move_iterator(fieldStars_.end()));
    fieldStars_.erase(deadBegin, fieldStars_.end());

    // 7) Mark spec_/specExtinct_/phot_/photExtinct_/lbol_/lbolCts_ as
    // stale; recomputed lazily, on demand, the next time spec()/
    // specExtinct()/phot()/photExtinct()/lbol() is actually called
    specCurrent_ = false;
    photCurrent_ = false;
    lbolCurrent_ = false;
    lbolCtsCurrent_ = false;

    // 8) Update current time
    curTime_ = t;

    // 9) Update yields_/fieldYields_ now, eagerly -- unlike
    // spec_/phot_/lbol_ (recomputed lazily, from scratch, on demand),
    // yields_/fieldYields_ only ever accumulate, and rely on
    // deadFieldStars_/each cluster's own mDead_, both of which only
    // ever hold this one step's own deaths (see step 6's own comment,
    // and Cluster::updateLivingStars()'s), so computeYields() must run
    // here, before the next advance() call's own step 6 clears
    // deadFieldStars_ -- see lastYieldTime_'s own comment for the data
    // loss that used to result otherwise.
    computeYields();
    lastYieldTime_ = curTime_;
}

// Sum spec_/specExtinct_ (and, if a nebular emission grid was
// requested, specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_) over
// every cluster in clusters_ and disruptedClusters_ -- see this method's own header
// comment for the null-guards this mirrors from Cluster::computeSpec()
// -- then, if fCluster() < 1, hands off to addContinuousSpec() to add
// the continuously-treated (non-clustered) share of the population's
// own spectrum together with every field star's own contribution
void core::Galaxy::computeSpec()
{
    const auto& sc = controls_.get();
    const auto synth = sc.specsyn();
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

    const auto ext = sc.extinct();
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

    // Sum specNeb_/specNebExtinct_/lineLum_/lineLumExtinct_ over the
    // same clusters, exactly as spec_/specExtinct_ are above, if a nebular emission
    // grid was requested -- see addClusterSpecNeb()'s own comment.
    const auto neb = sc.nebular();
    if (neb != nullptr) { addClusterSpecNeb(ext.get(), neb.get()); }

    // Add the purely continuous (non-clustered, below minStochMass())
    // share of the population's own spectrum, together with every
    // currently-alive field star's own contribution -- see
    // addContinuousSpec()'s own comment for why the two are combined
    // there. Skipped entirely whenever fCluster() == 1 (every
    // non-clustered star is clustered after all, so there is neither a
    // continuous share nor any field star to add) -- fCluster() < 1 is
    // exactly the condition under which fieldStars_ can be non-empty in
    // the first place (see advance()'s own comment), so this single
    // guard covers both contributions.
    if (sc.fCluster() < 1.0) { addContinuousSpec(ext.get(), neb.get()); }
}

// Sum specNeb_/lineLum_ (and specNebExtinct_/lineLumExtinct_) over
// every cluster in clusters_ and disruptedClusters_, exactly mirroring computeSpec()'s
// own spec_/specExtinct_ summing -- split out purely to keep
// computeSpec()'s own cognitive complexity down, not for any reuse
// elsewhere. Field stars' own nebular contribution is folded in
// separately, by addContinuousSpec() -- see its own comment.
void core::Galaxy::addClusterSpecNeb(const extinct::Extinct* ext, const nebular::Nebular* neb)
{
    specNeb_.assign(controls_.get().specsyn()->wl().size(), 0.0);
    lineLum_.assign(neb->lineWl().size(), 0.0);
    const auto sumSpecNeb = [this](std::vector<Cluster>& clusterList)
    {
        for (auto& cluster : clusterList)
        {
            const auto& clusterSpecNeb = cluster.specNeb();
            for (std::size_t i = 0; i < specNeb_.size(); ++i) { specNeb_[i] += clusterSpecNeb[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterSpecNeb has size wl().size() by Cluster::computeSpec()'s own contract, matching specNeb_'s size set just above
            const auto& clusterLineLum = cluster.lineLum();
            for (std::size_t i = 0; i < lineLum_.size(); ++i) { lineLum_[i] += clusterLineLum[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterLineLum has size neb->lineWl().size() by Cluster::computeSpec()'s own contract, matching lineLum_'s size set just above
        }
    };
    sumSpecNeb(clusters_);
    sumSpecNeb(disruptedClusters_);

    if (ext != nullptr)
    {
        specNebExtinct_.assign(ext->wl().size(), 0.0);
        lineLumExtinct_.assign(neb->lineWl().size(), 0.0);
        const auto sumSpecNebExtinct = [this](std::vector<Cluster>& clusterList)
        {
            for (auto& cluster : clusterList)
            {
                const auto& clusterSpecNebExtinct = cluster.specNebExtinct();
                for (std::size_t i = 0; i < specNebExtinct_.size(); ++i) { specNebExtinct_[i] += clusterSpecNebExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterSpecNebExtinct has size extinct()->wl().size() by Cluster::computeSpec()'s own contract, matching specNebExtinct_'s size set just above
                const auto& clusterLineLumExtinct = cluster.lineLumExtinct();
                for (std::size_t i = 0; i < lineLumExtinct_.size(); ++i) { lineLumExtinct_[i] += clusterLineLumExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- clusterLineLumExtinct has size neb->lineWl().size() by Cluster::computeSpec()'s own contract, matching lineLumExtinct_'s size set just above
            }
        };
        sumSpecNebExtinct(clusters_);
        sumSpecNebExtinct(disruptedClusters_);
    }
}

void core::Galaxy::addContinuousSpec(const extinct::Extinct* ext, const nebular::Nebular* neb)
{
    // If Lbol was also requested, gets it via specAndLbolCts() (a
    // byproduct of the same integral) rather than paying for a second
    // one via the standalone Lbol path -- see lbolCtsCurrent_'s own
    // comment. Skipped (contSpec left at all-zero) if fracStochMass()
    // == 1: there is no purely continuous share at all, though
    // fieldStars_ may still be non-empty and need adding below.
    const auto& sc = controls_.get();
    const auto synth = sc.specsyn();
    const double fCluster = sc.fCluster();

    std::vector<double> contSpec;
    if (sc.fracStochMass() < 1.0)
    {
        if (sc.computeLbol())
        {
            auto [s, l] = synth->specAndLbolCts(sfr(), sc.imf(), sc.fehDist(), curTime_,
                fCluster, sc.imf().getMin(), sc.minStochMass());
            contSpec = std::move(s);
            lbolCts_ = l;
            lbolCtsCurrent_ = true;
        }
        else
        {
            contSpec = synth->specCts(sfr(), sc.imf(), sc.fehDist(), curTime_,
                fCluster, sc.imf().getMin(), sc.minStochMass());
        }
    }
    else
    {
        contSpec.assign(synth->wl().size(), 0.0);
    }

    // Add every currently-alive field star's own contribution directly
    // into contSpec, before extinction or nebular emission is applied
    // to it below -- see this method's own header comment for why.
    if (!fieldStars_.empty())
    {
        const auto props = getFieldStarProps();
        for (std::size_t j = 0; j < fieldStars_.size(); ++j)
        {
            // A star outside the tracks' own mass range contributes
            // nothing, as in Cluster::computeSpec()
            const auto& starProps = props[j]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- props has size fieldStars_.size() by getFieldStarProps()'s own contract, and j is bounded by fieldStars_.size()
            if (!starProps.has_value()) { continue; }
            const auto starSpec = synth->spec(*starProps, fieldStars_[j].feh_); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- as above
            for (std::size_t i = 0; i < contSpec.size(); ++i) { contSpec[i] += starSpec[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- starSpec has size wl().size() by Specsyn::spec()'s own contract, matching contSpec's size set just above
        }
    }

    for (std::size_t i = 0; i < spec_.size(); ++i) { spec_[i] += contSpec[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- contSpec has size wl().size() by construction, matching spec_'s size set in computeSpec()
    if (ext != nullptr)
    {
        // Unlike a bound cluster, this combined share of the
        // population is not individually tracked, so there is no
        // single A_V to apply -- applyExtinctionCts() instead applies
        // the *expected* attenuation over SimControls::avDistField()
        // -- see its own comment.
        const auto contSpecExtinct = ext->applyExtinctionCts(contSpec);
        for (std::size_t i = 0; i < specExtinct_.size(); ++i) { specExtinct_[i] += contSpecExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- contSpecExtinct has size specExtinct_.size() (wl_.size()) by Extinct::applyExtinctionCts()'s own contract, matching specExtinct_'s size set in computeSpec()
    }

    // Add this same combined contribution's own nebular-reprocessed
    // share, if a nebular emission grid was requested -- see
    // addContinuousNebSpec()'s own comment.
    if (neb != nullptr) { addContinuousNebSpec(ext, neb, contSpec); }
}

// See Galaxy.hpp's own header comment for this method's exact
// contract. sc.fehDist().expectationValue() stands in for the full
// [Fe/H] distribution getGalaxy() itself can't take directly -- see
// addContinuousSpec()'s own header comment for why.
void core::Galaxy::addContinuousNebSpec(const extinct::Extinct* ext, const nebular::Nebular* neb,
    const std::vector<double>& contSpec)
{
    const auto& sc = controls_.get();
    auto [nebContSpec, nebContLineLum] = neb->getGalaxy(contSpec, sc.fehDist().expectationValue());
    for (std::size_t i = 0; i < specNeb_.size(); ++i) { specNeb_[i] += nebContSpec[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- nebContSpec has size wl().size() by Nebular::getGalaxy()'s own contract, matching specNeb_'s size set in computeSpec()
    for (std::size_t i = 0; i < lineLum_.size(); ++i) { lineLum_[i] += nebContLineLum[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- nebContLineLum has size neb->lineWl().size() by Nebular::getGalaxy()'s own contract, matching lineLum_'s size set in computeSpec()
    if (ext == nullptr) { return; }

    const auto nebContSpecExtinct = ext->applyExtinctionCts(nebContSpec);
    for (std::size_t i = 0; i < specNebExtinct_.size(); ++i) { specNebExtinct_[i] += nebContSpecExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- nebContSpecExtinct has size specNebExtinct_.size() (wl_.size()) by Extinct::applyExtinctionCts()'s own contract, matching specNebExtinct_'s size set in computeSpec()
    const auto nebContLineLumExtinct = ext->applyExtinctionCtsLines(nebContLineLum);
    for (std::size_t i = 0; i < lineLumExtinct_.size(); ++i) { lineLumExtinct_[i] += nebContLineLumExtinct[i]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- nebContLineLumExtinct has size lineLumExtinct_.size() (neb->lineWl().size()) by Extinct::applyExtinctionCtsLines()'s own contract, matching lineLumExtinct_'s size set in addClusterSpecNeb()
}

// Update the galaxy's photometry (and, if an extinction curve and/or
// nebular emission grid was requested, extincted and/or nebular
// photometry) from the current spec()/specExtinct()/specNeb()/
// specNebExtinct() -- identical to Cluster::computePhot(), just
// reading this Galaxy's own lazy getters (not the raw members
// directly, so this forces a current summed spectrum first if needed,
// regardless of call order)
void core::Galaxy::computePhot()
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
    else if (sc.fCluster() < 1.0 && sc.fracStochMass() < 1.0)
    {
        // Lbol was requested and there is a continuous-population
        // share to account for, but computeSpec() hasn't run since the
        // last advance() to compute it as a byproduct -- e.g. a caller
        // asked for lbol() without ever asking for spec(). Get it via
        // the standalone path instead, which doesn't pay for a full
        // spectrum it wasn't asked for. Skipped, like addContinuousSpec()'s
        // own identical guard, whenever fracStochMass() == 1 -- see its
        // own comment.
        computeLbolCts();
        lbol_ += lbolCts_;
    }

    // Add every currently-alive field star's own contribution -- 10^logL,
    // read directly off getFieldStarProps() -- independent of
    // lbolCtsCurrent_/lbolCts_ (see this method's own header comment
    // for why).
    for (const auto& props : getFieldStarProps())
    {
        if (!props.has_value()) { continue; } // outside the tracks' own mass range: no luminosity, as in Cluster::computeLbol()
        const double logL = (*props)[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
        lbol_ += std::pow(10.0, logL);
    }
}

auto core::Galaxy::lbolCtsIntegrand(
    const double age,
    const pdfs::PDF& imf,
    const double feh
) const -> std::vector<double>
{
    const auto& sc = controls_.get();
    const double logAge = std::max(std::log10(age), sc.tracks()->logTMin());
    // Called many times per integral: with a fixed [Fe/H], use the
    // slice SimControls has precomputed, rather than having
    // tracks() slice the tracks lazily on every call
    const auto isochrone = sc.constFeH() ?
        sc.tracks2D()->getIsochrone(logAge) : sc.tracks()->getIsochrone(logAge, feh);

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
        const double b = std::min(sc.minStochMass(), seg->xMax());
        if (a >= b) { continue; } // empty intersection with [imf.getMin(), minStochMass()]

        const auto segResult = integrator.integrate(a, b, *seg);
        lbolRaw += segResult[0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- segResult is a std::array<double, 1>, so index 0 is always valid
    }
    return { lbolRaw };
}

void core::Galaxy::computeLbolCts()
{
    const auto& sc = controls_.get();
    const auto& sfrPdf = sfr();
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
    const pdfs::PDFReflect sfrAge(sfrPdf, 0.5 * curTime_);
    const double ageMin = std::min(1e4, 1e-3 * curTime_);
    const bool logAge = curTime_ > ageMin;

    const bool singleFeh = (fehDist.getMin() == fehDist.getMax());
    const double absTol = sc.intAbsTol() * sfrPdf.integral(0.0, curTime_);

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
        // Integrate over [Fe/H] with an outer PDFIntegrator weighted by
        // fehDist -- see Specsyn::specCtsHelper()'s own identical
        // treatment
        auto fehIntegrand = [&](const double feh) -> std::vector<double>
        { return integrator.integrate(ageMin, curTime_, this, imf, feh); };
        const utils::PDFIntegrator<decltype(fehIntegrand), utils::GKOrder::GK15> fehIntegrator(
            fehDist, fehIntegrand, 1, false, sc.intMaxIter(), absTol, sc.intRelTol());
        const auto result = fehIntegrator.integrate(fehDist.getMin(), fehDist.getMax());
        lbolRaw = result[0] / fehDist.integral(fehDist.getMin(), fehDist.getMax()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has exactly 1 element, by construction (nInt == 1)
    }

    // imf is normalized by number, so lbolRaw is per star formed;
    // convert to per unit non-stochastic mass formed -- see
    // SimControls::nonStochIMFMass()'s own comment
    const double massInt = sc.nonStochIMFMass();
    lbolCts_ = massInt > 0.0 ?
        lbolRaw * (1.0 - fCluster) * (1.0 - sc.fracStochMass()) / massInt : 0.0;
    lbolCtsCurrent_ = true;
}

// Per-unit-stellar-mass instantaneous yield rate of the purely
// continuous population, at a given age and [Fe/H] -- see this
// method's own header comment
auto core::Galaxy::yieldsIntegrand(const double t, const double feh) const -> std::vector<double>
{
    const auto& sc = controls_.get();
    const auto yields = sc.yields();
    const bool decomposed = sc.yieldsChannelDecomposed();

    const std::size_t n = decomposed
        ? yields->yieldChannels().size() * yields->isotopes().size()
        : yields->isotopes().size();
    std::vector<double> result(n, 0.0);

    // See lbolCtsIntegrand()'s own identical choice of tracks
    const auto massDeriv = sc.constFeH() ?
        sc.tracks2D()->massAndDerivFromLifetime(std::log10(t)) :
        sc.tracks()->massAndDerivFromLifetime(std::log10(t), feh);
    for (const auto& [m, dmDlogT] : massDeriv)
    {
        if (m > sc.minStochMass()) { continue; } // stochastically-sampled, handled separately

        const double dmDt = dmDlogT / (t * std::numbers::ln10); // chain rule
        const std::vector<double> y = decomposed
            ? yields->yield(m, feh).second
            : yields->yieldSum(m, feh);
        const double weight = std::abs(dmDt) * sc.imf()(m) / sc.imf().expectationValue();
        for (std::size_t k = 0; k < result.size(); ++k)
        {
            result[k] += y[k] * weight; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- y and result are both sized n by construction, and k is bounded by result.size()
        }
    }

    return result;
}

// The continuous population's own instantaneous per-isotope yield
// rate, integrated over the star formation history -- see this
// method's own header comment
auto core::Galaxy::yieldsRate(const double t, const double feh) const -> std::vector<double>
{
    const auto& sc = controls_.get();
    const auto yields = sc.yields();
    if (yields == nullptr) { return {}; }

    const std::size_t n = sc.yieldsChannelDecomposed()
        ? yields->yieldChannels().size() * yields->isotopes().size()
        : yields->isotopes().size();

    // See computeLbolCts()'s own comment for why this reflects sfr()
    // about t / 2, so this dimension's own coordinate becomes age
    // directly
    const pdfs::PDFReflect sfrAge(sfr(), 0.5 * t);
    const double absTol = sc.intAbsTol() * 1e-6 * sfr().integral(0.0, t);

    using IntegrandFn = std::vector<double> (Galaxy::*)(double, double) const;
    const utils::PDFIntegrator<IntegrandFn, utils::GKOrder::GK15> integrator(
        sfrAge, static_cast<IntegrandFn>(&Galaxy::yieldsIntegrand),
        n, false, sc.intMaxIter(), absTol, sc.intRelTol());

    // This is the instantaneous rate of mass return at time t --
    // decay it forward to curTime_ (unless noDecay() is true) so that
    // what computeYields() integrates over t is each moment's own
    // present-day (curTime_), rather than as-produced, contribution.
    auto result = integrator.integrate(0.0, t, this, feh);
    if (!sc.noDecay())
    {
        yields->applyDecay(curTime_ - t, result, sc.yieldsChannelDecomposed());
    }
    return result;
}

// The continuous population's own instantaneous per-isotope yield
// rate, averaged over [Fe/H] -- see this method's own header comment
auto core::Galaxy::yieldsRate(const double t) const -> std::vector<double>
{
    const auto& sc = controls_.get();
    const auto& fehDist = sc.fehDist();

    if (fehDist.getMin() == fehDist.getMax())
    {
        return yieldsRate(t, fehDist.getMin());
    }

    // Multi-feh: integrate yieldsRate(t, feh) over [Fe/H] with a
    // PDFIntegrator weighted by fehDist -- see computeLbolCts()'s own
    // identical treatment. The number of components is only known once
    // yieldsRate() has run, so it is taken from yields() directly,
    // exactly as yieldsRate(t, feh) itself sizes its own result.
    const auto yields = sc.yields();
    if (yields == nullptr) { return {}; }
    const std::size_t n = sc.yieldsChannelDecomposed()
        ? yields->yieldChannels().size() * yields->isotopes().size()
        : yields->isotopes().size();
    const double absTol = sc.intAbsTol() * 1e-6 * sfr().integral(0.0, t);
    auto fehIntegrand = [this, t](const double feh) -> std::vector<double>
    { return yieldsRate(t, feh); };
    const utils::PDFIntegrator<decltype(fehIntegrand), utils::GKOrder::GK15> fehIntegrator(
        fehDist, fehIntegrand, n, false, sc.intMaxIter(), absTol, sc.intRelTol());
    auto result = fehIntegrator.integrate(fehDist.getMin(), fehDist.getMax());
    const double fehNorm = fehDist.integral(fehDist.getMin(), fehDist.getMax());
    for (double& r : result) { r /= fehNorm; }
    return result;
}

// Update yields_/fieldYields_ from the stars that died since
// lastYieldTime_ -- see this method's own header comment for the
// three populations summed (clustered, individually-tracked field
// stars, and the purely continuous population), and Cluster::
// computeYields()'s own comment for the same null-guard/accumulation
// pattern this mirrors one level up
void core::Galaxy::computeYields()
{
    const auto& sc = controls_.get();
    const auto yields = sc.yields();
    if (yields == nullptr) { return; }

    // yields_ itself is rebuilt from scratch every call (like lbol_,
    // unlike fieldYields_) because clusters_/disruptedClusters_ each
    // keep growing their own cumulative yields() total on their own,
    // so it must be re-summed here rather than accumulated onto
    yields_.assign(yields_.size(), 0.0);
    for (auto& cluster : clusters_)
    {
        const auto& clusterYields = cluster.yields();
        for (std::size_t k = 0; k < clusterYields.size(); ++k)
        {
            yields_[k] += clusterYields[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- yields_ and clusterYields are both sized identically (isotopes, times channels if decomposed) by construction
        }
    }
    for (auto& cluster : disruptedClusters_)
    {
        const auto& clusterYields = cluster.yields();
        for (std::size_t k = 0; k < clusterYields.size(); ++k)
        {
            yields_[k] += clusterYields[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- see above
        }
    }

    // Age fieldYields_'s own already-accumulated total forward by the
    // time elapsed since it was last updated, before adding in this
    // step's new contributions below (each of which is added using its
    // own exact dtDecay for the time since it was produced, since it
    // is only now being added) -- mirrors Cluster::computeYields()'s
    // own identical aging step for yields_, see its own comment for
    // why this incremental approach is exact (not an approximation),
    // by the decay operator's own compositional (semigroup) property:
    // applying decay for dt1 then dt2 gives the same result as
    // applying it once for dt1 + dt2, for whatever abundances are
    // present at the start of each step -- no need to separately track
    // when each contribution was originally produced.
    const bool decomposed = sc.yieldsChannelDecomposed();
    if (!sc.noDecay())
    {
        yields->applyDecay(curTime_ - lastYieldTime_, fieldYields_, decomposed);
    }

    // Individually-tracked field stars that died during this step
    // alone (deadFieldStars_ only ever holds those, mirroring
    // Cluster::mDead_'s own per-step convention) -- accumulated onto
    // fieldYields_ rather than recomputed, since earlier steps' own
    // dead field stars are no longer available to re-sum. dtDecay is
    // the time elapsed since each field star actually died
    // (curTime_ - deathTime_) if noDecay() is false, or 0 (no decay
    // applied at all) if it is true -- mirroring
    // Cluster::computeYields()'s own identical convention for its
    // stochastic population.
    for (const auto& fieldStar : deadFieldStars_)
    {
        // A star above the tracks' own mass range has a death time of
        // -infinity (see fieldStarLifetime()'s own comment), meaning it
        // was already dead when it formed: its yield has been decaying
        // since formTime_
        const double timeDied = std::isfinite(fieldStar.deathTime_) ?
            fieldStar.deathTime_ : fieldStar.formTime_;
        const double dtDecay = sc.noDecay() ? 0.0 : curTime_ - timeDied;
        const auto contribution = decomposed
            ? yields->yield(fieldStar.mass_, fieldStar.feh_, dtDecay).second
            : yields->yieldSum(fieldStar.mass_, fieldStar.feh_, dtDecay);
        for (std::size_t k = 0; k < contribution.size(); ++k)
        {
            fieldYields_[k] += contribution[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- fieldYields_ and contribution are both sized identically by construction
        }
    }

    // Purely continuous (non-clustered, below minStochMass()) share:
    // integrate yieldsRate(t)'s own instantaneous rate directly over
    // real time from lastYieldTime_ to curTime_ -- skipped whenever
    // there is no such population at all, either because every
    // non-clustered star is stochastically sampled (minStochMass() ==
    // 0) or because there is no non-clustered population to begin
    // with (fCluster() == 1)
    if (sc.minStochMass() > 0.0 && sc.fCluster() < 1.0)
    {
        using YieldsRateFn = std::vector<double> (Galaxy::*)(double) const;
        const utils::GKIntegrator<YieldsRateFn, utils::GKOrder::GK15> integrator(
            static_cast<YieldsRateFn>(&Galaxy::yieldsRate), fieldYields_.size(),
            sc.intMaxIter(), sc.intAbsTol(), sc.intRelTol());
        const auto result = integrator.integrate(lastYieldTime_, curTime_, this);
        for (std::size_t k = 0; k < result.size(); ++k)
        {
            fieldYields_[k] += result[k] * (1.0 - sc.fCluster()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- fieldYields_ and result are both sized identically by construction
        }
    }

    for (std::size_t k = 0; k < yields_.size(); ++k)
    {
        yields_[k] += fieldYields_[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- yields_ and fieldYields_ are both sized identically by construction
    }
}

auto core::Galaxy::getFieldStarProps() const -> std::vector<std::optional<specsyn::Specsyn::StarData>>
{
    const auto& sc = controls_.get();
    const std::size_t n = fieldStars_.size();
    std::vector<std::optional<specsyn::Specsyn::StarData>> props(n);
    const double logTMin = sc.tracks()->logTMin();
    const double mMin = sc.tracks()->mMin();
    const double mMax = sc.tracks()->mMax();
    // A star outside the tracks' own mass range has no properties to
    // look up (see fieldStarLifetime()'s own comment); its entry is
    // left empty
    const auto inRange = [mMin, mMax](const FieldStar& fs) -> bool
    { return fs.mass_ >= mMin && fs.mass_ <= mMax; };

    if (sc.constFeH())
    {
        const auto tracks2D = sc.tracks2D();
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& fs = fieldStars_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < n == fieldStars_.size() by construction
            if (!inRange(fs)) { continue; }
            const double logT = std::max(std::log10(curTime_ - fs.formTime_), logTMin);
            props[i] = tracks2D->getStar(fs.mass_, logT); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- props has size n by construction, and i is bounded by n
        }
        return props;
    }

    // Non-degenerate [Fe/H]: evaluate each star at its own feh_, via
    // tracks()' own lazily-evaluated slices -- see this method's own
    // header comment. A star whose feh_ lies outside the tracks' own
    // [Fe/H] grid (possible only if SimControls::setTracks() replaced
    // the tracks after it was drawn) is skipped like one outside their
    // mass range
    const auto tracks3D = sc.tracks();
    const double fehLo = tracks3D->feH().front();
    const double fehHi = tracks3D->feH().back();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& fs = fieldStars_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < n == fieldStars_.size() by construction
        if (!inRange(fs) || fs.feh_ < fehLo || fs.feh_ > fehHi) { continue; }
        const double logT = std::max(std::log10(curTime_ - fs.formTime_), logTMin);
        props[i] = tracks3D->getStar(fs.mass_, logT, fs.feh_); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- props has size n by construction, and i is bounded by n
    }
    return props;
}
