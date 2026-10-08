/**
 * @file Extinct.cpp
 * @author Mark Krumholz
 * @brief Implementation of Extinct
 * @date 2026-08-04
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * Out-of-line so this file, not Extinct.hpp, is the one that needs
 * io::SimControls's complete type -- SimControls.hpp itself includes
 * Extinct.hpp (for its own extinct_ member), so Extinct.hpp can only
 * forward-declare io::SimControls without creating a header cycle.
 * Mirrors Specsyn.cpp's identical situation/comment exactly.
 */

#include "Extinct.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimControls.hpp"
#include "../utils/GKIntegratorData.hpp"
#include "../utils/PDFIntegrator.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

auto extinct::Extinct::wlObs() const -> std::vector<double>
{
    if (wl_.empty())
    {
        throw std::runtime_error(
            "Extinct::wlObs: this Extinct has no wavelength grid -- "
            "controls_.specsyn() was null the last time rebuildCache() ran");
    }
    const double z = controls_.z();
    std::vector<double> wlObs(wl_.size());
    std::ranges::transform(wl_, wlObs.begin(),
        [z](const double wl) -> double { return wl * (1.0 + z); });
    return wlObs;
}

void extinct::Extinct::rebuildCache()
{
    Extinct tmp(*this);
    tmp.rebuildCacheImpl();
    commitFrom(tmp);
}

void extinct::Extinct::rebuildCacheImpl()
{
    if (controls_.specsyn() == nullptr)
    {
        // No wavelength grid to interpolate the curve onto -- see this
        // method's own header comment for why every cached quantity,
        // not just wl_/extinct_, has to be cleared rather than left as
        // whatever it was before (normalize() needs a non-empty wl_/
        // extinct_ to derive its own V-band scale factor, so
        // extinctLines_ can't be meaningfully normalized either)
        wl_.clear();
        extinct_.clear();
        wlOffset_ = 0;
        extinctLines_.clear();
        extinctionFacCts_.clear();
        extinctionFacCtsNeb_.clear();
        extinctionFacCtsLines_.clear();
        return;
    }
    const auto& wl = controls_.specsyn()->wl();

    // Build an interpolator for the native curve data
    const interp::Interpolator1D<1> interp(wlDat_, extinctDat_);

    // Chop wl down to the interpolator's own coverage -- wl is
    // assumed sorted ascending (a spectral wavelength grid), so the
    // kept elements are a single contiguous run; wlOffset_ records
    // how many leading elements were dropped, so applyExtinction()
    // can later line up a spectrum tabulated on this same wl without
    // having to rediscover the chop -- then interpolate the curve
    // onto what remains. wl_/extinct_ are cleared first (rather than
    // relying on them starting empty, as the old constructor-only
    // call site could) since rebuildCache() may run more than once
    // over this Extinct's lifetime.
    wl_.clear();
    extinct_.clear();
    const auto firstIt = std::ranges::find_if(wl,
        [&interp](const double w) -> bool { return w >= interp.xMin(); });
    wlOffset_ = static_cast<std::size_t>(std::distance(wl.begin(), firstIt));
    for (auto it = firstIt; it != wl.end() && *it <= interp.xMax(); ++it)
    {
        wl_.push_back(*it);
        extinct_.push_back(interp(*it));
    }

    // Interpolate the curve onto every nebular emission line's own
    // wavelength too, if a nebular emission grid was requested -- see
    // initExtinctLines()'s own comment.
    initExtinctLines(interp);

    // Normalize the curve (and, if any, the line-wavelength curve
    // above) to a V-band extinction of 1 mag
    normalize(wl_, extinct_, extinctLines_);

    // Recompute extinctionFacCts_/extinctionFacCtsNeb_/
    // extinctionFacCtsLines_ -- see their own comments
    computeExtinctionFacCts();
    computeExtinctionFacCtsLines();
}

void extinct::Extinct::initExtinctLines(const interp::Interpolator1D<1>& interp)
{
    const auto neb = controls_.nebular();
    if (neb == nullptr)
    {
        // No nebular emission grid was requested -- extinctLines_
        // must be left empty, not merely untouched: a prior
        // rebuildCache() call, made while controls_.nebular() was
        // still non-null, may have left it populated.
        extinctLines_.clear();
        return;
    }

    const auto& lineWl = neb->lineWl();
    extinctLines_.resize(lineWl.size());
    for (std::size_t ell = 0; ell < lineWl.size(); ++ell)
    {
        extinctLines_.at(ell) =
            (lineWl.at(ell) >= interp.xMin() && lineWl.at(ell) <= interp.xMax()) ?
            interp(lineWl.at(ell)) : 0.0;
    }
}

auto extinct::Extinct::expectedExtinctFac(const ExtinctFacFn fac, const std::size_t nInt,
    const bool nebular) const -> std::vector<double>
{
    const auto& avDistField = controls_.avDistField();

    // An invalid avDistField() (no explicit distribution at all -- see
    // this class's own comment for when that happens, e.g. constructed
    // directly against a bare default-constructed SimControls, as
    // tests/extinct/testExtinct.hpp's own tests do) is treated as a
    // delta at A_V = 0, mirroring Cluster's own avDist().valid() ?
    // draw() : 0.0 convention, so the result is all 1s whatever the
    // nebular factor is.
    if (!avDistField.valid()) { return (this->*fac)(0.0); }

    // The nebular-to-stellar extinction ratio f: exactly 1 for stellar
    // (non-nebular) light, and likewise if avNebFac() is invalid
    // (SimControls always sets it when reading an input deck, but a
    // bare default-constructed SimControls leaves it unset)
    const auto& avNebFac = controls_.avNebFac();
    const bool useNebFac = nebular && avNebFac.valid();

    // A degenerate (single-point) distribution has no meaningful density
    // for utils::PDFIntegrator to evaluate pointwise: PDFSegmentDelta::
    // operator() would throw if pdfs::PDF::operator() ever actually
    // called it, but in practice PDF::operator()'s own boundary check (a
    // strict getMin() < x) excludes the delta's own point exactly, so it
    // would instead silently return a density of 0 everywhere,
    // integrating to a wrong, all-zero result. Delta distributions are
    // therefore handled directly, by evaluating at their one point,
    // rather than ever handed to PDFIntegrator as a domain to integrate
    // over.
    const bool avIsDelta = avDistField.getMin() == avDistField.getMax();
    const bool facIsDelta = !useNebFac || avNebFac.getMin() == avNebFac.getMax();
    const double avValue = avDistField.getMin(); // only meaningful if avIsDelta
    const double facValue = useNebFac ? avNebFac.getMin() : 1.0; // only meaningful if facIsDelta
    const auto maxIter = controls_.intMaxIter();
    const auto absTol = controls_.intAbsTol();
    const auto relTol = controls_.intRelTol();

    // Both deltas: a single evaluation
    if (avIsDelta && facIsDelta) { return (this->*fac)(avValue * facValue); }

    // A_V distributed, f a delta: integrate fac(A_V f) over avDistField()
    if (facIsDelta)
    {
        const auto integrand = [this, fac, facValue](const double aV) -> std::vector<double>
        { return (this->*fac)(aV * facValue); };
        const utils::PDFIntegrator<decltype(integrand), utils::GKOrder::GK15> integrator(
            avDistField, integrand, nInt, false, maxIter, absTol, relTol);
        return integrator.integrate(avDistField.getMin(), avDistField.getMax());
    }

    // The integral of fac(A_V f) over f ~ avNebFac(), at fixed A_V -- the
    // whole answer if A_V is a delta, and otherwise the inner integral
    // of the double integral below
    const auto integrateOverNebFac = [this, fac, nInt, &avNebFac, maxIter, absTol, relTol](
        const double aV) -> std::vector<double>
    {
        const auto integrand = [this, fac, aV](const double f) -> std::vector<double>
        { return (this->*fac)(aV * f); };
        const utils::PDFIntegrator<decltype(integrand), utils::GKOrder::GK15> integrator(
            avNebFac, integrand, nInt, false, maxIter, absTol, relTol);
        return integrator.integrate(avNebFac.getMin(), avNebFac.getMax());
    };

    // A_V a delta, f distributed: a single integral over avNebFac()
    if (avIsDelta) { return integrateOverNebFac(avValue); }

    // Both distributed: the double integral, as two nested
    // PDFIntegrators -- the outer one over A_V ~ avDistField(), whose
    // integrand is itself the inner integral over f ~ avNebFac()
    const utils::PDFIntegrator<decltype(integrateOverNebFac), utils::GKOrder::GK15> integrator(
        avDistField, integrateOverNebFac, nInt, false, maxIter, absTol, relTol);
    return integrator.integrate(avDistField.getMin(), avDistField.getMax());
}

void extinct::Extinct::computeExtinctionFacCts()
{
    // Stellar light (A_V only), then nebular emission (A_V times the
    // nebular-to-stellar ratio) -- see expectedExtinctFac()
    extinctionFacCts_ = expectedExtinctFac(&Extinct::extinctFac, wl_.size(), false);
    extinctionFacCtsNeb_ = expectedExtinctFac(&Extinct::extinctFac, wl_.size(), true);
}

void extinct::Extinct::computeExtinctionFacCtsLines()
{
    // No nebular emission grid was requested, so there are no lines
    // to compute this for -- extinctionFacCtsLines_ must be left
    // empty, matching extinctLines_ itself; clear() rather than a
    // bare return, since a prior rebuildCache() call, made while
    // extinctLines_ was still non-empty, may have left it populated
    if (extinctLines_.empty())
    {
        extinctionFacCtsLines_.clear();
        return;
    }

    // Lines are always nebular emission, so integrate over the
    // nebular-to-stellar extinction ratio too -- see
    // expectedExtinctFac()
    extinctionFacCtsLines_ = expectedExtinctFac(&Extinct::extinctFacLines, extinctLines_.size(), true);
}
