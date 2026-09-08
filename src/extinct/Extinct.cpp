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
    const double z = controls_.z();
    std::vector<double> wlObs(wl_.size());
    std::ranges::transform(wl_, wlObs.begin(),
        [z](const double wl) -> double { return wl * (1.0 + z); });
    return wlObs;
}

void extinct::Extinct::rebuildCache()
{
    if (controls_.specsyn() == nullptr)
    {
        throw std::runtime_error(
            "Extinct::rebuildCache: controls has no spectral synthesizer "
            "(SimControls::specsyn() is null), so no wavelength grid is "
            "available to interpolate the extinction curve onto");
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

    // Recompute extinctionFacCts_/extinctionFacCtsLines_ -- see their
    // own comments
    computeExtinctionFacCts();
    computeExtinctionFacCtsLines();
}

void extinct::Extinct::initExtinctLines(const interp::Interpolator1D<1>& interp)
{
    const auto* neb = controls_.nebular();
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

void extinct::Extinct::computeExtinctionFacCts()
{
    const auto& avDistField = controls_.avDistField();

    // A degenerate (single-point) or invalid (no explicit distribution
    // at all -- see this class's own comment for when that happens,
    // e.g. constructed directly against a bare default-constructed
    // SimControls, as tests/extinct/testExtinct.hpp's own tests do)
    // avDistField() has no meaningful density for utils::PDFIntegrator
    // to evaluate pointwise: PDFSegmentDelta::operator() would throw if
    // pdfs::PDF::operator() ever actually called it, but in practice
    // PDF::operator()'s own boundary check (a strict getMin() < x)
    // excludes the delta's own point exactly, so it would instead
    // silently return a density of 0 everywhere, integrating to a
    // wrong, all-zero extinctionFacCts_ -- and an invalid avDistField()
    // has NaN getMin()/getMax(), which would never satisfy any of
    // GKIntegrator::integrate()'s own convergence checks (all strict
    // less-than comparisons against NaN), risking an infinite loop.
    // Handled directly here instead, with any invalid avDistField()
    // treated as a delta at A_V = 0 -- mirroring Cluster's own
    // identical avDist().valid() ? draw() : 0.0 convention -- rather
    // than ever handing PDFIntegrator a degenerate or NaN-bounded
    // domain to integrate over.
    if (!avDistField.valid() || avDistField.getMin() == avDistField.getMax())
    {
        const double A_V = avDistField.valid() ? avDistField.getMin() : 0.0; // NOLINT(readability-identifier-naming) -- see applyExtinction()'s own identical NOLINT
        extinctionFacCts_ = extinctFac(A_V);
        return;
    }

    // General case: a genuine, non-degenerate distribution -- integrate
    // extinctFac(A_V) against avDistField() itself via PDFIntegrator,
    // exactly the \int exp[-A_V * extinct(lambda)] p(A_V) dA_V this
    // class's own comment describes, over avDistField()'s own full
    // support.
    using ExtinctFacFn = std::vector<double> (Extinct::*)(double) const;
    const utils::PDFIntegrator<ExtinctFacFn, utils::GKOrder::GK15> integrator(
        avDistField, static_cast<ExtinctFacFn>(&Extinct::extinctFac),
        wl_.size(), false, controls_.intMaxIter(), controls_.intAbsTol(), controls_.intRelTol());
    extinctionFacCts_ = integrator.integrate(avDistField.getMin(), avDistField.getMax(), this);
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

    const auto& avDistField = controls_.avDistField();

    // See computeExtinctionFacCts()'s own comment on this degenerate/
    // invalid avDistField() handling -- identical here, just for
    // extinctFacLines()/extinctionFacCtsLines_ rather than
    // extinctFac()/extinctionFacCts_
    if (!avDistField.valid() || avDistField.getMin() == avDistField.getMax())
    {
        const double A_V = avDistField.valid() ? avDistField.getMin() : 0.0; // NOLINT(readability-identifier-naming) -- see applyExtinction()'s own identical NOLINT
        extinctionFacCtsLines_ = extinctFacLines(A_V);
        return;
    }

    // General case -- see computeExtinctionFacCts()'s own comment
    using ExtinctFacLinesFn = std::vector<double> (Extinct::*)(double) const;
    const utils::PDFIntegrator<ExtinctFacLinesFn, utils::GKOrder::GK15> integrator(
        avDistField, static_cast<ExtinctFacLinesFn>(&Extinct::extinctFacLines),
        extinctLines_.size(), false, controls_.intMaxIter(), controls_.intAbsTol(), controls_.intRelTol());
    extinctionFacCtsLines_ = integrator.integrate(avDistField.getMin(), avDistField.getMax(), this);
}
