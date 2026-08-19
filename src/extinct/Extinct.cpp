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
#include "../io/SimControls.hpp"
#include "../utils/GKIntegratorData.hpp"
#include "../utils/PDFIntegrator.hpp"
#include <algorithm>
#include <vector>

auto extinct::Extinct::wlObs() const -> std::vector<double>
{
    const double z = controls_.z();
    std::vector<double> wlObs(wl_.size());
    std::ranges::transform(wl_, wlObs.begin(),
        [z](const double wl) -> double { return wl * (1.0 + z); });
    return wlObs;
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
