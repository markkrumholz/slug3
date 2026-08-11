/**
 * @file Specsyn.cpp
 * @author Mark Krumholz
 * @brief Implementation of Specsyn
 * @date 2026-08-02
 * @details
 * Out-of-line so this file, not Specsyn.hpp, is the one that needs
 * io::SimControls's complete type -- SimControls.hpp itself includes
 * Specsyn.hpp (for its own specsyn_ member), so Specsyn.hpp can only
 * forward-declare io::SimControls without creating a header cycle.
 */

#include "Specsyn.hpp"
#include "../io/SimControls.hpp"
#include "../pdfs/PDF.hpp"
#include "../utils/Constants.hpp"
#include "../utils/PDFIntegrator.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

auto specsyn::Specsyn::intRelTol() const -> double { return controls_.intRelTol(); }

auto specsyn::Specsyn::intAbsTol() const -> double { return controls_.intAbsTol(); }

auto specsyn::Specsyn::intMaxIter() const -> std::size_t { return controls_.intMaxIter(); }

auto specsyn::Specsyn::wlObs() const -> std::vector<double>
{
    const double z = controls_.z();
    std::vector<double> wlObs(wl_.size());
    std::ranges::transform(wl_, wlObs.begin(),
        [z](const double wl) -> double { return wl * (1.0 + z); });
    return wlObs;
}

auto specsyn::Specsyn::specCts(
    const Isochrone& isochrone,
    const pdfs::PDF& imf,
    const double mTot,
    const double mMin,
    const double mMax,
    const double feh
) const -> std::vector<double>
{
    // Integrate lambda * dL/dlambda (via specWl) rather than
    // dL/dlambda directly, and scale intAbsTol() (specified in units
    // of dL/dlambda's own natural scale) by utils::Lsun to match --
    // see specWl()'s own doc comment for why.
    using SpecSegFn = std::vector<double> (Specsyn::*)(double, const Segment&, double) const;
    const utils::PDFIntegrator integrator(
        imf, static_cast<SpecSegFn>(&Specsyn::specWl), static_cast<unsigned>(wl_.size()),
        std::array<bool, 1>{}, intMaxIter(), intAbsTol() * utils::Lsun, intRelTol());

    std::vector<double> result(wl_.size(), 0.0);
    for (const auto& seg : isochrone)
    {
        const double a = std::max(mMin, seg->xMin());
        const double b = std::min(mMax, seg->xMax());
        if (a >= b) { continue; } // empty intersection with [mMin, mMax]

        const auto segResult = integrator.integrate(a, b, this, *seg, feh);
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] += segResult[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- segResult has the same size as result (both sized to wl_.size()), and i is bounded by result.size()
        }
    }
    // Scale by mTot and divide back out by wl_ elementwise, undoing
    // specWl()'s multiplication to recover dL/dlambda
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = result[i] * mTot / wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl_ has the same size as result (both sized to wl_.size()), and i is bounded by result.size()
    }
    return result;
}
