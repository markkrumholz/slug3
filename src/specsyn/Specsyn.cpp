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
#include "../tracks/Tracks3D.hpp"
#include "../utils/Constants.hpp"
#include "../utils/PDFIntegratorND.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <mdspan> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLibWR.hpp's own <mdspan> include
#include <set>
#include <utility>
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
    //
    // Uses cubature's p-adaptive routine (CubatureMethod::pAdaptive),
    // in linear (not log) mass space, rather than PDFIntegrator's own
    // default of h-adaptive: benchmarked head to head against
    // h-adaptive (linear and log) and p-adaptive-in-log-space on a
    // full-scale, 100-output-time non-stochastic cluster run (real
    // MIST tracks and spectral libraries, so the comparison reflects
    // this integrand's actual cost -- a spectral flux that is smooth
    // in mass but spans many orders of magnitude in value), p-adaptive
    // in linear space was the clear winner: roughly 2x faster than
    // h-adaptive at identical accuracy (integrated luminosity agreed
    // to within ~0.1% across every configuration and output time
    // checked), while log-transforming the mass coordinate under
    // p-adaptive was instead slower than the old h-adaptive default --
    // this integrand isn't sharply peaked enough near either edge of
    // its mass domain for the log transform to pay for the extra
    // nonlinearity it introduces.
    using SpecSegFn = std::vector<double> (Specsyn::*)(double, const Segment&, double) const;
    const utils::PDFIntegratorND<SpecSegFn, 1, false, utils::CubatureMethod::pAdaptive> integrator(
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

template <std::size_t Ndim>
auto specsyn::Specsyn::continuousSpecIntegrand(
    const std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, Ndim>> points, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
    std::map<std::pair<double, double>, Isochrone>& cache,
    const double curTime
) const -> std::vector<double>
{
    static_assert(Ndim == 2 || Ndim == 3, "continuousSpecIntegrand only supports Ndim == 2 or 3");

    const std::size_t npts = points.extent(0);

    // Step 1: find the unique (age, feh) pairs represented in points --
    // age = curTime - (each point's own time coordinate, column 0);
    // feh is column 1 if Ndim == 3, or -- since Ndim == 2 means
    // specCts()'s own fehDist was degenerate -- controls_.fehDist()'s
    // single value otherwise.
    std::set<std::pair<double, double>> uniqueAgeFeh;
    for (std::size_t i = 0; i < npts; ++i)
    {
        const double age = curTime - points[i, 0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < npts == points.extent(0) by the loop bound
        double feh = 0.0;
        if constexpr (Ndim == 3) { feh = points[i, 1]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        else { feh = controls_.fehDist().getMin(); }
        uniqueAgeFeh.emplace(age, feh);
    }

    // Step 2: build (or reuse, from cache) the isochrone at every one
    // of those (age, feh) pairs -- log(age) is floored at
    // controls_.tracks().logTMin(), mirroring Cluster::advance()'s own
    // identical floor, so a point landing at (or numerically near)
    // age == 0 doesn't take log(0)
    for (const auto& ageFeh : uniqueAgeFeh)
    {
        if (cache.contains(ageFeh)) { continue; }
        const auto& [age, feh] = ageFeh;
        const double logAge = std::max(std::log10(age), controls_.tracks().logTMin());
        cache.emplace(ageFeh, controls_.tracks().getIsochrone(logAge, feh));
    }

    // Step 3: evaluate each point's own spectrum -- lambda *
    // dL/dlambda (via specWl()), matching the other specCts()
    // overload's own convention -- against its own cached isochrone,
    // leaving a point's row at zero if its own mass falls in none of
    // that isochrone's segments (a star already dead at this age)
    std::vector<double> resultFlat(npts * wl_.size(), 0.0);
    const std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
        resultView(resultFlat.data(), npts, wl_.size());
    for (std::size_t i = 0; i < npts; ++i)
    {
        const double age = curTime - points[i, 0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < npts == points.extent(0) by the loop bound
        double feh = 0.0;
        double mass = 0.0;
        if constexpr (Ndim == 3) { feh = points[i, 1]; mass = points[i, 2]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        else { feh = controls_.fehDist().getMin(); mass = points[i, 1]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

        const auto& isochrone = cache.at({ age, feh });
        const Segment* seg = nullptr;
        for (const auto& s : isochrone)
        {
            if (mass >= s->xMin() && mass <= s->xMax()) { seg = s.get(); break; }
        }
        if (seg == nullptr) { continue; } // dead star: leave this row's zeros as-is

        const auto starSpecWl = specWl(mass, *seg, feh);
        for (std::size_t k = 0; k < wl_.size(); ++k)
        {
            resultView[i, k] = starSpecWl[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- starSpecWl has size wl_.size() by specWl()'s own contract, matching k's own loop bound
        }
    }
    return resultFlat;
}

// Explicit instantiation for the two dimensionalities specCts() ever
// uses this with (2: time+mass, single feh; 3: time+feh+mass) --
// keeps this template's implementation in this .cpp file, as with
// every other template class/method in src/specsyn.
template auto specsyn::Specsyn::continuousSpecIntegrand<2>(
    std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>,
    std::map<std::pair<double, double>, Isochrone>&, double) const -> std::vector<double>;
template auto specsyn::Specsyn::continuousSpecIntegrand<3>(
    std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 3>>,
    std::map<std::pair<double, double>, Isochrone>&, double) const -> std::vector<double>;

auto specsyn::Specsyn::specCts(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster
) const -> std::vector<double>
{
    // See this overload's own header comment for why the time
    // dimension is never log-transformed (its lower bound is always
    // exactly 0), while the mass dimension always is.
    const bool singleFeh = (fehDist.getMin() == fehDist.getMax());
    const double absTol = intAbsTol() * utils::Lsun * sfr.integral(0.0, curTime);

    std::vector<double> result;
    if (singleFeh)
    {
        using IntegrandFn = std::vector<double> (Specsyn::*)(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>,
            std::map<std::pair<double, double>, Isochrone>&, double) const;
        const utils::PDFIntegratorND<IntegrandFn, 2, true, utils::CubatureMethod::pAdaptive> integrator(
            std::array<std::reference_wrapper<const pdfs::PDF>, 2>{ std::cref(sfr), std::cref(imf) },
            static_cast<IntegrandFn>(&Specsyn::continuousSpecIntegrand<2>),
            static_cast<unsigned>(wl_.size()),
            std::array<bool, 2>{ false, true },
            intMaxIter(), absTol, intRelTol());
        result = integrator.integrate(
            std::array<double, 2>{ 0.0, imf.getMin() },
            std::array<double, 2>{ curTime, imf.getMax() },
            *this, isochroneCache_, curTime);
    }
    else
    {
        using IntegrandFn = std::vector<double> (Specsyn::*)(
            std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 3>>,
            std::map<std::pair<double, double>, Isochrone>&, double) const;
        const utils::PDFIntegratorND<IntegrandFn, 3, true, utils::CubatureMethod::pAdaptive> integrator(
            std::array<std::reference_wrapper<const pdfs::PDF>, 3>{
                std::cref(sfr), std::cref(fehDist), std::cref(imf) },
            static_cast<IntegrandFn>(&Specsyn::continuousSpecIntegrand<3>),
            static_cast<unsigned>(wl_.size()),
            std::array<bool, 3>{ false, false, true },
            intMaxIter(), absTol, intRelTol());
        result = integrator.integrate(
            std::array<double, 3>{ 0.0, fehDist.getMin(), imf.getMin() },
            std::array<double, 3>{ curTime, fehDist.getMax(), imf.getMax() },
            *this, isochroneCache_, curTime);
    }

    // Undo continuousSpecIntegrand()'s own lambda * dL/dlambda
    // weighting, and scale by (1 - fCluster) for the continuously-
    // treated share of the population.
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = result[i] * (1.0 - fCluster) / wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl_ has the same size as result (both sized to wl_.size()), and i is bounded by result.size()
    }
    isochroneCache_.clear();
    return result;
}
