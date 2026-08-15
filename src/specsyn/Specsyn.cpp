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
#include "../pdfs/PDFReflect.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/Constants.hpp"
#include "../utils/PDFIntegratorND.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <mdspan> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLibWR.hpp's own <mdspan> include
#include <memory>
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

auto specsyn::Specsyn::continuousSpecIntegrand(
    const std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>> points, // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
    const bool computeLbol,
    const double feh
) const -> std::vector<double>
{
    const std::size_t npts = points.extent(0);
    const std::size_t nPerPoint = wl_.size() + (computeLbol ? 1 : 0);

    // Evaluate each point's own spectrum -- lambda *
    // dL/dlambda (via specWl()), matching the other specCts()
    // overload's own convention -- against its own isochrone, leaving
    // a point's row at zero (Lbol element included, if present) if its
    // own mass falls in none of that isochrone's segments (a star
    // already dead at this age); if computeLbol, also evaluates that
    // same (mass, segment) directly to read off the star's own
    // bolometric luminosity, storing it as the row's final element
    // (see this function's own header comment for both)
    std::vector<double> resultFlat(npts * nPerPoint, 0.0);
    const std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on the <mdspan> include above
        resultView(resultFlat.data(), npts, nPerPoint);

    // Single-slot isochrone cache, keyed on age alone -- feh is fixed
    // for this whole call (see this function's own header comment) --
    // see this function's own header comment for why a single slot,
    // rather than a persistent map keyed on every distinct age a batch
    // touches, is what keeps memory bounded regardless of how many
    // distinct ages a batch spans.
    double cachedAge = std::numeric_limits<double>::quiet_NaN();
    std::unique_ptr<Isochrone> cachedIsochrone;

    for (std::size_t i = 0; i < npts; ++i)
    {
        const double age = points[i, 0]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < npts == points.extent(0) by this loop's own bound
        const double mass = points[i, 1]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

        if (cachedIsochrone == nullptr || age != cachedAge)
        {
            const double logAge = std::max(std::log10(age), controls_.tracks().logTMin());
            cachedIsochrone = std::make_unique<Isochrone>(controls_.tracks().getIsochrone(logAge, feh));
            cachedAge = age;
        }

        const Segment* seg = nullptr;
        for (const auto& s : *cachedIsochrone)
        {
            if (mass >= s->xMin() && mass <= s->xMax()) { seg = s.get(); break; }
        }
        if (seg == nullptr) { continue; } // dead star: leave this row's zeros as-is

        const auto starSpecWl = specWlForIntegration(mass, *seg, feh);
        for (std::size_t k = 0; k < wl_.size(); ++k)
        {
            resultView[i, k] = starSpecWl[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- starSpecWl has size wl_.size() by specWl()'s own contract, matching k's own loop bound
        }
        if (computeLbol)
        {
            const auto props = (*seg)(mass);
            const double logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
            // erg/s, not Lsun, despite Cluster::lbol()'s own Lsun
            // convention: specCtsHelper()'s own absolute tolerance is
            // in erg/s (like the spectral elements), so a Lbol value
            // of order unity (Lsun) would look converged to that
            // tolerance immediately, regardless of its actual
            // accuracy -- specCtsHelper() itself divides this back
            // down to Lsun once the integral is done.
            resultView[i, wl_.size()] = std::pow(10.0, logL) * utils::Lsun;
        }
    }
    return resultFlat;
}

auto specsyn::Specsyn::specCtsHelper(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster,
    const bool computeLbol
) const -> std::vector<double>
{
    // Reflect sfr about half the current time, so this dimension's
    // own coordinate -- what continuousSpecIntegrand() reads as
    // points[i, 0] -- becomes stellar age (curTime - time)
    // directly, rather than time itself (see PDFReflect::reflect()'s
    // own comment: pivot = curTime / 2 makes reflect(x) = curTime - x).
    // The luminosity this integral is dominated by -- young, hot, blue
    // stars -- is concentrated in a narrow sliver of *age* near 0
    // (equivalently, *time* near curTime); on a linear time axis, that
    // sliver's width relative to the full [0, curTime] domain shrinks
    // as curTime grows, needing on the order of curTime / (feature
    // width) quadrature subdivisions to resolve at large curTime --
    // catastrophically slow for pAdaptive's fixed-degree grid.
    // Log-transforming age instead spreads that near-zero feature out
    // relative to the quadrature's own sample spacing, the same way
    // log-transforming mass helps resolve a sharply peaked mass
    // function. sfr's own support can't be reflected about directly
    // (see PDFReflect's own two-argument constructor comment): sfr's
    // support is deliberately far larger than any realistic curTime
    // (see io::SimControls's own buildConstantSFR()), so reflecting
    // about its own midpoint would leave the age coordinate nowhere
    // near 0 -- an explicit pivot of curTime / 2 is required instead.
    //
    // Floored at ageMin = min(1e4 yr, 1e-3 * curTime) -- not at
    // tracks().logTMin(), which reports an extreme sentinel
    // (std::numeric_limits<float>::lowest(), not a genuine log10(yr)
    // floor) for every track set actually in use, real or test alike,
    // making it useless as a positive lower bound here. 1e4 yr is a
    // physical floor, not a numerical one: at ages that young, "the
    // track" isn't really a well-defined single curve at all -- a
    // star's properties still depend on its own detailed accretion
    // history during formation, not just its age and mass -- so there
    // is nothing more meaningful to resolve below it regardless of how
    // finely the quadrature samples that end of the domain. The
    // 1e-3 * curTime term instead guards the case where curTime itself
    // is only a few times 1e4 yr (e.g. a pathologically early output
    // time), so the floor never approaches curTime and leave nothing
    // for the log-transformed dimension to span. If curTime doesn't
    // exceed ageMin (only possible if curTime <= 0, since ageMin is
    // always < curTime whenever curTime > 0), the age dimension falls
    // back to a plain linear transform over [0, curTime], exactly as
    // before.
    const pdfs::PDFReflect sfrAge(sfr, 0.5 * curTime);
    const double ageMin = std::min(1e4, 1e-3 * curTime);
    const bool logAge = curTime > ageMin;

    const bool singleFeh = (fehDist.getMin() == fehDist.getMax());
    const double absTol = intAbsTol() * utils::Lsun * sfr.integral(0.0, curTime);
    const auto nInt = static_cast<unsigned>(wl_.size()) + (computeLbol ? 1U : 0U);

    using IntegrandFn = std::vector<double> (Specsyn::*)(
        std::mdspan<double, std::extents<std::size_t, std::dynamic_extent, 2>>, bool, double) const;
    const utils::PDFIntegratorND<IntegrandFn, 2, true, utils::CubatureMethod::hAdaptive> integrator(
        std::array<std::reference_wrapper<const pdfs::PDF>, 2>{ std::cref(sfrAge), std::cref(imf) },
        static_cast<IntegrandFn>(&Specsyn::continuousSpecIntegrand),
        nInt,
        std::array<bool, 2>{ logAge, true },
        intMaxIter(), absTol, intRelTol());

    std::vector<double> result;
    if (singleFeh)
    {
        result = integrator.integrate(
            std::array<double, 2>{ ageMin, imf.getMin() },
            std::array<double, 2>{ curTime, imf.getMax() },
            *this, computeLbol, fehDist.getMin());
    }
    else
    {
        // Integrate over [Fe/H] by running the same 2D (age, mass)
        // integral once at every grid point the tracks are actually
        // defined at, then interpolating and integrating those
        // discrete results over [Fe/H] -- see this function's own
        // header comment for why, in place of a joint 3D (feh, age,
        // mass) cubature integral.
        //
        // Deliberately uses the tracks' own full (padded) grid, not
        // just the points inside [fehDist.getMin(), fehDist.getMax()]:
        // those padding points carry real information about how the
        // spectrum varies with [Fe/H] near the domain edges (the slope
        // Interpolator1D's spline needs to shape correctly right up to
        // the true edges), not just filler. The spectral synthesis
        // libraries themselves are guaranteed wide enough in [Fe/H] to
        // cover this whole padded grid -- see
        // io::SimControls::readSpectra()'s own comment for why -- so
        // evaluating here never runs outside their own domain.
        const auto& fehGrid = controls_.tracks().feH();
        const std::size_t nFeh = fehGrid.size();
        const auto nQty = static_cast<std::size_t>(nInt);

        // rawResults[f] holds this call's nInt raw (lambda * dL/dlambda,
        // plus optional Lbol in erg/s) integrator outputs at fehGrid[f];
        // fehWeight[f] is fehDist evaluated at that same grid point (0
        // for any padding grid points outside fehDist's own support --
        // see pdfs::PDF::operator()'s own comment).
        std::vector<std::vector<double>> rawResults(nFeh);
        std::vector<double> fehWeight(nFeh);
        for (std::size_t f = 0; f < nFeh; ++f)
        {
            rawResults[f] = integrator.integrate(
                std::array<double, 2>{ ageMin, imf.getMin() },
                std::array<double, 2>{ curTime, imf.getMax() },
                *this, computeLbol, fehGrid[f]);
            fehWeight[f] = fehDist(fehGrid[f]);
        }

        // Normalizing denominator: the integral of fehDist alone over
        // its own domain (fehDist need not itself integrate to exactly
        // 1 -- e.g. if given as an unnormalized weight function).
        const interp::Interpolator1D<1> weightInterp(fehGrid, fehWeight);
        const double weightIntegral = weightInterp.integ(fehDist.getMin(), fehDist.getMax());

        result.assign(nQty, 0.0);
        std::vector<double> quantityAtFeh(nFeh);
        for (std::size_t k = 0; k < nQty; ++k)
        {
            for (std::size_t f = 0; f < nFeh; ++f)
            {
                quantityAtFeh[f] = rawResults[f][k] * fehWeight[f]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- rawResults[f] has size nInt == nQty by the integrator's own contract, and k is bounded by nQty
            }
            const interp::Interpolator1D<1> quantityInterp(fehGrid, quantityAtFeh);
            result[k] = quantityInterp.integ(fehDist.getMin(), fehDist.getMax()) / weightIntegral; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result has size nQty by the assign() just above, and k is bounded by nQty
        }
    }

    // Undo continuousSpecIntegrand()'s own lambda * dL/dlambda
    // weighting over the spectral elements alone (the trailing Lbol
    // element, if present, is a genuine luminosity already, not a
    // lambda-weighted flux, so it's excluded from this division), and
    // scale everything -- spectrum and Lbol alike -- by (1 - fCluster)
    // for the continuously-treated share of the population.
    for (std::size_t i = 0; i < wl_.size(); ++i)
    {
        result[i] = result[i] * (1.0 - fCluster) / wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl_.size() <= result.size() by construction (result.size() == wl_.size() + computeLbol), and i is bounded by wl_.size()
    }
    if (computeLbol)
    {
        // continuousSpecIntegrand() reports Lbol in erg/s, not Lsun --
        // see its own comment for why -- so convert back to Lsun here,
        // matching Cluster::lbol()'s own units, alongside the same
        // (1 - fCluster) scaling every other element gets.
        result[wl_.size()] *= (1.0 - fCluster) / utils::Lsun; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result.size() == wl_.size() + 1 when computeLbol is true, by construction
    }
    return result;
}

auto specsyn::Specsyn::specCts(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster
) const -> std::vector<double>
{
    return specCtsHelper(sfr, imf, fehDist, curTime, fCluster, false);
}

auto specsyn::Specsyn::specAndLbolCts(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster
) const -> std::pair<std::vector<double>, double>
{
    auto spec = specCtsHelper(sfr, imf, fehDist, curTime, fCluster, true);
    const double lBol = spec.back();
    spec.pop_back();
    return { std::move(spec), lBol };
}
