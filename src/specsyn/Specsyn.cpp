/**
 * @file Specsyn.cpp
 * @author Mark Krumholz
 * @brief Implementation of Specsyn
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
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
#include "../utils/Constants.hpp"
#include "../utils/GKIntegratorData.hpp"
#include "../utils/PDFIntegrator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace
{
    /**
     * @brief Integral of m p(m) dm over a mass range, for an IMF p normalized by number
     * @param imf The IMF
     * @param mMin Lower mass limit, in Msun; clamped up to imf.getMin()
     * @param mMax Upper mass limit, in Msun; clamped down to imf.getMax()
     * @return The mass per star, averaged over the whole of imf, in
     *   stars within [mMin, mMax] -- the normalization that converts
     *   an integral of a per-star quantity against imf over [mMin,
     *   mMax] into that quantity per unit mass of stars in [mMin,
     *   mMax] (see io::SimControls::nonStochIMFMass()'s own comment);
     *   0 if imf has no stars in the clamped range
     */
    auto imfMassIntegral(const pdfs::PDF& imf, const double mMin, const double mMax) -> double
    {
        // A range covering all of imf uses its whole-range integral
        // and mean directly -- see SimControls::nonStochIMFMass()'s
        // own identical special case for why (a delta-function imf)
        if (mMin <= imf.getMin() && mMax >= imf.getMax())
        {
            return imf.expectationValue() * imf.integral();
        }
        const double a = std::max(mMin, imf.getMin());
        const double b = std::min(mMax, imf.getMax());
        if (b <= a) { return 0.0; }
        return imf.expectationValue(a, b) * imf.integral(a, b);
    }
} // namespace

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

auto specsyn::Specsyn::specCtsImpl(
    const Isochrone& isochrone,
    const pdfs::PDF& imf,
    const double mTot,
    const double mMin,
    const double mMax,
    const double feh,
    const bool forIntegration,
    const bool computeLbol
) const -> std::vector<double>
{
    const std::size_t nWl = wl_.size();
    const std::size_t nQty = nWl + (computeLbol ? 1 : 0);

    // Evaluate a single star's own row -- lambda * dL/dlambda (see
    // specWl()) at each wavelength, plus -- only if computeLbol -- one
    // further element, that star's own bolometric luminosity in erg/s
    // (not Cluster::lbol()'s own Lsun -- see this function's own
    // header comment for why). Dispatches to specWlForIntegration()
    // rather than specWl() itself when forIntegration is true -- see
    // that function's own comment for why continuousSpecIntegrand()
    // needs it, in place of specWl()'s ordinary spec() call.
    const auto evalPoint = [this, forIntegration, computeLbol, nWl, nQty](
        const double m, const Segment& seg, const double fehArg) -> std::vector<double>
    {
        std::vector<double> row(nQty, 0.0);
        std::vector<double> starSpecWl;
        double logL = 0.0;
        if (forIntegration)
        {
            // Evaluate the segment once, sharing it between the
            // spectrum and (if requested) the Lbol contribution below,
            // rather than evaluating it twice.
            const auto props = seg(m);
            starSpecWl = specWlForIntegration(props, fehArg);
            if (computeLbol)
            {
                logL = props[static_cast<std::size_t>(tracks::FieldIdx::logL)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logL is one of its compile-time-known indices
            }
        }
        else
        {
            starSpecWl = specWl(m, seg, fehArg);
        }
        for (std::size_t k = 0; k < nWl; ++k)
        {
            row[k] = starSpecWl[k]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- starSpecWl has size wl_.size() == nWl by specWl()/specWlForIntegration()'s own contract, and k is bounded by nWl
        }
        if (computeLbol)
        {
            row[nWl] = std::pow(10.0, logL) * utils::Lsun; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- row has size nQty == nWl + 1 whenever computeLbol is true, so index nWl is always valid
        }
        return row;
    };
    using EvalFn = decltype(evalPoint);

    // Integrate lambda * dL/dlambda (via evalPoint) rather than
    // dL/dlambda directly, and scale intAbsTol() (specified in units
    // of dL/dlambda's own natural scale) by utils::Lsun to match --
    // see specWl()'s own doc comment for why.
    //
    // Uses PDFIntegrator (GKOrder::GK15), in linear (not log) mass
    // space, rather than the cubature-package-based PDFIntegratorND
    // this integrator used previously: benchmarked head to head on a
    // full-scale, 400-output-time non-stochastic cluster run (real
    // MIST tracks and spectral libraries), GK15 was consistently
    // ~3x faster than PDFIntegratorND's own best configuration
    // (CubatureMethod::pAdaptive) at identical accuracy, with GK31
    // essentially tied and GK61 slower (this integrand -- a spectral
    // flux that is smooth in mass -- doesn't need GK61's own extra
    // per-point accuracy enough to offset its higher per-point cost).
    const utils::PDFIntegrator<EvalFn, utils::GKOrder::GK15> integrator(
        imf, evalPoint, nQty, false, intMaxIter(), intAbsTol() * utils::Lsun, intRelTol());

    std::vector<double> result(nQty, 0.0);
    for (const auto& seg : isochrone)
    {
        const double a = std::max(mMin, seg->xMin());
        const double b = std::min(mMax, seg->xMax());
        if (a >= b) { continue; } // empty intersection with [mMin, mMax]

        const auto segResult = integrator.integrate(a, b, *seg, feh);
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] += segResult[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- segResult has the same size as result (both sized to nQty), and i is bounded by result.size()
        }
    }

    if (!forIntegration)
    {
        // Scale by mTot per unit IMF mass in [mMin, mMax] (converting
        // the per-star integral above into a total for a population of
        // mass mTot -- see imfMassIntegral()'s own comment), and divide
        // back out by wl_ elementwise, undoing specWl()'s
        // multiplication to recover dL/dlambda. Skipped when
        // forIntegration is true: continuousSpecIntegrand()'s own
        // caller (specCtsHelper()) does this exact scaling and division
        // itself, once, after its own outer age integral is complete --
        // see specCtsImpl()'s own header comment.
        const double massInt = imfMassIntegral(imf, mMin, mMax);
        const double scale = massInt > 0.0 ? mTot / massInt : 0.0;
        for (std::size_t i = 0; i < nWl; ++i)
        {
            result[i] = result[i] * scale / wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl_ has size nWl, and i is bounded by nWl
        }
    }
    return result;
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
    return specCtsImpl(isochrone, imf, mTot, mMin, mMax, feh, false, false);
}

auto specsyn::Specsyn::continuousSpecIntegrand(
    const double age,
    const pdfs::PDF& imf,
    const double mMin,
    const double mMax,
    const bool computeLbol,
    const double feh
) const -> std::vector<double>
{
    const double logAge = std::max(std::log10(age), controls_.tracks()->logTMin());
    // Called many times per integral: with a fixed [Fe/H], use the
    // slice controls_ has precomputed, rather than having tracks()
    // slice the tracks lazily on every call
    const auto isochrone = controls_.constFeH() ?
        controls_.tracks2D()->getIsochrone(logAge) :
        controls_.tracks()->getIsochrone(logAge, feh);
    return specCtsImpl(isochrone, imf, 1.0, mMin, mMax, feh, true, computeLbol);
}

auto specsyn::Specsyn::specCtsHelper(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster,
    const double mMin,
    const double mMax,
    const bool computeLbol
) const -> std::vector<double>
{
    // Reflect sfr about half the current time, so this dimension's
    // own coordinate -- what continuousSpecIntegrand() reads as its
    // own age parameter -- becomes stellar age (curTime - time)
    // directly, rather than time itself (see PDFReflect::reflect()'s
    // own comment: pivot = curTime / 2 makes reflect(x) = curTime - x).
    // The luminosity this integral is dominated by -- young, hot, blue
    // stars -- is concentrated in a narrow sliver of *age* near 0
    // (equivalently, *time* near curTime); on a linear time axis, that
    // sliver's width relative to the full [0, curTime] domain shrinks
    // as curTime grows, needing on the order of curTime / (feature
    // width) quadrature subdivisions to resolve at large curTime --
    // catastrophically slow for any fixed-tolerance adaptive scheme
    // that has to keep bisecting down to that same narrow sliver.
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

    // A PDFIntegrator (GKOrder::GK15) over age alone, whose integrand
    // (continuousSpecIntegrand()) performs a complete nested 1D
    // integral over mass internally (itself also a PDFIntegrator --
    // see specCtsImpl()'s own comment), at each age point visited; see
    // continuousSpecIntegrand()'s own comment and this class's own
    // specCts() overload's header comment for why nesting two 1D
    // integrals this way, rather than one joint 2D (age, mass)
    // integral, is what actually fixes the cost-cliff problem that
    // motivated this design. Was a plain 1D utils::PDFIntegrator
    // (CubatureMethod::hAdaptive) until benchmarked head to head
    // against PDFIntegrator on the same full-scale run described in
    // specCtsImpl()'s own comment: swapping both this outer age
    // integral and specCtsImpl()'s own inner mass integral to
    // PDFIntegrator (GK15 for both) cut that run's total time from
    // 449s to 37s -- a much larger win than the inner loop's own ~3x,
    // since the old hAdaptive age integral was itself the larger
    // remaining bottleneck.
    using IntegrandFn = std::vector<double> (Specsyn::*)(
        double, const pdfs::PDF&, double, double, bool, double) const;
    const utils::PDFIntegrator<IntegrandFn, utils::GKOrder::GK15> integrator(
        sfrAge, static_cast<IntegrandFn>(&Specsyn::continuousSpecIntegrand),
        nInt, logAge, intMaxIter(), absTol, intRelTol());

    std::vector<double> result;
    if (singleFeh)
    {
        result = integrator.integrate(
            ageMin, curTime, this, imf, mMin, mMax, computeLbol, fehDist.getMin());
    }
    else
    {
        // Integrate over [Fe/H] with a third, outermost PDFIntegrator,
        // weighted by fehDist, whose integrand is the complete nested
        // (age, mass) integral above at each [Fe/H] it visits. The
        // tracks are evaluated lazily at arbitrary [Fe/H] (see
        // tracks::Tracks3D's own class comment), so each such point
        // costs about as much as one at a track grid point. Divided by
        // fehDist's own integral over its support, in case fehDist is
        // not normalized.
        auto fehIntegrand = [&](const double feh) -> std::vector<double>
        {
            return integrator.integrate(
                ageMin, curTime, this, imf, mMin, mMax, computeLbol, feh);
        };
        const utils::PDFIntegrator<decltype(fehIntegrand), utils::GKOrder::GK15> fehIntegrator(
            fehDist, fehIntegrand, nInt, false, intMaxIter(), absTol, intRelTol());
        result = fehIntegrator.integrate(fehDist.getMin(), fehDist.getMax());
        const double fehNorm = fehDist.integral(fehDist.getMin(), fehDist.getMax());
        for (double& r : result) { r /= fehNorm; }
    }

    // Undo continuousSpecIntegrand()'s own lambda * dL/dlambda
    // weighting over the spectral elements alone (the trailing Lbol
    // element, if present, is a genuine luminosity already, not a
    // lambda-weighted flux, so it's excluded from this division), and
    // scale everything -- spectrum and Lbol alike -- by
    // (1 - fCluster) * (1 - controls_.fracStochMass()) for the
    // continuously-treated share of the population -- see specCts()'s
    // own comment for why the second factor is needed alongside mMin/
    // mMax already restricting the integral itself.
    // The last factor converts the per-star mass integral into one
    // per unit mass of stars in [mMin, mMax] -- see imfMassIntegral()'s
    // own comment
    const double massInt = imfMassIntegral(imf, mMin, mMax);
    const double ctsFrac = massInt > 0.0 ?
        (1.0 - fCluster) * (1.0 - controls_.fracStochMass()) / massInt : 0.0;
    for (std::size_t i = 0; i < wl_.size(); ++i)
    {
        result[i] = result[i] * ctsFrac / wl_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- wl_.size() <= result.size() by construction (result.size() == wl_.size() + computeLbol), and i is bounded by wl_.size()
    }
    if (computeLbol)
    {
        // continuousSpecIntegrand() reports Lbol in erg/s, not Lsun --
        // see its own comment for why -- so convert back to Lsun here,
        // matching Cluster::lbol()'s own units, alongside the same
        // ctsFrac scaling every other element gets.
        result[wl_.size()] *= ctsFrac / utils::Lsun; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result.size() == wl_.size() + 1 when computeLbol is true, by construction
    }
    return result;
}

auto specsyn::Specsyn::specCts(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster,
    const double mMin,
    const double mMax
) const -> std::vector<double>
{
    return specCtsHelper(sfr, imf, fehDist, curTime, fCluster, mMin, mMax, false);
}

auto specsyn::Specsyn::specAndLbolCts(
    const pdfs::PDF& sfr,
    const pdfs::PDF& imf,
    const pdfs::PDF& fehDist,
    const double curTime,
    const double fCluster,
    const double mMin,
    const double mMax
) const -> std::pair<std::vector<double>, double>
{
    auto spec = specCtsHelper(sfr, imf, fehDist, curTime, fCluster, mMin, mMax, true);
    const double lBol = spec.back();
    spec.pop_back();
    return { std::move(spec), lBol };
}
