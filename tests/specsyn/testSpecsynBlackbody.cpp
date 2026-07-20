/**
 * @file testSpecsynBlackbody.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SpecsynBlackbody class.
 * @date 2026-07-18
 */

#include "../src/pdfs/PDF.hpp"
#include "../src/pdfs/PDFFileParser.hpp"
#include "../src/specsyn/Specsyn.hpp"
#include "../src/specsyn/SpecsynBlackbody.hpp"
#include "../src/tracks/TrackCommons.hpp"
#include "testSpecsynBlackbody.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_const_cgsm.h> // NOLINT(misc-include-cleaner)
#include <gsl/gsl_interp.h> // NOLINT(misc-include-cleaner)
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

namespace
{
    // Mirror the physical constants and formulas used internally by
    // SpecsynBlackbody, so this test can independently recompute the
    // expected blackbody spectrum for made-up stellar data and check
    // it against the class's actual output, rather than just checking
    // that spec() runs without crashing.
    constexpr double planckH = GSL_CONST_CGSM_PLANCKS_CONSTANT_H;
    constexpr double speedOfLight = GSL_CONST_CGSM_SPEED_OF_LIGHT;
    constexpr double stefanBoltzmann = GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT;
    constexpr double boltzmannK = GSL_CONST_CGSM_BOLTZMANN;
    constexpr double solarLuminosity = 3.828e33; // erg/s, IAU 2015 nominal value
    constexpr double pi = std::numbers::pi_v<double>;
    constexpr double angstromToCm = 1e-8;
    constexpr double relTol = 1e-6;

    // Arbitrary [Fe/H] value, used only because spec()/specCts() now
    // require one; SpecsynBlackbody ignores it entirely, since a
    // blackbody spectrum depends only on temperature and radius
    constexpr double testFeh = -0.5;
} // namespace

// Check that the wavelength grid has the documented size and is
// strictly increasing
static auto testWlGrid(const std::vector<double>& wl) -> int
{
    constexpr std::size_t expectedSize = 1000;
    if (wl.size() != expectedSize)
    {
        std::cerr << "testSpecsynBlackbody: expected a " << expectedSize
            << "-point wavelength grid, got " << wl.size() << "\n";
        return 1;
    }
    for (std::size_t i = 1; i < wl.size(); ++i)
    {
        if (wl.at(i) <= wl.at(i - 1))
        {
            std::cerr << "testSpecsynBlackbody: wavelength grid is not "
                "strictly increasing at index " << i << "\n";
            return 1;
        }
    }
    return 0;
}

// Check spec()'s output, for some made-up (T, L), against an
// independently-computed expected blackbody spectrum
static auto testSpec(const specsyn::SpecsynBlackbody& synth) -> int
{
    // Bogus, made-up stellar data: T = 10^4 K, L = 100 Lsun
    constexpr double logTeff = 4.0;
    constexpr double logL = 2.0;
    specsyn::Specsyn::StarData props{};
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;

    const auto result = synth.spec(props, testFeh);
    const auto& wl = synth.wl();
    if (result.size() != wl.size())
    {
        std::cerr << "testSpecsynBlackbody: spec() returned " << result.size()
            << " values, expected " << wl.size() << "\n";
        return 1;
    }

    const double temperature = std::pow(10.0, logTeff);
    const double luminosity = std::pow(10.0, logL) * solarLuminosity;
    const double temperature4 = temperature * temperature * temperature * temperature;
    const double radius = std::sqrt(luminosity / (4.0 * pi * stefanBoltzmann * temperature4));
    const double area = 4.0 * pi * radius * radius;

    for (std::size_t i = 0; i < wl.size(); ++i)
    {
        const double wlCm = wl.at(i) * angstromToCm;
        const double x = planckH * speedOfLight / (wlCm * boltzmannK * temperature);
        const double bLambda = (2.0 * planckH * speedOfLight * speedOfLight) /
            ((wlCm * wlCm * wlCm * wlCm * wlCm) * (std::exp(x) - 1.0));
        const double expected = area * bLambda * angstromToCm;

        const bool ok = (expected == 0.0)
            ? (result.at(i) == 0.0)
            : (std::abs(result.at(i) - expected) <= relTol * std::abs(expected));
        if (!ok)
        {
            std::cerr << "testSpecsynBlackbody: at wl = " << wl.at(i)
                << " Angstrom, expected " << expected
                << " erg/s/Angstrom, got " << result.at(i) << "\n";
            return 1;
        }
    }

    return 0;
}

// Build a single-segment isochrone spanning [mLo, mHi] with constant
// stellar properties (logTeff, logL) at every mass -- since spec()
// then does not depend on mass at all, the population spectrum
// specCts() computes reduces to a simple, independently verifiable
// closed form (see testSpecCts)
static auto makeConstPropsIsochrone(const double logTeff, const double logL,
    const double mLo, const double mHi) -> specsyn::Specsyn::Isochrone
{
    constexpr auto nF = static_cast<std::size_t>(tracks::FieldIdx::nTrackQty);
    const std::vector<double> masses = { mLo, mHi };
    std::array<std::vector<double>, nF> props;
    for (auto& p : props) { p = { 0.0, 0.0 }; }
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = { logL, logL };
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = { logTeff, logTeff };

    specsyn::Specsyn::Isochrone isochrone;
    isochrone.push_back(std::make_unique<specsyn::Specsyn::Segment>(
        masses, props, gsl_interp_linear));
    return isochrone;
}

// Check specCts()'s output against an independently-computed expected
// value, using a real IMF (data/imfs/chabrier.toml) and a made-up,
// mass-independent isochrone (see makeConstPropsIsochrone): since
// spec() does not depend on mass in this construction, it factors out
// of the population integral entirely, so
// specCts() == mTot * PDF::integral(mMin, mMax) * spec(props)
static auto testSpecCts(const specsyn::SpecsynBlackbody& synth) -> int
{
    constexpr double logTeff = 4.0;
    constexpr double logL = 2.0;
    constexpr double mLo = 0.1;
    constexpr double mHi = 100.0;
    const auto isochrone = makeConstPropsIsochrone(logTeff, logL, mLo, mHi);

    const pdfs::PDF imf = pdfs::parsePDFDescriptor("data/imfs/chabrier.toml");

    constexpr double mTot = 1e5;
    constexpr double mMin = 1.0;
    constexpr double mMax = 50.0;
    const auto result = synth.specCts(isochrone, imf, mTot, mMin, mMax, testFeh);

    specsyn::Specsyn::StarData props{};
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
    props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
    const auto starSpec = synth.spec(props, testFeh);

    if (result.size() != starSpec.size())
    {
        std::cerr << "testSpecsynBlackbody: specCts: expected " << starSpec.size()
            << " wavelength points, got " << result.size() << "\n";
        return 1;
    }

    const double weight = mTot * imf.integral(mMin, mMax);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        const double expected = weight * starSpec.at(i);
        const bool ok = (expected == 0.0)
            ? (result.at(i) == 0.0)
            : (std::abs(result.at(i) - expected) <= relTol * std::abs(expected));
        if (!ok)
        {
            std::cerr << "testSpecsynBlackbody: specCts: at i = " << i
                << " expected " << expected << ", got " << result.at(i) << "\n";
            return 1;
        }
    }
    return 0;
}

auto testSpecsynBlackbody() -> int
{
    const specsyn::SpecsynBlackbody synth;
    int result = 0;
    result += testWlGrid(synth.wl());
    result += testSpec(synth);
    result += testSpecCts(synth);
    return result;
}
