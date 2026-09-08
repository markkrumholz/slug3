/**
 * @file testExtinct.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Extinct class
 * @date 2026-08-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTEXTINCT_HPP
#define TESTEXTINCT_HPP

#include "../src/extinct/Extinct.hpp"
#include "../src/interpolation/Interpolator1D.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterTabulated.hpp"
#include "../src/specsyn/SpecsynBlackbody.hpp"
#include "../src/utils/HDF5Utils.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // A wide logspace wavelength grid, deliberately extending well
    // past both ends of every standard extinction curve's own native
    // coverage (to exercise truncation), attached to controls as its
    // own spectral synthesizer -- Extinct now always reads its
    // wavelength grid from controls.specsyn()->wl(), so every test
    // below needs a real (if physically meaningless) one attached
    // before constructing an Extinct against it.
    void attachWideWlGrid(io::SimControls& controls) // NOLINT(llvm-prefer-static-over-anonymous-namespace)
    {
        controls.setSpecsyn(std::make_unique<specsyn::SpecsynBlackbody>(
            100.0, 30000.0, 2000, controls));
    }
} // namespace

/**
 * @brief Unit test for the Extinct class
 * @returns 0 if the test passes, 1 if it fails
 */
auto testExtinct() -> int
{
    const std::string registryName = "data/extinct/extinct.toml";
    const std::string curveName = "Calzetti_starburst";

    // Independently read the same curve's raw data directly from the
    // HDF5 file, bypassing Extinct entirely -- ground truth to check
    // Extinct's own wlDat()/extinctDat() against
    // NOLINTBEGIN(misc-include-cleaner) -- see HDF5Utils.hpp's own comment
    const hid_t file = H5Fopen("data/extinct/extinct.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    const hid_t grp = H5Gopen2(file, curveName.c_str(), H5P_DEFAULT);
    const auto wlRaw = utils::readDataset1D(grp, "wavelength", "testExtinct");
    const auto kappaRaw = utils::readDataset1D(grp, "kappa", "testExtinct");
    H5Gclose(grp);
    H5Fclose(file);
    // NOLINTEND(misc-include-cleaner)

    io::SimControls testControls;
    attachWideWlGrid(testControls);
    const auto& wl = testControls.specsyn()->wl();

    const extinct::Extinct ext(curveName, testControls, registryName);

    // Data should have been loaded correctly
    if (ext.wlDat() != wlRaw)
    {
        std::cerr << "testExtinct: wlDat() does not match raw HDF5 data\n";
        return 1;
    }
    if (ext.extinctDat() != kappaRaw)
    {
        std::cerr << "testExtinct: extinctDat() does not match raw HDF5 data\n";
        return 1;
    }

    // wl() should be truncated to the native curve's own coverage
    if (ext.wl().front() < wlRaw.front() || ext.wl().back() > wlRaw.back())
    {
        std::cerr << "testExtinct: wl() extends beyond native curve coverage\n";
        return 1;
    }
    std::size_t expectedKept = 0;
    for (const double w : wl)
    {
        if (w >= wlRaw.front() && w <= wlRaw.back()) { expectedKept++; }
    }
    if (ext.wl().size() != expectedKept)
    {
        std::cerr << "testExtinct: wl() has " << ext.wl().size()
            << " points, expected " << expectedKept << "\n";
        return 1;
    }
    if (ext.wl().size() != ext.extinct().size())
    {
        std::cerr << "testExtinct: wl() and extinct() have different sizes\n";
        return 1;
    }

    // ext.extinct() should be a pointwise rescaling (by normalize()'s
    // single uniform scale factor -- see testExtinctNormalization()
    // for a dedicated check of that scale factor's own meaning) of
    // the native curve, independently reinterpolated here directly
    // from wlRaw/kappaRaw -- checked at every point in ext.wl(), not
    // just a few hand-picked ones
    const interp::Interpolator1D<1> refInterp(wlRaw, kappaRaw);
    double scale = 0.0;
    bool haveScale = false;
    for (std::size_t i = 0; i < ext.wl().size(); i++)
    {
        const double refVal = refInterp(ext.wl().at(i));
        if (refVal == 0.0)
        {
            if (!utils::approxEqual(ext.extinct().at(i), 0.0))
            {
                std::cerr << "testExtinct: extinct()[" << i << "] = "
                    << ext.extinct().at(i) << ", expected 0 (native curve is 0 there)\n";
                return 1;
            }
            continue;
        }
        const double thisScale = ext.extinct().at(i) / refVal;
        if (!haveScale) { scale = thisScale; haveScale = true; }
        else if (!utils::approxEqual(thisScale, scale))
        {
            std::cerr << "testExtinct: interpolated/native ratio at wl=" << ext.wl().at(i)
                << " is " << thisScale << ", expected " << scale
                << " (the same ratio found at other points)\n";
            return 1;
        }
    }

    // wlObs() should read z live from testControls: 0 (the default)
    // means it matches wl() exactly, and a nonzero z should redshift
    // every element by exactly (1 + z)
    if (ext.wlObs() != ext.wl())
    {
        std::cerr << "testExtinct: wlObs() should equal wl() when z = 0\n";
        return 1;
    }
    io::SimControls redshiftedControls;
    attachWideWlGrid(redshiftedControls);
    redshiftedControls.setZ(0.1);
    const extinct::Extinct extRedshifted(curveName, redshiftedControls, registryName);
    for (std::size_t i = 0; i < extRedshifted.wl().size(); i++)
    {
        if (!utils::approxEqual(extRedshifted.wlObs().at(i), extRedshifted.wl().at(i) * 1.1))
        {
            std::cerr << "testExtinct: wlObs() does not match wl() * (1 + z) at index " << i << "\n";
            return 1;
        }
    }

    // An unrecognized curve name should raise, not crash
    try
    {
        const extinct::Extinct bad("NotARealCurve", testControls, registryName);
        std::cerr << "testExtinct: expected exception for unknown curve name\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0; // Passed
}

/**
 * @brief Unit test for Extinct's V-band normalization and applyExtinction()
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Builds a flat (constant F_lambda) spectrum on this Extinct's own
 * wavelength grid, photometers it through the V filter, then for
 * several A_V values extinguishes it via applyExtinction() and
 * photometers the result again -- the resulting magnitude should be
 * fainter by almost exactly A_V mag, since normalize() calibrates
 * every curve's own extinct() to A_V = 1 mag and applyExtinction()
 * applies exp(-A_V * extinct()). "Almost" rather than "exactly"
 * because averaging kappa*R over frequency (the definition
 * normalize() uses) and averaging F_lambda*R over ln(lambda) (what
 * phot() itself does) are two different weightings of the same
 * spectrum, so they don't cancel perfectly -- but the mismatch should
 * be at most a percent or two.
 */
auto testExtinctNormalization() -> int
{
    const phot::FilterTabulated vFilt("Generic", "Johnson", "V", "data/filters/V_filter.toml");

    io::SimControls testControls;
    attachWideWlGrid(testControls);
    const extinct::Extinct ext("Calzetti_starburst", testControls);

    // applyExtinction()'s own spec argument must be tabulated on
    // exactly controls.specsyn()->wl() -- the full, untruncated grid
    // -- not ext.wl() (a possibly-narrower, truncated copy of it; see
    // wlOffset()'s own comment): it indexes spec[wlOffset() + i]
    // internally, so a spec sized to ext.wl() instead would run past
    // its own end whenever wlOffset() > 0, as it is here (the wide
    // grid attachWideWlGrid() attaches extends past the curve's own
    // native coverage on both ends, deliberately, to also exercise
    // truncation). Filter::phot() itself accepts any wavelength grid
    // (it interpolates the filter's own response onto whatever's
    // given, not the other way around), so using the full grid for
    // photBefore too, rather than vFilt's own, is equally valid.
    const auto& fullWl = testControls.specsyn()->wl();
    constexpr double f0 = 1.0;
    const std::vector<double> specBefore(fullWl.size(), f0);
    const double photBefore = vFilt.phot(fullWl, specBefore);
    const double magBefore = -2.5 * std::log10(photBefore);

    constexpr double relTol = 0.02; // a percent or two
    for (const double AV : {0.5, 1.0, 2.0, 3.5})
    {
        const auto specAfter = ext.applyExtinction(AV, specBefore);
        const double photAfter = vFilt.phot(ext.wl(), specAfter);
        const double magAfter = -2.5 * std::log10(photAfter);
        const double deltaMag = magAfter - magBefore;

        const double relErr = std::abs(deltaMag - AV) / AV;
        if (relErr > relTol)
        {
            std::cerr << "testExtinctNormalization: A_V=" << AV
                << ": extinguished spectrum is " << deltaMag
                << " mag fainter (relative error " << relErr
                << ", tolerance " << relTol << ")\n";
            return 1;
        }
    }

    return 0; // Passed
}

/**
 * @brief Unit test for Extinct::applyExtinctionCts() with an invalid avDistField()
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * A bare default-constructed SimControls (beyond attachWideWlGrid()'s
 * own setSpecsyn() call) has an invalid avDistField() -- see
 * io::SimControls::avDistField()'s own comment. computeExtinctionFacCts()
 * treats this as a delta at A_V = 0, so applyExtinctionCts() should
 * leave an unattenuated spectrum completely unchanged (exp(0) = 1
 * everywhere), rather than either throwing (PDFSegmentDelta::operator()
 * cannot be evaluated at a point) or hanging (a NaN-bounded
 * utils::PDFIntegrator call would never satisfy its own convergence
 * checks).
 */
auto testExtinctApplyExtinctionCtsInvalid() -> int
{
    io::SimControls testControls;
    attachWideWlGrid(testControls);
    const extinct::Extinct ext("Calzetti_starburst", testControls);

    if (testControls.avDistField().valid())
    {
        std::cerr << "testExtinctApplyExtinctionCtsInvalid: test bug: expected "
            "a bare default-constructed SimControls to have an invalid "
            "avDistField()\n";
        return 1;
    }

    // spec must be tabulated on the full controls.specsyn()->wl(),
    // not ext.wl() -- see testExtinctNormalization()'s own identical
    // comment for why
    const std::vector<double> spec(testControls.specsyn()->wl().size(), 1.0);
    const auto result = ext.applyExtinctionCts(spec);
    for (std::size_t i = 0; i < result.size(); i++)
    {
        if (!utils::approxEqual(result.at(i), 1.0))
        {
            std::cerr << "testExtinctApplyExtinctionCtsInvalid: expected "
                "applyExtinctionCts() to leave an unattenuated spectrum "
                "unchanged (A_V = 0) when avDistField() is invalid, got "
                << result.at(i) << " at index " << i << "\n";
            return 1;
        }
    }
    return 0; // Passed
}

/**
 * @brief Unit test for Extinct's line-luminosity extinction support with no nebular emission grid
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * A bare default-constructed SimControls (beyond attachWideWlGrid()'s
 * own setSpecsyn() call) has SimControls::nebular() == nullptr, so
 * Extinct's own extinctLines_/extinctionFacCtsLines_ should both stay
 * empty, and applyExtinctionLines()/applyExtinctionCtsLines() should
 * both return an empty vector regardless of A_V -- there are no lines
 * to extinguish. See tests/core/testCluster.cpp's own
 * testClusterExtinctLines() for the positive-path check, with a real
 * nebular emission grid present, since building one needs
 * SimControls's full toml-deck constructor (and the tracks/spectra it
 * depends on), not available in this minimal test target.
 */
auto testExtinctLinesEmpty() -> int
{
    io::SimControls testControls;
    attachWideWlGrid(testControls);
    if (testControls.nebular() != nullptr)
    {
        std::cerr << "testExtinctLinesEmpty: test bug: expected a bare "
            "default-constructed SimControls to have a null nebular()\n";
        return 1;
    }

    const extinct::Extinct ext("Calzetti_starburst", testControls);

    const std::vector<double> noLines;
    if (!ext.applyExtinctionLines(1.0, noLines).empty())
    {
        std::cerr << "testExtinctLinesEmpty: expected applyExtinctionLines() "
            "to return an empty vector when no nebular emission grid was "
            "requested\n";
        return 1;
    }
    if (!ext.applyExtinctionCtsLines(noLines).empty())
    {
        std::cerr << "testExtinctLinesEmpty: expected applyExtinctionCtsLines() "
            "to return an empty vector when no nebular emission grid was "
            "requested\n";
        return 1;
    }

    return 0; // Passed
}

/**
 * @brief Unit test for loadCurve()'s failure-atomicity
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Reloading a bogus curve name into an already-successfully-loaded
 * Extinct must leave every one of its cached quantities exactly as
 * they were beforehand, not some inconsistent mix of the previous
 * curve and a partially-applied new one -- see loadCurve()'s own
 * comment on why it (and rebuildCache()) is failure-atomic. Also
 * checks that a subsequent, valid reload still works normally
 * afterward, i.e. the failed reload didn't leave the object broken in
 * some other way.
 */
auto testExtinctLoadCurveAtomic() -> int
{
    io::SimControls testControls;
    attachWideWlGrid(testControls);
    extinct::Extinct ext("Calzetti_starburst", testControls);

    const auto wlDatBefore = ext.wlDat();
    const auto extinctDatBefore = ext.extinctDat();
    const auto wlBefore = ext.wl();
    const auto extinctBefore = ext.extinct();
    const auto wlOffsetBefore = ext.wlOffset();

    try
    {
        ext.loadCurve("NotARealCurve");
        std::cerr << "testExtinctLoadCurveAtomic: expected exception for unknown curve name\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    if (ext.wlDat() != wlDatBefore || ext.extinctDat() != extinctDatBefore ||
        ext.wl() != wlBefore || ext.extinct() != extinctBefore ||
        ext.wlOffset() != wlOffsetBefore)
    {
        std::cerr << "testExtinctLoadCurveAtomic: a failed loadCurve() call "
            "left this Extinct's cached state changed from before the call\n";
        return 1;
    }

    // The object should still be fully usable afterward, including for
    // a subsequent, valid reload of a genuinely different curve --
    // Bouchet_SMC shares Calzetti_starburst's own native wavelength
    // grid (both tabulated on the same instrument grid), so it's
    // extinctDat() (the actual curve values), not wlDat(), that must
    // differ here
    ext.loadCurve("Bouchet_SMC");
    if (ext.extinctDat() == extinctDatBefore)
    {
        std::cerr << "testExtinctLoadCurveAtomic: test bug: expected "
            "Bouchet_SMC's own extinctDat() to differ from Calzetti_starburst's\n";
        return 1;
    }

    return 0; // Passed
}

#endif // TESTEXTINCT_HPP
