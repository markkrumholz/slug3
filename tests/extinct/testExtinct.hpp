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
#include "../src/io/SimControls.hpp"
#include "../src/phot/FilterTabulated.hpp"
#include "../src/utils/HDF5Utils.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

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

    // Build a request grid that extends well past both ends of the
    // native curve (to exercise truncation), plus a few points that
    // land exactly on the native grid (so interpolated values can be
    // checked exactly against the raw data, since any interpolant
    // reproduces its own input nodes)
    std::vector<double> wl;
    for (double w = 100.0; w <= 30000.0; w += 137.0) { wl.push_back(w); }
    for (const double w : {1000.0, 5000.0, 10000.0, wlRaw.back()}) { wl.push_back(w); }
    std::ranges::sort(wl);

    const io::SimControls testControls;
    const extinct::Extinct ext(curveName, wl, testControls, registryName);

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

    // At wavelengths landing exactly on the native grid, the
    // interpolated value should exactly reproduce the native data,
    // up to the single uniform scale factor Extinct's constructor
    // applies via normalize() (see testExtinctNormalization() for a
    // dedicated check of that scale factor's own meaning) -- so the
    // ratio of interpolated to native values should be the same at
    // every such point
    double scale = 0.0;
    for (const double w : {1000.0, 5000.0, 10000.0, wlRaw.back()})
    {
        const auto rawIt = std::ranges::find(wlRaw, w);
        if (rawIt == wlRaw.end())
        {
            std::cerr << "testExtinct: test bug: " << w << " is not on the native grid\n";
            return 1;
        }
        const auto rawIdx = static_cast<std::size_t>(std::distance(wlRaw.begin(), rawIt));

        const auto extIt = std::ranges::find(ext.wl(), w);
        if (extIt == ext.wl().end())
        {
            std::cerr << "testExtinct: " << w << " missing from wl()\n";
            return 1;
        }
        const auto extIdx = static_cast<std::size_t>(std::distance(ext.wl().begin(), extIt));

        const double thisScale = ext.extinct().at(extIdx) / kappaRaw.at(rawIdx);
        if (scale == 0.0) { scale = thisScale; }
        else if (!utils::approxEqual(thisScale, scale))
        {
            std::cerr << "testExtinct: interpolated/native ratio at w=" << w
                << " is " << thisScale << ", expected " << scale
                << " (the same ratio found at other native grid points)\n";
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
    redshiftedControls.setZ(0.1);
    const extinct::Extinct extRedshifted(curveName, wl, redshiftedControls, registryName);
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
        const extinct::Extinct bad("NotARealCurve", wl, testControls, registryName);
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
 * Builds a flat (constant F_lambda) spectrum across the V band,
 * photometers it, then for several A_V values extinguishes it via
 * applyExtinction() and photometers the result again -- the resulting
 * magnitude should be fainter by almost exactly A_V mag, since
 * normalize() calibrates every curve's own extinct() to A_V = 1 mag
 * and applyExtinction() applies exp(-A_V * extinct()). "Almost" rather
 * than "exactly" because averaging kappa*R over frequency (the
 * definition normalize() uses) and averaging F_lambda*R over
 * ln(lambda) (what phot() itself does) are two different weightings
 * of the same spectrum, so they don't cancel perfectly -- but the
 * mismatch should be at most a percent or two.
 */
auto testExtinctNormalization() -> int
{
    const phot::FilterTabulated vFilt("Generic", "Johnson", "V", "data/filters/V_filter.toml");

    // A flat spectrum on the V filter's own native wavelength grid
    constexpr double f0 = 1.0;
    const std::vector<double> specBefore(vFilt.wl().size(), f0);
    const double photBefore = vFilt.phot(vFilt.wl(), specBefore);
    const double magBefore = -2.5 * std::log10(photBefore);

    // Any curve will do; V's own wavelength coverage sits comfortably
    // inside every curve's own native range, so wl() below should
    // come back identical (untruncated) to vFilt.wl(), i.e. wlOffset_
    // should be 0 and specBefore is already on the right grid for
    // applyExtinction()
    const io::SimControls testControls;
    const extinct::Extinct ext("Calzetti_starburst", vFilt.wl(), testControls);
    if (ext.wl() != vFilt.wl())
    {
        std::cerr << "testExtinctNormalization: test bug: Extinct truncated "
            "the V filter's own wavelength grid\n";
        return 1;
    }

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
 * A bare default-constructed SimControls (as testExtinct()'s own
 * construction above uses) has an invalid avDistField() -- see
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
    const phot::FilterTabulated vFilt("Generic", "Johnson", "V", "data/filters/V_filter.toml");
    const io::SimControls testControls;
    const extinct::Extinct ext("Calzetti_starburst", vFilt.wl(), testControls);

    if (testControls.avDistField().valid())
    {
        std::cerr << "testExtinctApplyExtinctionCtsInvalid: test bug: expected "
            "a bare default-constructed SimControls to have an invalid "
            "avDistField()\n";
        return 1;
    }

    // vFilt.wl() falls entirely within the Calzetti curve's own native
    // coverage (see testExtinctNormalization()'s own identical
    // assumption), so ext.wl() == vFilt.wl() and wlOffset() == 0 here
    const std::vector<double> spec(ext.wl().size(), 1.0);
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
 * A bare default-constructed SimControls (as testExtinct()'s own
 * construction above uses) has SimControls::nebular() == nullptr, so
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
    const phot::FilterTabulated vFilt("Generic", "Johnson", "V", "data/filters/V_filter.toml");
    const io::SimControls testControls;
    if (testControls.nebular() != nullptr)
    {
        std::cerr << "testExtinctLinesEmpty: test bug: expected a bare "
            "default-constructed SimControls to have a null nebular()\n";
        return 1;
    }

    const extinct::Extinct ext("Calzetti_starburst", vFilt.wl(), testControls);

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

#endif // TESTEXTINCT_HPP
