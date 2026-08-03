/**
 * @file testExtinct.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Extinct class
 * @date 2026-08-03
 */

#ifndef TESTEXTINCT_HPP
#define TESTEXTINCT_HPP

#include "../src/extinct/Extinct.hpp"
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

    const extinct::Extinct ext(curveName, wl, registryName);

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

    // An unrecognized curve name should raise, not crash
    try
    {
        const extinct::Extinct bad("NotARealCurve", wl, registryName);
        std::cerr << "testExtinct: expected exception for unknown curve name\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0; // Passed
}

/**
 * @brief Unit test for Extinct's V-band normalization
 * @returns 0 if the test passes, 1 if it fails
 * @details
 * Builds a flat (constant F_lambda) spectrum across the V band,
 * photometers it, then extinguishes it by multiplying by
 * exp(-extinct()) and photometers the result again -- the resulting
 * magnitude should be fainter by almost exactly 1 mag, since Extinct
 * normalizes every curve to A_V = 1 mag. "Almost" rather than
 * "exactly" because averaging kappa*R over frequency (the definition
 * used to normalize) and averaging F_lambda*R over ln(lambda) (what
 * phot() itself does) are two different weightings of the same
 * spectrum, so they don't cancel perfectly -- but the mismatch should
 * be at most a percent or two.
 */
auto testExtinctNormalization() -> int
{
    const phot::FilterTabulated vFilt("Generic", "Johnson", "V", "data/filters/V_filter.toml");

    // A flat spectrum on the V filter's own native wavelength grid
    constexpr double f0 = 1.0;
    const std::vector<double> specBefore(vFilt.wl().size(), f0);
    const double photBefore = vFilt.phot(vFilt.wl(), specBefore);

    // Any curve will do; V's own wavelength coverage sits comfortably
    // inside every curve's own native range, so wl() below should
    // come back identical (untruncated) to vFilt.wl()
    const extinct::Extinct ext("Calzetti_starburst", vFilt.wl());
    if (ext.wl() != vFilt.wl())
    {
        std::cerr << "testExtinctNormalization: test bug: Extinct truncated "
            "the V filter's own wavelength grid\n";
        return 1;
    }

    std::vector<double> specAfter(specBefore.size());
    for (std::size_t i = 0; i < specBefore.size(); i++)
    {
        specAfter.at(i) = specBefore.at(i) * std::exp(-ext.extinct().at(i));
    }
    const double photAfter = vFilt.phot(ext.wl(), specAfter);

    const double magBefore = -2.5 * std::log10(photBefore);
    const double magAfter = -2.5 * std::log10(photAfter);
    const double deltaMag = magAfter - magBefore;

    constexpr double expected = 1.0;
    constexpr double tol = 0.02; // a percent or two
    if (!utils::approxEqual(deltaMag, expected, tol))
    {
        std::cerr << "testExtinctNormalization: extinguished spectrum is "
            << deltaMag << " mag fainter, expected " << expected
            << " (tolerance " << tol << ")\n";
        return 1;
    }

    return 0; // Passed
}

#endif // TESTEXTINCT_HPP
