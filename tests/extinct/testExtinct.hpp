/**
 * @file testExtinct.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Extinct class
 * @date 2026-08-03
 */

#ifndef TESTEXTINCT_HPP
#define TESTEXTINCT_HPP

#include "../src/extinct/Extinct.hpp"
#include "../src/utils/HDF5Utils.hpp"
#include "../src/utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <algorithm>
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
    // interpolated value should exactly match the native data
    for (const double w : {1000.0, 5000.0, 10000.0})
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

        if (!utils::approxEqual(ext.extinct().at(extIdx), kappaRaw.at(rawIdx)))
        {
            std::cerr << "testExtinct: interpolated value at w=" << w
                << " does not match native data: got " << ext.extinct().at(extIdx)
                << ", expected " << kappaRaw.at(rawIdx) << "\n";
            return 1;
        }
    }

    // The last point of wl() (the native curve's own max) should also
    // match exactly
    if (!utils::approxEqual(ext.extinct().back(), kappaRaw.back()))
    {
        std::cerr << "testExtinct: interpolated value at native wl max does not match\n";
        return 1;
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

#endif // TESTEXTINCT_HPP
