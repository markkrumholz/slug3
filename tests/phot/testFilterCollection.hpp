/**
 * @file testFilterCollection.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the FilterCollection class.
 * @details
 * Uses a mix of the existing tabulated-filter test fixture (tests/
 * phot/assets/filters_test.toml, the same one testFilterTabulated.hpp
 * uses) and idealized filters (see testFilterIdeal.hpp for the same
 * naming conventions exercised in more detail on their own). Rather
 * than re-deriving each filter type's own integration formulas here
 * (already covered by testFilterTabulated.hpp/testFilterIdeal.hpp),
 * these tests check FilterCollection's own responsibilities: parsing
 * filter names into the right Filter subclass, and correctly ordering
 * and converting phot()'s results -- by comparing FilterCollection's
 * output against directly-constructed reference Filter objects and
 * (for photometric-system conversion) direct calls to PhotConvert.
 * @date 2026-07-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTFILTERCOLLECTION_HPP
#define TESTFILTERCOLLECTION_HPP

#include "../../src/phot/FilterCollection.hpp"
#include "../../src/phot/FilterIdeal.hpp"
#include "../../src/phot/FilterTabulated.hpp"
#include "../../src/phot/PhotCommons.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    const std::string registryName = "tests/phot/assets/filters_test.toml"; // NOLINT(cert-err58-cpp) -- built from a fixed string literal, so construction cannot throw

    // Build a constant spectrum F_lambda = f0 on a uniform grid of n
    // points over [wlLo, wlHi] -- since a constant spectrum's
    // integral doesn't depend on a filter's response shape, this
    // gives an exact closed form for every filter type tested here
    // (tabulated or idealized) without needing to duplicate any of
    // their own integration machinery
    auto makeConstSpec(const double wlLo, const double wlHi, const std::size_t n, const double f0)
        -> std::pair<std::vector<double>, std::vector<double>>
    {
        std::vector<double> wl(n);
        const std::vector<double> spec(n, f0);
        for (std::size_t i = 0; i < n; ++i)
        {
            wl.at(i) = wlLo + (wlHi - wlLo) * static_cast<double>(i) / static_cast<double>(n - 1);
        }
        wl.back() = wlHi; // pin endpoint to avoid round-off
        return {std::move(wl), spec};
    }

    // Disable linting for the next two functions -- including hdf5.h
    // wholesale (rather than individual headers) is the paradigm
    // HDF5 itself wants, which confuses misc-include-cleaner -- see
    // FilterCollection.cpp's own identical suppression
    // NOLINTBEGIN(misc-include-cleaner)

    // Read a 1D double dataset from an already-open HDF5 file --
    // independent, minimal re-implementation of FilterCollection.cpp's
    // own private loadVegaSpectrum/readDataset1D, so this test
    // verifies against the raw file contents rather than against
    // FilterCollection's own loading code
    auto readDataset1D(const hid_t file, const std::string& name) -> std::vector<double>
    {
        const hid_t dset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
        const hid_t space = H5Dget_space(dset);
        hsize_t dims = 0;
        H5Sget_simple_extent_dims(space, &dims, nullptr);
        std::vector<double> data(dims);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dset);
        return data;
    }

    // Load the Vega reference spectrum's wl/flux datasets directly
    auto loadVegaSpectrumForTest(const std::string& vegaName)
        -> std::pair<std::vector<double>, std::vector<double>>
    {
        const auto vegaPath = utils::getFilePath(vegaName);
        const hid_t file = H5Fopen(vegaPath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        std::vector<double> wlVega = readDataset1D(file, "wl");
        std::vector<double> fluxVega = readDataset1D(file, "flux");
        H5Fclose(file);
        return { std::move(wlVega), std::move(fluxVega) };
    }
    // NOLINTEND(misc-include-cleaner)

    // Index of the wlVega entry closest to target
    auto closestIndex(const std::vector<double>& wlVega, const double target) -> std::size_t
    {
        std::size_t best = 0;
        double bestDist = std::abs(wlVega.at(0) - target);
        for (std::size_t i = 1; i < wlVega.size(); ++i)
        {
            const double dist = std::abs(wlVega.at(i) - target);
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        return best;
    }
} // namespace

/**
 * @brief Test that FilterCollection builds the right mix of filter types and orders results correctly
 * @return 0 on pass, 1 on failure
 * @details
 * Builds a FilterCollection from one tabulated filter name
 * (SLUGTEST.CAM1.G500, an energy-flux filter) and three idealized
 * filter names (ideal_energy_700_1500, an energy-flux filter;
 * ideal_phot_700_1500 and Q(HI), both photon-count filters), with
 * photSystem = Flambda (so no conversion applies to any of them), and
 * checks filterNames(), filterUnits(), and phot() all agree, in
 * order, with the same four filters constructed directly.
 */
inline auto testFilterCollectionMixedConstruction() -> int
{
    constexpr double wlLo = 500.0, wlHi = 9000.0;
    constexpr std::size_t n = 5000;
    constexpr double f0 = 3.5;
    const auto [wl, spec] = makeConstSpec(wlLo, wlHi, n, f0);

    const std::vector<std::string> names = {
        "SLUGTEST.CAM1.G500", "ideal_energy_700_1500", "ideal_phot_700_1500", "Q(HI)"};

    try
    {
        const phot::FilterCollection fc(names, phot::PhotSystem::Flambda, registryName);

        const auto gotNames = fc.filterNames();
        const std::vector<std::string> expectedNames = {
            "SLUGTEST.CAM1.G500", "ideal_energy_700_1500", "ideal_phot_700_1500", "Q(HI)"};
        if (gotNames != expectedNames)
        {
            std::cerr << "testFilterCollectionMixedConstruction: filterNames() mismatch\n";
            return 1;
        }

        const auto gotUnits = fc.filterUnits();
        const std::vector<std::string> expectedUnits = {
            "erg/(s Angstrom)", "erg/(s Angstrom)", "photon/s", "photon/s"};
        if (gotUnits != expectedUnits)
        {
            std::cerr << "testFilterCollectionMixedConstruction: filterUnits() mismatch\n";
            return 1;
        }

        // Reference filters, constructed directly (not through
        // FilterCollection), to compare phot() against
        const phot::FilterTabulated refTab("SLUGTEST", "CAM1", "G500", registryName);
        const phot::FilterIdeal refIdealEnergy("ideal_energy_700_1500");
        const phot::FilterIdeal refIdealPhot("ideal_phot_700_1500");
        const phot::FilterIdeal refQHI("Q(HI)");

        const std::vector<double> expected = {
            refTab.phot(wl, spec), refIdealEnergy.phot(wl, spec),
            refIdealPhot.phot(wl, spec), refQHI.phot(wl, spec)};
        const auto got = fc.phot(wl, spec);

        if (got.size() != expected.size())
        {
            std::cerr << "testFilterCollectionMixedConstruction: phot() returned "
                << got.size() << " values, expected " << expected.size() << "\n";
            return 1;
        }
        constexpr double relTol = 1e-9;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const double relErr = std::abs(got.at(i) - expected.at(i)) /
                std::max(std::abs(expected.at(i)), 1.0);
            if (relErr > relTol)
            {
                std::cerr << "testFilterCollectionMixedConstruction: at i = " << i
                    << " got " << got.at(i) << ", expected " << expected.at(i)
                    << " (relative error " << relErr << ", tolerance " << relTol << ")\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterCollectionMixedConstruction: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test that FilterCollection converts energy-flux filters to a non-Flambda photSystem
 * @return 0 on pass, 1 on failure
 * @details
 * Builds the same four filters as testFilterCollectionMixedConstruction,
 * but with photSystem = AB, and checks: (1) filterUnits() reports
 * "mag(AB)" for the two energy-flux filters and "photon/s" (unchanged)
 * for the two photon-count filters; (2) phot()'s energy-flux entries
 * match phot::PhotConvert<Flambda, AB> applied directly to the same
 * reference filters' raw Flambda phot() values, at each filter's own
 * wlPivot(); (3) phot()'s photon-count entries are unconverted.
 */
inline auto testFilterCollectionPhotSystemConversion() -> int
{
    constexpr double wlLo = 500.0, wlHi = 9000.0;
    constexpr std::size_t n = 5000;
    constexpr double f0 = 3.5;
    const auto [wl, spec] = makeConstSpec(wlLo, wlHi, n, f0);

    const std::vector<std::string> names = {
        "SLUGTEST.CAM1.G500", "ideal_energy_700_1500", "ideal_phot_700_1500", "Q(HI)"};

    try
    {
        const phot::FilterCollection fc(names, phot::PhotSystem::AB, registryName);

        const auto gotUnits = fc.filterUnits();
        const std::vector<std::string> expectedUnits = {
            "mag(AB)", "mag(AB)", "photon/s", "photon/s"};
        if (gotUnits != expectedUnits)
        {
            std::cerr << "testFilterCollectionPhotSystemConversion: filterUnits() mismatch\n";
            return 1;
        }

        const phot::FilterTabulated refTab("SLUGTEST", "CAM1", "G500", registryName);
        const phot::FilterIdeal refIdealEnergy("ideal_energy_700_1500");
        const phot::FilterIdeal refIdealPhot("ideal_phot_700_1500");
        const phot::FilterIdeal refQHI("Q(HI)");

        // PhotConvert<Flambda, AB> now internally divides its input by
        // fourPiTenPcSq to convert the specific luminosity Filter::phot()
        // returns into the flux observed at the standard distance of
        // 10 pc before applying AB's own zero point -- see PhotCommons.hpp's
        // own comments -- so this test need not do that conversion itself.
        const std::vector<double> expected = {
            phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(
                refTab.phot(wl, spec), refTab.wlPivot()),
            phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(
                refIdealEnergy.phot(wl, spec), refIdealEnergy.wlPivot()),
            refIdealPhot.phot(wl, spec), // photon-count: unconverted
            refQHI.phot(wl, spec),       // photon-count: unconverted
        };
        const auto got = fc.phot(wl, spec);

        if (got.size() != expected.size())
        {
            std::cerr << "testFilterCollectionPhotSystemConversion: phot() returned "
                << got.size() << " values, expected " << expected.size() << "\n";
            return 1;
        }
        constexpr double relTol = 1e-9;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const double relErr = std::abs(got.at(i) - expected.at(i)) /
                std::max(std::abs(expected.at(i)), 1.0);
            if (relErr > relTol)
            {
                std::cerr << "testFilterCollectionPhotSystemConversion: at i = " << i
                    << " got " << got.at(i) << ", expected " << expected.at(i)
                    << " (relative error " << relErr << ", tolerance " << relTol << ")\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterCollectionPhotSystemConversion: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test the instrument-omitted "facility.filter" tabulated name form
 * @return 0 on pass, 1 on failure
 * @details
 * tests/phot/assets/filters_test.toml's SLUGTEST facility has exactly
 * one instrument (CAM1), so "SLUGTEST.G500" should resolve to the same
 * filter as the explicit "SLUGTEST.CAM1.G500" -- FilterTabulated's own
 * registry constructor always names the result "facility.instrument.
 * filter", so filterNames() should report the resolved instrument
 * even though the input name omitted it.
 */
inline auto testFilterCollectionInstrumentOmitted() -> int
{
    constexpr double wlLo = 2000.0, wlHi = 8000.0;
    constexpr std::size_t n = 5000;
    constexpr double f0 = 3.5;
    const auto [wl, spec] = makeConstSpec(wlLo, wlHi, n, f0);

    try
    {
        const phot::FilterCollection fc({"SLUGTEST.G500"}, phot::PhotSystem::Flambda, registryName);

        const auto gotNames = fc.filterNames();
        if (gotNames.size() != 1 || gotNames.at(0) != "SLUGTEST.CAM1.G500")
        {
            std::cerr << "testFilterCollectionInstrumentOmitted: expected filterNames() "
                "== {\"SLUGTEST.CAM1.G500\"}, got a different result\n";
            return 1;
        }

        const phot::FilterTabulated refTab("SLUGTEST", "CAM1", "G500", registryName);
        const auto got = fc.phot(wl, spec);
        const double expected = refTab.phot(wl, spec);
        if (got.size() != 1)
        {
            std::cerr << "testFilterCollectionInstrumentOmitted: expected 1 phot() value, "
                "got " << got.size() << "\n";
            return 1;
        }
        constexpr double relTol = 1e-9;
        const double relErr = std::abs(got.at(0) - expected) / std::abs(expected);
        if (relErr > relTol)
        {
            std::cerr << "testFilterCollectionInstrumentOmitted: got " << got.at(0)
                << ", expected " << expected << " (relative error " << relErr
                << ", tolerance " << relTol << ")\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterCollectionInstrumentOmitted: unexpected exception: "
            << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test that unparseable or invalid filter names throw
 * @return 0 on pass, 1 on failure
 */
inline auto testFilterCollectionErrors() -> int
{
    // No dots, not an idealized-filter name either
    try
    {
        const phot::FilterCollection fc({"not_a_valid_name"}, phot::PhotSystem::Flambda, registryName);
        std::cerr << "testFilterCollectionErrors: \"not_a_valid_name\" should have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // Too many dot-separated fields
    try
    {
        const phot::FilterCollection fc({"A.B.C.D"}, phot::PhotSystem::Flambda, registryName);
        std::cerr << "testFilterCollectionErrors: \"A.B.C.D\" should have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // Unknown facility
    try
    {
        const phot::FilterCollection fc({"BOGUS.CAM1.G500"}, phot::PhotSystem::Flambda, registryName);
        std::cerr << "testFilterCollectionErrors: \"BOGUS.CAM1.G500\" should have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    // Malformed idealized-filter name
    try
    {
        const phot::FilterCollection fc({"ideal_energy_500"}, phot::PhotSystem::Flambda, registryName);
        std::cerr << "testFilterCollectionErrors: \"ideal_energy_500\" should have thrown\n";
        return 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    return 0;
}

/**
 * @brief Test that PhotSystem::Vega populates fluxVega() correctly
 * @return 0 on pass, 1 on failure
 * @details
 * Builds a FilterCollection with photSystem = Vega, from a mix of
 * energy-flux (SLUGTEST.CAM1.G500, ideal_energy_4500_5500) and
 * photon-count (ideal_phot_700_1500, Q(HI)) filters, using the real,
 * committed data/spectra/vega.h5 (this collection's default vegaName).
 * Checks: (1) every photCount() filter's fluxVega() is exactly 0 (the
 * Vega spectrum is meaningless for a photon-count filter, so
 * setFluxVega() is never called on one -- see FilterCollection's
 * constructor); (2) every other filter's fluxVega() is within 5% of
 * the Vega flux read directly from data/spectra/vega.h5 at the
 * wavelength closest to that filter's own wlPivot(). This "closest
 * point" comparison is only meaningful where the Vega spectrum varies
 * smoothly across the filter's passband; ideal_energy_4500_5500 (well
 * within the optical continuum) is chosen for that reason, unlike
 * e.g. a passband straddling the Lyman limit (~912 Angstrom), where
 * flux changes by orders of magnitude and a single nearby point is
 * not a good stand-in for the passband mean.
 */
inline auto testFilterCollectionVega() -> int
{
    const std::vector<std::string> names = {
        "SLUGTEST.CAM1.G500", "ideal_energy_4500_5500", "ideal_phot_700_1500", "Q(HI)"};

    try
    {
        const phot::FilterCollection fc(names, phot::PhotSystem::Vega, registryName);
        const auto& filters = fc.filters();
        if (filters.size() != names.size())
        {
            std::cerr << "testFilterCollectionVega: expected " << names.size()
                << " filters, got " << filters.size() << "\n";
            return 1;
        }

        const auto [wlVega, fluxVega] = loadVegaSpectrumForTest(phot::defaultVegaSpec);

        constexpr double relTol = 0.05;
        for (const auto& filt : filters)
        {
            if (filt->photCount())
            {
                if (filt->fluxVega() != 0.0)
                {
                    std::cerr << "testFilterCollectionVega: filter '" << filt->name()
                        << "' is photCount(); expected fluxVega() == 0, got "
                        << filt->fluxVega() << "\n";
                    return 1;
                }
                continue;
            }

            const auto idx = closestIndex(wlVega, filt->wlPivot());
            const double expected = fluxVega.at(idx);
            const double got = filt->fluxVega();
            const double relErr = std::abs(got - expected) / std::abs(expected);
            if (relErr > relTol)
            {
                std::cerr << "testFilterCollectionVega: filter '" << filt->name()
                    << "' fluxVega() = " << got << ", expected ~" << expected
                    << " (Vega flux at wl = " << wlVega.at(idx) << " Ang, filter "
                    "pivot = " << filt->wlPivot() << " Ang; relative error "
                    << relErr << ", tolerance " << relTol << ")\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterCollectionVega: unexpected exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Test that FilterCollection::phot() reports NaN for filters truncated by a chopped wavelength grid
 * @return 0 on pass, 1 on failure
 * @details
 * Uses a constant spectrum on a full grid [500, 9000] Angstrom, and
 * the same spectrum chopped to [1000, 8000] Angstrom (as an
 * extinction curve's coverage would chop it), and checks the
 * three-argument phot() against the expected verdict for each kind
 * of filter:
 * - ideal_energy_2000_3000, entirely inside the chopped grid: finite,
 *   and equal to the two-argument phot() on the chopped grid
 * - ideal_energy_700_1500 and ideal_energy_7500_8500, each running
 *   past one edge of the chopped grid: NaN
 * - ideal_energy_100_400, entirely outside the full grid: not NaN
 *   (left as the two-argument phot() would compute it)
 * - Q(HI), with no lower bound and wlMax = 911.6 Angstrom: NaN
 * - a tabulated filter whose response is nonzero only in
 *   (1100, 5000) Angstrom, padded with zeros out to [500, 8500]:
 *   finite, since only its padding extends past the chopped grid
 *
 * It also checks that passing the full grid as both wl and wlFull
 * (no truncation) gives exactly the two-argument result, with
 * Q(HI) finite; that an empty wl gives NaN for every filter that
 * overlaps the full grid; and the tabulated filter's wlSupport().
 */
inline auto testFilterCollectionTruncation() -> int
{
    constexpr double f0 = 3.5;
    const auto [wlFull, specFull] = makeConstSpec(500.0, 9000.0, 5000, f0);
    std::vector<double> wl;
    std::vector<double> spec;
    for (std::size_t i = 0; i < wlFull.size(); ++i)
    {
        if (wlFull.at(i) >= 1000.0 && wlFull.at(i) <= 8000.0)
        {
            wl.push_back(wlFull.at(i));
            spec.push_back(specFull.at(i));
        }
    }

    try
    {
        phot::FilterCollection fc({"ideal_energy_2000_3000", "ideal_energy_700_1500",
            "ideal_energy_7500_8500", "ideal_energy_100_400", "Q(HI)"},
            phot::PhotSystem::Flambda, registryName);
        const std::vector<double> padWl = {500.0, 900.0, 1100.0, 2000.0, 3000.0, 5000.0, 8500.0};
        const std::vector<double> padResp = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
        auto padded = std::make_unique<phot::FilterTabulated>("PADDED", padWl, padResp, 2500.0);
        const auto [supMin, supMax] = padded->wlSupport();
        if (supMin != 1100.0 || supMax != 5000.0)
        {
            std::cerr << "testFilterCollectionTruncation: padded filter wlSupport() = ["
                << supMin << ", " << supMax << "], expected [1100, 5000]\n";
            return 1;
        }
        fc.addFilter(std::move(padded));

        // Expected NaN verdicts, in filter order
        const std::vector<bool> expectNaN = {false, true, true, false, true, false};
        const auto got = fc.phot(wl, spec, wlFull);
        const auto ref = fc.phot(wl, spec);
        for (std::size_t i = 0; i < expectNaN.size(); ++i)
        {
            if (std::isnan(got.at(i)) != expectNaN.at(i))
            {
                std::cerr << "testFilterCollectionTruncation: filter " << i << " ("
                    << fc.filterNames().at(i) << ") gave " << got.at(i)
                    << ", expected " << (expectNaN.at(i) ? "NaN" : "a non-NaN value") << "\n";
                return 1;
            }
            if (!expectNaN.at(i) && got.at(i) != ref.at(i))
            {
                std::cerr << "testFilterCollectionTruncation: filter " << i << " ("
                    << fc.filterNames().at(i) << ") gave " << got.at(i)
                    << ", expected the two-argument phot() value " << ref.at(i) << "\n";
                return 1;
            }
        }

        // No truncation: identical to the two-argument overload, with
        // Q(HI) (index 4) finite
        const auto gotFull = fc.phot(wlFull, specFull, wlFull);
        const auto refFull = fc.phot(wlFull, specFull);
        for (std::size_t i = 0; i < refFull.size(); ++i)
        {
            if (std::isnan(gotFull.at(i)) || gotFull.at(i) != refFull.at(i))
            {
                std::cerr << "testFilterCollectionTruncation: untruncated filter " << i << " ("
                    << fc.filterNames().at(i) << ") gave " << gotFull.at(i)
                    << ", expected " << refFull.at(i) << "\n";
                return 1;
            }
        }

        // Empty wl: NaN for everything overlapping the full grid,
        // i.e. everything but ideal_energy_100_400 (index 3)
        const auto gotEmpty = fc.phot({}, {}, wlFull);
        for (std::size_t i = 0; i < gotEmpty.size(); ++i)
        {
            if (i != 3 && !std::isnan(gotEmpty.at(i)))
            {
                std::cerr << "testFilterCollectionTruncation: empty wl, filter " << i << " ("
                    << fc.filterNames().at(i) << ") gave " << gotEmpty.at(i) << ", expected NaN\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterCollectionTruncation: unexpected exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Run all FilterCollection unit tests
 * @return 0 if all tests pass, positive count of failures otherwise
 */
inline auto testFilterCollection() -> int
{
    int result = 0;
    result += testFilterCollectionMixedConstruction();
    result += testFilterCollectionPhotSystemConversion();
    result += testFilterCollectionInstrumentOmitted();
    result += testFilterCollectionErrors();
    result += testFilterCollectionVega();
    result += testFilterCollectionTruncation();
    return result;
}

#endif // TESTFILTERCOLLECTION_HPP
