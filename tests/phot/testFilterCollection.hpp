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
 */

#ifndef TESTFILTERCOLLECTION_HPP
#define TESTFILTERCOLLECTION_HPP

#include "../../src/phot/FilterCollection.hpp"
#include "../../src/phot/FilterIdeal.hpp"
#include "../../src/phot/FilterTabulated.hpp"
#include "../../src/phot/PhotCommons.hpp"
#include "../../src/utils/Constants.hpp"
#include "../../src/utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <numbers>
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

    // Convert a Filter::phot() luminosity-like value to the flux
    // observed at the standard distance of 10 pc, matching
    // FilterCollection's own convertFlambda() -- ST/AB/Vega
    // magnitudes are defined in terms of a flux, not a luminosity
    auto luminosityToFlux(const double value) -> double
    {
        constexpr double pi = std::numbers::pi_v<double>;
        constexpr double tenPc = 10.0 * utils::pc;
        return value / (4.0 * pi * tenPc * tenPc);
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
            "erg/(s Angstrom)", "erg/(s Angstrom)", "photons/s", "photons/s"};
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
 * "mag(AB)" for the two energy-flux filters and "photons/s" (unchanged)
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
            "mag(AB)", "mag(AB)", "photons/s", "photons/s"};
        if (gotUnits != expectedUnits)
        {
            std::cerr << "testFilterCollectionPhotSystemConversion: filterUnits() mismatch\n";
            return 1;
        }

        const phot::FilterTabulated refTab("SLUGTEST", "CAM1", "G500", registryName);
        const phot::FilterIdeal refIdealEnergy("ideal_energy_700_1500");
        const phot::FilterIdeal refIdealPhot("ideal_phot_700_1500");
        const phot::FilterIdeal refQHI("Q(HI)");

        const std::vector<double> expected = {
            phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(
                luminosityToFlux(refTab.phot(wl, spec)), refTab.wlPivot()),
            phot::PhotConvert<phot::PhotSystem::Flambda, phot::PhotSystem::AB>(
                luminosityToFlux(refIdealEnergy.phot(wl, spec)), refIdealEnergy.wlPivot()),
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
    return result;
}

#endif // TESTFILTERCOLLECTION_HPP
