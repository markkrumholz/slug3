/**
 * @file testFilterTabulated.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the FilterTabulated class.
 * @details
 * The registry-based tests use a small synthetic filter registry and
 * HDF5 file stored under tests/phot/assets so that they can run
 * without access to the full-size filter data under data/filters,
 * which is too large to store in the repository -- see
 * data/tools/make_filter_test_fixture.py for how that fixture was
 * generated.
 * @date 2026-07-27
 */

#ifndef TESTFILTERTABULATED_HPP
#define TESTFILTERTABULATED_HPP

#include "../../src/phot/FilterTabulated.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Unit test for FilterTabulated's direct (wl, response) constructor
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Uses a 3-point constant-response ("top hat") filter -- exactly at
 * Interpolator1D's Steffen minimum size, so it silently falls back to
 * linear interpolation (see Interpolator1D::interpInit's own fallback
 * condition), making response_(u) == 1 exactly everywhere in its
 * domain -- together with a 3-point constant spectrum, so that both
 * norm() and phot() have exact closed-form expected values (rather
 * than merely approximate ones), checked to floating-point precision.
 */
inline auto testFilterTabulatedDirect() -> int
{
    const std::vector<double> wl = {3000.0, 6000.0, 9000.0};
    const std::vector<double> response = {1.0, 1.0, 1.0};
    constexpr double wlPivot = 6000.0;
    const phot::FilterTabulated filt("test_filter", wl, response, wlPivot);

    if (filt.name() != "test_filter")
    {
        std::cerr << "testFilterTabulatedDirect: expected name() to be "
            "\"test_filter\", got \"" << filt.name() << "\"\n";
        return 1;
    }
    if (filt.photCount())
    {
        std::cerr << "testFilterTabulatedDirect: expected photCount() to be "
            "false for a FilterTabulated, but it was true\n";
        return 1;
    }
    if (filt.wlPivot() != wlPivot)
    {
        std::cerr << "testFilterTabulatedDirect: expected wlPivot() to be "
            << wlPivot << ", got " << filt.wlPivot() << "\n";
        return 1;
    }

    // norm() = integral of R(lambda) = 1 with respect to ln(lambda)
    // over [3000, 9000] = ln(9000 / 3000) exactly
    constexpr double relTol = 1e-9;
    const double expectedNorm = std::log(9000.0 / 3000.0);
    const double normRelErr = std::abs(filt.norm() - expectedNorm) / expectedNorm;
    if (normRelErr > relTol)
    {
        std::cerr << "testFilterTabulatedDirect: norm() gave " << filt.norm()
            << ", expected " << expectedNorm << " (relative error "
            << normRelErr << ", tolerance " << relTol << ")\n";
        return 1;
    }

    // A constant spectrum F_lambda = F0 should give phot() == F0
    // exactly, since dividing by norm() cancels R's own normalization
    constexpr double f0 = 4.0;
    const std::vector<double> spec = {f0, f0, f0};
    const double result = filt.phot(wl, spec);
    const double photRelErr = std::abs(result - f0) / f0;
    if (photRelErr > relTol)
    {
        std::cerr << "testFilterTabulatedDirect: phot() gave " << result
            << ", expected " << f0 << " (relative error " << photRelErr
            << ", tolerance " << relTol << ")\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for FilterTabulated's registry-based constructor
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Loads the synthetic SLUGTEST/CAM1/G500 filter from
 * tests/phot/assets/filters_test.toml (a 25-point Gaussian response,
 * dense enough that Interpolator1D actually uses its default Steffen
 * spline rather than falling back to linear interpolation, matching
 * the shape of a real tabulated filter response), then checks that a
 * constant spectrum F_lambda = F0, evaluated on a wavelength grid
 * spanning the filter's own domain, recovers phot() == F0 to within a
 * generous tolerance. This is not an exact closed-form check (norm_
 * uses Interpolator1D's own spline integral of the fitted response
 * curve, while phot()'s product integral instead resamples that same
 * response only at its own original grid points and integrates their
 * product with the spectrum piecewise-linearly -- see
 * Interpolator1D::integ's own comments -- so the two are not bound to
 * agree to machine precision for a genuinely curved, many-point
 * response), but it is still a strong regression check: an error in
 * the ln(wavelength) change of variables, a units mistake, or a wrong
 * axis would each produce a discrepancy many orders of magnitude
 * larger than the tolerance used here.
 */
inline auto testFilterTabulatedRegistry() -> int
{
    const std::string registryName = "tests/phot/assets/filters_test.toml";

    try
    {
        const phot::FilterTabulated filt("SLUGTEST", "CAM1", "G500", registryName);

        if (filt.name() != "SLUGTEST.CAM1.G500")
        {
            std::cerr << "testFilterTabulatedRegistry: expected name() to be "
                "\"SLUGTEST.CAM1.G500\", got \"" << filt.name() << "\"\n";
            return 1;
        }
        if (filt.photCount())
        {
            std::cerr << "testFilterTabulatedRegistry: expected photCount() "
                "to be false for a FilterTabulated, but it was true\n";
            return 1;
        }
        if (filt.norm() <= 0.0)
        {
            std::cerr << "testFilterTabulatedRegistry: expected norm() > 0, "
                "got " << filt.norm() << "\n";
            return 1;
        }
        // wl_ref = 5000.0 in tests/phot/assets/filters_test.toml's
        // [SLUGTEST.CAM1.G500] entry
        if (filt.wlPivot() != 5000.0)
        {
            std::cerr << "testFilterTabulatedRegistry: expected wlPivot() to "
                "be 5000.0, got " << filt.wlPivot() << "\n";
            return 1;
        }

        constexpr double relTol = 1e-4;
        constexpr double f0 = 3.0;
        constexpr int n = 5000;
        std::vector<double> wl(n);
        const std::vector<double> spec(n, f0);
        for (int i = 0; i < n; ++i)
        {
            wl[static_cast<size_t>(i)] = 2000.0 + (8000.0 - 2000.0) *
                static_cast<double>(i) / static_cast<double>(n - 1);
        }

        const double result = filt.phot(wl, spec);
        const double relErr = std::abs(result - f0) / f0;
        if (relErr > relTol)
        {
            std::cerr << "testFilterTabulatedRegistry: phot() gave " << result
                << ", expected " << f0 << " (relative error " << relErr
                << ", tolerance " << relTol << ")\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testFilterTabulatedRegistry: failed to construct "
            "FilterTabulated from " << registryName << ": " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test for FilterTabulated's registry constructor error handling
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Checks that constructing a FilterTabulated from a nonexistent
 * registry, or with a facility/instrument/filter combination not
 * present in tests/phot/assets/filters_test.toml, throws rather than
 * silently succeeding.
 */
inline auto testFilterTabulatedRegistryErrors() -> int
{
    const std::string registryName = "tests/phot/assets/filters_test.toml";

    struct Case
    {
        std::string label;
        std::string facility;
        std::string instrument;
        std::string filter;
        std::string registry;
    };
    const std::vector<Case> cases = {
        {"nonexistent registry", "SLUGTEST", "CAM1", "G500",
            "tests/phot/assets/no_such_registry.toml"},
        {"unknown facility", "NOSUCHFACILITY", "CAM1", "G500", registryName},
        {"unknown instrument", "SLUGTEST", "NOSUCHINSTRUMENT", "G500", registryName},
        {"unknown filter", "SLUGTEST", "CAM1", "NOSUCHFILTER", registryName},
    };

    for (const auto& c : cases)
    {
        try
        {
            const phot::FilterTabulated filt(c.facility, c.instrument, c.filter, c.registry);
            std::cerr << "testFilterTabulatedRegistryErrors: expected an "
                "exception for case '" << c.label << "', but none was thrown\n";
            return 1;
        }
        catch (const std::exception&) { /* this is the expected outcome */ }
    }

    return 0;
}

/**
 * @brief Unit tests for the FilterTabulated class
 * @return 0 if the test passes, 1 if it fails.
 */
inline auto testFilterTabulated() -> int
{
    int result = 0;
    result += testFilterTabulatedDirect();
    result += testFilterTabulatedRegistry();
    result += testFilterTabulatedRegistryErrors();
    return result;
}

#endif // TESTFILTERTABULATED_HPP
