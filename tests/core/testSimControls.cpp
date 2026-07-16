/**
 * @file testSimControls.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimControls class.
 * @date 2026-07-16
 */

#include "../src/core/SimControls.hpp"
#include "../src/extern/tomlplusplus/toml.hpp"
#include "testSimControls.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // Compare two vectors of doubles for approximate equality, reporting
    // a descriptive error through std::cerr if they don't match.
    auto checkOutTimes(const std::vector<double>& actual,
        const std::vector<double>& expected,
        const std::string& label) -> int
    {
        constexpr double tolerance = 1e-9;
        if (actual.size() != expected.size())
        {
            std::cerr << "testSimControls: " << label
                << ": outTimes() has size " << actual.size()
                << ", expected " << expected.size() << "\n";
            return 1;
        }
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const double denom = std::max(std::abs(expected[i]), 1.0);
            if (std::abs(actual[i] - expected[i]) / denom > tolerance)
            {
                std::cerr << "testSimControls: " << label
                    << ": outTimes()[" << i << "] = " << actual[i]
                    << ", expected " << expected[i] << "\n";
                return 1;
            }
        }
        return 0;
    }
} // namespace

// Verify that model_name, verbosity, and n_trial fall back to their
// documented defaults when not specified in the input deck, and that
// the start_time/end_time/ntime output option (linear spacing) is
// correctly expanded.
static auto testSimControlsDefaults() -> int
{
    const std::string fileName = "tests/core/assets/testCluster.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const core::SimControls controls(inputDeck);

        if (controls.modelName() != "slug_sim")
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default modelName() == \"slug_sim\", got "
                << controls.modelName() << "\n";
            return 1;
        }
        if (controls.verbosity() != 0)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default verbosity() == 0, got "
                << controls.verbosity() << "\n";
            return 1;
        }
        if (controls.nTrial() != 1)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected default nTrial() == 1, got "
                << controls.nTrial() << "\n";
            return 1;
        }

        return checkOutTimes(controls.outTimes(), { 0.0, 5.0, 10.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify that model_name, verbosity, and n_trial are correctly read
// from the input deck when explicitly specified.
static auto testSimControlsExplicit() -> int
{
    const std::string fileName = "tests/core/assets/testControlsExplicit.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const core::SimControls controls(inputDeck);

        if (controls.modelName() != "my_test_model")
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected modelName() == \"my_test_model\", got "
                << controls.modelName() << "\n";
            return 1;
        }
        if (controls.verbosity() != 3)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected verbosity() == 3, got "
                << controls.verbosity() << "\n";
            return 1;
        }
        if (controls.nTrial() != 7)
        {
            std::cerr << "testSimControls: " << fileName
                << ": expected nTrial() == 7, got "
                << controls.nTrial() << "\n";
            return 1;
        }

        return checkOutTimes(controls.outTimes(), { 0.0, 5.0, 10.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 1: outputs.output_time_dist draws a single output time
// from a distribution. Here the distribution is a delta function at
// 5.0, so the draw is deterministic.
static auto testSimControlsOutputTimeDist() -> int
{
    const std::string fileName = "tests/core/assets/testControlsOutputTimeDist.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const core::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 5.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 2: outputs.output_times is read back exactly as an
// explicit array of output times.
static auto testSimControlsOutputTimesArray() -> int
{
    const std::string fileName = "tests/core/assets/testControlsOutputTimesArray.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const core::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 1.0, 2.5, 9.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify option 3a: a log-spaced start_time/end_time/ntime grid produces
// a geometric sequence of output times.
static auto testSimControlsOutputTimesLog() -> int
{
    const std::string fileName = "tests/core/assets/testControlsOutputTimesLog.in";
    try
    {
        const toml::table inputDeck = toml::parse_file(fileName);
        const core::SimControls controls(inputDeck);
        return checkOutTimes(controls.outTimes(), { 1.0, 10.0, 100.0 }, fileName);
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimControls: failed to parse valid input deck "
            << fileName << ": " << error.what() << "\n";
        return 1;
    }
}

// Verify that constructing SimControls from a deck with no output time
// option at all throws.
static auto testSimControlsNoOutputs() -> int
{
    const std::string fileName = "tests/core/assets/testControlsNoOutputs.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const core::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck specifying both
// outputs.output_time_dist and outputs.output_times throws.
static auto testSimControlsOutputTimesConflict() -> int
{
    const std::string fileName = "tests/core/assets/testControlsOutputTimesConflict.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const core::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

// Verify that constructing SimControls from a deck specifying only
// outputs.start_time and outputs.end_time, without outputs.ntime, throws.
static auto testSimControlsOutputTimesPartialRange() -> int
{
    const std::string fileName = "tests/core/assets/testControlsOutputTimesPartialRange.in";
    const toml::table inputDeck = toml::parse_file(fileName);
    try
    {
        const core::SimControls controls(inputDeck);
        std::cerr << "testSimControls: " << fileName
            << ": expected construction to throw, but it succeeded\n";
        return 1;
    }
    catch (const std::runtime_error&)
    {
        return 0;
    }
}

auto testSimControls() -> int
{
    int result = 0;
    result += testSimControlsDefaults();
    result += testSimControlsExplicit();
    result += testSimControlsOutputTimeDist();
    result += testSimControlsOutputTimesArray();
    result += testSimControlsOutputTimesLog();
    result += testSimControlsNoOutputs();
    result += testSimControlsOutputTimesConflict();
    result += testSimControlsOutputTimesPartialRange();
    return result;
}
