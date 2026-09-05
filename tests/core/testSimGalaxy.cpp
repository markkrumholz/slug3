/**
 * @file testSimGalaxy.cpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimGalaxy class.
 * @date 2026-08-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "../src/core/SimGalaxy.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerAscii.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "testSimGalaxy.hpp"
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#ifdef _OPENMP
#   include <omp.h>
#endif // _OPENMP

static constexpr unsigned long nTrial = 8;
static constexpr int nThreads = 4;

// Build a galaxy-simulation input deck, reusing the deck already
// exercised by testSimControls's own galaxy-parsing tests, with
// model_name, out_dir, and n_trial injected so it can drive a real
// end-to-end run -- mirrors testSimCluster.cpp's own makeInputDeck.
static auto makeInputDeck(const std::string& modelName,
    const std::filesystem::path& outDir) -> toml::table
{
    toml::table inputDeck = toml::parse_file("tests/core/assets/testGalaxy.in");
    if (toml::table* outputTbl = inputDeck["output"].as_table())
    {
        outputTbl->insert("model_name", modelName);
        outputTbl->insert("out_dir", outDir.string());
    }
    else
    {
        inputDeck.insert("output", toml::table{
            { "model_name", modelName }, { "out_dir", outDir.string() } });
    }
    inputDeck.insert("n_trial", static_cast<int64_t>(nTrial));
    return inputDeck;
}

// Parse the deck, build SimControls/OutputManager/SimGalaxy (mirroring
// main.cpp's end-to-end setup), and run, forcing real multi-threaded
// execution so SimGalaxy::run's parallel for loop actually spans
// multiple threads -- mirrors testSimCluster.cpp's own runEndToEnd.
static void runEndToEnd(const toml::table& inputDeck)
{
    const io::SimControls simControls(inputDeck);

    std::unique_ptr<io::OutputManager> outputManager;
    if (simControls.outputMode() == io::SimControls::OutputMode::h5)
    {
        outputManager = std::make_unique<io::OutputManagerH5>(simControls);
    }
    else
    {
        outputManager = std::make_unique<io::OutputManagerAscii>(simControls);
    }

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#endif // _OPENMP

    core::SimGalaxy simGalaxy(simControls, std::move(outputManager));
    if (simGalaxy.trialsCompleted() != 0)
    {
        throw std::runtime_error("testSimGalaxy: trialsCompleted() should start at 0");
    }
    simGalaxy.run();
    if (simGalaxy.trialsCompleted() != simControls.nTrial())
    {
        throw std::runtime_error(
            "testSimGalaxy: trialsCompleted() should equal nTrial() after run() completes");
    }
}

// There is no output to check yet -- SimGalaxy::run() only updates
// each trial's Galaxy in memory (see its own comment) -- so this just
// verifies that a real galaxy-type input deck can be parsed and run
// end to end, across multiple trials and threads, without crashing or
// throwing. Once output writing is wired up (a future commit), this
// test can be expanded to check the resulting output the way
// testSimCluster.cpp already does for cluster simulations.
auto testSimGalaxy() -> int
{
    const auto outDir = std::filesystem::temp_directory_path() / "slugTestSimGalaxy";
    std::filesystem::remove_all(outDir);
    std::filesystem::create_directories(outDir);

    try
    {
        runEndToEnd(makeInputDeck("test_sim_galaxy", outDir));
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSimGalaxy: end-to-end run failed: "
            << error.what() << "\n";
        return 1;
    }
    return 0;
}
