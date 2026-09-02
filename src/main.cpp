/**
 * @file main.cpp
 * @author Mark Krumholz
 * @brief This is the main routine that drives slug.
 * @date 14-07-2026
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "core/SimCluster.hpp"
#include "core/SimGalaxy.hpp"
#include "io/OutputManager.hpp"
#include "io/OutputManagerAscii.hpp"
#include "io/OutputManagerH5.hpp"
#include "io/SimControls.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <toml.hpp>
#include <utility>

auto main(int argc, char *argv[]) -> int
{
    // Check arguments: either just the input deck, or an optional
    // --restart/-R flag ahead of it, requesting that this run resume
    // a previous, interrupted run from its most recent checkpoint
    // (see io::OutputManagerH5::restartSetup()'s own comment) rather
    // than starting over from trial 0
    if (argc != 2 && argc != 3)
    {
        std::cerr << "Usage: slug [--restart|-R] slug.in\n";
        return 1;
    }
    auto args = std::span(argv, static_cast<size_t>(argc));
    bool restart = false;
    if (argc == 3)
    {
        const std::string flag = args[1];
        if (flag == "--restart" || flag == "-R") { restart = true; }
        else
        {
            std::cerr << "Usage: slug [--restart|-R] slug.in\n";
            return 1;
        }
    }

    // Parse input file
    toml::table inputDeck;
    try
    {
        inputDeck = toml::parse_file(args.back());
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to parse input file "
            << args.back() << ": " << e.what() << '\n';
        return 1;
    }

    // Use the input deck to initialize simulation controls (control
    // flow and physics settings together)
    const io::SimControls simControls(inputDeck);

    // Restarting only makes sense with HDF5 output, since only HDF5
    // output ever produces the checkpoints a restart resumes from
    // (see io::OutputManagerAscii::checkpoint()'s own comment).
    // Checked here, before constructing the output manager or
    // SimCluster/SimGalaxy, so the illegal combination is always
    // caught by this one check rather than relying on
    // io::OutputManagerH5's own constructor-time check (never
    // reached, since outputMode() == ascii routes to
    // io::OutputManagerAscii below instead) or on
    // io::OutputManagerAscii::restartTrialsDone()'s own throw (never
    // reached either, precisely because of this check).
    if (restart && simControls.outputMode() == io::SimControls::OutputMode::ascii)
    {
        std::cerr << "slug: --restart/-R is not supported with ascii output\n";
        return 1;
    }

    // Construct the output manager
    std::unique_ptr<io::OutputManager> outputManager;
    if (simControls.outputMode() == io::SimControls::OutputMode::h5 ||
        simControls.outputMode() == io::SimControls::OutputMode::h5divided)
    {
        outputManager = std::make_unique<io::OutputManagerH5>(simControls, inputDeck, restart);
    }
    else
    {
        outputManager = std::make_unique<io::OutputManagerAscii>(simControls, inputDeck);
    }

    // Run the simulation. Wrapped in its own try/catch (mirroring the
    // toml::parse_file one above) so a genuine simulation failure --
    // e.g. SimCluster::run()/SimGalaxy::run() rethrowing a per-trial
    // exception once every OpenMP trial has finished -- exits cleanly
    // with a message and a nonzero status instead of propagating
    // uncaught out of main() and calling std::terminate()/abort().
    // exitCode stays 0 for a SimType::none deck (nothing to run);
    // for cluster/galaxy decks it becomes whatever SimCluster::run()/
    // SimGalaxy::run() itself returns -- 0 for a normal completion, or
    // SimCluster::sigtermExitCode/SimGalaxy::sigtermExitCode (143) if a
    // SIGTERM was caught and the run stopped early instead (see their
    // own comments).
    int exitCode = 0;
    try
    {
        if (simControls.simType() == io::SimControls::SimType::cluster)
        {
            core::SimCluster simCluster(simControls, std::move(outputManager), restart);
            exitCode = simCluster.run();
        }
        else if (simControls.simType() == io::SimControls::SimType::galaxy)
        {
            core::SimGalaxy simGalaxy(simControls, std::move(outputManager), restart);
            exitCode = simGalaxy.run();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "slug: simulation failed: " << e.what() << '\n';
        return 1;
    }

    return exitCode;
}