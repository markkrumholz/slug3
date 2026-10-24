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
#include "io/SlugVersion.hpp"
#include "utils/MPIUtils.hpp"
#include "utils/UniqueIDManager.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <toml.hpp>
#include <utility>

namespace
{
    /**
     * @brief Print the one-line usage summary
     * @param os The stream to print to
     */
    void printUsage(std::ostream& os)
    {
        os << "Usage: slug [--restart|-R] slug.in\n"
              "       slug --help|-h\n"
              "       slug --version|-V\n";
    }

    /**
     * @brief Print the usage summary, followed by a description of the
     *   arguments and where the input deck format is documented
     */
    void printHelp()
    {
        printUsage(std::cout);
        std::cout << "\n"
            "Run the slug simulation described by the TOML input deck slug.in.\n"
            "\n"
            "Options:\n"
            "  -R, --restart   resume an interrupted run from its most recent checkpoint\n"
            "                  (HDF5 output only), rather than starting over\n"
            "  -h, --help      print this help and exit\n"
            "  -V, --version   print slug's version and the git commit it was built from,\n"
            "                  and exit\n"
            "\n"
            "The input deck's parameters are documented at\n"
            "https://markkrumholz.github.io/slug3/parameters.html\n";
    }

    /**
     * @brief Print slug's version number and the git commit it was built from
     */
    void printVersion()
    {
        std::cout << "slug " << io::slugVersion << " (git " << io::slugGitHash << ")\n";
    }

    /**
     * @brief Handle the command line
     * @param args The arguments, starting with the program name
     * @param restart Set to true if --restart/-R was given ahead of the
     *   input deck, and left unchanged otherwise
     * @return An exit status if slug should exit without running a
     *   simulation -- 0 after printing help (--help/-h) or the version
     *   (--version/-V), or 1 after printing a usage error -- or
     *   std::nullopt if it should go on to run the input deck,
     *   args.back()
     * @details
     * The accepted forms are slug.in alone, --restart/-R followed by
     * slug.in, or --help/-h or --version/-V alone. Any other argument
     * starting with "-", whether alone or in the input deck's place
     * after --restart/-R, is an error rather than the name of an input
     * deck; a deck whose name starts with "-" can be given as ./-name.
     * Help and version output is the same on every MPI rank, so only
     * the I/O rank prints it.
     */
    auto parseArgs(std::span<char*> args, bool& restart) -> std::optional<int>
    {
        if (args.size() != 2 && args.size() != 3)
        {
            printUsage(std::cerr);
            return 1;
        }
        const std::string_view first = args[1];
        const bool isRestart = first == "--restart" || first == "-R";
        if (args.size() == 3)
        {
            if (isRestart)
            {
                const std::string_view deck = args[2];
                if (deck.starts_with('-'))
                {
                    std::cerr << "slug: " << first << " needs an input deck, not '" << deck << "'\n";
                    printUsage(std::cerr);
                    return 1;
                }
                restart = true;
                return std::nullopt;
            }
            std::cerr << "slug: unrecognized option '" << first << "'\n";
            printUsage(std::cerr);
            return 1;
        }
        if (first == "--help" || first == "-h")
        {
            if (utils::isIORank()) { printHelp(); }
            return 0;
        }
        if (first == "--version" || first == "-V")
        {
            if (utils::isIORank()) { printVersion(); }
            return 0;
        }
        if (isRestart)
        {
            std::cerr << "slug: " << first << " needs an input deck\n";
            printUsage(std::cerr);
            return 1;
        }
        if (first.starts_with('-'))
        {
            std::cerr << "slug: unrecognized option '" << first << "'\n";
            printUsage(std::cerr);
            return 1;
        }
        return std::nullopt;
    }
} // namespace

auto main(int argc, char *argv[]) -> int // NOLINT(bugprone-exception-escape) -- every throwing call below (SimControls, OutputManagerH5/OutputManagerAscii construction, SimCluster::run()/SimGalaxy::run()) is inside the try block below, catching std::exception by reference; verified at runtime that a genuine constructor throw (e.g. an invalid deck) is caught cleanly rather than escaping. This check's own known limitation is that it cannot see through calls into other translation units (OpenMP-parallel code among them) well enough to confirm every thrown type derives from std::exception, so it stays conservative here regardless.
{
    // Must be the very first thing main() does, and must live for its
    // entire remaining scope -- see MPIGuard's own comment for why every
    // one of this function's several exit paths is safe as a result.
    // A no-op (argc/argv left untouched) when this build was not
    // compiled with SLUG_MPI.
    const utils::MPIGuard mpiGuard(&argc, &argv);

    // Force utils::uniqueID()'s function-local static to construct here,
    // on this thread, right now -- rather than lazily, wherever the
    // first call to uniqueID() from deep inside SimCluster::run()/
    // SimGalaxy::run() happens to land, which (for a fresh, non-restart
    // run) is inside a "#pragma omp parallel for" loop body, with no
    // guarantee that lands on this same thread. UniqueIDManager's own
    // constructor reads mpiRank(), which initMPI() cached above (so it
    // makes no MPI call of its own, and would be safe from any thread
    // anyway) -- constructing it here just keeps that one-time setup
    // on the main thread, before any parallel region has had a chance
    // to open.
    // Restarting instead constructs it explicitly, just as early (see
    // OutputManagerH5::restartSetup()'s own comment), so this call is a
    // harmless no-op in that case -- constructing a C++11 function-
    // local static twice just returns the same already-built instance.
    static_cast<void>(utils::uniqueID());

    // Check arguments: either just the input deck, or an optional
    // --restart/-R flag ahead of it, requesting that this run resume
    // a previous, interrupted run from its most recent checkpoint
    // (see io::OutputManagerH5::restartSetup()'s own comment) rather
    // than starting over from trial 0; or, alone, --help/-h or
    // --version/-V (see parseArgs())
    auto args = std::span(argv, static_cast<size_t>(argc));
    bool restart = false;
    if (const auto status = parseArgs(args, restart)) { return *status; }

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

    // Everything from here on -- SimControls construction, output
    // manager construction, and actually running the simulation -- is
    // wrapped in one try/catch (mirroring the toml::parse_file one
    // above), since every one of those steps can throw (SimControls
    // validates the deck it's given; OutputManagerH5's own constructor
    // validates the restart/checkpointing/output-mode combination and
    // can fail opening its output file(s); SimCluster::run()/
    // SimGalaxy::run() rethrow a per-trial exception once every OpenMP
    // trial has finished) and none of them should be allowed to
    // propagate uncaught out of main() and call std::terminate()/
    // abort() -- exiting cleanly with a message and a nonzero status
    // instead. exitCode stays 0 for a SimType::none deck (nothing to
    // run); for cluster/galaxy decks it becomes whatever
    // SimCluster::run()/SimGalaxy::run() itself returns -- 0 for a
    // normal completion, or SimCluster::sigtermExitCode/
    // SimGalaxy::sigtermExitCode (143) if a SIGTERM was caught and the
    // run stopped early instead (see their own comments).
    int exitCode = 0;
    try
    {
        // Use the input deck to initialize simulation controls
        // (control flow and physics settings together)
        const io::SimControls simControls(inputDeck);

        // Restarting only makes sense with HDF5 output, since only
        // HDF5 output ever produces the checkpoints a restart resumes
        // from (see io::OutputManagerAscii::checkpoint()'s own
        // comment). Checked here, before constructing the output
        // manager or SimCluster/SimGalaxy, so the illegal combination
        // is always caught by this one check rather than relying on
        // io::OutputManagerH5's own constructor-time check (never
        // reached, since outputMode() == ascii routes to
        // io::OutputManagerAscii below instead) or on
        // io::OutputManagerAscii::restartTrialsDone()'s own throw
        // (never reached either, precisely because of this check).
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
            outputManager = std::make_unique<io::OutputManagerH5>(simControls, restart);
        }
        else
        {
            outputManager = std::make_unique<io::OutputManagerAscii>(simControls);
        }

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