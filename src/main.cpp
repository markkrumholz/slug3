/**
 * @file main.cpp
 * @author Mark Krumholz
 * @brief This is the main routine that drives slug.
 * @date 14-07-2026
 */

#include "io/OutputManager.hpp"
#include "io/SimControls.hpp"
#include "io/SimPhysics.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <toml.hpp>

auto main(int argc, char *argv[]) -> int 
{
    // Check arguments
    if (argc != 2)
    {
        std::cerr << "Usage: slug slug.in\n";
        return 1;
    }
    auto args = std::span(argv, static_cast<size_t>(argc));

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

    // Use the input deck to initialize simulation control flow
    // and physics
    const io::SimControls simControls(inputDeck);
    const io::SimPhysics simPhysics(inputDeck, simControls.simType());

    // Construct the output manager
    if (simControls.outputMode() == io::SimControls::OutputMode::h5)
    {
        const io::OutputManager<io::SimControls::OutputMode::h5> outputManager(simControls, inputDeck);
    }
    else
    {
        const io::OutputManager<io::SimControls::OutputMode::ascii> outputManager(simControls, inputDeck);
    }

    // Run the simulation
    if (simControls.simType() == io::SimControls::SimType::cluster)
    {
    }
    else if (simControls.simType() == io::SimControls::SimType::galaxy)
    {
        // Galaxy simulation support will be added in a future PR
    }
}