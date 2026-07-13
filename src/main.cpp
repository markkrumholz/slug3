/**
 * @file main.cpp
 * @author Mark Krumholz
 * @brief This is the main routine that drives slug.
 */

#include "extern/tomlplusplus/toml.hpp"
#include "core/SimPhysics.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>

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
    toml::table inputs;
    try
    {
        inputs = toml::parse_file(args.back());
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to parse input file " 
            << args.back() << ": " << e.what() << '\n';
        return 1;
    }
 
    // Use the input deck to initialize simulation physics
    const core::SimPhysics simPhysics(inputs);
}