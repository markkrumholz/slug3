/**
 * @file main.cpp
 * @author Mark Krumholz
 * @brief This is the main routine that drives slug.
 */

#include "extern/tomlplusplus/toml.hpp"
#include "core/SimPhysics.hpp"
#include <iostream>

auto main(int argc, char *argv[]) -> int 
{
    // Ingest the input deck
    if (argc != 2)
    {
        std::cerr << "Usage: slug slug.in\n";
        exit(1);
    }
    auto inputs = toml::parse_file(argv[1]);

    // Use the input deck to initialize simulation physics
    core::SimPhysics simPhysics(inputs);
}