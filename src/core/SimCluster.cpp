/**
 * @file SimCluster.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimCluster
 * @date 2026-07-17
 */

#include "SimCluster.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include "../io/SimPhysics.hpp"
#include <memory>
#include <utility>

core::SimCluster::SimCluster(const io::SimControls& simControls,
    const io::SimPhysics& simPhysics,
    std::unique_ptr<io::OutputManager> outputManager) :
    simControls_(simControls),
    simPhysics_(simPhysics),
    outputManager_(std::move(outputManager))
{
}

void core::SimCluster::run()
{
    // Implementation to be added in a future PR
}
