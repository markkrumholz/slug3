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
#include "Cluster.hpp"
#include <iostream>
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
    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: cluster simulation starting with "
            << simControls_.nTrial() << " trials\n";
    }

    while (true)
    {
        const auto trialNum = simControls_.startTrial();
        if (trialNum == 0) { break; }

        if (simControls_.verbosity() > 2)
        {
            std::cout << "slug: starting trial " << trialNum << " / "
                << simControls_.nTrial() << "\n";
        }

        Cluster cluster(0, simPhysics_.cmf().draw(), 0, simPhysics_);

        const auto outTimes = simControls_.outTimes();
        for (const auto outTime : outTimes)
        {
            cluster.advance(outTime);
            outputManager_->writeCluster(trialNum, cluster);
        }
    }

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: simulation complete\n";
    }
}
