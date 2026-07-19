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
#include "../utils/UniqueIDManager.hpp"
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

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (unsigned long trialNum = 0; trialNum < simControls_.nTrial(); ++trialNum)
    {
        if (simControls_.verbosity() > 2)
        {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                std::cout << "slug: starting trial " << trialNum << " / "
                    << simControls_.nTrial() << "\n";
            }
        }

        // Create cluster for this trial
        Cluster cluster(utils::getID(), simPhysics_.cmf().draw(), 0, simPhysics_);

        // Write time-invariant cluster properties to output
        outputManager_->writeCluster(trialNum, cluster);

        // Loop over output times
        const auto outTimes = simControls_.outTimes();
        for (const auto outTime : outTimes)
        {
            cluster.advance(outTime);
        }
    }

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: simulation complete\n";
    }
}
