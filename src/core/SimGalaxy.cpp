/**
 * @file SimGalaxy.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimGalaxy
 * @date 2026-08-10
 */

#include "SimGalaxy.hpp"
#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include "Galaxy.hpp"
#include <iostream>
#include <memory>
#include <utility>

core::SimGalaxy::SimGalaxy(const io::SimControls& simControls,
    std::unique_ptr<io::OutputManager> outputManager) :
    simControls_(simControls),
    outputManager_(std::move(outputManager))
{
}

void core::SimGalaxy::run()
{
    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: galaxy simulation starting with "
            << simControls_.nTrial() << " trials\n";
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (unsigned long trialNum = 0; trialNum < simControls_.nTrial(); ++trialNum)
    {
        if (simControls_.verbosity() > 1)
        {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                std::cout << "slug: starting trial " << trialNum << " / "
                    << simControls_.nTrial() << "\n";
            }
        }

        // Create galaxy for this trial
        Galaxy galaxy(simControls_);

        // Loop over output times, advancing the galaxy's internal
        // state and writing it out at each one -- writeGalaxy/
        // writeGalaxySpec/writeGalaxyPhot each also write out every
        // currently-alive cluster in the galaxy, so no separate
        // writeCluster/writeClusterSpec/writeClusterPhot calls are
        // needed here (unlike SimCluster::run's single writeCluster
        // call, there is no single moment when every cluster this
        // trial will ever form already exists, since new clusters
        // keep forming as the galaxy advances)
        const auto outTimes = simControls_.outTimes();
        for (const auto outTime : outTimes)
        {
            galaxy.advance(outTime);
            outputManager_->writeGalaxy(trialNum, outTime, galaxy);
            outputManager_->writeGalaxySpec(trialNum, outTime, galaxy);
            outputManager_->writeGalaxyPhot(trialNum, outTime, galaxy);
        }
    }

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: simulation complete\n";
    }
}
