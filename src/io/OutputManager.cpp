/**
 * @file OutputManager.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManager
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "OutputManager.hpp"
#include "../utils/RngThread.hpp"
#include "SimControls.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

io::OutputManager::OutputManager(const SimControls& simControls) :
    simControls_(simControls)
{
    // Sanity check 1: if every output relevant to this simulation's
    // own SimType is disabled, nothing at all would ever be written --
    // almost certainly a mistake. write_galaxy* is only relevant for a
    // galaxy-type simulation (there is no Galaxy object, and so no
    // galaxy/galaxy_spectra/galaxy_phot group/file, for a cluster-type
    // simulation), so it is excluded from this check otherwise.
    bool anyOutput = simControls_.writeCluster() || simControls_.writeClusterSpec() ||
        simControls_.writeClusterPhot();
    if (simControls_.simType() == SimControls::SimType::galaxy)
    {
        anyOutput = anyOutput || simControls_.writeGalaxy() ||
            simControls_.writeGalaxySpec() || simControls_.writeGalaxyPhot();
    }
    if (!anyOutput)
    {
        throw std::runtime_error(
            "OutputManager: every output.write_* key relevant to this "
            "simulation is false, so nothing would ever be written");
    }

    // Sanity check 2: if photometry was requested (SimControls::filters()
    // is non-null) but both the cluster- and galaxy-photometry outputs
    // are disabled, the computed photometry would never be written
    // anywhere -- almost certainly a mistake, rather than a deliberate
    // "compute filters() for some other purpose but discard the
    // result" request.
    if (!simControls_.writeClusterPhot() && !simControls_.writeGalaxyPhot() &&
        simControls_.filters() != nullptr)
    {
        throw std::runtime_error(
            "OutputManager: phot.filters was given, but both "
            "output.write_cluster_phot and output.write_galaxy_phot are "
            "false, so the computed photometry would never be written");
    }
}

// Return the current local date (YYYY-MM-DD) and time (HH:MM:SS) as
// a pair of strings
auto io::OutputManager::currentDateAndTime() -> std::pair<std::string, std::string>
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowC = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&nowC, &tmBuf);

    std::ostringstream dateStream;
    dateStream << std::put_time(&tmBuf, "%Y-%m-%d");
    std::ostringstream timeStream;
    timeStream << std::put_time(&tmBuf, "%H:%M:%S");

    return { dateStream.str(), timeStream.str() };
}

// Return the calling thread's current rng state as a string,
// suitable for writing to disk so a run can later be reproduced
auto io::OutputManager::currentRngStateString() -> std::string
{
    const auto state = utils::rng().getState();
    return { state.data() };
}
