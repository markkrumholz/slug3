/**
 * @file OutputManager.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManager
 * @date 2026-07-16
 */

#include "OutputManager.hpp"
#include "../utils/RngThread.hpp"
#include "SimControls.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <toml.hpp>
#include <utility>

io::OutputManager::OutputManager(const SimControls& simControls, const toml::table& inputDeck) :
    simControls_(simControls),
    inputDeck_(inputDeck)
{
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
