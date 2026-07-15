/**
 * @file SimControls.cpp
 * @author Mark Krumholz
 * @date 2026-07-16
 * @brief Implmenentation of SimControls class
 */

#include "SimControls.hpp"
#include "../extern/tomlplusplus/toml.hpp"
#include "../utils/ParseUtils.hpp"
#include <string>

core::SimControls::SimControls(const toml::table& inputDeck) :
    outputMode_(OutputMode::h5),
    nTrial_(1),
    nTrialRemain_(1)
{
    // Read output mode
    const auto outputMode = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "output_mode");
    if (outputMode.has_value())
    {
        if (outputMode.value() == "h5" || outputMode.value() == "hdf5")
        { outputMode_ = OutputMode::h5; }
        else if (outputMode.value() == "ascii" || outputMode.value() == "txt")
        { outputMode_ = OutputMode::ascii; }
        else
        {
            throw std::runtime_error("SimControls: unknown output_mode "
                 + outputMode.value());
        }
    }

    // Read number of trials
    const auto nTrialInput = utils::getTOMLKeyWithError<unsigned long>(
        inputDeck, "n_trial");
    if (nTrialInput.has_value())
    {
        nTrial_ = nTrialInput.value();
        nTrialRemain_ = nTrial_;    
    }
}