/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow
 * @date 16-07-2026
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../extern/tomlplusplus/toml.hpp"
#include <cstdint>

namespace core
{

    /**
     * @class SimControls
     * @brief A class to hold simulation control flow information
     */
    class SimControls
    {
    public:

        /**
         * @brief An enum to hold output modes
         */
        enum class OutputMode : std::uint8_t {
            h5,      /**< HDF5 output */
            ascii   /**< ASCII output */
        };

        /**
         * @brief Initialize the simulation controls from the input deck
         * @param inputDeck A toml table holding the input deck
         */
        SimControls(const toml::table& inputDeck);

        // Getters for control flow
        /**
         * @brief Return number of trials in the simulation
         * @return Number of trials
         */
        [[nodiscard]] auto nTrial() const { return nTrial_; }
        /**
         * @brief Return number of trials remaining in the simulation
         * @return Number of trials remaining
         */
        [[nodiscard]] auto nTrialRemain() const { return nTrialRemain_; }


    private:

        // Simulation control parameters
        OutputMode outputMode_;      /**< Output mode */
        unsigned long nTrial_;       /**< Number of trials */
        unsigned long nTrialRemain_; /**< Number of trials remaining */
    };

} // namespace core

#endif // SIMCONTROLS_HPP