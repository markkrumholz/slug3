/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow
 * @date 16-07-2026
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../extern/tomlplusplus/toml.hpp"
#include "../pdfs/PDF.hpp"
#include <cstdint>
#include <string>
#include <vector>

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
         * @brief Return output mode
         * @return Output mode
         */
        [[nodiscard]] auto outputMode() const { return outputMode_; }

        /**
         * @brief Return model name
         * @return Model name
         */
        [[nodiscard]] auto modelName() const { return modelName_; }

        /**
         * @brief Return verbosity level
         * @return Verbosity level
         */
        [[nodiscard]] auto verbosity() const { return verbosity_; }

         /**
         * @brief Return number of trials in the simulation
         * @return Number of trials
         */
        [[nodiscard]] auto nTrial() const { return nTrial_; }

        /**
         * @brief Return number of trials remaining in the simulation
         * @return Number of trials remaining
         * @details
         * This routine properly handles inter-thread synchronization
         * when openMP is enabled.
         */
        [[nodiscard]] auto nTrialRemain() const
        {
#ifdef _OPENMP
#pragma omp atomic
#endif
            const auto nRemain = nTrialRemain_;
            return nRemain;
        }

        /**
         * @brief Return the times at which output should occur
         * @return A vector of output times
         * @details
         * If explicit output times were specified in the input deck
         * (either directly, or as a uniformly- or log-spaced grid),
         * returns a copy of that array. If instead a distribution of
         * output times was specified, returns a single-element vector
         * containing one time drawn from that distribution.
         */
        [[nodiscard]] auto outTimes() const -> std::vector<double>
        {
            if (!outTimes_.empty()) { return outTimes_; }
            return { outTimeDist_.draw() };
        }

        // Setters
        /**
         * @brief Update the remaining trials count
         * @details
         * This routine properly handles inter-thread synchronization
         * when openMP is enabled.
         */
        [[nodiscard]] auto trialDone()
        {
#ifdef _OPENMP
#pragma omp atomic
#endif
            --nTrialRemain_;
        }


    private:

        // Simulation control parameters
        unsigned int verbosity_;       /**< Level of verbosity */
        unsigned long nTrial_;         /**< Number of trials */
        unsigned long nTrialRemain_;   /**< Number of trials remaining */
        OutputMode outputMode_;        /**< Output mode */
        std::string modelName_;        /**< Name of this model */
        std::vector<double> outTimes_; /**< Times to write output */
        pdfs::PDF outTimeDist_;        /**< Distribution of output times */

        /**
         * @brief Compute output times
         * @param inputDeck A toml table holding the input deck
         */
        void setOutputTimes(const toml::table& inputDeck);
    };

} // namespace core

#endif // SIMCONTROLS_HPP