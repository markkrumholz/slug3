/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow
 * @date 16-07-2026
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../pdfs/PDF.hpp"
#include <cstdint>
#include <string>
#include <toml.hpp>
#include <vector>

namespace io
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
         * @brief An enum to hold simulation types
         */
        enum class SimType : std::uint8_t {
            cluster, /**< Cluster simulation */
            galaxy,  /**< Galaxy simulation */
            none     /**< Dummy value */
        };

        /**
         * @brief Initialize the simulation controls from the input deck
         * @param inputDeck A toml table holding the input deck
         */
        SimControls(const toml::table& inputDeck);

        // Getters for control flow

        /**
         * @brief Return simulation type
         * @return Simulation type
         */
        [[nodiscard]] auto simType() const { return simType_; }

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
         * @brief Return output directory
         * @return Output directory
         * @details
         * This is the directory into which output will be written.
         * An empty string (the default) means output will be written
         * into the current working directory.
         */
        [[nodiscard]] auto outDir() const { return outDir_; }

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
            unsigned long nRemain{};
#ifdef _OPENMP
#pragma omp atomic read
#endif
            nRemain = nTrialRemain_;
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

        /**
         * @brief Return whether outputs include individual clusters
         * @return True if outputs include individual clusters
         * @details
         * In a galaxy-type simulation, this controls whether the
         * outputs include individual clusters, or only the integrated
         * properties of the entire simulated galaxy. This is only read
         * from the input deck for a galaxy-type simulation; it defaults
         * to true otherwise.
         */
        [[nodiscard]] auto outputClusters() const { return outputClusters_; }

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
        SimType simType_ = SimType::none;              /**< Simulation type */
        unsigned int verbosity_ = 0;                   /**< Level of verbosity */
        unsigned long nTrial_ = 1;                     /**< Number of trials */
        unsigned long nTrialRemain_ = 1;               /**< Number of trials remaining */
        OutputMode outputMode_ = OutputMode::h5;       /**< Output mode */
        std::string modelName_ = "slug_sim";           /**< Name of this model */
        std::string outDir_;                           /**< Directory into which output will be written */
        std::vector<double> outTimes_;                 /**< Times to write output */
        pdfs::PDF outTimeDist_;                        /**< Distribution of output times */
        bool outputClusters_ = true;                   /**< Whether outputs include individual clusters (galaxy sims only) */

        /**
         * @brief Compute output times
         * @param inputDeck A toml table holding the input deck
         */
        void setOutputTimes(const toml::table& inputDeck);
    };

} // namespace io

#endif // SIMCONTROLS_HPP