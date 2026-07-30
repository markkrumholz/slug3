/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow
 * @date 16-07-2026
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../pdfs/PDF.hpp"
#include <cstddef>
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
         * @brief Default-construct a SimControls with all defaults
         * @details
         * Produces a SimControls with every parameter at its default
         * value (simType = none, verbosity = 0, nTrial = 1, etc.).
         * Intended for use as a default argument at Specsyn-derived
         * class constructors, so call sites that don't have an input
         * deck can still compile without a SimControls.
         */
        SimControls() = default;

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

        /**
         * @brief Return the relative tolerance for PDF integration
         * @return Relative tolerance passed to PDFIntegrator (default 1e-6)
         */
        [[nodiscard]] auto intRelTol() const { return intRelTol_; }

        /**
         * @brief Return the absolute tolerance for PDF integration
         * @return Absolute tolerance passed to PDFIntegrator (default 0)
         */
        [[nodiscard]] auto intAbsTol() const { return intAbsTol_; }

        /**
         * @brief Return the maximum number of evaluations for PDF integration
         * @return Max evaluations passed to PDFIntegrator (0 = unlimited, the default)
         */
        [[nodiscard]] auto intMaxIter() const { return intMaxIter_; }

        /**
         * @brief Set the relative tolerance for PDF integration
         * @param tol New relative tolerance
         */
        void setIntRelTol(double tol) { intRelTol_ = tol; }

        /**
         * @brief Set the absolute tolerance for PDF integration
         * @param tol New absolute tolerance
         */
        void setIntAbsTol(double tol) { intAbsTol_ = tol; }

        /**
         * @brief Set the maximum number of evaluations for PDF integration
         * @param n New maximum (0 = unlimited)
         */
        void setIntMaxIter(std::size_t n) { intMaxIter_ = n; }

    private:

        // Simulation control parameters
        SimType simType_ = SimType::none;              /**< Simulation type */
        unsigned int verbosity_ = 0;                   /**< Level of verbosity */
        unsigned long nTrial_ = 1;                     /**< Number of trials */
        OutputMode outputMode_ = OutputMode::h5;       /**< Output mode */
        std::string modelName_ = "slug_sim";           /**< Name of this model */
        std::string outDir_;                           /**< Directory into which output will be written */
        std::vector<double> outTimes_;                 /**< Times to write output */
        pdfs::PDF outTimeDist_;                        /**< Distribution of output times */
        bool outputClusters_ = true;                   /**< Whether outputs include individual clusters (galaxy sims only) */
        double intRelTol_ = 1e-6;                      /**< Relative tolerance for PDF integrator */
        double intAbsTol_ = 0.0;                       /**< Absolute tolerance for PDF integrator */
        std::size_t intMaxIter_ = 0;                   /**< Max evaluations for PDF integrator (0 = unlimited) */

        /**
         * @brief Compute output times
         * @param inputDeck A toml table holding the input deck
         */
        void setOutputTimes(const toml::table& inputDeck);
    };

} // namespace io

#endif // SIMCONTROLS_HPP