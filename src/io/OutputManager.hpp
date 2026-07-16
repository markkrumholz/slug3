/**
 * @file OutputManager.hpp
 * @author Mark Krumholz
 * @brief A class to manage writing simulation outputs to disk
 * @date 2026-07-16
 */

#ifndef OUTPUTMANAGER_HPP
#define OUTPUTMANAGER_HPP

#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <fstream>
#include <toml.hpp>

namespace io
{

    /**
     * @class OutputManager
     * @brief A class to manage writing simulation outputs to disk
     * @tparam Mode The output format this instantiation writes
     * @details
     * The ascii and HDF5 output formats differ enough that almost every
     * member function needs its own implementation for each, so rather
     * than branching on Mode inside a single shared implementation, this
     * class is declared here only as a template (with no generic
     * definition), and is instead fully specialized below for each of
     * the two values of SimControls::OutputMode. Only those two
     * specializations exist; instantiating OutputManager for any other
     * value is a compile error.
     */
    template <SimControls::OutputMode Mode>
    class OutputManager;

    /**
     * @brief Ascii-output specialization of OutputManager
     */
    template <>
    class OutputManager<SimControls::OutputMode::ascii>
    {
    public:

        /**
         * @brief Open the output file and write its header
         * @param simControls Simulation control flow settings
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManager.
         */
        OutputManager(const SimControls& simControls, const toml::table& inputDeck);

        /**
         * @brief Close the cluster output file, if it was opened
         */
        ~OutputManager();

        // Disallow copying and moving: this object represents exclusive
        // ownership of on-disk output, so duplicating it makes no sense
        OutputManager(const OutputManager&) = delete;
        auto operator=(const OutputManager&) -> OutputManager& = delete;
        OutputManager(OutputManager&&) = delete;
        auto operator=(OutputManager&&) -> OutputManager& = delete;

    private:

        const SimControls& simControls_; /**< Simulation control flow settings */
        const toml::table& inputDeck_;   /**< The simulation's toml input deck */
        std::ofstream clustersFile_;     /**< Handle to the open cluster output file */
    };

    /**
     * @brief HDF5-output specialization of OutputManager
     */
    template <>
    class OutputManager<SimControls::OutputMode::h5>
    {
    public:

        /**
         * @brief Open the output file and write its header
         * @param simControls Simulation control flow settings
         * @param inputDeck The simulation's toml input deck
         * @details
         * simControls and inputDeck are stored by reference, so the
         * objects passed in must outlive this OutputManager.
         */
        OutputManager(const SimControls& simControls, const toml::table& inputDeck);

        /**
         * @brief Close the output file
         */
        ~OutputManager();

        // Disallow copying and moving: this object owns an open HDF5
        // file handle, so duplicating it would lead to a double close
        OutputManager(const OutputManager&) = delete;
        auto operator=(const OutputManager&) -> OutputManager& = delete;
        OutputManager(OutputManager&&) = delete;
        auto operator=(OutputManager&&) -> OutputManager& = delete;

    private:

        const SimControls& simControls_; /**< Simulation control flow settings */
        const toml::table& inputDeck_;   /**< The simulation's toml input deck */
        hid_t file_ = -1; /**< Handle to the open HDF5 output file */ // NOLINT(misc-include-cleaner)
        hid_t clustersGroup_ = -1; /**< Handle to the open clusters group, if any */ // NOLINT(misc-include-cleaner)
    };

} // namespace io

#endif // OUTPUTMANAGER_HPP
