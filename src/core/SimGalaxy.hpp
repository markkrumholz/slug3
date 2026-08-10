/**
 * @file SimGalaxy.hpp
 * @author Mark Krumholz
 * @brief A class to drive a galaxy-type simulation end to end
 * @date 2026-08-10
 */

#ifndef SIMGALAXY_HPP
#define SIMGALAXY_HPP

#include "../io/OutputManager.hpp"
#include "../io/SimControls.hpp"
#include <memory>

namespace core
{

    /**
     * @class SimGalaxy
     * @brief Drives a galaxy-type simulation end to end
     */
    class SimGalaxy
    {
    public:

        /**
         * @brief Initialize a galaxy simulation
         * @param simControls Simulation controls (physics settings and
         *   control-flow settings together)
         * @param outputManager Output manager to which simulation
         *   results should be written; ownership is transferred to
         *   this SimGalaxy
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this SimGalaxy.
         */
        SimGalaxy(const io::SimControls& simControls,
            std::unique_ptr<io::OutputManager> outputManager);

        // Disallow copying and moving: this object owns the output
        // manager exclusively, so duplicating or relocating it makes
        // no sense
        SimGalaxy(const SimGalaxy&) = delete;
        auto operator=(const SimGalaxy&) -> SimGalaxy& = delete;
        SimGalaxy(SimGalaxy&&) = delete;
        auto operator=(SimGalaxy&&) -> SimGalaxy& = delete;

        ~SimGalaxy() = default;

        /**
         * @brief Run the simulation
         */
        void run();

    private:

        const io::SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */
        std::unique_ptr<io::OutputManager> outputManager_; /**< Output manager */
    };

} // namespace core

#endif // SIMGALAXY_HPP
