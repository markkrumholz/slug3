/**
 * @file SimPhysics.hpp
 * @author Mark Krumholz
 * @brief A class to hold physics settings for the simulation
 * @date 2026-07-12
 */

#ifndef SIMPHYSICS_HPP
#define SIMPHYSICS_HPP

#include "../extern/tomlplusplus/toml.hpp"
#include "../pdfs/PDF.hpp"
#include "../tracks/Tracks3D.hpp"
#include <cstdint>

namespace core
{
    
    /**
     * @class SimPhysics
     * @brief A class to hold physics settings for a simulation
     */
    class SimPhysics
    {
    public:

        /**
         * @brief An enum to hold simulation types types
         */
        enum class SimType : std::uint8_t {
            cluster, /**< Cluster simulation */
            galaxy,  /**< Galaxy simulation */
            none     /**< Dummy value */
        };

        /**
         * @brief Initialize the simulation physics settings
         * @param inputs A toml table holding the input deck
         */
        SimPhysics(const toml::table& input);

        // Getters for the physics settings
        /**
         * @brief Get simulation type
         * @return Simulation type
         */
        [[nodiscard]] auto simType() const { return simType_; }

        /**
         * @brief Get simulation initial mass function
         * @return Pointer to the simulation initial mass function
         */
        [[nodiscard]] auto imf() const -> const auto& { return imf_; }

        /**
         * @brief Get simulation cluster mass function
         * @return Pointer to the simulation cluster mass function
         */
        [[nodiscard]] auto cmf() const -> const auto& { return cmf_; }

        /**
         * @brief Get simulation [Fe/H] distribution
         * @return Pointer to the simulation [Fe/H] distribution
         */
        [[nodiscard]] auto fehDist() const -> const auto& { return fehDist_; }

        /**
         * @brief Check whether the simulation has a spread in [Fe/H]
         * @return True if the simulation has a fixed value of [Fe/H]
         */
        [[nodiscard]] auto constFeH() const { return fehDist_.getMin() == fehDist_.getMax(); }

        /**
         * @brief Get simulation star formation rate
         * @return Pointer to the simulation star formation rate
         */
        [[nodiscard]] auto sfr() const -> const auto& { return sfr_; }

        /**
         * @brief Get simulation cluster lifetime function
         * @return Pointer to the simulation cluster lifetime function
         */
        [[nodiscard]] auto clf() const -> const auto& { return clf_; }

        /**
         * @brief Get simulation stellar tracks
         * @return Pointer to the simulation stellar tracks
         */
        [[nodiscard]] auto tracks() const -> const auto& { return tracks_; }

        /**
         * @brief Get the tracks sliced to this simulation's fixed [Fe/H]
         * @return A const reference to a Tracks2D object sliced at
         *         fehDist_'s (single) value
         * @details
         * Only valid to call if constFeH() is true. This slice is
         * computed once, in the constructor, and cached for the
         * lifetime of this SimPhysics object, so that Cluster objects
         * sharing a single [Fe/H] value across a simulation can all
         * query it directly instead of each computing (or racing to
         * compute) their own copy.
         */
        [[nodiscard]] auto tracks2D() const -> const auto& { return constFeHTracks_; }

        /**
         * @brief Get minimum mass for fully stochastic treatment
         * @return Minimum mass for fully stochastic treatment
         */
        [[nodiscard]] auto minStochMass() const { return minStochMass_; }

    private:

        /**
         * @brief Load a set of tracks specified by input deck
         * @param inputDeck Name of input deck
         * @returns A Tracks3D object with the correct tracks loaded
         */
        void readTracks(const toml::table& inputDeck);

        // Physics settings
        SimType simType_;          /**< Simulation type */
        pdfs::PDF imf_;            /**< The IMF to use for the simulation */
        pdfs::PDF cmf_;            /**< Cluster mass function */
        pdfs::PDF fehDist_;        /**< [Fe/H] distribution */
        pdfs::PDF sfr_;            /**< Star formation rate */
        pdfs::PDF clf_;            /**< Cluster lifetime function */
        tracks::Tracks3D tracks_;  /**< Stellar tracks */
        tracks::Tracks2D constFeHTracks_; /**< Tracks sliced at fehDist_'s value, if constFeH() */
        double minStochMass_;      /**< Minimum mass for fully stochastic treatment */

    };

} // namespace core

#endif // SIMPHYSICS_HPP