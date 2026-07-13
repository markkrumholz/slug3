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

    private:

        // Physics settings
        SimType simType_;          /**< Simulation type */
        pdfs::PDF imf_;            /**< The IMF to use for the simulation */
        pdfs::PDF cmf_;            /**< Cluster mass function */
        pdfs::PDF fehDist_;        /**< [Fe/H] distribution */
        pdfs::PDF sfr_;            /**< Star formation rate */
        pdfs::PDF clf_;            /**< Cluster lifetime function */
        tracks::Tracks3D tracks_;  /**< Stellar tracks */

    };

} // namespace core

#endif // SIMPHYSICS_HPP