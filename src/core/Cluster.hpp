/**
 * @file SimPhysics.hpp
 * @author Mark Krumholz
 * @brief A class to represent a mono-age star cluster
 * @date 2026-07-13
 */

#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "SimPhysics.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/RngThread.hpp"
#include <functional>
#include <variant>
#include <vector>

namespace core
{

    /**
     * @brief A class to represent a mono-age stellar cluster
     */
    class Cluster
    {
    public:

        // Shorten type names
        using interp1d = std::vector<
            std::unique_ptr<interp::Interpolator1D<
            static_cast<size_t>(tracks::FieldIdx::nTrackQty)
            >>>;
        using tracks2d = std::variant<tracks::Tracks2D, 
            std::reference_wrapper<const tracks::Tracks2D>>;

        /**
         * @brief Initialize a cluster
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param physics Simulation physics objectg
         */
        Cluster(unsigned long uid,
            double mass,
            double time,
            const SimPhysics& physics);

        /**
         * @brief Advance the cluster in time
         * @param t The time to which to advance
         */
        void advance(double t);

        /**
         * @brief Get the stellar tracks at this cluster's [Fe/H]
         * @return A const reference to a Tracks2D object at this
         *         cluster's [Fe/H]
         * @details
         * If the simulation has a fixed [Fe/H], this returns a
         * reference to the slice shared by SimPhysics (and thus by
         * every Cluster in the simulation). Otherwise it returns a
         * reference to the slice computed for, and owned by, this
         * Cluster alone. Either way, callers can use the returned
         * Tracks2D uniformly without needing to know which case
         * applies.
         */
        [[nodiscard]] auto tracks() const -> const tracks::Tracks2D&;

    private:

        // Data set on creation
        utils::RngState rngState_;  /**< State of the rng at birth */
        unsigned long uid_;         /**< Unique identifier */
        double targetMass_;         /**< Target mass */
        double formTime_;           /**< Formation time */
        double feH_;                /**< [Fe/H] of cluster */
        std::reference_wrapper<const SimPhysics> 
            physics_;       /**< Simulation physics */

        // Masses
        std::vector<double> m_;     /**< Stellar masses */
        std::vector<double> mDead_; /**< Mass of dead stars */
        double birthMass_;          /**< Actual mass at birth */

        // Times
        double disruptTime_;        /**< Time when this cluster will disrupt */
        double curTime_;            /**< Current time */

        // Other state information
        bool isDisrupted_;          /**< Has this cluster disrupted */
        interp1d isochrone_;        /**< Isochrone for the current time */

        /**
         * Tracks for this cluster's [Fe/H]: either owned outright (when
         * the simulation has a variable [Fe/H], so each cluster needs
         * its own slice) or a reference to the slice shared via
         * SimPhysics (when [Fe/H] is fixed for the whole simulation).
         * Use tracks() rather than this member directly.
         */
        tracks2d tracks_;            /**< 2d track holder */

        /**
         * @brief Update the lists of living and dead stars
         */
        void updateLivingStars();

    };

} // namespace core

#endif // CLUSTER_HPP
