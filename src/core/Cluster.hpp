/**
 * @file Cluster.hpp
 * @author Mark Krumholz
 * @brief A class to represent a mono-age star cluster
 * @date 2026-07-13
 */

#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimPhysics.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/RngThread.hpp"
#include <cstddef>
#include <functional>
#include <memory>
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
        using Interp1dPtr = std::vector<
            std::unique_ptr<interp::Interpolator1D<
            static_cast<size_t>(tracks::FieldIdx::nTrackQty)
            >>>;
        using Track2DVar = std::variant<tracks::Tracks2D, 
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
            const io::SimPhysics& physics);

        /**
         * @brief Reconstruct a cluster's stellar masses from a previously recorded rng state
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param physics Simulation physics object
         * @param rngState The rng state to draw star masses from --
         *   typically one previously returned by rngState() (e.g. read
         *   back from an output file), so that the resulting
         *   starMasses()/birthMass() come out bitwise identical to the
         *   original cluster's
         * @details
         * Sets every member the same way the primary constructor does,
         * except for feH_, m_, and birthMass_: rngState_ is set to
         * rngState directly, since that is this cluster's own birth
         * state by construction, rather than one captured live.
         * feH_ and m_ are then both drawn together, in the same order
         * the primary constructor draws them, from rngState rather
         * than the live rng stream -- saving the live state first and
         * restoring it immediately afterward, so this constructor has
         * no lasting effect on the ambient rng stream. Both draws,
         * not just m_'s, must be replayed from rngState for the result
         * to come out bitwise identical to the original cluster's:
         * PDF::draw() consumes rng state (via a
         * std::discrete_distribution used to pick a segment) even for
         * a single-segment/delta [Fe/H] distribution, so drawing feH_
         * live first would already have advanced the stream before
         * m_'s draw ever saw rngState, decoupling the two draws from
         * the sequence that originally produced them. birthMass_
         * (which starts at 0.0, overwritten once m_ is drawn) is then
         * reduced from m_ before m_ is sorted, matching the primary
         * constructor's own order (there, forced by computing
         * birthMass_ in the member initializer list, which runs
         * before the constructor body's own sort) -- reducing in a
         * different order would reproduce the same set of masses but
         * not necessarily the same birthMass_ bit-for-bit, since
         * floating-point addition is not associative.
         */
        Cluster(unsigned long uid,
            double mass,
            double time,
            const io::SimPhysics& physics,
            const utils::RngState& rngState);

        // Observers

        /**
         * @brief Return the state of the rng at this cluster's birth
         * @return A const reference to the serialized rng state (see
         *   utils::RngThread::getState()) in effect immediately before
         *   this cluster's own stochastic draws, sufficient to exactly
         *   reproduce its starMasses()/birthMass() later via the
         *   rngState-accepting constructor overload
         */
        [[nodiscard]] auto rngState() const -> const utils::RngState& { return rngState_; }

        /**
         * @brief Return the cluster's unique identifier
         * @return Unique ID of cluster
         */
        [[nodiscard]] auto uid() const { return uid_; }

        /**
         * @brief Return the cluster's target mass
         * @return Target cluster mass in Msun
         */
        [[nodiscard]] auto targetMass() const { return targetMass_; }

        /**
         * @brief Return the cluster's actual mass at birth
         * @return Actual cluster mass at birth in Msun
         */
        [[nodiscard]] auto birthMass() const { return birthMass_; }

        /**
         * @brief Return the cluster's formation time
         * @return Cluster formation time
         */
        [[nodiscard]] auto formTime() const { return formTime_; }

        /**
         * @brief Return the cluster's [Fe/H]
         * @return [Fe/H] of cluster
         */
        [[nodiscard]] auto feH() const { return feH_; }

        /**
         * @brief Return the current list of living stellar masses
         * @return Masses of currently alive stars in Msun
         */
        [[nodiscard]] auto starMasses() const -> const auto& { return m_; }

        /**
         * @brief Return the list of dead stellar masses
         * @return Masses of dead stars in Msun
         */
        [[nodiscard]] auto deadStarMasses() const -> const auto& { return mDead_; }

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

        /**
         * @brief Return the cluster's continuously-sampled spectrum
         * @return A const reference to the spectrum of the
         *   non-stochastically-sampled part of the population, on
         *   the wavelength grid of the simulation's spectral
         *   synthesizer, or an empty vector if no spectral
         *   synthesizer was requested (SimPhysics::specsyn() is
         *   null)
         */
        [[nodiscard]] auto spec() const -> const auto& { return spec_; }

        /**
         * @brief Return the cluster's photometry
         * @return A const reference to the photometric value computed
         *   from spec() by each filter in SimPhysics::filters(), in
         *   the same order as FilterCollection::filterNames()/
         *   filterUnits(), or an empty vector if no filter collection
         *   was requested (SimPhysics::filters() is null)
         */
        [[nodiscard]] auto phot() const -> const auto& { return phot_; }

        /**
         * @brief Return whether the cluster has disrupted
         * @return True if the cluster has disrupted
         */
        [[nodiscard]] auto isDisrupted() const { return isDisrupted_; }

        /**
         * @brief Advance the cluster in time
         * @param t Time to which to advance, in yr
         */
        void advance(double t);
        
    private:

        // Data set on creation
        utils::RngState rngState_;  /**< State of the rng at birth */
        unsigned long uid_;         /**< Unique identifier */
        double targetMass_;         /**< Target mass */
        double formTime_;           /**< Formation time */
        double feH_;                /**< [Fe/H] of cluster */
        std::reference_wrapper<const io::SimPhysics>
            physics_;       /**< Simulation physics */

        // Masses
        std::vector<double> m_;     /**< Stellar masses */
        std::vector<double> mDead_; /**< Mass of dead stars */
        double birthNonStochMass_;  /**< Mass in non-stochastic part of IMF at birth */
        double birthMass_;          /**< Actual mass at birth */

        // Times
        double disruptTime_;        /**< Time when this cluster will disrupt */
        double curTime_;            /**< Current time */

        // Other state information
        bool isDisrupted_ = false;  /**< Has this cluster disrupted */
        bool advanced_ = false;     /**< Has advance() ever run its body (as opposed to a same-time no-op)? */
        Interp1dPtr isochrone_;     /**< Isochrone for the current time */
        std::vector<double> spec_;  /**< Spectrum of the continuously-sampled part of the population at the current time */
        std::vector<double> phot_;  /**< Photometry of spec_ through each filter in SimPhysics::filters(), at the current time */

        /**
         * Tracks for this cluster's [Fe/H]: either owned outright (when
         * the simulation has a variable [Fe/H], so each cluster needs
         * its own slice) or a reference to the slice shared via
         * SimPhysics (when [Fe/H] is fixed for the whole simulation).
         * Use tracks() rather than this member directly.
         */
        Track2DVar tracks_;         /**< 2d track holder */

        /**
         * @brief Update the lists of living and dead stars
         * @param logAge log10 of the cluster age in yr
         */
        void updateLivingStars(double logAge);

        /**
         * @brief Update spec_ from the current isochrone and star lists
         * @details
         * Does nothing if SimPhysics::specsyn() is null (no spectral
         * synthesizer was requested). Otherwise sets spec_ to the sum
         * of the continuously-sampled (non-stochastic) part of the
         * population, if any, and each individually-sampled
         * (stochastic) star in m_.
         */
        void computeSpec();

    };

} // namespace core

#endif // CLUSTER_HPP
