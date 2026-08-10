/**
 * @file Galaxy.hpp
 * @author Mark Krumholz
 * @brief A class to represent a galaxy, built from a time-evolving population of star clusters
 * @date 2026-08-10
 */

#ifndef GALAXY_HPP
#define GALAXY_HPP

#include "../io/SimControls.hpp"
#include "Cluster.hpp"
#include <functional>
#include <vector>

namespace core
{

    /**
     * @brief A class to represent a galaxy
     * @details
     * Unlike Cluster, which represents a single mono-age population,
     * a Galaxy represents an entire, continuously star-forming system:
     * advance() draws new clusters from SimControls::sfr()/cmf() as
     * time passes, advances every cluster formed so far (whether still
     * bound or already disrupted), and sums their individual spectra,
     * photometry, and bolometric luminosities into this Galaxy's own
     * spec()/specExtinct()/phot()/photExtinct()/lbol().
     */
    class Galaxy
    {
    public:

        /**
         * @brief Initialize a galaxy
         * @param controls Simulation controls (physics settings and
         *   control-flow/integrator-tolerance settings together);
         *   stored by reference, so it must outlive this Galaxy --
         *   see controls_'s own comment
         * @details
         * curTime_, lbol_, targetMass_, and actualMass_ all start at 0,
         * and clusters_/disruptedClusters_/spec_/specExtinct_/phot_/
         * photExtinct_ all start empty -- no clusters exist, and no
         * spectrum/photometry has been computed, until the first call
         * to advance().
         */
        explicit Galaxy(const io::SimControls& controls);

        // Observers

        /**
         * @brief Return the galaxy's current time
         * @return Current simulation time, in yr
         */
        [[nodiscard]] auto curTime() const { return curTime_; }

        /**
         * @brief Return the galaxy's currently-alive (non-disrupted) clusters
         * @return A const reference to the list of clusters formed so
         *   far that have not yet disrupted
         */
        [[nodiscard]] auto clusters() const -> const auto& { return clusters_; }

        /**
         * @brief Return the galaxy's disrupted clusters
         * @return A const reference to the list of clusters formed so
         *   far that have already disrupted
         */
        [[nodiscard]] auto disruptedClusters() const -> const auto& { return disruptedClusters_; }

        /**
         * @brief Return the galaxy's continuously-sampled spectrum
         * @return A const reference to the sum of spec() over every
         *   cluster in clusters() and disruptedClusters(), on the
         *   wavelength grid of the simulation's spectral synthesizer,
         *   or an empty vector if no spectral synthesizer was
         *   requested (SimControls::specsyn() is null)
         */
        [[nodiscard]] auto spec() const -> const auto& { return spec_; }

        /**
         * @brief Return the galaxy's extincted spectrum
         * @return A const reference to the sum of specExtinct() over
         *   every cluster in clusters() and disruptedClusters(), on
         *   the wavelength grid returned by SimControls::extinct()'s
         *   own wl(); an empty vector if no extinction curve was
         *   requested (SimControls::extinct() is null) or spec()
         *   itself is empty (no spectral synthesizer was requested)
         */
        [[nodiscard]] auto specExtinct() const -> const auto& { return specExtinct_; }

        /**
         * @brief Return the galaxy's photometry
         * @return A const reference to the photometric value computed
         *   from spec() by each filter in SimControls::filters(), in
         *   the same order as FilterCollection::filterNames()/
         *   filterUnits(), or an empty vector if no filter collection
         *   was requested (SimControls::filters() is null)
         */
        [[nodiscard]] auto phot() const -> const auto& { return phot_; }

        /**
         * @brief Return the galaxy's extincted photometry
         * @return A const reference to the photometric value computed
         *   from specExtinct() by each filter in
         *   SimControls::filters(), in the same order as phot(); an
         *   empty vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or no filter collection
         *   was requested (SimControls::filters() is null)
         */
        [[nodiscard]] auto photExtinct() const -> const auto& { return photExtinct_; }

        /**
         * @brief Return the galaxy's bolometric luminosity
         * @return The sum of lbol() over every cluster in clusters()
         *   and disruptedClusters(), in Lsun, at the current time, or
         *   0 if advance() has never run with
         *   SimControls::computeLbol() true
         */
        [[nodiscard]] auto lbol() const { return lbol_; }

        /**
         * @brief Return the total target mass of clusters formed so far
         * @return The sum, over every advance() call so far, of the
         *   target mass (sfr().integral() over that call's own
         *   (curTime(), t] step) of clusters that should have formed,
         *   in Msun -- may differ from actualMass() due to stochastic
         *   sampling from SimControls::cmf() (see Galaxy::advance())
         */
        [[nodiscard]] auto targetMass() const { return targetMass_; }

        /**
         * @brief Return the total actual mass of clusters formed so far
         * @return The sum, over every advance() call so far, of the
         *   target mass (Cluster::targetMass()) of every cluster
         *   actually drawn from SimControls::cmf() during that call,
         *   in Msun
         */
        [[nodiscard]] auto actualMass() const { return actualMass_; }

        /**
         * @brief Advance the galaxy in time
         * @param t Time to which to advance, in yr; must be >= curTime()
         * @details
         * Draws and forms new clusters over (curTime(), t] from
         * SimControls::sfr()/cmf(), accumulating the target and actual
         * mass of that step's new clusters into targetMass()/
         * actualMass(), advances every cluster formed so far (in
         * clusters() and disruptedClusters()) to t, moves any cluster
         * that disrupted during this step from clusters() to
         * disruptedClusters(), then recomputes spec()/specExtinct()/
         * phot()/photExtinct()/lbol() as sums over the resulting
         * cluster population, before finally updating curTime() to t.
         */
        void advance(double t);

    private:

        double curTime_ = 0.0;                  /**< Current simulation time */
        std::vector<Cluster> clusters_;          /**< Currently alive (non-disrupted) clusters */
        std::vector<Cluster> disruptedClusters_; /**< Disrupted clusters */

        std::vector<double> spec_;         /**< Sum of spec() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> specExtinct_;  /**< Sum of specExtinct() over every cluster in clusters_/disruptedClusters_, at the current time */
        std::vector<double> phot_;         /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_;  /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;                /**< Sum of lbol() over every cluster in clusters_/disruptedClusters_, at the current time */
        double targetMass_ = 0.0;          /**< Cumulative target mass of clusters formed so far, over every advance() call, in Msun */
        double actualMass_ = 0.0;          /**< Cumulative actual mass of clusters formed so far, over every advance() call, in Msun */

        /**
         * @brief Simulation controls (physics and control-flow settings) this galaxy was built from
         * @details
         * See Cluster::controls_'s own comment: read live wherever a
         * physics setting or integrator tolerance is needed, rather
         * than snapshotted at construction.
         */
        std::reference_wrapper<const io::SimControls> controls_;

        /**
         * @brief Update spec_ (and specExtinct_) from the current cluster population
         * @details
         * Mirrors Cluster::computeSpec()'s own null-guard, but sums
         * over clusters rather than stars: does nothing if
         * SimControls::specsyn() is null. Otherwise sets spec_ to the
         * sum of spec() over every cluster in clusters_ and
         * disruptedClusters_; if SimControls::extinct() is also
         * non-null, also sets specExtinct_ to the sum of specExtinct()
         * over the same clusters.
         */
        void computeSpec();

        /**
         * @brief Update phot_ (and photExtinct_) from the current spec_ (and specExtinct_)
         * @details
         * Identical to Cluster::computePhot(), but reading this
         * Galaxy's own spec_/specExtinct_ rather than a Cluster's.
         */
        void computePhot();

        /**
         * @brief Update lbol_ from the current cluster population
         * @details
         * Sets lbol_ to the sum of lbol() over every cluster in
         * clusters_ and disruptedClusters_.
         */
        void computeLbol();

    };

} // namespace core

#endif // GALAXY_HPP
