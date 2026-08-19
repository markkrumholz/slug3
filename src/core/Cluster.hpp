/**
 * @file Cluster.hpp
 * @author Mark Krumholz
 * @brief A class to represent a mono-age star cluster
 * @date 2026-07-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../utils/RngThread.hpp"
#include <array>
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
        using Segment = interp::Interpolator1D<
            static_cast<size_t>(tracks::FieldIdx::nTrackQty)>;
        using Interp1dPtr = std::vector<std::unique_ptr<Segment>>;
        using Track2DVar = std::variant<tracks::Tracks2D,
            std::reference_wrapper<const tracks::Tracks2D>>;

        /**
         * @brief Initialize a cluster
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param controls Simulation controls (physics settings and
         *   control-flow/integrator-tolerance settings together);
         *   stored by reference, so it must outlive this Cluster --
         *   see controls_'s own comment
         */
        Cluster(unsigned long uid,
            double mass,
            double time,
            const io::SimControls& controls);

        /**
         * @brief Reconstruct a cluster's stellar masses from a previously recorded rng state
         * @param uid Unique ID of cluster
         * @param mass Target cluster mass
         * @param time Cluster formation time
         * @param controls Simulation controls; see the primary
         *   constructor's own comment
         * @param rngState The rng state to draw star masses from --
         *   typically one previously returned by rngState() (e.g. read
         *   back from an output file), so that the resulting
         *   starMasses()/birthMass() come out bitwise identical to the
         *   original cluster's
         * @details
         * Sets every member the same way the primary constructor does,
         * except for feH_, aV_, m_, and birthMass_: rngState_ is set to
         * rngState directly, since that is this cluster's own birth
         * state by construction, rather than one captured live.
         * feH_, aV_ (if SimControls::avDist() is valid), and m_ are
         * then all drawn together, in the same order the primary
         * constructor draws them, from rngState rather than the live
         * rng stream -- saving the live state first and restoring it
         * immediately afterward, so this constructor has no lasting
         * effect on the ambient rng stream. All of these draws, not
         * just m_'s, must be replayed from rngState for the result
         * to come out bitwise identical to the original cluster's:
         * PDF::draw() consumes rng state (via a
         * std::discrete_distribution used to pick a segment) even for
         * a single-segment/delta [Fe/H] distribution, so drawing feH_
         * (and aV_) live first would already have advanced the stream
         * before m_'s draw ever saw rngState, decoupling the draws
         * from the sequence that originally produced them. birthMass_
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
            const io::SimControls& controls,
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
         * @brief Return the cluster's V-band extinction
         * @return A_V, in magnitudes, drawn from
         *   SimControls::avDist() at construction, or 0 if
         *   SimControls::avDist() is not valid (extinct.AV was not
         *   given in the input deck)
         */
        [[nodiscard]] auto aV() const { return aV_; }

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
         * reference to the slice shared by SimControls (and thus by
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
         *   synthesizer was requested (SimControls::specsyn() is
         *   null)
         * @details
         * Computed lazily: if advance() has run since spec_/specExtinct_
         * were last computed, this triggers computeSpec() (which
         * computes both together) before returning, so the result is
         * always current as of the last advance() -- see specCurrent_'s
         * own comment. Not const, since it may need to run that
         * computation.
         */
        [[nodiscard]] auto spec() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return spec_;
        }

        /**
         * @brief Return the cluster's extincted spectrum
         * @return A const reference to spec(), attenuated by aV()
         *   through SimControls::extinct(), on the wavelength grid
         *   returned by SimControls::extinct()'s own wl(); an empty
         *   vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or spec() itself is
         *   empty (no spectral synthesizer was requested)
         * @details
         * Computed lazily -- see spec()'s own comment; both share the
         * same specCurrent_ flag, since a single computeSpec() call
         * computes both together.
         */
        [[nodiscard]] auto specExtinct() -> const auto&
        {
            if (!specCurrent_) { computeSpec(); specCurrent_ = true; }
            return specExtinct_;
        }

        /**
         * @brief Return the cluster's photometry
         * @return A const reference to the photometric value computed
         *   from spec() by each filter in SimControls::filters(), in
         *   the same order as FilterCollection::filterNames()/
         *   filterUnits(), or an empty vector if no filter collection
         *   was requested (SimControls::filters() is null)
         * @details
         * Computed lazily -- see spec()'s own comment; computePhot()
         * itself calls spec()/specExtinct() (rather than reading
         * spec_/specExtinct_ directly), so calling phot() alone, with
         * no prior call to spec(), still computes a spectrum current
         * as of the last advance() first.
         */
        [[nodiscard]] auto phot() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return phot_;
        }

        /**
         * @brief Return the cluster's extincted photometry
         * @return A const reference to the photometric value computed
         *   from specExtinct() by each filter in
         *   SimControls::filters(), in the same order as phot(); an
         *   empty vector if no extinction curve was requested
         *   (SimControls::extinct() is null) or no filter collection
         *   was requested (SimControls::filters() is null)
         * @details
         * Computed lazily -- see phot()'s own comment; both share the
         * same photCurrent_ flag, since a single computePhot() call
         * computes both together.
         */
        [[nodiscard]] auto photExtinct() -> const auto&
        {
            if (!photCurrent_) { computePhot(); photCurrent_ = true; }
            return photExtinct_;
        }

        /**
         * @brief Return the cluster's bolometric luminosity
         * @return The population's total bolometric luminosity, in
         *   Lsun, at the current time, or 0 if SimControls::computeLbol()
         *   is false (Lbol was never requested)
         * @details
         * Computed lazily -- see spec()'s own comment. computeLbol()
         * itself is a no-op (leaving lbol_ at 0) unless
         * SimControls::computeLbol() is true, mirroring computeSpec()/
         * computePhot()'s own null-guards.
         */
        [[nodiscard]] auto lbol()
        {
            if (!lbolCurrent_) { computeLbol(); lbolCurrent_ = true; }
            return lbol_;
        }

        /**
         * @brief Return whether the cluster has disrupted
         * @return True if the cluster has disrupted
         */
        [[nodiscard]] auto isDisrupted() const { return isDisrupted_; }

        /**
         * @brief Advance the cluster in time
         * @param t Time to which to advance, in yr
         * @details
         * Updates the living/dead star lists and the isochrone for the
         * new time, then marks spec_/specExtinct_/phot_/photExtinct_/
         * lbol_ as stale (see specCurrent_/photCurrent_/lbolCurrent_'s
         * own comments) rather than recomputing them itself -- they are
         * instead recomputed lazily, on demand, the next time spec()/
         * specExtinct()/phot()/photExtinct()/lbol() is actually called.
         */
        void advance(double t);
        
    private:

        // Data set on creation
        utils::RngState rngState_;  /**< State of the rng at birth */
        unsigned long uid_;         /**< Unique identifier */
        double targetMass_;         /**< Target mass */
        double formTime_;           /**< Formation time */
        double feH_;                /**< [Fe/H] of cluster */
        double aV_;                 /**< V-band extinction, in magnitudes, drawn from SimControls::avDist() (0 if that PDF is not valid) */

        /**
         * @brief Simulation controls (physics and control-flow settings) this cluster was built from
         * @details
         * Read live wherever this cluster needs a physics setting or
         * an integrator tolerance (e.g. computeLbol()'s own
         * utils::PDFIntegrator), rather than snapshotted at
         * construction -- so a change to controls_'s own tolerances
         * after this Cluster is built takes effect the next time it
         * integrates, with no need to rebuild the Cluster.
         */
        std::reference_wrapper<const io::SimControls> controls_;

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
        std::vector<double> specExtinct_; /**< spec_ attenuated by aV_ through SimControls::extinct(), at the current time */
        std::vector<double> phot_;  /**< Photometry of spec_ through each filter in SimControls::filters(), at the current time */
        std::vector<double> photExtinct_; /**< Photometry of specExtinct_ through each filter in SimControls::filters(), at the current time */
        double lbol_ = 0.0;         /**< Bolometric luminosity of the population, in Lsun, at the current time */

        /**
         * @brief Whether spec_/specExtinct_ are current as of curTime_
         * @details
         * True at construction (spec_/specExtinct_'s own empty
         * in-class defaults are already the correct, current value
         * before advance() has ever run), and after every spec()/
         * specExtinct() call recomputes them; set back to false at the
         * end of every advance() call, so the next spec()/specExtinct()
         * call recomputes them lazily -- see advance()'s and spec()'s
         * own comments. Letting spec()/phot()/lbol() compute only when
         * actually requested, rather than unconditionally inside
         * advance() itself, avoids e.g. computing this cluster's own
         * photometry (an unwanted expense) when a galaxy this cluster
         * belongs to needs only this cluster's spectrum, to build the
         * galaxy's own summed spectrum, but per-cluster photometry
         * output was disabled (output.write_cluster_phot = false).
         */
        bool specCurrent_ = true;

        /**
         * @brief Whether phot_/photExtinct_ are current as of curTime_
         * @details
         * Mirrors specCurrent_'s own comment, for phot()/photExtinct().
         */
        bool photCurrent_ = true;

        /**
         * @brief Whether lbol_ is current as of curTime_
         * @details
         * Mirrors specCurrent_'s own comment, for lbol().
         */
        bool lbolCurrent_ = true;

        /**
         * Tracks for this cluster's [Fe/H]: either owned outright (when
         * the simulation has a variable [Fe/H], so each cluster needs
         * its own slice) or a reference to the slice shared via
         * SimControls (when [Fe/H] is fixed for the whole simulation).
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
         * Does nothing if SimControls::specsyn() is null (no spectral
         * synthesizer was requested). Otherwise sets spec_ to the sum
         * of the continuously-sampled (non-stochastic) part of the
         * population, if any, and each individually-sampled
         * (stochastic) star in m_.
         */
        void computeSpec();

        /**
         * @brief Update phot_ (and photExtinct_) from the current spec_ (and specExtinct_)
         * @details
         * Does nothing if SimControls::filters() is null (no filter
         * collection was requested). Otherwise sets phot_ from spec()
         * (the lazy getter, not spec_ directly, so that calling phot()
         * alone -- with no prior call to spec() -- still computes a
         * current spectrum first, regardless of call order); if
         * SimControls::extinct() is also non-null, also sets
         * photExtinct_ from specExtinct() likewise.
         */
        void computePhot();

        /**
         * @brief Update lbol_ from the current isochrone and star lists
         * @details
         * Does nothing if SimControls::computeLbol() is false (Lbol was
         * never requested), mirroring computeSpec()/computePhot()'s own
         * null-guards. Otherwise mirrors computeSpec()'s own two-part
         * structure: sums 10^logL (logL read directly off the isochrone
         * via Segment's single-quantity operator()(x, idx), rather than
         * interpolating every quantity) over each individually-sampled
         * (stochastic) star in m_ with a valid isochrone segment, then
         * adds the continuously-sampled (non-stochastic) part of the
         * population's own contribution, integrated against the IMF
         * over each isochrone segment via lbolStar() and
         * utils::PDFIntegrator -- the same per-segment integration
         * Specsyn::specCts() uses, and for the same reason (an
         * isochrone may have gaps between segments that the quadrature
         * routine has no way to know to avoid).
         */
        void computeLbol();

        /**
         * @brief Bolometric luminosity of a single star, given its mass and isochrone segment
         * @param m Stellar mass, in Msun; must lie within segment's
         *   valid domain (segment.xMin() <= m <= segment.xMax())
         * @param segment A single isochrone segment (one element of
         *   the Isochrone returned by Tracks2D::getIsochrone) to
         *   evaluate at mass m
         * @return 10^logL at mass m, in Lsun, wrapped in a
         *   single-element array
         * @details
         * Exists so computeLbol() can hand it to utils::PDFIntegrator,
         * which expects a callable taking the integration variable
         * (here, mass) as its first argument and returning a fixed- or
         * dynamically-sized container of doubles -- mirroring
         * Specsyn::specWl()'s own role for specCts(), but wrapped in a
         * std::array<double, 1> rather than a per-wavelength vector,
         * since there is only one quantity (Lbol) here, not one per
         * wavelength. Static since it needs no instance state, which
         * lets computeLbol() hand PDFIntegrator a plain function
         * pointer rather than a pointer to member function.
         */
        [[nodiscard]] static auto lbolStar(double m, const Segment& segment) -> std::array<double, 1>;

    };

} // namespace core

#endif // CLUSTER_HPP
