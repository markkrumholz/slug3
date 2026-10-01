/**
 * @file Winds.hpp
 * @author Mark Krumholz
 * @brief A class to compute stellar wind feedback (e.g. terminal wind velocity)
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef WINDS_HPP
#define WINDS_HPP

#include "../specsyn/Specsyn.hpp"

namespace io
{
    class SimControls;
} // namespace io

/**
 * @brief A namespace to hold machinery for stellar feedback (winds, and eventually other channels such as ionizing/non-ionizing radiation)
 */
namespace feedback
{

    /**
     * @class Winds
     * @brief Computes stellar wind feedback quantities (e.g. terminal wind velocity) for individual stars
     * @details
     * This is the first class in the feedback subsystem, which will
     * eventually provide the full set of stellar feedback quantities
     * (wind mechanical power, ionizing/non-ionizing photon rates,
     * etc.) slug can compute from a star's interpolated track
     * properties. For now it provides only vWindWR(), the terminal
     * wind velocity of a Wolf-Rayet star.
     */
    class Winds
    {
    public:

        /**
         * @brief Construct a Winds object
         * @param controls Simulation controls; stored live for the
         *   rest of this object's lifetime (see controls_'s own
         *   comment), and read on every call for the choice of wind
         *   model (e.g. controls.wrWindModel(), see vWindWR()), so a
         *   change to it takes effect immediately. Must outlive this
         *   Winds.
         */
        explicit Winds(const io::SimControls& controls) :
            controls_(controls) { }

        // Copyable and movable (matching the C++ reference semantics
        // of controls_, which just gets rebound to the same referent
        // on copy), but assignment is deleted since a reference member
        // cannot be rebound after construction -- same pattern as
        // Specsyn's own controls_ (see its own comment)
        Winds(const Winds&) = default;
        Winds(Winds&&) = default;
        auto operator=(const Winds&) -> Winds& = delete;
        auto operator=(Winds&&) -> Winds& = delete;
        ~Winds() = default;

        /**
         * @brief Get the SimControls this Winds was constructed against
         * @return A const reference to controls_
         */
        [[nodiscard]] auto controls() const -> const io::SimControls& { return controls_; }

        /**
         * @brief Compute the terminal wind velocity of a Wolf-Rayet star
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass -- assumed to already be a Wolf-Rayet
         *   star (e.g. as classified by
         *   specsyn::SpecsynLibWR::getWRType); this method does not
         *   itself check that
         * @return The star's WR wind terminal velocity, in cm/s,
         *   computed by whichever model controls_.wrWindModel()
         *   selects (see @details)
         * @details
         * The model used is read live from controls_.wrWindModel()
         * on every call:
         *   - WRwindModel::none_: no WR wind at all; returns exactly 0.
         *   - WRwindModel::lOverc_: the single-scattering momentum
         *     limit, v_wind = L / (mdot c), with L the star's
         *     luminosity and mdot its mass-loss rate -- the same
         *     velocity SpecsynLibWR uses to derive a star's
         *     transformed radius (see its computeRawLogRt()). Not
         *     clamped.
         *   - WRwindModel::nugisLamers00_ (the default): as described
         *     below.
         *
         * WRwindModel::nugisLamers00_ implements the empirical
         * wind-velocity prescription of Nugis & Lamers (2000), A&A,
         * 360, 227
         * (https://ui.adsabs.harvard.edu/abs/2000A%26A...360..227N/abstract).
         * The star's own terminal wind speed v_wind is expressed
         * relative to its core escape speed
         *
         *   v_esc = sqrt(G M (1 - Gamma_e) / R),
         *
         * where M and R are the star's current mass and radius (R
         * obtained from L and T_eff via the Stefan-Boltzmann law,
         * since props carries no radius of its own) and Gamma_e is
         * the electron-scattering Eddington factor,
         *
         *   Gamma_e = 7.66e-5 (L/Lsun) / (M/Msun) * 0.401 (X + Y/2 + Z/4),
         *
         * with X, Y, Z the star's surface H, He, and metal (1 - X - Y)
         * mass fractions. Nugis & Lamers give two calibrations, one
         * for nitrogen-sequence (WN) stars and one for carbon-sequence
         * (WC) stars,
         *
         *   log(v_wind/v_esc) = 0.61 - 0.13 log(L) + 0.3 log(Y)          [WN]
         *   log(v_wind/v_esc) = -2.37 + 0.43 log(L) - 0.07 log(Z)        [WC]
         *
         * with L in Lsun. This method selects between the two using
         * the same surface C/N comparison SpecsynLibWR::getWRType uses
         * to distinguish its own WNE/WC subtypes (cSurf < nSurf: WN;
         * otherwise: WC) -- and, since Nugis & Lamers provide no
         * separate calibration for hydrogen-rich WNL stars, applies
         * the WN formula to those as well, on the same C/N test,
         * rather than leaving them without a wind velocity at all.
         *
         * One guard keeps this well-defined outside the regime the
         * above formulas were actually derived for: Gamma_e > 1
         * (equivalently, 1 - Gamma_e < 0 -- super-Eddington; not a
         * hypothetical case -- roughly 0.4% of Wolf-Rayet-classified
         * points across a full sweep of MIST's own tracks land here,
         * concentrated among the most massive, most metal-poor, most
         * luminous stars) makes v_esc's own radicand negative, so
         * v_esc -- and everything computed from it -- is undefined;
         * rather than let that propagate as NaN through the final
         * clamp below (std::clamp leaves a NaN argument unchanged,
         * since every comparison against NaN is false), this returns
         * the clamp's own lower bound, per instruction. (Gamma_e
         * itself can never be zero or negative in the first place: X,
         * Y, Z, L, and M are all positive-definite, so every term
         * feeding it is too.)
         * Every other input is computed from the formulas above, then
         * clamped (std::clamp) to [740, 5500] km/s, the full range of
         * terminal velocities actually present in Nugis & Lamers's own
         * WN/WC sample (their Table 1), since neither formula is
         * bounded outside the luminosity range that sample spans and
         * both can be pushed well outside it by sufficiently extreme
         * (L, Y, Z) values.
         */
        [[nodiscard]] auto vWindWR(const specsyn::Specsyn::StarData& props) const -> double;

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Specsyn's/Extinct's/Yields's own identical controls_ members exactly -- see any of their own comments for why. Only ever used through the same non-copyable-by-assignment pattern as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
    };

} // namespace feedback

#endif // WINDS_HPP
