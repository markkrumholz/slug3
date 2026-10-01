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
     * properties. For now it provides only terminal wind velocities:
     * vWindWR() for Wolf-Rayet stars, vWindOB() for O and B stars,
     * vWindAGB() for AGB stars, and vWindOther() for all other stars.
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

        /**
         * @brief Compute the terminal wind velocity of an O or B star
         * @param props Stellar properties, as for vWindWR() -- assumed
         *   to already be an O or B star; this method does not itself
         *   check that
         * @param feh The star's metallicity [Fe/H], used as a proxy
         *   for log(Z/Zsun) by both non-trivial models below
         * @return The star's terminal wind velocity, in cm/s, computed
         *   by whichever model controls_.obWindModel() selects (see
         *   @details)
         * @details
         * The model used is read live from controls_.obWindModel() on
         * every call:
         *   - OBwindModel::none_: no OB wind at all; returns exactly 0.
         *   - OBwindModel::vink01_: Vink, de Koter, & Lamers (2001),
         *     A&A, 369, 574
         *     (https://ui.adsabs.harvard.edu/abs/2001A%26A...369..574V).
         *   - OBwindModel::vinkSander21_: Vink & Sander (2021), MNRAS,
         *     504, 2051
         *     (https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2051V).
         *
         * Both non-trivial models switch between two prescriptions at
         * the bistability jump temperature of Vink et al. (2001), eqs.
         * (14) and (15),
         *
         *   log rho = -13.636 + 0.889 [Fe/H],
         *   T_jump  = 61.2 + 2.59 log rho   [kK],
         *
         * using the cool-side prescription for Teff < T_jump and the
         * hot-side one for Teff >= T_jump.
         *
         * OBwindModel::vink01_ takes v_wind as a fixed multiple of the
         * star's effective escape speed
         *
         *   v_esc = sqrt(2 G M (1 - Gamma_e) / R),
         *
         * the convention of Lamers et al. on which Vink et al.'s
         * ratios are based. Here M is the star's current mass, R its
         * radius, obtained from L and T_eff via the Stefan-Boltzmann
         * law, and Gamma_e its electron-scattering Eddington factor,
         * computed exactly as for vWindWR()'s nugisLamers00_ model
         * (fully ionized gas of the star's own surface composition).
         * A star with Gamma_e >= 1 has no effective escape speed, and
         * so gets v_wind = 0. Then
         *
         *   v_wind = 1.3 v_esc (10^[Fe/H])^0.13   [Teff <  T_jump]
         *   v_wind = 2.6 v_esc (10^[Fe/H])^0.13   [Teff >= T_jump]
         *
         * OBwindModel::vinkSander21_ uses the fits of Vink & Sander
         * (2021), eqs. (3) and (4), with v_wind in km/s, L in Lsun,
         * and Teff in K:
         *
         *   log v_wind = -7.79 - 0.07 log L + 2.57 log Teff - 0.003 [Fe/H]   [Teff <  T_jump]
         *   log v_wind =  0.39 - 0.04 log L + 0.74 log Teff + 0.19  [Fe/H]   [Teff >= T_jump]
         *
         * Vink & Sander fit eq. (4) only for Teff <= 20 kK and eq. (3)
         * only for Teff >= 25 kK; using T_jump to divide the two
         * extends each to meet the other inside the gap between them.
         * Neither model is clamped.
         */
        [[nodiscard]] auto vWindOB(const specsyn::Specsyn::StarData& props, double feh) const -> double;

        /**
         * @brief Compute the terminal wind velocity of an AGB star
         * @param props Stellar properties, as for vWindWR() -- assumed
         *   to already be an AGB star; this method does not itself
         *   check that
         * @param feh The star's metallicity [Fe/H]
         * @return The star's terminal wind velocity, in cm/s, computed
         *   by whichever model controls_.agbWindModel() selects (see
         *   @details)
         * @details
         * The model used is read live from controls_.agbWindModel()
         * on every call:
         *   - AGBwindModel::none_: no AGB wind at all; returns exactly 0.
         *   - AGBwindModel::slug2_: the prescription used by slug2,
         *     described below.
         *
         * AGBwindModel::slug2_ uses the Elitzur & Ivezic (2001, MNRAS,
         * 327, 403) scaling v_wind ~ R_gd^(-1/2) L^(1/4) for dust-driven
         * winds, where R_gd is the gas-to-dust ratio. The
         * proportionality constant comes from the empirical calibration
         * of Goldman et al. (2017, MNRAS, 465, 403), which gives
         * v_wind = 9.4 km/s at L = 10^4 Lsun and a Milky Way / Solar
         * gas-to-dust ratio. Taking R_gd to scale inversely with
         * metallicity Z = 10^[Fe/H] gives
         *
         *   v_wind = 9.4 km/s (L / 10^4 Lsun)^(1/4) (10^[Fe/H])^(1/2).
         *
         * Note that this L scaling is not what Goldman et al. find
         * empirically: this combination of the Elitzur & Ivezic
         * scaling with the Goldman et al. normalization follows a
         * recommendation from Jacco van Loon. The metallicity scaling
         * almost certainly fails for metal-poor stars, whose winds are
         * not dust-driven, but no better estimate is available for
         * that case. Not clamped.
         */
        [[nodiscard]] auto vWindAGB(const specsyn::Specsyn::StarData& props, double feh) const -> double;

        /**
         * @brief Compute the terminal wind velocity of a star not
         *   covered by vWindWR(), vWindOB(), or vWindAGB()
         * @param props Stellar properties, as for vWindWR()
         * @return The star's terminal wind velocity, in cm/s, computed
         *   by whichever model controls_.otherWindModel() selects (see
         *   @details)
         * @details
         * The model used is read live from controls_.otherWindModel()
         * on every call:
         *   - OtherwindModel::none_: no wind at all; returns exactly 0.
         *   - OtherwindModel::vesc_: the star's surface escape speed,
         *     v_wind = sqrt(2 G M / R), where M is the star's current
         *     mass and R its radius, obtained from L and T_eff via the
         *     Stefan-Boltzmann law.
         */
        [[nodiscard]] auto vWindOther(const specsyn::Specsyn::StarData& props) const -> double;

        /**
         * @brief Compute the terminal wind velocity of any star,
         *   dispatching to the appropriate per-type method
         * @param props Stellar properties, as for vWindWR()
         * @param feh The star's metallicity [Fe/H]
         * @return The star's terminal wind velocity, in cm/s
         * @details
         * Classifies the star, and dispatches to the corresponding
         * method, checking in order:
         *   1. Wolf-Rayet star, per specsyn::SpecsynLibWR::getWRType():
         *      vWindWR(). If controls_.specsyn() is a
         *      specsyn::SpecsynLibChained, getWRType() is given that
         *      library's own wnlTeffRanges() and normalLogTeffMax(), so
         *      a star is classified as WR here exactly when the
         *      spectral synthesis treats it as one; otherwise (no
         *      specsyn, or one that is not chained), both are NaN, so
         *      no star is classified as WNL (see getWRType()'s own
         *      comment).
         *   2. Teff > 11 kK: an O or B star, vWindOB().
         *   3. log g < 3.5 (g in cm/s^2, from the star's mass and its
         *      radius via the Stefan-Boltzmann law): an AGB star,
         *      vWindAGB().
         *   4. Anything else: vWindOther().
         */
        [[nodiscard]] auto vWind(const specsyn::Specsyn::StarData& props, double feh) const -> double;

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Specsyn's/Extinct's/Yields's own identical controls_ members exactly -- see any of their own comments for why. Only ever used through the same non-copyable-by-assignment pattern as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
    };

} // namespace feedback

#endif // WINDS_HPP
