/**
 * @file SpecsynLibWR.hpp
 * @author Mark Krumholz
 * @brief A SpecsynLib for Wolf-Rayet stars (stars with optically thick winds)
 * @date 2026-07-22
 */

#ifndef SPECSYNLIBWR_HPP
#define SPECSYNLIBWR_HPP

#include "../io/SimControls.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynLibWR
     * @brief A SpecsynLib specialization for Wolf-Rayet stars
     * @tparam Policy See SpecsynLib.
     * @details
     * Covers the Potsdam Wolf-Rayet (PoWR) model grids fetched by
     * data/tools/spectra/fetch_powr.py: powr_wne.h5, powr_wnl_h20.h5,
     * powr_wnl_h40.h5, powr_wnl_h60.h5, and powr_wc.h5. Unlike
     * SpecsynLibNoWind's (FeH, logg, Teff) tensor
     * grid, PoWR's WR atmospheres are parameterized by [Fe/H], the
     * stellar temperature, and the "transformed radius" (a function of
     * mass-loss rate and stellar radius that captures the optically
     * thick wind BOSZ/TLUSTY-style stars don't have) -- and PoWR's own
     * grid is regular in log(T_*) and log(R_t), not their linear
     * values (see fetch_powr.py's log_teff/log_rt conversion), so
     * dim1_, dim2_, dim3_ (inherited from SpecsynLib) hold FeH,
     * log(R_t), and log(Teff) respectively, aliased here as FeH_,
     * logRt_, and logTeff_ for readability.
     *
     * Unlike BOSZ/TLUSTY, where every spectrum in a library shares one
     * native wavelength grid, every individual PoWR model has its own
     * distinct wavelength sampling (see fetch_powr.py's per-dataset
     * "..._wave" companion datasets), so this class's constructor
     * resamples every populated grid point onto a single common
     * wavelength grid (via SpecsynLibChained::makeCommonWlGrid and the
     * same technique as SpecsynLib::resample) before storing them,
     * rather than reading one shared grid the way SpecsynLibNoWind
     * does.
     *
     * This is the counterpart to SpecsynLibNoWind, which instead
     * covers stars without optically thick winds (BOSZ, TLUSTY).
     *
     * PoWR's WNL grids carry an extra surface-hydrogen (xh) axis this
     * class does not model as a tensor axis. Rather than collapsing it
     * (the earlier, now-replaced approach: keep only the xh = 0.20
     * ("H20") group at each [Fe/H], which was safe only under the
     * now-replaced Georgy et al. 2012 classification scheme, which
     * never produced a WNL star with a surface H mass fraction above
     * 0.3), fetch_powr.py instead splits PoWR's WNL grids into three
     * separate files -- powr_wnl_h20.h5, powr_wnl_h40.h5,
     * powr_wnl_h60.h5, registered as POWR_WNL_H20/H40/H60 -- one per
     * xh bucket, and this class treats each as its own WRType (see
     * WRType and getWRType), constructed as its own separate
     * SpecsynLibWR instance the same way POWR_WNE/POWR_WC are. This
     * matches getWRType's Roy et al. 2020 classification, under which
     * a WNL star can retain a substantially larger hydrogen envelope
     * than Georgy et al. 2012 ever allowed, so a genuinely H-rich WNL
     * star now has a matching grid point instead of being silently
     * forced onto H20's.
     *
     * Besides the spectra themselves, the constructor also reads two
     * quantities spec() needs but that the parent SpecsynLib knows
     * nothing about: each populated grid
     * point's log10(L/Lsun) (into logL_, a per-point scalar grid
     * analogous to spectra_ but holding a single number instead of a
     * whole spectrum), and each [Fe/H] group's wind clumping density
     * contrast D_infinity (into dInf_, one value per FeH_ entry, since
     * -- unlike log_teff, log_rt, and logl -- it does not vary within
     * a group at all).
     */
    template <OOBPolicy Policy>
    class SpecsynLibWR : public SpecsynLib<Policy>
    {
    public:

        /**
         * @brief Construct a SpecsynLibWR from a spectral library on disk
         * @param spectraName Name of the spectral model (e.g.
         *   "POWR_WNE", "POWR_WNL_H40", or "POWR_WC")
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param registryName Name of the spectral library registry file
         * @param wlMin Minimum wavelength of the output grid, in
         *   Angstrom; if 0 (the default), used together with wlMax
         *   and nWl below (see @details) as a flag to fall back on
         *   the library's own native wavelength coverage
         * @param wlMax Maximum wavelength of the output grid, in
         *   Angstrom; see wlMin
         * @param nWl Number of points in the output grid; if 0 (the
         *   default), used as a flag to fall back on the library's
         *   own native wavelength grid -- see @details
         * @param controls Simulation controls; forwarded unchanged to
         *   Specsyn's own constructor -- see its comment. Has no
         *   default of its own (unlike every parameter above): a
         *   reference bound to a temporary default-constructed
         *   SimControls would dangle the moment this constructor
         *   returned, since this class stores it live rather than
         *   copying out of it.
         * @throws std::runtime_error if spectraName does not contain
         *   "wne", "wc", or "wnl" together with one of "h20"/"h40"/
         *   "h60" (case-insensitively), since type_ cannot be
         *   determined otherwise
         * @details
         * Sets type_ from spectraName (see type_'s own comment), then
         * finds every registry group matching (spectraName, fehMin,
         * fehMax) -- afe/cfe/microTurb/r do not apply to PoWR's WR
         * registry entries (see fetch_powr.py), so they are passed to
         * findMatchingSpectra at their library-wide defaults, which
         * that function ignores anyway for any group missing those
         * attributes -- and reads every populated (log(R_t),
         * log(Teff)) point's own flux and wavelength grid within each
         * matching [Fe/H] group. If nWl is 0, builds a single common
         * wavelength grid spanning every populated point's own native
         * grid; otherwise uses nWl points log-spaced from wlMin to
         * wlMax instead -- or, if wlMin is 0 there too (nWl alone was
         * requested, without an explicit range), from the global
         * minimum to the global maximum spanned by every populated
         * point's own native grid, keeping the caller's requested
         * point count. Either way, every populated point's own flux
         * is then resampled from its native wavelength grid onto that
         * common grid before being stored in spectra_.
         */
        SpecsynLibWR(
            const std::string& spectraName,
            double fehMin,
            double fehMax,
            const std::string& registryName,
            double wlMin,
            double wlMax,
            std::size_t nWl,
            const io::SimControls& controls);

        /**
         * @brief Compute a star's spectrum by trilinear interpolation on the library grid
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star; needed because it is
         *   not carried by props itself
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom; a
         *   size-0 vector if the star falls outside this library's
         *   domain and Policy is OOBPolicy::silent
         * @throws std::runtime_error if the star falls outside this
         *   library's domain and Policy is OOBPolicy::raise
         * @details
         * First classifies props via getWRType: a mismatch against
         * type_ means this library's spectra don't apply to this star
         * at all, so that alone is grounds for the OOB policy. Then
         * derives the (FeH, logRt, logTeff) point this star maps to --
         * D_infinity by linear interpolation on dInf_, a wind velocity
         * from the star's luminosity and mass-loss rate, and the
         * transformed radius from that wind velocity, D_infinity, and
         * the star's radius (Todt et al. 2015, eq. 2, one of the PoWR
         * references) -- and checks that point against FeH_ and
         * logTeff_'s ranges. logRt is instead moved (feh and logTeff
         * both left fixed) to the nearest populated grid value, since
         * PoWR's own (feh, logRt, logTeff) grids are ragged and get
         * sparser still at their hottest edge -- see spec()'s own
         * implementation comment for the full mechanism -- before
         * delegating the actual trilinear interpolation to
         * SpecsynLib::spec(double, double, double). The result is
         * finally scaled by 10^(logL_star - logLGrid),
         * with logLGrid obtained by trilinear interpolation on
         * logLGrid_ at the same point, since the model spectrum
         * stored on the grid is normalized to that model's own
         * luminosity rather than this particular star's.
         */
        [[nodiscard]] auto spec(const Specsyn::StarData& props, double feh) const
        -> std::vector<double> override;

        /**
         * @brief Compute a star's spectrum, forcing a result by moving to the nearest populated (log(R_t), log(Teff)) grid point if needed
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star; needed because it is
         *   not carried by props itself
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom --
         *   never a size-0 vector
         * @throws std::runtime_error if feh itself falls entirely
         *   outside FeH_'s own range (so not even a bracketing feh
         *   corner exists to search), or if neither of the (up to two)
         *   bracketing feh corners has any populated (log(R_t),
         *   log(Teff)) grid point at all
         * @details
         * Overrides Specsyn::specForce() -- see its own comment for
         * when SpecsynLibChained calls this rather than spec(). Unlike
         * SpecsynLibNoWind::specForce()/SpecsynLibWD::specForce(),
         * which each bracket every axis except the one being rescued
         * (log(g)) and search only that one, this class rescues along
         * *two* axes at once: spec()'s own bounds check rejects a star
         * whose log(Teff) alone falls outside logTeff_'s range, and
         * that same star's log(R_t) -- always a crude, single-scattering
         * estimate (see spec()'s own implementation comment) -- has no
         * reason to be any more trustworthy once log(Teff) already
         * is not. So rather than bracket log(Teff) and search log(R_t)
         * (or vice versa), this brackets only feh (the one axis with
         * no physical reason to be unreliable) and, at each of the (up
         * to two) bracketing feh corners, searches every populated
         * (log(R_t), log(Teff)) grid point for whichever is closest --
         * by ordinary Euclidean distance in (log(R_t), log(Teff))
         * space, both O(1)-dex quantities of comparable dynamic range,
         * so no relative weighting between them is applied -- to this
         * star's own raw (unclamped) (log(R_t), log(Teff)). Each
         * chosen point's own spectrum is rescaled by
         * 10^(logL_star - logLGrid) (logLGrid read directly off
         * logLGrid_ at that point, not trilinearly interpolated, since
         * this is a single discrete grid point rather than a bracketed
         * cell) before being weighted by the ordinary feh bilinear
         * weight and renormalized by the sum of weights actually used
         * -- the same OOBPolicy::coerce-style renormalization pattern
         * SpecsynLibNoWind::specForce()/SpecsynLibWD::specForce() use
         * for their own single-corner/blended results.
         */
        [[nodiscard]] auto specForce(const Specsyn::StarData& props, double feh) const
        -> std::vector<double> override;

        /**
         * @brief A star's Wolf-Rayet spectral subtype
         * @details
         * Follows the classification scheme of Roy et al. 2020, MNRAS,
         * 494, 3861
         * (https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.3861R/abstract),
         * which -- unlike the surface-hydrogen-gated scheme of Georgy
         * et al. 2012 this replaced -- classifies mainly on surface
         * helium (and, for the He-rich case, C/N) mass fraction, so a
         * WNL star can retain a substantial hydrogen envelope. Roy et
         * al. 2020 does not itself subdivide WNL further, but PoWR's
         * WNL grids come in three surface-hydrogen (xh) buckets --
         * H20, H40, H60 -- each its own file/registry entry (see
         * fetch_powr.py), so getWRType further sub-classifies a WNL
         * star by its own surface H mass fraction into WNLH20/WNLH40/
         * WNLH60, matching it to exactly one of those files. Unlike
         * Roy et al. 2020 itself, a candidate WNL subtype is only
         * actually assigned if the star's own log(Teff) falls within
         * that specific bucket's own PoWR grid coverage (see
         * getWRType's own comment for why this matters and what
         * happens when it doesn't). WO and WNC (which Georgy et al.
         * 2012 split out separately) are folded into WC here, since
         * PoWR has no spectral models for either.
         */
        enum class WRType : std::uint8_t
        {
            None,   /**< Not a Wolf-Rayet star */ // NOLINT(readability-identifier-naming) -- capitalized to match WNE/WNL/WC's fixed spectral-classification naming below, rather than the project's usual camelBack enum-constant convention
            WNE,    /**< Hydrogen-free nitrogen-sequence WR star */ // NOLINT(readability-identifier-naming) -- WNE is a fixed spectral-classification abbreviation; lowercasing it would make it unrecognizable
            WNLH20, /**< Nitrogen-sequence WR star, hydrogen-rich, surface H mass fraction < 0.3 (PoWR's "H20" WNL grid) */ // NOLINT(readability-identifier-naming) -- see WNE
            WNLH40, /**< Nitrogen-sequence WR star, hydrogen-rich, surface H mass fraction in [0.3, 0.5] (PoWR's "H40" WNL grid) */ // NOLINT(readability-identifier-naming) -- see WNE
            WNLH60, /**< Nitrogen-sequence WR star, hydrogen-rich, surface H mass fraction > 0.5 (PoWR's "H60" WNL grid) */ // NOLINT(readability-identifier-naming) -- see WNE
            WC      /**< Carbon-sequence WR star; also covers Georgy et al.'s WO and WNC subtypes, for which PoWR has no spectral models */ // NOLINT(readability-identifier-naming) -- see WNE
        };

        /**
         * @brief Classify a star's Wolf-Rayet spectral subtype
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param wnlTeffRanges The [min, max] log(Teff) actually
         *   spanned by each of PoWR's three WNL surface-hydrogen grids
         *   -- index 0 for WNLH20, 1 for WNLH40, 2 for WNLH60 (see
         *   wnlTeffRanges_'s own comment for how this is normally
         *   populated). A NaN entry (either bound) means that bucket's
         *   own range is unknown, so a star that would otherwise land
         *   in it never actually does -- see this function's own
         *   comment for why that matters.
         * @return The star's WRType
         * @details
         * Follows Roy et al. 2020's classification scheme on surface
         * He mass fraction, with a surface-H-mass-fraction subdivision
         * of WNL into WNLH20/WNLH40/WNLH60 (not part of Roy et al. 2020
         * itself, which treats WNL as one subtype) matching PoWR's own
         * H20/H40/H60 grid split (see fetch_powr.py and WRType's own
         * comment), and gated by wnlTeffRanges so that a star is only
         * ever assigned a WNL subtype if its own log(Teff) actually
         * falls within that subtype's real PoWR grid coverage.
         *
         * That temperature gate exists because surface composition
         * alone is a poor guide to WNL-ness at the cool end: ordinary
         * post-main-sequence mixing can nudge a star's surface He mass
         * fraction above 0.4 -- squarely inside the WNL composition
         * window -- long before it is anywhere near hot enough to
         * actually be a Wolf-Rayet star (real WNL spectra exist only
         * for O-star-and-hotter temperatures). Gating on each bucket's
         * own actual grid coverage, rather than a single fixed ceiling
         * shared by all three buckets, keeps such a star from being
         * misclassified as WNL (and then failing to find a covering
         * spectrum) merely because its composition happens to match --
         * it instead falls through to WRType::None, letting it be
         * classified as an ordinary (e.g. O-star) grid star downstream.
         * The WNE/WC branch below has no equivalent gate: a He-depleted,
         * C/O-rich WC/WO star's surface He mass fraction can itself
         * fall below 0.4 once He has been burned through to C/O,
         * despite being unambiguously a Wolf-Rayet star, so that branch
         * is reached by a log(Teff) floor alone, independent of
         * wnlTeffRanges and of the WNL composition window entirely.
         *
         * A mass floor guards that same log(Teff) >= log10(50000 K)
         * escape hatch: it exists specifically so a He-depleted,
         * C/O-rich WC/WO star isn't missed purely for having surface He
         * mass fraction below 0.4, but log(Teff) alone can equally well
         * be triggered by a low-mass, ordinary-composition (H-rich)
         * evolved star transiently passing through the same hot regime
         * -- e.g. a MIST post-AGB "phase 6" star, whose effective
         * temperature briefly exceeds 50000 K on its way to becoming a
         * white dwarf despite having nothing like a stripped
         * Wolf-Rayet envelope. Checked against MIST's own grids (see
         * the WD/hot-atmosphere-coverage project notes): every
         * genuinely WR-composition (surface He mass fraction >= 0.4)
         * star has a current mass of at least ~17 Msun, while the
         * low-mass post-AGB stars this hot-Teff escape hatch would
         * otherwise misclassify never exceed ~1.1 Msun at this phase --
         * a wide, clean gap (up to the next mass at which a genuine,
         * massive He-depleted WC/WO star appears, ~13 Msun) that an
         * 8 Msun floor sits comfortably inside. Checked sequentially:
         *   1) Mass < 8 Msun: not a Wolf-Rayet star at all
         *      (WRType::None) -- see the mass-floor discussion above.
         *   2) Surface He mass fraction in [0.4, 0.9] and log(Teff)
         *      within the log(Teff) range of the WNL bucket implied by
         *      surface H mass fraction (< 0.3 for WNLH20; [0.3, 0.5]
         *      for WNLH40; > 0.5 for WNLH60, per wnlTeffRanges[0/1/2]
         *      respectively): that WNL subtype. Unlike the Georgy et
         *      al. 2012 scheme this replaced, this does not require the
         *      surface to be hydrogen-poor -- a WNL star here can
         *      retain a substantial hydrogen envelope. If the He mass
         *      fraction condition holds but the log(Teff) gate does
         *      not (see this function's own comment above for why that
         *      can happen), falls through to step 3 rather than
         *      returning here.
         *   3) log(Teff) > log10(50000 K): WRType::WNE if surface C
         *      mass fraction < surface N mass fraction
         *      (nitrogen-sequence, hydrogen-free), else WRType::WC.
         *      Georgy et al. 2012 further split the WC case into WC,
         *      WO, and WNC subtypes; PoWR has no spectral models for
         *      WO or WNC, so all three are lumped into WC here.
         *   4) Otherwise: WRType::None.
         */
        [[nodiscard]] static auto getWRType(
            const Specsyn::StarData& props,
            const std::array<std::pair<double, double>, 3>& wnlTeffRanges) -> WRType;

        /**
         * @brief Which WR subtype this library's models are for
         * @details
         * Exposed for SpecsynLibChained's benefit, which needs to know
         * every chained SpecsynLibWR library's own type() (alongside
         * its logTeff()) to build the wnlTeffRanges getWRType needs --
         * see wnlTeffRanges_'s own comment.
         */
        [[nodiscard]] auto type() const -> WRType { return type_; }

        /**
         * @brief Tell this library the real log(Teff) range of each WNL bucket
         * @param ranges See getWRType's own wnlTeffRanges parameter
         * @details
         * Called by SpecsynLibChained once every chained library has
         * been constructed, so that this library's own spec() -- which
         * must call getWRType with the same ranges classifyGridType
         * uses, or the two could disagree about whether a given star is
         * even this library's own WRType at all -- sees the combined
         * range spanning every chained WNL library, not just
         * wnlTeffRanges_'s own single-bucket default. See
         * wnlTeffRanges_'s own comment for what happens when this is
         * never called (e.g. a SpecsynLibWR used standalone, outside
         * SpecsynLibChained, as in this class's own unit tests).
         */
        void setWNLTeffRanges(const std::array<std::pair<double, double>, 3>& ranges)
        {
            wnlTeffRanges_ = ranges;
        }

        /**
         * @brief This library's log10(T_*) grid points
         * @details
         * Exposed for SpecsynLibChained's benefit, which needs to scan
         * every chained library's own logTeff_ range (alongside
         * SpecsynLibNoWind's) to derive a global (logTeffMin, logTeffMax)
         * clamp -- see its tClamp constructor argument.
         */
        [[nodiscard]] auto logTeff() const -> const std::vector<double>& { return logTeff_; }

        /**
         * @brief The minimum [Fe/H] this library has real spectral data for
         * @return FeH_.front() -- see Specsyn::fehMin()'s own comment
         *   for why this override exists
         */
        [[nodiscard]] auto fehMin() const -> double override { return FeH_.front(); }

        /**
         * @brief The maximum [Fe/H] this library has real spectral data for
         * @return FeH_.back() -- see Specsyn::fehMin()'s own comment
         *   for why this override exists
         */
        [[nodiscard]] auto fehMax() const -> double override { return FeH_.back(); }

    private:

        /**
         * @brief Derive the transformed radius log(R_t) a star maps to
         * @param props Stellar properties; see spec()'s own props parameter
         * @param feh [Fe/H] value of the star; needed to look up D_infinity
         * @return The star's raw (unclamped, possibly outside logRt_'s
         *   own range) log(R_t), by Todt et al. 2015, eq. 2
         * @details
         * Factored out of spec() (which calls this, then moves the
         * result to the nearest populated logRt_ value) so specForce()
         * can reuse the exact same derivation without duplicating it --
         * see spec()'s own implementation comment for the full physical
         * derivation this performs (D_infinity by interpolation on
         * dInf_, wind velocity from luminosity and mass-loss rate, then
         * the transformed radius itself).
         */
        [[nodiscard]] auto computeRawLogRt(const Specsyn::StarData& props, double feh) const -> double;

        /** @brief The shape of logLGrid_, the mdspan view onto logL_ */
        using ScalarGrid = std::mdspan<double, std::dextents<std::size_t, 3>>; // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.hpp's SpectraGrid alias

        // References into the parent class's dim1_/dim2_/dim3_, named
        // for what they actually hold in this (FeH, log_rt, log_teff)
        // specialization -- see SpecsynLibNoWind's identical FeH_,
        // logg_, Teff_ pattern, and its own NOLINT justification for
        // why these are references rather than owned copies.
        std::vector<double>& FeH_;     /**< [Fe/H] values spanned by the tensor grid (alias for SpecsynLib::dim1_) */ // NOLINT(readability-identifier-naming, cppcoreguidelines-avoid-const-or-ref-data-members)
        std::vector<double>& logRt_;   /**< log10(R_t) values spanned by the tensor grid (alias for SpecsynLib::dim2_) */ // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        std::vector<double>& logTeff_; /**< log10(T_*) values spanned by the tensor grid (alias for SpecsynLib::dim3_) */ // NOLINT(readability-identifier-naming, cppcoreguidelines-avoid-const-or-ref-data-members)

        /**
         * @brief log10(L/Lsun) at each point in the (FeH, logRt, logTeff) tensor grid
         * @details
         * A flattened array of shape (FeH_.size(), logRt_.size(),
         * logTeff_.size()) -- viewed through logLGrid_ -- holding one
         * scalar value per populated grid point (unlike spectra_, this
         * is a single number per point rather than a whole spectrum).
         * Unpopulated points hold quiet_NaN, rather than being left as
         * some other sentinel, so an accidental read of an
         * unpopulated point's "value" is at least easy to notice.
         */
        std::vector<double> logL_;
        ScalarGrid logLGrid_; /**< mdspan view onto logL_, shaped (FeH_.size(), logRt_.size(), logTeff_.size()) */

        /**
         * @brief Wind clumping density contrast D_infinity at each [Fe/H]
         * @details
         * One value per FeH_ entry (dInf_.size() == FeH_.size()).
         * Unlike log_teff, log_rt, and logl, D_infinity is constant
         * across every model within a grid -- fetch_powr.py stores it
         * as a single attribute on each [Fe/H] group rather than per
         * model -- so there is no (logRt, logTeff) axis to it at all,
         * and no need for a 3D grid or an mdspan view.
         */
        std::vector<double> dInf_;

        /**
         * @brief Which WR subtype this library's models are for
         * @details
         * Set by the constructor from spectraName (e.g. "POWR_WNE"
         * gives WRType::WNE, "POWR_WNL_H40" gives WRType::WNLH40),
         * rather than read from the HDF5 file -- every model in a
         * given library is the same subtype, since fetch_powr.py
         * writes one file per subtype (powr_wne.h5, powr_wnl_h20.h5,
         * powr_wnl_h40.h5, powr_wnl_h60.h5, powr_wc.h5), so there is
         * nothing to read per group or per model. Never WRType::None:
         * the constructor throws if spectraName doesn't identify a
         * subtype at all.
         */
        WRType type_;

        /**
         * @brief The [min, max] log(Teff) range of each WNL bucket, as known to this instance
         * @details
         * Index 0 is WNLH20's own range, 1 is WNLH40's, 2 is WNLH60's --
         * see getWRType's own wnlTeffRanges parameter for how these are
         * used. Defaults to {quiet_NaN(), quiet_NaN()} at every index
         * except type_'s own (if type_ is itself one of WNLH20/WNLH40/
         * WNLH60), which the constructor instead sets to this library's
         * own logTeff_.front()/back() -- the only range this instance
         * can know on its own, without help from any sibling library --
         * so that a standalone SpecsynLibWR (e.g. in this class's own
         * unit tests, or any other use outside SpecsynLibChained) still
         * correctly recognizes stars of its own WRType. SpecsynLibChained
         * overwrites this default via setWNLTeffRanges() once every
         * chained library exists, with the combined range spanning
         * every chained WNL library -- necessary because a star of, say,
         * WNLH40 composition must be correctly told apart from one of
         * WNLH20 composition even when checked from a WNE or WC
         * library's own spec() (which needs to know it is NOT looking
         * at a WNL star at all), not just from a WNLH40 library's own.
         */
        std::array<std::pair<double, double>, 3> wnlTeffRanges_ = {{
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()},
        }};
    };

} // namespace specsyn

#endif // SPECSYNLIBWR_HPP
