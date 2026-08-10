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
#include <cstddef>
#include <cstdint>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <string>
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
         * WNLH60, matching it to exactly one of those files. The
         * None/WNL boundary also carries two log(Teff) conditions, not
         * part of Roy et al. 2020 itself: one that keeps a He-depleted,
         * C/O-rich WC/WO star from being misclassified as None or WNL,
         * and an absolute floor (resurrecting Georgy et al. 2012's own
         * cool-star cutoff) that keeps an ordinary cool evolved star
         * from being misclassified as WNL merely because ordinary
         * mixing has nudged its surface He mass fraction above 0.4 --
         * see getWRType for the full rule and why. WO and WNC (which
         * Georgy et al. 2012 split out separately) are folded into WC
         * here, since PoWR has no spectral models for either.
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
         * @return The star's WRType
         * @details
         * Follows Roy et al. 2020's classification scheme on surface
         * He mass fraction, extended with log(Teff) conditions at both
         * the None/WNL boundary and as an absolute floor (neither part
         * of Roy et al. 2020 itself, which does not distinguish WC
         * from WO/WNC and does not address cool stars at all), and
         * with a surface-H-mass-fraction subdivision of WNL into
         * WNLH20/WNLH40/WNLH60 (not part of Roy et al. 2020 either,
         * which treats WNL as one subtype) matching PoWR's own H20/
         * H40/H60 grid split (see fetch_powr.py and WRType's own
         * comment). The first log(Teff) extension keeps a He-depleted,
         * C/O-rich WC/WO star -- whose surface He mass fraction can
         * itself fall below 0.4 once He has been burned through to
         * C/O, despite being unambiguously a Wolf-Rayet star -- from
         * being misclassified as WRType::None, and keeps a star this
         * hot from being misclassified as a WNL subtype instead of
         * falling through to the WNE/WC check below. The second
         * extension (log(Teff) < log10(10000 K), resurrecting the
         * Georgy et al. 2012 scheme's own absolute floor) keeps an
         * ordinary cool evolved star -- whose surface He mass fraction
         * can drift somewhat above 0.4 from ordinary post-main-sequence
         * mixing, with nothing to do with Wolf-Rayet-style stripping --
         * from being misclassified as WNL: no genuine Wolf-Rayet star,
         * of any subtype, is this cool.
         *
         * A third condition, on mass rather than log(Teff), guards the
         * same log(Teff) >= log10(50000 K) escape hatch the first
         * extension above relies on: that hatch exists specifically so
         * a He-depleted, C/O-rich WC/WO star isn't misclassified as
         * WRType::None purely for having surface He mass fraction below
         * 0.4, but it makes no reference to mass, so it can equally
         * well be triggered by a low-mass, ordinary-composition
         * (H-rich) evolved star transiently passing through the same
         * hot regime -- e.g. a MIST post-AGB "phase 6" star, whose
         * effective temperature briefly exceeds 50000 K on its way to
         * becoming a white dwarf despite having nothing like a
         * stripped Wolf-Rayet envelope. Checked against MIST's own
         * grids (see the WD/hot-atmosphere-coverage project notes):
         * every genuinely WR-composition (surface He mass fraction >=
         * 0.4) star has a current mass of at least ~17 Msun, while the
         * low-mass post-AGB stars this hot-Teff escape hatch would
         * otherwise misclassify never exceed ~1.1 Msun at this phase --
         * a wide, clean gap (up to the next mass at which a genuine,
         * massive He-depleted WC/WO star appears, ~13 Msun) that an
         * 8 Msun floor sits comfortably inside. Checked sequentially:
         *   1) Mass < 8 Msun, or (surface He mass fraction < 0.4 and
         *      log(Teff) < log10(50000 K)), or log(Teff) <
         *      log10(10000 K): not a Wolf-Rayet star at all
         *      (WRType::None). The mass clause is what keeps a
         *      low-mass, hot, but ordinary-composition star from
         *      slipping through the log(Teff) escape hatch described
         *      above. The log(Teff) clause itself is what keeps a
         *      He-depleted WC/WO star from landing here: below that
         *      temperature, a genuinely evolved WR star's surface He
         *      has not yet been burned down through 0.4, so a cool,
         *      He-poor star really is just an ordinary, non-WR star.
         *      The absolute floor clause is what keeps an ordinary
         *      cool star with He mass fraction >= 0.4 from landing in
         *      a WNL subtype instead -- see this function's own
         *      comment above for both.
         *   2) Surface He mass fraction <= 0.9 and log(Teff) <
         *      log10(1e5 K): a WNL subtype, sub-classified by surface
         *      H mass fraction: < 0.3 gives WRType::WNLH20; [0.3, 0.5]
         *      gives WRType::WNLH40; > 0.5 gives WRType::WNLH60.
         *      Unlike the Georgy et al. 2012 scheme this replaced,
         *      this does not require the surface to be hydrogen-poor
         *      -- a WNL star here can retain a substantial hydrogen
         *      envelope. The log(Teff) clause keeps a star hot enough
         *      to be a WC/WO star, but with He mass fraction in this
         *      same range (see step 1's own comment), from being
         *      misclassified as a WNL subtype instead of falling
         *      through to the WNE/WC check below.
         *   3) Surface C mass fraction < surface N mass fraction:
         *      WRType::WNE (nitrogen-sequence, hydrogen-free).
         *   4) Otherwise: WRType::WC. Georgy et al. 2012 further split
         *      this remaining case into WC, WO, and WNC subtypes;
         *      PoWR has no spectral models for WO or WNC, so all
         *      three are lumped into WC here.
         */
        [[nodiscard]] static auto getWRType(const Specsyn::StarData& props) -> WRType;

        /**
         * @brief This library's log10(T_*) grid points
         * @details
         * Exposed for SpecsynLibChained's benefit, which needs to scan
         * every chained library's own logTeff_ range (alongside
         * SpecsynLibNoWind's) to derive a global (logTeffMin, logTeffMax)
         * clamp -- see its tClamp constructor argument.
         */
        [[nodiscard]] auto logTeff() const -> const std::vector<double>& { return logTeff_; }

    private:

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
    };

} // namespace specsyn

#endif // SPECSYNLIBWR_HPP
