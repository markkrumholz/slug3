/**
 * @file SpecsynLibWD.hpp
 * @author Mark Krumholz
 * @brief A SpecsynLib2D for white dwarf atmosphere grids
 * @date 2026-08-07
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef SPECSYNLIBWD_HPP
#define SPECSYNLIBWD_HPP

#include "../io/SimControls.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib2D.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynLibWD
     * @brief A SpecsynLib2D specialization for white dwarf (and similarly log(g)/Teff-only) atmosphere grids
     * @tparam Policy See SpecsynLib.
     * @details
     * Covers spectral libraries -- like the Tremblay et al. pure-
     * hydrogen (DA) white dwarf grids fetched by
     * data/tools/spectra/fetch_tremblay.py, or the Rauch et al. NLTE
     * hot star grid fetched by data/tools/spectra/fetch_rauch.py --
     * whose spectra are parameterized by (log(g), Teff) alone, with
     * no [Fe/H]/[alpha/Fe]/[C/Fe] axis at all, stored as four flat
     * top-level HDF5 datasets:
     * "wl", "logg", "log_Teff", and "flux". "flux" may be either a
     * filled (n_logg, n_logTeff, n_wl) tensor (Tremblay's grids,
     * already a full rectangular grid as distributed) or a (n_models,
     * n_wl) array alongside per-model "logg"/"log_Teff" values that
     * don't together form a filled grid (Rauch's grid, missing some
     * (log g, Teff) combinations at its hottest Teff values) -- the
     * constructor detects which from "flux"'s own rank and dispatches
     * to readFilledTensorGrid or readSparseGrid accordingly (see
     * SpecsynLibWD.cpp). Either way this differs from the
     * one-group-per-[Fe/H] layout SpecsynLibNoWind reads. dim2_, dim3_
     * (inherited from SpecsynLib, via SpecsynLib2D) hold log(g) and
     * log(Teff) respectively, aliased here as logg_ and logTeff_ for
     * readability -- mirroring SpecsynLibNoWind's identical
     * FeH_/logg_/logTeff_ aliasing exactly. dim1_ is left empty (see
     * SpecsynLib2D's own comment): this class has no first axis at
     * all, not even a degenerate one populated with a single
     * placeholder value.
     */
    template <OOBPolicy Policy>
    class SpecsynLibWD : public SpecsynLib2D<Policy>
    {
    public:

        /**
         * @brief Construct a SpecsynLibWD from a white dwarf atmosphere grid on disk
         * @param spectraName Name of the spectral model
         * @param registryName Name of the spectral library registry file
         * @param wlMin Minimum wavelength of the output grid, in
         *   Angstrom; if 0 (the default), used together with wlMax
         *   and nWl below (see @details) as a flag to fall back on
         *   the library's own native wavelength grid
         * @param wlMax Maximum wavelength of the output grid, in
         *   Angstrom; see wlMin
         * @param nWl Number of points in the output grid; if 0 (the
         *   default), used as a flag to fall back on the library's
         *   own native wavelength grid -- see @details
         * @param controls Simulation controls; forwarded unchanged to
         *   SpecsynLib2D's own constructor -- see its comment. Has no
         *   default of its own (unlike every parameter above): a
         *   reference bound to a temporary default-constructed
         *   SimControls would dangle the moment this constructor
         *   returned, since this class stores it live rather than
         *   copying out of it.
         * @throws std::runtime_error if spectraName is not found in
         *   the registry, or its HDF5 file cannot be read, or its
         *   "flux" dataset's shape does not match its "wl"/"logg"/
         *   "log_Teff" datasets' own sizes
         * @details
         * Unlike SpecsynLibNoWind/SpecsynLibWR, this constructor takes
         * no [Fe/H]/[alpha/Fe]/[C/Fe]/microturbulence/resolution
         * arguments at all: white dwarf atmospheres here have none of
         * those axes, and the registry's "file" entry names the one
         * and only HDF5 file this model reads from directly (no
         * per-[Fe/H] group filtering, unlike findMatchingSpectra).
         *
         * If nWl is nonzero, resamples onto nWl points log-spaced from
         * wlMin to wlMax (via SpecsynLib::resample) after reading the
         * library's own native wavelength grid; if wlMin is 0 there
         * (nWl alone was requested, without an explicit range), wlMin/
         * wlMax instead fall back to the front/back of that just-read
         * native grid, keeping the caller's requested point count. If
         * nWl is 0, wl() simply returns the native grid as read from
         * disk, unresampled. Mirrors SpecsynLibNoWind's own identical
         * resampling logic exactly.
         */
        SpecsynLibWD(
            const std::string& spectraName,
            const std::string& registryName,
            double wlMin,
            double wlMax,
            std::size_t nWl,
            const io::SimControls& controls);

        /**
         * @brief Compute a star's spectrum by bilinear interpolation on the library grid
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star; unused, since this
         *   library has no [Fe/H] axis at all -- present only because
         *   the Specsyn base class's own spec() signature requires it
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom; a
         *   size-0 vector if the star falls outside this library's
         *   (logg, Teff) grid and Policy is OOBPolicy::silent
         * @throws std::runtime_error if the star falls outside this
         *   library's grid and Policy is OOBPolicy::raise
         * @details
         * Derives the star's log(Teff) directly from props and checks
         * it against logTeff_'s own range; then derives log(g) (and
         * surface area) via Specsyn::getSAandLogg and checks that
         * against logg_'s own range. If both checks pass, delegates
         * the actual bilinear interpolation to
         * SpecsynLib2D::spec(double, double) and scales the result by
         * the star's surface area to convert specific flux at the
         * surface into specific luminosity.
         */
        [[nodiscard]] auto spec(const Specsyn::StarData& props, double feh) const
        -> std::vector<double> override;

        /**
         * @brief Compute a star's spectrum, forcing a result by moving log(g) to the nearest populated grid value if needed
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star; unused, since this
         *   library has no [Fe/H] axis at all -- present only because
         *   Specsyn's own specForce() signature requires it
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom --
         *   never a size-0 vector
         * @throws std::runtime_error if log(Teff) itself falls
         *   entirely outside logg_/logTeff_'s own range (so not even
         *   a bracketing logTeff column exists to search), or if
         *   neither bracketing logTeff column has any populated
         *   log(g) value at all
         * @details
         * Overrides Specsyn::specForce() -- see its own comment for
         * when SpecsynLibChained calls this rather than spec().
         * Finds the (up to two) logTeff_ grid values bracketing this
         * star's own log(Teff); at each, searches every logg_ value
         * for whichever is both populated and closest to this star's
         * own log(g) (derived via Specsyn::getSAandLogg), regardless
         * of how far out of logg_'s own range that log(g) is. If
         * only one bracketing logTeff column has any populated
         * log(g) value at all, returns that single grid point's own
         * spectrum (scaled by surface area); if both do, linearly
         * interpolates the two (independently log(g)-snapped)
         * spectra in log(Teff), using the ordinary bracket weight
         * for each -- mirroring the renormalized-weight pattern
         * SpecsynLib2D::spec(double, double)'s own OOBPolicy::coerce
         * handling uses for a partially-populated interpolation
         * cell.
         */
        [[nodiscard]] auto specForce(const Specsyn::StarData& props, double feh) const
        -> std::vector<double> override;

        /**
         * @brief This library's log10(effective temperature) grid points
         * @details
         * Exposed for SpecsynLibChained's benefit, which needs to scan
         * every chained library's own logTeff_ range (alongside
         * SpecsynLibNoWind's and SpecsynLibWR's) to derive a global
         * (logTeffMin, logTeffMax) clamp -- see its tClamp constructor
         * argument. Mirrors SpecsynLibNoWind's identical logTeff()
         * exactly.
         */
        [[nodiscard]] auto logTeff() const -> const std::vector<double>& { return logTeff_; }

        /**
         * @brief This library's log(g) grid points
         * @details
         * Exposed for SpecsynLibChained's benefit, which needs to scan
         * every chained SpecsynLibNoWind/SpecsynLibWD library's own
         * logg_ range to derive a global (loggMin, loggMax) clamp --
         * see its tClamp constructor argument. Mirrors
         * SpecsynLibNoWind's identical logg() exactly.
         */
        [[nodiscard]] auto logg() const -> const std::vector<double>& { return logg_; }

        /**
         * @brief The minimum log(g) this library has real spectral data for
         * @return logg_.front() -- see Specsyn::loggMin()'s own comment
         *   for why this override exists
         */
        [[nodiscard]] auto loggMin() const -> double override { return logg_.front(); }

        /**
         * @brief The maximum log(g) this library has real spectral data for
         * @return logg_.back() -- see Specsyn::loggMin()'s own comment
         *   for why this override exists
         */
        [[nodiscard]] auto loggMax() const -> double override { return logg_.back(); }

    private:

        // References into the parent class's dim2_/dim3_, named for
        // what they actually hold in this (logg, Teff) specialization
        // -- mirrors SpecsynLibNoWind's identical FeH_/logg_/logTeff_
        // aliasing exactly, including the same rationale (deliberately
        // references rather than owned copies: the actual storage
        // lives in, and is sized by, the parent).
        std::vector<double>& logg_;    /**< log(g) values spanned by the tensor grid (alias for SpecsynLib::dim2_) */ // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        std::vector<double>& logTeff_; /**< log10(effective temperature) values spanned by the tensor grid (alias for SpecsynLib::dim3_) */ // NOLINT(readability-identifier-naming, cppcoreguidelines-avoid-const-or-ref-data-members)
    };

} // namespace specsyn

#endif // SPECSYNLIBWD_HPP
