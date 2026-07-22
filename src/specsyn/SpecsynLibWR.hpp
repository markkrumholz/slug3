/**
 * @file SpecsynLibWR.hpp
 * @author Mark Krumholz
 * @brief A SpecsynLib for Wolf-Rayet stars (stars with optically thick winds)
 * @date 2026-07-22
 */

#ifndef SPECSYNLIBWR_HPP
#define SPECSYNLIBWR_HPP

#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <cstddef>
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
     * data/tools/fetch_powr.py: powr_wne.h5, powr_wnl.h5, and
     * powr_wc.h5. Unlike SpecsynLibNoWind's (FeH, logg, Teff) tensor
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
     * Besides the spectra themselves, the constructor also reads two
     * quantities spec() (once implemented) will need but that the
     * parent SpecsynLib knows nothing about: each populated grid
     * point's log10(L/Lsun) (into logL_, a per-point scalar grid
     * analogous to spectra_ but holding a single number instead of a
     * whole spectrum), and each [Fe/H] group's wind clumping density
     * contrast D_infinity (into dInf_, one value per FeH_ entry, since
     * -- unlike log_teff, log_rt, and logl -- it does not vary within
     * a group at all).
     *
     * This constructor currently only supports the WNE and WC grids,
     * which have no additional H-fraction axis; WNL support (an extra
     * axis, handled outside this class -- see fetch_powr.py's xh group
     * attribute) is not yet implemented.
     */
    template <OOBPolicy Policy>
    class SpecsynLibWR : public SpecsynLib<Policy>
    {
    public:

        /**
         * @brief Construct a SpecsynLibWR from a spectral library on disk
         * @param spectraName Name of the spectral model (e.g.
         *   "POWR_WNE" or "POWR_WC")
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param registryName Name of the spectral library registry file
         * @param z The redshift; defaults to zero
         * @details
         * Finds every registry group matching (spectraName, fehMin,
         * fehMax) -- afe/cfe/microTurb/r do not apply to PoWR's WR
         * registry entries (see fetch_powr.py), so they are passed to
         * findMatchingSpectra at their library-wide defaults, which
         * that function ignores anyway for any group missing those
         * attributes -- reads every populated (log(R_t), log(Teff))
         * point's own flux and wavelength grid within each matching
         * [Fe/H] group, builds a single common wavelength grid
         * spanning all of them, and resamples every point onto it
         * before storing it in spectra_.
         */
        SpecsynLibWR(
            const std::string& spectraName,
            double fehMin,
            double fehMax,
            const std::string& registryName = defaultRegistry,
            double z = 0.0);

        // spec() is not yet implemented -- SpecsynLibWR remains
        // abstract (inheriting SpecsynLib's pure virtual
        // spec(const StarData&, double) unimplemented) until it is.

    private:

        /** @brief The shape of logLGrid_, the mdspan view onto logL_ */
        using ScalarGrid = std::mdspan<double, std::dextents<std::size_t, 3>>;

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
    };

} // namespace specsyn

#endif // SPECSYNLIBWR_HPP
