/**
 * @file SpecsynLibNoWind.hpp
 * @author Mark Krumholz
 * @brief A SpecsynLib for stars without optically thick winds
 * @date 2026-07-22
 */

#ifndef SPECSYNLIBNOWIND_HPP
#define SPECSYNLIBNOWIND_HPP

#include "../tracks/TrackCommons.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <limits>
#include <string>
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynLibNoWind
     * @brief A SpecsynLib specialization for stars without optically thick winds
     * @tparam Policy See SpecsynLib.
     * @details
     * Covers spectral libraries -- like BOSZ and TLUSTY -- whose
     * spectra sit on a (FeH, logg, Teff) tensor grid, one HDF5 file
     * per library, following the conventions of
     * data/tools/fetch_bosz.py and data/tools/fetch_tlusty.py: groups
     * named spectra_feh<feh>... holding one dataset per (Teff, logg)
     * pair actually present, plus a top-level wavelengths group.
     * dim1_, dim2_, dim3_ (inherited from SpecsynLib) hold FeH, logg,
     * and Teff respectively, aliased here as FeH_, logg_, and Teff_
     * for readability.
     *
     * This is the counterpart to SpecsynLibWR, which instead covers
     * Wolf-Rayet spectral libraries -- stars with optically thick
     * winds -- parameterized by fundamentally different variables
     * (stellar temperature and transformed radius, rather than Teff
     * and logg).
     */
    template <OOBPolicy Policy>
    class SpecsynLibNoWind : public SpecsynLib<Policy>
    {
    public:

        /**
         * @brief Construct a SpecsynLibNoWind from a spectral library on disk
         * @param spectraName Name of the spectral model
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param afe Value of [alpha/Fe]
         * @param cfe Value of [C/Fe]
         * @param microTurb Microturbulent velocity, in km/s; if NaN
         *   (the default), uses this library's own micro_default value
         *   from the registry instead, since a sensible default
         *   microturbulence is library-specific (e.g. hot, massive OB
         *   stars are conventionally modeled with substantially more
         *   microturbulence than cooler, lower-mass stars) rather than
         *   a single constant that fits every library equally well
         * @param r Spectral resolution
         * @param registryName Name of the spectral library registry file
         * @param z The redshift; defaults to zero
         */
        SpecsynLibNoWind(
            const std::string& spectraName,
            double fehMin,
            double fehMax,
            double afe = tracks::defaultAFe,
            double cfe = defaultCFe,
            double microTurb = std::numeric_limits<double>::quiet_NaN(),
            double r = defaultR,
            const std::string& registryName = defaultRegistry,
            double z = 0.0);

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
         *   (FeH, logg, Teff) grid and Policy is OOBPolicy::silent
         * @throws std::runtime_error if the star falls outside this
         *   library's grid and Policy is OOBPolicy::Throw
         * @details
         * Derives the star's Teff and log(g) from props (via
         * Specsyn::getSAandLogg for the latter) and checks that
         * (feh, logg, Teff) falls within the grid built by the
         * constructor, then delegates the actual trilinear
         * interpolation to SpecsynLib::spec(double, double, double)
         * -- which also confirms every one of the 8 neighboring grid
         * points has an actual spectrum, since interpolating across
         * an unpopulated point would be meaningless -- and finally
         * scales the result by the star's surface area to convert
         * specific flux at the surface into specific luminosity.
         */
        [[nodiscard]] auto spec(const Specsyn::StarData& props, double feh) const
        -> std::vector<double> override;

    private:

        // References into the parent class's dim1_/dim2_/dim3_, named
        // for what they actually hold in this (FeH, logg, Teff)
        // specialization, so the constructor logic inherited from the
        // pre-refactor SpecsynLib can stay unchanged in everything but
        // name. Deliberately references rather than owned copies --
        // the actual storage lives in (and is sized by) the parent,
        // which knows nothing about what its three axes mean -- so the
        // avoid-const-or-ref-data-members check's usual objection
        // (disabling implicit copy/move assignment) doesn't apply in
        // practice: this class is only ever used through the same
        // non-copyable, non-movable ownership pattern (unique_ptr in
        // SpecsynLibChained) as every other Specsyn.
        std::vector<double>& FeH_;  /**< [Fe/H] values spanned by the tensor grid (alias for SpecsynLib::dim1_) */ // NOLINT(readability-identifier-naming, cppcoreguidelines-avoid-const-or-ref-data-members)
        std::vector<double>& logg_; /**< log(g) values spanned by the tensor grid (alias for SpecsynLib::dim2_) */ // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        std::vector<double>& Teff_; /**< Effective temperature values spanned by the tensor grid (alias for SpecsynLib::dim3_) */ // NOLINT(readability-identifier-naming, cppcoreguidelines-avoid-const-or-ref-data-members)

        double AFe_;       /**< [alpha/Fe] value of this spectral library */ // NOLINT(readability-identifier-naming)
        double CFe_;       /**< [C/Fe] value of this spectral library */     // NOLINT(readability-identifier-naming)
        double microTurb_; /**< Microturbulent velocity of this spectral library, in km/s */
        double r_;         /**< Spectral resolution of this spectral library */
    };

} // namespace specsyn

#endif // SPECSYNLIBNOWIND_HPP
