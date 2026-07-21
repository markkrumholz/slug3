/**
 * @file SpecsynLib.hpp
 * @author Mark Krumholz
 * @brief A spectral synthesizer that loads spectra from a library on disk
 * @date 2026-07-20
 */

#ifndef SPECSYNLIB_HPP
#define SPECSYNLIB_HPP

#include "../tracks/TrackCommons.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include <string>
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynLib
     * @brief A Specsyn specialization that reads spectra from an HDF5 spectral library
     * @tparam Policy How this SpecsynLib should handle a star whose
     *   (logg, Teff) properties fall outside its tensor grid -- see
     *   OOBPolicy. This is a template parameter, rather than a
     *   constructor argument or runtime flag, so that spec() (to be
     *   implemented) can compile the chosen behavior directly into
     *   its hot path instead of branching on it at runtime.
     * @details
     * This class is the spectral equivalent of tracks::Tracks3D: rather
     * than computing spectra from a formula (as SpecsynBlackbody does),
     * it loads pre-computed spectra from an HDF5 spectral library on
     * disk, such as the BOSZ library fetched by data/tools/fetch_bosz.py.
     */
    template <OOBPolicy Policy>
    class SpecsynLib : public Specsyn
    {
    public:

        /**
         * @brief Construct a SpecsynLib object from a spectral library on disk
         * @param spectraName Name of the spectral model
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param afe Value of [alpha/Fe]
         * @param cfe Value of [C/Fe]
         * @param microTurb Microturbulent velocity, in km/s
         * @param r Spectral resolution
         * @param registryName Name of the spectral library registry file
         * @param z The redshift; defaults to zero
         */
        SpecsynLib(
            const std::string& spectraName,
            double fehMin,
            double fehMax,
            double afe = tracks::defaultAFe,
            double cfe = defaultCFe,
            double microTurb = defaultMicroTurb,
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
         * Specsyn::getSAandLogg for the latter), checks that
         * (feh, logg, Teff) falls within the grid built by the
         * constructor, and if so, trilinearly interpolates the 8
         * neighboring grid points -- all of which must have an
         * actual spectrum in the library, since interpolating across
         * an unpopulated point would be meaningless -- to get the
         * star's specific flux at the surface, then scales by the
         * star's surface area to get specific luminosity.
         */
        [[nodiscard]] auto spec(const StarData& props, double feh) const
        -> std::vector<double> override;

        /**
         * @brief Resample every spectrum in this library onto a new wavelength grid
         * @param wlNew The new wavelength grid, in Angstrom
         * @details
         * For every populated (non-empty) point in the (FeH, logg,
         * Teff) tensor grid, builds an Interpolator1D of that point's
         * flux versus this library's existing wavelength grid (wl_),
         * then evaluates it at every wavelength in wlNew to produce
         * that point's resampled flux; wavelengths in wlNew that fall
         * outside the range spanned by wl_ are assigned a flux of
         * zero rather than extrapolated. Once every populated point
         * has been resampled this way, wl_ itself is replaced with
         * wlNew, so wl() and every subsequent spec() call reflect the
         * new grid. Unpopulated grid points are left as empty vectors,
         * exactly as before.
         */
        void resample(const std::vector<double>& wlNew);

    private:

        /**
         * @brief Handle a star that falls outside this library's grid
         * @param message Description of why the star is out of bounds
         * @return A size-0 vector, if Policy is OOBPolicy::silent
         * @throws std::runtime_error with message, if Policy is OOBPolicy::Throw
         */
        [[nodiscard]] static auto outOfBoundsResult(const std::string& message) -> std::vector<double>;

        // Spectral library data
        std::vector<double> FeH_;  /**< [Fe/H] values spanned by the tensor grid */ // NOLINT(readability-identifier-naming)
        std::vector<double> logg_; /**< log(g) values spanned by the tensor grid */
        std::vector<double> Teff_; /**< Effective temperature values spanned by the tensor grid */ // NOLINT(readability-identifier-naming)

        /**
         * @brief Spectra on the (FeH, logg, Teff) tensor grid
         * @details
         * A flattened array of shape (FeH_.size(), logg_.size(),
         * Teff_.size()) -- see the SpectraGrid alias in SpecsynLib.cpp
         * for the mdspan used to index it -- holding one spectrum
         * (a vector of the same length as wl_) per populated grid
         * point. Not every (FeH, logg, Teff) combination in this outer
         * product actually has a spectrum in the library; unpopulated
         * points are left as empty (size-0) vectors.
         */
        std::vector<std::vector<double>> spectra_;

        double AFe_;       /**< [alpha/Fe] value of this spectral library */ // NOLINT(readability-identifier-naming)
        double CFe_;       /**< [C/Fe] value of this spectral library */     // NOLINT(readability-identifier-naming)
        double microTurb_; /**< Microturbulent velocity of this spectral library, in km/s */
        double r_;         /**< Spectral resolution of this spectral library */
    };

} // namespace specsyn

#endif // SPECSYNLIB_HPP
