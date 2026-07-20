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
     * @details
     * This class is the spectral equivalent of tracks::Tracks3D: rather
     * than computing spectra from a formula (as SpecsynBlackbody does),
     * it loads pre-computed spectra from an HDF5 spectral library on
     * disk, such as the BOSZ library fetched by data/tools/fetch_bosz.py.
     */
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

    private:

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
