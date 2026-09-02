/**
 * @file VegaSpectrum.hpp
 * @author Mark Krumholz
 * @brief Holds the Vega reference spectrum, and a lazy global accessor for it
 * @date 2026-08-02
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef VEGASPECTRUM_HPP
#define VEGASPECTRUM_HPP

#include "PhotCommons.hpp"
#include <string>
#include <utility>
#include <vector>

namespace phot
{
    /**
     * @class VegaSpectrum
     * @brief Holds the Vega reference spectrum, read once from an HDF5 file
     */
    class VegaSpectrum
    {
    public:

        /**
         * @brief Construct a VegaSpectrum, reading the reference spectrum from a file
         * @param vegaName Name of the HDF5 file holding the Vega
         *   reference spectrum ("wl" and "flux" datasets -- see
         *   data/tools/spectra/fetch_vega.py)
         * @throws std::runtime_error if vegaName cannot be found or read
         */
        explicit VegaSpectrum(const std::string& vegaName = defaultVegaSpec);

        ~VegaSpectrum() = default;

        // Disable copy and move constructors and assignment operators,
        // since this class is meant to be used only via a single,
        // shared instance -- see vegaSpectrum()
        VegaSpectrum(const VegaSpectrum&) = delete;
        VegaSpectrum(VegaSpectrum&&) = delete;
        auto operator=(const VegaSpectrum&) -> VegaSpectrum& = delete;
        auto operator=(VegaSpectrum&&) -> VegaSpectrum& = delete;

        /**
         * @brief Get the Vega reference spectrum
         * @return A pair of const references: the wavelength grid, in
         *   Angstrom, and the flux, in erg/s/cm^2/Angstrom
         */
        [[nodiscard]] auto spectrum() const
            -> std::pair<const std::vector<double>&, const std::vector<double>&>
        {
            return {wl_, flux_};
        }

    private:
        std::vector<double> wl_;   /**< Vega reference spectrum wavelength grid, in Angstrom */
        std::vector<double> flux_; /**< Vega reference spectrum flux, in erg/s/cm^2/Angstrom */
    };

    /**
     * @brief Return the global Vega reference spectrum
     * @param vegaName Name of the HDF5 file holding the Vega reference
     *   spectrum; only has any effect the first time this function is
     *   called anywhere in the program -- see below
     * @return The same pair VegaSpectrum::spectrum() returns
     * @details
     * The VegaSpectrum instance is a function-local static, so it is
     * constructed on first use rather than before main() begins, and
     * is shared by every caller for the remaining lifetime of the
     * program (the same lazy construct-once model used by
     * utils::uniqueID()/utils::rng()). This means vegaName only has any
     * effect on the very first call to this function anywhere in the
     * program; every later call, regardless of what vegaName it is
     * given, returns the spectrum already loaded by that first call. A
     * caller that needs a non-default Vega file must therefore call
     * this function with that vegaName before anything else in the
     * program can call it with the default (see
     * SimControls::readFilters()). Not marked [[nodiscard]]: unlike
     * VegaSpectrum::spectrum(), a caller is sometimes deliberately
     * calling this only to force that first, file-selecting
     * construction, with no use for the returned spectrum itself.
     */
    inline auto vegaSpectrum(const std::string& vegaName = defaultVegaSpec)
        -> std::pair<const std::vector<double>&, const std::vector<double>&>
    {
        static const VegaSpectrum instance(vegaName);
        return instance.spectrum();
    }

} // namespace phot

#endif // VEGASPECTRUM_HPP
