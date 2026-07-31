/**
 * @file FilterIdeal.hpp
 * @author Mark Krumholz
 * @brief A class representing an idealized, top-hat photometric filter
 * @date 2026-07-31
 */

#ifndef FILTERIDEAL_HPP
#define FILTERIDEAL_HPP

#include "Filter.hpp"
#include <string>
#include <utility>

namespace phot
{
    /**
     * @class FilterIdeal
     * @brief An idealized photometric filter whose response is a top-hat function of wavelength
     * @details
     * A FilterIdeal's response is exactly 1 within [wlMin, wlMax] and
     * exactly 0 everywhere else -- useful both for idealized
     * experiments and for computing photon fluxes above (or between)
     * particular ionization thresholds, without needing a real
     * instrument's tabulated response curve.
     */
    class FilterIdeal : public Filter
    {
    public:

        // Constructors

        /**
         * @brief Construct a FilterIdeal directly from a wavelength range
         * @param name Name of this filter, for output purposes
         * @param wlMin Minimum wavelength of the filter's response, in Angstrom
         * @param wlMax Maximum wavelength of the filter's response, in Angstrom
         * @param photCount If true, this filter returns photon counts
         *   rather than F_lambda values; see Filter::phot() for details
         */
        FilterIdeal(std::string name, double wlMin, double wlMax, bool photCount = false)
        : Filter(std::move(name), photCount), wlMin_(wlMin), wlMax_(wlMax) { }

        /**
         * @brief Construct a FilterIdeal by parsing its name
         * @param name Name of this filter; must match one of the
         *   recognized ideal-filter naming conventions (see below)
         * @details
         * Recognizes two naming conventions:
         *
         * 1. ideal_<kind>_X_Y, where X and Y are wavelengths in
         *    Angstrom (Y may also be "inf", meaning no upper bound)
         *    giving wlMin and wlMax respectively, and <kind> is either:
         *    - "energy": an energy-flux filter (photCount() is false)
         *    - "phot":   a photon-counting filter (photCount() is true)
         *
         * 2. Q(<spec>), where <spec> is an atomic symbol followed by a
         *    Roman numeral giving the ionization state in astronomical
         *    notation (e.g. Q(HI) = photons that ionize neutral
         *    hydrogen, Q(CII) = photons that ionize C+ to C++).  The
         *    filter is set to wlMin = h*c/IP (the ionization-threshold
         *    wavelength in Angstrom, sourced from CRC data), wlMax =
         *    infinity, and photCount = true.
         *
         * @throws std::runtime_error if name does not match either
         *   recognized convention, or if the requested ionization
         *   potential is not available in the CRC data
         */
        explicit FilterIdeal(std::string name);

        FilterIdeal(const FilterIdeal&) = default;
        auto operator=(const FilterIdeal&) -> FilterIdeal& = default;
        FilterIdeal(FilterIdeal&&) = default;
        auto operator=(FilterIdeal&&) -> FilterIdeal& = default;
        ~FilterIdeal() override = default;

        // Observers

        /**
         * @brief Get the minimum wavelength of this filter's response
         * @return The minimum wavelength of this filter's response, in Angstrom
         */
        [[nodiscard]] auto wlMin() const -> double { return wlMin_; }

        /**
         * @brief Get the maximum wavelength of this filter's response
         * @return The maximum wavelength of this filter's response, in Angstrom
         */
        [[nodiscard]] auto wlMax() const -> double { return wlMax_; }

        // phot() is intentionally not yet overridden here -- left for
        // a future commit -- so FilterIdeal remains abstract for now

    private:
        double wlMin_; /**< Minimum wavelength of this filter's response, in Angstrom */
        double wlMax_; /**< Maximum wavelength of this filter's response, in Angstrom */
    };

} // namespace phot

#endif // FILTERIDEAL_HPP
