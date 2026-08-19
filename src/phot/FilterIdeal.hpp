/**
 * @file FilterIdeal.hpp
 * @author Mark Krumholz
 * @brief A class representing an idealized, top-hat photometric filter
 * @date 2026-07-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef FILTERIDEAL_HPP
#define FILTERIDEAL_HPP

#include "Filter.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
         * @throws std::runtime_error if photCount is false and wlMin
         *   <= 0 or wlMax is infinite -- phot()'s energy-flux mode
         *   integrates in ln(wavelength), which breaks at either
         *   extreme (see phot()'s own comment); a photCount filter
         *   has no such restriction
         */
        FilterIdeal(std::string name, double wlMin, double wlMax, bool photCount = false)
        : Filter(std::move(name), photCount), wlMin_(wlMin), wlMax_(wlMax)
        {
            if (!photCount_ && (wlMin_ <= 0.0 || std::isinf(wlMax_)))
            {
                throw std::runtime_error(
                    "FilterIdeal: an energy-flux filter (photCount = false) "
                    "must have a finite, positive wavelength range; got "
                    "wlMin = " + std::to_string(wlMin_) + ", wlMax = " +
                    std::to_string(wlMax_));
            }
        }

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
         *    hydrogen, Q(CII) = photons that ionize C+ to C++).
         *    Ionizing photons have energy above the ionization
         *    potential IP, i.e. wavelength below the threshold
         *    lambda = h*c/IP, so the filter is set to wlMin = 0
         *    (no lower bound), wlMax = h*c/IP (the ionization-threshold
         *    wavelength in Angstrom, sourced from CRC data), and
         *    photCount = true.
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

        /**
         * @brief Get this filter's pivot wavelength
         * @return The midpoint of [wlMin, wlMax]; wlMax if wlMin is 0
         *   (as for a Q(*) ionization-threshold filter, whose passband
         *   has no finite lower end), or wlMin if wlMax is infinite
         *   (as for an ideal_phot_X_inf filter, whose passband has no
         *   finite upper end)
         */
        [[nodiscard]] auto wlPivot() const -> double override
        {
            if (std::isinf(wlMax_)) { return wlMin_; }
            if (wlMin_ <= 0.0) { return wlMax_; }
            return 0.5 * (wlMin_ + wlMax_);
        }

        /**
         * @brief Compute the photometric response of this filter to a spectrum
         * @param wl The wavelength grid, in Angstrom
         * @param spec The spectrum, in erg/s/Angstrom
         * @return If photCount() is false: the mean value of spec over
         *   the filter's passband [wlMin, wlMax] on a log-wavelength
         *   basis -- (integral of spec d(ln λ)) / ln(wlMax / wlMin),
         *   in erg/s/Angstrom, matching Filter::phot()'s own
         *   documented units and FilterTabulated::phot()'s own
         *   dlnλ convention -- over the intersection of
         *   [wlMin, wlMax] with the supplied wavelength grid; if
         *   photCount() is true: the total photon-count rate, in
         *   photons/s, computed by integrating spec × λ / (h c) dλ
         *   over the same range (a count rate, not a mean, so not
         *   normalized)
         */
        [[nodiscard]] auto phot(const std::vector<double>& wl,
            const std::vector<double>& spec) const -> double override;

    private:
        double wlMin_; /**< Minimum wavelength of this filter's response, in Angstrom */
        double wlMax_; /**< Maximum wavelength of this filter's response, in Angstrom */
    };

} // namespace phot

#endif // FILTERIDEAL_HPP
