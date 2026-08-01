/**
 * @file Filter.hpp
 * @author Mark Krumholz
 * @brief Abstract base class representing a photometric filter
 * @date 2026-07-27
 */

#ifndef FILTER_HPP
#define FILTER_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace phot
{
    /**
     * @class Filter
     * @brief Abstract base class representing a photometric filter
     * @details
     * This class defines the interface for a photometric filter --
     * something that, given a spectrum, computes the photometric
     * response of that spectrum in the filter. Derived classes must
     * implement phot() according to however the filter's own response
     * is represented (e.g. tabulated, analytic).
     */
    class Filter
    {
    public:

        // Constructor and destructor
        /**
         * @brief Construct a Filter
         * @param name Name of this filter, for output purposes
         * @param photCount If true, this filter returns photon counts
         *   rather than F_lambda values; see phot() for details
         */
        explicit Filter(std::string name, bool photCount = false)
        : name_(std::move(name)), photCount_(photCount) {}
        Filter(const Filter&) = default;
        auto operator=(const Filter&) -> Filter& = default;
        Filter(Filter&&) = default;
        auto operator=(Filter&&) -> Filter& = default;
        virtual ~Filter() = default;

        /**
         * @brief Get the name of this filter
         * @return The name of this filter
         */
        [[nodiscard]] auto name() const -> const std::string& { return name_; }

        /**
         * @brief Get whether this filter returns photon counts
         * @return True if this filter returns photon counts rather
         *   than F_lambda values, false otherwise
         */
        [[nodiscard]] auto photCount() const -> bool { return photCount_; }

        /**
         * @brief Get this filter's pivot wavelength
         * @return A single characteristic wavelength for this filter,
         *   in Angstrom
         * @details
         * Used, e.g., to convert an F_lambda value returned by phot()
         * into F_nu or a wavelength-dependent magnitude system via
         * PhotConvert, since phot() itself collapses a filter's
         * response to a single number with no wavelength of its own.
         */
        [[nodiscard]] virtual auto wlPivot() const -> double = 0;

        /**
         * @brief Compute the photometric response in this filter to a given spectrum
         * @param wl The wavelength grid, in Angstrom, on which spec is computed
         * @param spec The spectrum to which to compute the photometric response
         * @return The photometric response, in erg/s/cm^2/Angstrom if
         *   photCount() is false, or photons/s if photCount() is true
         */
        [[nodiscard]] virtual auto phot(const std::vector<double>& wl,
            const std::vector<double>& spec) const -> double = 0;

        /**
         * @brief Get this filter's mean Vega flux
         * @return The filter-mean flux of Vega in this filter, in
         *   erg/s/cm^2/Angstrom, as last set by setFluxVega() (0 if
         *   setFluxVega() has never been called)
         */
        [[nodiscard]] auto fluxVega() const -> double { return fluxVega_; }

        /**
         * @brief Set this filter's mean Vega flux from the Vega reference spectrum
         * @param wlVega Wavelength grid of the Vega reference spectrum, in Angstrom
         * @param fluxVega The Vega reference spectrum, in erg/s/cm^2/Angstrom
         * @details
         * Computes fluxVega_ by evaluating this filter's own phot()
         * on the Vega reference spectrum, and stores the result --
         * i.e. fluxVega_ becomes this filter's own idea of "the flux
         * of Vega," on whatever basis (tabulated response, top-hat
         * passband, etc.) phot() itself uses.
         */
        void setFluxVega(const std::vector<double>& wlVega, const std::vector<double>& fluxVega)
        {
            fluxVega_ = phot(wlVega, fluxVega);
        }

    protected:
        // Protected (rather than private with a setter) so a derived
        // class's name-parsing constructor (e.g. FilterIdeal's) can
        // set it directly after base construction, once it has
        // determined it by parsing the name passed to the base class
        bool photCount_; /**< True if this filter returns photon counts rather than F_lambda values */

        /**
         * @brief Element-wise natural logarithm of a wavelength grid
         * @param wl Wavelength grid, in Angstrom
         * @return ln(wl), element-wise
         * @details
         * Shared by every derived class that integrates its
         * photometric response in ln(wavelength) rather than
         * wavelength itself (FilterTabulated and FilterIdeal, as of
         * this writing), so it lives here instead of being
         * duplicated in each of them.
         */
        static auto lnGrid(const std::vector<double>& wl) -> std::vector<double>
        {
            std::vector<double> result(wl.size());
            std::ranges::transform(wl, result.begin(), [](const double w) -> double { return std::log(w); });
            return result;
        }

    private:
        std::string name_;      /**< Name of this filter, for output purposes */
        double fluxVega_ = 0.0; /**< Filter-mean flux of Vega, in erg/s/cm^2/Angstrom; see setFluxVega() */
    };

} // namespace phot

#endif // FILTER_HPP
