/**
 * @file Filter.hpp
 * @author Mark Krumholz
 * @brief Abstract base class representing a photometric filter
 * @date 2026-07-27
 */

#ifndef FILTER_HPP
#define FILTER_HPP

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
         * @brief Compute the photometric response in this filter to a given spectrum
         * @param wl The wavelength grid, in Angstrom, on which spec is computed
         * @param spec The spectrum to which to compute the photometric response
         * @return The photometric response, in erg/s/cm^2/Angstrom if
         *   photCount() is false, or photons/s if photCount() is true
         */
        [[nodiscard]] virtual auto phot(const std::vector<double>& wl,
            const std::vector<double>& spec) const -> double = 0;

    protected:
        // Protected (rather than private with a setter) so a derived
        // class's name-parsing constructor (e.g. FilterIdeal's) can
        // set it directly after base construction, once it has
        // determined it by parsing the name passed to the base class
        bool photCount_; /**< True if this filter returns photon counts rather than F_lambda values */

    private:
        std::string name_; /**< Name of this filter, for output purposes */
    };

} // namespace phot

#endif // FILTER_HPP
