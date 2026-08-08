/**
 * @file FilterCollection.hpp
 * @author Mark Krumholz
 * @brief A class holding the set of photometric filters used in a simulation
 * @date 2026-07-31
 */

#ifndef FILTERCOLLECTION_HPP
#define FILTERCOLLECTION_HPP

#include "Filter.hpp"
#include "FilterCommons.hpp"
#include "PhotCommons.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace phot
{
    /**
     * @class FilterCollection
     * @brief Holds all the filters for which photometry will be computed in a simulation
     * @details
     * Constructed from a list of filter names, each of which is parsed
     * to determine whether it names a tabulated filter (a
     * facility.filter or facility.instrument.filter name, resolved
     * against a filter registry -- see FilterTabulated) or an
     * idealized filter (one of FilterIdeal's own naming conventions),
     * and builds the corresponding Filter for each. phot() then
     * evaluates every filter in the collection against a spectrum in
     * one call, converting each energy-flux filter's result to the
     * requested photometric system.
     */
    class FilterCollection
    {
    public:

        /**
         * @brief Construct a FilterCollection from a list of filter names
         * @param filterNames Names of the filters to include; each
         *   must be either a tabulated-filter name (facility.filter or
         *   facility.instrument.filter, the latter's instrument
         *   omitted only for a facility with exactly one instrument in
         *   the registry -- see FilterTabulated) or an idealized-
         *   filter name (ideal_energy_X_Y, ideal_phot_X_Y, or Q(...)
         *   -- see FilterIdeal)
         * @param photSystem The photometric system phot() converts
         *   energy-flux (photCount() == false) filters' results into
         * @param registryName Name of the filter registry file to
         *   resolve tabulated filter names against
         * @throws std::runtime_error if any entry of filterNames does
         *   not match a recognized tabulated- or idealized-filter
         *   naming convention, or if constructing the corresponding
         *   Filter itself throws (e.g. an unknown facility/instrument/
         *   filter, an ambiguous instrument-omitted facility, or an
         *   unparseable idealized-filter name)
         * @details
         * If photSystem is PhotSystem::Vega, each non-photCount()
         * filter's Vega zero point is computed lazily, from the global
         * Vega reference spectrum, the first time it is actually
         * needed -- see Filter::fluxVega() and phot::vegaSpectrum().
         */
        FilterCollection(const std::vector<std::string>& filterNames,
            PhotSystem photSystem,
            const std::string& registryName = defaultRegistry);

        // Virtual destructor (making this class polymorphic), and
        // correspondingly explicit copy/move declarations, since
        // declaring any destructor suppresses the implicitly-declared
        // move constructor/assignment: mirrors Tracks3D's own
        // identical declarations, needed for the same reason -- a
        // non-copyable class (filters_ holds unique_ptr<Filter>
        // elements) being polymorphic lets pybind11's type_caster use
        // its RTTI-based reference/pointer-return path, rather than
        // one that requires attempting (and failing to compile) a
        // copy
        virtual ~FilterCollection() = default;
        FilterCollection(const FilterCollection&) = delete;
        auto operator=(const FilterCollection&) -> FilterCollection& = delete;
        FilterCollection(FilterCollection&&) = default;
        auto operator=(FilterCollection&&) -> FilterCollection& = default;

        /**
         * @brief Parse a filter name and add the resulting filter to this collection
         * @param name Name of the filter to add; see the constructor's
         *   own filterNames parameter for the recognized naming
         *   conventions
         * @param registryName Name of the filter registry file to
         *   resolve a tabulated filter name against; unused for an
         *   idealized filter name
         * @throws std::runtime_error if name does not match a
         *   recognized tabulated- or idealized-filter naming
         *   convention, or if constructing the corresponding Filter
         *   itself throws
         * @details
         * Used by the constructor to build each filter named in
         * filterNames, one at a time; also public so a caller (e.g.
         * from Python, where a FilterCollection can be built up
         * incrementally rather than from a single up-front list of
         * names) can add filters to an already-constructed
         * FilterCollection one name at a time.
         */
        void addFilter(const std::string& name, const std::string& registryName = defaultRegistry);

        /**
         * @brief Add an already-constructed filter to this collection
         * @param filter The filter to add; ownership is transferred to
         *   this FilterCollection
         * @details
         * Lets a caller (e.g. from Python, where FilterIdeal and
         * FilterTabulated both have their own accessible constructors)
         * add a filter it built directly, rather than by name via the
         * other addFilter() overload.
         */
        void addFilter(std::unique_ptr<Filter> filter) { filters_.push_back(std::move(filter)); }

        /**
         * @brief Compute the photometric response of every filter in this collection to a spectrum
         * @param wl The wavelength grid, in Angstrom, on which spec is computed
         * @param spec The spectrum to which to compute the photometric response
         * @return One value per filter, in the same order as
         *   filterNames()/filterUnits(): for a photCount() filter,
         *   Filter::phot()'s own photon-count value, unconverted
         *   (PhotSystem conversions apply only to energy fluxes); for
         *   an energy-flux filter, Filter::phot()'s F_lambda value,
         *   converted to this collection's photSystem via PhotConvert
         *   (left as-is if photSystem is already Flambda; for Vega,
         *   using this filter's own fluxVega() -- lazily computed the
         *   first time it is needed, see Filter::fluxVega() -- as the
         *   zero point)
         */
        [[nodiscard]] auto phot(const std::vector<double>& wl,
            const std::vector<double>& spec) const -> std::vector<double>;

        /**
         * @brief Get the names of every filter in this collection
         * @return The name() of every filter, in the same order as phot()/filterUnits()
         */
        [[nodiscard]] auto filterNames() const -> std::vector<std::string>;

        /**
         * @brief Get the units of every filter's phot() value
         * @return For a photCount() filter: "photon/s"; for an
         *   energy-flux filter: "erg/(s Angstrom)" (Flambda),
         *   "Jy" (Fnu), "mag(ST)" (ST), "mag(AB)" (AB), or "mag"
         *   (Vega -- astropy has no dedicated Vega-magnitude unit)
         *   matching this collection's photSystem -- in the
         *   same order as phot()/filterNames()
         */
        [[nodiscard]] auto filterUnits() const -> std::vector<std::string>;

        /**
         * @brief Get const access to the filters in this collection
         * @return A const reference to the underlying filters, in the
         *   same order as phot()/filterNames()/filterUnits()
         */
        [[nodiscard]] auto filters() const -> const std::vector<std::unique_ptr<Filter>>& { return filters_; }

        /**
         * @brief Get a single filter by index
         * @param i Index of the filter to get, in the same order as
         *   phot()/filterNames()/filterUnits()
         * @return A const reference to the i'th filter
         * @throws std::out_of_range if i >= filters().size()
         */
        [[nodiscard]] auto getFilter(std::size_t i) const -> const Filter&;

        /**
         * @brief Get a single filter by name
         * @param name Name to search for
         * @return A const reference to the filter whose name() exactly matches name
         * @throws std::runtime_error if no filter's name() matches name
         */
        [[nodiscard]] auto getFilter(const std::string& name) const -> const Filter&;

    private:
        std::vector<std::unique_ptr<Filter>> filters_; /**< The filters in this collection */
        PhotSystem photSystem_; /**< The photometric system phot() converts energy-flux filters' results into */
    };

} // namespace phot

#endif // FILTERCOLLECTION_HPP
