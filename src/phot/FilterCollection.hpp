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
#include <memory>
#include <string>
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
         * @param vegaName Name of the HDF5 file holding the Vega
         *   reference spectrum ("wl" and "flux" datasets -- see
         *   data/tools/fetch_vega.py); only read if photSystem is
         *   Vega, in which case every non-photCount() filter's
         *   fluxVega() is set from it (see Filter::setFluxVega())
         * @throws std::runtime_error if any entry of filterNames does
         *   not match a recognized tabulated- or idealized-filter
         *   naming convention, or if constructing the corresponding
         *   Filter itself throws (e.g. an unknown facility/instrument/
         *   filter, an ambiguous instrument-omitted facility, or an
         *   unparseable idealized-filter name); also throws if
         *   photSystem is Vega and vegaName cannot be read
         */
        FilterCollection(const std::vector<std::string>& filterNames,
            PhotSystem photSystem,
            const std::string& registryName = defaultRegistry,
            const std::string& vegaName = defaultVegaSpec);

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
         *   using this filter's own fluxVega() -- populated from
         *   vegaName at construction, see the constructor -- as the
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
         * @return For a photCount() filter: "photons/s"; for an
         *   energy-flux filter: "erg/s/Angstrom" (Flambda),
         *   "erg/s/Hz" (Fnu), "STmag" (ST), "ABmag" (AB), or "VegaMag"
         *   (Vega), matching this collection's photSystem -- in the
         *   same order as phot()/filterNames()
         */
        [[nodiscard]] auto filterUnits() const -> std::vector<std::string>;

        /**
         * @brief Get const access to the filters in this collection
         * @return A const reference to the underlying filters, in the
         *   same order as phot()/filterNames()/filterUnits()
         */
        [[nodiscard]] auto filters() const -> const std::vector<std::unique_ptr<Filter>>& { return filters_; }

    private:
        std::vector<std::unique_ptr<Filter>> filters_; /**< The filters in this collection */
        PhotSystem photSystem_; /**< The photometric system phot() converts energy-flux filters' results into */
    };

} // namespace phot

#endif // FILTERCOLLECTION_HPP
