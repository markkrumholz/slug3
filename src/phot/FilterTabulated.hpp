/**
 * @file FilterTabulated.hpp
 * @author Mark Krumholz
 * @brief A class representing a photometric filter with a tabulated response
 * @date 2026-07-27
 */

#ifndef FILTERTABULATED_HPP
#define FILTERTABULATED_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "Filter.hpp"
#include "FilterCommons.hpp"
#include <string>
#include <utility>
#include <vector>

namespace phot
{
    /**
     * @class FilterTabulated
     * @brief A photometric filter whose response is tabulated as a function of wavelength
     */
    class FilterTabulated : public Filter
    {
    public:

        // Constructors

        /**
         * @brief Construct a FilterTabulated directly from a tabulated response
         * @param name Name of this filter, for output purposes
         * @param wl Wavelengths at which the response is tabulated, in Angstrom
         * @param response Filter response at each wavelength in wl
         * @param wlPivot This filter's pivot wavelength, in Angstrom
         *   (see Filter::wlPivot()); mandatory here since, unlike the
         *   registry-based constructor, there is no registry entry to
         *   read one from
         * @details
         * A FilterTabulated always returns an energy flux (photCount()
         * is always false); slug's photCount concept -- used for
         * filter-like objects that report photon counts rather than
         * energy fluxes, e.g. hydrogen-ionizing photon counters -- is
         * unrelated to SVO's own "photon counter" vs. "energy counter"
         * detector-type distinction, and is not exposed here.
         */
        FilterTabulated(std::string name, const std::vector<double>& wl,
            const std::vector<double>& response, double wlPivot)
        : Filter(std::move(name)), wl_(wl), lnWl_(lnGrid(wl_)), responseData_(response),
          response_(lnWl_, responseData_),
          norm_(response_.integ(response_.xMin(), response_.xMax())),
          wlPivot_(wlPivot)
        { }

        /**
         * @brief Construct a FilterTabulated from an entry in a filter registry
         * @param facility Facility that operates the instrument the filter belongs to (e.g. "HST")
         * @param instrument Instrument the filter belongs to (e.g. "ACS_WFC")
         * @param filter Name of the filter (e.g. "F435W")
         * @param registryName Name of the filter registry file
         * @details
         * Follows the facility/instrument/filter hierarchy used by
         * data/tools/filters/fetch_filter_vo.py to fetch filter data and by
         * the filters.h5 file it writes to store it. name() is set to
         * "facility.instrument.filter".
         */
        explicit FilterTabulated(const std::string& facility,
            const std::string& instrument,
            const std::string& filter,
            const std::string& registryName = defaultRegistry);

        // Disallow copy and move: response_ is an Interpolator1D, which
        // is itself neither copyable nor movable (see its own comment),
        // so neither can this class be
        FilterTabulated(const FilterTabulated&) = delete;
        auto operator=(const FilterTabulated&) -> FilterTabulated& = delete;
        FilterTabulated(FilterTabulated&&) = delete;
        auto operator=(FilterTabulated&&) -> FilterTabulated& = delete;
        ~FilterTabulated() override = default;

        // Observers

        /**
         * @brief Get the integral of the filter response with respect to ln(wavelength)
         * @return The integral of the filter response with respect to ln(wavelength)
         */
        [[nodiscard]] auto norm() const -> double { return norm_; }

        /**
         * @brief Get this filter's pivot wavelength
         * @return The pivot wavelength, in Angstrom -- either supplied
         *   directly (the (wl, response) constructor) or read from the
         *   registry entry's wl_ref field (the registry constructor)
         */
        [[nodiscard]] auto wlPivot() const -> double override { return wlPivot_; }

        /**
         * @brief Compute the photometric response in this filter to a given spectrum
         * @param wl The wavelength grid, in Angstrom, on which spec is computed
         * @param spec The spectrum to which to compute the photometric response
         * @return The photometric response, in erg/s/cm^2/Angstrom
         * @details
         * Evaluates \f$\int F_\lambda(\lambda) R(\lambda)\, d\ln\lambda
         * \big/ \int R(\lambda)\, d\ln\lambda\f$ (the second integral
         * being norm()) -- see this class's own file-level comment in
         * FilterTabulated.cpp for why this form (rather than the
         * photon-counting \f$\int [F_\lambda R / \lambda]\, d\lambda
         * \big/ \int [R / \lambda]\, d\lambda\f$ it is analytically
         * equivalent to) is the one actually evaluated here.
         */
        [[nodiscard]] auto phot(const std::vector<double>& wl,
            const std::vector<double>& spec) const -> double override;

        /**
         * @brief Get the wavelengths at which the response is tabulated
         * @return A const reference to the wavelength grid, in Angstrom
         */
        [[nodiscard]] auto wl() const -> const std::vector<double>& { return wl_; }

        /**
         * @brief Get the tabulated filter response data
         * @return A const reference to the filter response at each
         *   wavelength in wl()
         */
        [[nodiscard]] auto responseData() const -> const std::vector<double>& { return responseData_; }

        /**
         * @brief Get the interpolator built from the tabulated response
         * @return A const reference to the Interpolator1D built from
         *   ln(wl()) and responseData(), as used internally by phot()
         */
        [[nodiscard]] auto response() const -> const interp::Interpolator1D<1>& { return response_; }

    private:

        /**
         * @brief Data needed to build a FilterTabulated, read from a registry entry
         */
        struct RegistryData
        {
            std::vector<double> wl_;           /**< Wavelengths at which the response is tabulated, in Angstrom */
            std::vector<double> response_;     /**< Filter response at each wavelength in wl_ */
            double wlPivot_ = 0.0;              /**< Pivot wavelength read from the registry entry's wl_ref field, in Angstrom */
        };

        /**
         * @brief Read a filter's tabulated response from a registry entry
         * @param facility Facility that operates the instrument the filter belongs to
         * @param instrument Instrument the filter belongs to
         * @param filter Name of the filter
         * @param registryName Name of the filter registry file
         * @return The wavelengths and response for the requested filter
         */
        static auto loadFromRegistry(const std::string& facility,
            const std::string& instrument,
            const std::string& filter,
            const std::string& registryName)
        -> RegistryData;

        /**
         * @brief Construct a FilterTabulated from a name and data read from a registry entry
         * @param name Name of this filter, for output purposes
         * @param data Wavelengths and response read by loadFromRegistry
         */
        FilterTabulated(std::string name, const RegistryData& data)
        : FilterTabulated(std::move(name), data.wl_, data.response_, data.wlPivot_) {}

        std::vector<double> wl_;             /**< Wavelengths at which the response is tabulated, in Angstrom */
        std::vector<double> lnWl_;           /**< ln(wl_) */
        std::vector<double> responseData_;   /**< Filter response at each wavelength in wl_ */
        interp::Interpolator1D<1> response_; /**< Interpolator for responseData_ as a function of lnWl_ */
        double norm_;                        /**< Integral of the filter response with respect to ln(wavelength) */
        double wlPivot_;                     /**< Pivot wavelength, in Angstrom */
    };

} // namespace phot

#endif // FILTERTABULATED_HPP
