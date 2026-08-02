/**
 * @file Filter.hpp
 * @author Mark Krumholz
 * @brief Abstract base class representing a photometric filter
 * @date 2026-07-27
 */

#ifndef FILTER_HPP
#define FILTER_HPP

#include "VegaSpectrum.hpp"
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
         *   erg/s/cm^2/Angstrom, computing and caching it (via
         *   setFluxVega()) against the global Vega reference spectrum
         *   (see vegaSpectrum()) the first time it is needed; always 0
         *   if photCount() is true, since a photon-count filter's
         *   phot() value has no Flambda meaning, so the Vega spectrum
         *   is meaningless for it
         */
        [[nodiscard]] auto fluxVega() const -> double
        {
            if (photCount_) { return 0.0; }
            if (fluxVega_ == 0.0)
            {
#ifdef _OPENMP
#pragma omp critical(filterSetFluxVega)
#endif
                {
                    // Re-check inside the critical section: another
                    // thread may have already computed fluxVega_ while
                    // this one was waiting to enter it (e.g. multiple
                    // cluster trials, sharing this same Filter, running
                    // concurrently in SimCluster::run()'s parallel for)
                    if (fluxVega_ == 0.0) { setFluxVega(); }
                }
            }
            return fluxVega_;
        }

    protected:
        // Protected (rather than private with a setter) so a derived
        // class's name-parsing constructor (e.g. FilterIdeal's) can
        // set it directly after base construction, once it has
        // determined it by parsing the name passed to the base class
        bool photCount_; /**< True if this filter returns photon counts rather than F_lambda values */

        /**
         * @brief Set this filter's mean Vega flux from the global Vega reference spectrum
         * @details
         * Computes fluxVega_ by evaluating this filter's own phot() on
         * the global Vega reference spectrum (see vegaSpectrum()), and
         * stores the result -- i.e. fluxVega_ becomes this filter's
         * own idea of "the flux of Vega," on whatever basis (tabulated
         * response, top-hat passband, etc.) phot() itself uses. Called
         * automatically, at most once, by fluxVega() the first time it
         * is needed; not normally called directly.
         */
        void setFluxVega() const
        {
            const auto [wlVega, flux] = vegaSpectrum();
            fluxVega_ = phot(wlVega, flux);
        }

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
        std::string name_; /**< Name of this filter, for output purposes */
        mutable double fluxVega_ = 0.0; /**< Filter-mean flux of Vega, in erg/s/cm^2/Angstrom; lazily set by fluxVega(); see setFluxVega() */
    };

} // namespace phot

#endif // FILTER_HPP
