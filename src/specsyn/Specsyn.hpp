/**
 * @file Specsyn.hpp
 * @author Mark Krumholz
 * @brief Defines the common interface for stellar spectral synthesis
 * @date 2026-07-18
 */

#ifndef SPECSYN_HPP
#define SPECSYN_HPP

#include <algorithm>
#include <vector>

namespace specsyn
{

    /**
     * @class Specsyn
     * @brief A base class defining a common interface for spectral synthesis
     * @details
     * This class defines a common interface to spectral synthesis
     * classes, whose purpose is to take as input a set of stellar
     * parameters produced by interpolation on the tracks, and return
     * as output stellar spectra on a pre-defined wavelength grid.
     */
    class Specsyn
    {
    public:

        /**
         * @brief Construct a Specsyn
         * @param z The redshift; defaults to zero
         */
        explicit Specsyn(double z = 0.0) : z_(z) { }

        virtual ~Specsyn() = default;
        Specsyn(const Specsyn&) = default;
        Specsyn(Specsyn&&) = default;
        auto operator=(const Specsyn&) -> Specsyn& = default;
        auto operator=(Specsyn&&) -> Specsyn& = default;

        /**
         * @brief Return the rest-frame wavelength grid
         * @return A const reference to the wavelength grid, in Angstrom
         */
        [[nodiscard]] auto wl() const -> const std::vector<double>& { return wl_; }

        /**
         * @brief Return the observed-frame wavelength grid
         * @return The wavelength grid, in Angstrom, redshifted by (1 + z)
         */
        [[nodiscard]] auto wlObs() const -> std::vector<double>
        {
            std::vector<double> wlObs(wl_.size());
            std::ranges::transform(wl_, wlObs.begin(),
                [this](const double wl) -> double { return wl * (1.0 + z_); });
            return wlObs;
        }

    protected:

        double z_;                   /**< Redshift */
        std::vector<double> wl_;     /**< Wavelength grid for the spectral synthesizer, in Angstrom */
    };

} // namespace specsyn

#endif // SPECSYN_HPP
