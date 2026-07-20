/**
 * @file SpecsynBlackbody.hpp
 * @author Mark Krumholz
 * @brief A blackbody-spectrum specialization of Specsyn
 * @date 2026-07-18
 */

#ifndef SPECSYNBLACKBODY_HPP
#define SPECSYNBLACKBODY_HPP

#include "Specsyn.hpp"
#include <cstddef>
#include <gsl/gsl_const_cgsm.h> // NOLINT(misc-include-cleaner)
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynBlackbody
     * @brief An example Specsyn specialization, for testing and validation only
     * @details
     * This class is an example specialization of the Specsyn class,
     * provided for testing and validation of the spectral synthesis
     * architecture only; it should not be used in production
     * simulations. It treats every star as a perfect blackbody
     * radiating at its effective temperature, with a radius set by
     * its luminosity and temperature via the Stefan-Boltzmann law.
     */
    class SpecsynBlackbody : public Specsyn
    {
        // Physical constants, in cgs units, used by this class
        static constexpr double planckH = GSL_CONST_CGSM_PLANCKS_CONSTANT_H; /**< Planck constant, erg s */
        static constexpr double speedOfLight = GSL_CONST_CGSM_SPEED_OF_LIGHT; /**< Speed of light, cm/s */
        static constexpr double rydberg = GSL_CONST_CGSM_RYDBERG;            /**< Rydberg energy, erg */
        static constexpr double boltzmannK = GSL_CONST_CGSM_BOLTZMANN;       /**< Boltzmann constant, erg/K */

        // Number of points in the wavelength grid
        static constexpr std::size_t nWl = 1000;

    public:

        /**
         * @brief Construct a SpecsynBlackbody
         * @param z The redshift; defaults to zero
         * @details
         * Sets wl_ to a grid of nWl points, logarithmically spaced
         * from hc / (10 Ry) to hc / (0.01 Ry).
         */
        explicit SpecsynBlackbody(double z = 0.0);

        /**
         * @brief Compute the blackbody spectrum of a single star
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom
         */
        [[nodiscard]] auto spec(const StarData& props) const -> std::vector<double> override;
    };

} // namespace specsyn

#endif // SPECSYNBLACKBODY_HPP
