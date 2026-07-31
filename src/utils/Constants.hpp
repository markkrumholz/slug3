/**
 * @file Constants.hpp
 * @author Mark Krumholz
 * @brief Physical constants and unit conversions, in cgs units
 * @date 2026-07-26
 */

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <gsl/gsl_const_cgsm.h> // NOLINT(misc-include-cleaner)

namespace utils
{
    // Physical constants, taken directly from GSL's cgs values
    // (gsl_const_cgsm.h)
    inline constexpr double c = GSL_CONST_CGSM_SPEED_OF_LIGHT;                  /**< Speed of light, cm/s */
    inline constexpr double h = GSL_CONST_CGSM_PLANCKS_CONSTANT_H;              /**< Planck's constant, erg s */
    inline constexpr double Ryd = GSL_CONST_CGSM_RYDBERG;                       /**< Rydberg energy, erg */ // NOLINT(readability-identifier-naming) -- capitalized to match standard astrophysical/physical notation (Ryd, G, Msun, Rsun, Lsun, Jy below), rather than the project's usual camelBack variable convention
    inline constexpr double G = GSL_CONST_CGSM_GRAVITATIONAL_CONSTANT;          /**< Gravitational constant, cm^3/g/s^2 */ // NOLINT(readability-identifier-naming) -- see Ryd
    inline constexpr double pc = GSL_CONST_CGSM_PARSEC;                         /**< Parsec, cm */
    inline constexpr double Angstrom = GSL_CONST_CGSM_ANGSTROM;                 /**< Angstrom, cm */ // NOLINT(readability-identifier-naming) -- see Ryd
    inline constexpr double sigmaSB = GSL_CONST_CGSM_STEFAN_BOLTZMANN_CONSTANT; /**< Stefan-Boltzmann constant, erg/cm^2/s/K^4 */
    inline constexpr double kB = GSL_CONST_CGSM_BOLTZMANN;                      /**< Boltzmann's constant, erg/K */
    inline constexpr double eV = GSL_CONST_CGSM_ELECTRON_VOLT;                  /**< Electron volt, erg */
    inline constexpr double Msun = GSL_CONST_CGSM_SOLAR_MASS;                   /**< Solar mass, g */ // NOLINT(readability-identifier-naming) -- see Ryd
    inline constexpr double day = GSL_CONST_CGSM_DAY;                           /**< Day, s */

    // Constants and units GSL has no cgs value for
    inline constexpr double yr = 365.25 * day; /**< Julian year (365.25 days exactly, IAU definition), s */
    inline constexpr double Rsun = 6.957e10;   /**< Solar radius (IAU 2015 nominal value, exact), cm */ // NOLINT(readability-identifier-naming) -- see Ryd
    inline constexpr double Lsun = 3.828e33;   /**< Solar luminosity (IAU 2015 nominal value, exact), erg/s */ // NOLINT(readability-identifier-naming) -- see Ryd
    inline constexpr double Jy = 1e-23;        /**< Jansky (definition, exact), erg/s/cm^2/Hz */ // NOLINT(readability-identifier-naming) -- see Ryd

} // namespace utils

#endif // CONSTANTS_HPP
