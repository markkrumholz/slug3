/**
 * @file testGalaxySpecsynFullNonStoch.hpp
 * @author Mark Krumholz
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @date 2026-08-14
 */

#ifndef TESTGALAXYSPECSYNFULLNONSTOCH_HPP
#define TESTGALAXYSPECSYNFULLNONSTOCH_HPP

/**
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @return 0 if the test passes (including if it was skipped because the
 *   required data files are not present), 1 if it fails.
 */
auto testGalaxySpecsynFullNonStoch() -> int;

#endif // TESTGALAXYSPECSYNFULLNONSTOCH_HPP
