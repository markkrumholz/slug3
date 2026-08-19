/**
 * @file testGalaxySpecsynFullNonStoch.hpp
 * @author Mark Krumholz
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @date 2026-08-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTGALAXYSPECSYNFULLNONSTOCH_HPP
#define TESTGALAXYSPECSYNFULLNONSTOCH_HPP

/**
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @return 0 if both of this test's own two runs pass (including if
 *   both were skipped because the required data files are not
 *   present), 1 if either fails -- see the .cpp file's own comment
 *   for why there are two.
 */
auto testGalaxySpecsynFullNonStoch() -> int;

#endif // TESTGALAXYSPECSYNFULLNONSTOCH_HPP
