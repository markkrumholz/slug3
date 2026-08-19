/**
 * @file testClusterSpecsynFullNonStoch.hpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation with min_stoch_mass at the IMF max.
 * @date 2026-07-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTCLUSTERSPECSYNFULLNONSTOCH_HPP
#define TESTCLUSTERSPECSYNFULLNONSTOCH_HPP

/**
 * @brief Optional full end-to-end test of a cluster simulation with min_stoch_mass at the IMF max.
 * @return 0 if the test passes (including if it was skipped because the
 *   required data files are not present), 1 if it fails.
 */
auto testClusterSpecsynFullNonStoch() -> int;

#endif // TESTCLUSTERSPECSYNFULLNONSTOCH_HPP
