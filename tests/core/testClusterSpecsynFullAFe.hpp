/**
 * @file testClusterSpecsynFullAFe.hpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation at alphaFe = 0.2.
 * @date 2026-07-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTCLUSTERSPECSYNFULLAFE_HPP
#define TESTCLUSTERSPECSYNFULLAFE_HPP

/**
 * @brief Optional full end-to-end test of a cluster simulation at alphaFe = 0.2.
 * @return 0 if the test passes (including if it was skipped because the
 *   required data files are not present), 1 if it fails.
 */
auto testClusterSpecsynFullAFe() -> int;

#endif // TESTCLUSTERSPECSYNFULLAFE_HPP
