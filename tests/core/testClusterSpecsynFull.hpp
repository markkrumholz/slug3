/**
 * @file testClusterSpecsynFull.hpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation with real tracks/spectra.
 * @date 2026-07-24
 */

#ifndef TESTCLUSTERSPECSYNFULL_HPP
#define TESTCLUSTERSPECSYNFULL_HPP

/**
 * @brief Optional full end-to-end test of a cluster simulation with real tracks/spectra.
 * @return 0 if the test passes (including if it was skipped because the
 *   required data files are not present), 1 if it fails.
 */
auto testClusterSpecsynFull() -> int;

#endif // TESTCLUSTERSPECSYNFULL_HPP
