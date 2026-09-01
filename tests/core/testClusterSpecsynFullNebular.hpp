/**
 * @file testClusterSpecsynFullNebular.hpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of nebular emission against the real cloudy grid.
 * @date 2026-08-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTCLUSTERSPECSYNFULLNEBULAR_HPP
#define TESTCLUSTERSPECSYNFULLNEBULAR_HPP

/**
 * @brief Optional full end-to-end test of nebular emission against the real cloudy grid.
 * @return 0 if the test passes (including if it was skipped because the
 *   required data files -- real tracks/spectra and/or the real
 *   data/nebular/nebular.h5 cloudy grid -- are not present locally --
 *   see the .cpp file's own comment), 1 if it fails.
 */
auto testClusterSpecsynFullNebular() -> int;

#endif // TESTCLUSTERSPECSYNFULLNEBULAR_HPP
