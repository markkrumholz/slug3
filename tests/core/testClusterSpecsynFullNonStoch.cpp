/**
 * @file testClusterSpecsynFullNonStoch.cpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation with min_stoch_mass at the IMF max.
 * @details
 * The non-stochastic-sampling counterpart to testClusterSpecsynFull:
 * identical real-data, full-scale run (see that file's own comment
 * for the shared rationale, and its required-data-file skip
 * behavior), except at min_stoch_mass = 120 (chabrier.toml's own
 * maximum mass) rather than the default 0 -- see
 * tests/core/assets/testClusterSpecsynFullNonStoch.in's own comment
 * for why that value drives SimPhysics::fracStochMass() to exactly
 * zero, forcing every cluster's entire population through
 * Cluster::computeSpec()'s continuously-sampled (Specsyn::specCts())
 * path against real spectral libraries at full scale, rather than the
 * individually-sampled path testClusterSpecsynFull/AFe both exercise
 * instead.
 * @date 2026-07-30
 */

#include "testClusterSpecsynFullNonStoch.hpp"
#include "testClusterSpecsynFullCommon.hpp"

auto testClusterSpecsynFullNonStoch() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    return runClusterSpecsynFull(
        "tests/core/assets/testClusterSpecsynFullNonStoch.in",
        "test_cluster_specsyn_full_nonstoch");
}
