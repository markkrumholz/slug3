/**
 * @file testClusterSpecsynFullAFe.cpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation at alphaFe = 0.2.
 * @details
 * The alpha/Fe-interpolation counterpart to testClusterSpecsynFull:
 * identical real-data, full-scale run (see that file's own comment
 * for the shared rationale, and its required-data-file skip
 * behavior), except at alphaFe = 0.2 rather than 0.0 -- see
 * tests/core/assets/testClusterSpecsynFullAFe.in's own comment for why
 * that value specifically exercises SpecsynLibNoWind's alpha/Fe
 * interpolation path (added for BOSZ in the afe-interp-spectra
 * branch) against real BOSZ data, across the same flat
 * [Fe/H] = [-1, 0] distribution testClusterSpecsynFull uses.
 * @date 2026-07-30
 */

#include "testClusterSpecsynFullAFe.hpp"
#include "testClusterSpecsynFullCommon.hpp"

auto testClusterSpecsynFullAFe() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    return runClusterSpecsynFull(
        "tests/core/assets/testClusterSpecsynFullAFe.in", "test_cluster_specsyn_full_afe");
}
