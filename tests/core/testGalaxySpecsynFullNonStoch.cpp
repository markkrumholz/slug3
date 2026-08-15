/**
 * @file testGalaxySpecsynFullNonStoch.cpp
 * @author Mark Krumholz
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @details
 * The Galaxy counterpart to testClusterSpecsynFullNonStoch: identical
 * real-data, full-scale run (same tracks/IMF/FeH distribution/spectra
 * chain, wl grid, and output times), except sim_type is "galaxy" and
 * clusters.f_cluster = 0.0 rather than stars.min_stoch_mass being
 * raised to the IMF's own maximum -- see
 * tests/core/assets/testGalaxySpecsynFullNonStoch.in's own comment.
 * Where testClusterSpecsynFullNonStoch exercises
 * Specsyn::specCts()/Cluster::computeSpec()'s single-mono-age-
 * population path, this exercises Specsyn::specAndLbolCts()/
 * Galaxy::computeSpec()'s own continuously-star-forming-history path
 * -- the joint (time, feh, mass) PDFIntegratorND integral -- against
 * real spectral libraries at full scale.
 * @date 2026-08-14
 */

#include "testGalaxySpecsynFullNonStoch.hpp"
#include "testClusterSpecsynFullCommon.hpp"
#include "testGalaxySpecsynFullCommon.hpp"

auto testGalaxySpecsynFullNonStoch() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    return runGalaxySpecsynFull(
        "tests/core/assets/testGalaxySpecsynFullNonStoch.in",
        "test_galaxy_specsyn_full_nonstoch");
}
