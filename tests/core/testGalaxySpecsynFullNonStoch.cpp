/**
 * @file testGalaxySpecsynFullNonStoch.cpp
 * @author Mark Krumholz
 * @brief Full end-to-end test of a fully continuous (f_cluster = 0) galaxy simulation.
 * @details
 * The Galaxy counterpart to testClusterSpecsynFullNonStoch: identical
 * real-data, full-scale run (same tracks/IMF/FeH distribution/spectra
 * chain, wl grid, and output times), except sim_type is "galaxy" and
 * clusters.f_cluster = 0.0 rather than stars.min_stoch_mass being
 * raised to the IMF's own maximum. Where testClusterSpecsynFullNonStoch
 * exercises Specsyn::specCts()/Cluster::computeSpec()'s single-mono-
 * age-population path, this exercises Specsyn::specAndLbolCts()/
 * Galaxy::computeSpec()'s own continuously-star-forming-history path
 * -- nested 1D integrals over age and mass, run once per [Fe/H] grid
 * point when [Fe/H] is non-degenerate (see Specsyn::specCtsHelper()'s
 * own comment) -- against real spectral libraries at full scale.
 *
 * Split into two decks/runs, rather than one covering every output
 * time and the real [Fe/H] distribution together: that full cross
 * product was found to take somewhere in the 15-25 minute range (an
 * extrapolation -- the run was killed before completing, since nothing
 * needs to wait that long to verify correctness), driven by two mostly
 * independent axes of cost (more output times; a non-degenerate [Fe/H]
 * distribution, which multiplies the whole nested integral by the
 * number of [Fe/H] grid points the tracks are defined at). Running
 * those two axes separately -- testGalaxySpecsynFullNonStochTimes.in
 * (7 output times, [Fe/H] pinned to a single value) and
 * testGalaxySpecsynFullNonStochFeH.in (a single output time, the real
 * [Fe/H] distribution) -- exercises both without paying for their full
 * product, and each one actually finishes and gets checked to
 * completion, unlike the combined version.
 * @date 2026-08-14
 */

#include "testGalaxySpecsynFullNonStoch.hpp"
#include "testClusterSpecsynFullCommon.hpp"
#include "testGalaxySpecsynFullCommon.hpp"

auto testGalaxySpecsynFullNonStoch() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    int result = runGalaxySpecsynFull(
        "tests/core/assets/testGalaxySpecsynFullNonStochTimes.in",
        "test_galaxy_specsyn_full_nonstoch_times");
    result += runGalaxySpecsynFull(
        "tests/core/assets/testGalaxySpecsynFullNonStochFeH.in",
        "test_galaxy_specsyn_full_nonstoch_feh");
    return result;
}
