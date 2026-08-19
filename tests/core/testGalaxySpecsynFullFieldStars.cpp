/**
 * @file testGalaxySpecsynFullFieldStars.cpp
 * @author Mark Krumholz
 * @brief Full end-to-end test of a partially-stochastic (field-star) galaxy simulation.
 * @details
 * The Galaxy counterpart to testGalaxySpecsynFullNonStoch, but for the
 * partially-stochastic non-clustered population -- individually-drawn
 * field stars above stars.min_stoch_mass, evaluated star by star via
 * Galaxy::getFieldStarProps()/computeSpec() -- rather than the fully
 * continuous or fully clustered extremes. See
 * tests/core/assets/testGalaxySpecsynFullFieldStars.in's own comment
 * for why clusters.f_cluster and stars.min_stoch_mass are both set to
 * genuinely intermediate values here, and why a single, modest output
 * time keeps the resulting field-star count (and hence this run's own
 * per-star spectral synthesis cost) tractable.
 * @date 2026-08-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testGalaxySpecsynFullFieldStars.hpp"
#include "testClusterSpecsynFullCommon.hpp"
#include "testGalaxySpecsynFullCommon.hpp"

auto testGalaxySpecsynFullFieldStars() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    return runGalaxySpecsynFull(
        "tests/core/assets/testGalaxySpecsynFullFieldStars.in",
        "test_galaxy_specsyn_full_field_stars");
}
