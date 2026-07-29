/**
 * @file testClusterSpecsynFull.cpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end test of a cluster simulation with real tracks/spectra.
 * @details
 * This is the first full-scale, end-to-end test of the code including
 * spectral synthesis, using the complete MIST tracks and the complete
 * POWR_WC/POWR_WNE/POWR_WNL_H20/POWR_WNL_H40/POWR_WNL_H60/TLUSTY_O/
 * TLUSTY_B/BOSZ/CK04 spectral libraries -- as opposed to every other
 * spectral-synthesis test in this repository, which uses small
 * synthetic or trimmed-down fixtures. Its purpose is to turn up any
 * gaps in coverage between
 * the tracks and the spectral libraries: for example, a star whose
 * (Teff, logg) or (Teff, transformed radius) the tracks produce but
 * none of the spectral libraries cover would surface here as an
 * out-of-bounds error from the final (OOBPolicy::raise) library in
 * the chain. CK04 is listed last, after BOSZ, since it exists
 * specifically to catch stars in one of BOSZ's own gaps (BOSZ has
 * essentially no models for Teff >~ 8000 K combined with log g
 * <~ 1.5) that this test itself found.
 *
 * The data files this test needs are too large to store in the
 * repository (see .gitignore's data/tracks and data/spectra
 * exclusions), so, mirroring tests/tracks/testTracks2D.hpp's own
 * optional-file pattern, this test runs only if every one of them is
 * present locally (i.e. has been fetched separately via
 * data/tools/fetch_mist.py, fetch_powr.py, fetch_tlusty.py,
 * fetch_bosz.py, and fetch_ck04.py); otherwise it is skipped,
 * returning an automatic pass rather than a failure. See
 * testClusterSpecsynFullAFe for the alphaFe = 0.2 counterpart of this
 * same test.
 * @date 2026-07-24
 */

#include "testClusterSpecsynFull.hpp"
#include "testClusterSpecsynFullCommon.hpp"

// Run the full simulation described by
// tests/core/assets/testClusterSpecsynFull.in (MIST tracks; a flat
// [Fe/H] = [-1, 0] distribution -- see that file's own comments;
// alphaFe = 0, v/vcrit = 0.4; a spectra.model chain of POWR_WC,
// POWR_WNE, POWR_WNL_H20, POWR_WNL_H40, POWR_WNL_H60, TLUSTY_O,
// TLUSTY_B, BOSZ, CK04, and MARCS; a CMF fixed at 10^4 Msun; output at t = 2, 3, 4,
// and 10 Myr, spanning stellar evolution before, during, and after the
// Wolf-Rayet phase; 10 trials) and check that the resulting HDF5
// output has the expected shape and contains only finite, non-trivial
// spectra. There is no independently-computed expected spectrum to
// check against here -- this test's job is to confirm the full
// pipeline runs to completion without an out-of-bounds or other
// error, which is exactly what would happen if the tracks ever
// produced a star outside every spectral library's coverage.
auto testClusterSpecsynFull() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    return runClusterSpecsynFull(
        "tests/core/assets/testClusterSpecsynFull.in", "test_cluster_specsyn_full");
}
