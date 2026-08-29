/**
 * @file testClusterSpecsynFullGrid.hpp
 * @author Mark Krumholz
 * @brief Optional full end-to-end sweep over every (track, [Fe/H], v/vcrit) combination.
 * @date 2026-08-29
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTCLUSTERSPECSYNFULLGRID_HPP
#define TESTCLUSTERSPECSYNFULLGRID_HPP

/**
 * @brief Run one full end-to-end cluster simulation per registered (track, [Fe/H], v/vcrit) combination
 * @return 0 if every combination ran to completion with finite,
 *   non-trivial cluster spectra (or the real data files this test
 *   needs are missing, in which case it is silently skipped -- see
 *   its own @details); 1 (after printing a diagnostic naming every
 *   failing combination, not just the first) otherwise
 * @details
 * testClusterSpecsynFull/AFe/NonStoch each exercise exactly one
 * hand-picked (track, [Fe/H], v/vcrit) combination -- real enough to
 * catch some bugs, but not remotely exhaustive: the hot, metal-poor
 * coverage gap in cloudy_grid_genuine_slug_failures_report.txt was a
 * genuine, reproducible slug crash that none of them happened to
 * touch, and only surfaced from an unrelated pipeline run trying
 * every combination the tracks registry actually offers.
 *
 * This test closes that gap systematically: for MIST, Stromlo, and
 * PARSEC_comp (the three track sets make_slug_grid.py's own
 * TRACK_SETS actually builds decks for -- PARSEC_vms/PARSEC_rot are
 * internal building blocks for PARSEC_comp, not something a real deck
 * selects directly), it reads that track set's own Fe_H and v_vcrit
 * arrays straight from the tracks registry (data/tracks/tracks.toml)
 * and runs one full simulation, from
 * tests/core/assets/testClusterSpecsynFullGrid.in (parsed fresh each
 * time, with [stars] tracks/FeH/v_vcrit overwritten per combination --
 * see that file's own comment for every setting shared across all of
 * them, including why min_stoch_mass=120 there matters here in
 * particular: it forces the *entire* IMF mass range through
 * Specsyn::specCts() every single time, so one trial per combination
 * deterministically exercises that track's whole Teff/logg reach at
 * that [Fe/H]/v/vcrit, rather than depending on which few masses a
 * stochastic draw happens to land on).
 *
 * A combination "passing" only means the run completed and produced
 * finite spectra with some positive flux somewhere -- there is no
 * independently-computed expected spectrum to check against (nwl=64
 * and integrator.rel_tol=0.1 in the shared deck make the numbers
 * themselves too coarse to be meaningful for that anyway); this test
 * exists purely to catch a combination that cannot be run *at all*,
 * the way the [Fe/H]=-2.75 MIST crash could not.
 *
 * Every combination is attempted even after an earlier one fails
 * (unlike testClusterSpecsynFull/AFe/NonStoch, which each check a
 * single combination and can simply return on the first problem):
 * the point of a systematic sweep is to know the full extent of any
 * gap in one run, not to stop at the first one found.
 *
 * Skipped entirely (returning 0 immediately, with a diagnostic on
 * stderr) if any of the real, gitignored data files it needs --
 * every file allRequiredDataFilesExist() already checks, plus
 * data/tracks/stromlo.h5 and data/tracks/parsec_composite.h5, which
 * none of the single-combination tests need but this one does --
 * are not present on this machine.
 */
auto testClusterSpecsynFullGrid() -> int;

#endif // TESTCLUSTERSPECSYNFULLGRID_HPP
