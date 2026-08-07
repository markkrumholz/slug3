/**
 * @file testClusterSpecsynFullCommon.hpp
 * @author Mark Krumholz
 * @brief Shared helpers for the full end-to-end cluster-spectral-synthesis tests.
 * @date 2026-07-30
 */

#ifndef TESTCLUSTERSPECSYNFULLCOMMON_HPP
#define TESTCLUSTERSPECSYNFULLCOMMON_HPP

#include <string>

/**
 * @brief Check that every data file the full-scale cluster tests need is present
 * @return true if every required file exists; false (after printing a
 *   diagnostic identifying the first missing one) otherwise
 * @details
 * Both testClusterSpecsynFull and testClusterSpecsynFullAFe need the
 * exact same set of real, gitignored tracks/spectra data files (see
 * runClusterSpecsynFull's own comment for why they can share this),
 * so this check is shared between them rather than duplicated.
 */
auto allRequiredDataFilesExist() -> bool;

/**
 * @brief Check that the real photometry data files testClusterSpecsynFull needs are present
 * @return true if every required file exists; false (after printing a
 *   diagnostic identifying the first missing one) otherwise
 * @details
 * Separate from allRequiredDataFilesExist() above, since only
 * testClusterSpecsynFull's own deck has a [phot] section -- gating
 * testClusterSpecsynFullAFe/NonStoch on this too would skip them
 * unnecessarily on a machine that has fetched tracks/spectra data but
 * not the (also gitignored) real filter registry/Vega spectrum.
 */
auto allPhotDataFilesExist() -> bool;

/**
 * @brief Run a full end-to-end SimCluster simulation and check its HDF5 output
 * @param inputFile Path to the TOML input deck to run
 * @param modelName Name to give this run's output model, used to name
 *   its output HDF5 file -- kept distinct between callers sharing this
 *   helper so their runs don't collide on the same output path
 * @return 0 if the run completed and its output has the expected
 *   shape and contains only finite, non-trivial spectra (and, if the
 *   deck requested photometry, only finite cluster_phot values); 1
 *   (with a diagnostic on stderr) otherwise
 * @details
 * Shared by testClusterSpecsynFull, testClusterSpecsynFullAFe, and
 * testClusterSpecsynFullNonStoch, which differ only in which input
 * deck they run (see each one's own comment for what that changes).
 * n_trial is read back from the deck itself (via SimControls) rather
 * than assumed, since not every caller's deck uses the same value --
 * testClusterSpecsynFullNonStoch's uses n_trial = 1, since its
 * min_stoch_mass setting makes every trial's stellar population
 * non-stochastic, so multiple trials would just repeat the same
 * exercise. The number of output times is likewise read back from the
 * deck (via SimControls::outTimes().size()) rather than assumed,
 * since testClusterSpecsynFull/NonStoch's decks now list 7 entries
 * (to also exercise white dwarf coverage at 1e8-1e10 yr) while
 * testClusterSpecsynFullAFe's still lists the original 4. The
 * cluster_phot check is conditional on that group actually existing
 * in the output, since only testClusterSpecsynFull's own deck
 * requests photometry.
 */
auto runClusterSpecsynFull(const std::string& inputFile, const std::string& modelName) -> int;

#endif // TESTCLUSTERSPECSYNFULLCOMMON_HPP
