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
 * @brief Run a full end-to-end SimCluster simulation and check its HDF5 output
 * @param inputFile Path to the TOML input deck to run
 * @param modelName Name to give this run's output model, used to name
 *   its output HDF5 file -- kept distinct between callers sharing this
 *   helper so their runs don't collide on the same output path
 * @return 0 if the run completed and its output has the expected
 *   shape and contains only finite, non-trivial spectra; 1 (with a
 *   diagnostic on stderr) otherwise
 * @details
 * Shared by testClusterSpecsynFull and testClusterSpecsynFullAFe,
 * which differ only in which input deck they run (see each one's own
 * comment for what that changes) -- both decks share every other
 * setting, including n_trial = 10 and an output_times with 4 entries,
 * so those are fixed constants here rather than further parameters.
 */
auto runClusterSpecsynFull(const std::string& inputFile, const std::string& modelName) -> int;

#endif // TESTCLUSTERSPECSYNFULLCOMMON_HPP
