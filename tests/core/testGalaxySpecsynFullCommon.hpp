/**
 * @file testGalaxySpecsynFullCommon.hpp
 * @author Mark Krumholz
 * @brief Shared helpers for the full end-to-end galaxy-spectral-synthesis tests.
 * @date 2026-08-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTGALAXYSPECSYNFULLCOMMON_HPP
#define TESTGALAXYSPECSYNFULLCOMMON_HPP

#include <string>

/**
 * @brief Run a full end-to-end SimGalaxy simulation and check its HDF5 output
 * @param inputFile Path to the TOML input deck to run
 * @param modelName Name to give this run's output model, used to name
 *   its output HDF5 file -- kept distinct between callers sharing this
 *   helper (and from the Cluster-side runClusterSpecsynFull's own
 *   callers) so their runs don't collide on the same output path
 * @return 0 if the run completed and its galaxy_spectra output has
 *   the expected shape and contains only finite, non-trivial spectra;
 *   1 (with a diagnostic on stderr) otherwise
 * @details
 * The Galaxy counterpart to testClusterSpecsynFullCommon.cpp's own
 * runClusterSpecsynFull() -- see its own comment for the shared
 * rationale -- reading io::OutputManagerH5's "galaxy_spectra" group
 * (wl/trial/time/spec, the same shape as "cluster_spectra") rather
 * than "clusters"/"cluster_spectra", and driving a core::SimGalaxy
 * rather than a core::SimCluster. Requires the same real (gitignored)
 * tracks/spectra data files runClusterSpecsynFull's own callers do --
 * reuse allRequiredDataFilesExist() (declared in
 * testClusterSpecsynFullCommon.hpp) to check for them before calling
 * this.
 */
auto runGalaxySpecsynFull(const std::string& inputFile, const std::string& modelName) -> int;

#endif // TESTGALAXYSPECSYNFULLCOMMON_HPP
