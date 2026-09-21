/**
 * @file testMPIUtils.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the MPI wrappers in utils/MPIUtils.hpp.
 * @details
 * SLUG_MPI is defined only for the slug executable, not for this test
 * binary (see CMakeLists.txt), so what can be checked in-process is the
 * single-process behavior every non-MPI build takes; the multi-rank
 * behavior is checked end to end by tests/mpi/test_mpi_e2e.py.
 * @date 2026-09-21
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTMPIUTILS_HPP
#define TESTMPIUTILS_HPP

#include "../../src/utils/MPIUtils.hpp"
#include <iostream>
#include <utility>

/**
 * @brief Outside MPI, this process is the only rank, and is the I/O rank
 * @returns 0 on success, 1 on failure
 */
inline auto testMPIUtilsSingleProcess() -> int
{
    if (utils::mpiRank() != 0 || utils::mpiSize() != 1)
    {
        std::cerr << "testMPIUtilsSingleProcess: expected rank 0 of 1, got rank "
            << utils::mpiRank() << " of " << utils::mpiSize() << "\n";
        return 1;
    }
    if (!utils::isIORank())
    {
        std::cerr << "testMPIUtilsSingleProcess: isIORank() should be true "
            "for a single-process run\n";
        return 1;
    }

    // With one rank, the partition of a range is the range itself
    const auto range = utils::mpiPartitionRange(3, 10);
    if (range != std::make_pair(3UL, 10UL))
    {
        std::cerr << "testMPIUtilsSingleProcess: mpiPartitionRange(3, 10) "
            "should be [3, 10) for one rank, got [" << range.first << ", "
            << range.second << ")\n";
        return 1;
    }
    return 0;
}

#endif // TESTMPIUTILS_HPP
