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
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
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

    // With one rank, the end-of-batch agreement is just this rank's
    // own flags
    for (const bool sigterm : { false, true })
    {
        for (const bool error : { false, true })
        {
            const auto stop = utils::mpiAgreeBatchStop(sigterm, error);
            if (stop.sigterm_ != sigterm || stop.error_ != error)
            {
                std::cerr << "testMPIUtilsSingleProcess: mpiAgreeBatchStop(" << sigterm
                    << ", " << error << ") should return its own inputs for one rank\n";
                return 1;
            }
        }
    }
    return 0;
}

/**
 * @brief Test rethrowAgreedError(), errorIsAgreed(), and unwindingFromLocalError() in a single process
 * @returns 0 on success, 1 on failure
 * @details
 * rethrowAgreedError() sets process-wide state that is never reset,
 * so this must run after every other test that depends on
 * errorIsAgreed() being false.
 */
inline auto testMPIUtilsAgreedError() -> int
{
    // With a single rank there are no other ranks to disagree with,
    // so even an exception in flight is never a "local" one
    try
    {
        struct CheckWhileUnwinding
        {
            bool* sawLocal_;
            CheckWhileUnwinding(const CheckWhileUnwinding&) = delete;
            CheckWhileUnwinding(CheckWhileUnwinding&&) = delete;
            auto operator=(const CheckWhileUnwinding&) -> CheckWhileUnwinding& = delete;
            auto operator=(CheckWhileUnwinding&&) -> CheckWhileUnwinding& = delete;
            explicit CheckWhileUnwinding(bool* sawLocal) : sawLocal_(sawLocal) {}
            ~CheckWhileUnwinding() { *sawLocal_ = utils::unwindingFromLocalError(); }
        };
        bool sawLocal = true;
        try
        {
            const CheckWhileUnwinding check(&sawLocal);
            throw std::runtime_error("test");
        }
        catch (const std::runtime_error&) { /* expected */ }
        if (sawLocal)
        {
            std::cerr << "testMPIUtilsAgreedError: unwindingFromLocalError() should "
                "be false with a single rank\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testMPIUtilsAgreedError: unexpected exception: " << e.what() << "\n";
        return 1;
    }

    if (utils::errorIsAgreed())
    {
        std::cerr << "testMPIUtilsAgreedError: errorIsAgreed() should be false "
            "before rethrowAgreedError() is called\n";
        return 1;
    }

    // A rank that had the error rethrows its own exception
    try
    {
        try { throw std::logic_error("local trial failure"); }
        catch (...) { utils::rethrowAgreedError(std::current_exception()); }
    }
    catch (const std::logic_error& e)
    {
        if (std::string_view(e.what()) != "local trial failure")
        {
            std::cerr << "testMPIUtilsAgreedError: rethrowAgreedError() rethrew the "
                "wrong exception: " << e.what() << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testMPIUtilsAgreedError: rethrowAgreedError() threw the wrong "
            "type for a non-null exception_ptr: " << e.what() << "\n";
        return 1;
    }
    if (!utils::errorIsAgreed())
    {
        std::cerr << "testMPIUtilsAgreedError: errorIsAgreed() should be true "
            "after rethrowAgreedError()\n";
        return 1;
    }

    // A rank that did not have the error throws a generic one
    try
    {
        utils::rethrowAgreedError(nullptr);
    }
    catch (const std::runtime_error& e)
    {
        if (!std::string_view(e.what()).contains("another MPI rank"))
        {
            std::cerr << "testMPIUtilsAgreedError: unexpected message for a null "
                "exception_ptr: " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
}

#endif // TESTMPIUTILS_HPP
