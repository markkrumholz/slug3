/**
 * @file testCoreFullAll.cpp
 * @author Mark Krumholz
 * @brief Slow, full-scale end-to-end tests for the classes in src/core.
 * @details
 * Counterpart to testCoreAll.cpp, for tests that are too slow to run
 * every time (full-scale runs against real, gitignored data files
 * that take upwards of a minute even when they run at all -- see e.g.
 * testClusterSpecsynFull's own doc comment for why it may not run at
 * all on a given machine). Built into its own executable, and run via
 * its own CTest entry tagged LABELS "slow" (see CMakeLists.txt),
 * specifically so a quick `ctest -L quick` (or an explicit `ctest -LE
 * slow`) run can skip this file's tests entirely rather than only
 * skipping whichever of them happen to find their own required data
 * files missing. Add future slow, full-scale core tests here as this
 * grows, following the same pattern as testCoreAll.cpp.
 * @date 2026-07-26
 */

#include "testClusterSpecsynFull.hpp"
#include "testClusterSpecsynFullAFe.hpp"
#include <exception>
#include <iostream>

auto main() -> int {
    try
    {
        int result = 0;
        result += testClusterSpecsynFull();
        result += testClusterSpecsynFullAFe();
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testCoreFullAll: uncaught exception: "
            << error.what() << "\n";
        return 1;
    }
}
