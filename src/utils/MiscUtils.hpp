/**
 * @file MiscUtils.hpp
 * @author Mark Krumholz
 * @brief Miscellaneous utility functions
 * @date 2024-06-19
 */

#ifndef MISCUTILS_HPP
#define MISCUTILS_HPP

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
namespace utils
{
    /**
     * @brief Compare two floating-point numbers for approximate equality.
     * @param a The first number to compare.
     * @param b The second number to compare.
     * @param tol The tolerance for comparison (default is 1e-6).
     * @return True if the numbers are approximately equal within the specified tolerance, false otherwise.
     */
    inline auto approxEqual(double a, double b, double tol = 1e-6) -> bool {
        return std::fabs(a - b) < tol;
    }

    /**
     * @brief Look for a file in the current working directory, SLUG_DIR, or REPO_DIR
     * @param fileName File name
     * @param prefix Prefix within SLUG_DIR/REPO_DIR to search
     * @returns Path to file
     * @details
     * This routine searches for files with the name fileName in the current
     * working directory, the directory specified by the environment variable
     * SLUG_DIR, and REPO_DIR (the directory containing the project's
     * top-level CMakeLists.txt, baked in at compile time -- see
     * CMakeLists.txt), with the following resolution rules:
     * (1) If a file matching fileName exists in the current working directory, return
     *     the path to it.
     * (2) If a matching file is not found and fileName specifies an absolute path, return
     *     an empty path.
     * (3) If fileName is not an absolute path, and the environment variable SLUG_DIR
     *     is set, search for a file named SLUG_DIR/prefix/fileName, and return a path to it
     *     if found.
     * (4) If still not found, search for a file named REPO_DIR/prefix/fileName, and
     *     return a path to it if found.
     * (5) Otherwise, return an empty path.
     */
    inline auto getFilePath(const std::string& fileName,
        const std::string& prefix = "")
    {
        std::filesystem::path filePath(fileName);
        if (std::filesystem::exists(filePath)) { return filePath; }
        if (filePath.is_absolute()) { return std::filesystem::path(); }

        auto *slugDir = std::getenv("SLUG_DIR"); // NOLINT(concurrency-mt-unsafe) -- no thread-safe standard alternative; only ever called during single-threaded setup
        if (slugDir != nullptr)
        {
            auto slugDirPath = std::filesystem::path(slugDir) /
                std::filesystem::path(prefix) / filePath;
            if (std::filesystem::exists(slugDirPath)) { return slugDirPath; }
        }

        auto repoDirPath = std::filesystem::path(REPO_DIR) /
            std::filesystem::path(prefix) / filePath;
        if (std::filesystem::exists(repoDirPath)) { return repoDirPath; }

        return std::filesystem::path();
    }

} // namespace utils

#endif // MISCUTILS_HPP