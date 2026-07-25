/**
 * @file MiscUtils.hpp
 * @author Mark Krumholz
 * @brief Miscellaneous utility functions
 * @date 2024-06-19
 */

#ifndef MISCUTILS_HPP
#define MISCUTILS_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
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

    /**
     * @brief Generate a logarithmically spaced grid of points
     * @param xMin The minimum value of the grid
     * @param xMax The maximum value of the grid
     * @param n The number of points in the grid
     * @returns A vector of n points, logarithmically spaced from xMin
     *   to xMax inclusive (i.e. result.front() == xMin and
     *   result.back() == xMax)
     * @details
     * Equivalent to numpy's logspace function (with base = e, i.e.
     * numpy's logspace(log(xMin), log(xMax), n, base=e)). xMin and
     * xMax must both be strictly positive, and n must be >= 2, since
     * the spacing between points is (log(xMax) - log(xMin)) / (n - 1);
     * neither is checked here.
     */
    inline auto logspace(const double xMin, const double xMax, const std::size_t n) -> std::vector<double>
    {
        std::vector<double> result(n);
        const double logXMin = std::log(xMin);
        const double logXMax = std::log(xMax);
        const double dLogX = (logXMax - logXMin) / static_cast<double>(n - 1);
        for (std::size_t i = 0; i < n; ++i)
        {
            result.at(i) = std::exp(logXMin + (static_cast<double>(i) * dLogX));
        }
        return result;
    }

} // namespace utils

#endif // MISCUTILS_HPP