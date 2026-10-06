/**
 * @file MiscUtils.hpp
 * @author Mark Krumholz
 * @brief Miscellaneous utility functions
 * @date 2024-06-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef MISCUTILS_HPP
#define MISCUTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <limits>
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
     * @brief Split a search-path string into its component directories
     * @param searchPath A list of directories separated by the
     *   platform's path-list separator (':' on POSIX, ';' on Windows),
     *   as in the PATH environment variable
     * @return The non-empty entries of searchPath, in order
     */
    inline auto splitSearchPath(const std::string& searchPath) -> std::vector<std::string>
    {
#ifdef _WIN32
        constexpr char sep = ';';
#else
        constexpr char sep = ':';
#endif
        std::vector<std::string> dirs;
        std::size_t start = 0;
        while (start <= searchPath.size())
        {
            const auto end = std::min(searchPath.find(sep, start), searchPath.size());
            if (end > start) { dirs.push_back(searchPath.substr(start, end - start)); }
            start = end + 1;
        }
        return dirs;
    }

    /**
     * @brief Look for a file in the current working directory, SLUG_DIR, SLUG_DATA_PATH, or REPO_DIR
     * @param fileName File name
     * @param prefix Prefix within SLUG_DIR/SLUG_DATA_PATH/REPO_DIR to search
     * @returns Path to file
     * @details
     * This routine searches for files with the name fileName in the current
     * working directory, the directory specified by the environment variable
     * SLUG_DIR, each directory listed in the environment variable
     * SLUG_DATA_PATH, and REPO_DIR (the directory containing the project's
     * top-level CMakeLists.txt, baked in at compile time -- see
     * CMakeLists.txt), with the following resolution rules:
     * (1) If a file matching fileName exists in the current working directory, return
     *     the path to it.
     * (2) If a matching file is not found and fileName specifies an absolute path, return
     *     an empty path.
     * (3) If fileName is not an absolute path, and the environment variable SLUG_DIR
     *     is set, search for a file named SLUG_DIR/prefix/fileName, and return a path to it
     *     if found.
     * (4) If still not found, and the environment variable SLUG_DATA_PATH is set,
     *     treat it as a list of directories separated as in PATH (see
     *     splitSearchPath), and search each one, in order, for a file named
     *     dir/prefix/fileName, returning a path to the first one found.
     *     slugpy sets this (see slugpy/_paths.py) so that a pip-installed copy
     *     of slug can find both its own bundled data files and any large data
     *     downloaded into a per-user data directory.
     * (5) If still not found, and REPO_DIR is not empty, search for a file named
     *     REPO_DIR/prefix/fileName, and return a path to it if found. REPO_DIR is
     *     empty for a pip-installed build, whose source tree does not outlive
     *     installation.
     * (6) Otherwise, return an empty path.
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

        auto *slugDataPath = std::getenv("SLUG_DATA_PATH"); // NOLINT(concurrency-mt-unsafe) -- see SLUG_DIR above
        if (slugDataPath != nullptr)
        {
            for (const auto& dir : splitSearchPath(slugDataPath))
            {
                auto dataDirPath = std::filesystem::path(dir) /
                    std::filesystem::path(prefix) / filePath;
                if (std::filesystem::exists(dataDirPath)) { return dataDirPath; }
            }
        }

        const std::string repoDir = REPO_DIR;
        if (!repoDir.empty())
        {
            auto repoDirPath = std::filesystem::path(repoDir) /
                std::filesystem::path(prefix) / filePath;
            if (std::filesystem::exists(repoDirPath)) { return repoDirPath; }
        }

        return std::filesystem::path();
    }

    /**
     * @brief Generate a logarithmically spaced grid of points
     * @param xMin The minimum value of the grid
     * @param xMax The maximum value of the grid
     * @param n The number of points in the grid
     * @returns A vector of n points, logarithmically spaced from xMin
     *   to xMax inclusive; result.front() == xMin and
     *   result.back() == xMax exactly (see @details)
     * @details
     * Equivalent to numpy's logspace function (with base = e, i.e.
     * numpy's logspace(log(xMin), log(xMax), n, base=e)). xMin and
     * xMax must both be strictly positive, and n must be >= 2, since
     * the spacing between points is (log(xMax) - log(xMin)) / (n - 1);
     * neither is checked here. Every interior point is computed as
     * exp(log(xMin) + i * dLogX); the first and last points would be
     * exp(log(xMin)) and exp(log(xMax)) respectively if computed the
     * same way, neither of which is guaranteed to be bit-identical to
     * xMin/xMax (exp and log are not exact inverses at the level of
     * individual floating-point rounding), so both are instead set
     * directly to xMin/xMax. This matters because a caller may rely
     * on the returned grid never straying outside [xMin, xMax] -- e.g.
     * to stay within some other interpolant's valid range -- which a
     * few ULPs of round-off on either end could otherwise violate.
     */
    /**
     * @brief Convert a Roman numeral string to an integer
     * @param s Roman numeral string (e.g. "III", "IV", "XXVI"); must
     *   consist only of the characters I, V, X, L, C, D, M
     * @return The integer value, or 0 if s is empty or contains any
     *   unrecognized character
     */
    inline auto romanToInt(const std::string& s) -> int
    {
        if (s.empty()) { return 0; }
        const auto romanVal = [](char c) -> int {
            switch (c) {
                case 'I': return 1;
                case 'V': return 5;
                case 'X': return 10;
                case 'L': return 50;
                case 'C': return 100;
                case 'D': return 500;
                case 'M': return 1000;
                default:  return 0;
            }
        };
        int result = 0;
        int prev = 0;
        for (auto it = s.rbegin(); it != s.rend(); ++it) { // NOLINT(modernize-loop-convert) -- range-based reverse iteration requires C++20 views::reverse
            const int val = romanVal(*it);
            if (val == 0) { return 0; }
            result += (val < prev) ? -val : val;
            prev = val;
        }
        return result;
    }

    inline auto logspace(const double xMin, const double xMax, const std::size_t n) -> std::vector<double>
    {
        std::vector<double> result(n);
        result[0] = xMin; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- n >= 2 by this function's own (unchecked) precondition, so result is non-empty
        result[n - 1] = xMax; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- n >= 2 by this function's own (unchecked) precondition, so n - 1 < result.size()
        const double logXMin = std::log(xMin);
        const double logXMax = std::log(xMax);
        const double dLogX = (logXMax - logXMin) / static_cast<double>(n - 1);
        for (std::size_t i = 1; i < n - 1; ++i)
        {
            result[i] = std::exp(logXMin + (static_cast<double>(i) * dLogX)); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < n - 1 < n == result.size() by the loop bound
        }
        return result;
    }

    /**
     * @brief The smallest nonzero magnitude among a set of values
     * @param v The values to scan
     * @param fallback Value to return if every element of v is zero
     *   (or v is empty)
     * @return min |v[i]| over every nonzero v[i], or fallback if there
     *   are none
     * @details
     * Used to put a vector-valued integrand whose elements span many
     * orders of magnitude, and may include exact zeros, on a common
     * dimensionless scale before integrating it, so that a single
     * absolute tolerance (SimControls::intAbsTol()) is meaningful for
     * every element: an element that integrates to (nearly) zero can
     * then still meet the absolute tolerance, rather than failing the
     * relative one forever. Pass fallback = +infinity to combine the
     * results over several vectors with std::min.
     */
    inline auto minNonZeroMagnitude(const std::vector<double>& v, const double fallback = 1.0) -> double
    {
        double m = std::numeric_limits<double>::infinity();
        for (const double x : v) { if (x != 0.0) { m = std::min(m, std::abs(x)); } }
        return std::isfinite(m) ? m : fallback;
    }

    /**
     * @brief A characteristic scale for a vector-valued integrand, from a few sample evaluations
     * @tparam F A callable taking a double and returning a
     *   std::vector<double>
     * @param f The integrand
     * @param points The points to evaluate f at
     * @return The smallest nonzero magnitude among every element of
     *   f(x), over every x in points, but at least 1e-300 times the
     *   largest; or 1 if all are zero
     * @details
     * Dividing an integrand by this scale makes it dimensionless and
     * puts its smallest nonzero elements at order unity, so that
     * SimControls::intAbsTol(), times the integral of the weighting
     * function over the integration range, is a meaningful absolute
     * tolerance for every element -- see minNonZeroMagnitude()'s own
     * comment. The floor at 1e-300 times the largest magnitude only
     * matters if the elements span more than 300 orders of magnitude
     * (e.g. a subnormal entry alongside an ordinary one): it keeps the
     * largest scaled element at most 1e300, so it never overflows,
     * while the smallest stays representable for any span up to about
     * 1e600 -- beyond that (a subnormal alongside a value near the
     * largest double) the smallest may underflow to 0, a negligible
     * contribution in any case.
     */
    template <class F>
    auto integrandScale(const F& f, const std::vector<double>& points) -> double
    {
        constexpr double maxScaled = 1e300;
        double scale = std::numeric_limits<double>::infinity();
        double maxAbs = 0.0;
        for (const double x : points)
        {
            const auto v = f(x);
            scale = std::min(scale, minNonZeroMagnitude(v, scale));
            for (const double e : v) { maxAbs = std::max(maxAbs, std::abs(e)); }
        }
        if (!std::isfinite(scale)) { return 1.0; }
        return std::max(scale, maxAbs / maxScaled);
    }

} // namespace utils

#endif // MISCUTILS_HPP