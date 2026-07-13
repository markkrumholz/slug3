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
     * @brief Look for a file both in the current working directory and in SLUG_DIR
     * @param fileName File name
     * @param prefix Prefix within in SLUG_DIR to search
     * @returns Path to file
     * @details
     * This routine searches for files with the name fileName in both the current
     * working directory and in the directory specified by the environment variable
     * SLUG_DIR, with the following resolution rules: 
     * (1) If a file matching fileName exists in the current working directory, return 
     *     the path to it.
     * (2) If a matching file is not found and fileName specifies an absolute path, return
     *     an empty path.
     * (3) If fileName is not an absolute path, and the environment variable SLUG_DIR
     *     is set, search for a file named SLUG_DIR/prefix/fileName, and return a path to it
     *     if found.
     * (4) Otherwise, return an empty path.
     */
    inline auto getFilePath(const std::string& fileName,
        const std::string& prefix = "")
    {
        std::filesystem::path filePath(fileName);
        if (std::filesystem::exists(filePath)) { return filePath; }
        if (filePath.is_absolute()) { return std::filesystem::path(); }
        auto slugDir = std::getenv("SLUG_DIR");
        if (!slugDir) { return std::filesystem::path(); }
        filePath = std::filesystem::path(slugDir) / 
            std::filesystem::path(prefix) / filePath;
        if (std::filesystem::exists(filePath)) { return filePath; }
        return std::filesystem::path();
    }

} // namespace utils

#endif // MISCUTILS_HPP