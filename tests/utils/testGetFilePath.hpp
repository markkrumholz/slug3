/**
 * @file testGetFilePath.hpp
 * @author Mark Krumholz
 * @brief Unit tests for utils::getFilePath's SLUG_DIR/SLUG_DATA_PATH search and utils::splitSearchPath
 * @date 2026-10-06
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTGETFILEPATH_HPP
#define TESTGETFILEPATH_HPP

#include "../../src/utils/MiscUtils.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Test utils::splitSearchPath on empty, single, and multi-entry inputs
 * @return 0 on success, 1 on failure
 */
inline auto testSplitSearchPath() -> int
{
#ifdef _WIN32
    const std::string sep = ";";
#else
    const std::string sep = ":";
#endif
    const auto check = [](const std::string& in, const std::vector<std::string>& expected) -> bool
    {
        const auto got = utils::splitSearchPath(in);
        if (got != expected)
        {
            std::cerr << "testSplitSearchPath: splitting '" << in << "' gave "
                << got.size() << " entries, expected " << expected.size() << "\n";
            return false;
        }
        return true;
    };
    bool ok = check("", {});
    ok = check("/a", { "/a" }) && ok;
    ok = check("/a" + sep + "/b/c", { "/a", "/b/c" }) && ok;
    // Empty entries (leading, trailing, or doubled separators) are dropped
    ok = check(sep + "/a" + sep + sep + "/b" + sep, { "/a", "/b" }) && ok;
    return ok ? 0 : 1;
}

#ifndef _WIN32
/**
 * @brief Set an environment variable for the lifetime of this object, restoring its previous value (or unsetting it) on destruction
 */
class ScopedEnv
{
public:
    /**
     * @brief Set (or, if value is std::nullopt, unset) an environment variable
     * @param name The variable's name
     * @param value The value to set it to, or std::nullopt to unset it
     */
    ScopedEnv(std::string name, const std::optional<std::string>& value) :
        name_(std::move(name))
    {
        const char* const old = std::getenv(name_.c_str()); // NOLINT(concurrency-mt-unsafe) -- single-threaded test
        if (old != nullptr) { old_ = std::string(old); }
        if (value.has_value()) { setenv(name_.c_str(), value->c_str(), 1); } // NOLINT(concurrency-mt-unsafe,misc-include-cleaner) -- as above; setenv is POSIX <stdlib.h>
        else { unsetenv(name_.c_str()); } // NOLINT(concurrency-mt-unsafe,misc-include-cleaner) -- as above
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv(ScopedEnv&&) = delete;
    auto operator=(const ScopedEnv&) -> ScopedEnv& = delete;
    auto operator=(ScopedEnv&&) -> ScopedEnv& = delete;
    ~ScopedEnv()
    {
        if (old_.has_value()) { setenv(name_.c_str(), old_->c_str(), 1); } // NOLINT(concurrency-mt-unsafe,misc-include-cleaner) -- as above
        else { unsetenv(name_.c_str()); } // NOLINT(concurrency-mt-unsafe,misc-include-cleaner) -- as above
    }

private:
    std::string name_;              /**< Name of the variable */
    std::optional<std::string> old_; /**< Its value before construction, if it was set */
};
#endif // _WIN32

/**
 * @brief Test that getFilePath searches SLUG_DATA_PATH's entries in order, after SLUG_DIR
 * @return 0 on success, 1 on failure
 * @details
 * Builds two scratch data roots, each holding prefix/file.toml, and
 * checks that: a file found in neither returns an empty path; the
 * first SLUG_DATA_PATH entry holding the file wins over a later one;
 * a later entry is used when an earlier one lacks the file; and
 * SLUG_DIR takes precedence over SLUG_DATA_PATH. POSIX-only, since it
 * sets environment variables via setenv/unsetenv.
 */
inline auto testGetFilePathDataPath() -> int
{
#ifdef _WIN32
    return 0;
#else
    namespace fs = std::filesystem;
    const fs::path base = fs::temp_directory_path() / "slugTestGetFilePath";
    fs::remove_all(base);
    const fs::path rootA = base / "a";
    const fs::path rootB = base / "b";
    const std::string prefix = "data/test";
    fs::create_directories(rootA / prefix);
    fs::create_directories(rootB / prefix);
    std::ofstream(rootA / prefix / "both.toml") << "x = 1\n";
    std::ofstream(rootB / prefix / "both.toml") << "x = 2\n";
    std::ofstream(rootB / prefix / "onlyB.toml") << "x = 3\n";

    int result = 0;
    const auto expect = [&result](const fs::path& got, const fs::path& want, const std::string& what)
    {
        if (got != want)
        {
            std::cerr << "testGetFilePathDataPath: " << what << ": got '" << got.string()
                << "', expected '" << want.string() << "'\n";
            result = 1;
        }
    };

    {
        const ScopedEnv noSlugDir("SLUG_DIR", std::nullopt);
        const ScopedEnv dataPath("SLUG_DATA_PATH", rootA.string() + ":" + rootB.string());
        expect(utils::getFilePath("both.toml", prefix), rootA / prefix / "both.toml",
            "first SLUG_DATA_PATH entry should win");
        expect(utils::getFilePath("onlyB.toml", prefix), rootB / prefix / "onlyB.toml",
            "later SLUG_DATA_PATH entry should be searched");
        expect(utils::getFilePath("missing.toml", prefix), fs::path(),
            "a file in no search directory should give an empty path");
    }
    {
        const ScopedEnv slugDir("SLUG_DIR", rootB.string());
        const ScopedEnv dataPath("SLUG_DATA_PATH", rootA.string());
        expect(utils::getFilePath("both.toml", prefix), rootB / prefix / "both.toml",
            "SLUG_DIR should take precedence over SLUG_DATA_PATH");
    }

    fs::remove_all(base);
    return result;
#endif // _WIN32
}

#endif // TESTGETFILEPATH_HPP
