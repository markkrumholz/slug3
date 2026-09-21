/**
 * @file testTrackedDeck.hpp
 * @author Mark Krumholz
 * @brief Unit tests for utils::TrackedDeck.
 * @date 2026-09-21
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTTRACKEDDECK_HPP
#define TESTTRACKEDDECK_HPP

#include "../../src/pdfs/PDF.hpp"
#include "../../src/utils/TrackedDeck.hpp"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

/**
 * @brief The paths of a list of reports, in order
 * @param reports Reports as returned by TrackedDeck::unused()/ignored()
 * @returns Each report's path_
 */
inline auto trackedDeckPaths(const std::vector<utils::DeckKeyReport>& reports) -> std::vector<std::string>
{
    std::vector<std::string> paths;
    for (const auto& r : reports) { paths.push_back(r.path_); }
    return paths;
}

/**
 * @brief Compare a list of paths against the expected one, reporting any mismatch
 * @param label Name of the check, for the failure message
 * @param actual The paths found
 * @param expected The paths expected
 * @returns 0 if they match, 1 otherwise
 */
inline auto checkTrackedDeckPaths(const std::string& label, const std::vector<std::string>& actual,
    const std::vector<std::string>& expected) -> int
{
    if (actual == expected) { return 0; }
    std::cerr << label << ": expected [";
    for (const auto& p : expected) { std::cerr << " " << p; }
    std::cerr << " ], got [";
    for (const auto& p : actual) { std::cerr << " " << p; }
    std::cerr << " ]\n";
    return 1;
}

/** @brief A small deck shared by several tests: top-level, nested, array, and doubly-nested keys */
inline auto trackedDeckSample() -> toml::table
{
    return toml::parse(R"(n_trial = 5
[output]
model_name = "x"
times = [1.0, 2.0]
[stars.sub]
IMF = 1.0
)");
}

/**
 * @brief Every leaf of a deck is reported unused before anything is read, with its line number
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Leaves are values and arrays, found at any depth; tables ([output],
 * [stars.sub]) are not themselves reported.
 */
inline auto testTrackedDeckLeaves() -> int
{
    const toml::table deck = trackedDeckSample();
    const utils::TrackedDeck tracked(deck);
    const auto unused = tracked.unused();

    int result = checkTrackedDeckPaths("testTrackedDeckLeaves (paths)", trackedDeckPaths(unused),
        { "n_trial", "output.model_name", "output.times", "stars.sub.IMF" });

    const std::vector<std::uint32_t> lines{ 1, 3, 4, 6 };
    for (std::size_t i = 0; i < unused.size() && i < lines.size(); ++i)
    {
        if (unused[i].line_ != lines[i])
        {
            std::cerr << "testTrackedDeckLeaves: " << unused[i].path_ << " on line " <<
                unused[i].line_ << ", expected " << lines[i] << "\n";
            result = 1;
        }
    }
    if (!tracked.ignored().empty())
    {
        std::cerr << "testTrackedDeckLeaves: expected nothing to be ignored initially\n";
        result = 1;
    }
    return result;
}

/**
 * @brief A deck built in code rather than parsed has no line numbers, but is tracked otherwise
 * @return 0 if the test passes, 1 if it fails
 */
inline auto testTrackedDeckNoSourcePositions() -> int
{
    const toml::table deck{ { "a", 1 }, { "b", toml::table{ { "c", 2 } } } };
    const utils::TrackedDeck tracked(deck);
    const auto unused = tracked.unused();
    int result = checkTrackedDeckPaths("testTrackedDeckNoSourcePositions", trackedDeckPaths(unused),
        { "a", "b.c" });
    for (const auto& r : unused)
    {
        if (r.line_ != 0 || r.column_ != 0)
        {
            std::cerr << "testTrackedDeckNoSourcePositions: expected no position for " << r.path_ << "\n";
            result = 1;
        }
    }
    return result;
}

/**
 * @brief Reading a leaf marks it used; looking up a table, or a missing key, marks nothing
 * @return 0 if the test passes, 1 if it fails
 * @details
 * The table lookups are the important part: SimControls tests whether
 * [yields.channel2] exists before reading it, and that must not count
 * as using the keys inside it. An index into an array marks the array.
 */
inline auto testTrackedDeckMarking() -> int
{
    const toml::table deck = trackedDeckSample();
    const utils::TrackedDeck tracked(deck);
    int result = 0;

    static_cast<void>(tracked.atPath("output"));      // a table
    static_cast<void>(tracked.atPath("stars.sub"));   // a nested table
    static_cast<void>(tracked.atPath("nosuch.key"));  // absent
    result += checkTrackedDeckPaths("testTrackedDeckMarking (tables and absent keys mark nothing)",
        trackedDeckPaths(tracked.unused()),
        { "n_trial", "output.model_name", "output.times", "stars.sub.IMF" });

    static_cast<void>(tracked.atPath("output.model_name"));
    result += checkTrackedDeckPaths("testTrackedDeckMarking (a leaf)", trackedDeckPaths(tracked.unused()),
        { "n_trial", "output.times", "stars.sub.IMF" });

    static_cast<void>(tracked.atPath("output.times[1]"));
    result += checkTrackedDeckPaths("testTrackedDeckMarking (an array element marks the array)",
        trackedDeckPaths(tracked.unused()), { "n_trial", "stars.sub.IMF" });

    // atPath must still return the real node
    if (!tracked.atPath("n_trial") || tracked.atPath("n_trial").value<int>().value_or(0) != 5)
    {
        std::cerr << "testTrackedDeckMarking: atPath did not return the node for n_trial\n";
        result = 1;
    }
    return result;
}

/**
 * @brief value() and initPDF() record the access, and behave exactly like the functions they wrap
 * @return 0 if the test passes, 1 if it fails
 */
inline auto testTrackedDeckValueAndPDF() -> int
{
    const toml::table deck = trackedDeckSample();
    const utils::TrackedDeck tracked(deck);
    int result = 0;

    if (tracked.value<unsigned long>("n_trial").value_or(0) != 5UL)
    {
        std::cerr << "testTrackedDeckValueAndPDF: value<unsigned long>(n_trial) != 5\n";
        result = 1;
    }
    if (tracked.value<double>("no.such.key").has_value())
    {
        std::cerr << "testTrackedDeckValueAndPDF: an absent optional key should give an empty optional\n";
        result = 1;
    }
    const auto pdf = tracked.initPDF("stars.sub.IMF");
    if (pdf.getMin() != 1.0 || pdf.getMax() != 1.0)
    {
        std::cerr << "testTrackedDeckValueAndPDF: initPDF did not give a delta function at 1.0\n";
        result = 1;
    }
    result += checkTrackedDeckPaths("testTrackedDeckValueAndPDF (marked used)",
        trackedDeckPaths(tracked.unused()), { "output.model_name", "output.times" });

    // Errors are unchanged: a required missing key, and a value of the wrong type
    try
    {
        static_cast<void>(tracked.value<double>("no.such.key", true));
        std::cerr << "testTrackedDeckValueAndPDF: a required missing key should throw\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }
    try
    {
        static_cast<void>(tracked.value<double>("output.model_name"));
        std::cerr << "testTrackedDeckValueAndPDF: a string read as a double should throw\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }
    return result;
}

/**
 * @brief markIgnored() moves unread keys to the ignored list; markUsedAll() consumes a whole table
 * @return 0 if the test passes, 1 if it fails
 */
inline auto testTrackedDeckIgnoredAndUsedAll() -> int
{
    const toml::table deck = trackedDeckSample();
    const utils::TrackedDeck tracked(deck);
    int result = 0;

    tracked.markIgnored("output", "output is switched off");
    result += checkTrackedDeckPaths("testTrackedDeckIgnoredAndUsedAll (unused)",
        trackedDeckPaths(tracked.unused()), { "n_trial", "stars.sub.IMF" });
    const auto ignored = tracked.ignored();
    result += checkTrackedDeckPaths("testTrackedDeckIgnoredAndUsedAll (ignored)",
        trackedDeckPaths(ignored), { "output.model_name", "output.times" });
    for (const auto& r : ignored)
    {
        if (r.reason_ != "output is switched off")
        {
            std::cerr << "testTrackedDeckIgnoredAndUsedAll: wrong reason for " << r.path_ <<
                ": " << r.reason_ << "\n";
            result = 1;
        }
    }

    // A key that is later read is used, not ignored
    static_cast<void>(tracked.atPath("output.times"));
    result += checkTrackedDeckPaths("testTrackedDeckIgnoredAndUsedAll (read after ignoring)",
        trackedDeckPaths(tracked.ignored()), { "output.model_name" });

    // A prefix must match whole path segments: "out" is not "output"
    tracked.markUsedAll("out");
    result += checkTrackedDeckPaths("testTrackedDeckIgnoredAndUsedAll (prefix is not a substring match)",
        trackedDeckPaths(tracked.unused()), { "n_trial", "stars.sub.IMF" });

    tracked.markUsedAll("stars");
    result += checkTrackedDeckPaths("testTrackedDeckIgnoredAndUsedAll (markUsedAll)",
        trackedDeckPaths(tracked.unused()), { "n_trial" });
    return result;
}

/**
 * @brief An unused key is matched to the key the code looked for: right key in the wrong section, or a misspelling
 * @return 0 if the test passes, 1 if it fails
 */
inline auto testTrackedDeckHints() -> int
{
    const toml::table deck = toml::parse(R"([output]
n_trial = 5
[stars]
v_vcirt = 0.1
zzzzzz = 1
ab = 2
)");
    const utils::TrackedDeck tracked(deck);
    // What the code reads: n_trial (top level), stars.v_vcrit, and stars.ac
    static_cast<void>(tracked.value<unsigned long>("n_trial"));
    static_cast<void>(tracked.value<double>("stars.v_vcrit"));
    static_cast<void>(tracked.value<double>("stars.ac"));

    int result = 0;
    for (const auto& r : tracked.unused())
    {
        std::string expected;  // empty means no hint expected
        if (r.path_ == "output.n_trial") { expected = "did you mean 'n_trial'?"; }
        if (r.path_ == "stars.v_vcirt") { expected = "did you mean 'stars.v_vcrit'?"; }
        // stars.zzzzzz is too far from anything, and stars.ab too short to match stars.ac
        if (r.hint_ != expected)
        {
            std::cerr << "testTrackedDeckHints: hint for " << r.path_ << " was '" << r.hint_ <<
                "', expected '" << expected << "'\n";
            result = 1;
        }
    }
    return result;
}

/**
 * @brief A key whose name contains a dot is shown quoted, and can never be read, so is always unused
 * @return 0 if the test passes, 1 if it fails
 * @details
 * toml++'s path syntax splits on every '.' with no way to quote one, so
 * no lookup can reach a key named "x.y", and no SLUG keyword is named
 * that way. Such a key in a deck must therefore be reported, however it
 * is looked up, while an ordinary key beside it is unaffected.
 */
inline auto testTrackedDeckDottedKeyName() -> int
{
    const toml::table deck = toml::parse(R"([t]
"x.y" = 1
z = 2
)");
    const utils::TrackedDeck tracked(deck);
    int result = 0;
    static_cast<void>(tracked.atPath("t.z"));
    static_cast<void>(tracked.atPath("t.x.y"));
    static_cast<void>(tracked.atPath("t.\"x.y\""));
    result += checkTrackedDeckPaths("testTrackedDeckDottedKeyName", trackedDeckPaths(tracked.unused()),
        { "t.\"x.y\"" });
    return result;
}

/**
 * @brief Run every TrackedDeck unit test
 * @return 0 if all pass, otherwise the number of failing tests
 */
inline auto testTrackedDeck() -> int
{
    int result = 0;
    result += testTrackedDeckLeaves();
    result += testTrackedDeckNoSourcePositions();
    result += testTrackedDeckMarking();
    result += testTrackedDeckValueAndPDF();
    result += testTrackedDeckIgnoredAndUsedAll();
    result += testTrackedDeckHints();
    result += testTrackedDeckDottedKeyName();
    return result;
}

#endif // TESTTRACKEDDECK_HPP
