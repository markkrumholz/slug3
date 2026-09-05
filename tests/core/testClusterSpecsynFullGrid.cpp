/**
 * @file testClusterSpecsynFullGrid.cpp
 * @author Mark Krumholz
 * @brief Implementation of testClusterSpecsynFullGrid.hpp
 * @date 2026-08-29
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "testClusterSpecsynFullGrid.hpp"
#include "../src/core/SimCluster.hpp"
#include "../src/io/OutputManager.hpp"
#include "../src/io/OutputManagerH5.hpp"
#include "../src/io/SimControls.hpp"
#include "../src/tracks/TrackUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "testClusterSpecsynFullCommon.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace
{
    // Additional real data files this sweep needs beyond what
    // allRequiredDataFilesExist() already checks (MIST + every
    // spectral library) -- Stromlo and PARSEC_comp, the other two
    // track sets make_slug_grid.py's own TRACK_SETS builds decks for.
    const std::array<std::string, 2> extraRequiredDataFiles = { { // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- see requiredDataFiles's own identical NOLINT in testClusterSpecsynFullCommon.cpp
        "data/tracks/stromlo.h5",
        "data/tracks/parsec_composite.h5",
    }};

    auto extraRequiredDataFilesExist() -> bool
    {
        for (const auto& path : extraRequiredDataFiles)
        {
            if (!std::filesystem::exists(path))
            {
                std::cerr << "testClusterSpecsynFullGrid: required data file "
                    << path << " not found; skipping this optional "
                    "full-data end-to-end test\n";
                return false;
            }
        }
        return true;
    }

    // Read a track's own registry entry's array field (e.g. "Fe_H",
    // "v_vcrit") as a vector of doubles, ignoring any non-numeric
    // entries -- mirrors utils::getStringArrayField/stringArrayContents
    // (TOMLUtils.hpp), which only handle strings, not numbers.
    auto doubleArrayField(
        const toml::table& registry, const std::string& trackName, const std::string& key)
        -> std::vector<double>
    {
        std::vector<double> result;
        const auto* arr = registry.at_path(trackName).at_path(key).as_array();
        if (arr != nullptr)
        {
            arr->for_each([&result](auto&& el) -> void {
                if constexpr (toml::is_number<decltype(el)>)
                {
                    // el's static type here is toml::value<int64_t> or
                    // toml::value<double> -- its own leaf class is
                    // named "value", so calling the inherited
                    // node::value<double>() accessor by that same name
                    // is ambiguous; operator* (returning the
                    // underlying int64_t/double directly) sidesteps
                    // that name collision entirely.
                    result.push_back(static_cast<double>(*el));
                }
            });
        }
        return result;
    }

    // One (track, [Fe/H], v/vcrit) combination to run
    struct Combo
    {
        std::string track_;
        double feh_;
        double vvcrit_;
    };

    // Every combination the tracks registry offers for the three
    // track sets make_slug_grid.py's own TRACK_SETS builds decks for
    // -- see testClusterSpecsynFullGrid()'s own comment for why just
    // these three. Each track set's own Fe_H and v_vcrit arrays are
    // read straight from the registry, so this automatically tracks
    // the registry's own current grid (e.g. Stromlo's [Fe/H] max
    // dropping to +0.5 after PR #184) rather than needing to be kept
    // in sync by hand.
    auto allCombos() -> std::vector<Combo>
    {
        static constexpr std::array<const char*, 3> trackSets = { "MIST", "Stromlo", "PARSEC_comp" };
        const auto registry = tracks::parseRegistry().first;

        std::vector<Combo> combos;
        for (const auto* track : trackSets)
        {
            const auto fehVals = doubleArrayField(registry, track, "Fe_H");
            // Mirrors make_slug_grid.py's own get_vvcrit_grid(): a
            // track set with no v_vcrit axis at all in the registry
            // (e.g. PARSEC_comp before it gained an explicit one) is
            // still v/vcrit=0.0, not a track set to skip entirely.
            auto vvcritVals = doubleArrayField(registry, track, "v_vcrit");
            if (vvcritVals.empty()) { vvcritVals.push_back(0.0); }
            if (fehVals.empty())
            {
                // A required track set with no valid Fe_H values (the
                // key is absent, empty, or every entry is
                // non-numeric) is a registry problem, not a track set
                // to skip -- silently continuing here would silently
                // shrink this test's own coverage to less than it
                // claims, defeating its whole purpose.
                throw std::runtime_error(
                    "testClusterSpecsynFullGrid: track set " + std::string(track) +
                    " has no valid Fe_H values in the registry");
            }
            for (const double feh : fehVals)
            {
                for (const double vvcrit : vvcritVals)
                {
                    combos.push_back({ track, feh, vvcrit });
                }
            }
        }
        return combos;
    }

    // Look up table's own sub-table at key, inserting an empty one
    // first if it doesn't already exist -- mirrors the same
    // check-or-insert idiom runClusterSpecsynFull() (testClusterSpecsynFullCommon.cpp)
    // uses for exactly the same reason: the shared base deck may or
    // may not already have a given top-level table.
    auto tableAt(toml::table& table, const std::string& key) -> toml::table&
    {
        if (toml::table* existing = table[key].as_table()) { return *existing; }
        return *table.insert(key, toml::table{}).first->second.as_table();
    }

    // Run a single combination's full deck, returning an empty string
    // on success or a diagnostic message on failure. Never throws --
    // every failure, whether an exception from the simulation itself
    // or a problem found in its output, is converted to a returned
    // message instead -- so the caller can attempt every combination
    // even after an earlier one fails; see this test's own header
    // comment for why that matters here specifically.
    auto runOneCombo(const toml::table& baseDeck, const Combo& combo,
        const std::filesystem::path& outDir, const std::filesystem::path& h5Path) -> std::string
    {
        try
        {
            toml::table inputDeck = baseDeck;
            toml::table& starsTbl = tableAt(inputDeck, "stars");
            starsTbl.insert_or_assign("tracks", combo.track_);
            starsTbl.insert_or_assign("FeH", combo.feh_);
            starsTbl.insert_or_assign("v_vcrit", combo.vvcrit_);
            tableAt(inputDeck, "output").insert_or_assign("model_name", std::string("grid_test"));
            tableAt(inputDeck, "output").insert_or_assign("out_dir", outDir.string());

            std::filesystem::remove_all(outDir);
            std::filesystem::create_directories(outDir);

            {
                // Scoped so simCluster -- and the OutputManagerH5 it
                // owns -- is destroyed, and its OpenMP per-thread
                // files thereby consolidated into h5Path, before the
                // reopen attempt below -- see runClusterSpecsynFull()'s
                // own identical comment.
                const io::SimControls simControls(inputDeck);
                std::unique_ptr<io::OutputManager> outputManager =
                    std::make_unique<io::OutputManagerH5>(simControls, inputDeck);
                core::SimCluster simCluster(simControls, std::move(outputManager));
                simCluster.run();
            }

            // NOLINTBEGIN(misc-include-cleaner)
            const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file < 0) { return "unable to reopen output file"; }

            const hid_t specGrp = H5Gopen2(file, "cluster_spectra", H5P_DEFAULT);
            const hid_t dset = H5Dopen2(specGrp, "spec", H5P_DEFAULT);
            const hid_t space = H5Dget_space(dset);
            std::array<hsize_t, 2> dims{};
            H5Sget_simple_extent_dims(space, dims.data(), nullptr);
            std::vector<double> spec(static_cast<size_t>(dims.at(0) * dims.at(1)));
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, spec.data());
            H5Sclose(space);
            H5Dclose(dset);
            H5Gclose(specGrp);
            H5Fclose(file);
            // NOLINTEND(misc-include-cleaner)

            const auto nRows = dims.at(0);
            const auto nCols = dims.at(1);
            if (nRows == 0 || nCols == 0) { return "cluster_spectra/spec is empty"; }
            // Positive flux is checked across the *whole* spec array,
            // not row by row: unlike testClusterSpecsynFull(AFe)'s own
            // single hand-picked combination (chosen so its population
            // stays alive at every one of its own output times), this
            // sweep's shared output_times run out to 1e10 yr for every
            // track set alike, including Stromlo, whose own mass floor
            // (10 Msun) means every star in any Stromlo-based
            // population has already died by 1e8 yr -- a row of
            // legitimately all-zero flux there, not a coverage bug.
            // Every value still has to be finite everywhere, and the
            // dataset as a whole still has to have positive flux
            // *somewhere*, so a genuinely broken (all-zero throughout)
            // run still fails.
            bool anyPositive = false;
            for (hsize_t row = 0; row < nRows; ++row)
            {
                for (hsize_t col = 0; col < nCols; ++col)
                {
                    const double value = spec.at((static_cast<size_t>(row) * nCols) + static_cast<size_t>(col));
                    if (!std::isfinite(value))
                    {
                        std::ostringstream msg;
                        msg << "spec row " << row << ", column " << col << " is not finite (" << value << ")";
                        return msg.str();
                    }
                    anyPositive = anyPositive || (value > 0.0);
                }
            }
            if (!anyPositive) { return "no positive flux anywhere in the whole spec dataset"; }
        }
        catch (const std::exception& error)
        {
            return error.what();
        }
        return {};
    }
} // namespace

auto testClusterSpecsynFullGrid() -> int
{
    if (!allRequiredDataFilesExist()) { return 0; }
    if (!extraRequiredDataFilesExist()) { return 0; }

    // Every exception below (including allCombos() throwing on a
    // registry with no valid Fe_H values for a required track set) is
    // caught and reported as a plain failing return value, not left
    // to propagate -- testCoreFullAll()'s own main() runs this
    // alongside several other slow tests in one uninterrupted
    // sequence (result += testX() for each), so an uncaught exception
    // here would abort every later test in that sequence rather than
    // just this one.
    try
    {
        const toml::table baseDeck = toml::parse_file("tests/core/assets/testClusterSpecsynFullGrid.in");
        const auto outDir = std::filesystem::temp_directory_path() / "slugTestClusterSpecsynFullGrid";
        const auto h5Path = outDir / "grid_test.h5";

        const auto combos = allCombos();
        std::cout << "testClusterSpecsynFullGrid: running " << combos.size()
            << " (track, [Fe/H], v/vcrit) combinations\n";

        std::vector<std::string> failures;
        for (const auto& combo : combos)
        {
            const auto message = runOneCombo(baseDeck, combo, outDir, h5Path);
            if (!message.empty())
            {
                std::ostringstream line;
                line << "track=" << combo.track_ << " FeH=" << combo.feh_
                    << " v_vcrit=" << combo.vvcrit_ << ": " << message;
                failures.push_back(line.str());
            }
        }

        if (!failures.empty())
        {
            std::cerr << "testClusterSpecsynFullGrid: " << failures.size() << " of "
                << combos.size() << " combinations failed:\n";
            for (const auto& failure : failures) { std::cerr << "  " << failure << "\n"; }
            return 1;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "testClusterSpecsynFullGrid: uncaught exception: " << error.what() << "\n";
        return 1;
    }
}
