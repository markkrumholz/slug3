/**
 * @file testTracksSB99Full.hpp
 * @author Mark Krumholz
 * @brief Slow test of the legacy starburst99 track sets in data/tracks
 * @date 2026-09-24
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTTRACKSSB99FULL_HPP
#define TESTTRACKSSB99FULL_HPP

#include "../../src/tracks/TrackCommons.hpp"
#include "../../src/tracks/TrackUtils.hpp"
#include "../../src/tracks/Tracks3D.hpp"
#include "../../src/utils/HDF5Utils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "trackFieldFixture.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <vector>

// Suppress clang-tidy warnings in this file caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the
// paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

namespace testsb99
{
    /**
     * @brief Description of one legacy starburst99 track set
     */
    struct SB99Set
    {
        const char* name_;          /**< Registry name of the track set */
        std::array<double, 5> z_;   /**< Metallicities Z of the track set, in increasing order */
        double mMin_;               /**< Smallest initial mass in the track set */
        std::size_t nMass_;         /**< Number of initial masses in the track set */
    };

    // Every track set data/tools/tracks/import_sb99.py produces: the
    // four model families (Geneva standard / 2x mass loss, Padova
    // with / without TP-AGB), each with five metallicities
    inline constexpr std::array<SB99Set, 4> sb99Sets = { {
        { "sb99_modc", { 0.001, 0.004, 0.008, 0.020, 0.040 }, 0.8, 22 },
        { "sb99_mode", { 0.001, 0.004, 0.008, 0.020, 0.040 }, 0.8, 22 },
        { "sb99_modp", { 0.0004, 0.004, 0.008, 0.020, 0.050 }, 0.15, 42 },
        { "sb99_mods", { 0.0004, 0.004, 0.008, 0.020, 0.050 }, 0.15, 42 },
    } };

    inline constexpr double zSun = 0.02;   /**< Solar Z for these track sets */
    inline constexpr double mMax = 120.0;  /**< Largest initial mass in every set */

    /**
     * @brief Check that every data file this test needs exists
     * @return True if the registry and every sb99 HDF5 file exist
     */
    inline auto dataFilesExist() -> bool
    {
        const std::filesystem::path trackDir =
            std::filesystem::path(tracks::defaultRegistry).parent_path();
        std::vector<std::filesystem::path> required = { tracks::defaultRegistry };
        for (const auto& set : sb99Sets)
        {
            required.push_back(trackDir / (std::string(set.name_) + ".h5"));
        }
        for (const auto& path : required)
        {
            if (!std::filesystem::exists(path))
            {
                std::cerr << "testTracksSB99Full: required data file "
                    << path.string() << " not found; skipping this "
                    "optional full-data test\n";
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Accumulates and reports failures for one track set
     */
    struct Reporter
    {
        std::string name_;  /**< Name of the track set being checked */
        int nFail_ = 0;     /**< Number of failures reported so far */

        /**
         * @brief Report one failure
         * @param msg Description of the failure
         */
        void fail(const std::string& msg)
        {
            std::cerr << "testTracksSB99Full: " << name_ << ": " << msg << "\n";
            ++nFail_;
        }
    };

    /**
     * @brief On-disk column of the age and of each canonical track quantity
     */
    struct Columns
    {
        size_t age_;                                /**< Column holding age */
        std::array<size_t, testutil::nQty> qty_;    /**< Column holding each quantity, in tracks::FieldIdx order */

        /**
         * @brief Column holding a given canonical quantity
         * @param idx The quantity
         * @return Its on-disk column
         */
        [[nodiscard]] auto operator[](const tracks::FieldIdx idx) const -> size_t
        { return qty_.at(static_cast<size_t>(idx)); }
    };

    /**
     * @brief Read the on-disk column layout of a track group
     * @param grp Handle to an open track group
     * @return The column layout, derived from the field_names attribute
     */
    inline auto readColumns(const hid_t grp) -> Columns
    {
        const auto fieldNames = testutil::readFieldNames(grp);
        auto col = [&fieldNames](const std::string_view fld) -> size_t {
            return static_cast<size_t>(std::distance(fieldNames.begin(),
                std::ranges::find(fieldNames, std::string(fld))));
        };
        Columns cols{ .age_ = col("age"), .qty_ = {} };
        for (size_t k = 0; k < testutil::nQty; ++k)
        {
            cols.qty_.at(k) = col(tracks::fieldStr.at(k));
        }
        return cols;
    }

    /**
     * @brief Check a track set's registry entry
     * @param registry The parsed track registry
     * @param fehs Expected [Fe/H] values of the track set, in increasing order
     * @param rep Reporter for the track set
     * @details
     * The entry should list exactly the [Fe/H] values in fehs, and
     * v/vcrit = 0 only.
     */
    inline void checkRegistry(const toml::table& registry,
        const std::span<const double> fehs, Reporter& rep)
    {
        const auto* fehArr = registry.at_path(rep.name_).at_path("Fe_H").as_array();
        const auto* vvcArr = registry.at_path(rep.name_).at_path("v_vcrit").as_array();
        bool fehOk = fehArr != nullptr && fehArr->size() == fehs.size();
        for (size_t i = 0; fehOk && i < fehs.size(); ++i)
        {
            fehOk = std::fabs(fehArr->at(i).value_or(1.0e30) - fehs[i]) <= 1.0e-10;
        }
        if (!fehOk) { rep.fail("registry Fe_H does not match the expected values"); }
        if (vvcArr == nullptr || vvcArr->size() != 1 ||
            vvcArr->at(0).value_or(1.0e30) != 0.0)
        {
            rep.fail("registry v_vcrit is not [0.0]");
        }
    }

    /**
     * @brief Check one stored row of a track
     * @param row The raw stored row
     * @param actual The track quantities Tracks3D returns at that row's age
     * @param cols Column layout of row
     * @param m Initial mass of the track
     * @param rep Reporter for the track set
     * @details
     * Checks that actual reproduces the stored row, that the surface
     * abundances sum to between 0.85 and 1.2, and that the mass loss
     * rate is finite, non-negative, and below 1e-2 Msun/yr.
     */
    inline void checkRow(const std::span<const double> row,
        const std::array<double, testutil::nQty>& actual,
        const Columns& cols, const double m, Reporter& rep)
    {
        const double age = row[cols.age_];
        for (size_t k = 0; k < testutil::nQty; ++k)
        {
            if (!testutil::fieldsMatch(actual.at(k), row[cols.qty_.at(k)]))
            {
                rep.fail(std::format("m = {}, age = {}: field {} is {}, expected {}",
                    m, age, tracks::fieldStr.at(k), actual.at(k), row[cols.qty_.at(k)]));
            }
        }

        const double abundSum = row[cols[tracks::FieldIdx::hSurf]] +
            row[cols[tracks::FieldIdx::heSurf]] + row[cols[tracks::FieldIdx::cSurf]] +
            row[cols[tracks::FieldIdx::nSurf]] + row[cols[tracks::FieldIdx::oSurf]];
        if (abundSum < 0.85 || abundSum > 1.2)
        {
            rep.fail(std::format("m = {}, age = {}: surface abundances sum to {}",
                m, age, abundSum));
        }

        const double mdot = row[cols[tracks::FieldIdx::mdot]];
        if (!std::isfinite(mdot) || mdot < 0.0 || mdot > 1.0e-2)
        {
            rep.fail(std::format("m = {}, age = {}: implausible mdot {}", m, age, mdot));
        }
    }

    /**
     * @brief Check one track of a track set against its raw stored data
     * @param tracks3d The track set
     * @param feh [Fe/H] of the track set
     * @param grp Handle to the track set's open HDF5 group
     * @param cols Column layout of the group's track datasets
     * @param m Initial mass of the track to check
     * @param rep Reporter for the track set
     */
    inline void checkTrack(const tracks::Tracks3D& tracks3d, const double feh,
        const hid_t grp, const Columns& cols, const double m, Reporter& rep)
    {
        const auto [data, shape] = utils::readDataset2D(
            grp, std::format("track_m{:.3f}", m), "testTracksSB99Full");
        const auto [nRow, nCol] = shape;
        const std::span<const double> allRows(data);

        // Lifetime should be the age of the final row
        const double lifetime = tracks3d.starLifetime(m, feh);
        const double finalAge = allRows[((nRow - 1) * nCol) + cols.age_];
        if (!testutil::fieldsMatch(lifetime, finalAge))
        {
            rep.fail(std::format("m = {}: lifetime {} != final age {}",
                m, lifetime, finalAge));
        }

        // Every row should be reproduced exactly by the track
        const auto track = tracks3d.getTrack(m, feh);
        for (size_t i = 0; i < nRow; ++i)
        {
            const auto row = allRows.subspan(i * nCol, nCol);
            checkRow(row, (*track)(std::log10(row[cols.age_])), cols, m, rep);
        }
    }

    /**
     * @brief Check every track in one [Fe/H] group of a track set against its raw stored data
     * @param tracks3d The track set
     * @param feh [Fe/H] of the group to check
     * @param file Handle to the track set's open HDF5 file
     * @param nMass Expected number of initial masses in the group
     * @param rep Reporter for the track set
     */
    inline void checkGroup(const tracks::Tracks3D& tracks3d, const double feh,
        const hid_t file, const std::size_t nMass, Reporter& rep)
    {
        const std::string grpName = tracks::findTrack(rep.name_, feh);
        const hid_t grp = H5Gopen2(file, grpName.c_str(), H5P_DEFAULT);
        if (grp < 0)
        {
            rep.fail(std::format("unable to open group for [Fe/H] = {}", feh));
            return;
        }
        try
        {
            const Columns cols = readColumns(grp);
            const auto masses = utils::readDataset1D(grp, "masses", "testTracksSB99Full");
            if (masses.size() != nMass)
            {
                rep.fail(std::format("[Fe/H] = {}: {} masses, expected {}",
                    feh, masses.size(), nMass));
            }
            for (const double m : masses)
            {
                checkTrack(tracks3d, feh, grp, cols, m, rep);
            }
        }
        catch (const std::exception& e)
        {
            rep.fail(std::format("[Fe/H] = {}: unexpected exception: {}", feh, e.what()));
        }
        H5Gclose(grp);
    }

    /**
     * @brief Check that a track set can be evaluated between its tabulated [Fe/H] values
     * @param tracks3d The track set
     * @param fehs Tabulated [Fe/H] values of the track set, in increasing order
     * @param rep Reporter for the track set
     * @details
     * At the midpoint between each pair of adjacent tabulated [Fe/H]
     * values, evaluates a 20 Msun star at half its lifetime, and
     * checks that every returned quantity is finite; this exercises
     * interpolation across [Fe/H] groups, which requires every group
     * to share a single mass grid.
     */
    inline void checkOffGrid(const tracks::Tracks3D& tracks3d,
        const std::span<const double> fehs, Reporter& rep)
    {
        constexpr double m = 20.0;
        for (size_t i = 0; i + 1 < fehs.size(); ++i)
        {
            const double feh = 0.5 * (fehs[i] + fehs[i + 1]);
            const double logT = std::log10(0.5 * tracks3d.starLifetime(m, feh));
            const auto star = tracks3d.getStar(m, logT, feh);
            if (!std::ranges::all_of(star, [](const double v) -> bool { return std::isfinite(v); }))
            {
                rep.fail(std::format("non-finite star properties at m = {}, "
                    "log t = {}, [Fe/H] = {}", m, logT, feh));
            }
        }
    }

    /**
     * @brief Check one sb99 track set
     * @param registry The parsed track registry
     * @param set The track set to check
     * @return The number of failed checks
     */
    inline auto checkSet(const toml::table& registry, const SB99Set& set) -> int
    {
        Reporter rep{ .name_ = set.name_ };
        std::array<double, 5> fehs{};
        std::ranges::transform(set.z_, fehs.begin(),
            [](const double z) -> double { return std::log10(z / zSun); });
        checkRegistry(registry, fehs, rep);

        // Build the tracks spanning the full [Fe/H] range, and check
        // the mass range
        const tracks::Tracks3D tracks3d(rep.name_, fehs.front(), fehs.back());
        if (std::fabs(tracks3d.mMin() - set.mMin_) > 1.0e-10 ||
            std::fabs(tracks3d.mMax() - mMax) > 1.0e-10)
        {
            rep.fail(std::format("mass range is [{}, {}], expected [{}, {}]",
                tracks3d.mMin(), tracks3d.mMax(), set.mMin_, mMax));
        }

        // Check every group against the raw data, read independently
        // of Tracks3D
        const std::string h5Path =
            (std::filesystem::path(tracks::defaultRegistry).parent_path()
            / (rep.name_ + ".h5")).string();
        const hid_t file = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) { rep.fail("unable to open " + h5Path); return rep.nFail_; }
        for (const double feh : fehs) { checkGroup(tracks3d, feh, file, set.nMass_, rep); }
        H5Fclose(file);

        checkOffGrid(tracks3d, fehs, rep);
        return rep.nFail_;
    }
} // namespace testsb99

/**
 * @brief Slow test of the legacy starburst99 track sets
 * @return 0 if every check passes (or the data files this test needs
 *   are missing, in which case it is skipped); 1 otherwise
 * @details
 * For every track set produced by data/tools/tracks/import_sb99.py
 * (the sb99_modc, sb99_mode, sb99_modp, and sb99_mods entries in
 * data/tracks/tracks.toml), this checks that:
 * - the registry entry lists the five [Fe/H] = log10(Z / 0.02)
 *   values of the family's source files, and v/vcrit = 0 only;
 * - a Tracks3D object spanning the full [Fe/H] range can be built
 *   from it (which requires every [Fe/H] group to share one mass
 *   grid), with the expected initial mass range, and with the
 *   expected number of masses in every group;
 * - for every [Fe/H] group and every initial mass, starLifetime()
 *   matches the final age of the track, and getTrack() reproduces
 *   every stored row exactly (every mass is on the mesh's mass grid,
 *   every [Fe/H] is on its [Fe/H] grid, and every stored age is on
 *   that mass's own age grid, so no interpolation error is
 *   expected), with the raw rows read directly from HDF5
 *   independently of Tracks3D;
 * - every stored row has surface abundances summing to between 0.85
 *   and 1.2 and a finite, non-negative mass loss rate below 1e-2
 *   Msun/yr, which catches, e.g., gaps in the original data files or
 *   a misparsed log mass loss rate; and
 * - a star can be evaluated, with finite properties, at [Fe/H]
 *   values between the tabulated ones.
 *
 * Every track set is checked even after an earlier one fails, so a
 * single run reports every problem. Skipped entirely (returning 0,
 * with a diagnostic on stderr) if the registry or any of the sb99
 * HDF5 files are not present, since, like all the track data, they
 * are too large for the repository and are downloaded separately.
 */
inline auto testTracksSB99Full() -> int
{
    if (!testsb99::dataFilesExist()) { return 0; }

    int nFail = 0;
    try
    {
        const auto registry = tracks::parseRegistry().first;
        for (const auto& set : testsb99::sb99Sets)
        {
            try
            {
                nFail += testsb99::checkSet(registry, set);
            }
            catch (const std::exception& e)
            {
                std::cerr << "testTracksSB99Full: " << set.name_
                    << ": unexpected exception: " << e.what() << "\n";
                ++nFail;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracksSB99Full: unable to parse registry: "
            << e.what() << "\n";
        return 1;
    }
    return nFail > 0 ? 1 : 0;
}

// NOLINTEND(misc-include-cleaner)

#endif // TESTTRACKSSB99FULL_HPP
