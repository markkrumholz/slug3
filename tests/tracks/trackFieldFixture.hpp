/**
 * @file trackFieldFixture.hpp
 * @author Mark Krumholz
 * @brief Test helper for reading raw track field data directly from HDF5
 * @date 2026-07-11
 * @details
 * This header provides a way to read a single row of raw track data
 * from an HDF5 track file, independently of the (fixed, but
 * previously buggy) field-column-mapping logic in Tracks2D.cpp and
 * Tracks3D.cpp, so that tests can check the output of those classes
 * against ground truth read straight from the file.
 */

#ifndef TRACKFIELDFIXTURE_HPP
#define TRACKFIELDFIXTURE_HPP

#include "../../src/tracks/TrackCommons.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace testutil
{
    constexpr size_t nQty = static_cast<size_t>(tracks::FieldIdx::nTrackQty);

    // Suppress clang-tidy warnings iun this namespace caused by just including
    // hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
    // that HDF5 wants
    // NOLINTBEGIN(misc-include-cleaner)

    /**
     * @brief Read the field_names attribute of an HDF5 group
     * @param grp Handle to the group
     * @returns The field names, in on-disk column order
     * @details
     * This duplicates, rather than reuses, the equivalent (private)
     * logic in TrackUtils.cpp/Tracks3D.cpp, so that this fixture
     * provides ground truth that is independent of the loading code
     * under test.
     */
    inline auto readFieldNames(const hid_t grp) -> std::vector<std::string>
    {
        const hid_t attr = H5Aopen(grp, "field_names", H5P_DEFAULT);
        if (attr < 0)
        {
            throw std::runtime_error(
                "trackFieldFixture: unable to open field_names attribute");
        }
        const hid_t aspace = H5Aget_space(attr);
        const auto npoints =
            static_cast<size_t>(H5Sget_simple_extent_npoints(aspace));
        const hid_t memtype = H5Aget_type(attr);

        std::vector<char*> buf(npoints);
        H5Aread(attr, memtype, static_cast<void *>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        std::vector<std::string> names;
        names.reserve(npoints);
        for (const auto* s : buf) { names.emplace_back(s); }

        H5Dvlen_reclaim(memtype, aspace, H5P_DEFAULT,
            static_cast<void *>(buf.data())); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        H5Tclose(memtype);
        H5Sclose(aspace);
        H5Aclose(attr);

        return names;
    }

    /**
     * @brief Read one row of raw track data for a given mass
     * @param grp Handle to an open feh/afe/vvcrit group
     * @param mass The (exact, on-grid) mass whose track dataset to read
     * @param rowIdx The row to read
     * @returns The age at that row, and the nQty canonical field
     *   values (in tracks::FieldIdx order) at that row
     * @details
     * Independently (re-)derives the on-disk column holding the age
     * field and each of the nQty canonical quantities from the
     * field_names attribute, rather than relying on the mapping built
     * by Tracks2D/Tracks3D, so that the values this returns serve as
     * ground truth for testing that mapping.
     */
    inline auto readRawFields(const hid_t grp, const double mass, const size_t rowIdx)
        -> std::pair<double, std::array<double, nQty>>
    {
        const auto fieldNames = readFieldNames(grp);

        const auto ageIt = std::ranges::find(fieldNames, std::string("age"));
        if (ageIt == fieldNames.end())
        {
            throw std::runtime_error("trackFieldFixture: group has no 'age' field");
        }
        const auto ageCol = static_cast<size_t>(std::distance(fieldNames.begin(), ageIt));

        std::array<size_t, nQty> qtyCol{};
        for (size_t k = 0; k < nQty; ++k)
        {
            const auto it = std::ranges::find(fieldNames, std::string(tracks::fieldStr.at(k)));
            if (it == fieldNames.end())
            {
                throw std::runtime_error(
                    "trackFieldFixture: group is missing field " +
                    std::string(tracks::fieldStr.at(k)));
            }
            qtyCol.at(k) = static_cast<size_t>(std::distance(fieldNames.begin(), it));
        }

        const auto name = std::format("track_m{:.3f}", mass);
        const hid_t dset = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
        {
            throw std::runtime_error("trackFieldFixture: unable to open dataset " + name);
        }
        const hid_t space = H5Dget_space(dset);
        std::array<hsize_t, 2> dims = {0, 0};
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        if (rowIdx >= dims[0])
        {
            H5Sclose(space);
            H5Dclose(dset);
            throw std::runtime_error("trackFieldFixture: row index out of range");
        }

        const std::array<hsize_t, 2> start = {static_cast<hsize_t>(rowIdx), 0};
        const std::array<hsize_t, 2> count = {1, dims[1]};
        H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);
        const hid_t memspace = H5Screate_simple(2, count.data(), nullptr);
        std::vector<double> row(dims[1]);
        H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, space, H5P_DEFAULT, row.data());
        H5Sclose(memspace);
        H5Sclose(space);
        H5Dclose(dset);

        const double age = row.at(ageCol);
        std::array<double, nQty> values{};
        for (size_t k = 0; k < nQty; ++k) { values.at(k) = row.at(qtyCol.at(k)); }

        return { age, values };
    }

    /**
     * @brief Compare two field values for approximate equality
     * @param actual The value returned by the code under test
     * @param expected The ground-truth value
     * @returns True if actual and expected agree to within a combined
     *   relative and absolute tolerance
     * @details
     * Track fields span many orders of magnitude (e.g. mdot ~1e-10 vs.
     * age ~1e8), so a purely absolute tolerance is either too loose
     * for small quantities or too tight for large ones; this combines
     * both, matching the standard technique for comparing
     * floating-point values across scales.
     */
    inline auto fieldsMatch(const double actual, const double expected) -> bool
    {
        const double scale = std::max(std::fabs(actual), std::fabs(expected));
        return std::fabs(actual - expected) <= (1e-8 * scale + 1e-12);
    }

    // NOLINTEND(misc-include-cleaner)

} // namespace testutil

#endif // TRACKFIELDFIXTURE_HPP
