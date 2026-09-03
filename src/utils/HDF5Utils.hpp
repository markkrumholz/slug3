/**
 * @file HDF5Utils.hpp
 * @author Mark Krumholz
 * @brief Shared utility functions for reading and writing HDF5 files
 * @date 2026-08-01
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef HDF5UTILS_HPP
#define HDF5UTILS_HPP

// Suppress clang-tidy warnings caused by just including hdf5.h, instead of
// the individual HDF5 headers, since this is the paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)
#include "hdf5.h"
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace utils
{
    /**
     * @brief Read a scalar double attribute from an HDF5 object, if present
     * @param obj Handle to the object (a group or a dataset)
     * @param name Name of the attribute
     * @returns The attribute's value, or an empty optional if the object
     *   has no attribute of that name
     */
    auto readScalarAttrIfPresent(hid_t obj, const std::string& name) -> std::optional<double>;

    /**
     * @brief Read a required scalar double attribute from an HDF5 object
     * @param obj Handle to the object (a group or a dataset)
     * @param name Name of the attribute
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The attribute's value
     * @throws std::runtime_error if obj has no attribute of that name
     */
    auto readRequiredScalarAttr(hid_t obj, const std::string& name,
        const std::string& context) -> double;

    /**
     * @brief Read a 1D double dataset from an HDF5 group (or file)
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents
     */
    auto readDataset1D(hid_t grp, const std::string& name,
        const std::string& context) -> std::vector<double>;

    /**
     * @brief Get the rank (number of dimensions) of a dataset without reading its data
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset's rank
     */
    auto datasetRank(hid_t grp, const std::string& name, const std::string& context) -> int;

    /**
     * @brief Get the shape of a 2D dataset without reading its data
     * @param grp Handle to the group containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The (nrow, ncol) shape of the dataset
     */
    auto dataset2DShape(hid_t grp, const std::string& name,
        const std::string& context) -> std::pair<size_t, size_t>;

    /**
     * @brief Read a full 2D double dataset from an HDF5 group
     * @param grp Handle to the group containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents, and its (nrow, ncol) shape
     */
    auto readDataset2D(hid_t grp, const std::string& name, const std::string& context)
        -> std::pair<std::vector<double>, std::pair<size_t, size_t>>;

    /**
     * @brief Read a 1D fixed-length string dataset from an HDF5 group (or file)
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset's strings, each trimmed at its own first null byte
     */
    auto readStringDataset1D(hid_t grp, const std::string& name,
        const std::string& context) -> std::vector<std::string>;

    /**
     * @brief Read a full 3D double dataset from an HDF5 group
     * @param grp Handle to the group (or already-open file) containing the dataset
     * @param name Name of the dataset
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The dataset contents, flattened in the same row-major
     *   (C) order HDF5/h5py itself stores a multidimensional dataset
     *   in -- i.e. element (i, j, k) of a dataset shaped (n0, n1, n2)
     *   is at flat index (i * n1 + j) * n2 + k -- and its (n0, n1, n2) shape
     */
    auto readDataset3D(hid_t grp, const std::string& name, const std::string& context)
        -> std::pair<std::vector<double>, std::array<hsize_t, 3>>;

    /**
     * @brief Read the field_names attribute of an HDF5 group
     * @param grp Handle to the group
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The names of the fields stored in each track dataset,
     *   in the order in which they appear
     */
    auto readFieldNames(hid_t grp, const std::string& context) -> std::vector<std::string>;

    /**
     * @brief List the names of every dataset directly inside an HDF5 group
     * @param grp Handle to the group
     * @returns The names of the group's child datasets
     */
    auto listGroupDatasetNames(hid_t grp) -> std::vector<std::string>;

    /**
     * @brief Create (and return) a variable-length, UTF-8 HDF5 string datatype
     * @returns The new datatype's handle
     */
    auto vlenStrType() -> hid_t;

    /**
     * @brief Create (and return) a fixed-length HDF5 string datatype
     * @param size Size of the datatype, in bytes
     * @returns The new datatype's handle
     * @details
     * Used for e.g. an "rng" dataset, whose elements are
     * utils::RngState's own fixed-width, null-padded buffers rather
     * than variable-length text.
     */
    auto fixedStrType(size_t size) -> hid_t;

    /**
     * @brief Write a scalar string attribute on an HDF5 object
     * @param loc Handle to the object (a group, dataset, or file) to
     *   attach the attribute to
     * @param name Name of the attribute
     * @param value Value of the attribute
     * @throws std::runtime_error if the attribute cannot be created
     */
    void writeStringAttr(hid_t loc, const std::string& name, const std::string& value);

    /**
     * @brief Write a scalar unsigned long attribute on an HDF5 object
     * @param loc Handle to the object (a group, dataset, or file) to
     *   attach the attribute to
     * @param name Name of the attribute
     * @param value Value of the attribute
     * @throws std::runtime_error if the attribute cannot be created
     */
    void writeULongAttr(hid_t loc, const std::string& name, unsigned long value);

    /**
     * @brief Read a scalar unsigned long attribute from an HDF5 object
     * @param loc Handle to the object (a group, dataset, or file) the
     *   attribute is attached to
     * @param name Name of the attribute
     * @returns The attribute's value
     * @throws std::runtime_error if the attribute does not exist or
     *   cannot be opened
     * @details
     * The inverse of writeULongAttr().
     */
    auto readULongAttr(hid_t loc, const std::string& name) -> unsigned long;

    /**
     * @brief Write a scalar string dataset into an HDF5 group
     * @param loc Handle to the group (or file) to create the dataset in
     * @param name Name of the dataset
     * @param value Value of the dataset
     * @throws std::runtime_error if the dataset cannot be created
     */
    void writeStringDataset(hid_t loc, const std::string& name, const std::string& value);

    /**
     * @brief Write a 1D array-of-strings attribute on an HDF5 object
     * @param loc Handle to the object (a group, dataset, or file) to
     *   attach the attribute to
     * @param name Name of the attribute
     * @param values Values of the attribute
     * @throws std::runtime_error if the attribute cannot be created
     */
    void writeStringArrayAttr(hid_t loc, const std::string& name,
        const std::vector<std::string>& values);

    /**
     * @brief Create a 1D, extensible dataset in an HDF5 group
     * @param loc Handle to the group (or file) to create the dataset in
     * @param name Name of the dataset
     * @param type HDF5 datatype of the dataset's elements
     * @returns The new dataset's handle -- left open; the caller owns
     *   closing it (via H5Dclose())
     * @throws std::runtime_error if the dataset cannot be created
     * @details
     * Chunked, so appendToDataset() can grow it one element at a time.
     */
    auto createExtensible1dDataset(hid_t loc, const std::string& name, hid_t type) -> hid_t;

    /**
     * @brief Create a 2D, row-extensible dataset in an HDF5 group
     * @param loc Handle to the group (or file) to create the dataset in
     * @param name Name of the dataset
     * @param type HDF5 datatype of the dataset's elements
     * @param nCols Number of columns; fixed for the dataset's lifetime
     * @returns The new dataset's handle -- left open; the caller owns
     *   closing it (via H5Dclose())
     * @throws std::runtime_error if the dataset cannot be created
     * @details
     * Chunked, so appendRowToDataset2d() can grow it one row at a time.
     */
    auto createExtensible2dDataset(hid_t loc, const std::string& name, hid_t type,
        hsize_t nCols) -> hid_t;

    /**
     * @brief Create a 1D, non-extensible dataset in an HDF5 group and immediately write its data
     * @param loc Handle to the group (or file) to create the dataset in
     * @param name Name of the dataset
     * @param type HDF5 datatype of the dataset's elements
     * @param data Pointer to len contiguous elements of type type
     * @param len Number of elements
     * @param units Value written as the new dataset's own scalar
     *   "units" string attribute
     * @throws std::runtime_error if the dataset cannot be created
     */
    void writeFixed1dDataset(hid_t loc, const std::string& name, hid_t type,
        const void* data, hsize_t len, const std::string& units);

    /**
     * @brief Append a single element to the end of an extensible 1D dataset
     * @param loc Handle to the group (or file) containing the dataset
     * @param name Name of the dataset
     * @param memType HDF5 datatype value points to in memory
     * @param value Pointer to the single element to append
     * @throws std::runtime_error if the dataset cannot be opened
     */
    void appendToDataset(hid_t loc, const std::string& name, hid_t memType, const void* value);

    /**
     * @brief Append a single row to the end of an extensible 2D dataset
     * @param loc Handle to the group (or file) containing the dataset
     * @param name Name of the dataset
     * @param memType HDF5 datatype rowData points to in memory
     * @param rowData Pointer to the row to append -- the dataset's own
     *   fixed number of columns' worth of contiguous elements, taken
     *   from the dataset's own current extent
     * @throws std::runtime_error if the dataset cannot be opened
     */
    void appendRowToDataset2d(hid_t loc, const std::string& name, hid_t memType,
        const void* rowData);

    /**
     * @brief Return the name of the idx-th direct child link of an HDF5 group
     * @param loc Handle to the group (or file, which is itself a group)
     * @param idx Index of the child link to name
     * @returns The child's name
     */
    auto childNameByIdx(hid_t loc, hsize_t idx) -> std::string;

    /**
     * @brief Check whether a dataset is extensible
     * @param dset Handle to the dataset
     * @returns true if any one of dset's own dimensions has an
     *   unlimited maximum extent (i.e. it was created via
     *   createExtensible1dDataset()/createExtensible2dDataset()),
     *   false otherwise (e.g. a dataset created via
     *   writeFixed1dDataset() or writeStringDataset())
     */
    auto isExtensibleDataset(hid_t dset) -> bool;

    /**
     * @brief Append every row of one extensible dataset onto another
     * @param srcDset Handle to the dataset to read rows from
     * @param dstDset Handle to the dataset to append those rows onto
     * @details
     * srcDset and dstDset are assumed to share the same rank, element
     * type, and (for a 2D dataset) number of columns. A no-op if
     * srcDset is currently empty (no rows to append).
     */
    void appendDatasetContents(hid_t srcDset, hid_t dstDset);

    /**
     * @brief Append every extensible dataset in one HDF5 file onto the correspondingly-named dataset of the same group in another
     * @param srcFile Handle to the file to read from
     * @param dstFile Handle to the file to append onto -- must already
     *   contain a group/dataset structure matching srcFile's own,
     *   built by an identical prior call to whatever created srcFile
     * @details
     * Recurses one level into each of srcFile's own top-level groups
     * (not further); every extensible dataset directly inside one of
     * those groups (per isExtensibleDataset()) is appended via
     * appendDatasetContents(), every other dataset is left alone
     * (assumed already identical, fixed, shared metadata).
     */
    void appendFileContents(hid_t srcFile, hid_t dstFile);

} // namespace utils
// NOLINTEND(misc-include-cleaner)

#endif // HDF5UTILS_HPP
