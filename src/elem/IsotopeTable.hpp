/**
 * @file IsotopeTable.hpp
 * @author Mark Krumholz
 * @brief Table of all known isotopes, indexed by (Z, A)
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef ISOTOPETABLE_HPP
#define ISOTOPETABLE_HPP

#include "IsotopeData.hpp"
#include <cstddef>
#include <map>
#include <string>
#include <utility>

namespace elem
{
    /**
     * @class IsotopeTable
     * @brief Reads the isotope data HDF5 file and holds every isotope's
     *   IsotopeData record, indexed by (Z, A)
     * @details The underlying HDF5 file (see data/tools/elem/
     * build_isotope_table.py, which builds it from slug2's own
     * lib/yields/isotope_data.txt) stores the ragged per-isotope
     * daughter lists in a flat CSR layout -- parallel Z/A/lifetime
     * datasets plus a daughterOffset index into flat daughterZ/
     * daughterA/branchingRatio datasets -- read here in one pass per
     * dataset rather than opening one HDF5 group per isotope.
     */
    class IsotopeTable
    {
    public:

        /**
         * @brief Construct an IsotopeTable by reading an isotope data HDF5 file
         * @param fileName Path to the isotope data HDF5 file, resolved via
         *   utils::getFilePath (current directory, then SLUG_DIR, then
         *   REPO_DIR)
         * @throws std::runtime_error if fileName cannot be found, or the
         *   file's contents are malformed
         */
        explicit IsotopeTable(const std::string& fileName = "data/elem/isotopes.h5");

        /** @brief Return the full table, indexed by (Z, A) */
        [[nodiscard]] auto table() const noexcept
            -> const std::map<std::pair<unsigned int, unsigned int>, IsotopeData>&
            { return table_; }

        /** @brief Return the number of isotopes in the table */
        [[nodiscard]] auto size() const noexcept -> std::size_t { return table_.size(); }

    private:
        std::map<std::pair<unsigned int, unsigned int>, IsotopeData> table_;
    };

} // namespace elem

#endif // ISOTOPETABLE_HPP
