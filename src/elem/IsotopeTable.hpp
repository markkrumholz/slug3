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
#include <cstdlib>
#include <exception>
#include <iostream>
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

    /**
     * @brief Return a reference to the global isotope table
     * @details
     * The instance is a function-local static, so it is constructed on
     * first use (reading data/elem/isotopes.h5) rather than before
     * main() begins; this ensures that any exception thrown during
     * construction (e.g. the data file can't be found) is caught here
     * rather than escaping before main() can run. There is no
     * reasonable way for isotope/yield code to continue without a
     * working isotope table, so a construction failure is treated as
     * fatal: it is reported and the program exits immediately, rather
     * than letting the exception propagate to callers that have no
     * better way to handle it either. Being a local static in an inline
     * function also guarantees a single, program-wide instance, rather
     * than one per translation unit. Mirrors utils::rng()'s identical
     * pattern in RngThread.hpp.
     */
    inline auto isotopeTable() -> IsotopeTable&
    {
        try
        {
            static IsotopeTable instance;
            return instance;
        }
        catch (const std::exception& error)
        {
            std::cerr << "FATAL: isotope table initialization failed: "
                << error.what() << "\n";
            std::exit(1); // NOLINT(concurrency-mt-unsafe) -- the program is terminating regardless; thread-safety of the shutdown path doesn't matter
        }
    }

    /**
     * @brief Look up a single isotope's data by (Z, A) in the global isotope table
     * @param z Atomic number
     * @param a Mass number
     * @returns A const reference to the isotope's IsotopeData record
     * @throws std::out_of_range if no isotope with this (Z, A) exists in the table
     * @details
     * Shorthand for isotopeTable().table().at({z, a}), so callers
     * elsewhere in the codebase can just write isotopeTable(z, a)
     * rather than spelling out the map lookup themselves each time.
     */
    inline auto isotopeTable(unsigned int z, unsigned int a) -> const IsotopeData&
    {
        return isotopeTable().table().at({z, a});
    }

} // namespace elem

#endif // ISOTOPETABLE_HPP
