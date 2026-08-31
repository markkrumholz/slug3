/**
 * @file IsotopeTable.cpp
 * @author Mark Krumholz
 * @brief Implementation of elem::IsotopeTable
 * @date 2026-08-31
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "IsotopeTable.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "IonizationData.hpp"
#include "IsotopeData.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace elem
{
    IsotopeTable::IsotopeTable(const std::string& fileName)
    {
        const auto filePath = utils::getFilePath(fileName);
        if (filePath.empty())
        {
            throw std::runtime_error(
                "IsotopeTable: unable to locate isotope data file " + fileName);
        }

        // NOLINTBEGIN(misc-include-cleaner) -- see HDF5Utils.hpp's own comment
        const hid_t file = H5Fopen(filePath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "IsotopeTable: unable to open HDF5 file " + filePath.string());
        }

        std::vector<double> zData;
        std::vector<double> aData;
        std::vector<double> lifetimeData;
        std::vector<double> offsetData;
        std::vector<double> daughterZData;
        std::vector<double> daughterAData;
        std::vector<double> branchData;
        try
        {
            zData = utils::readDataset1D(file, "Z", "IsotopeTable");
            aData = utils::readDataset1D(file, "A", "IsotopeTable");
            lifetimeData = utils::readDataset1D(file, "lifetime", "IsotopeTable");
            offsetData = utils::readDataset1D(file, "daughterOffset", "IsotopeTable");
            daughterZData = utils::readDataset1D(file, "daughterZ", "IsotopeTable");
            daughterAData = utils::readDataset1D(file, "daughterA", "IsotopeTable");
            branchData = utils::readDataset1D(file, "branchingRatio", "IsotopeTable");
        }
        catch (...)
        {
            H5Fclose(file);
            throw;
        }
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        if (offsetData.size() != zData.size() + 1)
        {
            throw std::runtime_error(
                "IsotopeTable: daughterOffset has " + std::to_string(offsetData.size()) +
                " entries, expected " + std::to_string(zData.size() + 1) +
                " (one per isotope, plus a final total)");
        }

        for (std::size_t i = 0; i < zData.size(); ++i)
        {
            const auto z = static_cast<unsigned int>(zData.at(i));
            const auto a = static_cast<unsigned int>(aData.at(i));
            const double lifetime = lifetimeData.at(i);

            const auto begin = static_cast<std::size_t>(offsetData.at(i));
            const auto end = static_cast<std::size_t>(offsetData.at(i + 1));
            std::vector<IsotopeDecayData> daughters;
            daughters.reserve(end - begin);
            for (std::size_t j = begin; j < end; ++j)
            {
                daughters.push_back({
                    static_cast<unsigned int>(daughterZData.at(j)),
                    static_cast<unsigned int>(daughterAData.at(j)),
                    branchData.at(j)
                });
            }

            // ionizationData (the ionization-potential table) is already the
            // canonical Z -> symbol mapping in this namespace, so reuse
            // it here rather than duplicating symbols in the isotope
            // data file
            const bool inserted = table_.emplace(std::make_pair(z, a),
                IsotopeData(ionizationData.at(z - 1).symbol(), z, a, lifetime, std::move(daughters))).second;
            if (!inserted)
            {
                throw std::runtime_error(
                    "IsotopeTable: duplicate isotope entry (Z=" + std::to_string(z) +
                    ", A=" + std::to_string(a) + ") in " + filePath.string());
            }
        }
    }

} // namespace elem
