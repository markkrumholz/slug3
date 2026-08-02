/**
 * @file VegaSpectrum.cpp
 * @author Mark Krumholz
 * @brief Implementation of VegaSpectrum.hpp
 * @date 2026-08-02
 */

#include "VegaSpectrum.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <stdexcept>
#include <string>

// Disable linting for the constructor body -- including hdf5.h
// wholesale (rather than individual headers) is the paradigm HDF5
// itself wants, which confuses misc-include-cleaner -- see
// FilterCollection.cpp's own identical suppression
// NOLINTBEGIN(misc-include-cleaner)
phot::VegaSpectrum::VegaSpectrum(const std::string& vegaName)
{
    const auto vegaPath = utils::getFilePath(vegaName);
    if (vegaPath.empty())
    {
        throw std::runtime_error(
            "VegaSpectrum: Vega reference spectrum " + vegaName + " not found");
    }
    const hid_t file = H5Fopen(vegaPath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        throw std::runtime_error(
            "VegaSpectrum: unable to open HDF5 file " + vegaPath.string());
    }
    wl_ = utils::readDataset1D(file, "wl", "VegaSpectrum");
    flux_ = utils::readDataset1D(file, "flux", "VegaSpectrum");
    H5Fclose(file);
}
// NOLINTEND(misc-include-cleaner)
