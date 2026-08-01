/**
 * @file FilterTabulated.cpp
 * @author Mark Krumholz
 * @brief Implementation of FilterTabulated.hpp
 * @date 2026-07-27
 * @details
 * phot() evaluates \f$\int F_\lambda(\lambda) R(\lambda)\, d\ln\lambda
 * \big/ \int R(\lambda)\, d\ln\lambda\f$ rather than integrating
 * directly in \f$\lambda\f$. This is not an approximation: the
 * photon-counting quantity slug actually wants is \f$\int [F_\lambda
 * R / \lambda]\, d\lambda \big/ \int [R / \lambda]\, d\lambda\f$ (the
 * \f$1/\lambda\f$ factors converting an energy flux into a photon
 * count), and substituting \f$u = \ln\lambda\f$ (so \f$d\lambda =
 * \lambda\, du\f$) turns \f$[R/\lambda]\, d\lambda\f$ into \f$R\, du\f$
 * exactly, in both the numerator and denominator. Working in
 * \f$\ln\lambda\f$ throughout therefore computes the same ratio with
 * one fewer factor of \f$1/\lambda\f$ evaluated per grid point in both
 * response_ (built from lnWl_ rather than wl_) and phot()'s specInterp.
 */

#include "FilterTabulated.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

namespace phot
{
    auto FilterTabulated::phot(const std::vector<double>& wl,
        const std::vector<double>& spec) const -> double
    {
        const interp::Interpolator1D<1> specInterp(lnGrid(wl), spec);
        return response_.integ(specInterp, specInterp.xMin(), specInterp.xMax()) / norm_;
    }

    auto FilterTabulated::loadFromRegistry(
        const std::string& facility,
        const std::string& instrument,
        const std::string& filter,
        const std::string& registryName)
    -> RegistryData
    {
        // Locate and parse the registry file
        const auto [registry, registryPath] =
            utils::parseTOMLFile(registryName, "FilterTabulated");

        // Validate that the registry actually lists this
        // facility/instrument/filter combination
        auto instruments = utils::stringArrayContents(registry[facility]["instruments"].as_array()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- toml::table/node_view::operator[] is a keyed lookup, not a bounds-checkable container index; a missing key just yields a null node_view, handled here and by stringArrayContents/as_array returning nullptr
        if (std::ranges::find(instruments, instrument) == instruments.end())
        {
            throw std::runtime_error(
                "FilterTabulated: facility '" + facility + "' has no instrument '" +
                instrument + "' in registry " + registryPath.string());
        }
        auto filters = utils::stringArrayContents(registry[facility][instrument]["filters"].as_array()); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see instruments above
        if (std::ranges::find(filters, filter) == filters.end())
        {
            throw std::runtime_error(
                "FilterTabulated: instrument '" + instrument + "' (facility '" + facility +
                "') has no filter '" + filter + "' in registry " + registryPath.string());
        }

        // Each filter's own entry (e.g. [SLUGTEST.CAM1.G500]) carries a
        // wl_ref field giving this filter's pivot wavelength (see
        // Filter::wlPivot())
        const auto wlRef = registry[facility][instrument][filter]["wl_ref"].value<double>(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see instruments above
        if (!wlRef.has_value())
        {
            throw std::runtime_error(
                "FilterTabulated: registry entry '" + facility + "." + instrument +
                "." + filter + "' is missing required 'wl_ref' field in registry " +
                registryPath.string());
        }

        // The registry's top-level "file" entry names the HDF5 file
        // holding the actual tabulated data, relative to the directory
        // containing the registry itself -- the same convention used
        // by the track and spectra registries (see TrackUtils.cpp's
        // parseRegistry/findMatchingTracks), and written by
        // fetch_filter_vo.py's update_registry().
        const auto h5Name = registry["file"].value<std::string>(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see instruments above
        if (!h5Name.has_value())
        {
            throw std::runtime_error(
                "FilterTabulated: registry " + registryPath.string() +
                " is missing required 'file' field");
        }
        const auto h5Path = registryPath.parent_path() / h5Name.value();

        // NOLINTBEGIN(misc-include-cleaner) -- including hdf5.h wholesale
        // (rather than individual headers) is the paradigm HDF5 itself
        // wants, which confuses misc-include-cleaner
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "FilterTabulated: unable to open HDF5 file " + h5Path.string());
        }
        const std::string groupPath = facility + "/" + instrument + "/" + filter;
        const hid_t grp = H5Gopen2(file, groupPath.c_str(), H5P_DEFAULT);
        if (grp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                "FilterTabulated: unable to open group " + groupPath +
                " in HDF5 file " + h5Path.string());
        }

        RegistryData data;
        data.wl_ = utils::readDataset1D(grp, "wavelength", "FilterTabulated");
        data.response_ = utils::readDataset1D(grp, "transmission", "FilterTabulated");
        data.wlPivot_ = wlRef.value();

        H5Gclose(grp);
        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        return data;
    }

    FilterTabulated::FilterTabulated(
        const std::string& facility,
        const std::string& instrument,
        const std::string& filter,
        const std::string& registryName)
    : FilterTabulated(facility + "." + instrument + "." + filter,
        loadFromRegistry(facility, instrument, filter, registryName))
    { }

} // namespace phot
