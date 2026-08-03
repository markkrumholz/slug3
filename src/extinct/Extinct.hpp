/**
 * @file Extinct.hpp
 * @author Mark Krumholz
 * @brief Class to represent a dust extinction curve.
 * @date 2026-08-03
 */

#ifndef EXTINCT_HPP
#define EXTINCT_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

namespace extinct
{
    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("extinct")
        / std::filesystem::path("extinct.toml")); /**< Default registry */

    /**
     * @class Extinct
     * @brief A dust extinction curve, interpolated onto a caller-supplied wavelength grid.
     * @details
     * An Extinct is built from a named curve in an extinction curve
     * registry (see data/extinct/extinct.toml/extinct.h5 and
     * data/tools/add_extinction_curve.py/fetch_draine_extinction.py
     * for how these are populated): the curve's own native
     * (wavelength, kappa) tabulation is read from the registry, then
     * interpolated onto the wavelength grid the caller supplies,
     * clipped to the native curve's own coverage. kappa is in
     * arbitrary units -- only the curve's shape matters, since a
     * caller scales it by a separately supplied A_V.
     */
    class Extinct
    {
    public:

        /**
         * @brief Construct an Extinct from a named registry entry
         * @param extinctName Name of the extinction curve to load (e.g. "Calzetti_starburst")
         * @param wl Wavelength grid, in Angstrom, to interpolate the curve onto
         * @param registryName Name of the extinction curve registry file
         * @throws std::runtime_error if extinctName is not found in the
         *   registry, or the registry/HDF5 file cannot be read
         * @details
         * wl is clipped to the native curve's own [min, max] wavelength
         * coverage before interpolating -- see wl()'s own comment.
         */
        Extinct(const std::string& extinctName,
            const std::vector<double>& wl,
            const std::string& registryName = defaultRegistry)
        {
            // Locate and parse the registry file
            const auto [registry, registryPath] =
                utils::parseTOMLFile(registryName, "Extinct");

            // Validate that the registry actually lists this curve
            const auto curves = utils::getStringArrayField(registry, "curves");
            if (std::ranges::find(curves, extinctName) == curves.end())
            {
                throw std::runtime_error(
                    "Extinct: registry " + registryPath.string() +
                    " has no extinction curve '" + extinctName + "'");
            }

            // The registry's top-level "file" entry names the HDF5 file
            // holding the actual curve data, relative to the directory
            // containing the registry itself
            const auto h5Name = registry["file"].value<std::string>(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- toml::table::operator[] is a keyed lookup, not a bounds-checkable container index; a missing key just yields a null node_view, handled by value<std::string>() returning nullopt
            if (!h5Name.has_value())
            {
                throw std::runtime_error(
                    "Extinct: registry " + registryPath.string() +
                    " is missing required 'file' field");
            }
            const auto h5Path = registryPath.parent_path() / h5Name.value();

            // NOLINTBEGIN(misc-include-cleaner) -- see HDF5Utils.hpp's own comment
            const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file < 0)
            {
                throw std::runtime_error(
                    "Extinct: unable to open HDF5 file " + h5Path.string());
            }
            const hid_t grp = H5Gopen2(file, extinctName.c_str(), H5P_DEFAULT);
            if (grp < 0)
            {
                H5Fclose(file);
                throw std::runtime_error(
                    "Extinct: unable to open group " + extinctName +
                    " in HDF5 file " + h5Path.string());
            }

            wlDat_ = utils::readDataset1D(grp, "wavelength", "Extinct");
            extinctDat_ = utils::readDataset1D(grp, "kappa", "Extinct");

            H5Gclose(grp);
            H5Fclose(file);
            // NOLINTEND(misc-include-cleaner)

            // Build an interpolator for the native curve data
            const interp::Interpolator1D<1> interp(wlDat_, extinctDat_);

            // Chop wl down to the interpolator's own coverage, then
            // interpolate the curve onto what remains
            std::ranges::copy_if(wl, std::back_inserter(wl_),
                [&interp](const double w) -> bool
                { return w >= interp.xMin() && w <= interp.xMax(); });
            extinct_.reserve(wl_.size());
            for (const double w : wl_) { extinct_.push_back(interp(w)); }
        }

        // Observers

        /**
         * @brief Get the native extinction curve wavelength grid
         * @return A const reference to the wavelength grid, in Angstrom,
         *   as read directly from the registry entry
         */
        [[nodiscard]] auto wlDat() const -> const std::vector<double>& { return wlDat_; }

        /**
         * @brief Get the native extinction curve
         * @return A const reference to the extinction curve, in
         *   arbitrary units, at each wavelength in wlDat()
         */
        [[nodiscard]] auto extinctDat() const -> const std::vector<double>& { return extinctDat_; }

        /**
         * @brief Get the interpolated wavelength grid
         * @return A const reference to the wavelength grid, in
         *   Angstrom, supplied to the constructor and clipped to
         *   wlDat()'s own [min, max] coverage
         */
        [[nodiscard]] auto wl() const -> const std::vector<double>& { return wl_; }

        /**
         * @brief Get the interpolated extinction curve
         * @return A const reference to the extinction curve, in
         *   arbitrary units, interpolated onto wl()
         */
        [[nodiscard]] auto extinct() const -> const std::vector<double>& { return extinct_; }

    private:

        std::vector<double> wlDat_;      /**< Native extinction curve wavelength grid, in Angstrom */
        std::vector<double> extinctDat_; /**< Native extinction curve, in arbitrary units */
        std::vector<double> wl_;         /**< Interpolated wavelength grid, in Angstrom */
        std::vector<double> extinct_;    /**< Extinction curve interpolated onto wl_ */
    };

} // namespace extinct

#endif // EXTINCT_HPP
