/**
 * @file Extinct.hpp
 * @author Mark Krumholz
 * @brief Class to represent a dust extinction curve.
 * @date 2026-08-03
 */

#ifndef EXTINCT_HPP
#define EXTINCT_HPP

#include "../interpolation/Interpolator1D.hpp"
#include "../phot/FilterTabulated.hpp"
#include "../utils/Constants.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner) -- see HDF5Utils.hpp's own comment on including hdf5.h wholesale
#include <algorithm>
#include <cmath>
#include <cstddef>
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

    inline static const std::string defaultVRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- see defaultRegistry above
        (std::filesystem::path("data") / std::filesystem::path("filters")
        / std::filesystem::path("V_filter.toml")); /**< Default V-band filter registry */

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

            // Chop wl down to the interpolator's own coverage -- wl is
            // assumed sorted ascending (a spectral wavelength grid),
            // so the kept elements are a single contiguous run;
            // wlOffset_ records how many leading elements were
            // dropped, so applyExtinction() can later line up a
            // spectrum tabulated on this same original wl without
            // having to rediscover the chop -- then interpolate the
            // curve onto what remains
            const auto firstIt = std::ranges::find_if(wl,
                [&interp](const double w) -> bool { return w >= interp.xMin(); });
            wlOffset_ = static_cast<std::size_t>(std::distance(wl.begin(), firstIt));
            for (auto it = firstIt; it != wl.end() && *it <= interp.xMax(); ++it)
            {
                wl_.push_back(*it);
                extinct_.push_back(interp(*it));
            }

            // Normalize the curve to a V-band extinction of 1 mag
            normalize(wl_, extinct_);
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

        /**
         * @brief Apply this extinction curve to a spectrum
         * @param A_V V-band extinction to apply, in magnitudes
         * @param spec Spectrum to extinguish, tabulated on exactly the
         *   same wavelength grid as the wl originally passed to the
         *   constructor
         * @returns The extinguished spectrum, on the wavelength grid
         *   returned by wl()
         * @details
         * spec's first wlOffset_ elements (those falling outside this
         * curve's own wavelength coverage) are discarded; each of the
         * remaining wl().size() elements is multiplied by
         * exp(-A_V * extinct()) at the corresponding wavelength.
         */
        [[nodiscard]] auto applyExtinction(const double A_V, // NOLINT(readability-identifier-naming) -- see above
            const std::vector<double>& spec) const -> std::vector<double>
        {
            std::vector<double> result(wl_.size());
            for (std::size_t i = 0; i < wl_.size(); i++)
            {
                result.at(i) = spec.at(wlOffset_ + i) * std::exp(-A_V * extinct_.at(i));
            }
            return result;
        }

    private:

        /**
         * @brief Normalize an extinction curve to a V-band extinction of 1 mag
         * @param wl Wavelength grid, in Angstrom, that extinct is defined on
         * @param extinct Extinction curve; rescaled in place so that it
         *   corresponds to a V-band extinction A_V = 1 mag
         * @param vRegistry Name of the V-band filter registry file
         * @details
         * The mean V-band opacity is defined as
         * \f$\kappa_V = \int \kappa(\nu) R(\nu)\, d\nu \big/ \int R(\nu)\, d\nu\f$,
         * where \f$\kappa(\nu)\f$ is extinct and \f$R(\nu)\f$ is the V
         * filter's response, both expressed as functions of frequency
         * \f$\nu\f$ (by convention, rather than wavelength); the
         * V-band extinction in magnitudes is then
         * \f$A_V = (5 / \ln 100) \kappa_V\f$. FilterTabulated's own
         * phot() can't be used for this integral, since it integrates
         * over \f$\ln\lambda\f$ rather than \f$\nu\f$, so the V
         * filter's raw wl()/responseData() are instead converted to
         * frequency and rebuilt into a fresh Interpolator1D here.
         */
        static void normalize(const std::vector<double>& wl,
            std::vector<double>& extinct,
            const std::string& vRegistry = defaultVRegistry)
        {
            const phot::FilterTabulated vFilt("Generic", "Johnson", "V", vRegistry);

            // V filter response, converted from wavelength to
            // frequency and reversed back into increasing order (wl
            // increases, so nu = c/wl decreases), then rebuilt as a
            // frequency-space interpolator
            std::vector<double> filtNu(vFilt.wl().size());
            std::ranges::transform(vFilt.wl(), filtNu.begin(),
                [](const double w) -> double { return utils::c / (w * utils::Angstrom); });
            std::ranges::reverse(filtNu);
            const std::vector<double> filtResp(vFilt.responseData().rbegin(), vFilt.responseData().rend());
            const interp::Interpolator1D<1> respInterp(filtNu, filtResp);

            // Same wavelength-to-frequency transformation for the
            // extinction curve
            std::vector<double> extNu(wl.size());
            std::ranges::transform(wl, extNu.begin(),
                [](const double w) -> double { return utils::c / (w * utils::Angstrom); });
            std::ranges::reverse(extNu);
            const std::vector<double> extRev(extinct.rbegin(), extinct.rend());
            const interp::Interpolator1D<1> extInterp(extNu, extRev);

            const double norm = (std::log(100.0) / 5.0) *
                respInterp.integ(respInterp.xMin(), respInterp.xMax()) /
                respInterp.integ(extInterp, respInterp.xMin(), respInterp.xMax());

            for (double& e : extinct) { e *= norm; }
        }

        std::vector<double> wlDat_;      /**< Native extinction curve wavelength grid, in Angstrom */
        std::vector<double> extinctDat_; /**< Native extinction curve, in arbitrary units */
        std::vector<double> wl_;         /**< Interpolated wavelength grid, in Angstrom */
        std::vector<double> extinct_;    /**< Extinction curve interpolated onto wl_ */
        std::size_t wlOffset_ = 0;       /**< Number of leading elements of the constructor's own wl chopped off wl_'s front */
    };

} // namespace extinct

#endif // EXTINCT_HPP
