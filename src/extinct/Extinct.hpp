/**
 * @file Extinct.hpp
 * @author Mark Krumholz
 * @brief Class to represent a dust extinction curve.
 * @date 2026-08-03
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
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

namespace io
{
    class SimControls;
} // namespace io

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
     * data/tools/extinct/add_extinction_curve.py/fetch_draine_extinction.py
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
         * @param controls Simulation controls this Extinct reads its
         *   redshift (see wlObs()) from, live, for the rest of its
         *   lifetime -- see controls_'s own comment. Must outlive this
         *   Extinct. Has no default of its own (unlike registryName
         *   below): a reference bound to a temporary default-
         *   constructed SimControls would dangle the moment this
         *   constructor returned, since this class stores it live
         *   rather than copying out of it.
         * @param registryName Name of the extinction curve registry file
         * @throws std::runtime_error if extinctName is not found in the
         *   registry, or the registry/HDF5 file cannot be read
         * @details
         * wl is clipped to the native curve's own [min, max] wavelength
         * coverage before interpolating -- see wl()'s own comment.
         */
        Extinct(const std::string& extinctName,
            const std::vector<double>& wl,
            const io::SimControls& controls,
            const std::string& registryName = defaultRegistry) :
            controls_(controls)
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

            // Interpolate the curve onto every nebular emission line's
            // own wavelength too, if a nebular emission grid was
            // requested -- see initExtinctLines()'s own comment. Needs
            // io::SimControls's complete type (to call
            // controls_.nebular()), so is defined out-of-line, in
            // Extinct.cpp, exactly like computeExtinctionFacCts()
            // below -- see that method's own comment for why.
            initExtinctLines(interp);

            // Normalize the curve (and, if any, the line-wavelength
            // curve above) to a V-band extinction of 1 mag
            normalize(wl_, extinct_, extinctLines_);

            // Precompute extinctionFacCts_/extinctionFacCtsLines_ --
            // see their own comments
            computeExtinctionFacCts();
            computeExtinctionFacCtsLines();
        }

        // Copyable (rebinding controls_ to the same referent) but not
        // assignable (controls_ can't be reseated), matching Specsyn's
        // own identical copy/move declarations exactly -- see its
        // comment. Never actually copied/assigned in practice: every
        // Extinct is held via unique_ptr (see SimControls's own
        // extinct_).
        Extinct(const Extinct&) = default;
        Extinct(Extinct&&) = default;
        auto operator=(const Extinct&) -> Extinct& = delete;
        auto operator=(Extinct&&) -> Extinct& = delete;
        ~Extinct() = default;

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
         * @brief Get the observed-frame interpolated wavelength grid
         * @return wl(), redshifted by (1 + z), with z read live from controls_
         * @details
         * Defined out-of-line, in Extinct.cpp -- see that file's own
         * comment for why (controls_.z() needs io::SimControls's
         * complete type, which this header can only forward-declare;
         * mirrors Specsyn::wlObs()'s identical situation).
         */
        [[nodiscard]] auto wlObs() const -> std::vector<double>;

        /**
         * @brief Get the number of leading elements chopped off the constructor's own wl
         * @return wlOffset_ -- the number of leading elements of the wl
         *   passed to the constructor that fell below the native
         *   curve's own coverage and so are absent from wl()/extinct()
         *   (see the constructor's own comment). Lets a caller line up
         *   a spectrum tabulated on that original wl with wl()'s own,
         *   narrower grid, exactly as applyExtinction() does internally.
         */
        [[nodiscard]] auto wlOffset() const { return wlOffset_; }

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
                result[i] = spec[wlOffset_ + i] * std::exp(-A_V * extinct_[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinct_ both have size wl_.size(), and i is bounded by wl_.size(); spec has the same size as the wl originally passed to the constructor (this class's own contract), so wlOffset_ + i stays in bounds
            }
            return result;
        }

        /**
         * @brief Apply this extinction curve's own expected attenuation to a continuously-distributed population's spectrum
         * @param spec Spectrum to extinguish, tabulated on exactly the
         *   same wavelength grid as the wl originally passed to the
         *   constructor -- see applyExtinction()'s own spec parameter
         * @returns The expected extinguished spectrum, on the
         *   wavelength grid returned by wl()
         * @details
         * Unlike applyExtinction(), which attenuates a single star (or
         * cluster) with one known A_V, this is for a population whose
         * members are not individually tracked, so there is no single
         * A_V to apply -- instead, each of spec's remaining wl().size()
         * elements is multiplied by extinctionFacCts_ at the
         * corresponding wavelength: the expectation value of
         * exp(-A_V * extinct()) over the field-star A_V distribution
         * (io::SimControls::avDistField()), precomputed once, at
         * construction -- see extinctionFacCts_'s own comment.
         */
        [[nodiscard]] auto applyExtinctionCts(const std::vector<double>& spec) const -> std::vector<double>
        {
            std::vector<double> result(wl_.size());
            for (std::size_t i = 0; i < wl_.size(); i++)
            {
                result[i] = spec[wlOffset_ + i] * extinctionFacCts_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinctionFacCts_ both have size wl_.size() (extinctionFacCts_ by computeExtinctionFacCts()'s own contract), and i is bounded by wl_.size(); spec has the same size as the wl originally passed to the constructor, so wlOffset_ + i stays in bounds -- see applyExtinction()'s own identical indexing
            }
            return result;
        }

        /**
         * @brief Apply this extinction curve to a set of nebular emission line luminosities
         * @param A_V V-band extinction to apply, in magnitudes
         * @param lineLum Luminosity of each of
         *   controls_.nebular()->lineWl()'s own lines, in erg/s
         * @returns The extinguished line luminosities, in the same
         *   order as lineLum/controls_.nebular()->lineWl()
         * @details
         * Line-luminosity analog of applyExtinction(): each element of
         * lineLum is multiplied by exp(-A_V * extinctLines_) at the
         * corresponding line. Unlike applyExtinction(), no elements are
         * dropped -- extinctLines_ already reads 0 (rather than being
         * absent) for any line outside the native curve's own coverage
         * -- so lineLum and the result share the same size and
         * indexing as extinctLines_/controls_.nebular()->lineWl(), with
         * no analog of wlOffset_.
         */
        [[nodiscard]] auto applyExtinctionLines(const double A_V, // NOLINT(readability-identifier-naming) -- see applyExtinction()'s own identical NOLINT
            const std::vector<double>& lineLum) const -> std::vector<double>
        {
            std::vector<double> result(extinctLines_.size());
            for (std::size_t i = 0; i < extinctLines_.size(); i++)
            {
                result[i] = lineLum[i] * std::exp(-A_V * extinctLines_[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinctLines_ both have size extinctLines_.size(), and i is bounded by extinctLines_.size(); lineLum has the same size, by this method's own contract
            }
            return result;
        }

        /**
         * @brief Apply this extinction curve's own expected attenuation to a continuously-distributed population's line luminosities
         * @param lineLum Luminosity of each of
         *   controls_.nebular()->lineWl()'s own lines, in erg/s -- see
         *   applyExtinctionLines()'s own lineLum parameter
         * @returns The expected extinguished line luminosities, in the
         *   same order as lineLum/controls_.nebular()->lineWl()
         * @details
         * Line-luminosity analog of applyExtinctionCts(): each element
         * of lineLum is multiplied by extinctionFacCtsLines_ at the
         * corresponding line -- the expectation value of
         * exp(-A_V * extinctLines_) over the field-star A_V
         * distribution (io::SimControls::avDistField()), precomputed
         * once, at construction -- see extinctionFacCtsLines_'s own
         * comment. See applyExtinctionLines()'s own comment for why
         * there is no analog of wlOffset_ here.
         */
        [[nodiscard]] auto applyExtinctionCtsLines(const std::vector<double>& lineLum) const -> std::vector<double>
        {
            std::vector<double> result(extinctLines_.size());
            for (std::size_t i = 0; i < extinctLines_.size(); i++)
            {
                result[i] = lineLum[i] * extinctionFacCtsLines_[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinctionFacCtsLines_ both have size extinctLines_.size() (extinctionFacCtsLines_ by computeExtinctionFacCtsLines()'s own contract), and i is bounded by extinctLines_.size(); lineLum has the same size, by this method's own contract
            }
            return result;
        }

    private:

        /**
         * @brief Normalize an extinction curve to a V-band extinction of 1 mag
         * @param wl Wavelength grid, in Angstrom, that extinct is defined on
         * @param extinct Extinction curve; rescaled in place so that it
         *   corresponds to a V-band extinction A_V = 1 mag
         * @param extinctLines Line-wavelength extinction curve (see
         *   extinctLines_'s own comment); rescaled in place by the same
         *   normalization factor as extinct, so that A_V continues to
         *   mean the same thing for both. Empty (a no-op) if no nebular
         *   emission grid was requested.
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
            std::vector<double>& extinctLines,
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
            for (double& e : extinctLines) { e *= norm; }
        }

        /**
         * @brief The per-wavelength attenuation factor for a single, known A_V
         * @param A_V V-band extinction, in magnitudes
         * @return exp(-A_V * extinct()) at each wavelength in wl()
         * @details
         * The same factor applyExtinction() itself multiplies a
         * spectrum by, exposed as its own method so
         * computeExtinctionFacCts() can hand it to utils::PDFIntegrator
         * as the integrand of an integral over A_V, rather than
         * duplicating this same exponential there.
         */
        [[nodiscard]] auto extinctFac(const double A_V) const -> std::vector<double> // NOLINT(readability-identifier-naming) -- see applyExtinction()'s own identical NOLINT
        {
            std::vector<double> result(wl_.size());
            for (std::size_t i = 0; i < wl_.size(); i++)
            {
                result[i] = std::exp(-A_V * extinct_[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinct_ both have size wl_.size(), and i is bounded by wl_.size()
            }
            return result;
        }

        /**
         * @brief The per-line attenuation factor for a single, known A_V
         * @param A_V V-band extinction, in magnitudes
         * @return exp(-A_V * extinctLines_) at each of
         *   controls_.nebular()->lineWl()'s own lines
         * @details
         * Line-luminosity analog of extinctFac() -- the same factor
         * applyExtinctionLines() itself multiplies a set of line
         * luminosities by, exposed as its own method so
         * computeExtinctionFacCtsLines() can hand it to
         * utils::PDFIntegrator as the integrand of an integral over
         * A_V, rather than duplicating this same exponential there.
         */
        [[nodiscard]] auto extinctFacLines(const double A_V) const -> std::vector<double> // NOLINT(readability-identifier-naming) -- see applyExtinction()'s own identical NOLINT
        {
            std::vector<double> result(extinctLines_.size());
            for (std::size_t i = 0; i < extinctLines_.size(); i++)
            {
                result[i] = std::exp(-A_V * extinctLines_[i]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- result and extinctLines_ both have size extinctLines_.size(), and i is bounded by extinctLines_.size()
            }
            return result;
        }

        /**
         * @brief Initialize extinctLines_ from the native curve's own interpolator
         * @param interp Interpolator built (in the constructor) from
         *   wlDat_/extinctDat_ -- the same one wl_/extinct_ are
         *   themselves interpolated from
         * @details
         * A no-op (extinctLines_ left empty) if no nebular emission
         * grid was requested (controls_.nebular() == nullptr).
         * Otherwise resizes extinctLines_ to
         * controls_.nebular()->lineWl().size() and, for each line,
         * evaluates interp at that line's own wavelength if it falls
         * within interp's own [xMin(), xMax()] coverage, or sets that
         * line's entry to 0 (no attenuation) otherwise -- see
         * extinctLines_'s own comment for why lines are never dropped
         * the way wl_/extinct_'s own out-of-coverage leading/trailing
         * elements are. Needs io::SimControls's complete type (to call
         * controls_.nebular()), so is defined out-of-line, in
         * Extinct.cpp -- see computeExtinctionFacCts()'s own comment
         * for why. Called once, from the constructor, immediately
         * after the wl_/extinct_ loop, before normalize().
         */
        void initExtinctLines(const interp::Interpolator1D<1>& interp);

        /**
         * @brief Compute extinctionFacCts_
         * @details
         * Defined out-of-line, in Extinct.cpp -- see that file's own
         * comment for why (needs io::SimControls's complete type, to
         * call controls_.avDistField()/intMaxIter()/intAbsTol()/
         * intRelTol(); mirrors wlObs()'s identical situation). Called
         * once, from the constructor, immediately after normalize().
         */
        void computeExtinctionFacCts();

        /**
         * @brief Compute extinctionFacCtsLines_
         * @details
         * Line-luminosity analog of computeExtinctionFacCts() -- see
         * its own comment; identical in every respect except that it
         * integrates extinctFacLines() (over extinctLines_.size()
         * quantities) rather than extinctFac(), storing the result
         * into extinctionFacCtsLines_ rather than extinctionFacCts_. A
         * no-op (an empty extinctionFacCtsLines_) if extinctLines_ is
         * itself empty, i.e. no nebular emission grid was requested.
         * Defined out-of-line, in Extinct.cpp, for the same reason as
         * computeExtinctionFacCts(). Called once, from the
         * constructor, immediately after computeExtinctionFacCts().
         */
        void computeExtinctionFacCtsLines();

        /**
         * @brief Simulation controls this Extinct reads its redshift from
         * @details
         * Read live, not snapshotted, every time wlObs() is called --
         * changing controls_'s own redshift after this Extinct is
         * built takes effect immediately, with no need to rebuild it.
         * Bound once, at construction, from whichever SimControls
         * actually built this Extinct (see SimControls::readExtinct());
         * never reseated afterward, so that SimControls must outlive
         * this Extinct. Mirrors Specsyn's own controls_ member exactly
         * -- see its comment for the rationale.
         */
        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy: see this member's own comment for why. Only ever used through the same non-copyable, non-movable ownership pattern (unique_ptr in SimControls's own extinct_) as every other class with a reference member in this codebase (e.g. Specsyn's own controls_), so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.

        std::vector<double> wlDat_;      /**< Native extinction curve wavelength grid, in Angstrom */
        std::vector<double> extinctDat_; /**< Native extinction curve, in arbitrary units */
        std::vector<double> wl_;         /**< Interpolated wavelength grid, in Angstrom */
        std::vector<double> extinct_;    /**< Extinction curve interpolated onto wl_ */
        std::size_t wlOffset_ = 0;       /**< Number of leading elements of the constructor's own wl chopped off wl_'s front */

        /**
         * @brief Extinction curve interpolated onto controls_.nebular()->lineWl()
         * @details
         * One entry per line in controls_.nebular()->lineWl(), in the
         * same order -- unlike extinct_ (interpolated onto wl_, a
         * chopped-down copy of the constructor's own wl), no line is
         * ever dropped here: a line falling outside the native curve's
         * own [wlDat_.front(), wlDat_.back()] coverage simply reads an
         * extinction of 0 (no attenuation) rather than being excluded,
         * since a nebular emission line, unlike a point on a
         * continuous wavelength grid, cannot simply be left out of the
         * output. Left empty if no nebular emission grid was requested
         * (controls_.nebular() == nullptr).
         */
        std::vector<double> extinctLines_;

        /**
         * @brief The expected value of exp(-A_V * extinct()) over controls_.avDistField(), at each wavelength in wl()
         * @details
         * \f$\int \exp[-A_V \cdot \mathrm{extinct}(\lambda)]\, p(A_V)\, dA_V\f$,
         * where \f$p\f$ is controls_.avDistField() -- the multiplicative
         * factor applyExtinctionCts() itself applies to a continuously-
         * distributed population's spectrum, since no single A_V
         * applies to every member of that population. Computed once,
         * by computeExtinctionFacCts(), at construction (avDistField()
         * never changes after that, so neither does this).
         */
        std::vector<double> extinctionFacCts_;

        /**
         * @brief The expected value of exp(-A_V * extinctLines_) over controls_.avDistField(), at each line in controls_.nebular()->lineWl()
         * @details
         * Line-luminosity analog of extinctionFacCts_ -- the
         * multiplicative factor applyExtinctionCtsLines() itself
         * applies to a continuously-distributed population's line
         * luminosities. Computed once, by
         * computeExtinctionFacCtsLines(), at construction. Left empty
         * if extinctLines_ is itself empty, i.e. no nebular emission
         * grid was requested.
         */
        std::vector<double> extinctionFacCtsLines_;
    };

} // namespace extinct

#endif // EXTINCT_HPP
