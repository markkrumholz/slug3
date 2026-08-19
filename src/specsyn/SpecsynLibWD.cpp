/**
 * @file SpecsynLibWD.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibWD.hpp
 * @date 2026-08-07
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SpecsynLibWD.hpp"
#include "../io/SimControls.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include "SpecsynLib2D.hpp"
#include "SpecsynUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm> // NOLINT(misc-include-cleaner) -- see the identical NOLINT on SpecsynLib.cpp's findBracket for why std::ranges::lower_bound needs this
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{
    // Holds everything the constructor needs to finish building a
    // SpecsynLibWD once the (logg, logTeff) grid and its spectra have
    // been assembled, by either readFilledTensorGrid or
    // readSparseGrid below -- mirrors SpecsynLibNoWind's own identical
    // GridBuildResult/buildExactMatchGrid split exactly, including the
    // rationale for factoring this out as a plain (Policy-independent)
    // free function rather than a member function: the grid-reading
    // logic itself has nothing to do with OOBPolicy, so there is no
    // reason to instantiate it three times over.
    struct WDGridBuildResult
    {
        std::vector<double> logg_;
        std::vector<double> logTeff_;
        std::vector<std::vector<double>> spectra_;
    };

    // Reads a flux dataset stored as a filled (n_logg, n_logTeff,
    // n_wl) tensor -- e.g. the Tremblay et al. white dwarf grids
    // fetched by fetch_tremblay.py, whose (logg, Teff) coverage is
    // already a full rectangular grid as distributed.
    // NOLINTBEGIN(misc-include-cleaner)
    static auto readFilledTensorGrid(const hid_t file, const std::size_t nWlNative,
        const std::string& h5pathStr) -> WDGridBuildResult
    {
        WDGridBuildResult result;
        result.logg_ = utils::readDataset1D(file, "logg", "SpecsynLibWD");
        result.logTeff_ = utils::readDataset1D(file, "log_Teff", "SpecsynLibWD");

        auto [flux, shape] = utils::readDataset3D(file, "flux", "SpecsynLibWD");
        const auto nLogg = static_cast<std::size_t>(shape.at(0));
        const auto nLogTeff = static_cast<std::size_t>(shape.at(1));
        const auto nWl = static_cast<std::size_t>(shape.at(2));
        if (nLogg != result.logg_.size() || nLogTeff != result.logTeff_.size() ||
            nWl != nWlNative)
        {
            throw std::runtime_error(
                "SpecsynLibWD: 'flux' dataset shape (" + std::to_string(nLogg) +
                ", " + std::to_string(nLogTeff) + ", " + std::to_string(nWl) +
                ") does not match the sizes of 'logg' (" + std::to_string(result.logg_.size()) +
                "), 'log_Teff' (" + std::to_string(result.logTeff_.size()) + "), and 'wl' (" +
                std::to_string(nWlNative) + ") in " + h5pathStr);
        }

        result.spectra_.assign(nLogg * nLogTeff, std::vector<double>{});
        for (std::size_t i = 0; i < nLogg; ++i)
        {
            for (std::size_t j = 0; j < nLogTeff; ++j)
            {
                const auto offset = (i * nLogTeff + j) * nWl;
                result.spectra_.at(i * nLogTeff + j) = std::vector<double>(
                    flux.begin() + static_cast<std::ptrdiff_t>(offset),
                    flux.begin() + static_cast<std::ptrdiff_t>(offset + nWl));
            }
        }
        return result;
    }
    // NOLINTEND(misc-include-cleaner)

    // Reads a flux dataset stored as a (n_models, n_wl) array
    // alongside per-model "logg"/"log_Teff" datasets (also n_models
    // long, and free to repeat values) -- e.g. the Rauch et al. NLTE
    // hot star grid fetched by fetch_rauch.py, whose (logg, Teff)
    // coverage is only partially filled (log g = 5 is missing above
    // Teff = 100000 K), so there is no way to lay the models out as a
    // full rectangular grid as distributed. Builds the (logg,
    // logTeff) tensor axes from the sorted set of unique values
    // actually present -- mirrors SpecsynLibNoWind's own
    // buildExactMatchGrid exactly -- and leaves every (logg, logTeff)
    // grid point with no matching model unpopulated (an empty
    // spectrum), exactly as SpecsynLibNoWind does for its own missing
    // grid points.
    // NOLINTBEGIN(misc-include-cleaner)
    static auto readSparseGrid(const hid_t file, const std::size_t nWlNative,
        const std::string& h5pathStr) -> WDGridBuildResult
    {
        const auto loggPerModel = utils::readDataset1D(file, "logg", "SpecsynLibWD");
        const auto logTeffPerModel = utils::readDataset1D(file, "log_Teff", "SpecsynLibWD");
        auto [flux, shape] = utils::readDataset2D(file, "flux", "SpecsynLibWD");
        const auto nModels = shape.first;
        const auto nWl = shape.second;
        if (loggPerModel.size() != nModels || logTeffPerModel.size() != nModels ||
            nWl != nWlNative)
        {
            throw std::runtime_error(
                "SpecsynLibWD: 'flux' dataset shape (" + std::to_string(nModels) +
                ", " + std::to_string(nWl) + ") does not match the sizes of 'logg' (" +
                std::to_string(loggPerModel.size()) + "), 'log_Teff' (" +
                std::to_string(logTeffPerModel.size()) + "), and 'wl' (" +
                std::to_string(nWlNative) + ") in " + h5pathStr);
        }

        WDGridBuildResult result;
        const std::set<double> loggSet(loggPerModel.begin(), loggPerModel.end());
        const std::set<double> logTeffSet(logTeffPerModel.begin(), logTeffPerModel.end());
        result.logg_.assign(loggSet.begin(), loggSet.end());
        result.logTeff_.assign(logTeffSet.begin(), logTeffSet.end());

        result.spectra_.assign(result.logg_.size() * result.logTeff_.size(), std::vector<double>{});
        for (std::size_t m = 0; m < nModels; ++m)
        {
            const auto iLogg = static_cast<std::size_t>(
                std::ranges::lower_bound(result.logg_, loggPerModel.at(m)) - result.logg_.begin());
            const auto iLogTeff = static_cast<std::size_t>(
                std::ranges::lower_bound(result.logTeff_, logTeffPerModel.at(m)) - result.logTeff_.begin());
            const auto offset = m * nWl;
            result.spectra_.at(iLogg * result.logTeff_.size() + iLogTeff) = std::vector<double>(
                flux.begin() + static_cast<std::ptrdiff_t>(offset),
                flux.begin() + static_cast<std::ptrdiff_t>(offset + nWl));
        }
        return result;
    }
    // NOLINTEND(misc-include-cleaner)

    template <OOBPolicy Policy>
    SpecsynLibWD<Policy>::SpecsynLibWD(
        const std::string& spectraName,
        const std::string& registryName,
        double wlMin,
        double wlMax,
        const std::size_t nWl,
        const io::SimControls& controls) :
        SpecsynLib2D<Policy>(controls),
        logg_(this->dim2_),
        logTeff_(this->dim3_)
    {
        // Step 1: find and open the HDF5 file this model names in the registry
        auto [registry, registryPath] = parseRegistry(registryName);
        const auto modelEntry = registry.at_path(spectraName);
        if (!modelEntry)
        {
            throw std::runtime_error(
                "SpecsynLibWD: spectral model '" + spectraName +
                "' not found in registry " + registryPath.string());
        }
        const auto h5name = modelEntry.at_path("file").value<std::string>();
        if (!h5name.has_value())
        {
            throw std::runtime_error(
                "SpecsynLibWD: registry entry '" + spectraName +
                "' in " + registryPath.string() + " is missing a 'file' field");
        }
        const auto h5path = registryPath.parent_path() / h5name.value();

        // NOLINTBEGIN(misc-include-cleaner)
        const hid_t file = H5Fopen(h5path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "SpecsynLibWD: unable to open HDF5 file " + h5path.string());
        }

        // Everything from here through the matching H5Fclose below is
        // wrapped in a try/catch so the file handle always gets
        // closed, regardless of which of the several things that can
        // go wrong while reading this file's four datasets actually
        // does -- mirrors SpecsynLibNoWind's own identical pattern.
        try
        {
            // Step 2: the wavelength grid
            this->wl_ = utils::readDataset1D(file, "wl", "SpecsynLibWD");

            // Step 3+4: logg_/logTeff_ (dim2_/dim3_) and the flux
            // data -- dispatched on the "flux" dataset's own rank
            // (a property of the data itself, not a registry flag)
            // rather than assuming a filled tensor, since a grid like
            // Rauch's stores flux as a (n_models, n_wl) array instead
            // of a filled (n_logg, n_logTeff, n_wl) tensor -- see
            // readFilledTensorGrid/readSparseGrid's own comments.
            // dim1_ is left empty entirely -- see SpecsynLib2D's own
            // comment for why.
            const int fluxRank = utils::datasetRank(file, "flux", "SpecsynLibWD");
            WDGridBuildResult built;
            if (fluxRank == 3)
            {
                built = readFilledTensorGrid(file, this->wl_.size(), h5path.string());
            }
            else if (fluxRank == 2)
            {
                built = readSparseGrid(file, this->wl_.size(), h5path.string());
            }
            else
            {
                throw std::runtime_error(
                    "SpecsynLibWD: 'flux' dataset in " + h5path.string() +
                    " has rank " + std::to_string(fluxRank) + " (expected 2 or 3)");
            }
            logg_ = std::move(built.logg_);
            logTeff_ = std::move(built.logTeff_);
            this->spectra_ = std::move(built.spectra_);
            using SpectraGrid = typename SpecsynLib<Policy>::SpectraGrid;
            this->grid_ = SpectraGrid(this->spectra_.data(), 1, logg_.size(), logTeff_.size());
        }
        catch (...)
        {
            H5Fclose(file);
            throw;
        }

        H5Fclose(file);
        // NOLINTEND(misc-include-cleaner)

        // If the caller requested an explicit output grid rather than
        // this library's own native one, resample onto it now --
        // mirrors SpecsynLibNoWind's own identical "nWl alone" handling
        if (nWl != 0)
        {
            if (wlMin == 0.0)
            {
                wlMin = this->wl_.front();
                wlMax = this->wl_.back();
            }
            this->resample(utils::logspace(wlMin, wlMax, nWl));
        }
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWD<Policy>::spec(const Specsyn::StarData& props, const double /*feh*/) const -> std::vector<double>
    {
        // Step 1: check log(Teff) against the grid's bounds
        const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logTe is one of its compile-time-known indices
        if (logTeff < logTeff_.front() || logTeff > logTeff_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWD: star with log(Teff) = " + std::to_string(logTeff) +
                " is outside this library's grid");
        }

        // Step 2: surface area and log(g), then bounds-check log(g)
        const auto [area, logg] = this->getSAandLogg(props);
        if (logg < logg_.front() || logg > logg_.back())
        {
            return SpecsynLib<Policy>::outOfBoundsResult(
                "SpecsynLibWD: star with log(g) = " + std::to_string(logg) +
                " is outside this library's grid");
        }

        // Step 3: bilinear interpolation is handled entirely by the
        // parent class, which knows nothing about surface area --
        // scale its result to convert specific flux at the surface
        // into specific luminosity
        auto result = this->SpecsynLib2D<Policy>::spec(logg, logTeff);
        for (auto& v : result) { v *= area; }
        return result;
    }

    template <OOBPolicy Policy>
    auto SpecsynLibWD<Policy>::specForce(const Specsyn::StarData& props, const double /*feh*/) const -> std::vector<double>
    {
        // Reject only if log(Teff) itself can't even be bracketed --
        // log(g) is never checked against its own range here, since
        // the whole point of this function is to search for a
        // populated log(g) value regardless of how far out of range
        // this star's own log(g) starts.
        const double logTeff = props[static_cast<std::size_t>(tracks::FieldIdx::logTe)]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- StarData is a fixed-size std::array, and logTe is one of its compile-time-known indices
        if (logTeff < logTeff_.front() || logTeff > logTeff_.back())
        {
            throw std::runtime_error(
                "SpecsynLibWD::specForce: star with log(Teff) = " + std::to_string(logTeff) +
                " is entirely outside this library's grid");
        }

        const auto [area, logg] = this->getSAandLogg(props);
        std::size_t cacheIdx = 0;
        const auto bTeff = detail::findBracket(logTeff_, logTeff, cacheIdx);

        // For each of the (up to two) bracketing logTeff columns,
        // search every logg_ value for whichever is both populated
        // and closest to this star's own (real, unclamped) log(g).
        // Renormalizing by wSum at the end -- exactly as
        // SpecsynLib2D::spec()'s own OOBPolicy::coerce handling
        // does for a partially-populated cell -- means a single
        // populated column's own spectrum is returned unscaled if
        // the other column has nothing populated at all, and a
        // proper log(Teff)-weighted blend of both otherwise.
        std::vector<double> result(this->wl_.size(), 0.0);
        double wSum = 0.0;
        for (int ti = 0; ti < 2; ++ti)
        {
            const std::size_t t = (ti == 0) ? bTeff.lo_ : bTeff.hi_;
            const double weight = (ti == 0) ? (1.0 - bTeff.t_) : bTeff.t_;
            if (weight == 0.0) { continue; } // degenerate axis or exact grid hit: skip a zero-weight column

            double bestDist = std::numeric_limits<double>::infinity();
            std::size_t bestG = 0;
            bool found = false;
            for (std::size_t g = 0; g < logg_.size(); ++g)
            {
                if (this->grid_[0, g, t].empty()) { continue; }
                const double dist = std::abs(logg_[g] - logg); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- g < logg_.size() by construction
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestG = g;
                    found = true;
                }
            }
            if (!found) { continue; } // this column has no populated log(g) value at all

            wSum += weight;
            const auto& spectrum = this->grid_[0, bestG, t];
            for (std::size_t w = 0; w < result.size(); ++w) { result[w] += weight * spectrum[w]; }
        }

        if (wSum == 0.0)
        {
            throw std::runtime_error(
                "SpecsynLibWD::specForce: no populated grid point found near "
                "log(Teff) = " + std::to_string(logTeff) + " at any log(g)");
        }
        for (auto& v : result) { v = (v / wSum) * area; }
        return result;
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the class's implementation in this .cpp file, as with
    // every other class in src/specsyn, rather than forcing it into
    // the header just because it is a template.
    template class SpecsynLibWD<OOBPolicy::raise>;
    template class SpecsynLibWD<OOBPolicy::silent>;
    template class SpecsynLibWD<OOBPolicy::coerce>;

} // namespace specsyn
