/**
 * @file Nebular.cpp
 * @author Mark Krumholz
 * @brief Implementation of Nebular.hpp
 * @date 2026-08-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Nebular.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../io/SimControls.hpp"
#include "../utils/Constants.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Disable linting in this file caused by including hdf5.h wholesale
// (rather than individual headers) -- the paradigm HDF5 itself
// wants -- see VegaSpectrum.cpp's own identical suppression
// NOLINTBEGIN(misc-include-cleaner)

namespace
{
    // This object's own nebular spectral grid, and the subset of the
    // table's own line list that falls inside it -- see
    // Nebular.hpp's own wl_ comment for the algorithm this implements
    struct WavelengthGridResult
    {
        std::vector<double> wl_;
        std::vector<double> lineWl_;
        std::vector<std::string> lineLabel_;
    };

    auto buildWavelengthGrid(const std::vector<double>& specsynWl,
        const std::vector<double>& lineWlAll,
        const std::vector<std::string>& lineLabelAll,
        const nebular::NebularControls& nebControls) -> WavelengthGridResult
    {
        WavelengthGridResult result;
        result.wl_ = specsynWl;

        const double wlMin = specsynWl.front();
        const double wlMax = specsynWl.back();

        // lineWidth_ is in km/s; utils::c is in cm/s -- convert
        // lineWidth_ to cm/s before dividing so the ratio (and so
        // fracHalfWidth below) comes out dimensionless
        const double lineWidthCgs = nebControls.lineWidth_ * 1.0e5;
        const double fracHalfWidth = nebControls.lineExtent_ * lineWidthCgs / utils::c;

        for (size_t i = 0; i < lineWlAll.size(); ++i)
        {
            const double lambda0 = lineWlAll.at(i);
            if (lambda0 < wlMin || lambda0 > wlMax) { continue; }

            result.lineWl_.push_back(lambda0);
            result.lineLabel_.push_back(lineLabelAll.at(i));

            const double lo = lambda0 * (1.0 - fracHalfWidth);
            const double hi = lambda0 * (1.0 + fracHalfWidth);
            for (size_t k = 0; k < nebControls.nGridLine_; ++k)
            {
                const double frac = (nebControls.nGridLine_ > 1)
                    ? static_cast<double>(k) / static_cast<double>(nebControls.nGridLine_ - 1)
                    : 0.0;
                result.wl_.push_back(lo + (frac * (hi - lo)));
            }
        }

        std::sort(result.wl_.begin(), result.wl_.end());
        return result;
    }

    // Every child group of parent that has an attribute called
    // attrName, as (value, open group handle) pairs sorted ascending
    // by that value. Every returned handle is the caller's own to
    // close, exactly once.
    auto childrenByAttr(const hid_t parent, const std::string& attrName,
        const std::string& context) -> std::vector<std::pair<double, hid_t>>
    {
        std::vector<std::pair<double, hid_t>> result;
        for (const auto& name : utils::listGroupDatasetNames(parent))
        {
            const hid_t child = H5Gopen2(parent, name.c_str(), H5P_DEFAULT);
            if (child < 0)
            {
                std::string msg = context;
                msg += ": unable to open group ";
                msg += name;
                throw std::runtime_error(msg);
            }
            result.emplace_back(utils::readRequiredScalarAttr(child, attrName, context), child);
        }
        std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return result;
    }

    // Every logU child group of vvcritGrp that has its own subtype
    // (e.g. "cluster" or "galaxy") subgroup, as (logU, open subtype
    // subgroup handle) pairs sorted ascending by logU -- the parent
    // logU group itself is closed here, since only its subtype
    // subgroup is needed by the caller. Every returned handle is the
    // caller's own to close, exactly once.
    auto logUCandidates(const hid_t vvcritGrp, const std::string& subtype,
        const std::string& context) -> std::vector<std::pair<double, hid_t>>
    {
        std::vector<std::pair<double, hid_t>> result;
        for (const auto& name : utils::listGroupDatasetNames(vvcritGrp))
        {
            const hid_t logUGrp = H5Gopen2(vvcritGrp, name.c_str(), H5P_DEFAULT);
            if (logUGrp < 0)
            {
                std::string msg = context;
                msg += ": unable to open group ";
                msg += name;
                throw std::runtime_error(msg);
            }
            if (H5Lexists(logUGrp, subtype.c_str(), H5P_DEFAULT) <= 0)
            {
                H5Gclose(logUGrp);
                continue;
            }
            const double logU = utils::readRequiredScalarAttr(logUGrp, "logU", context);
            const hid_t subGrp = H5Gopen2(logUGrp, subtype.c_str(), H5P_DEFAULT);
            H5Gclose(logUGrp);
            if (subGrp < 0)
            {
                std::string msg = context;
                msg += ": unable to open group ";
                msg += subtype;
                throw std::runtime_error(msg);
            }
            result.emplace_back(logU, subGrp);
        }
        std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return result;
    }

    // Read a flat "spec" dataset out of a cluster/galaxy subtype
    // group -- galaxy's own spec is already 1D (nwl,); cluster's own
    // is 2D (ntime, nwl), read here as the same flattened,
    // row-major layout
    auto readFlatSpec(const hid_t subGrp, const bool is2D,
        const std::string& context) -> std::vector<double>
    {
        if (is2D) { return utils::readDataset2D(subGrp, "spec", context).first; }
        return utils::readDataset1D(subGrp, "spec", context);
    }

    // Linearly blend (or, on an exact hit, select) the logU-bracketing
    // "spec" data closest to requestedLogU among candidates, closing
    // every one of candidates' own group handles exactly once before
    // returning. Throws if candidates is empty, or requestedLogU
    // falls outside the range of logU values candidates actually
    // covers.
    auto blendSpecByLogU(std::vector<std::pair<double, hid_t>> candidates,
        const bool is2D, const double requestedLogU,
        const std::string& context) -> std::vector<double>
    {
        if (candidates.empty())
        {
            throw std::runtime_error(context + ": no logU data available to interpolate");
        }

        const double logUMin = candidates.front().first;
        const double logUMax = candidates.back().first;
        if (requestedLogU < logUMin || requestedLogU > logUMax)
        {
            for (const auto& [logU, grp] : candidates) { H5Gclose(grp); }
            throw std::runtime_error(context + ": requested logU is outside the tabulated range");
        }

        std::vector<double> result;
        size_t hiIdx = 0;
        while (hiIdx < candidates.size() && candidates.at(hiIdx).first < requestedLogU) { ++hiIdx; }

        if (utils::approxEqual(candidates.at(hiIdx).first, requestedLogU))
        {
            result = readFlatSpec(candidates.at(hiIdx).second, is2D, context);
        }
        else
        {
            const auto& lo = candidates.at(hiIdx - 1);
            const auto& hi = candidates.at(hiIdx);
            const auto loData = readFlatSpec(lo.second, is2D, context);
            const auto hiData = readFlatSpec(hi.second, is2D, context);
            const double frac = (requestedLogU - lo.first) / (hi.first - lo.first);
            result.resize(loData.size());
            for (size_t k = 0; k < result.size(); ++k)
            {
                result.at(k) = loData.at(k) + (frac * (hiData.at(k) - loData.at(k)));
            }
        }

        for (const auto& [logU, grp] : candidates) { H5Gclose(grp); }
        return result;
    }

    // Interpolate nativeSpec (defined on nativeWl) onto targetWl,
    // using a Steffen-spline Interpolator1D built from (nativeWl,
    // nativeSpec); targetWl points outside [nativeWl.front(),
    // nativeWl.back()] are set to zero rather than evaluated, mirroring
    // data/tools/cloudy/process_cloudy_grid.py's own _normalize_row
    // (which zero-pads a row's continuum wherever the global grid
    // extends past that row's own native wl range)
    auto resampleZeroPad(const std::vector<double>& nativeWl,
        const std::vector<double>& nativeSpec,
        const std::vector<double>& targetWl) -> std::vector<double>
    {
        const interp::Interpolator1D<1> interpolator(nativeWl, nativeSpec);
        std::vector<double> result(targetWl.size(), 0.0);
        for (size_t k = 0; k < targetWl.size(); ++k)
        {
            const double x = targetWl.at(k);
            if (x >= nativeWl.front() && x <= nativeWl.back())
            {
                result.at(k) = interpolator(x);
            }
        }
        return result;
    }
} // namespace

nebular::Nebular::Nebular(
    const std::string& tableName,
    const std::string& trackName,
    const io::SimControls& simControls,
    const double vvcrit) :
    simControls_(simControls)
{
    static constexpr auto context = "Nebular";

    const auto tablePath = utils::getFilePath(tableName);
    if (tablePath.empty())
    {
        throw std::runtime_error(
            std::string(context) + ": nebular emission table " + tableName + " not found");
    }

    const auto& specsynWl = simControls_.specsyn()->wl();

    // The HDF5 build linked here is not safe to call from two threads
    // at once at all, even across entirely separate files -- see
    // OutputManagerH5::openOutputFile()'s own comment for the full
    // story -- so this shares that same global critical section
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        const hid_t file = H5Fopen(tablePath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                std::string(context) + ": unable to open HDF5 file " + tablePath.string());
        }

        const auto nativeWl = utils::readDataset1D(file, "wl", context);
        const auto lineWlAll = utils::readDataset1D(file, "line_wl", context);
        const auto lineLabelAll = utils::readStringDataset1D(file, "line_label", context);
        clusterAge_ = utils::readDataset1D(file, "time", context);

        auto grid = buildWavelengthGrid(specsynWl, lineWlAll, lineLabelAll, simControls_.nebControls());
        wl_ = std::move(grid.wl_);
        lineWl_ = std::move(grid.lineWl_);
        lineLabel_ = std::move(grid.lineLabel_);

        const hid_t trackGrp = H5Gopen2(file, trackName.c_str(), H5P_DEFAULT);
        if (trackGrp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                std::string(context) + ": track " + trackName + " not found in table " +
                tablePath.string());
        }

        auto fehGroups = childrenByAttr(trackGrp, "FeH", context);
        feH_.clear();
        feH_.reserve(fehGroups.size());
        for (const auto& [feh, grp] : fehGroups) { feH_.push_back(feh); }

        const size_t nWl = wl_.size();
        const size_t nTimeNative = clusterAge_.size();
        ctmLumPerQGalaxyData_.assign(feH_.size() * nWl, 0.0);
        ctmLumPerQClusterData_.assign(feH_.size() * nTimeNative * nWl, 0.0);

        const double logURequested = simControls_.nebControls().logU_;

        for (size_t i = 0; i < fehGroups.size(); ++i)
        {
            const hid_t fehGrp = fehGroups.at(i).second;

            auto vvcritGroups = childrenByAttr(fehGrp, "v_vcrit", context);
            const auto vvcritMatch = std::find_if(vvcritGroups.begin(), vvcritGroups.end(),
                [vvcrit](const auto& entry) { return utils::approxEqual(entry.first, vvcrit); });
            if (vvcritMatch == vvcritGroups.end())
            {
                for (const auto& [vc, grp] : vvcritGroups) { H5Gclose(grp); }
                for (const auto& [feh, grp] : fehGroups) { H5Gclose(grp); }
                H5Gclose(trackGrp);
                H5Fclose(file);
                throw std::runtime_error(
                    std::string(context) + ": no exact v/vcrit match for track " + trackName +
                    " at [Fe/H] = " + std::to_string(feH_.at(i)));
            }
            const hid_t vvcritGrp = vvcritMatch->second;

            const auto galaxySpec = blendSpecByLogU(
                logUCandidates(vvcritGrp, "galaxy", context), false, logURequested, context);
            const auto galaxyResampled = resampleZeroPad(nativeWl, galaxySpec, wl_);
            std::copy(galaxyResampled.begin(), galaxyResampled.end(),
                ctmLumPerQGalaxyData_.begin() + static_cast<std::ptrdiff_t>(i * nWl));

            const auto clusterSpec = blendSpecByLogU(
                logUCandidates(vvcritGrp, "cluster", context), true, logURequested, context);
            for (size_t t = 0; t < nTimeNative; ++t)
            {
                const std::vector<double> row(
                    clusterSpec.begin() + static_cast<std::ptrdiff_t>(t * nativeWl.size()),
                    clusterSpec.begin() + static_cast<std::ptrdiff_t>((t + 1) * nativeWl.size()));
                const auto rowResampled = resampleZeroPad(nativeWl, row, wl_);
                std::copy(rowResampled.begin(), rowResampled.end(),
                    ctmLumPerQClusterData_.begin() +
                        static_cast<std::ptrdiff_t>(((i * nTimeNative) + t) * nWl));
            }

            for (const auto& [vc, grp] : vvcritGroups) { H5Gclose(grp); }
        }

        for (const auto& [feh, grp] : fehGroups) { H5Gclose(grp); }
        H5Gclose(trackGrp);
        H5Fclose(file);

        ctmLumPerQGalaxy_ = Grid2D(ctmLumPerQGalaxyData_.data(), feH_.size(), nWl);
        ctmLumPerQCluster_ = Grid3D(ctmLumPerQClusterData_.data(), feH_.size(), nTimeNative, nWl);
    }
}

// NOLINTEND(misc-include-cleaner)
