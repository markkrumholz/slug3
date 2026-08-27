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
#include <cmath>
#include <cstddef>
#include <numbers>
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

    // Read a flat dataset (either "spec" or "line_lum") out of a
    // cluster/galaxy subtype group -- galaxy's own copy of either is
    // already 1D (nwl,)/(nline,); cluster's own is 2D
    // (ntime, nwl)/(ntime, nline), read here as the same flattened,
    // row-major layout
    auto readFlatDataset(const hid_t subGrp, const std::string& datasetName,
        const bool is2D, const std::string& context) -> std::vector<double>
    {
        if (is2D) { return utils::readDataset2D(subGrp, datasetName, context).first; }
        return utils::readDataset1D(subGrp, datasetName, context);
    }

    // Linearly blend (or, on an exact hit, select) the logU-bracketing
    // datasetName data closest to requestedLogU among candidates,
    // closing every one of candidates' own group handles exactly once
    // before returning. Throws if candidates is empty, or
    // requestedLogU falls outside the range of logU values candidates
    // actually covers.
    auto blendByLogU(std::vector<std::pair<double, hid_t>> candidates,
        const std::string& datasetName, const bool is2D, const double requestedLogU,
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
            result = readFlatDataset(candidates.at(hiIdx).second, datasetName, is2D, context);
        }
        else
        {
            const auto& lo = candidates.at(hiIdx - 1);
            const auto& hi = candidates.at(hiIdx);
            const auto loData = readFlatDataset(lo.second, datasetName, is2D, context);
            const auto hiData = readFlatDataset(hi.second, datasetName, is2D, context);
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

    // Threshold below which a line's own deposited power fraction in a
    // wavelength bin is treated as negligible -- see
    // computeLineDepositWindows()'s own comment for how this bounds
    // the width of each line's own deposit window
    constexpr double lineDepositThreshold = 1.0e-3;

    // For every line in lineWl (in the same order), the precomputed
    // data Nebular's own line-deposition step (implemented in a future
    // commit) will need to add that line's power into wl's own bins --
    // see Nebular.hpp's own lineCenterIdx_/lineDepositFrac_ comments
    struct LineDepositWindows
    {
        std::vector<size_t> centerIdx_;         /**< Each line's own index into wl of the bin its central wavelength falls into; 0 (with an empty frac_ entry) for a line whose central wavelength falls outside wl's own range */
        std::vector<std::vector<double>> frac_; /**< Each line's own fraction of power landing in each bin of the window (odd length, centered on centerIdx_) around its own central bin; empty for a line whose central wavelength falls outside wl's own range */
    };

    // For a line with central wavelength wlCen and, expressed as a
    // wavelength, standard deviation deltaWl (a Gaussian line profile
    // is assumed), the fraction of that line's own power landing in
    // wl's own bin binIdx -- whose own edges are the geometric mean of
    // wl's own bracketing pairs of central wavelengths (boundaries.at(i)
    // is the edge shared by bins i and i+1), except at the two ends of
    // wl, which are treated as open (extending to -/+ infinity)
    auto binPowerFraction(const std::vector<double>& boundaries, const size_t nWl,
        const size_t binIdx, const double wlCen, const double deltaWl) -> double
    {
        const double erfLo = (binIdx == 0)
            ? -1.0
            : std::erf((boundaries.at(binIdx - 1) - wlCen) / (deltaWl * std::numbers::sqrt2));
        const double erfHi = (binIdx == nWl - 1)
            ? 1.0
            : std::erf((boundaries.at(binIdx) - wlCen) / (deltaWl * std::numbers::sqrt2));
        return 0.5 * (erfHi - erfLo);
    }

    // Build, for every line in lineWl whose own central wavelength
    // falls within wl's own range, the window of wl bins its own power
    // should be deposited into -- starting from that line's own
    // central bin and expanding outward, one bin at a time on each
    // side, until the newly added bin on both sides would receive less
    // than lineDepositThreshold of that line's own power (a bin beyond
    // wl's own edge is treated as receiving zero power, rather than
    // stopping the expansion outright, so the window can still grow on
    // its in-range side)
    auto computeLineDepositWindows(const std::vector<double>& wl,
        const std::vector<double>& lineWl, const double lineWidthKms) -> LineDepositWindows
    {
        const size_t nWl = wl.size();
        const size_t nLine = lineWl.size();

        // boundaries.at(i) is the edge shared by bins i and i+1
        std::vector<double> boundaries(nWl - 1);
        for (size_t i = 0; i < nWl - 1; ++i) { boundaries.at(i) = std::sqrt(wl.at(i) * wl.at(i + 1)); }

        // lineWidthKms is in km/s; utils::c is in cm/s -- convert
        // lineWidthKms to cm/s before dividing so the ratio (and so
        // deltaWl below) comes out dimensionless, leaving deltaWl in
        // wl's own units
        const double lineWidthCgs = lineWidthKms * 1.0e5;

        LineDepositWindows result;
        result.centerIdx_.assign(nLine, 0);
        result.frac_.assign(nLine, {});

        for (size_t ell = 0; ell < nLine; ++ell)
        {
            const double wlCen = lineWl.at(ell);
            if (wlCen < wl.front() || wlCen > wl.back()) { continue; }

            const double deltaWl = wlCen * lineWidthCgs / utils::c;

            const size_t centerIdx = static_cast<size_t>(std::distance(boundaries.begin(),
                std::upper_bound(boundaries.begin(), boundaries.end(), wlCen)));
            result.centerIdx_.at(ell) = centerIdx;

            std::vector<double> frac{binPowerFraction(boundaries, nWl, centerIdx, wlCen, deltaWl)};
            for (size_t step = 1;; ++step)
            {
                const bool hasLo = centerIdx >= step;
                const bool hasHi = centerIdx + step <= nWl - 1;
                const double fracLo =
                    hasLo ? binPowerFraction(boundaries, nWl, centerIdx - step, wlCen, deltaWl) : 0.0;
                const double fracHi =
                    hasHi ? binPowerFraction(boundaries, nWl, centerIdx + step, wlCen, deltaWl) : 0.0;
                if (fracLo < lineDepositThreshold && fracHi < lineDepositThreshold) { break; }
                frac.insert(frac.begin(), fracLo);
                frac.push_back(fracHi);
            }
            result.frac_.at(ell) = std::move(frac);
        }

        return result;
    }
} // namespace

nebular::Nebular::Nebular(
    const std::string& tableName,
    const std::string& trackName,
    const io::SimControls& simControls,
    const double vvcrit) :
    simControls_(simControls), qhiFilter_("Q(HI)")
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
        wl_ = specsynWl;
        lineWl_ = utils::readDataset1D(file, "line_wl", context);
        lineLabel_ = utils::readStringDataset1D(file, "line_label", context);
        clusterAge_ = utils::readDataset1D(file, "time", context);

        auto lineDeposit =
            computeLineDepositWindows(wl_, lineWl_, simControls_.nebControls().lineWidth_);
        lineCenterIdx_ = std::move(lineDeposit.centerIdx_);

        size_t depositWidth = 1;
        for (const auto& frac : lineDeposit.frac_) { depositWidth = std::max(depositWidth, frac.size()); }

        // Every line's own frac_ window is centered within the
        // depositWidth-wide row allotted to it here, zero-padded on
        // both sides out to depositWidth -- so a line with a narrower
        // window than depositWidth (or none at all, for a line whose
        // central wavelength fell outside wl_'s own range) simply
        // contributes zero power outside its own window
        lineDepositFracData_.assign(depositWidth * lineWl_.size(), 0.0);
        for (size_t ell = 0; ell < lineWl_.size(); ++ell)
        {
            const auto& frac = lineDeposit.frac_.at(ell);
            const size_t offset = (depositWidth - frac.size()) / 2;
            for (size_t k = 0; k < frac.size(); ++k)
            {
                lineDepositFracData_.at(((offset + k) * lineWl_.size()) + ell) = frac.at(k);
            }
        }
        lineDepositFrac_ = Grid2D(lineDepositFracData_.data(), depositWidth, lineWl_.size());

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
        const size_t nLine = lineWl_.size();
        const size_t nTimeNative = clusterAge_.size();
        ctmLumPerQGalaxyData_.assign(feH_.size() * nWl, 0.0);
        ctmLumPerQClusterData_.assign(feH_.size() * nTimeNative * nWl, 0.0);
        lineLumPerQGalaxyData_.assign(feH_.size() * nLine, 0.0);
        lineLumPerQClusterData_.assign(feH_.size() * nTimeNative * nLine, 0.0);

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

            const auto galaxySpec = blendByLogU(
                logUCandidates(vvcritGrp, "galaxy", context), "spec", false, logURequested, context);
            const auto galaxyResampled = resampleZeroPad(nativeWl, galaxySpec, wl_);
            std::copy(galaxyResampled.begin(), galaxyResampled.end(),
                ctmLumPerQGalaxyData_.begin() + static_cast<std::ptrdiff_t>(i * nWl));

            const auto clusterSpec = blendByLogU(
                logUCandidates(vvcritGrp, "cluster", context), "spec", true, logURequested, context);
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

            // Lines: no wavelength resampling needed (unlike the
            // continuum above) -- the blended line_lum data already
            // covers the table's full, global line list, matching
            // lineWl_/lineLabel_ exactly
            const auto galaxyLineLum = blendByLogU(
                logUCandidates(vvcritGrp, "galaxy", context), "line_lum", false, logURequested, context);
            std::copy(galaxyLineLum.begin(), galaxyLineLum.end(),
                lineLumPerQGalaxyData_.begin() + static_cast<std::ptrdiff_t>(i * nLine));

            const auto clusterLineLum = blendByLogU(
                logUCandidates(vvcritGrp, "cluster", context), "line_lum", true, logURequested, context);
            std::copy(clusterLineLum.begin(), clusterLineLum.end(),
                lineLumPerQClusterData_.begin() +
                    static_cast<std::ptrdiff_t>(i * nTimeNative * nLine));

            for (const auto& [vc, grp] : vvcritGroups) { H5Gclose(grp); }
        }

        for (const auto& [feh, grp] : fehGroups) { H5Gclose(grp); }
        H5Gclose(trackGrp);
        H5Fclose(file);

        ctmLumPerQGalaxy_ = Grid2D(ctmLumPerQGalaxyData_.data(), feH_.size(), nWl);
        ctmLumPerQCluster_ = Grid3D(ctmLumPerQClusterData_.data(), feH_.size(), nTimeNative, nWl);
        lineLumPerQGalaxy_ = Grid2D(lineLumPerQGalaxyData_.data(), feH_.size(), nLine);
        lineLumPerQCluster_ = Grid3D(lineLumPerQClusterData_.data(), feH_.size(), nTimeNative, nLine);
    }
}

// NOLINTEND(misc-include-cleaner)
