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
#include <exception>
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

    // Throw if a just-blended dataset's own size doesn't match what
    // the constructor is about to slice/copy it as -- e.g. a
    // malformed table whose "cluster"/"spec" dataset isn't exactly
    // (ntime, nwl) for the ntime/nwl this Nebular otherwise expects.
    // Factored out of Nebular's own constructor, where this same check
    // is needed after every blendByLogU() call, to keep it within its
    // cognitive-complexity budget.
    void checkDatasetSize(const std::string& context, const std::string& label,
        const size_t actual, const size_t expected)
    {
        if (actual != expected)
        {
            throw std::runtime_error(context + ": " + label + " dataset has " +
                std::to_string(actual) + " elements, expected " + std::to_string(expected));
        }
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

    // Best-effort close of whatever HDF5 group/file handles
    // Nebular::loadTable() had open when an exception interrupted it.
    // Factored out purely to keep loadTable() within its own
    // cognitive-complexity budget.
    void closeLoadTableHandles(const std::vector<std::pair<double, hid_t>>& vvcritGroups,
        const std::vector<std::pair<double, hid_t>>& fehGroups, const hid_t trackGrp, const hid_t file)
    {
        for (const auto& [vc, grp] : vvcritGroups) { if (grp >= 0) { H5Gclose(grp); } }
        for (const auto& [feh, grp] : fehGroups) { if (grp >= 0) { H5Gclose(grp); } }
        if (trackGrp >= 0) { H5Gclose(trackGrp); }
        if (file >= 0) { H5Fclose(file); }
    }

    // Load one [Fe/H] group's own galaxy/cluster spec/line_lum data
    // into index i of every ctmLumPerQ*/lineLumPerQ* output array.
    // Finds fehGrp's own v_vcrit child matching vvcrit exactly
    // (throwing if none exists), then blends/validates/copies its
    // galaxy and cluster spec/line_lum data -- see
    // Nebular::loadTable()'s own comment for why validation happens
    // here rather than trusting the table. vvcritGroups is scratch
    // storage the caller owns (so its own catch block can close it if
    // this function throws partway through); assigned here, and left
    // empty again on successful return. Factored out of
    // Nebular::loadTable() to keep it within its own cognitive-
    // complexity budget, not for any reuse elsewhere.
    void processFehGroup(const hid_t fehGrp, const double vvcrit, const std::string& trackName,
        const std::vector<double>& nativeWl, const std::vector<double>& wl, const std::string& context,
        const double logURequested, const size_t i, const size_t nWl, const size_t nLine,
        const size_t nTimeNative, const double feh, std::vector<std::pair<double, hid_t>>& vvcritGroups,
        std::vector<double>& ctmLumPerQGalaxyData, std::vector<double>& ctmLumPerQClusterData,
        std::vector<double>& lineLumPerQGalaxyData, std::vector<double>& lineLumPerQClusterData)
    {
        vvcritGroups = childrenByAttr(fehGrp, "v_vcrit", context);
        const auto vvcritMatch = std::find_if(vvcritGroups.begin(), vvcritGroups.end(),
            [vvcrit](const auto& entry) { return utils::approxEqual(entry.first, vvcrit); });
        if (vvcritMatch == vvcritGroups.end())
        {
            throw std::runtime_error(
                std::string(context) + ": no exact v/vcrit match for track " + trackName +
                " at [Fe/H] = " + std::to_string(feh));
        }
        const hid_t vvcritGrp = vvcritMatch->second;

        const auto galaxySpec = blendByLogU(
            logUCandidates(vvcritGrp, "galaxy", context), "spec", false, logURequested, context);
        checkDatasetSize(context, "galaxy spec", galaxySpec.size(), nativeWl.size());
        const auto galaxyResampled = resampleZeroPad(nativeWl, galaxySpec, wl);
        std::copy(galaxyResampled.begin(), galaxyResampled.end(),
            ctmLumPerQGalaxyData.begin() + static_cast<std::ptrdiff_t>(i * nWl));

        const auto clusterSpec = blendByLogU(
            logUCandidates(vvcritGrp, "cluster", context), "spec", true, logURequested, context);
        checkDatasetSize(context, "cluster spec", clusterSpec.size(), nTimeNative * nativeWl.size());
        for (size_t t = 0; t < nTimeNative; ++t)
        {
            const std::vector<double> row(
                clusterSpec.begin() + static_cast<std::ptrdiff_t>(t * nativeWl.size()),
                clusterSpec.begin() + static_cast<std::ptrdiff_t>((t + 1) * nativeWl.size()));
            const auto rowResampled = resampleZeroPad(nativeWl, row, wl);
            std::copy(rowResampled.begin(), rowResampled.end(),
                ctmLumPerQClusterData.begin() +
                    static_cast<std::ptrdiff_t>(((i * nTimeNative) + t) * nWl));
        }

        // Lines: no wavelength resampling needed (unlike the continuum
        // above) -- the blended line_lum data already covers the
        // table's full, global line list, matching lineWl_/lineLabel_
        // exactly
        const auto galaxyLineLum = blendByLogU(
            logUCandidates(vvcritGrp, "galaxy", context), "line_lum", false, logURequested, context);
        checkDatasetSize(context, "galaxy line_lum", galaxyLineLum.size(), nLine);
        std::copy(galaxyLineLum.begin(), galaxyLineLum.end(),
            lineLumPerQGalaxyData.begin() + static_cast<std::ptrdiff_t>(i * nLine));

        const auto clusterLineLum = blendByLogU(
            logUCandidates(vvcritGrp, "cluster", context), "line_lum", true, logURequested, context);
        checkDatasetSize(context, "cluster line_lum", clusterLineLum.size(), nLine * nTimeNative);
        std::copy(clusterLineLum.begin(), clusterLineLum.end(),
            lineLumPerQClusterData.begin() +
                static_cast<std::ptrdiff_t>(i * nTimeNative * nLine));

        for (const auto& [vc, grp] : vvcritGroups) { H5Gclose(grp); }
        vvcritGroups.clear();
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

            const auto centerIdx = static_cast<size_t>(std::distance(boundaries.begin(),
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

    // (lo_, hi_, frac_) bracketing value within grid (ascending), such
    // that value == grid.at(lo_) + frac_ * (grid.at(hi_) - grid.at(lo_));
    // lo_ == hi_ (frac_ = 0) on an exact match, including at either
    // end of grid. Throws if value falls outside [grid.front(), grid.back()].
    struct GridBracket
    {
        size_t lo_;
        size_t hi_;
        double frac_;
    };

    auto bracketGrid(const std::vector<double>& grid, const double value,
        const std::string& context, const std::string& gridName) -> GridBracket
    {
        if (value < grid.front() || value > grid.back())
        {
            throw std::runtime_error(
                context + ": requested " + gridName + " is outside the tabulated range");
        }

        size_t hi = 0;
        while (hi + 1 < grid.size() && grid[hi] < value) { ++hi; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hi + 1 < grid.size() already checked by the loop condition itself, so hi < grid.size()

        if (utils::approxEqual(grid[hi], value)) { return {hi, hi, 0.0}; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hi < grid.size() by construction: it starts at 0 and only ever increments while hi + 1 < grid.size()

        // hi >= 1 here: hi == 0 would already have matched grid.front()
        // exactly above (value >= grid.front() by the range check at
        // the top of this function, and the loop above only stops at
        // hi == 0 when grid[0] >= value, so the two sandwich grid[0]
        // == value exactly)
        const double frac = (value - grid[hi - 1]) / (grid[hi] - grid[hi - 1]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        return {hi - 1, hi, frac};
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

    // The HDF5 build linked here is not safe to call from two threads
    // at once at all, even across entirely separate files -- see
    // OutputManagerH5::openOutputFile()'s own comment for the full
    // story -- so this shares that same global critical section.
    // Letting an exception escape a #pragma omp critical block is not
    // something the OpenMP runtime reliably unwinds cleanly -- the
    // lock's own release can be skipped, permanently blocking every
    // other thread's own h5ThreadSafety-guarded HDF5 I/O (e.g.
    // OutputManagerH5's own row-writing) for the rest of the run --
    // so any exception loadTable() raises (e.g. a malformed or
    // incomplete table -- see blendByLogU()'s own "no logU data"/
    // "outside the tabulated range" throws) is instead caught here,
    // captured into caught (declared outside the critical block,
    // before it even opens), and only actually rethrown once this
    // block -- and so the critical section itself -- has exited
    // normally. loadTable() itself is responsible for cleaning up
    // every HDF5 handle it opens before letting an exception escape
    // it, so there is nothing further to clean up here.
    std::exception_ptr caught;
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        try
        {
            loadTable(trackName, vvcrit, tablePath);
        }
        catch (...)
        {
            caught = std::current_exception();
        }
    }
    if (caught) { std::rethrow_exception(caught); }
}

void nebular::Nebular::loadTable(const std::string& trackName, const double vvcrit,
    const std::filesystem::path& tablePath)
{
    static constexpr auto context = "Nebular";
    const auto& specsynWl = simControls_.specsyn()->wl();

    hid_t file = -1;
    hid_t trackGrp = -1;
    std::vector<std::pair<double, hid_t>> fehGroups;
    std::vector<std::pair<double, hid_t>> vvcritGroups;
    try
    {
        file = H5Fopen(tablePath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                std::string(context) + ": unable to open HDF5 file " + tablePath.string());
        }

        const auto nativeWl = utils::readDataset1D(file, "wl", context);
        wl_ = specsynWl;
        lineWl_ = utils::readDataset1D(file, "line_wl", context);
        lineLabel_ = utils::readStringDataset1D(file, "line_label", context);
        if (lineLabel_.size() != lineWl_.size())
        {
            throw std::runtime_error(
                std::string(context) + ": line_label has " + std::to_string(lineLabel_.size()) +
                " entries, but line_wl has " + std::to_string(lineWl_.size()));
        }
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

        trackGrp = H5Gopen2(file, trackName.c_str(), H5P_DEFAULT);
        if (trackGrp < 0)
        {
            throw std::runtime_error(
                std::string(context) + ": track " + trackName + " not found in table " +
                tablePath.string());
        }

        fehGroups = childrenByAttr(trackGrp, "FeH", context);
        if (fehGroups.empty())
        {
            throw std::runtime_error(
                std::string(context) + ": track " + trackName + " has no [Fe/H] groups");
        }
        if (clusterAge_.empty())
        {
            throw std::runtime_error(
                std::string(context) + ": table has no cluster ages (empty 'time' dataset)");
        }
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
            processFehGroup(fehGroups.at(i).second, vvcrit, trackName, nativeWl, wl_, context,
                logURequested, i, nWl, nLine, nTimeNative, feH_.at(i), vvcritGroups,
                ctmLumPerQGalaxyData_, ctmLumPerQClusterData_,
                lineLumPerQGalaxyData_, lineLumPerQClusterData_);
        }

        for (const auto& [feh, grp] : fehGroups) { H5Gclose(grp); }
        fehGroups.clear();
        H5Gclose(trackGrp);
        trackGrp = -1;
        H5Fclose(file);
        file = -1;

        ctmLumPerQGalaxy_ = Grid2D(ctmLumPerQGalaxyData_.data(), feH_.size(), nWl);
        ctmLumPerQCluster_ = Grid3D(ctmLumPerQClusterData_.data(), feH_.size(), nTimeNative, nWl);
        lineLumPerQGalaxy_ = Grid2D(lineLumPerQGalaxyData_.data(), feH_.size(), nLine);
        lineLumPerQCluster_ = Grid3D(lineLumPerQClusterData_.data(), feH_.size(), nTimeNative, nLine);
    }
    catch (...)
    {
        // Best-effort cleanup of whatever the try block above left
        // open, then rethrow -- the exception itself must propagate
        // out of loadTable() so the constructor's own critical-section
        // wrapper (see its own comment) can catch and re-raise it once
        // that section has exited.
        closeLoadTableHandles(vvcritGroups, fehGroups, trackGrp, file);
        throw;
    }
}

auto nebular::Nebular::stellarAboveEdge(const std::vector<double>& spec) const -> std::vector<double>
{
    const double edgeWl = qhiFilter_.wlMax();
    std::vector<double> result(spec.size(), 0.0);
    for (size_t k = 0; k < spec.size(); ++k)
    {
        // spec is always on wl_'s own grid (see this function's own
        // callers' shared contract), so k < spec.size() == wl_.size();
        // result was constructed with spec.size() elements above
        if (wl_[k] > edgeWl) { result[k] = spec[k]; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
    }
    return result;
}

void nebular::Nebular::depositLines(
    const std::vector<double>& lineLum, std::vector<double>& spec) const
{
    const size_t nWl = wl_.size();
    const size_t depositWidth = lineDepositFrac_.extent(0);
    const auto centerRow = static_cast<ptrdiff_t>((depositWidth - 1) / 2);

    for (size_t ell = 0; ell < lineLum.size(); ++ell)
    {
        // lineCenterIdx_ has one entry per line, the same nLine as
        // lineLum always has (see getGalaxy()'s/getCluster()'s own
        // construction of it), so ell < lineLum.size() == lineCenterIdx_.size()
        if (lineLum[ell] == 0.0) { continue; } // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

        const auto centerIdx = static_cast<ptrdiff_t>(lineCenterIdx_[ell]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        // Offsets range over the deposit window's own full half-width,
        // centerRow on each side of the center bin (offset = 0), so
        // every row lineDepositFrac_ actually stores -- including the
        // lo/hi neighbor bins computeLineDepositWindows() computed --
        // gets deposited, not just the center bin.
        for (ptrdiff_t offset = -centerRow; offset <= centerRow; ++offset)
        {
            const ptrdiff_t binIdx = centerIdx + offset;

            // A bin's own width (below) needs its own left and right
            // neighbors, so bin 0 (no left neighbor) and bin nWl-1 (no
            // right neighbor) -- and anything further off the grid --
            // are skipped rather than read out of bounds
            if (binIdx < 1 || binIdx >= static_cast<ptrdiff_t>(nWl) - 1) { continue; }

            // bin is in [1, nWl - 2] here (the skip above), so bin - 1
            // and bin + 1 both stay within [0, nWl - 1] == wl_'s own
            // valid range, and spec (always on wl_'s own grid, per
            // stellarAboveEdge()'s own contract) has the same size
            const auto bin = static_cast<size_t>(binIdx);
            const double binWidth = std::sqrt(wl_[bin + 1] * wl_[bin]) - // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
                std::sqrt(wl_[bin - 1] * wl_[bin]); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above

            const auto row = static_cast<size_t>(centerRow + offset);
            const double powerFrac = lineDepositFrac_[row, ell];

            spec[bin] += lineLum[ell] * powerFrac / binWidth; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- see above
        }
    }
}

auto nebular::Nebular::getGalaxy(const std::vector<double>& spec, const double feH) const
    -> std::pair<std::vector<double>, std::vector<double>>
{
    static constexpr auto context = "Nebular::getGalaxy";

    const auto feHBracket = bracketGrid(feH_, feH, context, "[Fe/H]");
    // covFac_ discounts Q(HI) for ionizing photons lost to dust-grain
    // absorption or escape outside the observational aperture, before
    // any of it is attributed to nebular emission below
    const double qhi = simControls_.nebControls().covFac_ * qhiFilter_.phot(wl_, spec);

    const size_t nLine = lineWl_.size();
    std::vector<double> lineLum(nLine);
    for (size_t ell = 0; ell < nLine; ++ell)
    {
        const double lo = lineLumPerQGalaxy_[feHBracket.lo_, ell];
        const double hi = lineLumPerQGalaxy_[feHBracket.hi_, ell];
        lineLum[ell] = qhi * (lo + (feHBracket.frac_ * (hi - lo))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- ell < nLine == lineLum.size() by construction above
    }

    auto outSpec = stellarAboveEdge(spec);

    const size_t nWl = wl_.size();
    for (size_t k = 0; k < nWl; ++k)
    {
        const double lo = ctmLumPerQGalaxy_[feHBracket.lo_, k];
        const double hi = ctmLumPerQGalaxy_[feHBracket.hi_, k];
        outSpec[k] += qhi * (lo + (feHBracket.frac_ * (hi - lo))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- k < nWl == wl_.size() == outSpec.size() (stellarAboveEdge() sizes its own result to spec.size(), and spec is always on wl_'s own grid)
    }

    depositLines(lineLum, outSpec);

    return { std::move(outSpec), std::move(lineLum) };
}

auto nebular::Nebular::getCluster(const std::vector<double>& spec, const double feH,
    const double age) const -> std::pair<std::vector<double>, std::vector<double>>
{
    static constexpr auto context = "Nebular::getCluster";

    // No cloudy grid data exists for a cluster this old -- return the
    // stellar spectrum alone (still edge-zeroed, per getGalaxy()'s own
    // convention) with no nebular continuum or line emission added
    if (age > clusterAge_.back())
    {
        return { stellarAboveEdge(spec), std::vector<double>(lineWl_.size(), 0.0) };
    }

    const auto feHBracket = bracketGrid(feH_, feH, context, "[Fe/H]");
    // Ages below clusterAge_'s own minimum are pinned to that minimum
    // rather than treated as an error, unlike an out-of-range [Fe/H]
    const double ageClamped = std::max(age, clusterAge_.front());
    const auto ageBracket = bracketGrid(clusterAge_, ageClamped, context, "cluster age");

    // covFac_ discounts Q(HI) for ionizing photons lost to dust-grain
    // absorption or escape outside the observational aperture, before
    // any of it is attributed to nebular emission below
    const double qhi = simControls_.nebControls().covFac_ * qhiFilter_.phot(wl_, spec);

    const size_t nLine = lineWl_.size();
    std::vector<double> lineLum(nLine);
    for (size_t ell = 0; ell < nLine; ++ell)
    {
        const double loFloT = lineLumPerQCluster_[feHBracket.lo_, ageBracket.lo_, ell];
        const double loFhiT = lineLumPerQCluster_[feHBracket.lo_, ageBracket.hi_, ell];
        const double hiFloT = lineLumPerQCluster_[feHBracket.hi_, ageBracket.lo_, ell];
        const double hiFhiT = lineLumPerQCluster_[feHBracket.hi_, ageBracket.hi_, ell];
        const double loF = loFloT + (ageBracket.frac_ * (loFhiT - loFloT));
        const double hiF = hiFloT + (ageBracket.frac_ * (hiFhiT - hiFloT));
        lineLum[ell] = qhi * (loF + (feHBracket.frac_ * (hiF - loF))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- ell < nLine == lineLum.size() by construction above
    }

    auto outSpec = stellarAboveEdge(spec);

    const size_t nWl = wl_.size();
    for (size_t k = 0; k < nWl; ++k)
    {
        const double loFloT = ctmLumPerQCluster_[feHBracket.lo_, ageBracket.lo_, k];
        const double loFhiT = ctmLumPerQCluster_[feHBracket.lo_, ageBracket.hi_, k];
        const double hiFloT = ctmLumPerQCluster_[feHBracket.hi_, ageBracket.lo_, k];
        const double hiFhiT = ctmLumPerQCluster_[feHBracket.hi_, ageBracket.hi_, k];
        const double loF = loFloT + (ageBracket.frac_ * (loFhiT - loFloT));
        const double hiF = hiFloT + (ageBracket.frac_ * (hiFhiT - hiFloT));
        outSpec[k] += qhi * (loF + (feHBracket.frac_ * (hiF - loF))); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- k < nWl == wl_.size() == outSpec.size() (stellarAboveEdge() sizes its own result to spec.size(), and spec is always on wl_'s own grid)
    }

    depositLines(lineLum, outSpec);

    return { std::move(outSpec), std::move(lineLum) };
}

// NOLINTEND(misc-include-cleaner)
