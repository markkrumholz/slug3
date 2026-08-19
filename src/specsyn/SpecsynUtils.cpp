/**
 * @file SpecsynUtils.cpp
 * @author Mark Krumholz
 * @brief Implementations for SpecsynUtils.hpp
 * @date 2026-07-20
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SpecsynUtils.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/MiscUtils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "SpecsynCommons.hpp"
#include "hdf5.h"   // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <filesystem>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace specsyn
{
    auto parseRegistry(const std::string& registryName)
    -> std::pair<toml::table, std::filesystem::path>
    {
        // requireFeH=false: see this function's own header comment for why
        [[maybe_unused]] auto [registry, registryPath, spectraSets] =
            utils::parseSetRegistry(registryName, "spectra_sets", "spectra", "parseRegistry", false);
        return { std::move(registry), std::move(registryPath) };
    }

    auto findMatchingSpectra( // NOLINT(readability-function-cognitive-complexity)
        const std::string& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const double microTurb,
        const double r,
        const std::string& registryName)
    -> std::pair<std::vector<double>, std::vector<std::string>>
    {
        // First parse the registry
        auto [registry, registryPath] = parseRegistry(registryName);

        // Now check the registry for spectra matching the given spectra name
        auto spectraSets = utils::getStringArrayField(registry, "spectra_sets");
        auto it = std::ranges::find(spectraSets.begin(), spectraSets.end(), spectraName);
        if (it == spectraSets.end())
        {
            throw std::runtime_error("findMatchingSpectra: no spectra set named " +
                spectraName + " found in spectra registry " + registryName);
        }

        // Get the h5file name for this spectra set from the registry
        const auto h5name =
            registry.at_path(spectraName).at_path("file").value_or(std::string{});

        // The h5 file name is given relative to the directory
        // containing the registry file itself
        const auto h5path = registryPath.parent_path() / h5name;

        // Suppress clang-tidy warnings iun this namespace caused by just including
        // hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
        // that HDF5 wants
        // NOLINTBEGIN(misc-include-cleaner)

        // Open the HDF5 file
        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error("findMatchingSpectra: unable to open "
                "HDF5 file " + h5path.string());
        }

        // Find the number of top-level links (groups) in the file
        H5G_info_t ginfo{};
        H5Gget_info(file, &ginfo);

        // Loop over every group in the file, collecting the feh value
        // and name of each one whose afe, cfe, micro, and r attributes
        // all match the input values; a group missing one of these
        // attributes is treated as matching regardless of the
        // corresponding input value. Groups with no feh attribute at
        // all (e.g. "wavelengths", "logg_Teff_grid") are simply
        // skipped. The feh range is deliberately not applied here --
        // every afe/cfe/micro/r match is collected, and the feh range
        // is instead used below to select a bracketing subset of them
        // (see this function's own doc comment for why).
        std::vector<std::pair<double, std::string>> matches;
        for (hsize_t i = 0; i < ginfo.nlinks; ++i)
        {
            const auto nameLen = H5Lget_name_by_idx(file, ".",
                H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
            if (nameLen < 0) { continue; }
            std::vector<char> nameBuf(static_cast<size_t>(nameLen) + 1);
            H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                nameBuf.data(), nameBuf.size(), H5P_DEFAULT);
            const std::string groupName(nameBuf.data());

            const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
            if (grp < 0) { continue; }

            const auto fehVal = utils::readScalarAttrIfPresent(grp, "feh");
            bool isMatch = fehVal.has_value();

            if (isMatch)
            {
                const auto afeVal = utils::readScalarAttrIfPresent(grp, "afe");
                if (afeVal && !utils::approxEqual(*afeVal, afe))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto cfeVal = utils::readScalarAttrIfPresent(grp, "cfe");
                if (cfeVal && !utils::approxEqual(*cfeVal, cfe))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto microVal = utils::readScalarAttrIfPresent(grp, "micro");
                if (microVal && !utils::approxEqual(*microVal, microTurb))
                {
                    isMatch = false;
                }
            }
            if (isMatch)
            {
                const auto rVal = utils::readScalarAttrIfPresent(grp, "r");
                if (rVal && !utils::approxEqual(*rVal, r))
                {
                    isMatch = false;
                }
            }

            if (isMatch) { matches.emplace_back(*fehVal, groupName); }

            H5Gclose(grp);
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        // Sort all the afe/cfe/micro/r-matching spectra by feh, from
        // lowest to highest
        std::ranges::sort(matches, {},
            &std::pair<double, std::string>::first);

        if (matches.empty())
        {
            return { {}, {} };
        }

        // Find the minimal bracketing range of indices into matches
        // whose feh values encompass [fehMin, fehMax]: from the last
        // index with feh <= fehMin (or index 0, if no feh is that low),
        // through the first index with feh >= fehMax (or the last
        // index, if no feh is that high) -- identical in purpose to
        // TrackUtils::findMatchingTracks's own bracketing logic (see
        // its own comment). std::ranges::upper_bound and lower_bound
        // trip up misc-include-cleaner on some libc++ versions (it
        // can't find a header to attribute them to, even with
        // <algorithm> already included), hence the NOLINTs below.
        const auto loBound = std::ranges::upper_bound(matches, fehMin, //NOLINT(misc-include-cleaner)
            {}, &std::pair<double, std::string>::first);
        size_t loIdx = (loBound == matches.begin()) ? 0 :
            static_cast<size_t>(loBound - matches.begin()) - 1;
        const auto hiBound = std::ranges::lower_bound(matches, fehMax, //NOLINT(misc-include-cleaner)
            {}, &std::pair<double, std::string>::first);
        size_t hiIdx = (hiBound == matches.end()) ?
            matches.size() - 1 : static_cast<size_t>(hiBound - matches.begin());

        // Unlike a track set's groups, a spectral library's groups are
        // not guaranteed to have distinct feh values -- nothing here
        // assumes a 1:1 (feh, group) correspondence, so a caller whose
        // registry entry happens to have two groups share a feh (see
        // tests/specsyn/assets/TIEDFEH_test.h5, built purely to
        // exercise this) still gets correct behavior. loIdx/hiIdx
        // above each land on just one
        // arbitrary member of such a tied run (std::ranges::sort is
        // not required to be stable, so even which member is
        // unspecified) whenever fehMin or fehMax falls at or beyond
        // that run, so expand each outward to cover every group tied
        // at that same feh value: loIdx back to the first index
        // sharing matches[loIdx]'s feh, hiIdx forward to the last
        // index sharing matches[hiIdx]'s feh. This also handles the
        // case where fehMin == fehMax lands exactly on a tied run --
        // loIdx and hiIdx there start out inverted (loIdx, the last
        // index <= fehMin, past hiIdx, the first index >= fehMax, both
        // within that same run), but expanding each to the full run
        // brings loIdx back to its start and hiIdx to its end, so the
        // two never need reconciling against each other directly.
        while (loIdx > 0 && matches[loIdx - 1].first == matches[loIdx].first) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- loIdx > 0 checked in the loop condition, and loIdx < matches.size() throughout since it only ever decreases from an initially valid index
        {
            --loIdx;
        }
        while (hiIdx + 1 < matches.size() && matches[hiIdx + 1].first == matches[hiIdx].first) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- hiIdx + 1 < matches.size() checked in the loop condition, and hiIdx stays < matches.size() throughout since it only ever increases from an initially valid index
        {
            ++hiIdx;
        }

        // Extract the bracketing range of matches into the output vectors
        std::vector<double> fehOut;
        std::vector<std::string> nameOut;
        const size_t nOut = hiIdx - loIdx + 1;
        fehOut.reserve(nOut);
        nameOut.reserve(nOut);
        for (size_t idx = loIdx; idx <= hiIdx; ++idx)
        {
            auto& match = matches.at(idx);
            fehOut.push_back(match.first);
            nameOut.push_back(std::move(match.second));
        }

        return { std::move(fehOut), std::move(nameOut) };
    }

    auto getMicroDefault(
        const std::string& spectraName, const std::string& registryName) -> double
    {
        auto registry = parseRegistry(registryName).first;

        auto spectraSets = utils::getStringArrayField(registry, "spectra_sets");
        if (std::ranges::find(spectraSets, spectraName) == spectraSets.end())
        {
            throw std::runtime_error("getMicroDefault: no spectra set named " +
                spectraName + " found in spectra registry " + registryName);
        }

        // A missing micro_default means this library has no
        // microturbulence axis at all (e.g. CK04), rather than a
        // caller error -- unlike the missing-spectraName case above,
        // which is always a mistake. defaultMicroTurb is a harmless
        // placeholder in that case: a library with no microturbulence
        // axis also has no "micro" attribute on any of its groups, so
        // findMatchingSpectra ignores this value entirely rather than
        // ever actually filtering on it.
        return registry.at_path(spectraName).at_path("micro_default")
            .value<double>().value_or(defaultMicroTurb);
    }

    namespace
    {
        // Returns a group's afe attribute if it has both feh and afe
        // attributes and its cfe/micro/r attributes (where present)
        // match the input constraints; nullopt otherwise. Factored out
        // of findAfeValues to keep that function's cognitive complexity
        // down.
        auto groupAfeIfMatching( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const hid_t grp, const double cfe, const double microTurb, const double r)
            -> std::optional<double>
        {
            const auto fehVal = utils::readScalarAttrIfPresent(grp, "feh");
            const auto afeVal = utils::readScalarAttrIfPresent(grp, "afe");
            if (!fehVal || !afeVal)
            {
                return std::nullopt;
            }

            const auto cfeVal = utils::readScalarAttrIfPresent(grp, "cfe");
            if (cfeVal && !utils::approxEqual(*cfeVal, cfe))
            {
                return std::nullopt;
            }
            const auto microVal = utils::readScalarAttrIfPresent(grp, "micro");
            if (microVal && !utils::approxEqual(*microVal, microTurb))
            {
                return std::nullopt;
            }
            const auto rVal = utils::readScalarAttrIfPresent(grp, "r");
            if (rVal && !utils::approxEqual(*rVal, r))
            {
                return std::nullopt;
            }

            return afeVal;
        }
    } // namespace

    auto findAfeValues(
        const std::string& spectraName,
        const double cfe,
        const double microTurb,
        const double r,
        const std::string& registryName) -> std::vector<double>
    {
        // Parse registry and open the HDF5 file
        auto [registry, registryPath] = parseRegistry(registryName);

        auto spectraSets = utils::getStringArrayField(registry, "spectra_sets");
        if (std::ranges::find(spectraSets, spectraName) == spectraSets.end())
        {
            throw std::runtime_error("findAfeValues: no spectra set named " +
                spectraName + " found in spectra registry " + registryName);
        }

        const auto h5name =
            registry.at_path(spectraName).at_path("file").value_or(std::string{});
        const auto h5path = registryPath.parent_path() / h5name;

        // NOLINTBEGIN(misc-include-cleaner)

        const hid_t file = H5Fopen(h5path.string().c_str(),
            H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error("findAfeValues: unable to open "
                "HDF5 file " + h5path.string());
        }

        H5G_info_t ginfo{};
        H5Gget_info(file, &ginfo);

        // Collect all distinct afe values from groups that have an afe
        // attribute and also match the cfe, micro, and r constraints.
        // Groups with no feh attribute (non-spectra groups like
        // "wavelengths") are skipped.
        std::vector<double> afeVals;
        for (hsize_t i = 0; i < ginfo.nlinks; ++i)
        {
            const auto nameLen = H5Lget_name_by_idx(file, ".",
                H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
            if (nameLen < 0) { continue; }
            std::vector<char> nameBuf(static_cast<size_t>(nameLen) + 1);
            H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                nameBuf.data(), nameBuf.size(), H5P_DEFAULT);

            const hid_t grp = H5Gopen2(file, nameBuf.data(), H5P_DEFAULT);
            if (grp < 0) { continue; }

            const auto afeVal = groupAfeIfMatching(grp, cfe, microTurb, r);
            H5Gclose(grp);
            if (afeVal && std::ranges::find_if(afeVals, [a = *afeVal](const double v) {
                    return utils::approxEqual(v, a);
                }) == afeVals.end())
            {
                afeVals.push_back(*afeVal);
            }
        }

        H5Fclose(file);

        // NOLINTEND(misc-include-cleaner)

        std::ranges::sort(afeVals);
        return afeVals;
    }

} // namespace specsyn
