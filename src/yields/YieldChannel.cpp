/**
 * @file YieldChannel.cpp
 * @author Mark Krumholz
 * @brief Implementation of YieldChannel
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "YieldChannel.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <ranges>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

// Suppress clang-tidy warnings in this namespace caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the
// paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

namespace yields
{
    namespace
    {
        /**
         * @brief Scan, validate, and bracket the [Fe/H] groups in one channel's group
         * @param grp Handle to the channel's own already-open group
         * @param fehMin Minimum [Fe/H] value requested
         * @param fehMax Maximum [Fe/H] value requested
         * @param channelName Name of the channel (for error messages only)
         * @param modelName Name of the model (for error messages only)
         * @returns The minimal bracketing subset of available [Fe/H]
         *   values and their group names, ascending -- see
         *   YieldChannel::YieldChannel()'s own comment on this being
         *   the "largest available <= fehMin through smallest
         *   available >= fehMax" rule
         * @throws std::runtime_error if grp has no [Fe/H] groups at
         *   all, or if [fehMin, fehMax] extends outside the range
         *   those groups actually cover
         * @details
         * Scans every direct child of grp, collecting its own "Fe_H"
         * attribute and name -- mirroring findMatchingTracks()'s own
         * "scan the actual HDF5 groups by attribute" approach (the
         * authoritative source of which [Fe/H] values actually exist),
         * rather than trusting yields.toml's own Fe_H list, which --
         * like tracks.toml's -- is informational only and not itself
         * read here. A child that fails to open as a group (masses/
         * isotope_z/isotope_a, which are datasets, not groups, and so
         * cannot be H5Gopen2'd) is silently skipped, the same way
         * findMatchingTracks() skips "masses" by name; skipping by
         * failed-open here instead requires no hardcoded list of the
         * non-[Fe/H]-group dataset names to keep in sync as this
         * format grows. Leaves grp itself open either way -- the
         * caller owns closing it (and the file containing it).
         */
        auto findBracketingFeH( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const hid_t grp, const double fehMin, const double fehMax,
            const std::string& channelName, const std::string& modelName)
            -> std::vector<std::pair<double, std::string>>
        {
            H5G_info_t ginfo{};
            H5Gget_info(grp, &ginfo);
            std::vector<std::pair<double, std::string>> available;
            for (hsize_t i = 0; i < ginfo.nlinks; ++i)
            {
                const auto childName = utils::childNameByIdx(grp, i);
                const hid_t childGrp = H5Gopen2(grp, childName.c_str(), H5P_DEFAULT);
                if (childGrp < 0) { continue; }
                const auto fehVal = utils::readScalarAttrIfPresent(childGrp, "Fe_H");
                H5Gclose(childGrp);
                if (fehVal.has_value()) { available.emplace_back(*fehVal, childName); }
            }
            std::ranges::sort(available);

            if (available.empty())
            {
                throw std::runtime_error(
                    "YieldChannel: no [Fe/H] groups found for channel '" +
                    channelName + "', model '" + modelName + "'");
            }

            // Verify [fehMin, fehMax] can actually be accommodated --
            // thrown here, rather than allowing the bracketing search
            // below to silently narrow to whichever available value is
            // closest, exactly mirroring Tracks3D::Tracks3D()'s own
            // identical up-front range check (see its own comment).
            const double availFehMin = available.front().first;
            const double availFehMax = available.back().first;
            if (fehMin < availFehMin || fehMax > availFehMax)
            {
                throw std::runtime_error(
                    "YieldChannel: requested [Fe/H] range [" +
                    std::to_string(fehMin) + ", " + std::to_string(fehMax) +
                    "] extends outside the range available for channel '" +
                    channelName + "', model '" + modelName + "' ([" +
                    std::to_string(availFehMin) + ", " +
                    std::to_string(availFehMax) + "])");
            }

            // Find the minimal bracketing subset -- the largest
            // available value <= fehMin through the smallest available
            // value >= fehMax, inclusive -- so the loaded grid always
            // fully encloses [fehMin, fehMax] even when neither bound
            // falls exactly on a grid point. available is sorted
            // ascending (just above), so each loop can stop at the
            // first element that no longer satisfies its own condition.
            std::size_t loIdx = 0;
            for (std::size_t i = 0; i < available.size(); ++i)
            {
                if (available[i].first > fehMin) { break; }
                loIdx = i;
            }
            std::size_t hiIdx = available.size() - 1;
            for (std::size_t i = available.size(); i-- > 0; )
            {
                if (available[i].first < fehMax) { break; }
                hiIdx = i;
            }

            return { available.begin() + static_cast<std::ptrdiff_t>(loIdx),
                available.begin() + static_cast<std::ptrdiff_t>(hiIdx) + 1 };
        }

        /**
         * @brief Read and transpose the yield data for every bracketed [Fe/H] group
         * @param grp Handle to the channel's own already-open group
         * @param groupNames Names of the bracketed [Fe/H] groups, ascending
         * @param nmass Number of masses (masses_.size())
         * @param niso Number of isotopes (yldZ_.size())
         * @returns The flat backing storage for a (groupNames.size(),
         *   nmass, niso) mdspan (see YieldChannel::yld()'s own comment)
         * @throws std::runtime_error if a group cannot be opened, or
         *   its own "yield" dataset is not shaped (niso, nmass)
         * @details
         * Each group's own "yield" dataset is shaped (n_isotopes,
         * n_masses) on disk (see import_yield_tables.py's own comment
         * on the HDF5 format); the returned mdspan is (n_feh, n_masses,
         * n_isotopes), so this transposes the last two axes while
         * copying. Leaves grp itself open either way -- the caller owns
         * closing it (and the file containing it).
         */
        auto readYieldData( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const hid_t grp, const std::vector<std::string>& groupNames,
            const std::size_t nmass, const std::size_t niso) -> std::vector<double>
        {
            const std::size_t nfeh = groupNames.size();
            std::vector<double> yieldData(nfeh * nmass * niso, 0.0);
            const std::mdspan<double, std::dextents<std::size_t, 3>> yldView(
                yieldData.data(), nfeh, nmass, niso);

            for (std::size_t f = 0; f < nfeh; ++f)
            {
                const hid_t fehGrp = H5Gopen2(grp, groupNames[f].c_str(), H5P_DEFAULT);
                if (fehGrp < 0)
                {
                    throw std::runtime_error(
                        "YieldChannel: unable to open group " + groupNames[f]);
                }
                auto [data, shape] = utils::readDataset2D(fehGrp, "yield", "YieldChannel");
                H5Gclose(fehGrp);
                const auto [nrow, ncol] = shape;
                if (nrow != niso || ncol != nmass)
                {
                    throw std::runtime_error(
                        "YieldChannel: yield dataset in group " + groupNames[f] +
                        " has shape (" + std::to_string(nrow) + ", " +
                        std::to_string(ncol) + "), expected (" +
                        std::to_string(niso) + ", " + std::to_string(nmass) + ")");
                }
                const std::mdspan<const double, std::dextents<std::size_t, 2>> src(
                    data.data(), nrow, ncol);
                for (std::size_t m = 0; m < nmass; ++m)
                {
                    for (std::size_t iso = 0; iso < niso; ++iso)
                    {
                        yldView[f, m, iso] = src[iso, m];
                    }
                }
            }

            return yieldData;
        }
    } // namespace

    YieldChannel::YieldChannel(
        const Channel channel,
        const std::string& modelName,
        const double fehMin,
        const double fehMax,
        const std::string& registryName) :
        channel_(channel)
    {
        const std::string channelName(channelStr.at(static_cast<std::size_t>(channel)));

        // Step 1: locate and validate the registry entry for this
        // channel/model, and resolve the HDF5 file it names. Mirrors
        // Extinct::loadCurveImpl()'s own registry-navigation pattern,
        // just one level deeper (channel, then model within it) to
        // match yields.toml's own [channel] -> models -> [channel.model]
        // structure (see import_yield_tables.py's own comment).
        auto [registry, registryPath] = utils::parseTOMLFile(registryName, "YieldChannel");

        const auto* models = registry.at_path(channelName).at_path("models").as_array();
        const auto modelNames = utils::stringArrayContents(models);
        if (std::ranges::find(modelNames, modelName) == modelNames.end())
        {
            throw std::runtime_error(
                "YieldChannel: registry " + registryPath.string() +
                " has no model '" + modelName + "' for channel '" +
                channelName + "'");
        }

        const auto modelPath = channelName + "." + modelName;
        const auto h5Name = registry.at_path(modelPath).at_path("file").value<std::string>();
        if (!h5Name.has_value())
        {
            throw std::runtime_error(
                "YieldChannel: registry " + registryPath.string() +
                ", entry for " + modelPath + " is missing required 'file' field");
        }
        const auto h5Path = registryPath.parent_path() / h5Name.value();

        // Step 2: open the HDF5 file and the channel's own group within it
        const hid_t file = H5Fopen(h5Path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            throw std::runtime_error(
                "YieldChannel: unable to open HDF5 file " + h5Path.string());
        }
        const hid_t grp = H5Gopen2(file, channelName.c_str(), H5P_DEFAULT);
        if (grp < 0)
        {
            H5Fclose(file);
            throw std::runtime_error(
                "YieldChannel: unable to open group " + channelName +
                " in HDF5 file " + h5Path.string());
        }

        // Everything from here on can throw partway through, after grp
        // (and file) are already open -- caught once here, closing
        // both before rethrowing, rather than repeating that pair of
        // closes at every individual throw site the way
        // findBracketingFeH()/readYieldData() would otherwise each
        // need their own copy of.
        try
        {
            // Step 3: read masses/isotope_z/isotope_a, shared across
            // every [Fe/H] this channel's group holds
            masses_ = utils::readDataset1D(grp, "masses", "YieldChannel");
            if (!std::ranges::is_sorted(masses_))
            {
                throw std::runtime_error(
                    "YieldChannel: masses dataset in group " + channelName +
                    " of " + h5Path.string() + " is not sorted ascending");
            }

            const auto zData = utils::readDataset1D(grp, "isotope_z", "YieldChannel");
            const auto aData = utils::readDataset1D(grp, "isotope_a", "YieldChannel");
            if (zData.size() != aData.size())
            {
                throw std::runtime_error(
                    "YieldChannel: isotope_z and isotope_a in group " +
                    channelName + " of " + h5Path.string() + " have different lengths");
            }
            yldZ_.resize(zData.size());
            std::ranges::transform(zData, yldZ_.begin(),
                [](const double z) { return static_cast<unsigned int>(z); });
            yldA_.resize(aData.size());
            std::ranges::transform(aData, yldA_.begin(),
                [](const double a) { return static_cast<unsigned int>(a); });

            // Step 4: find the bracketing [Fe/H] subset -- see
            // findBracketingFeH()'s own comment
            const auto bracket = findBracketingFeH(grp, fehMin, fehMax, channelName, modelName);
            feH_.clear();
            std::vector<std::string> groupNames;
            for (const auto& [feh, name] : bracket)
            {
                feH_.push_back(feh);
                groupNames.push_back(name);
            }

            // Step 5: read the yield data for every bracketed [Fe/H]
            // group -- see readYieldData()'s own comment
            yieldData_ = readYieldData(grp, groupNames, masses_.size(), yldZ_.size());
        }
        catch (...)
        {
            H5Gclose(grp);
            H5Fclose(file);
            throw;
        }

        H5Gclose(grp);
        H5Fclose(file);
    }

} // namespace yields

// NOLINTEND(misc-include-cleaner)
