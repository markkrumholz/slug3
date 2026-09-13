/**
 * @file YieldChannel.cpp
 * @author Mark Krumholz
 * @brief Implementation of YieldChannel
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "YieldChannel.hpp"
#include "../elem/IsotopeTable.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <tuple>
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
         * @brief RAII guard that disables HDF5's automatic error-stack
         *   printing for its lifetime, restoring the prior handler on
         *   destruction -- including when unwinding past it due to an
         *   exception, unlike a bare save/set/restore triplet
         */
        class H5ErrorSuppressor //NOLINT(llvm-prefer-static-over-anonymous-namespace)
        {
        public:
            H5ErrorSuppressor()
            {
                H5Eget_auto2(H5E_DEFAULT, &oldFunc_, &oldClientData_);
                H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
            }
            ~H5ErrorSuppressor() { H5Eset_auto2(H5E_DEFAULT, oldFunc_, oldClientData_); }
            H5ErrorSuppressor(const H5ErrorSuppressor&) = delete;
            H5ErrorSuppressor(H5ErrorSuppressor&&) = delete;
            auto operator=(const H5ErrorSuppressor&) -> H5ErrorSuppressor& = delete;
            auto operator=(H5ErrorSuppressor&&) -> H5ErrorSuppressor& = delete;

        private:
            H5E_auto2_t oldFunc_ = nullptr;
            void* oldClientData_ = nullptr;
        };

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
         *
         * HDF5's own automatic error-stack printing is disabled (via
         * H5ErrorSuppressor, above) around the loop below: every
         * non-group child (masses/isotope_z/isotope_a) is *expected*
         * to fail H5Gopen2, and this codebase has no use for HDF5's
         * own multi-line stderr diagnostic dump on each such harmless,
         * anticipated failure. The RAII guard, rather than a bare
         * save/set/restore triplet, ensures the prior handler is put
         * back even if utils::childNameByIdx or
         * utils::readScalarAttrIfPresent throws.
         */
        auto findBracketingFeH( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const hid_t grp, const double fehMin, const double fehMax,
            const std::string& channelName, const std::string& modelName)
            -> std::vector<std::pair<double, std::string>>
        {
            H5G_info_t ginfo{};
            H5Gget_info(grp, &ginfo);
            std::vector<std::pair<double, std::string>> available;

            {
                const H5ErrorSuppressor suppressor;
                for (hsize_t i = 0; i < ginfo.nlinks; ++i)
                {
                    const auto childName = utils::childNameByIdx(grp, i);
                    const hid_t childGrp = H5Gopen2(grp, childName.c_str(), H5P_DEFAULT);
                    if (childGrp < 0) { continue; }
                    std::optional<double> fehVal;
                    try
                    {
                        fehVal = utils::readScalarAttrIfPresent(childGrp, "Fe_H");
                    }
                    catch (...)
                    {
                        H5Gclose(childGrp);
                        throw;
                    }
                    H5Gclose(childGrp);
                    if (fehVal.has_value()) { available.emplace_back(*fehVal, childName); }
                }
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
         * @param niso Number of isotopes (isotopesOrig_.size())
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
                // fehGrp must be closed even if readDataset2D throws
                // (e.g. this group's own "yield" dataset is missing or
                // malformed) -- caught, closed, and rethrown here
                // rather than leaking it.
                std::vector<double> data;
                std::pair<std::size_t, std::size_t> shape;
                try
                {
                    std::tie(data, shape) = utils::readDataset2D(fehGrp, "yield", "YieldChannel");
                }
                catch (...)
                {
                    H5Gclose(fehGrp);
                    throw;
                }
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

        /** @brief mdspan view used for both yieldData_/yieldDataOrig_ in rebuildYieldGrid() */
        using MutableArray3D = std::mdspan<double, std::dextents<std::size_t, 3>>; // NOLINT(misc-include-cleaner)

        /**
         * @brief For each rebuildYieldGrid() target isotope, its index in isotopesOrig_, if any
         * @details
         * isoMap[j] is the index into isotopesOrig_ (and origView's own
         * third axis) that target isotope j reads from, or nullopt if
         * isotopesOrig_ has no matching isotope at all -- see
         * rebuildYieldGrid()'s own comment. extrapolateMassColumn()/
         * interpolateMassColumn() both take this instead of a plain
         * isotope count, so they can skip (leaving newView at its
         * already-zero-initialized default) any target isotope with no
         * source column to read.
         */
        using IsotopeMap = std::vector<std::optional<std::size_t>>; //NOLINT(llvm-prefer-static-over-anonymous-namespace)

        /**
         * @brief Fill one masses_ column of newView by extrapolating a massesOrig_ column
         * @param origView Yield data over massesOrig_/isotopesOrig_, shape (nfeh, massesOrig_.size(), isotopesOrig_.size())
         * @param newView Yield data over masses_/isotopes_, shape (nfeh, masses_.size(), isoMap.size()) -- column mi is written
         * @param nfeh feH_.size()
         * @param isoMap See its own comment
         * @param mi Index into masses_ (and newView's second axis) to fill
         * @param src Index into massesOrig_ (and origView's second axis) of the nearest native column
         * @param ratio Requested mass divided by massesOrig_[src] -- see rebuildYieldGrid()'s own comment
         */
        void extrapolateMassColumn( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const YieldChannel::Array3D& origView, const MutableArray3D& newView,
            const std::size_t nfeh, const IsotopeMap& isoMap,
            const std::size_t mi, const std::size_t src, const double ratio)
        {
            for (std::size_t f = 0; f < nfeh; ++f)
            {
                for (std::size_t j = 0; j < isoMap.size(); ++j)
                {
                    if (!isoMap[j].has_value()) { continue; } // no matching isotope in isotopesOrig_ -- leave at 0
                    newView[f, mi, j] = ratio * origView[f, src, *isoMap[j]]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f/mi/src/j/*isoMap[j] all in range by construction, see rebuildYieldGrid()'s own comment
                }
            }
        }

        /**
         * @brief Fill one masses_ column of newView by interpolating between two massesOrig_ columns
         * @param origView Yield data over massesOrig_/isotopesOrig_, shape (nfeh, massesOrig_.size(), isotopesOrig_.size())
         * @param newView Yield data over masses_/isotopes_, shape (nfeh, masses_.size(), isoMap.size()) -- column mi is written
         * @param nfeh feH_.size()
         * @param isoMap See its own comment
         * @param mi Index into masses_ (and newView's second axis) to fill
         * @param bracket Bracketing indices/weight (into massesOrig_) from utils::findBracket --
         *   weight 0 or 1 reduces this to an exact copy of one massesOrig_ column, see
         *   rebuildYieldGrid()'s own comment
         */
        void interpolateMassColumn( //NOLINT(llvm-prefer-static-over-anonymous-namespace)
            const YieldChannel::Array3D& origView, const MutableArray3D& newView,
            const std::size_t nfeh, const IsotopeMap& isoMap,
            const std::size_t mi, const utils::Bracket& bracket)
        {
            for (std::size_t f = 0; f < nfeh; ++f)
            {
                for (std::size_t j = 0; j < isoMap.size(); ++j)
                {
                    if (!isoMap[j].has_value()) { continue; } // no matching isotope in isotopesOrig_ -- leave at 0
                    newView[f, mi, j] = // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- f/mi/j/bracket/*isoMap[j] all in range by construction, see rebuildYieldGrid()'s own comment
                        (1.0 - bracket.t_) * origView[f, bracket.lo_, *isoMap[j]] +
                        bracket.t_ * origView[f, bracket.hi_, *isoMap[j]];
                }
            }
        }
    } // namespace

    YieldChannel::YieldChannel(
        const YieldChannelDescriptor& descriptor,
        const double fehMin,
        const double fehMax,
        const std::string& registryName) :
        channel_(descriptor.channel_)
    {
        const std::string& modelName = descriptor.modelName_;
        const std::string channelName(channelStr.at(static_cast<std::size_t>(descriptor.channel_)));

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
            massesOrig_ = utils::readDataset1D(grp, "masses", "YieldChannel");
            // Not just "sorted": an empty grid would make
            // massesOrig_.front()/back() (rebuildYieldGrid()'s own
            // default mMin/mMax, below) undefined behavior, and a
            // merely non-decreasing (as opposed to strictly increasing)
            // grid could give findBracket() two equal-valued neighbors,
            // dividing by zero when it computes t. adjacent_find with
            // greater_equal locates the first adjacent pair that
            // violates strict ascent, if any.
            if (massesOrig_.empty() ||
                std::ranges::adjacent_find(massesOrig_, std::ranges::greater_equal{}) != massesOrig_.end())
            {
                throw std::runtime_error(
                    "YieldChannel: masses dataset in group " + channelName +
                    " of " + h5Path.string() + " is empty or not strictly ascending");
            }

            const auto zData = utils::readDataset1D(grp, "isotope_z", "YieldChannel");
            const auto aData = utils::readDataset1D(grp, "isotope_a", "YieldChannel");
            if (zData.size() != aData.size())
            {
                throw std::runtime_error(
                    "YieldChannel: isotope_z and isotope_a in group " +
                    channelName + " of " + h5Path.string() + " have different lengths");
            }
            isotopesOrig_.clear();
            isotopesOrig_.reserve(zData.size());
            for (const auto& [z, a] : std::views::zip(zData, aData))
            {
                const auto zInt = static_cast<unsigned int>(z);
                const auto aInt = static_cast<unsigned int>(a);
                try
                {
                    isotopesOrig_.emplace_back(elem::isotopeTable(zInt, aInt));
                }
                catch (const std::out_of_range&)
                {
                    throw std::runtime_error(
                        "YieldChannel: isotope (Z=" + std::to_string(zInt) + ", A=" +
                        std::to_string(aInt) + ") in group " + channelName + " of " +
                        h5Path.string() + " is not in the isotope table");
                }
            }

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
            // Same risk, and same fix, as masses_'s own identical
            // check above -- two feh_<value> groups that happened to
            // share a "Fe_H" attribute value (a malformed/hand-edited
            // file; findBracketingFeH()'s own group-name-derived values
            // can't produce this from a normal import) would divide by
            // zero in findBracket() exactly the same way.
            if (std::ranges::adjacent_find(feH_, std::ranges::greater_equal{}) != feH_.end())
            {
                throw std::runtime_error(
                    "YieldChannel: [Fe/H] groups in group " + channelName +
                    " of " + h5Path.string() + " do not have strictly ascending Fe_H values");
            }

            // Step 5: read the yield data for every bracketed [Fe/H]
            // group -- see readYieldData()'s own comment
            yieldDataOrig_ = readYieldData(grp, groupNames, massesOrig_.size(), isotopesOrig_.size());
        }
        catch (...)
        {
            H5Gclose(grp);
            H5Fclose(file);
            throw;
        }

        H5Gclose(grp);
        H5Fclose(file);

        // Deliberately does not call rebuildYieldGrid() here -- see its
        // own comment, and the constructor's own, for why masses_/
        // isotopes_/yieldData_ are left empty until a caller does so
        // explicitly.
    }

    void YieldChannel::rebuildYieldGrid(
        const std::optional<double> mMin, const std::optional<double> mMax, IsotopeList isotopes)
    {
        const double loMass = mMin.value_or(massesOrig_.front());
        const double hiMass = mMax.value_or(massesOrig_.back());
        if (!std::isfinite(loMass) || !std::isfinite(hiMass) || loMass <= 0.0 || hiMass <= 0.0)
        {
            throw std::invalid_argument(
                "YieldChannel::rebuildYieldGrid: mMin (" + std::to_string(loMass) +
                ") and mMax (" + std::to_string(hiMass) +
                ") must both be finite and strictly positive");
        }
        if (!(loMass < hiMass))
        {
            throw std::invalid_argument(
                "YieldChannel::rebuildYieldGrid: mMin (" + std::to_string(loMass) +
                ") must be strictly less than mMax (" + std::to_string(hiMass) + ")");
        }

        masses_.clear();
        masses_.push_back(loMass);
        for (const double m : massesOrig_)
        {
            if (m > loMass && m < hiMass) { masses_.push_back(m); }
        }
        masses_.push_back(hiMass);

        // An empty isotopes argument means "use isotopesOrig_ itself,
        // unchanged" -- see this method's own comment. remapIsotopes
        // distinguishes that identity case (isoMap below is trivially
        // j -> j, with no search needed) from a genuine caller-supplied
        // list (isoMap is instead found by IsotopeData equality, since
        // the caller's own list can reorder, subset, or extend relative
        // to isotopesOrig_).
        const bool remapIsotopes = !isotopes.empty();
        IsotopeList newIsotopes = remapIsotopes ? std::move(isotopes) : isotopesOrig_;

        IsotopeMap isoMap(newIsotopes.size());
        for (std::size_t j = 0; j < newIsotopes.size(); ++j)
        {
            if (!remapIsotopes) { isoMap[j] = j; continue; }
            const auto it = std::ranges::find_if(isotopesOrig_,
                [&](const auto& orig) { return orig.get() == newIsotopes[j].get(); });
            if (it != isotopesOrig_.end())
            {
                isoMap[j] = static_cast<std::size_t>(std::distance(isotopesOrig_.begin(), it));
            }
            // else: this channel's own model never tabulated
            // newIsotopes[j] -- isoMap[j] stays nullopt, leaving that
            // isotope's own yieldData_ entries at 0 (see
            // extrapolateMassColumn()/interpolateMassColumn())
        }

        const std::size_t nfeh = feH_.size();
        const std::size_t nmassOrig = massesOrig_.size();
        const std::size_t nmass = masses_.size();
        const std::size_t nisoOrig = isotopesOrig_.size();
        const std::size_t niso = newIsotopes.size();
        const Array3D origView(yieldDataOrig_.data(), nfeh, nmassOrig, nisoOrig);

        yieldData_.assign(nfeh * nmass * niso, 0.0);
        const MutableArray3D newView(yieldData_.data(), nfeh, nmass, niso);

        // Extrapolation (mt outside massesOrig_'s own range) can't just
        // be utils::findBracket, unlike the interior case: it clamps
        // out-of-range queries to weight 0 or 1 at the nearest edge
        // rather than reporting them as out of range -- see
        // extrapolateMassColumn()/interpolateMassColumn()'s own
        // comments for the two cases this splits into.
        std::size_t cache = 0;
        for (std::size_t mi = 0; mi < nmass; ++mi)
        {
            const double mt = masses_[mi]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- mi < nmass == masses_.size() by construction
            if (mt < massesOrig_.front() || mt > massesOrig_.back())
            {
                const bool below = mt < massesOrig_.front();
                const std::size_t src = below ? 0 : nmassOrig - 1;
                const double ratio = mt / (below ? massesOrig_.front() : massesOrig_.back());
                extrapolateMassColumn(origView, newView, nfeh, isoMap, mi, src, ratio);
            }
            else
            {
                const auto bracket = utils::findBracket(massesOrig_, mt, cache);
                interpolateMassColumn(origView, newView, nfeh, isoMap, mi, bracket);
            }
        }

        isotopes_ = std::move(newIsotopes);

        // massCache_'s cached indices were only ever valid against the
        // old masses_; a stale one could now be at or past the end of
        // a smaller new masses_, which yield()'s own unchecked
        // utils::findBracket call requires never happens. feH_/
        // fehCache_ are untouched -- this method never changes the
        // [Fe/H] axis.
        std::ranges::fill(massCache_, 0);
    }

} // namespace yields

// NOLINTEND(misc-include-cleaner)
