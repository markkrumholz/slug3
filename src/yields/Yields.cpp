/**
 * @file Yields.cpp
 * @author Mark Krumholz
 * @brief Implementation of Yields
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Yields.hpp"
#include "../elem/DecayChain.hpp"
#include "../elem/ElemCommons.hpp"
#include "../elem/IsotopeTable.hpp"
#include "../io/SimControls.hpp"
#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yields
{
    namespace
    {
        /**
         * @brief Force-expand an isotope list to include every decay-chain descendant of an unstable entry
         * @param isotopes The list to expand, in place -- see
         *   Yields::rebuildYieldGrid()'s own comment for why
         * @throws std::runtime_error if some unstable isotope's own
         *   daughters() names a (Z, A) pair not found in the global
         *   elem::isotopeTable()
         * @details
         * Repeatedly scans for any entry not yet present that some
         * already-present unstable isotope's own daughters() names,
         * appending it, until a full pass adds nothing further -- a
         * newly-added isotope can itself be unstable, with daughters()
         * of its own, so this cannot stop after a single pass (e.g.
         * Ni56 -> Co56 -> Fe56, discovering Co56 from Ni56's own
         * daughters() only reveals Fe56 -- if not already present -- on
         * the *next* pass, from Co56's).
         */
        void forceExpandDecayChain(elem::IsotopeList& isotopes)
        {
            bool addedAny = true;
            while (addedAny)
            {
                addedAny = false;
                // Iterate over a snapshot of isotopes' own current
                // contents: appending to isotopes itself while iterating
                // over it directly would invalidate the very iterators
                // driving the loop.
                const elem::IsotopeList current = isotopes;
                for (const auto& iso : current)
                {
                    if (iso.get().stable()) { continue; }
                    for (const auto& daughter : iso.get().daughters())
                    {
                        const bool present = std::ranges::any_of(isotopes,
                            [&daughter](const auto& candidate)
                            {
                                return candidate.get().Z() == daughter.Z_ &&
                                    candidate.get().A() == daughter.A_;
                            });
                        if (!present)
                        {
                            try
                            {
                                isotopes.emplace_back(elem::isotopeTable(daughter.Z_, daughter.A_));
                            }
                            catch (const std::out_of_range&)
                            {
                                throw std::runtime_error(
                                    "Yields::rebuildYieldGrid: " + iso.get().label() +
                                    "'s daughter (Z=" + std::to_string(daughter.Z_) + ", A=" +
                                    std::to_string(daughter.A_) + ") is not in the isotope table");
                            }
                            addedAny = true;
                        }
                    }
                }
            }
        }
    } // namespace

    Yields::Yields(const io::SimControls& controls, std::string registryName) :
        controls_(controls),
        registryName_(std::move(registryName))
    {
        // Warn (not throw) if the same channel was requested more than
        // once -- a suspicious pattern that's easy to hit by mistake
        // (e.g. a copy-pasted yields.channelN table with the "model"
        // key never updated), but not forbidden outright: there are
        // legitimate reasons for it too, e.g. comparing two different
        // yield tables for the same channel, or covering one channel's
        // own mass range with two different models over two different
        // sub-ranges.
        std::array<unsigned int, static_cast<std::size_t>(Channel::nChannel_)> channelCounts{};
        for (const auto& descriptor : controls_.yieldChannels())
        {
            ++channelCounts.at(static_cast<std::size_t>(descriptor.channel_));
        }
        for (std::size_t i = 0; i < channelCounts.size(); ++i)
        {
            if (channelCounts.at(i) > 1)
            {
                std::cout << "slug: warning: " << channelCounts.at(i) <<
                    " yield channels requested for channel '" << channelStr.at(i) <<
                    "'; this is allowed (e.g. to compare yield tables, or to cover "
                    "different mass ranges with different models) but is also an easy "
                    "mistake to make by accident -- make sure this is intentional\n";
            }
        }

        for (const auto& descriptor : controls_.yieldChannels())
        {
            addChannel(descriptor);
        }
        rebuildYieldGrid();
    }

    void Yields::addChannel(const YieldChannelDescriptor& descriptor)
    {
        yieldChannels_.push_back(std::make_shared<YieldChannel>(
            descriptor, controls_.fehDist().getMin(), controls_.fehDist().getMax(), registryName_));
    }

    void Yields::addChannel(std::unique_ptr<YieldChannel> channel)
    {
        if (!channel)
        {
            throw std::invalid_argument("Yields::addChannel: channel must not be null");
        }
        yieldChannels_.push_back(std::shared_ptr<YieldChannel>(std::move(channel)));
    }

    void Yields::deleteChannel(const std::size_t index)
    {
        if (index >= yieldChannels_.size())
        {
            throw std::out_of_range(
                "Yields::deleteChannel: index " + std::to_string(index) +
                " is out of range for yieldChannels() of size " +
                std::to_string(yieldChannels_.size()));
        }
        yieldChannels_.erase(yieldChannels_.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void Yields::setChannels(std::vector<std::unique_ptr<YieldChannel>> channels)
    {
        if (std::ranges::any_of(channels, [](const auto& channel) { return !channel; }))
        {
            throw std::invalid_argument("Yields::setChannels: channels must not contain any null entries");
        }
        std::vector<std::shared_ptr<YieldChannel>> newChannels;
        newChannels.reserve(channels.size());
        for (auto& channel : channels)
        {
            newChannels.push_back(std::shared_ptr<YieldChannel>(std::move(channel)));
        }
        yieldChannels_ = std::move(newChannels);
    }

    void Yields::setChannels(const std::vector<YieldChannelDescriptor>& descriptors)
    {
        // Built into a temporary vector, not straight into
        // yieldChannels_ via addChannel(), so that a descriptor partway
        // through that fails to load (e.g. an unknown model, or a
        // [Fe/H] range mismatch -- see the YieldChannel constructor's
        // own comment) leaves yieldChannels_ completely untouched,
        // rather than a half-built mix of some new channels and none
        // of the old ones.
        std::vector<std::shared_ptr<YieldChannel>> newChannels;
        newChannels.reserve(descriptors.size());
        for (const auto& descriptor : descriptors)
        {
            newChannels.push_back(std::make_shared<YieldChannel>(
                descriptor, controls_.fehDist().getMin(), controls_.fehDist().getMax(), registryName_));
        }
        yieldChannels_ = std::move(newChannels);
    }

    void Yields::rebuildYieldGrid(const elem::IsotopeList& isotopes)
    {
        // Union every loaded channel's own isotopesOrig() into one
        // deduplicated, sorted isotopes_ -- see this method's own
        // comment. IsotopeData's own operator</operator== (Z first,
        // then A) is reused here via the referenced object (.get()),
        // rather than reference_wrapper's own identity-based
        // comparison, so two channels that both tabulate the same
        // isotope (necessarily the very same elem::isotopeTable()
        // entry, since that table is a single, global, per-(Z,A) map)
        // are correctly recognized as one isotope, not kept as
        // duplicates.
        isotopes_.clear();
        for (const auto& channel : yieldChannels_)
        {
            isotopes_.insert(
                isotopes_.end(), channel->isotopesOrig().begin(), channel->isotopesOrig().end());
        }
        std::ranges::sort(isotopes_,
            [](const auto lhs, const auto rhs) { return lhs.get() < rhs.get(); }); // NOLINT(performance-unnecessary-value-param) -- by-value (not const&) deliberately, to sidestep a clang-analyzer false positive (cplusplus.Move) that otherwise fires on this comparator's parameters during std::sort's internal element swaps; reference_wrapper is pointer-sized, so this has no real cost
        const auto dup = std::ranges::unique(isotopes_,
            [](const auto& lhs, const auto& rhs) { return lhs.get() == rhs.get(); });
        isotopes_.erase(dup.begin(), dup.end());

        // Force-expand to decay-chain closure (see this method's own
        // comment), then re-sort/re-dedup the same way as above.
        forceExpandDecayChain(isotopes_);
        std::ranges::sort(isotopes_,
            [](const auto lhs, const auto rhs) { return lhs.get() < rhs.get(); }); // NOLINT(performance-unnecessary-value-param) -- see above
        const auto dup2 = std::ranges::unique(isotopes_,
            [](const auto& lhs, const auto& rhs) { return lhs.get() == rhs.get(); });
        isotopes_.erase(dup2.begin(), dup2.end());

        // If the caller passed a non-empty isotopes list, restrict
        // isotopes_ down to its intersection with that list -- an
        // empty isotopes (the default) leaves every channel's own
        // isotope free to appear, exactly as before this parameter
        // existed.
        if (!isotopes.empty())
        {
            const auto toDrop = std::ranges::remove_if(isotopes_,
                [&isotopes](const auto& candidate)
                {
                    return std::ranges::none_of(isotopes,
                        [&candidate](const auto& wanted) { return wanted.get() == candidate.get(); });
                });
            isotopes_.erase(toDrop.begin(), toDrop.end());

            // Every yieldChannels_ entry's own isotopesOrig() is always
            // non-empty in practice (this Yields is only ever
            // constructed, by SimControls::readYields(), once at least
            // one real yield channel/model has been loaded -- see this
            // class's own constructor), so an empty isotopes_ here can
            // only mean the caller's own isotopes list -- entirely, not
            // just partially, as testSimControlsYieldsIsotopesKeyword()'s
            // own "C12" case in testSimControls.cpp covers -- fails to
            // match anything any loaded channel tabulates: almost
            // certainly a mistake (e.g. every entry misspelled, or
            // naming isotopes from a channel that was never actually
            // requested). Caught here, before it can reach
            // OutputManagerH5::openClusterYieldsGroup()/
            // openGalaxyYieldsGroup(), which would otherwise try to
            // create a zero-column "yields" dataset -- H5Pset_chunk()
            // rejects a zero chunk dimension, so that would instead
            // surface as an opaque "unable to create dataset" failure
            // far from the actual cause.
            if (isotopes_.empty())
            {
                throw std::runtime_error(
                    "Yields::rebuildYieldGrid: the given isotopes list does not "
                    "intersect any isotope tabulated by any loaded channel");
            }
        }

        // Push isotopes_ (and each channel's own descriptor mMin_/mMax_)
        // back down into every channel, synchronizing them all onto the
        // same isotope list -- see this method's own comment. Each
        // channel's own descriptor() already caches its own mMin_/
        // mMax_, so this needs no separate, parallel record of its own
        // to stay in sync with yieldChannels_.
        for (const auto& channel : yieldChannels_)
        {
            const auto descriptor = channel->descriptor();
            channel->rebuildYieldGrid(descriptor.mMin_, descriptor.mMax_, isotopes_);
        }

        // Rebuild decayChain_ to match the now-final isotopes_ -- see
        // this method's own comment.
        decayChain_.emplace(isotopes_);
    }

    auto Yields::yield(const double mass, const double feH, const double dtDecay) const
        -> std::pair<Array2D, std::vector<double>>
    {
        const std::size_t nchannels = yieldChannels_.size();
        const std::size_t niso = isotopes_.size();
        std::vector<double> data(nchannels * niso, 0.0);
        for (std::size_t i = 0; i < nchannels; ++i)
        {
            if (!yieldChannels_[i]->hasYield(mass)) { continue; } // mass outside this channel's own range -- leave its row at 0, see this method's own comment // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels == yieldChannels_.size() by construction
            const auto& channelFeH = yieldChannels_[i]->feH(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
            if (feH < channelFeH.front() || feH > channelFeH.back()) { continue; } // feH outside this channel's own range -- leave its row at 0, same as an out-of-range mass above
            const auto row = yieldChannels_[i]->yield(mass, feH); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels == yieldChannels_.size() by construction
            assert(row.size() == niso); // guaranteed once rebuildYieldGrid() has synchronized every channel onto isotopes_ -- see this method's own comment
            for (std::size_t j = 0; j < niso; ++j)
            {
                data[(i * niso) + j] = row[j]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels, j < niso == row.size() (asserted above) by construction
            }
        }

        if (!controls_.noDecay())
        {
            assert(decayChain_.has_value()); // populated by rebuildYieldGrid(), always called at least once by the constructor
            // Computed once and reused across every channel's own row,
            // rather than recomputing the same O(m^3) matrix exponential
            // once per channel via the dtDecay-taking applyDecay() --
            // dtDecay is identical for every row here.
            const auto propagator = decayChain_->propagator(dtDecay); // NOLINT(bugprone-unchecked-optional-access) -- see assert above
            for (std::size_t i = 0; i < nchannels; ++i)
            {
                decayChain_->applyDecay(propagator, std::span<double>(data).subspan(i * niso, niso)); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index,bugprone-unchecked-optional-access) -- i < nchannels, so i * niso + niso <= nchannels * niso == data.size() by construction; decayChain_ engaged per assert above
            }
        }

        const Array2D view(data.data(), nchannels, niso);
        return { view, std::move(data) };
    }

    auto Yields::yieldSum(const double mass, const double feH, const double dtDecay) const -> std::vector<double>
    {
        // Decay is applied once below, to the summed total, not here --
        // see this method's own comment for why passing 0 suffices
        // (yield()'s own decay step is a no-op at dtDecay == 0).
        const auto [view, data] = yield(mass, feH, 0.0);
        const std::size_t nchannels = view.extent(0);
        const std::size_t niso = view.extent(1);
        std::vector<double> result(niso, 0.0);
        for (std::size_t i = 0; i < nchannels; ++i)
        {
            for (std::size_t j = 0; j < niso; ++j)
            {
                result[j] += view[i, j]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i < nchannels, j < niso by construction
            }
        }

        if (!controls_.noDecay())
        {
            applyDecay(dtDecay, result);
        }

        return result;
    }

    void Yields::applyDecay(const double dtDecay, const std::span<double> values) const
    {
        assert(values.size() == isotopes_.size());
        assert(decayChain_.has_value()); // populated by rebuildYieldGrid(), always called at least once by the constructor
        decayChain_->applyDecay(dtDecay, values); // NOLINT(bugprone-unchecked-optional-access) -- see assert above
    }

    void Yields::applyDecay(const double dtDecay, const std::span<double> values, const bool decomposed) const
    {
        if (!decomposed) { applyDecay(dtDecay, values); return; }
        assert(decayChain_.has_value()); // populated by rebuildYieldGrid(), always called at least once by the constructor
        const std::size_t niso = isotopes_.size();
        const std::size_t nchannels = values.size() / niso;
        // Computed once and reused across every channel's own row, same
        // reasoning as yield()'s own identical pattern -- dtDecay is
        // identical for every row here too.
        const auto propagator = decayChain_->propagator(dtDecay); // NOLINT(bugprone-unchecked-optional-access) -- see assert above
        for (std::size_t i = 0; i < nchannels; ++i)
        {
            decayChain_->applyDecay(propagator, values.subspan(i * niso, niso)); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index,bugprone-unchecked-optional-access) -- i < nchannels, so i * niso + niso <= nchannels * niso == values.size() by construction; decayChain_ engaged per assert above
        }
    }

} // namespace yields
