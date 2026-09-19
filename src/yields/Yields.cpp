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
#include "../io/SimControls.hpp"
#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yields
{
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

        // Rebuild decayChains_ to match the now-final isotopes_ -- see
        // this method's own comment.
        decayChains_.clear();
        for (const auto& iso : isotopes_)
        {
            if (!iso.get().stable())
            {
                decayChains_.try_emplace(iso.get(), iso.get());
            }
        }
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
            for (std::size_t i = 0; i < nchannels; ++i)
            {
                applyDecay(dtDecay, std::span<double>(data).subspan(i * niso, niso)); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels, so i * niso + niso <= nchannels * niso == data.size() by construction
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
        std::vector<double> deltaYield(values.size(), 0.0);
        for (std::size_t p = 0; p < isotopes_.size(); ++p)
        {
            const auto& parent = isotopes_[p].get(); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p < isotopes_.size() by construction
            if (parent.stable()) { continue; } // nothing to decay, and no decayChains_ entry for it either -- see rebuildYieldGrid()'s own comment
            const auto chainIt = decayChains_.find(parent);
            assert(chainIt != decayChains_.end()); // every unstable isotopes_ entry has one, by rebuildYieldGrid()'s own construction
            const auto fExpect = chainIt->second.yield(dtDecay);
            const auto& products = chainIt->second.products();
            const double parentMass = values[p]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- p < isotopes_.size() == values.size() by construction
            for (std::size_t k = 0; k < products.size(); ++k)
            {
                // products() (topologically sorted by DecayChain's own
                // construction) is not in isotopes_'s own (Z-then-A)
                // order, and may list isotopes isotopes_ doesn't
                // tabulate at all -- resolve by IsotopeData equality,
                // skipping any product not found, per this method's own
                // comment.
                const auto isoIt = std::ranges::find_if(isotopes_,
                    [&products, k](const auto& candidate)
                    { return candidate.get() == products[k].get(); }); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k < products.size() by construction
                if (isoIt == isotopes_.end()) { continue; }
                const auto idx = static_cast<std::size_t>(std::distance(isotopes_.begin(), isoIt));
                if (idx == p)
                {
                    // The parent isotope itself: fExpect[k] <= 1 is the
                    // surviving fraction, so this is always <= 0 (see
                    // this method's own comment for the sign).
                    deltaYield[idx] += parentMass * (fExpect[k] - 1.0); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- idx < isotopes_.size() == deltaYield.size(), k < products.size() == fExpect.size(), by construction
                }
                else
                {
                    deltaYield[idx] += parentMass * fExpect[k]; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- see above
                }
            }
        }

        for (std::size_t j = 0; j < values.size(); ++j) { values[j] += deltaYield[j]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- j < values.size() == deltaYield.size() by construction
    }

} // namespace yields
