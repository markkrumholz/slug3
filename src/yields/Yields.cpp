/**
 * @file Yields.cpp
 * @author Mark Krumholz
 * @brief Implementation of Yields
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Yields.hpp"
#include "../io/SimControls.hpp"
#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
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
        yieldChannels_.push_back(std::make_unique<YieldChannel>(
            descriptor, controls_.fehDist().getMin(), controls_.fehDist().getMax(), registryName_));
        descriptors_.push_back(descriptor);
    }

    void Yields::rebuildYieldGrid()
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

        // Push isotopes_ (and each channel's own descriptor mMin_/mMax_)
        // back down into every channel, synchronizing them all onto the
        // same isotope list -- see this method's own comment. descriptors_
        // is appended to by addChannel() in lockstep with yieldChannels_
        // itself, so this doesn't need to assume anything about
        // controls_.yieldChannels() staying in sync.
        for (std::size_t i = 0; i < yieldChannels_.size(); ++i)
        {
            yieldChannels_[i]->rebuildYieldGrid(descriptors_[i].mMin_, descriptors_[i].mMax_, isotopes_); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < yieldChannels_.size() == descriptors_.size() by construction, see this method's own comment
        }
    }

    auto Yields::yield(const double mass, const double feH) const
        -> std::pair<Array2D, std::vector<double>>
    {
        const std::size_t nchannels = yieldChannels_.size();
        const std::size_t niso = isotopes_.size();
        std::vector<double> data(nchannels * niso, 0.0);
        for (std::size_t i = 0; i < nchannels; ++i)
        {
            const auto row = yieldChannels_[i]->yield(mass, feH); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels == yieldChannels_.size() by construction
            assert(row.size() == niso); // guaranteed once rebuildYieldGrid() has synchronized every channel onto isotopes_ -- see this method's own comment
            for (std::size_t j = 0; j < niso; ++j)
            {
                data[(i * niso) + j] = row[j]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-constant-array-index) -- i < nchannels, j < niso == row.size() (asserted above) by construction
            }
        }
        const Array2D view(data.data(), nchannels, niso);
        return { view, std::move(data) };
    }

    auto Yields::yieldSum(const double mass, const double feH) const -> std::vector<double>
    {
        const auto [view, data] = yield(mass, feH);
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
        return result;
    }

} // namespace yields
