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
#include <memory>
#include <string>
#include <utility>

namespace yields
{
    Yields::Yields(const io::SimControls& controls, std::string registryName) :
        controls_(controls),
        registryName_(std::move(registryName))
    {
        for (const auto& descriptor : controls_.yieldChannels())
        {
            addChannel(descriptor);
        }

        // Union every loaded channel's own isotopes() into one
        // deduplicated, sorted isotopes_ -- see its own comment.
        // IsotopeData's own operator</operator== (Z first, then A) is
        // reused here via the referenced object (.get()), rather than
        // reference_wrapper's own identity-based comparison, so two
        // channels that both tabulate the same isotope (necessarily the
        // very same elem::isotopeTable() entry, since that table is a
        // single, global, per-(Z,A) map) are correctly recognized as
        // one isotope, not kept as duplicates.
        for (const auto& channel : yieldChannels_)
        {
            isotopes_.insert(isotopes_.end(), channel->isotopes().begin(), channel->isotopes().end());
        }
        std::ranges::sort(isotopes_,
            [](const auto& lhs, const auto& rhs) { return lhs.get() < rhs.get(); });
        const auto dup = std::ranges::unique(isotopes_,
            [](const auto& lhs, const auto& rhs) { return lhs.get() == rhs.get(); });
        isotopes_.erase(dup.begin(), dup.end());
    }

    void Yields::addChannel(const YieldChannelDescriptor& descriptor)
    {
        yieldChannels_.push_back(std::make_unique<YieldChannel>(
            descriptor, controls_.fehDist().getMin(), controls_.fehDist().getMax(), registryName_));
    }

} // namespace yields
