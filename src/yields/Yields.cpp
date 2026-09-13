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
    }

    void Yields::addChannel(const YieldChannelDescriptor& descriptor)
    {
        yieldChannels_.push_back(std::make_unique<YieldChannel>(
            descriptor, controls_.fehDist().getMin(), controls_.fehDist().getMax(), registryName_));
    }

} // namespace yields
