/**
 * @file Yields.hpp
 * @author Mark Krumholz
 * @brief A class to chain together nucleosynthetic yield channels
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDS_HPP
#define YIELDS_HPP

#include "../elem/IsotopeData.hpp"
#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

namespace yields
{
    /**
     * @class Yields
     * @brief Chains together the nucleosynthetic yield channels a
     *   SimControls requests, and is the connector between them and
     *   the rest of the code
     * @details
     * Built from controls.yieldChannels() (see the constructor's own
     * comment): one YieldChannel per descriptor, added via addChannel()
     * and owned in yieldChannels_. The constructor also collects every
     * loaded channel's own isotopes() into one deduplicated, sorted
     * isotopes_ -- the union of every isotope any requested channel
     * covers. There will be more to this class -- nothing yet combines
     * the channels' actual yield values together.
     */
    class Yields
    {
    public:

        /**
         * @brief Construct a Yields from a SimControls's own yieldChannels()
         * @param controls Simulation controls this Yields reads its
         *   yield-channel descriptors (controls.yieldChannels()) and
         *   [Fe/H] range (controls.fehDist()) from, live, for the rest
         *   of its lifetime -- see controls_'s own comment. Must
         *   outlive this Yields.
         * @param registryName Name of the yield registry file
         * @throws std::runtime_error if any descriptor in
         *   controls.yieldChannels() names a channel/model not found in
         *   registryName, or whose [Fe/H] range doesn't cover
         *   controls.fehDist() -- see addChannel()'s own comment
         * @details
         * Calls addChannel() once per entry in controls.yieldChannels(),
         * in order, then builds isotopes_ from the union of every
         * loaded channel's own isotopes() -- see isotopes()'s own
         * comment.
         */
        Yields(const io::SimControls& controls,
            std::string registryName = defaultRegistry);

        // Movable (moving yieldChannels_'s own ownership, and rebinding
        // controls_ to the same referent) but neither copyable (
        // yieldChannels_ holds move-only std::unique_ptr<YieldChannel>
        // elements) nor assignable (controls_ can't be reseated,
        // matching Extinct's/Specsyn's own identical controls_ members
        // -- see either one's own comment).
        Yields(const Yields&) = delete;
        Yields(Yields&&) = default;
        auto operator=(const Yields&) -> Yields& = delete;
        auto operator=(Yields&&) -> Yields& = delete;
        ~Yields() = default;

        /**
         * @brief Add one YieldChannel, built from a descriptor, to yieldChannels_
         * @param descriptor Which channel/model to load, and the mass
         *   range it should cover
         * @throws std::runtime_error if descriptor.channel_/modelName_
         *   is not found in the registry named by registryName(), or if
         *   controls_.fehDist()'s own range lies outside the [Fe/H]
         *   range actually available for that channel/model -- see
         *   YieldChannel::YieldChannel()'s own comment
         * @details
         * The new YieldChannel is loaded over controls_.fehDist()'s own
         * [min, max] range -- mirrors SimControls::readTracks()'s
         * identical use of fehDist_.getMin()/getMax() to pick the
         * [Fe/H] range a model is loaded over.
         */
        void addChannel(const YieldChannelDescriptor& descriptor);

        /**
         * @brief Return the SimControls this Yields reads its channel descriptors from
         * @return A const reference to controls_
         */
        [[nodiscard]] auto controls() const -> const io::SimControls& { return controls_; }

        /**
         * @brief Return the name of the yield registry this Yields reads from
         * @return A const reference to registryName_
         */
        [[nodiscard]] auto registryName() const -> const std::string& { return registryName_; }

        /**
         * @brief Return the yield channels loaded so far
         * @return A const reference to yieldChannels_, in the same
         *   order as controls().yieldChannels()
         */
        [[nodiscard]] auto yieldChannels() const -> const std::vector<std::unique_ptr<YieldChannel>>&
        {
            return yieldChannels_;
        }

        /**
         * @brief Return the union of every loaded channel's own isotopes()
         * @return A const reference to isotopes_: every isotope that
         *   appears in at least one of yieldChannels()'s own isotopes()
         *   lists, deduplicated and sorted (by IsotopeData's own Z-then-A
         *   ordering) -- unlike any one channel's own isotopes(), not
         *   necessarily in the order any single yield table tabulates
         *   them
         */
        [[nodiscard]] auto isotopes() const
            -> const std::vector<std::reference_wrapper<const elem::IsotopeData>>&
        {
            return isotopes_;
        }

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Extinct's/Specsyn's own identical controls_ members exactly -- see either one's own comment for why. Only ever used through the same non-copyable ownership pattern (unique_ptr in SimControls's own yields_) as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
        std::string registryName_;        /**< Name of the yield registry file */
        std::vector<std::unique_ptr<YieldChannel>> yieldChannels_; /**< Yield channels built via addChannel(), one per entry in controls_.yieldChannels() -- see yieldChannels()'s own comment */
        std::vector<std::reference_wrapper<const elem::IsotopeData>> isotopes_; /**< Union of every yieldChannels_ entry's own isotopes(), deduplicated and sorted -- see isotopes()'s own comment */

    };

} // namespace yields

#endif // YIELDS_HPP
