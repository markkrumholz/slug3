/**
 * @file Yields.hpp
 * @author Mark Krumholz
 * @brief A class to chain together nucleosynthetic yield channels
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDS_HPP
#define YIELDS_HPP

#include "YieldCommons.hpp"
#include <string>
#include <utility>

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
     * A stub for now: the constructor stores controls_/registryName_
     * but does not yet build any YieldChannel objects from
     * controls.yieldChannels() -- that, and the rest of this class's
     * actual behavior, comes in a follow-up commit.
     */
    class Yields
    {
    public:

        /**
         * @brief Construct a Yields from a SimControls's own yieldChannels()
         * @param controls Simulation controls this Yields reads its
         *   yield-channel descriptors (controls.yieldChannels()) from,
         *   live, for the rest of its lifetime -- see controls_'s own
         *   comment. Must outlive this Yields.
         * @param registryName Name of the yield registry file
         */
        Yields(const io::SimControls& controls,
            std::string registryName = defaultRegistry) :
            controls_(controls),
            registryName_(std::move(registryName))
        {}

        // Copyable (rebinding controls_ to the same referent) but not
        // assignable (controls_ can't be reseated), matching Extinct's
        // own identical copy/move declarations exactly -- see its
        // comment.
        Yields(const Yields&) = default;
        Yields(Yields&&) = default;
        auto operator=(const Yields&) -> Yields& = delete;
        auto operator=(Yields&&) -> Yields& = delete;
        ~Yields() = default;

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

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Extinct's/Specsyn's own identical controls_ members exactly -- see either one's own comment for why. Only ever used through the same non-copyable, non-movable ownership pattern (unique_ptr in SimControls's own yields_) as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
        std::string registryName_;        /**< Name of the yield registry file */

    };

} // namespace yields

#endif // YIELDS_HPP
