/**
 * @file YieldCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by nucleosynthetic yield classes
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDCOMMONS_HPP
#define YIELDCOMMONS_HPP

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

/**
 * @brief A namespace to hold items dealing with nucleosynthetic yields
*/
namespace yields
{
    /**
     * @brief enum of the known nucleosynthetic yield channels
     * @details
     * More channels will be added here as more yield sources are
     * imported (see data/tools/yields/import_yield_tables.py's own
     * comment); nChannel_ is not itself a real channel, only a sentinel
     * marking how many there are, matching tracks::FieldIdx::nTrackQty's
     * identical role.
     */
    enum class Channel
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore on each enumerator matches this project's own member-variable convention, used here (rather than plain camelBack, this project's actual style for enum constants -- see elem::Symbols's own identical, deliberate deviation) at the user's own explicit request
        ccsn_,             /**< Core-collapse supernova ejecta */
        massiveStarWinds_, /**< Massive star winds (pre-supernova mass loss) */
        nChannel_          /**< Number of known channels -- not a real channel */
        // NOLINTEND(readability-identifier-naming)
    };

    /**
     * @brief Names of each channel, matching the keys used in yields.toml
     *   and the top-level HDF5 group names in each yield model's own file
     */
    constexpr std::array<std::string_view,
        static_cast<std::size_t>(Channel::nChannel_)> channelStr{
        "ccsn",
        "massive_star_winds"
    };

    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("yields")
        / std::filesystem::path("yields.toml")); /**< Default registry */

} // namespace yields

#endif // YIELDCOMMONS_HPP
