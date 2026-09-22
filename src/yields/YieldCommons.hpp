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
#include <optional>
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
        agb_,              /**< Asymptotic giant branch stellar winds */
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
        "massive_star_winds",
        "agb"
    };

    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("yields")
        / std::filesystem::path("yields.toml")); /**< Default registry */

    /**
     * @struct YieldChannelDescriptor
     * @brief Everything needed to select and size one YieldChannel,
     *   other than the [Fe/H] range and registry a whole run's worth
     *   of channels share
     * @details
     * Members have the same meaning and type as the like-named
     * parameters of YieldChannel::YieldChannel() -- see its own
     * comment for each. Introduced so a caller building several
     * YieldChannel objects at once (Yields, the class that chains
     * multiple channels together) can hold "which channel/model/mass
     * range" as one value per channel, and so SimControls can read
     * yield-related keys and hand back a std::vector<
     * YieldChannelDescriptor> without also having to carry the
     * (single, run-wide) fehMin/fehMax/registryName alongside every
     * element of that vector.
     */
    struct YieldChannelDescriptor
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore matches this project's own member-variable convention (see utils::Bracket's identical public-struct-fields-with-trailing-underscore precedent), even though these particular names otherwise match YieldChannel::YieldChannel()'s own parameter names exactly
        Channel channel_;             /**< Which nucleosynthetic channel to load (e.g. Channel::ccsn_) */
        std::string modelName_;       /**< Name of the yield model (e.g. "sukhbold16") */
        std::optional<double> mMin_;  /**< Minimum stellar mass this channel should cover; nullopt for the model's own native minimum */
        std::optional<double> mMax_;  /**< Maximum stellar mass this channel should cover; nullopt for the model's own native maximum */
        // NOLINTEND(readability-identifier-naming)
    };

} // namespace yields

#endif // YIELDCOMMONS_HPP
