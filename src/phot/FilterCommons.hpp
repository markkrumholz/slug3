/**
 * @file FilterCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by photometric filter classes
 * @date 2026-07-27
 */

#ifndef FILTERCOMMONS_HPP
#define FILTERCOMMONS_HPP

#include <filesystem>
#include <string>

namespace phot
{
    inline static const std::string defaultRegistry = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("filters")
        / std::filesystem::path("filters.toml")); /**< Default filter registry */

} // namespace phot

#endif // FILTERCOMMONS_HPP
