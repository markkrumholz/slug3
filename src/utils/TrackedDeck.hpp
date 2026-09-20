/**
 * @file TrackedDeck.hpp
 * @author Mark Krumholz
 * @brief A read-only view of an input deck that records which of its keys are actually used
 * @date 2026-09-21
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TRACKEDDECK_HPP
#define TRACKEDDECK_HPP

#include "../pdfs/PDF.hpp"
#include "ParseUtils.hpp"
#include "TOMLUtils.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace utils
{
    /**
     * @struct DeckKeyReport
     * @brief One key of an input deck that was never used, or was deliberately ignored
     * @details
     * Returned by TrackedDeck::unused() and TrackedDeck::ignored(). Which
     * of hint_ and reason_ is filled in depends on which of the two
     * lists the report came from -- see each's own comment.
     */
    struct DeckKeyReport
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore matches this project's own member-variable convention (see yields::YieldChannelDescriptor's identical public-struct-fields-with-trailing-underscore precedent)
        std::string path_;            /**< Canonical dotted path of the key, e.g. "output.n_trial" */
        std::uint32_t line_ = 0;      /**< 1-based line of the key in the deck it was parsed from, or 0 if unknown (e.g. a deck built in code rather than parsed) */
        std::uint32_t column_ = 0;    /**< 1-based column of the key, or 0 if unknown */
        std::string hint_;            /**< For an unused key: a suggestion of what was probably meant, or empty if there is none */
        std::string reason_;          /**< For an ignored key: why it was not read, e.g. "nebular.compute_neb is false" */
        // NOLINTEND(readability-identifier-naming)
    };

    namespace trackeddeck_detail
    {
        /**
         * @brief Canonical text form of a TOML path
         * @param path A dotted path such as "stars.FeH"
         * @returns The path as toml++ itself would print it (so equivalent
         *   spellings compare equal), or path unchanged if it does not parse
         */
        inline auto canonicalPath(const std::string_view path) -> std::string
        {
            const toml::path parsed{ path };
            return parsed ? parsed.str() : std::string(path);
        }

        /**
         * @brief Append a table key to a dotted path, quoting the key if it contains anything but bare-key characters
         * @param prefix Path of the enclosing table, empty at the top level
         * @param key The key to append
         * @returns The extended path
         * @details
         * The quoting is only so that a key such as "x.y" is displayed
         * unambiguously in a report. toml++'s own path syntax splits on
         * every '.', with no way to quote one, so a key whose name
         * contains a dot cannot be looked up through toml::table::at_path()
         * at all; no SLUG keyword has such a name, so a deck key like
         * that is never read and is always reported as unused.
         */
        inline auto childPath(const std::string& prefix, const std::string_view key) -> std::string
        {
            const bool bare = !key.empty() && std::ranges::all_of(key, [](const unsigned char c)
                { return (std::isalnum(c) != 0) || c == '_' || c == '-'; });
            std::string text = bare ? std::string(key) : std::string("\"");
            if (!bare)
            {
                for (const char c : key)
                {
                    if (c == '"' || c == '\\') { text += '\\'; }
                    text += c;
                }
                text += '"';
            }
            return prefix.empty() ? text : prefix + "." + text;
        }

        /** @brief Whether path is prefix itself or lies beneath it (prefix followed by '.' or '[') */
        inline auto isUnder(const std::string_view path, const std::string_view prefix) -> bool
        {
            if (!path.starts_with(prefix)) { return false; }
            return path.size() == prefix.size() || path[prefix.size()] == '.' || path[prefix.size()] == '[';
        }

        /** @brief The part of a dotted path after its last '.', or the whole path if it has none */
        inline auto lastSegment(const std::string_view path) -> std::string
        {
            const auto dot = path.rfind('.');
            return std::string(dot == std::string_view::npos ? path : path.substr(dot + 1));
        }

        /** @brief Levenshtein edit distance between two strings */
        inline auto editDistance(const std::string_view a, const std::string_view b) -> std::size_t
        {
            std::vector<std::size_t> prev(b.size() + 1);
            std::vector<std::size_t> cur(b.size() + 1);
            for (std::size_t j = 0; j <= b.size(); ++j) { prev[j] = j; }
            for (std::size_t i = 1; i <= a.size(); ++i)
            {
                cur[0] = i;
                for (std::size_t j = 1; j <= b.size(); ++j)
                {
                    const std::size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                    cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, sub });
                }
                std::swap(prev, cur);
            }
            return prev[b.size()];
        }

        /**
         * @brief Suggest which key the user probably meant for an unused one
         * @param unusedPath Canonical path of the key that was never read
         * @param wanted Paths the code looked for but did not find
         * @returns A phrase such as "did you mean 'n_trial'?", or empty if nothing plausible
         * @details
         * Two rules, in order. First, a wanted path with the same last
         * segment as the unused key: the key is right but in the wrong
         * section, e.g. output.n_trial when n_trial is a top-level key.
         * Second, a wanted path whose last segment is within two edits of
         * the unused key's, and at least four characters long, so that
         * short names do not match everything: a misspelling, e.g.
         * stars.v_vcirt for stars.v_vcrit. The closest such path wins.
         */
        inline auto suggestKey(const std::string& unusedPath, const std::set<std::string>& wanted) -> std::string
        {
            const std::string leaf = lastSegment(unusedPath);
            const auto same = std::ranges::find_if(wanted,
                [&leaf](const std::string& w) { return lastSegment(w) == leaf; });
            if (same != wanted.end()) { return "did you mean '" + *same + "'?"; }

            constexpr std::size_t maxDistance = 2;
            constexpr std::size_t minLength = 4;
            if (leaf.size() < minLength) { return {}; }
            const std::string* best = nullptr;
            std::size_t bestDist = maxDistance + 1;
            for (const std::string& w : wanted)
            {
                const std::size_t d = editDistance(leaf, lastSegment(w));
                if (d < bestDist) { bestDist = d; best = &w; }
            }
            return best == nullptr ? std::string() : "did you mean '" + *best + "'?";
        }
    } // namespace trackeddeck_detail

    /**
     * @class TrackedDeck
     * @brief A read-only view of a toml input deck that records which keys are used
     * @details
     * SLUG silently ignores any input-deck key it never reads, so a
     * misspelled or misplaced key (e.g. n_trial given under [output]
     * rather than at the top level) has no effect and no warning.
     * TrackedDeck wraps the parsed deck so that every read is recorded,
     * and can then report the keys that were never read.
     *
     * Only *leaves* are tracked: values, and arrays (which are read as
     * a unit). A table is not itself a key, and looking one up -- for
     * example to test whether [yields.channel2] exists -- does not count
     * as using anything beneath it, so the keys inside it are still
     * reported if nothing reads them. Code that consumes a whole table
     * some other way must say so with markUsedAll().
     *
     * Some keys are valid but legitimately go unread, e.g.
     * nebular.log_U when nebular.compute_neb is false. Code that skips
     * a section for such a reason calls markIgnored() with the reason,
     * so those keys can be reported separately from genuinely unknown
     * ones.
     *
     * Every read must go through this class for the report to be
     * complete, which is why it does not expose the underlying
     * toml::table: pass it (as a const reference) to everything that
     * reads the deck, in place of the table itself. It only holds a
     * reference to the table, which must outlive it. The recorded state
     * is mutable so that reading is const, and is not thread safe.
     */
    class TrackedDeck
    {
    public:
        /**
         * @brief Wrap an input deck, recording every leaf key it contains
         * @param deck The parsed deck; must outlive this object
         */
        explicit TrackedDeck(const toml::table& deck) : deck_(std::cref(deck))
        {
            collectLeaves(deck, std::string());
        }

        /**
         * @brief Look up a key, recording the access
         * @param path Dotted path of the key, e.g. "stars.FeH"
         * @returns The same node_view toml::table::at_path() would; empty
         *   if the key does not exist
         * @details
         * A leaf that exists is marked used. A key that does not exist is
         * remembered as one the code looked for, which is what lets an
         * unused key be matched to the key it was probably meant to be
         * (see unused()). A table is neither: looking one up marks
         * nothing.
         */
        [[nodiscard]] auto atPath(const std::string_view path) const -> toml::node_view<const toml::node>
        {
            const auto node = deck_.get().at_path(path);
            const std::string key = trackeddeck_detail::canonicalPath(path);
            if (!node)
            {
                wanted_.insert(key);
            }
            else if (!node.is_table())
            {
                markLeafOrAncestor(key);
            }
            return node;
        }

        /**
         * @brief Read a typed value, recording the access
         * @tparam T Type to read; any type utils::getTOMLKeyWithError() is instantiated for
         * @param path Dotted path of the key
         * @param required Whether a missing key is an error
         * @returns The value, or an empty optional if the key is absent and not required
         * @throws std::runtime_error under exactly the conditions
         *   utils::getTOMLKeyWithError() does: a required key is missing,
         *   or the key cannot be read as a T
         */
        template<class T>
        [[nodiscard]] auto value(const std::string& path, const bool required = false) const -> std::optional<T>
        {
            static_cast<void>(atPath(path));
            return utils::getTOMLKeyWithError<T>(deck_.get(), path, required);
        }

        /**
         * @brief Read a probability distribution, recording the access
         * @param path Dotted path of the key
         * @param prefix Directory prefix used to locate a PDF file, as for utils::initPDFFromKey()
         * @returns The distribution
         * @throws std::runtime_error under exactly the conditions utils::initPDFFromKey() does
         */
        [[nodiscard]] auto initPDF(const std::string& path, const std::string& prefix = "") const -> pdfs::PDF
        {
            static_cast<void>(atPath(path));
            return utils::initPDFFromKey(deck_.get(), path, prefix);
        }

        /**
         * @brief Record that every key at or beneath a path has been consumed
         * @param prefix Dotted path of a table (or a single key)
         */
        void markUsedAll(const std::string_view prefix) const
        {
            const std::string p = trackeddeck_detail::canonicalPath(prefix);
            for (auto& [path, leaf] : leaves_)
            {
                if (trackeddeck_detail::isUnder(path, p)) { leaf.used_ = true; }
            }
        }

        /**
         * @brief Record that every not-yet-used key at or beneath a path is being deliberately ignored
         * @param prefix Dotted path of a table (or a single key)
         * @param reason Why the keys are not read, e.g. "nebular.compute_neb is false"
         * @details
         * The keys are not marked used, so a later read still counts as a
         * use, and they are reported by ignored() instead of unused().
         */
        void markIgnored(const std::string_view prefix, const std::string_view reason) const
        {
            const std::string p = trackeddeck_detail::canonicalPath(prefix);
            for (auto& [path, leaf] : leaves_)
            {
                if (!leaf.used_ && leaf.reason_.empty() && trackeddeck_detail::isUnder(path, p))
                {
                    leaf.reason_ = std::string(reason);
                }
            }
        }

        /**
         * @brief Keys that were never read and were not marked ignored
         * @returns One report per key, in path order, each with a hint if
         *   one could be found: the key the code looked for that this one
         *   most plausibly was meant to be
         */
        [[nodiscard]] auto unused() const -> std::vector<DeckKeyReport>
        {
            std::vector<DeckKeyReport> result;
            for (const auto& [path, leaf] : leaves_)
            {
                if (leaf.used_ || !leaf.reason_.empty()) { continue; }
                result.push_back({ path, leaf.line_, leaf.column_,
                    trackeddeck_detail::suggestKey(path, wanted_), std::string() });
            }
            return result;
        }

        /**
         * @brief Keys that were never read because the code deliberately skipped them
         * @returns One report per key, in path order, each with the reason
         *   given to markIgnored()
         */
        [[nodiscard]] auto ignored() const -> std::vector<DeckKeyReport>
        {
            std::vector<DeckKeyReport> result;
            for (const auto& [path, leaf] : leaves_)
            {
                if (leaf.used_ || leaf.reason_.empty()) { continue; }
                result.push_back({ path, leaf.line_, leaf.column_, std::string(), leaf.reason_ });
            }
            return result;
        }

    private:
        /** @brief What is recorded about one leaf key */
        struct Leaf
        {
            // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore matches this project's own member-variable convention, as for DeckKeyReport above
            bool used_ = false;          /**< Whether the key has been read */
            std::string reason_;         /**< Why the key is deliberately ignored; empty if not */
            std::uint32_t line_ = 0;     /**< 1-based line in the deck, or 0 if unknown */
            std::uint32_t column_ = 0;   /**< 1-based column in the deck, or 0 if unknown */
            // NOLINTEND(readability-identifier-naming)
        };

        /**
         * @brief Record every leaf beneath a table
         * @param table The table to walk
         * @param prefix Canonical path of table, empty for the top level
         */
        void collectLeaves(const toml::table& table, const std::string& prefix)
        {
            for (const auto& [key, node] : table)
            {
                const std::string path = trackeddeck_detail::childPath(prefix, key.str());
                if (const auto* sub = node.as_table()) { collectLeaves(*sub, path); continue; }
                Leaf leaf;
                leaf.line_ = node.source().begin.line;
                leaf.column_ = node.source().begin.column;
                leaves_.emplace(path, leaf);
            }
        }

        /**
         * @brief Mark the leaf at a path as used, or the leaf that contains it
         * @param key Canonical path of a node that exists
         * @details
         * A path that indexes into an array (e.g. "a.b[1]") names a
         * position inside a leaf rather than a leaf of its own, so
         * fall back to progressively shorter prefixes until one matches.
         */
        void markLeafOrAncestor(std::string key) const
        {
            while (!key.empty())
            {
                const auto it = leaves_.find(key);
                if (it != leaves_.end()) { it->second.used_ = true; return; }
                const auto cut = key.find_last_of(".[");
                if (cut == std::string::npos) { return; }
                key.resize(cut);
            }
        }

        std::reference_wrapper<const toml::table> deck_;  /**< The deck being tracked */
        mutable std::map<std::string, Leaf> leaves_;  /**< Canonical path -> state, for every leaf of the deck */
        mutable std::set<std::string> wanted_;        /**< Canonical paths the code looked for and did not find */
    };

} // namespace utils

#endif // TRACKEDDECK_HPP
