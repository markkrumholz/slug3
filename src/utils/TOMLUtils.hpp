/**
 * @file TOMLUtils.hpp
 * @author Mark Krumholz
 * @brief Shared utility functions for locating, parsing, and reading TOML registry files
 * @date 2026-08-01
 */

#ifndef TOMLUTILS_HPP
#define TOMLUTILS_HPP

#include "MiscUtils.hpp"
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace utils
{
    /**
     * @brief Get the string contents of a toml array, ignoring any non-string entries
     * @param arr Pointer to the array, as returned by node_view::as_array(); may be nullptr
     * @returns The array's string entries, in order
     */
    inline auto stringArrayContents(const toml::array* arr) -> std::vector<std::string>
    {
        std::vector<std::string> result;
        if (arr != nullptr)
        {
            arr->for_each([&result](auto&& el) -> void {
                if constexpr (toml::is_string<decltype(el)>)
                {
                    result.push_back(std::string(el));
                }
            });
        }
        return result;
    }

    /**
     * @brief Get the string contents of a top-level toml array field
     * @param registry The table to look up key in
     * @param key Name of the top-level key holding the array
     * @returns The array's string entries, in order, or an empty vector if
     *   registry has no array field named key
     */
    inline auto getStringArrayField(const toml::table& registry, const std::string& key)
    -> std::vector<std::string>
    {
        return stringArrayContents(registry.at_path(key).as_array());
    }

    /**
     * @brief Locate and parse a TOML file
     * @param fileName Name of the file to locate (via utils::getFilePath) and parse
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling class or function)
     * @returns The parsed toml table and the path it was read from
     * @throws std::runtime_error if fileName cannot be located via
     *   utils::getFilePath, or the file it resolves to cannot be parsed as toml
     */
    inline auto parseTOMLFile(const std::string& fileName, const std::string& context)
    -> std::pair<toml::table, std::filesystem::path>
    {
        auto path = getFilePath(fileName);
        if (path.empty())
        {
            throw std::runtime_error(
                context + ": registry " + fileName + " not found");
        }
        toml::table table;
        try
        {
            table = toml::parse_file(path.string());
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                context + ": registry " + path.string() + " is not a valid toml file");
        }
        return { std::move(table), std::move(path) };
    }

    /**
     * @brief Locate, parse, and validate a track/spectra-style registry file
     * @param registryName Name of the registry file
     * @param setsKey The top-level key naming the array of set names (e.g.
     *   "track_sets" or "spectra_sets")
     * @param itemNoun Plural noun for a set entry, used in error messages
     *   (e.g. "tracks" or "spectra")
     * @param context Prefix used in thrown error messages (typically the
     *   name of the calling function)
     * @returns The parsed table, the path it was read from, and the list of
     *   set names read from its setsKey field
     * @details
     * On top of parseTOMLFile's own validation, this further checks that
     * the registry has a non-empty setsKey array, and that every set it
     * names has its own table entry with both a "file" and a "Fe_H" field.
     * @throws std::runtime_error if any of the above conditions are not met
     */
    inline auto parseSetRegistry(
        const std::string& registryName,
        const std::string& setsKey,
        const std::string& itemNoun,
        const std::string& context)
    -> std::tuple<toml::table, std::filesystem::path, std::vector<std::string>>
    {
        auto [registry, registryPath] = parseTOMLFile(registryName, context);

        auto sets = getStringArrayField(registry, setsKey);
        if (sets.empty())
        {
            throw std::runtime_error(
                context + ": registry " + registryPath.string() +
                " does not contain a valid [" + setsKey + "] field");
        }

        for (const auto& s : sets)
        {
            if (!registry.contains(s))
            {
                std::string msg = context + ": registry " + registryPath.string();
                msg += " is missing entry for ";
                msg += itemNoun;
                msg += " ";
                msg += s;
                throw std::runtime_error(msg);
            }
            const auto entry = registry.at_path(s);
            if (!entry.at_path("file") || !entry.at_path("Fe_H"))
            {
                std::string msg = context + ": registry " + registryPath.string();
                msg += ", entry for ";
                msg += itemNoun;
                msg += " ";
                msg += s;
                msg += " is missing required 'file' or 'Fe_H' fields";
                throw std::runtime_error(msg);
            }
        }

        return { std::move(registry), std::move(registryPath), std::move(sets) };
    }

} // namespace utils

#endif // TOMLUTILS_HPP
