/**
 * @file ParseUtils.hpp
 * @author Mark Krumholz
 * @brief Utility functions for file and string parsing
 * @date 2024-06-12
 */

#ifndef PARSEUTILS_HPP
#define PARSEUTILS_HPP

#include "../pdfs/PDF.hpp"
#include <cstddef>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

/**
 * @brief A namespace to hold utility functions
 */
namespace utils {

    /**
     * @brief Convert string to double
     * @param str The string to be converted
     * @returns str converted to a double
     * @details
     * This is a thin wrapper around std::stod, which checks
     * to make sure that the entire input was converted
     * successfully, and throws an invalid_argument error
     * if not.
     */
    inline auto stod(const std::string& str) -> double
    {
        size_t processedLen = 0;
        auto result = std::stod(str, &processedLen);
        if (processedLen != str.size())
        {
            throw std::invalid_argument("");
        }
        return result;
    }

    /**
     * @brief Trim leading and trailing whitespace, and trailing comments
     * @param str The string to be trimmed
     * @returns The trimmed string
     * @details
     * The returned string has the leading and trailing whitespace
     * removed. It also has any trailing comments, starting with
     * the # character, removed.
     */
    auto trim(const std::string& str) -> std::string;

    /**
     * @brief Split a string into tokens based on whitespace
     * @param str The string to split
     * @returns A vector of strings containing str split by whitespace
     */
    inline auto tokenize(const std::string& str)
        -> std::vector<std::string>
    {
        std::istringstream stream(str);
        std::vector<std::string> tokens(
            (std::istream_iterator<std::string>(stream)),
            std::istream_iterator<std::string>()
        );
        return tokens;
    }

    /**
     * @brief Split a string on every occurrence of a delimiter character
     * @param str The string to split
     * @param delim The delimiter character
     * @returns str split on delim, keeping empty tokens (e.g. "a..b" splits
     *   to {"a", "", "b"}) rather than collapsing them, so a malformed
     *   input with an extra or missing delimiter fails to match the
     *   expected token count downstream instead of silently merging fields
     */
    inline auto splitOnChar(const std::string& str, const char delim)
        -> std::vector<std::string>
    {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        while (true)
        {
            const std::size_t pos = str.find(delim, start);
            if (pos == std::string::npos)
            {
                tokens.push_back(str.substr(start));
                break;
            }
            tokens.push_back(str.substr(start, pos - start));
            start = pos + 1;
        }
        return tokens;
    }

    /**
     * @brief Initialize a PDF from a string
     * @param value The string to interpret; plays the same role as
     *   the value of the toml key in TOMLUtils.hpp's initPDFFromKey()
     * @param prefix Prefix to look for PDF file relative to SLUG_DIR
     * @details
     * If value can be converted to a numerical value (via
     * utils::stod), this is interpreted as specifying a delta
     * function at that value. Otherwise, value is interpreted as the
     * name of a PDF file descriptor, with the path to the file
     * resolved by utils::getFilePath.
     * @throws std::runtime_error if value is not numeric and does not
     *   name a file that can be found
    */
    auto initPDFFromString(const std::string& value,
        const std::string& prefix = "") -> pdfs::PDF;

    /**
     * @brief Check for TOML key of type and return error if type is wrong
     * @tparam T The type to check for
     * @param inputDeck Input deck for simulation
     * @param key Name of key
     * @param required True if key is required
     * @details
     * This routine checks if the input key exists and can be retrieved
     * as of type T. Behavior is as follows: (1) if the key exists and can
     * be interpreted as type T, return a full optional<T>; (2) if the key
     * exists and cannot be interpreted as T, throw a runtime error; (3) if
     * the key does not exist and required is false, return ane empty
     * optional; (4) if the key does not exist and required is true, throw a
     * runtime error.
     */
    template<class T> auto getTOMLKeyWithError(
        const toml::table& inputDeck,
        const std::string& key,
        bool required = false) -> std::optional<T>;

} // namespace utils

#endif // PARSEUTILS_HPP