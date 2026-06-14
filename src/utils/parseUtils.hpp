/**
 * @file parseUtils.hpp
 * @author Mark Krumholz
 * @brief Utility functions for file and string parsing
 * @date 2024-06-12
 */

#ifndef PARSEUTILS_HPP
#define PARSEUTILS_HPP

#include <iterator>
#include <sstream>
#include <string>
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
        size_t processed_len;
        auto result = std::stod(str, &processed_len);
        if (processed_len != str.size())
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
    inline auto trim(const std::string& str) -> std::string
    {
        const std::string whitespace = " \t\n\r\f\v";
    
        // Find the first character that is not a whitespace
        auto first = str.find_first_not_of(whitespace);
        if (first == std::string::npos) {
            return ""; // The entire string is whitespace
        }
    
        // Find the last character that is not a whitespace
        auto last = str.find_last_not_of(whitespace);
    
        // Extract the substring without whitespace
        auto strNW = str.substr(first, (last - first + 1));

        // Find the first instance of the comment delimiter
        first = strNW.find_first_of("#");
        if (first != std::string::npos) {
            strNW = str.substr(0, first);
        }

        // Return
        return strNW;
    }

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
}

#endif // PARSEUTILS_HPP