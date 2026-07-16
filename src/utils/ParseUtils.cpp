/**
 * @file ParseUtils.cpp
 * @author Mark Krumholz
 * @brief Implementation of utility functions for file and string parsing
 * @date 2026-07-16
 */

#include "ParseUtils.hpp"
#include "../extern/tomlplusplus/toml.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include "../pdfs/PDFSegmentDelta.hpp"
#include "MiscUtils.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

// Trim leading and trailing whitespace, and trailing comments
auto utils::trim(const std::string& str) -> std::string
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
    first = strNW.find_first_of('#');
    if (first != std::string::npos) {
        strNW = str.substr(0, first);
    }

    // Return
    return strNW;
}

// Initialize a PDF from a toml key
auto utils::initPDFFromKey(const toml::table& inputDeck,
    const std::string& key,
    const std::string& prefix) -> pdfs::PDF
{
    // First check for numerical value
    const std::optional<double> num =
        inputDeck.at_path(key).value<double>();
    if (num.has_value())
    {
        auto delta = std::make_unique<pdfs::PDFSegmentDelta>(num.value());
        auto pdf = pdfs::PDF(std::move(delta));
        return std::move(pdf);
    }

    // Next check for string value
    const std::optional<std::string> pdfFile =
        inputDeck.at_path(key).value<std::string>();
    if (pdfFile.has_value())
    {
        auto pdfFilePath = utils::getFilePath(pdfFile.value(), prefix).string();
        if (pdfFilePath.empty())
        {
            throw std::runtime_error(
                "initPDFFromKey: pdf file " + pdfFile.value() + " not found");
        }
        auto pdf = pdfs::parsePDFDescriptor(pdfFilePath);
        return std::move(pdf);
    }

    // If we get here, key does not exist or has invalid type
    throw std::runtime_error(
        "initPDFFromKey: invalid entry for " + key);
}

// Check for TOML key of type and return error if type is wrong
template<class T> auto utils::getTOMLKeyWithError(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<T>
{
    const auto node = inputDeck.at_path(key);
    if (node)
    {
        auto ret = node.value<T>();
        if (ret.has_value()) {
            // Key found, return value
            return ret;
        }

        // Key found, but doesn't match expected type
        throw std::runtime_error(
            "getTOMLKeyWithError: cannot understand value for key " +
            key);
    }
    if (required)
    {
        // Required key not found, throw error
        throw std::runtime_error(
            "getTOMLKeyWithError: required key " + key +
            " not found");
    }
    // Optional key not found, return nullopt
    return std::nullopt;
}

// Explicit instantiations for the types currently in use. Add a new
// instantiation here whenever getTOMLKeyWithError is called with a new
// type T, since the definition above is not visible outside this file.
template auto utils::getTOMLKeyWithError<std::string>(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<std::string>;
template auto utils::getTOMLKeyWithError<double>(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<double>;
template auto utils::getTOMLKeyWithError<unsigned int>(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<unsigned int>;
template auto utils::getTOMLKeyWithError<unsigned long>(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<unsigned long>;
template auto utils::getTOMLKeyWithError<bool>(
    const toml::table& inputDeck,
    const std::string& key,
    bool required) -> std::optional<bool>;