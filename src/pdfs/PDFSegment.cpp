/**
 * @file PDFSegment.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegment routines
 * @date 2024-06,014
 */

#include <fstream>
#include "PDFSegment.hpp"
#include "../utils/parseUtils.hpp"

auto pdfs::PDFSegment::segmentParser(std::ifstream& file,
    std::vector<std::string>& tok)
    -> std::map<std::string, double>
{
    // Empty map to hold result
    std::map<std::string, double> result;

    // Process segment
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto t = utils::tokenize(line);

        // Valid lines consist of one of our expected tokens,
        // followed by a single number; throw an error if
        // format is not as expected
        if (t.size() != 2)
        {
            throw std::runtime_error(line);
        }
        
        // See if this line matches any of our expected tokens
        // that we have not yet recorded
        bool foundMatch = false;
        for (auto tExpect : tok)
        {
            if (t[0] == tExpect && !result.contains(t[0]))
            {
                // Found a match, so extract value and store
                // in result map
                foundMatch = true;
                try
                {
                    result[t[0]] = utils::stod(tok[1]);
                } catch (const std::exception& error) {
                    throw std::runtime_error(line);
                }
                break;
            }
        }

        // Make sure this token matched one of our expected ones,
        // and throw an error if not
        if (!foundMatch)
        {
            throw std::runtime_error(line);
        }

        // Check if we have now found all our expected tokens,
        // and return if we have
        if (result.size() == tok.size()) return result;
    }

    // If we get here, then we have reached the end of the file
    // without finding all the required tokens for this segment,
    // so we throw an error
    throw std::runtime_error("");
}
