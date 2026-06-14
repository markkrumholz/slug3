/**
 * @file PDFSegmentNormal.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentNormal routines
 * @date 2024-06-14
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include "PDFSegmentNormal.hpp"
#include "../utils/parseUtils.hpp"

// Basic mode constructor
pdfs::PDFSegmentNormal::PDFSegmentNormal(
    std::ifstream& file, double sMin, double sMax, rngType &rng) :
    PDFSegment(sMin, sMax, rng)
{
    // Process file; expect lines "mean MEAN" and "stddev STDDEV" in any order
    bool foundMean = false;
    bool foundStddev = false;
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        if (tok.size() != 2)
        {
            throw std::runtime_error(line);
        }
        else if (tok[0] == "mean")
        {
            try
            {
                mean_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundMean = true;
        }
        else if (tok[0] == "disp")
        {
            try
            {
                stddev_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundStddev = true;
        }
        else
        {
            throw std::runtime_error(line);
        }

        // Stop once we have both parameters
        if (foundMean && foundStddev) break;
    }

    // Throw error if something is missing
    if (!foundMean || !foundStddev)
    {
        throw std::runtime_error("reached end of file while "
            "parsing normal segment");
    }

    // Compute normalization constant
    norm_ = std::sqrt(2.0 / M_PI) / stddev_ /
        (std::erf((sMax_ - mean_) / (stddev_ * std::sqrt(2.0))) -
         std::erf((sMin_ - mean_) / (stddev_ * std::sqrt(2.0))));
}
