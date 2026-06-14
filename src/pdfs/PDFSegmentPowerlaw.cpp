/**
 * @file PDFSegmentPowerlaw.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentPowerlaw routines
 * @date 2024-06-14
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include "PDFSegmentPowerlaw.hpp"
#include "../utils/parseUtils.hpp"

// Basic mode constructor
pdfs::PDFSegmentPowerlaw::PDFSegmentPowerlaw(
    std::ifstream& file, double sMin, double sMax, rngType &rng) :
    PDFSegment(sMin, sMax, rng)
{
    // Process file; expect a single line "slope SLOPE"
    bool foundAll = false;
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        // The line should be of the form "slope SLOPE"
        if (tok[0] != "slope" || tok.size() != 2)
        {
            throw std::runtime_error(line);
        }
        try
        {
            alpha_ = utils::stod(tok[1]);
        } catch (const std::exception& error) {
            throw std::runtime_error(line);
        }
        foundAll = true;
        break;
    }

    // Throw error if something is missing
    if (!foundAll)
    {
        throw std::runtime_error("reached end of file while "
            "parsing powerlaw segment");
    }

    // Compute normalization constant
    if (alpha_ != -1) {
        norm_ = (alpha_ + 1) / (std::pow(sMax_, alpha_ + 1) - std::pow(sMin_, alpha_ + 1));
    } else {
        norm_ = 1.0 / std::log(sMax_ / sMin_);
    }
}
