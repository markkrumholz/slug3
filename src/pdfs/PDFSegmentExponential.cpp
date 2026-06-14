/**
 * @file PDFSegmentExponential.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentExponential routines
 * @date 2024-06,014
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include "PDFSegmentExponential.hpp"
#include "../utils/parseUtils.hpp"

// Basic mode constructor
pdfs::PDFSegmentExponential::PDFSegmentExponential(
    std::ifstream& file, double sMin, double sMax, rngType &rng) :
    PDFSegment(sMin, sMax, rng)
{
    // Process file
    bool foundAll = false;
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        // The line should be of the form "scale SCALE";
        // make sure it is, and if so read the scale
        if (tok[0] != "scale" || tok.size() != 2)
        {
            throw std::runtime_error(line);
        }
        try
        {
            scale_ = utils::stod(tok[1]);
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
            "parsing exponential segment");
    }

    // Calculate normalization constant for the PDF segment
    norm_ = 1.0 / (scale_ * (std::exp(-sMin_ / scale_) - std::exp(-sMax_ / scale_)));
}