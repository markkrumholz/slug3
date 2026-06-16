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

// File-based constructor
pdfs::PDFSegmentExponential::PDFSegmentExponential(
    std::ifstream& file, rngType& rng, 
    fileFormats::format fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax, rng)
{
    // Action depends on format
    if (fmt == fileFormats::basic)
    {
        // Basic format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens = { "scale" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        scale_ = contents["scale"];
    }
    else
    {
        // Advanced format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "scale", "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        scale_ = contents["scale"];
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ >= sMax_)
        {
            throw std::runtime_error(
                "exponential segments must have min < max");
        }
    }

    // Calculate normalization constant for the PDF segment
    norm_ = 1.0 / (scale_ * (std::exp(-sMin_ / scale_) - std::exp(-sMax_ / scale_)));
}