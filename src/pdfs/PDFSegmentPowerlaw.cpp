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

// File-based constructor
pdfs::PDFSegmentPowerlaw::PDFSegmentPowerlaw(
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
        std::vector<std::string> tokens = { "slope" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
    }
    else
    {
        // Advanced format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "slope", "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ >= sMax_)
        {
            throw std::runtime_error(
                "powerlaw segments must have min < max");
        }
    }
    
    // Compute normalization constant
    if (alpha_ != -1) {
        norm_ = (alpha_ + 1) / (std::pow(sMax_, alpha_ + 1) - std::pow(sMin_, alpha_ + 1));
    } else {
        norm_ = 1.0 / std::log(sMax_ / sMin_);
    }
}
