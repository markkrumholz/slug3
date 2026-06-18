/**
 * @file PDFSegmentLognormal.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentLognormal routines
 * @date 2024-06,014
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentLognormal.hpp"
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

// File-based constructor
pdfs::PDFSegmentLognormal::PDFSegmentLognormal(
    std::ifstream& file,
    FileFormats fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax)
{
    // Action depends on format
    if (fmt == FileFormats::basic)
    {
        // Basic format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens = { "mean", "disp" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        mean_ = contents["mean"];
        stddev_ = std::numbers::ln10 * contents["disp"]; // Convert to base e
    }
    else
    {
        // Advanced format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "mean", "disp", "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        mean_ = contents["mean"];
        stddev_ = std::numbers::ln10 * contents["disp"]; // Convert to base e
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ >= sMax_)
        {
            throw std::runtime_error(
                "lognormal segments must have min < max");
        }
    }

    // Calculate normalization constant
    logMean_ = std::log(mean_);
    root2dev_ = std::numbers::sqrt2 * stddev_;
    norm_ = std::numbers::sqrt2 * std::numbers::inv_sqrtpi / 
        stddev_ / (
        std::erf( -std::log(sMin_/mean_) / root2dev_ ) -
        std::erf( -std::log(sMax_/mean_) / root2dev_ )
        );
}