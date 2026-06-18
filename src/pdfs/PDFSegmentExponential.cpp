/**
 * @file PDFSegmentExponential.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentExponential routines
 * @date 2024-06,014
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentExponential.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// File-based constructor
pdfs::PDFSegmentExponential::PDFSegmentExponential(
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
    const double exMin = std::exp(-sMin_ / scale_);
    const double exMax = std::exp(-sMax_ / scale_);
    norm_ = 1.0 / (scale_ * (exMin - exMax));
}