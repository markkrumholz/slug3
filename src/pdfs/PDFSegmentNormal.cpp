/**
 * @file PDFSegmentNormal.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentNormal routines
 * @date 2024-06-14
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentNormal.hpp"
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

// File-based constructor
pdfs::PDFSegmentNormal::PDFSegmentNormal(
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
        stddev_ = contents["disp"];
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
        stddev_ = contents["disp"];
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ >= sMax_)
        {
            throw std::runtime_error(
                "normal segments must have min < max");
        }
    }

    // Compute normalization constant
    norm_ = std::numbers::sqrt2 * std::numbers::inv_sqrtpi / stddev_ /
        (std::erf((sMax_ - mean_) / (stddev_ * std::numbers::sqrt2)) -
         std::erf((sMin_ - mean_) / (stddev_ * std::numbers::sqrt2)));
}
