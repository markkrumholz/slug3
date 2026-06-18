/**
 * @file PDFSegmentSchechter.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentSchechter routines
 * @date 2024-06-14
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentSchechter.hpp"
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_gamma.h>
#include <stdexcept>
#include <string>
#include <vector>

// File-based constructor
pdfs::PDFSegmentSchechter::PDFSegmentSchechter(
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
        std::vector<std::string> tokens = { "slope", "xStar" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["xStar"];
    }
    else
    {
        // Advanced format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "slope", "xStar", "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["xStar"];
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ >= sMax_)
        {
            throw std::runtime_error(
                "schechter segments must have min < max");
        }
    }

    // Compute normalization constant
    norm_ = 1.0 / (
        std::pow(sStar_, alpha_ + 1) * (
            gsl_sf_gamma_inc(alpha_ + 1, sMin_ / sStar_) -
            gsl_sf_gamma_inc(alpha_ + 1, sMax_ / sStar_)
        )
    );
}
