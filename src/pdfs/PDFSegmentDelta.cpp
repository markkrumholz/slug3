/**
 * @file PDFSegmentExponential.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentExponential routines
 * @date 2024-06,014
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include "PDFSegmentDelta.hpp"
#include "../utils/parseUtils.hpp"

// File-based constructor
pdfs::PDFSegmentDelta::PDFSegmentDelta(
    std::ifstream& file, RngType& rng, 
    FileFormats fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax, rng)
{
    // Parameters expected only in advanced format
    if (fmt == FileFormats::advanced)
    {
         // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        sMin_ = contents["min"];
        sMax_ = contents["max"];
        wgt = contents["weight" ];

        // Safety check
        if (sMin_ != sMax_)
        {
            throw std::runtime_error(
                "delta segments must have min == max");
        }
    }
}
