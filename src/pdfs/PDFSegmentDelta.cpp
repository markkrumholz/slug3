/**
 * @file PDFSegmentExponential.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentExponential routines
 * @date 2024-06,014
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentDelta.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

// File-based constructor
pdfs::PDFSegmentDelta::PDFSegmentDelta(
    std::ifstream& file,
    FileFormats fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax)
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

// Toml-based constructor
pdfs::PDFSegmentDelta::PDFSegmentDelta(
    toml::node_view<const toml::node> node,
    FileFormats fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax)
{
    // Parameters expected only in advanced format
    if (fmt == FileFormats::advanced)
    {
        // Look for the keys we need
        std::vector<std::string> tokens =
            { "min", "max", "weight" };
        auto contents = segmentParserToml(node, tokens);

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

// Write this segment's parameters into a toml table -- the inverse of
// the toml-based constructor above, in advanced format
void pdfs::PDFSegmentDelta::toToml(toml::table& tbl) const
{
    tbl.insert_or_assign("type", "delta");
    tbl.insert_or_assign("min", sMin_);
    tbl.insert_or_assign("max", sMax_);
}
