/**
 * @file PDFSegmentSchechter.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentSchechter routines
 * @date 2024-06-14
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentSchechter.hpp"
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_gamma.h>
#include <stdexcept>
#include <string>
#include <toml.hpp>
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
        std::vector<std::string> tokens = { "slope", "x_star" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["x_star"];
    }
    else
    {
        // Advanced format

        // Call segment parser to get the tokens we need
        std::vector<std::string> tokens =
            { "slope", "x_star", "min", "max", "weight" };
        auto contents = segmentParser(file, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["x_star"];
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

// Toml-based constructor
pdfs::PDFSegmentSchechter::PDFSegmentSchechter(
    toml::node_view<const toml::node> node,
    FileFormats fmt,
    double& sMin, double& sMax, double& wgt) :
    PDFSegment(sMin, sMax)
{
    // Action depends on format
    if (fmt == FileFormats::basic)
    {
        // Basic format

        // Look for the keys we need
        std::vector<std::string> tokens = { "slope", "x_star" };
        auto contents = segmentParserToml(node, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["x_star"];
    }
    else
    {
        // Advanced format

        // Look for the keys we need
        std::vector<std::string> tokens =
            { "slope", "x_star", "min", "max", "weight" };
        auto contents = segmentParserToml(node, tokens);

        // Use the parsed results to set parameters
        alpha_ = contents["slope"];
        sStar_ = contents["x_star"];
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

// Write this segment's parameters into a toml table -- the inverse of
// the toml-based constructor above, in advanced format
void pdfs::PDFSegmentSchechter::toToml(toml::table& tbl) const
{
    tbl.insert_or_assign("type", "schechter");
    tbl.insert_or_assign("slope", alpha_);
    tbl.insert_or_assign("x_star", sStar_);
    tbl.insert_or_assign("min", sMin_);
    tbl.insert_or_assign("max", sMax_);
}
