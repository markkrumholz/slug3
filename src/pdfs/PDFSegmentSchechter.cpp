/**
 * @file PDFSegmentSchechter.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentSchechter routines
 * @date 2024-06-14
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <gsl/gsl_sf_gamma.h>
#include "PDFSegmentSchechter.hpp"
#include "../utils/parseUtils.hpp"

// Basic mode constructor
pdfs::PDFSegmentSchechter::PDFSegmentSchechter(
    std::ifstream& file, double sMin, double sMax, rngType &rng) :
    PDFSegment(sMin, sMax, rng)
{
    // Process file; expect lines "xStar XSTAR" and "slope SLOPE" in any order
    bool foundSStar = false;
    bool foundAlpha = false;
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        if (tok.size() != 2)
        {
            throw std::runtime_error(line);
        }
        else if (tok[0] == "xStar")
        {
            try
            {
                sStar_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundSStar = true;
        }
        else if (tok[0] == "slope")
        {
            try
            {
                alpha_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundAlpha = true;
        }
        else
        {
            throw std::runtime_error(line);
        }

        // Stop once we have both parameters
        if (foundSStar && foundAlpha) break;
    }

    // Throw error if something is missing
    if (!foundSStar || !foundAlpha)
    {
        throw std::runtime_error("reached end of file while "
            "parsing Schechter segment");
    }

    // Compute normalization constant
    norm_ = 1.0 / (
        std::pow(sStar_, alpha_ + 1) * (
            gsl_sf_gamma_inc(alpha_ + 1, sMin_ / sStar_) -
            gsl_sf_gamma_inc(alpha_ + 1, sMax_ / sStar_)
        )
    );
}
