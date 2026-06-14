/**
 * @file PDFSegmentLognormal.cpp
 * @author Mark Krumholz
 * @brief Implementations of PDFSegmentLognormal routines
 * @date 2024-06,014
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include "PDFSegmentLognormal.hpp"
#include "../utils/parseUtils.hpp"

// Basic mode constructor
pdfs::PDFSegmentLognormal::PDFSegmentLognormal(
    std::ifstream& file, double sMin, double sMax, rngType &rng) :
    PDFSegment(sMin, sMax, rng)
{
    // Process file
    bool foundMean = false;
    bool foundDisp = false;
    std::string line;
    while (std::getline(file, line))
    {
        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) continue; // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        // The line should be of the form "mean MEAN" or
        // "disp DISP"; make sure it is, and if so read
        if (tok.size() != 2)
        {
            throw std::runtime_error(line);
        }
        else if (tok[1] == "mean")
        {
            try
            {
                mean_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundMean = true;
        }
        else if (tok[1] == "disp")
        {
            try
            {
                stddev_ = utils::stod(tok[1]);
            } catch (const std::exception& error) {
                throw std::runtime_error(line);
            }
            foundDisp = true;
        }
        else
        {
            throw std::runtime_error(line);
        }

        // Stop if we found both
        if (foundMean && foundDisp) break;
    }

    // Throw error if something is missing
    if (!foundMean || !foundDisp)
    {
        throw std::runtime_error("reached end of file while "
            "parsing lognormal segment");
    }

    // Calculate normalization constant
    log_mean_ = std::log(mean_);
    root2dev_ = std::sqrt(2.0) * stddev_;
    norm_ = std::sqrt(2.0 / M_PI) / stddev_ / (
        std::erf( -std::log(sMin_/mean_) / root2dev_ ) -
        std::erf( -std::log(sMax_/mean_) / root2dev_ )
    );
}