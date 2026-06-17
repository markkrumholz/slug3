/**
 * @file PDFFileParser.cpp
 * @author Mark Krumholz
 * @brief Methods to parse PDF descriptor files
 * @details
 * This file provides a method to parse PDF files
 * and construct PDF objects based on them.
 * @date 2024-06-14
 */

#include "PDFCommons.hpp"
#include "PDFFileParser.hpp"
#include "PDFSegmentDelta.hpp"
#include "PDFSegmentExponential.hpp"
#include "PDFSegmentLognormal.hpp"
#include "PDFSegmentNormal.hpp"
#include "PDFSegmentPowerlaw.hpp"
#include "PDFSegmentSchechter.hpp"
#include "../utils/parseUtils.hpp"

// Little utility function to handle errors
[[noreturn ]]
void parseError(const std::string& err,
    const std::string& line,
    const std::string& fileName)
{
    std::string errMsg = "parsePDFDescriptor: " +
        err + "; file " + fileName;
    if (!line.empty()) errMsg += ", line " + line;
    throw std::runtime_error(errMsg);
}

namespace pdfs {

    // General parser to start and decide if file is basic or advanced
    auto parsePDFDescriptor(const std::string& fileName,
        RngType& rng) -> PDF
    {
        // First try to open file
        std::ifstream file(fileName);
        if (!file.is_open())
        {
            parseError("unable to open file", "", fileName);
        }

        // Read until we find a non-empty line
        std::string line;
        while (std::getline(file, line))
        {
            line = utils::trim(line);  // Trim whitespace
            if (line.empty()) continue; // Whitespace-only line
            break;  // If we are here, line is non-empty
        }

        // Tokenize the line
        auto tokens = utils::tokenize(line);
 
        // The first token must be either "advanced" (to indicate an
        // advanced mode file) or "breakpoints" followed by at least
        // two numbers (to indicate a basic mode file); check
        // that it is indeed one of these two
        FileFormats fmt;
        if (tokens[0] == "breakpoints" && tokens.size() > 2)
        {
            fmt = FileFormats::basic;
        } 
        else if (tokens[0] == "advanced") 
        {
            fmt = FileFormats::advanced;
        } 
        else
        {
            parseError("PDF file must start with "
                "'breakpoints BP1 BP2 ...' or 'advanced'",
                line, fileName);
        }

        // If we are in basic mode, extract the breakpoints, which are
        // stored in the rest of the line
        std::vector<double> breakpoints;
        if (fmt == FileFormats::basic)
        {
            tokens.erase(tokens.begin());  // Remove the token "breakpoints"
            for (auto t : tokens)
            {
                try
                {
                    breakpoints.push_back(utils::stod(t));
                }
                catch (const std::exception& error)
                {
                    parseError("could not convert string to float",
                        line, fileName);
                }
            }
            if (!std::is_sorted(breakpoints.begin(), breakpoints.end()))
            {
                parseError("breakpoints must be non-decreasing",
                    line, fileName);
            }
        }

        // Allocate holders for segments and weights
        std::vector<PDFSegment *> seg;
        std::vector<double> wgt;

        // Now parse the list of segments; segments are formatted
        // as follows:
        // segment
        // type TYPE
        // var1 VALUE
        // var2 VALUE
        // ...
        // The number and type of variables in var1, var2, etc. depend
        // on the type of segment.
        bool inSegment = false;
        size_t breakpointPtr = 0;
        SamplingMethods method;
        while (std::getline(file, line))
        {
 
            // Trim and tokenize the line
            line = utils::trim(line);
            if (line.empty()) continue; // Whitespace-only line; skip
            auto tok = utils::tokenize(line);

            // Check if we are currently a segment
            if (inSegment)
            {
                // We are in a segment, so check that the current line
                // declares the type of segment
                if (tok[0] != "type" || tok.size() != 2)
                {
                    parseError("expected 'type TYPE'", line, fileName);
                }

                // If we are in basic mode, extract lower and upper limits
                // for this segment
                double sMin = 0.0, sMax = 0.0;
                if (fmt == FileFormats::basic)
                {
                    if (breakpoints.size() < breakpointPtr + 2)
                    {
                        parseError("too few breakpoints for number of segments",
                            line, fileName);
                    }
                    sMin = breakpoints[breakpointPtr];
                    sMax = breakpoints[breakpointPtr+1];
                    breakpointPtr++;
                }

                // Now call the file-based constructor for the appropriate
                // segment type
                double w = 0.0;
                try
                {
                    if (tok[1] == "delta")
                    {
                        seg.push_back(new PDFSegmentDelta(file, rng, fmt, sMin, sMax, w));
                    }
                    else if (tok[1] == "exponential")
                    {
                        seg.push_back(
                            new PDFSegmentExponential(file, rng, fmt, sMin, sMax, w)
                        );
                    }
                    else if (tok[1] == "lognormal")
                    {
                        seg.push_back(
                            new PDFSegmentLognormal(file, rng, fmt, sMin, sMax, w)
                        );
                    }
                    else if (tok[1] == "normal")
                    {
                        seg.push_back(
                            new PDFSegmentNormal(file, rng, fmt, sMin, sMax, w)
                        );
                    }
                    else if (tok[1] == "powerlaw")
                    {
                        seg.push_back(
                            new PDFSegmentPowerlaw(file, rng, fmt, sMin, sMax, w)
                        );
                    }
                    else if (tok[1] == "schechter")
                    {
                        seg.push_back(
                            new PDFSegmentSchechter(file, rng, fmt, sMin, sMax, w)
                        );
                    } else {
                        parseError("unknown segment type " + tok[1],
                            line, fileName);
                    }
                }
                catch (const std::exception& error)
                {
                    if (strlen(error.what()) == 0)
                    {
                        parseError("reached end of file before "
                            "completing segment " + tok[1],
                            line, fileName);
                    }
                    else
                    {
                        parseError("invalid options for segment type " + tok[1],
                            line, fileName);                        
                    }
                }

                // In advanced format mode, store weight
                if (fmt == FileFormats::advanced) wgt.push_back(w);

                // Done with segment
                inSegment = false;

            }
            else
            {

                // If we're not in a segment, two types of line
                // are acceptable: declaration of a new segment,
                // taking the form
                //    segment
                // or a statement of what PDF policy we are using,
                // of the form
                //    method METHOD
                // Check for one of these two.

                if (tok[0] == "segment" && tok.size() == 1)
                {
                    // Start of a new segment
                    inSegment = true;
                    continue;
                }
                else if (tok[0] == "method" && tok.size() == 2)
                {
                    // Statement of drawing policy
                    if (tok[1] == "stop_nearest") method = SamplingMethods::stopNearest;
                    else if (tok[1] == "stop_before") method = SamplingMethods::stopBefore;
                    else if (tok[1] == "stop_after") method = SamplingMethods::stopAfter;
                    else if (tok[1] == "stop_50") method = SamplingMethods::stop50;
                    else if (tok[1] == "number") method = SamplingMethods::number;
                    else if (tok[1] == "poisson") method = SamplingMethods::poisson;
                    else if (tok[1] == "sorted_sampling") method = SamplingMethods::sorted;
                    else
                    {
                        parseError("unknown sampling method", line, fileName);
                    }
                }
                else
                {
                    parseError("unknown command", line, fileName);
                }
            }
        }

        // Close file
        file.close();

        // Final step depends on format
        if (fmt == FileFormats::basic)
        {
            // Basic format: calculate weights to make
            // segments continuous
            wgt.resize(seg.size());
            wgt[0] = 1.0;
            for (size_t i = 1; i < seg.size(); i++)
            {
                PDFSegment &s = *(seg[i]);
                PDFSegment &sPrev = *(seg[i-1]);
                wgt[i] = wgt[i-1] * sPrev(breakpoints[i]) /
                    s(breakpoints[i]);
            }

            // Create and return normalized PDF
            bool normalize = true;
            bool ownSegments = true;
            return PDF(seg, wgt, rng, method, 
                normalize, ownSegments);
        }
        else
        {
            // Advanced mode, so don't do any normalization,
            // just return
            bool normalize = false;
            bool ownSegments = true;
            return PDF(seg, wgt, rng, method, 
                normalize, ownSegments);            
        }
    }

}