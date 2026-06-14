/**
 * @file PDFFileParser.cpp
 * @author Mark Krumholz
 * @brief Methods to parse PDF descriptor files
 * @details
 * This file provides a method to parse PDF files
 * and construct PDF objects based on them.
 * @date 2024-06-14
 */

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
        rngType& rng) -> PDF
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

        // The first non-empty, non-comment line must be either
        // "advanced" (to indicate an advanced mode file) or
        // "breakpoints" (to indicate a basic mode file); check
        // that it is indeed one of these two, and call the appropriate
        // parser to finish parsing the descriptor file
        if (line.starts_with("breakpoints"))
        {
            return parsePDFDescriptorBasic(fileName, rng, line, file);
        } 
        else if (line.starts_with("advanced")) 
        {
            //return parsePDFDescriptorAdvanced(fileName, rng, file);
        } 
        else
        {
            parseError("PDF file must start with 'breakpoints' or 'advanced'",
                line, fileName);
        }
    }

    // Basic parser
    auto parsePDFDescriptorBasic(const std::string& fileName, 
        rngType &rng,
        std::string& breakpointLine,
        std::ifstream& file) -> PDF
    {
        // Create holder for segments and sampling method
        std::vector<PDFSegment *> seg;
        auto method = samplingMethods::stopNearest; // Default

        // From the line we have been given, extract the breakpoints
        auto tokens = utils::tokenize(breakpointLine);
        tokens.erase(tokens.begin());  // Remove the word "breakpoint"
        if (tokens.size() < 2)
        {
            parseError("PDF file must have at least two breakpoints",
                breakpointLine, fileName); // Need two breakpoints
        }
        std::vector<double> breakpoints;
        for (auto t : tokens)
        {
            try
            {
                breakpoints.push_back(utils::stod(t));
            }
            catch (const std::exception& error)
            {
                parseError("could not convert string to float",
                    breakpointLine, fileName);
            }
        }
        if (!std::is_sorted(breakpoints.begin(), breakpoints.end()))
        {
            parseError("breakpoints must be non-decreasing",
                breakpointLine, fileName);
        }

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
        std::string line;
        size_t breakpointPtr = 0;
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

                // Extract lower and upper limits for this segment
                double sMin, sMax;
                if (breakpoints.size() < breakpointPtr + 2)
                {
                    parseError("too few breakpoints for number of segments",
                        line, fileName);
                }
                sMin = breakpoints[breakpointPtr];
                sMax = breakpoints[breakpointPtr+1];
                breakpointPtr++;

                // Next step depends on segment type
                if (tok[1] == "delta")
                {
                    // No extra tokens expected, so just make sure
                    // breakpoints are correct, then create
                    if (sMin != sMax)
                    {
                        parseError("min and max breakpoints must match for a delta "
                            "function segment", line, fileName);
                    }
                    seg.push_back(new PDFSegmentDelta(sMin, rng));
                }
                else if (tok[1] == "exponential")
                {
                    // Call the file-based constructor
                    try {
                        seg.push_back(
                            new PDFSegmentExponential(file, sMin, sMax, rng)
                        );
                    } 
                    catch (const std::exception& error)
                    {
                        parseError("invalid exponential segment",
                            error.what(), fileName);
                    }
                } 
                else if (tok[1] == "lognormal")
                {
                    // Call the file-based constructor
                    try {
                        seg.push_back(
                            new PDFSegmentLognormal(file, sMin, sMax, rng)
                        );
                    }
                    catch (const std::exception& error)
                    {
                        parseError("invalid lognormal segment",
                            error.what(), fileName);
                    }
                } 
                else if (tok[1] == "normal")
                {
                    // Call the file-based constructor
                    try {
                        seg.push_back(
                            new PDFSegmentNormal(file, sMin, sMax, rng)
                        );
                    }
                    catch (const std::exception& error)
                    {
                        parseError("invalid normal segment",
                            error.what(), fileName);
                    }
                } 
                else if (tok[1] == "powerlaw")
                {
                    // Call the file-based constructor
                    try {
                        seg.push_back(
                            new PDFSegmentPowerlaw(file, sMin, sMax, rng)
                        );
                    }
                    catch (const std::exception& error)
                    {
                        parseError("invalid powerlaw segment",
                            error.what(), fileName);
                    }
                } 
                else if (tok[1] == "schechter")
                {
                    // Call the file-based constructor
                    try {
                        seg.push_back(
                            new PDFSegmentSchechter(file, sMin, sMax, rng)
                        );
                    }
                    catch (const std::exception& error)
                    {
                        parseError("invalid Schechter segment",
                            error.what(), fileName);
                    }
                }

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
                    if (tok[1] == "stop_nearest") method = samplingMethods::stopNearest;
                    else if (tok[1] == "stop_before") method = samplingMethods::stopBefore;
                    else if (tok[1] == "stop_after") method = samplingMethods::stopAfter;
                    else if (tok[1] == "stop_50") method = samplingMethods::stop50;
                    else if (tok[1] == "number") method = samplingMethods::number;
                    else if (tok[1] == "poisson") method = samplingMethods::poisson;
                    else if (tok[1] == "sorted") method = samplingMethods::sorted;
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

        // Consistency check: make sure number of segments found
        // matches number expected from breakpoints
        if (seg.size() != breakpoints.size() - 1)
        {
            parseError("mismatch between number of breakpoints "
                "and number of segments", "", fileName);
        }

        // Now figure out correct weight to assign to all segments
        // to ensure that segments are continuous
        std::vector<double> wgt(seg.size());
        wgt[0] = 1.0;
        for (size_t i = 1; i < seg.size(); i++)
        {
            PDFSegment &s = *(seg[i]);
            PDFSegment &sPrev = *(seg[i-1]);
            wgt[i] = wgt[i-1] * sPrev(breakpoints[i]) /
                s(breakpoints[i]);
        }

        // Construct the final PDF to return
        bool normalize = true;
        bool ownSegments = true;
        return PDF(seg, wgt, rng, method, normalize, ownSegments);

    }
}