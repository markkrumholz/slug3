/**
 * @file PDFFileParser.cpp
 * @author Mark Krumholz
 * @brief Methods to parse PDF descriptor files
 * @details
 * This file provides a method to parse PDF files
 * and construct PDF objects based on them.
 * @date 2024-06-14
 */

#include "../utils/ParseUtils.hpp"
#include "PDF.hpp"
#include "PDFCommons.hpp"
#include "PDFFileParser.hpp"
#include "PDFSegment.hpp"
#include "PDFSegmentDelta.hpp"
#include "PDFSegmentExponential.hpp"
#include "PDFSegmentLognormal.hpp"
#include "PDFSegmentNormal.hpp"
#include "PDFSegmentPowerlaw.hpp"
#include "PDFSegmentSchechter.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>



// Little utility function to handle errors
[[noreturn ]]
static void parseError(const std::string& err,   // NOLINT misc-use-anonymous-namespace
    const std::string& line,
    const std::string& fileName)
{
    std::string errMsg = "parsePDFDescriptor: " +
        err + "; file " + fileName;
    if (!line.empty()) { errMsg += ", line " + line; }
    throw std::runtime_error(errMsg);
}


// Function to parse a single segment
static void parseSegment(const std::string& fileName,  // NOLINT misc-use-anonymous-namespace
    const pdfs::FileFormats fmt,
    const std::string& line,
    const std::vector<std::string>& tok, 
    const std::vector<double>& breakpoints,
    const size_t& breakpointPtr, 
    std::ifstream& file,
    std::vector<std::unique_ptr<pdfs::PDFSegment> >& seg,
    std::vector<double>& wgt)
{
    if (tok.at(0) != "type" || tok.size() != 2)
    {
        parseError("expected 'type TYPE'", line, fileName);
    }

    // If we are in basic mode, extract lower and upper limits
    // for this segment
    double sMin = 0.0;
    double sMax = 0.0;
    if (fmt == pdfs::FileFormats::basic)
    {
        if (breakpoints.size() < breakpointPtr + 2)
        {
            parseError("too few breakpoints for number of segments",
                line, fileName);
        }
        sMin = breakpoints.at(breakpointPtr);
        sMax = breakpoints.at(breakpointPtr+1);
    }

    // Now call the file-based constructor for the appropriate
    // segment type
    double w = 0.0;
    try
    {
        if (tok.at(1) == "delta")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentDelta>
                (file, fmt, sMin, sMax, w));
        }
        else if (tok.at(1) == "exponential")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentExponential>
                (file, fmt, sMin, sMax, w)
            );
        }
        else if (tok.at(1) == "lognormal")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentLognormal>
                (file, fmt, sMin, sMax, w)
            );
        }
        else if (tok.at(1) == "normal")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentNormal>
                (file, fmt, sMin, sMax, w)
            );
        }
        else if (tok.at(1) == "powerlaw")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentPowerlaw>
                (file, fmt, sMin, sMax, w)
            );
        }
        else if (tok.at(1) == "schechter")
        {
            seg.emplace_back(
                std::make_unique<pdfs::PDFSegmentSchechter>
                (file, fmt, sMin, sMax, w)
            );
        } else {
            parseError("unknown segment type " + tok.at(1),
                line, fileName);
        }
    }
    catch (const std::exception& error)
    {
        if (strlen(error.what()) == 0)
        {
            parseError("reached end of file before "
                "completing segment " + tok.at(1),
                line, fileName);
        }
        else
        {
            parseError("invalid options for segment type " + tok.at(1),
                line, fileName);                        
        }
    }

    // In advanced format mode, store weight
    if (fmt == pdfs::FileFormats::advanced) { wgt.push_back(w); }

}

// Parse method
static auto parseMethod(const std::string& fileName,  // NOLINT misc-use-anonymous-namespace
    const std::string& line,
    std::vector<std::string>& tok) -> pdfs::SamplingMethods
{

    if (tok.at(1) == "stop_nearest") { return pdfs::SamplingMethods::stopNearest; }
    if (tok.at(1) == "stop_before") { return pdfs::SamplingMethods::stopBefore; }
    if (tok.at(1) == "stop_after") { return pdfs::SamplingMethods::stopAfter; }
    if (tok.at(1) == "stop_50") { return pdfs::SamplingMethods::stop50; }
    if (tok.at(1) == "number") { return pdfs::SamplingMethods::number; }
    if (tok.at(1) == "poisson") { return pdfs::SamplingMethods::poisson; }
    if (tok.at(1) == "sorted_sampling") { return pdfs::SamplingMethods::sorted; }
    // Error if we get here
    parseError("unknown sampling method", line, fileName);
}

// Method to set weights in basic mode
static void computeWgt(const std::vector<std::unique_ptr<pdfs::PDFSegment> >& seg,  // NOLINT misc-use-anonymous-namespace
    std::vector<double>& breakpoints,
    std::vector<double>& wgt)
{
    wgt.resize(seg.size());
    wgt[0] = 1.0; // NOLINT cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
    for (size_t i = 1; i < seg.size(); i++)
    {
        pdfs::PDFSegment &s = *(seg[i]); // NOLINT cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
        pdfs::PDFSegment &sPrev = *(seg[i-1]); // NOLINT cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
        wgt[i] = wgt[i-1] * sPrev(breakpoints[i]) / s(breakpoints[i]); // NOLINT cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
    }
}

// Utility function to extract breakpoints
static void parseBreakpoints(const std::string& fileName,  // NOLINT misc-use-anonymous-namespace
    const std::string& line,
    std::vector<std::string>& tokens,
    std::vector<double>& breakpoints)
{
    tokens.erase(tokens.begin());  // Remove the token "breakpoints"
    for (auto const& t : tokens)
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
    if (!std::ranges::is_sorted(breakpoints))
    {
        parseError("breakpoints must be non-decreasing",
            "", fileName);
    }
    if (breakpoints.size() < 2)
    {
        parseError("at least two breakpoints required",
            "", fileName);
    }
}

// Method to parse main body
static auto
parseBody(const std::string& fileName,  // NOLINT misc-use-anonymous-namespace
    const pdfs::FileFormats fmt,
    std::ifstream& file,
    std::vector<double>& breakpoints) ->
    std::tuple<std::vector<std::unique_ptr<pdfs::PDFSegment> >,
        std::vector<double>,
        pdfs::SamplingMethods>
{
    // Allocate holders for segments and weights
    std::vector<std::unique_ptr<pdfs::PDFSegment> > seg;
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
    std::string line;
    bool inSegment = false;
    size_t breakpointPtr = 0;
    pdfs::SamplingMethods method = pdfs::SamplingMethods::stopNearest;
    while (std::getline(file, line))
    {

        // Trim and tokenize the line
        line = utils::trim(line);
        if (line.empty()) { continue; } // Whitespace-only line; skip
        auto tok = utils::tokenize(line);

        // Check if we are currently a segment
        if (inSegment)
        {
            // Call the segment parser
            parseSegment(fileName, fmt, line, tok,
                breakpoints, breakpointPtr, file, seg, wgt);

            // Increment breakpoint counter
            breakpointPtr++;

            // Mark segment done
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

            if (tok.at(0) == "segment" && tok.size() == 1)
            {
                // Start of a new segment
                inSegment = true;
                continue;
            }
            
            if (tok.at(0) == "method" && tok.size() == 2)
            {
                method = parseMethod(fileName, line, tok);
                continue;
            }

            // Error if we reach here
            parseError("unknown command", line, fileName);
        }
    }

    // Return result
    return { std::move(seg), wgt, method };
}

namespace pdfs {

    // General parser to start and decide if file is basic or advanced
    auto parsePDFDescriptor(const std::string& fileName) -> PDF
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
            if (line.empty()) { continue; } // Whitespace-only line
            break;  // If we are here, line is non-empty
        }

        // Tokenize the line
        auto tokens = utils::tokenize(line);
 
        // The first token must be either "advanced" (to indicate an
        // advanced mode file) or "breakpoints" followed by at least
        // two numbers (to indicate a basic mode file); check
        // that it is indeed one of these two
        FileFormats fmt = FileFormats::basic;
        if (tokens.at(0) == "breakpoints" && tokens.size() > 2)
        {
            fmt = FileFormats::basic;
        } 
        else if (tokens.at(0) == "advanced") 
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
            parseBreakpoints(fileName, line, tokens, breakpoints);
        }

        // Parse main body
        auto [seg, wgt, method] = parseBody(fileName, fmt, file, breakpoints);

        // Close file
        file.close();

        // Check that we have the right number of segments
        if (fmt == FileFormats::basic)
        {
            if (seg.size() + 1 != breakpoints.size())
            {
                parseError("found inconsistent number of segments "
                    "and breakpoints", "", fileName);
            }
        }
        else
        {
            if (seg.empty())
            {
                parseError("need at least one segment", "", fileName);
            }
        }

        // Final step depends on format
        if (fmt == FileFormats::basic)
        {
            // Basic format: calculate weights to make
            // segments continuous
            computeWgt(seg, breakpoints, wgt);

            // Create and return normalized PDF
            const bool normalize = true;
            return { std::move(seg), wgt, method, normalize };
        }
        else  // NOLINT readability-else-after-return
        {
            // Advanced mode, so don't do any normalization,
            // just return
            const bool normalize = false;
            return { std::move(seg), wgt, method, normalize };            
        }
    }

} // namespace pdfs