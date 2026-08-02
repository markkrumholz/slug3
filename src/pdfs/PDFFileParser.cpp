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
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <tuple>
#include <utility>
#include <vector>



// Little utility function to handle errors
[[noreturn ]]
static void parseError(const std::string& err,
    const std::string& line,
    const std::string& fileName)
{
    std::string errMsg = "parsePDFDescriptor: " +
        err + "; file " + fileName;
    if (!line.empty()) { errMsg += ", line " + line; }
    throw std::runtime_error(errMsg);
}


// Function to parse a single segment
static void parseSegment(const std::string& fileName,
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
static auto parseMethod(const std::string& fileName,
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
static void computeWgt(const std::vector<std::unique_ptr<pdfs::PDFSegment> >& seg,
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
static void parseBreakpoints(const std::string& fileName,
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
parseBody(const std::string& fileName,
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

// Little utility function to handle errors in a toml-format PDF
// descriptor -- mirrors parseError() above, but a toml::table has no
// associated file name or line number of its own to report
[[noreturn]]
static void parseTomlError(const std::string& err)
{
    throw std::runtime_error("parsePDFToml: " + err);
}

// Map a sampling-method string to pdfs::SamplingMethods, mirroring
// parseMethod() above; method is std::nullopt if the toml "method"
// key was present but not a string, which -- like every unrecognized
// string -- falls through to the error at the end
static auto parseMethodToml(const std::optional<std::string>& method) -> pdfs::SamplingMethods
{
    if (method == "stop_nearest") { return pdfs::SamplingMethods::stopNearest; }
    if (method == "stop_before") { return pdfs::SamplingMethods::stopBefore; }
    if (method == "stop_after") { return pdfs::SamplingMethods::stopAfter; }
    if (method == "stop_50") { return pdfs::SamplingMethods::stop50; }
    if (method == "number") { return pdfs::SamplingMethods::number; }
    if (method == "poisson") { return pdfs::SamplingMethods::poisson; }
    if (method == "sorted_sampling") { return pdfs::SamplingMethods::sorted; }
    parseTomlError("'method' must be a string naming a recognized sampling method");
}

namespace
{
    // Result of parsing a toml PDF descriptor's header -- every
    // top-level key other than the segmentN tables -- see
    // parseTomlHeader() and parsePDFToml()'s own doc comment
    struct PDFTomlHeader
    {
        pdfs::FileFormats fmt_ = pdfs::FileFormats::basic; /**< Basic or advanced format */
        std::vector<double> breakpoints_; /**< Segment breakpoints, only set (and only meaningful) if fmt_ is basic */
        pdfs::SamplingMethods method_ = pdfs::SamplingMethods::stopNearest; /**< Sampling method to use */
    };
} // namespace

// Parse the header of a toml PDF descriptor -- every top-level key
// other than the segmentN tables -- and determine the format,
// breakpoints (if any), and sampling method. See parsePDFToml's own
// doc comment for the exact rules "format", "breakpoints", and
// "method" must follow.
static auto parseTomlHeader(const toml::table& deck) -> PDFTomlHeader
{
    PDFTomlHeader header;

    // "format": optional, defaults to basic
    if (const auto formatNode = deck.at_path("format"))
    {
        const auto formatStr = formatNode.value<std::string>();
        if (formatStr == "basic") { header.fmt_ = pdfs::FileFormats::basic; }
        else if (formatStr == "advanced") { header.fmt_ = pdfs::FileFormats::advanced; }
        else { parseTomlError("'format' must be 'basic' or 'advanced'"); }
    }

    // "breakpoints": required (and parsed) iff fmt is basic; an error
    // for it to be present at all if fmt is advanced
    const auto breakpointsNode = deck.at_path("breakpoints");
    const bool hasBreakpoints = static_cast<bool>(breakpointsNode);
    if (hasBreakpoints)
    {
        const auto* arr = breakpointsNode.as_array();
        if (arr == nullptr) { parseTomlError("'breakpoints' must be an array of numbers"); }
        arr->for_each([&header](auto&& el) -> void {
            if constexpr (toml::is_number<decltype(el)>)
            {
                header.breakpoints_.push_back(static_cast<double>(el.get()));
            }
        });
        if (header.breakpoints_.size() != arr->size())
        {
            parseTomlError("'breakpoints' must be an array of numbers");
        }
        if (header.breakpoints_.size() < 2)
        {
            parseTomlError("'breakpoints' must have at least two elements");
        }
    }
    if (header.fmt_ == pdfs::FileFormats::basic && !hasBreakpoints)
    {
        parseTomlError("'breakpoints' is required when format is 'basic'");
    }
    if (header.fmt_ == pdfs::FileFormats::advanced && hasBreakpoints)
    {
        parseTomlError("'breakpoints' must not be given when format is 'advanced'");
    }

    // "method": optional, defaults to stop-nearest sampling
    if (const auto methodNode = deck.at_path("method"))
    {
        header.method_ = parseMethodToml(methodNode.value<std::string>());
    }

    return header;
}

// Find every top-level segmentN table in a toml PDF descriptor,
// validate that their names are numbered sequentially starting at 1
// with no gaps, and return their numbers in ascending order (so the
// N'th entry of the result is always N + 1). Also validates that
// every top-level table (as opposed to a plain key like "format" or
// "breakpoints") is in fact named segmentN for some N -- any other
// table name is an error.
static auto findSegmentNumbers(const toml::table& deck) -> std::vector<int>
{
    static const std::string prefix = "segment";

    std::vector<int> numbers;
    for (const auto& [key, val] : deck)
    {
        if (!val.is_table()) { continue; }

        const std::string keyStr(key.str());
        const bool matchesPrefix = keyStr.size() > prefix.size() &&
            keyStr.compare(0, prefix.size(), prefix) == 0;
        const std::string numPart = matchesPrefix ? keyStr.substr(prefix.size()) : "";
        const bool allDigits = !numPart.empty() &&
            std::ranges::all_of(numPart,
                [](const unsigned char c) -> bool { return std::isdigit(c) != 0; });
        if (!matchesPrefix || !allDigits)
        {
            parseTomlError("table '" + keyStr +
                "' does not match the expected 'segmentN' naming");
        }

        try
        {
            numbers.push_back(std::stoi(numPart));
        }
        catch (const std::exception&)
        {
            parseTomlError("table '" + keyStr +
                "' does not match the expected 'segmentN' naming");
        }
    }

    std::ranges::sort(numbers);
    for (std::size_t i = 0; i < numbers.size(); i++)
    {
        if (numbers.at(i) != static_cast<int>(i) + 1)
        {
            parseTomlError("segment tables must be numbered sequentially "
                "as segment1, segment2, ... with no gaps");
        }
    }
    return numbers;
}

// Parse a single segmentN table, fully analogous to parseSegment()
// above: reads the required "type" key, then calls the toml-based
// constructor of the corresponding PDFSegment type, passing it node,
// fmt, sMin, sMax, and a weight to fill in. sMin/sMax are inputs if
// fmt is basic (the segment's own slice of the breakpoints array) and
// outputs if fmt is advanced (in which case the caller passes 0.0 for
// both, mirroring parseSegment()'s own identical convention).
static void parseSegmentToml(
    const std::string& key,
    const pdfs::FileFormats fmt,
    const toml::node_view<const toml::node>& node,
    double sMin,
    double sMax,
    std::vector<std::unique_ptr<pdfs::PDFSegment> >& seg,
    std::vector<double>& wgt)
{
    const auto type = node.at_path("type").value<std::string>();
    if (!type.has_value())
    {
        parseTomlError(key + ": expected a string 'type' key");
    }

    double w = 0.0;
    try
    {
        if (type == "delta")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentDelta>(node, fmt, sMin, sMax, w));
        }
        else if (type == "exponential")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentExponential>(node, fmt, sMin, sMax, w));
        }
        else if (type == "lognormal")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentLognormal>(node, fmt, sMin, sMax, w));
        }
        else if (type == "normal")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentNormal>(node, fmt, sMin, sMax, w));
        }
        else if (type == "powerlaw")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentPowerlaw>(node, fmt, sMin, sMax, w));
        }
        else if (type == "schechter")
        {
            seg.emplace_back(std::make_unique<pdfs::PDFSegmentSchechter>(node, fmt, sMin, sMax, w));
        }
        else
        {
            parseTomlError(key + ": unknown segment type '" + type.value() + "'");
        }
    }
    catch (const std::exception& error)
    {
        parseTomlError(key + ": invalid options for segment type '" +
            type.value() + "': " + error.what());
    }

    // In advanced format mode, store weight
    if (fmt == pdfs::FileFormats::advanced) { wgt.push_back(w); }
}

// Parse the body (the segmentN tables) of a toml PDF descriptor,
// fully analogous to parseBody() above -- header must already have
// been parsed (and validated) via parseTomlHeader()
static auto parseTomlBody(const toml::table& deck,
    const pdfs::FileFormats fmt,
    const std::vector<double>& breakpoints) ->
    std::pair<std::vector<std::unique_ptr<pdfs::PDFSegment> >, std::vector<double> >
{
    const auto segmentNumbers = findSegmentNumbers(deck);

    // Check that we have the right number of segments
    if (fmt == pdfs::FileFormats::basic)
    {
        if (segmentNumbers.size() + 1 != breakpoints.size())
        {
            parseTomlError("number of segment tables is inconsistent "
                "with the number of breakpoints");
        }
    }
    else if (segmentNumbers.empty())
    {
        parseTomlError("need at least one segment");
    }

    std::vector<std::unique_ptr<pdfs::PDFSegment> > seg;
    std::vector<double> wgt;
    seg.reserve(segmentNumbers.size());

    for (const int n : segmentNumbers)
    {
        const std::string key = "segment" + std::to_string(n);
        const double sMin = (fmt == pdfs::FileFormats::basic) ?
            breakpoints.at(static_cast<std::size_t>(n) - 1) : 0.0;
        const double sMax = (fmt == pdfs::FileFormats::basic) ?
            breakpoints.at(static_cast<std::size_t>(n)) : 0.0;
        parseSegmentToml(key, fmt, deck.at_path(key), sMin, sMax, seg, wgt);
    }

    return { std::move(seg), std::move(wgt) };
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

    // Toml-format counterpart to parsePDFDescriptor() -- see this
    // function's own doc comment (PDFFileParser.hpp) for the exact
    // format. Header parsing (format/breakpoints/method) is handled
    // by parseTomlHeader(), body parsing (the segmentN tables) by
    // parseTomlBody(), fully analogous to parsePDFDescriptor() above.
    auto parsePDFToml(const toml::table& deck) -> PDF
    {
        auto header = parseTomlHeader(deck);
        auto [seg, wgt] = parseTomlBody(deck, header.fmt_, header.breakpoints_);

        // Final step depends on format
        if (header.fmt_ == FileFormats::basic)
        {
            // Basic format: calculate weights to make
            // segments continuous
            computeWgt(seg, header.breakpoints_, wgt);

            // Create and return normalized PDF
            const bool normalize = true;
            return { std::move(seg), wgt, header.method_, normalize };
        }

        // Advanced mode, so don't do any normalization, just return
        const bool normalize = false;
        return { std::move(seg), wgt, header.method_, normalize };
    }

} // namespace pdfs