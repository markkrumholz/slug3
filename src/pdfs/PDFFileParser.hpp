/**
 * @file PDFFileParser.hpp
 * @author Mark Krumholz
 * @brief Methods to parse PDF descriptor files
 * @details
 * This file provides a method to parse PDF files
 * and construct PDF objects based on them.
 * @date 2024-06-14
 */

#ifndef PDFFILEPARSER_HPP
#define PDFFILEPARSER_HPP

#include "PDF.hpp"
#include <string>
#include <toml.hpp>

namespace pdfs
{
    /**
     * @brief Construct a PDF objects from a descriptor file
     * @param fileName Name of the file
     * @returns A PDF objects constructed from the file
     */
    auto parsePDFDescriptor(const std::string& fileName) -> PDF;

    /**
     * @brief Construct a PDF object from a toml-format PDF descriptor
     * @param deck A toml table holding the PDF descriptor
     * @returns A PDF object constructed from deck
     * @throws std::runtime_error if deck is not a valid PDF descriptor
     * @details
     * A toml-format counterpart to parsePDFDescriptor(), extending the
     * same basic/advanced PDF descriptor format to toml. The table's
     * top-level keys form a header, analogous to the first lines of a
     * parsePDFDescriptor() file:
     *   - "format" (optional string, "basic" or "advanced"; defaults
     *     to "basic")
     *   - "breakpoints" (an array of at least two numbers; required if
     *     format is "basic", and not allowed if format is "advanced")
     *   - "method" (optional string, one of the same sampling method
     *     names parsePDFDescriptor() itself recognizes; defaults to
     *     stop-nearest sampling if omitted)
     * Every other top-level key must be a table named segmentN (N =
     * 1, 2, ...), numbered sequentially with no gaps, one per PDF
     * segment, each analogous to one "segment"/"type TYPE"/... block
     * in a parsePDFDescriptor() file: a required "type" key naming one
     * of the recognized PDFSegment types, plus whatever keys that
     * segment type's own toml-based constructor requires. If format
     * is "basic", segmentN's own [sMin, sMax) range is taken from
     * breakpoints[N-1]/breakpoints[N], the number of segments must be
     * exactly breakpoints.size() - 1, and segment weights are computed
     * (as parsePDFDescriptor() itself does) so adjacent segments agree
     * at their shared breakpoint; if "advanced", each segmentN table
     * must additionally specify its own sMin/sMax/weight directly, and
     * no such continuity is enforced.
     */
    auto parsePDFToml(const toml::table& deck) -> PDF;
} // namespace pdfs

#endif // PDFFILEPARSER_HPP