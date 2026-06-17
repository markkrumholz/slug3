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

namespace pdfs 
{
    /**
     * @brief Construct a PDF objects from a descriptor file
     * @param fileName Name of the file
     * @returns A PDF objects constructed from the file
     */
    auto parsePDFDescriptor(const std::string& fileName) -> PDF;
} // namespace pdfs

#endif // PDFFILEPARSER_HPP