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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include "PDF.hpp"

namespace pdfs 
{
    /**
     * @brief Construct a PDF objects from a descriptor file
     * @param fileName Name of the file
     * @param rng The random number engine for the PDF object
     * @returns A PDF objects constructed from the file
     */
    auto parsePDFDescriptor(const std::string& fileName, 
        rngType &rng) -> PDF;
    /**
     * @brief Construct a PDF object from a basic descriptor file
     * @param fileName Name of the file
     * @param rng The random number engine for the PDF object
     * @param breakpointLine The line containing breakpoint information
     * @param file Open file object to finish being read
     */
    auto parsePDFDescriptorBasic(const std::string& fileName, 
        rngType &rng,
        std::string& breakpointLine,
        std::ifstream& file) -> PDF;
    /**
     * @brief Construct a PDF object from an advanced descriptor file
     * @param fileName Name of the file
     * @param rng The random number engine for the PDF object
     * @param file Open file object to finish being read
     */
    //auto parsePDFDescriptorAdvanced(const std::string& fileName, 
    //    rngType &rng,
    //    std::ifstream& file) -> PDF;
}

#endif // PDFFILEPARSER_HPP