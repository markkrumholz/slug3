/**
 * @file PDFCommons.hpp
 * @author Mark Krumholz
 * @brief Common namespaced and enums used by PDFs
 * @details
 * This file provides common definitions of enums and namespaces
 * used by modules for handlng PDFs.
 * @date 2024-06-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef PDFCOMMONS_HPP
#define PDFCOMMONS_HPP

#include <cstdint>
#include <stdexcept>
#include <string_view>

/**
 * @brief A namespace to hold PDFs and quantities related to them.
 */
namespace pdfs {

    /**
     * @brief Enum to hold sampling methods
     */
    enum class SamplingMethods : std::uint8_t {
        stopNearest,    /**< Stop-nearest sampling */
        stopBefore,     /**< Stop-before sampling */
        stopAfter,      /**< Stop-after sampling */
        stop50,         /**< Stop-50/50 sampling */
        number,         /**< Exact number sampling */
        poisson,        /**< Poisson sampling */
        sorted,         /**< Sorted sampling */
        none            /**< Invalid dummy method */
    };
    
    /**
     * @brief Enum to hold PDF file formats
     */
    enum class FileFormats : std::uint8_t {
        basic,       /**< Basic format */
        advanced     /**< Advanced format */
    };

    /**
     * @brief Name of a sampling method, as used in PDF descriptors
     * @param method The sampling method
     * @return The name a PDF descriptor's "method" key (or a legacy
     *   descriptor's "method" line) uses for method -- e.g.
     *   "stop_nearest" -- or "none" for the invalid dummy method
     */
    inline auto samplingMethodName(const SamplingMethods method) -> std::string_view
    {
        switch (method)
        {
            case SamplingMethods::stopNearest: return "stop_nearest";
            case SamplingMethods::stopBefore:  return "stop_before";
            case SamplingMethods::stopAfter:   return "stop_after";
            case SamplingMethods::stop50:      return "stop_50";
            case SamplingMethods::number:      return "number";
            case SamplingMethods::poisson:     return "poisson";
            case SamplingMethods::sorted:      return "sorted_sampling";
            case SamplingMethods::none:        return "none";
        }
        throw std::logic_error("samplingMethodName: unrecognized sampling method"); // unreachable; exhaustive switch above
    }

} // namespace pdfs

#endif // PDFCOMMONS_HPP