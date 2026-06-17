/**
 * @file PDFCommons.hpp
 * @author Mark Krumholz
 * @brief Common namespaced and enums used by PDFs
 * @details
 * This file provides common definitions of enums and namespaces
 * used by modules for handlng PDFs.
 * @date 2024-06-12
 */

 #ifndef PDFCOMMONS_HPP
 #define PDFCOMMONS_HPP

 #include <pcg_random.hpp>
 #include <cstdint>

/**
 * @brief A namespace to hold PDFs and quantities related to them.
 */
namespace pdfs {

    /**
     * @brief Namespace to hold sampling methods
     */
    enum class SamplingMethods : std::uint8_t {
        stopNearest,    /**< Stop-nearest sampling */
        stopBefore,     /**< Stop-before sampling */
        stopAfter,      /**< Stop-after sampling */
        stop50,         /**< Stop-50/50 sampling */
        number,         /**< Exact number sampling */
        poisson,        /**< Poisson sampling */
        sorted          /**< Sorted sampling */
    };
    
    /**
     * @brief Enum to hold PDF file formats
     */
    enum class FileFormats : std::uint8_t {
        basic,       /**< Basic format */
        advanced     /**< Advanced format */
    };

    using RngType = pcg64;  /**< Alias for random number type */

} // namespace pdfs

#endif // PDFCOMMONS_HPP