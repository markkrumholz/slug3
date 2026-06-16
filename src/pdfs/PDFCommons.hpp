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

 #include "../pcg-cpp/include/pcg_random.hpp"
 
/**
 * @brief A namespace to hold PDFs and quantities related to them.
 */
namespace pdfs {

    /**
     * @brief Namespace to hold sampling methods
     */
    namespace samplingMethods {

        /**
        * @brief An enum of known sampling methods
        */
        typedef enum {
            stopNearest,    /**< Stop-nearest sampling */
            stopBefore,     /**< Stop-before sampling */
            stopAfter,      /**< Stop-after sampling */
            stop50,         /**< Stop-50/50 sampling */
            number,         /**< Exact number sampling */
            poisson,        /**< Poisson sampling */
            sorted          /**< Sorted sampling */
        } method;

    }
    
    /**
     * @brief Namespace to hold PDF file formats
     */
    namespace fileFormats {
        typedef enum {
            basic, /** Basic mode */
            advanced
        } format;
    }

    typedef pcg64 rngType;  /**< Alias for random number type */
}

#endif // PDFCOMMONS_HPP