/**
 * @file PDF.hpp
 * @author Mark Krumholz
 * @brief Class to represent a PDF.
 * @details
 * This class represents a PDF composed of one or more
 * segments. This PDF can be properly normalized (so that
 * the integral over all segments is unity) or not. It
 * provides methods for sampling from the PDF and computing
 * various integrals and expectation values from it.
 * @date 2024-06-12
 */

#ifndef PDF_HPP
#define PDF_HPP

#include <array>
#include <ranges>
#include <valarray>
#include <vector>
#include "PDFSegment.hpp"

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
     * @class PDF
     * @brief A class representing a PDF composed of one or more segments
     * @tparam N Number of segments in the PDF
     * @details
     * This class represents a PDF composed of one or more
     * segments. This PDF can be properly normalized (so that
     * the integral over all segments is unity) or not. It
     * provides methods for sampling from the PDF and computing
     * various integrals and expectation values from it.
     */
    template<unsigned int N> class PDF {
    public:

        // Constructors and destructor
        /**
         * @brief Construct from array of segments and weights
         * @param seg Array of segments of segments
         * @param wgt Array of weights; all elements must be positive
         * @param method Sampling method
         * @param normalize Normalize or not?
         */
        PDF(std::array<PDFSegment*, N> seg, 
            std::array<double, N> wgt,
            samplingMethods::method meth = samplingMethods::stopNearest,
            bool normalize = true) : 
            seg_(seg),
            wgt_(wgt.data(), wgt.size()),
            method_(meth),
            normalized_(normalize)
        {
            if (wgt_.min() <= 0) { // Safety check
                    throw std::runtime_error("PDF: elements of wgt must be non-negative");                    
            }
            if (normalize) {
                normalizePDF();
            }
            sMin_ = seg[0]->getMin();
            sMax_ = seg[0]->getMax();
            for (auto s : seg_) {
                sMin_ = std::min(sMin_, s->getMin());
                sMax_ = std::max(sMax_, s->getMax());
            }
        }
        virtual ~PDF() = default;

        // Getters for the lower and upper limits of the segment
        /** @brief Get the lower limit of the segment.
         *  @return The lower limit of the segment.
         */
        auto getMin() const -> double { return sMin_; }
        /** @brief Get the upper limit of the segment.
         *  @return The upper limit of the segment.
         */
        auto getMax() const -> double { return sMax_; }

        // Normalization method
        /**
         * @brief Normalize the PDF
         */
        auto normalizePDF() -> void
        {
            wgt_ /= wgt_.sum();
            normalized_ = true;
        }
        /**
         * @brief Return whether the PDF is normalized
         */
        auto normalized() const -> bool { return normalized_; }

        // Evaluation functions
        /**
         * @brief Evaluate the PDF at a given point.
         * @param x The point at which to evaluate the PDF.
         * @return The value of the PDF at the given point.
         */
        auto operator()(double x) const -> double
        {
            double p = 0.0;
            for (auto const& [s,w] : std::views::zip(seg_,wgt_)) {
                p += w * (*s)(x);
            }
            return p;
        }
        /**
         * @brief Calculate the expectation value of the PDF segment.
         * @param a The lower limit of the range for expectation value calculation; if set to a value <= sMin, will be set to sMin_.
         * @param b The upper limit of the range for expectation value calculation; if set to a value >= sMax, will be set to sMax_.
         * @return The expectation value of the PDF segment.
         */
        auto expectationValue(const double a, const double b) const -> double
        {
            double e = 0.0;
            double wSum = 0.0;
            for (auto const& [s,w] : std::views::zip(seg_,wgt_)) {
                e += w * s->expectationValue(a, b);
                wSum += w;
            }
            return e / wSum;
        }
        /**
         * @brief Calculate the expectation value of the PDF segment over its entire range.
         * @return The expectation value of the PDF segment.
         */
        virtual auto expectationValue() const -> double
        {
            return expectationValue(sMin_, sMax_);
        }

    protected:

        std::array<PDFSegment*, N> seg_; /**< Segments in the PDF */
        std::valarray<double> wgt_;     /**< Weights of segments */
        double sMin_;                   /**< PDF lower limit */
        double sMax_;                   /**< PDF upper limit */
        samplingMethods::method method_; /**< Sampling method for this PDF */
        bool normalized_;               /**< Is this PDF properly normalized? */
    };

}

#endif // PDF_HPP
