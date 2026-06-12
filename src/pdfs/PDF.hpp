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

#include <ranges>
#include <vector>
#include "PDFSegment.hpp"

/**
 * @brief A namespace to hold PDFs and quantities related to them.
 */
namespace pdfs {

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
        sortedSampling  /**< Sorted sampling */
    } samplingMethod;

    /**
     * @class PDF
     * @brief A class representing a PDF composed of one or more segments
     * @details
     * This class represents a PDF composed of one or more
     * segments. This PDF can be properly normalized (so that
     * the integral over all segments is unity) or not. It
     * provides methods for sampling from the PDF and computing
     * various integrals and expectation values from it.
     */
    class PDF {
    public:

        // Constructors and destructor
        /**
         * @brief Construct from array of segments and weights
         * @param seg Vector of segments
         * @param wgt Vector of weights; must have same number of elements as seg
         * @param method Sampling method
         * @param normalize Normalize or not?
         */
        PDF(std::vector<PDFSegment> seg, 
            std::vector<double> wgt,
            samplingMethod method = stopNearest,
            bool normalize = true) : 
            seg_(seg),
            wgt_(wgt),
            method_(method),
            normalized_(normalize)
        {
            if (seg.size() < 1 || seg.size() != wgt.size()) {
                throw std::runtime_error("PDF: seg and wgt must have equal size >= 1.");
            }
            if (normalize) {
                normalizePDF();
            }
            sMin_ = seg[0].getMin();
            sMax_ = seg[0].getMax();
            for (auto s : seg) {
                sMin_ = std::min(sMin_, s.getMin());
                sMax_ = std::max(sMax_, s.getMax());
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
            double sumWgt = 0;
            for (auto w : wgt_) {
                sumWgt += w;
            }
            for (auto w : wgt_) {
                w /= sumWgt;
            }
            normalized_ = true;
        }
        /**
         * @brief Return whether the PDF is normalized
         */
        auto normalized() const -> bool\
        {
            return normalized_;
        }

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
                p += w * s(x);
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
                e += w * s.expectationValue(a, b);
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

        std::vector<PDFSegment> seg_; /**< Segments in the PDF */
        std::vector<double> wgt_;     /**< Weights of segments */
        double sMin_;                 /**< PDF lower limit */
        double sMax_;                 /**< PDF upper limit */
        samplingMethod method_;       /**< Sampling method for this PDF */
        bool normalized_;             /**< Is this PDF properly normalized? */
    };

}

#endif // PDF_HPP
