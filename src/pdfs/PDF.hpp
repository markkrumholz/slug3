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
#include <random>
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
         * @tparam SC Any contiguous container of PDFSegment pointers
         * @tparam WC Any contiguous container of doubles
         * @param seg Container of pointers to segments
         * @param wgt Container of weights; must have same size as seg, all elements must be positive
         * @param rng Reference to the random number generator to be used for sampling.
         * @param method Sampling method
         * @param normalize Normalize or not?
         */
        template <typename SC, typename WC>
        PDF(SC seg,
            WC wgt,
            rngType &rng,
            samplingMethods::method meth = samplingMethods::stopNearest,
            bool normalize = true) :
            seg_(std::begin(seg), std::end(seg)),
            wgt_(std::data(wgt), std::size(wgt)),
            rng_(rng),
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
                if (s->getMin() < x && s->getMax() >= x) {
                    p += w * (*s)(x);
                }
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
                double wi = w * s->integral(a, b);
                e += wi * s->expectationValue(a, b);
                wSum += wi;
            }
            return e / wSum;
        }
        /**
         * @brief Calculate the expectation value of the PDF segment over its entire range.
         * @return The expectation value of the PDF segment.
         */
        auto expectationValue() const -> double
        {
            return expectationValue(sMin_, sMax_);
        }
        /**
         * @brief Calculate the integral of the PDF over a specified range.
         * @param a The lower limit of the range for integral calculation; if set to a value <= sMin, will be set to sMin_.
         * @param b The upper limit of the range for integral calculation; if set to a value >= sMax, will be set to sMax_.
         * @return The integral of the PDF over the specified range.
         */
        auto integral(const double a, const double b) const -> double {
            if (a >= b) {
                return 0.0; // Interval is empty
            }
            double sum = 0.0;
            for (auto const& [s,w] : std::views::zip(seg_,wgt_)) {
                sum += w * s->integral(a,b);  // Segment overlaps range
            }
            return sum;
        }
        /**
         * @brief Calculate the integral of the PDF over its full range.
         * @return The integral of the PDF segment over the specified range.
         */
        auto integral() const -> double { return wgt_.sum(); }

        // Drawing functions
        /**
         * @brief Sample a random value from the PDF within a specified range.
         * @param a The lower limit of the sampling range (should be >= sMin_).
         * @param b The upper limit of the sampling range (should be <= sMax_).
         * @return A random value drawn from the PDF within the specified range.
         */
        auto draw(const double a, const double b) const -> double {
            std::vector<double> wLim(wgt_.size());
            for (auto const& [wL,s,w] : std::views::zip(wLim,seg_,wgt_)) {
                wL = w * s->integral(a,b);
                std::cout << "wL = " << wL << std::endl;
            }
            std::discrete_distribution<int> dist(wLim.begin(), wLim.end());
            return seg_[dist(rng_)]->draw(a,b);
        }
        /**
         * @brief Sample a random value from the PDF.
         * @return A random value drawn from the PDF within the specified range.
         */
        auto draw() const -> double { return draw(sMin_, sMax_); }
        /**
         * @brief Sample nDraw random values from the PDF.
         * @param nDraw Number of samples to draw.
         * @return A vector of random values drawn from the PDF within the specified range.
         */
        auto draw(unsigned int nDraw) const -> std::vector<double> 
        { 
            std::vector<double> result;
            for (unsigned int i = 0; i < nDraw; i++) {
                result.push_back(draw(sMin_, sMax_));
            }
            return result;
        }
        /**
         * @brief Sample nDraw random values from the PDF within the specified range.
         * @param nDraw Number of samples to draw.
         * @param a The lower limit of the sampling range (should be >= sMin_).
         * @param b The upper limit of the sampling range (should be <= sMax_).
         * @return A vector of nDraw random values drawn from the PDF within the specified range.
        */
        auto draw(unsigned int nDraw, double a, double b) const -> std::vector<double>
        {
            std::vector<double> result;
            for (unsigned int i = 0; i < nDraw; i++) {
                result.push_back(draw(a, b));
            }
            return result;        
        }  

    protected:

        std::vector<PDFSegment *> seg_; /**< Segments in the PDF */
        std::valarray<double> wgt_;     /**< Weights of segments */
        rngType& rng_;                  /**< Random engine */
        double sMin_;                   /**< PDF lower limit */
        double sMax_;                   /**< PDF upper limit */
        samplingMethods::method method_; /**< Sampling method for this PDF */
        bool normalized_;               /**< Is this PDF properly normalized? */
    };

}

#endif // PDF_HPP
