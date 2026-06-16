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
         * @param ownSegments True if this PDF owns the segments it contains
         */
        template <typename SC, typename WC>
        PDF(SC seg,
            WC wgt,
            rngType &rng,
            samplingMethods::method meth = samplingMethods::stopNearest,
            bool normalize = true,
            bool ownSegments = false) :
            seg_(std::begin(seg), std::end(seg)),
            wgt_(std::data(wgt), std::size(wgt)),
            rng_(rng),
            method_(meth),
            normalized_(normalize),
            ownSegments_(ownSegments)
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
        virtual ~PDF()
        {
            if (ownSegments_)
            {
                // We own the segments, so invoke delete each segment
                for (auto s : seg_) { delete s; }
            }
        }

        // Getters for internal state
        /** @brief Get the lower limit of the segment.
         *  @return The lower limit of the segment.
         */
        auto getMin() const -> double { return sMin_; }
        /** @brief Get the upper limit of the segment.
         *  @return The upper limit of the segment.
         */
        auto getMax() const -> double { return sMax_; }
        /**
         * @brief Get the weights of segments
         * @return The weights of the segments
         */
        auto getWeights() const -> const std::valarray<double>& { return wgt_; }

        // Sampling policy getter and setter
        /**
         * @brief Get sampling method
         * @return The current sampling method
         */
        auto getSampling() const -> samplingMethods::method
        {
            return method_;
        }
        /**
         * @brief Set sampling method
         * @param method Method to set
         */
        void setSampling(samplingMethods::method method)
        {
            method_ = method;
        }

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
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if unspecified).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if unspecified).
         * @return A random value drawn from the PDF within the specified range.
         */
        auto draw(const double a = std::numeric_limits<double>::lowest(),
         const double b = std::numeric_limits<double>::max()) const -> double {
            std::vector<double> wLim(wgt_.size());
            for (auto const& [wL,s,w] : std::views::zip(wLim,seg_,wgt_)) {
                wL = w * s->integral(a,b);
            }
            std::discrete_distribution<int> dist(wLim.begin(), wLim.end());
            return seg_[dist(rng_)]->draw(a,b);
        }
        /**
         * @brief Sample nDraw random values from the PDF within the specified range.
         * @param nDraw Number of samples to draw.
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if not set).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if not set).
         * @return A vector of nDraw random values drawn from the PDF within the specified range.
        */
        auto draw(unsigned int nDraw, 
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max())
            const -> std::vector<double>
        {
            std::vector<double> result;
            for (unsigned int i = 0; i < nDraw; i++) {
                result.push_back(draw(a, b));
            }
            return result;        
        }
        /**
         * @brief Draw a sample from the PDF targeting a fixed total value
         * @param target Target total to draw
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if not set).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if not set).
         */
        auto drawTarget(double target, 
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max()) const 
            -> std::vector<double>
        {
            std::vector<double> sample;
            switch (method_)
            {
                case samplingMethods::stopNearest:
                case samplingMethods::stopBefore:
                case samplingMethods::stopAfter:
                case samplingMethods::stop50:
                {
                    // Stop methods
                    double sum = 0.0;
                    while (true)
                    {
                        auto s = draw(a, b);
                        if (sum + s <= target)
                        {
                            // Taget not yet reached; add to sample and continue
                            sample.push_back(s);
                            sum += s;
                        } else {
                            // Target reached; decide whether to keep last number based on policy
                            std::uniform_real_distribution<double> dist(0.0, 1.0);
                            bool keep =
                                (
                                    method_ == samplingMethods::stopNearest &&
                                    s + sum - target < target - sum
                                ) || (
                                    method_ == samplingMethods::stopAfter
                                ) || (
                                    method_ == samplingMethods::stop50 &&
                                    dist(rng_) > 0.5
                                );
                            if (keep) sample.push_back(s);
                            break;
                        }
                    }
                    return sample;
                }
                case samplingMethods::number:
                {
                    // Number method: compute number of samples, then draw
                    unsigned int nSamp = std::round(target / expectationValue(a,b));
                    return draw(nSamp, a, b);
                }
                case samplingMethods::poisson:
                {
                    // Poisson method: draw number of samples, then draw
                    std::poisson_distribution<unsigned int> dist(target / expectationValue(a,b));
                    unsigned int nSamp = dist(rng_);
                    return draw(nSamp, a, b);
                }
                case samplingMethods::sorted:
                {
                    // Sorted sampling: first draw repeatedly using number
                    // method, until target is exceeded, then sort list and
                    // keep or discard last entry following stop nearest method.
                    std::vector<double> samples;
                    double sum = 0.0;
                    while (sum < target) {
                        unsigned int nSamp = std::round((target - sum) / expectationValue(a,b));
                        auto s = draw(nSamp, a, b);
                        samples.resize(samples.size() + nSamp);
                        samples.insert(samples.end(), s.begin(), s.end());
                        for (auto s_ : s) sum += s_;
                    }
                    std::sort(samples.begin(), samples.end());
                    if (std::fabs(target - (sum - samples.back())) <
                        std::fabs(target - sum))
                    {
                        samples.pop_back();
                    }
                    return samples;
                }
            }
        }

    protected:

        std::vector<PDFSegment *> seg_; /**< Segments in the PDF */
        std::valarray<double> wgt_;     /**< Weights of segments */
        rngType& rng_;                  /**< Random engine */
        double sMin_;                   /**< PDF lower limit */
        double sMax_;                   /**< PDF upper limit */
        samplingMethods::method method_; /**< Sampling method for this PDF */
        bool normalized_;               /**< Is this PDF properly normalized? */
        bool ownSegments_;              /**< True if we own the segments */
    };

}

#endif // PDF_HPP
