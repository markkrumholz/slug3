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

#include "../utils/RngThread.hpp"
#include "PDFCommons.hpp"
#include "PDFSegment.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <random>
#include <ranges>
#include <stdexcept>
#include <valarray>
#include <vector>

/**
 * @brief A namespace to hold PDFs and quantities related to them.
 */
namespace pdfs {

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
         * @param method Sampling method
         * @param normalize Normalize or not?
         */
        template <typename SC, typename WC>
        PDF(SC seg,
            WC wgt,
            SamplingMethods method = SamplingMethods::stopNearest,
            bool normalize = true) :
            seg_(std::move(seg)),
            method_(method),
            normalized_(normalize)
        {
            const auto n = std::size(wgt);
            const auto* ptr = std::data(wgt);
            if (n == 0 || ptr == nullptr) {
                throw std::runtime_error("PDF: must have at least one segment");
            }
            wgt_ = std::valarray<double>(ptr, n);
            if (wgt_.min() <= 0) { // Safety check
                throw std::runtime_error("PDF: elements of wgt must be positive");
            }
            if (seg_.size() != wgt_.size()) {
                throw std::runtime_error("PDF: seg and wgt must be of equal size");
            }
            if (normalize) {
                normalizePDF();
            }
            sMin_ = seg_.front()->getMin();
            sMax_ = seg_.front()->getMax();
            for (auto& s : seg_) {
                sMin_ = std::min(sMin_, s->getMin());
                sMax_ = std::max(sMax_, s->getMax());
            }
        }

        /**
         * @brief Construct a normalized PDF containing a single segment
         * @tparam SegT The concrete segment type; must derive from PDFSegment
         * @param seg A unique_ptr to the segment to use
         * @param wgt Weight of the PDF
         * @param method Sampling method
         * @details
         * Templated on the concrete segment type (rather than taking a
         * unique_ptr<PDFSegment> directly), even though seg is
         * immediately stored as a unique_ptr<PDFSegment>, so that
         * passing a unique_ptr to some derived segment type (the
         * normal way to call this) is an exact match here rather than
         * needing an implicit derived-to-base unique_ptr conversion;
         * without that, overload resolution would sometimes prefer
         * the array-of-segments-and-weights constructor above instead,
         * mistaking the sampling method argument for a weights
         * container.
         */
        template <typename SegT>
            requires std::derived_from<SegT, PDFSegment>
        explicit PDF(std::unique_ptr<SegT> seg,
            const double wgt = 1.0,
            SamplingMethods method = SamplingMethods::stopNearest) :
            wgt_(wgt, 1),
            method_(method),
            normalized_(wgt == 1.0)
        {
            sMin_ = seg->getMin();
            sMax_ = seg->getMax();
            seg_.push_back(std::move(seg));
        }

        /**
         * @brief Construct an empty PDF with no segments
         */
        PDF() :
            sMin_(std::numeric_limits<double>::quiet_NaN()),
            sMax_(std::numeric_limits<double>::quiet_NaN()),
            method_(pdfs::SamplingMethods::none),
            normalized_(false)
        { }
        virtual ~PDF() = default;

        // Disallow copying, since we use unique_ptr
        // objects to hold segments
        PDF(const PDF&) = delete;
        auto operator=(const PDF&) -> PDF& = delete;

        // Default moves
        PDF(PDF&&) = default; 
        auto operator=(PDF&&) -> PDF& = default; 

        // Getters for internal state
        /** @brief Report if PDF has been initialized
         *  @return True if the PDF is initialized and usable, false if not
         */
        [[nodiscard]] auto valid() const { return !seg_.empty(); }
        /** @brief Get the lower limit of the segment.
         *  @return The lower limit of the segment.
         */
        [[nodiscard]] auto getMin() const -> double { return sMin_; }
        /** @brief Get the upper limit of the segment.
         *  @return The upper limit of the segment.
         */
        [[nodiscard]] auto getMax() const -> double { return sMax_; }
        /**
         * @brief Get the weights of segments
         * @return The weights of the segments
         */
        [[nodiscard]] auto getWeights() const -> const std::valarray<double>& { return wgt_; }

        // Sampling policy getter and setter
        /**
         * @brief Get sampling method
         * @return The current sampling method
         */
        [[nodiscard]] auto getSampling() const -> SamplingMethods
        {
            return method_;
        }
        /**
         * @brief Set sampling method
         * @param method Method to set
         */
        void setSampling(SamplingMethods method)
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
        [[nodiscard]] auto normalized() const -> bool { return normalized_; }

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
        [[nodiscard]] auto expectationValue(const double a, const double b) const -> double
        {
            double e = 0.0;
            double wSum = 0.0;
            for (auto const& [s,w] : std::views::zip(seg_,wgt_)) {
                const double wi = w * s->integral(a, b);
                e += wi * s->expectationValue(a, b);
                wSum += wi;
            }
            return e / wSum;
        }
        /**
         * @brief Calculate the expectation value of the PDF segment over its entire range.
         * @return The expectation value of the PDF segment.
         */
        [[nodiscard]] auto expectationValue() const -> double
        {
            return expectationValue(sMin_, sMax_);
        }
        /**
         * @brief Calculate the integral of the PDF over a specified range.
         * @param a The lower limit of the range for integral calculation; if set to a value <= sMin, will be set to sMin_.
         * @param b The upper limit of the range for integral calculation; if set to a value >= sMax, will be set to sMax_.
         * @return The integral of the PDF over the specified range.
         */
        [[nodiscard]] auto integral(const double a, const double b) const -> double {
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
        [[nodiscard]] auto integral() const -> double { return wgt_.sum(); }

        // Drawing functions
        /**
         * @brief Sample a random value from the PDF within a specified range.
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if unspecified).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if unspecified).
         * @return A random value drawn from the PDF within the specified range.
         */
        [[nodiscard]]
        auto draw(const double a = std::numeric_limits<double>::lowest(),
         const double b = std::numeric_limits<double>::max()) const -> double {
            std::vector<double> wLim(wgt_.size());
            for (auto const& [wL,s,w] : std::views::zip(wLim,seg_,wgt_)) {
                wL = w * s->integral(a,b);
            }
            std::discrete_distribution<int> dist(wLim.begin(), wLim.end());
            return seg_.at(dist(utils::rng()()))->draw(a,b);
        }
        /**
         * @brief Sample nDraw random values from the PDF within the specified range.
         * @param nDraw Number of samples to draw.
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if not set).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if not set).
         * @return A vector of nDraw random values drawn from the PDF within the specified range.
        */
        [[nodiscard]] auto draw(unsigned int nDraw, 
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max())
            const -> std::vector<double>
        {
            std::vector<double> result(nDraw);
            for (auto &r : result) { r = draw(a, b); }
            return result;        
        }
        /**
         * @brief Draw a sample from the PDF targeting a fixed total value
         * @param target Target total to draw
         * @param a The lower limit of the sampling range (should be >= sMin_, equal to sMin_ if not set).
         * @param b The upper limit of the sampling range (should be <= sMax_, equal to sMax_ if not set).
         */
        [[nodiscard]] auto drawTarget(double target, 
            double a = std::numeric_limits<double>::lowest(),
            double b = std::numeric_limits<double>::max()) const 
            -> std::vector<double>
        {
            std::vector<double> sample;
            switch (method_)
            {
                case SamplingMethods::stopNearest:
                case SamplingMethods::stopBefore:
                case SamplingMethods::stopAfter:
                case SamplingMethods::stop50:
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
                            const bool keep =
                                (
                                    method_ == SamplingMethods::stopNearest &&
                                    s + sum - target < target - sum
                                ) || (
                                    method_ == SamplingMethods::stopAfter
                                ) || (
                                    method_ == SamplingMethods::stop50 &&
                                    dist(utils::rng()()) > 0.5
                                );
                            if (keep) { sample.push_back(s); }
                            break;
                        }
                    }
                    return sample;
                }
                case SamplingMethods::number:
                {
                    // Number method: compute number of samples, then draw
                    const auto nSamp = 
                        static_cast<unsigned int>(
                        std::round(target / expectationValue(a,b))
                        );
                    return draw(nSamp, a, b);
                }
                case SamplingMethods::poisson:
                {
                    // Poisson method: draw number of samples, then draw
                    std::poisson_distribution<unsigned int> dist(target / expectationValue(a,b));
                    const auto nSamp = dist(utils::rng()());
                    return draw(nSamp, a, b);
                }
                case SamplingMethods::sorted:
                {
                    // Sorted sampling: first draw repeatedly using number
                    // method, until target is exceeded, then sort list and
                    // keep or discard last entry following stop nearest method.
                    std::vector<double> samples;
                    double sum = 0.0;
                    while (sum < target) {
                        const auto nSamp = 
                            static_cast<unsigned int>(std::round((target - sum) / expectationValue(a,b)));
                        auto sampleSet = draw(nSamp, a, b);
                        samples.resize(samples.size() + nSamp);
                        samples.insert(samples.end(), sampleSet.begin(), sampleSet.end());
                        for (auto s : sampleSet) { sum += s; }
                    }
                    std::ranges::sort(samples);
                    if (std::fabs(target - (sum - samples.back())) <
                        std::fabs(target - sum))
                    {
                        samples.pop_back();
                    }
                    return samples;
                }
                case SamplingMethods::none:
                {
                    throw std::runtime_error("PDF: invalid sampling method");
                }
            }
        }

    protected:

        std::vector<std::unique_ptr<PDFSegment> > seg_; /**< Segments in the PDF */
        std::valarray<double> wgt_;     /**< Weights of segments */
        double sMin_;                   /**< PDF lower limit */
        double sMax_;                   /**< PDF upper limit */
        SamplingMethods method_;        /**< Sampling method for this PDF */
        bool normalized_;               /**< Is this PDF properly normalized? */
    };

} // namespace pdfs

#endif // PDF_HPP
