/**
 * @file PDFSegment.hpp
 * @author Mark Krumholz
 * @brief Class to represent a segment of a PDF.
 * @details
 * This class represents a segment of a PDF, defined by a lower
 * and upper limit. It defines an interface with some standard
 * functions on PDF segments that must be implemented by derived
 * classes representing segments with particular functional forms. 
 * @date 2024-06-12
 */

#ifndef PDFSEGMENT_HPP
#define PDFSEGMENT_HPP

#include <map>
#include <vector>
#include "PDFCommons.hpp"

namespace pdfs {

    /**
     * @class PDFSegment
     * @brief Abstract base class representing a segment of a PDF.
     * @details
     * This class defines the interface for a PDF segment, which includes
     * methods to evaluate the PDF at a given point and to sample a random
     * value from the PDF segment. Derived classes must implement these methods
     * according to the specific functional form of the PDF segment they represent.
     */
    class PDFSegment {
    public:

        // Constructors and destructor
        /**
         * @brief Constructor for PDFSegment.
         * @param lower The lower limit of the segment.
         * @param upper The upper limit of the segment.
         * @param rng Reference to the random number generator to be used for sampling.
         */
        PDFSegment(double sMin, double sMax, rngType &rng) : 
            sMin_(sMin), sMax_(sMax), rng_(rng) {}
        virtual ~PDFSegment() = default;

        // Getters for the lower and upper limits of the segment
        /** @brief Get the lower limit of the segment.
         *  @return The lower limit of the segment.
         */
        auto getMin() const -> double { return sMin_; }
        /** @brief Get the upper limit of the segment.
         *  @return The upper limit of the segment.
         */
        auto getMax() const -> double { return sMax_; }

        // Evaluation functions
        /**
         * @brief Evaluate the PDF segment at a given point.
         * @param x The point at which to evaluate the PDF segment.
         * @return The value of the PDF segment at the given point.
         */
        virtual auto operator()(double x) const -> double = 0;
        /**
         * @brief Calculate the expectation value of the PDF segment.
         * @param a The lower limit of the range for expectation value calculation; if set to a value <= sMin, will be set to sMin_.
         * @param b The upper limit of the range for expectation value calculation; if set to a value >= sMax, will be set to sMax_.
         * @return The expectation value of the PDF segment.
         */
        virtual auto expectationValue(const double a, const double b) const -> double = 0;
        /**
         * @brief Calculate the expectation value of the PDF segment over its entire range.
         * @return The expectation value of the PDF segment.
         */
        virtual auto expectationValue() const -> double = 0;

        /**
         * @brief Calculate the integral of the PDF segment over a specified range.
         * @param a The lower limit of the range for integral calculation; if set to a value <= sMin, will be set to sMin_.
         * @param b The upper limit of the range for integral calculation; if set to a value >= sMax, will be set to sMax_.
         * @return The integral of the PDF segment over the specified range.
         */
        virtual auto integral(const double a, const double b) const -> double = 0;

        // Drawing functions
        /**
         * @brief Sample a random value from the PDF segment within a specified range.
         * @param a The lower limit of the sampling range (should be >= sMin_).
         * @param b The upper limit of the sampling range (should be <= sMax_).
         * @return A random value drawn from the PDF segment within the specified range.
         */
        virtual auto draw(const double a, const double b) const -> double = 0;
        /**
         * @brief Sample a random value from the PDF segment.
         * @return A random value drawn from the PDF segment.
         */
        virtual auto draw() const -> double = 0;

    protected:

        /**
         * @brief Parse a segment declaration in a PDF file
         * @param file File stream to parse
         * @param tok List of expected tokens for that segment
         * @returns Map of token values
         */
        auto segmentParser(std::ifstream& file,
            std::vector<std::string>& tok) 
            -> std::map<std::string, double>;

        double sMin_; /**< The lower limit of the segment */
        double sMax_; /**< The upper limit of the segment */
        rngType &rng_; /**< Reference to the random number generator */
    };

}

#endif // PDFSEGMENT_HPP