/**
 * @file test_PDF.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the PDF class.
 * @details
 * This file contains unit tests for the PDF class,
 * which represents a PDF made up of multiple segments.
 * @date 2024-06-12
 */

#ifndef TEST_PDF_HPP
#define TEST_PDF_HPP

#include <cmath>
#include <cstdio>
#include "../src/pdfs/PDF.hpp"
#include "../tests/testUtils.hpp"

 /** 
  * @brief Unit test for the PDF class.
  * @return 0 if the test passes, 1 if it fails.
  * @details
  * This function tests the PDF class.
  */
auto test_PDF() -> int
{
    rngType rng(42); // Create a random number generator with a fixed seed for reproducibility

}

#endif // TEST_PDF_HPP