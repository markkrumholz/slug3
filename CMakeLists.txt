cmake_minimum_required(VERSION 3.15)
project(slug VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GSL
find_package(GSL REQUIRED)

# Main executable
add_executable(slug src/main.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
)

# Test executables
add_executable(slug_test_PDFs
    tests/pdfs/test_PDFs.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
)

# Include directories for the test executable
target_include_directories(slug_test_PDFs PRIVATE
    src
    src/pcg-cpp/include
)

# Link to GSL
target_link_libraries(slug PRIVATE GSL::gsl)
target_link_libraries(slug_test_PDFs PRIVATE GSL::gsl)

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_PDFs COMMAND slug_test_PDFs)

# Add command to copy assets to build dir
add_custom_command(
    TARGET slug_test_PDFs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pdfs/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets   
)