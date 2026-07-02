cmake_minimum_required(VERSION 3.15)
project(slug VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# On macOS, Apple Clang implicitly knows the SDK sysroot, but Homebrew LLVM
# tools (e.g. clang-tidy) do not.  Make it explicit so compile_commands.json
# is self-contained for any Clang-based tool.
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE APPLE_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    add_compile_options("--sysroot=${APPLE_SDK_PATH}")
endif()

# Find GSL
find_package(GSL REQUIRED)

# Main executable
add_executable(slug 
    src/main.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
)

# Test executables
add_executable(slugTestInterpolation
    src/interpolation/Mesh2DGrid.cpp
    tests/interpolation/testInterpolationAll.cpp
    tests/interpolation/testMesh2DGrid.cpp
    tests/interpolation/testMesh2DInterpolator.cpp)
add_executable(slugTestPDFs
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    tests/pdfs/testPDF.cpp
    tests/pdfs/testPDFAll.cpp
    tests/pdfs/testPDFSegmentPowerlaw.cpp
    tests/pdfs/testPDFSegmentSchechter.cpp
)

# Include directories: project headers as regular includes, pcg-cpp as a
# system include so clang-tidy and the compiler suppress warnings from it.
target_include_directories(slug PRIVATE src)
target_include_directories(slug SYSTEM PRIVATE src/pcg-cpp/include)

target_include_directories(slugTestInterpolation PRIVATE src)
target_include_directories(slugTestInterpolation SYSTEM PRIVATE src/pcg-cpp/include)

target_include_directories(slugTestPDFs PRIVATE src)
target_include_directories(slugTestPDFs SYSTEM PRIVATE src/pcg-cpp/include)

# Link to GSL
target_link_libraries(slug PRIVATE GSL::gsl)
target_link_libraries(slugTestInterpolation PRIVATE GSL::gsl)
target_link_libraries(slugTestPDFs PRIVATE GSL::gsl)

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_Interpolation COMMAND slugTestInterpolation)
add_test(NAME test_PDFs COMMAND slugTestPDFs)

# Add command to copy assets to build dir
add_custom_command(
    TARGET slugTestPDFs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pdfs/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets   
)