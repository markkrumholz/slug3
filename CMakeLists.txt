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

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Find OpenMP; threading is optional, so fall back to a single-threaded
# build with a warning if no OpenMP implementation is available. Code that
# depends on OpenMP must guard itself with #ifdef _OPENMP.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found: building without threading support")
endif()

# Determine the current git commit hash, so it can be embedded in output
# files to record exactly which version of slug produced them. Falls back
# to "unknown" if git isn't available or this isn't a git checkout (e.g.
# a source tarball). This is captured at configure time, so re-run cmake
# to refresh the embedded hash after committing.
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SLUG_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE SLUG_GIT_HASH_RESULT
)
if(NOT SLUG_GIT_HASH_RESULT EQUAL 0 OR NOT SLUG_GIT_HASH)
    set(SLUG_GIT_HASH "unknown")
endif()
configure_file(
    ${CMAKE_SOURCE_DIR}/src/io/SlugVersion.hpp.in
    ${CMAKE_BINARY_DIR}/generated/io/SlugVersion.hpp
    @ONLY
)

# pybind11, for the Python frontend
add_subdirectory(src/extern/pybind11)

# Main executable
add_executable(slug
    src/main.cpp
    src/core/Cluster.cpp
    src/core/SimCluster.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/io/SimPhysics.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/tracks/TrackUtils.cpp
    src/io/SimControls.cpp
    src/utils/ParseUtils.cpp
)

# Test executables
add_executable(slugTestInterpolation
    src/interpolation/Mesh2DGrid.cpp
    tests/interpolation/testInterpolationAll.cpp
    tests/interpolation/testMesh2DGrid.cpp
    tests/interpolation/testMesh2DInterpolator.cpp
    tests/interpolation/testMesh3DInterpolator.cpp
)
add_executable(slugTestPDFs
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/utils/ParseUtils.cpp
    tests/pdfs/testPDF.cpp
    tests/pdfs/testPDFAll.cpp
    tests/pdfs/testPDFSegmentPowerlaw.cpp
    tests/pdfs/testPDFSegmentSchechter.cpp
)
add_executable(slugTestUtils
    tests/utils/testUtilsAll.cpp
)
add_executable(slugTestTracks
    src/interpolation/Mesh2DGrid.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    tests/tracks/testTracksAll.cpp
)
add_executable(slugTestCore
    src/core/Cluster.cpp
    src/io/SimControls.cpp
    src/io/SimPhysics.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/core/testCluster.cpp
    tests/core/testCoreAll.cpp
)
add_executable(slugTestIo
    src/core/Cluster.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/io/SimControls.cpp
    src/io/SimPhysics.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/io/testIoAll.cpp
    tests/io/testOutputManager.cpp
    tests/io/testSimControls.cpp
    tests/io/testSimPhysics.cpp
)

# Python module, for testing the Python frontend. The CMake target is named
# slugPython (the "slug" target name is already taken by the main
# executable), but the output module file is named slug to match the
# PYBIND11_MODULE name in Bindings.cpp.
pybind11_add_module(slugPython
    src/pybind/Bindings.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
)
set_target_properties(slugPython PROPERTIES OUTPUT_NAME slug)

# Include directories: project headers as regular includes; pcg-cpp and
# tomlplusplus as system includes so clang-tidy and the compiler suppress
# warnings from them. (pybind11's own include directory is already added
# as SYSTEM automatically by pybind11_add_module below, for slugPython.)
# This only takes effect for headers actually reached via this search
# list (e.g. an angle-bracket include) -- a relative "../extern/..."
# include resolves via the including file's own directory first and
# never consults this list, so it would NOT be treated as a system
# header despite living under a SYSTEM include path. This is why the
# toml.hpp and pybind11 includes throughout the codebase are written in
# angle-bracket form.
target_include_directories(slug PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slug SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestInterpolation PRIVATE src)
target_include_directories(slugTestInterpolation SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestPDFs PRIVATE src)
target_include_directories(slugTestPDFs SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)


target_include_directories(slugTestUtils PRIVATE src)
target_include_directories(slugTestUtils SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestTracks PRIVATE src)
target_include_directories(slugTestTracks SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestCore PRIVATE src)
target_include_directories(slugTestCore SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestIo PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestIo SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugPython PRIVATE src)
target_include_directories(slugPython SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slug SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestInterpolation SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestPDFs SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestUtils SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestTracks SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestCore SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestIo SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugPython SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})

# Link to GSL
target_link_libraries(slug PRIVATE GSL::gsl)
target_link_libraries(slugTestInterpolation PRIVATE GSL::gsl)
target_link_libraries(slugTestPDFs PRIVATE GSL::gsl)
target_link_libraries(slugTestUtils PRIVATE GSL::gsl)
target_link_libraries(slugTestTracks PRIVATE GSL::gsl)
target_link_libraries(slugTestCore PRIVATE GSL::gsl)
target_link_libraries(slugTestIo PRIVATE GSL::gsl)
target_link_libraries(slugPython PRIVATE GSL::gsl)

# Link to HDF5
target_link_libraries(slug PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestInterpolation PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestPDFs PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestUtils PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestTracks PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestCore PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestIo PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugPython PRIVATE ${HDF5_CXX_LIBRARIES})

# Link to OpenMP, if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(slug PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestInterpolation PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestPDFs PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestUtils PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestTracks PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestCore PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestIo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugPython PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_Interpolation COMMAND slugTestInterpolation)
add_test(NAME test_PDFs COMMAND slugTestPDFs)
add_test(NAME test_Utils COMMAND slugTestUtils)
add_test(NAME test_Tracks COMMAND slugTestTracks
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# test_Core exercises SimPhysics's file resolution via getFilePath's
# SLUG_DIR + prefix rule (e.g. for stars.IMF = "chabrier.toml", which
# resolves to data/imfs/chabrier.toml), so SLUG_DIR must be set here.
add_test(NAME test_Core COMMAND slugTestCore
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_Core PROPERTIES
    ENVIRONMENT "SLUG_DIR=${CMAKE_SOURCE_DIR}")

# test_Io exercises SimPhysics's file resolution via getFilePath's
# SLUG_DIR + prefix rule, same as test_Core above.
add_test(NAME test_Io COMMAND slugTestIo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_Io PROPERTIES
    ENVIRONMENT "SLUG_DIR=${CMAKE_SOURCE_DIR}")

# Python binding tests: run via pytest, using the same Python interpreter
# pybind11 built slugPython against, so that this always tests the module
# that was just built rather than whatever "python" happens to be on PATH.
# Requires pytest and numpy to be installed for that interpreter.
add_test(NAME test_PythonBindings
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/pybind -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_PythonBindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:slugPython>")

# Add command to copy assets to build dir
add_custom_command(
    TARGET slugTestPDFs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pdfs/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets   
)