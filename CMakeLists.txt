cmake_minimum_required(VERSION 3.15)
project(slug VERSION 0.1 LANGUAGES CXX C)

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

# GCC's -Wpsabi fires at every call site of a function returning a small
# struct/pair by value (e.g. Interpolator1D::xRange()), noting that GCC 9
# and GCC >= 10 pass such types differently under the C++17 ABI. This is
# purely an ABI-compatibility note between different GCC major versions;
# since we always build from source with a single, consistent compiler,
# it carries no actual risk here, so silence it rather than have it drown
# out real warnings.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
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

# The directory the top-level CMakeLists.txt (this file) lives in, baked
# in at compile time as a last-resort fallback for utils::getFilePath, so
# data files under the repo (e.g. data/imfs/chabrier.toml) can still be
# found even when SLUG_DIR isn't set and the working directory isn't the
# repo root -- e.g. when slugPython is imported from an arbitrary
# directory. Applies directory-wide, to every target below, since
# getFilePath is reachable from most of them.
add_compile_definitions(REPO_DIR="${CMAKE_SOURCE_DIR}")

# pybind11, for the Python frontend
add_subdirectory(src/extern/pybind11)

# Main executable
add_executable(slug
    src/main.cpp
    src/core/Cluster.cpp
    src/core/Galaxy.cpp
    src/core/SimCluster.cpp
    src/core/SimGalaxy.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/tracks/TrackUtils.cpp
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
)

# Test executables
add_executable(slugTestInterpolation
    src/interpolation/Mesh2DGrid.cpp
    tests/interpolation/testInterpolationAll.cpp
    tests/interpolation/testMesh2DGrid.cpp
    tests/interpolation/testMesh2DInterpolator.cpp
    tests/interpolation/testMesh3DInterpolator.cpp
)
add_executable(slugTestExtinct
    src/extinct/Extinct.cpp
    src/phot/FilterTabulated.cpp
    tests/extinct/testExtinctAll.cpp
)
add_executable(slugTestPhot
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    tests/phot/testPhotAll.cpp
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
    tests/utils/testPDFIntegrator.cpp
    tests/utils/testPDFIntegratorND.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
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
    src/core/Galaxy.cpp
    src/core/SimCluster.cpp
    src/core/SimGalaxy.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
    tests/core/testCluster.cpp
    tests/core/testCoreAll.cpp
    tests/core/testGalaxy.cpp
    tests/core/testSimCluster.cpp
    tests/core/testSimGalaxy.cpp
    tests/core/testSpecsynChain.cpp
)
# slugTestCoreFull: the slow, full-scale end-to-end counterpart to
# slugTestCore above -- same core-area sources, but built and run as
# its own executable/CTest entry (LABELS "slow" below) specifically so
# it can be skipped independently of slugTestCore's fast tests. See
# testCoreFullAll.cpp's own comment for why, and for how to add more
# tests here as this grows.
add_executable(slugTestCoreFull
    src/core/Cluster.cpp
    src/core/Galaxy.cpp
    src/core/SimCluster.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
    tests/core/testClusterSpecsynFullCommon.cpp
    tests/core/testClusterSpecsynFull.cpp
    tests/core/testClusterSpecsynFullAFe.cpp
    tests/core/testClusterSpecsynFullNonStoch.cpp
    tests/core/testCoreFullAll.cpp
)
# slugTestCoreFull drives a real end-to-end run (real MIST tracks, real
# multi-GB spectral libraries) through the same interpolation-heavy
# code paths as production use, so building it at whatever
# CMAKE_BUILD_TYPE the rest of the project happens to be configured
# with (often an unoptimized Debug build, for everything else's
# debuggability) makes it dramatically slower to run than it needs to
# be. Force optimized flags for just this one target, independent of
# CMAKE_BUILD_TYPE, so this slow test stays fast to run regardless of
# how the rest of the project is configured.
target_compile_options(slugTestCoreFull PRIVATE -O3)
target_compile_definitions(slugTestCoreFull PRIVATE NDEBUG)
add_executable(slugTestIo
    src/core/Cluster.cpp
    src/core/Galaxy.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
    tests/io/testIoAll.cpp
    tests/io/testOutputManager.cpp
    tests/io/testSimControls.cpp
)
add_executable(slugTestSpecsyn
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    tests/specsyn/testSpecsynAll.cpp
    tests/specsyn/testSpecsynBlackbody.cpp
    tests/specsyn/testSpecsynUtils.cpp
    tests/specsyn/testSpecsynLib.cpp
    tests/specsyn/testSpecsynLibWR.cpp
    tests/specsyn/testSpecsynLibWD.cpp
    tests/specsyn/testSpecsynLibChained.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/utils/ParseUtils.cpp
)

# Python module. The CMake target is named slugPython (the "slug" target
# name is already taken by the main executable), but the output module
# file is named _slug to match the PYBIND11_MODULE name in Bindings.cpp.
# It is built directly into slugpy/, the top-level pure-Python package
# that wraps it (see slugpy/__init__.py), rather than into the build
# directory, so that package's compiled and pure-Python halves always
# live side by side and "import slugpy" works straight out of the
# source tree with no separate install step.
pybind11_add_module(slugPython
    src/pybind/Bindings.cpp
    src/pybind/BindInterpolator1D.cpp
    src/pybind/BindInterpolator1DScalar.cpp
    src/pybind/BindTracks2D.cpp
    src/pybind/BindTracks3D.cpp
    src/pybind/BindSimControls.cpp
    src/pybind/BindSpecsyn.cpp
    src/pybind/BindCluster.cpp
    src/pybind/BindGalaxy.cpp
    src/pybind/BindFilter.cpp
    src/pybind/BindFilterIdeal.cpp
    src/pybind/BindFilterTabulated.cpp
    src/pybind/BindFilterCollection.cpp
    src/pybind/BindPhotConvert.cpp
    src/pybind/BindPDF.cpp
    src/pybind/BindOutputManager.cpp
    src/pybind/BindOutputManagerH5.cpp
    src/pybind/BindOutputManagerAscii.cpp
    src/pybind/BindSimCluster.cpp
    src/pybind/BindSimGalaxy.cpp
    src/core/Cluster.cpp
    src/core/Galaxy.cpp
    src/core/SimCluster.cpp
    src/core/SimGalaxy.cpp
    src/extinct/Extinct.cpp
    src/io/SimControls.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    src/extern/cubature/pcubature.c
    src/extern/cubature/hcubature.c
)
set_target_properties(slugPython PROPERTIES
    OUTPUT_NAME _slug
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/slugpy)

# Force optimized flags for slugPython, independent of CMAKE_BUILD_TYPE,
# for the same reason as slugTestCoreFull above: it drives the same
# interpolation-heavy production code paths, so building it at whatever
# (often unoptimized) CMAKE_BUILD_TYPE the rest of the project happens
# to be configured with would make every slugpy-based analysis
# needlessly slow.
target_compile_options(slugPython PRIVATE -O3)
target_compile_definitions(slugPython PRIVATE NDEBUG)

# Include directories: project headers as regular includes; pcg-cpp,
# tomlplusplus, and cubature as system includes so clang-tidy and the
# compiler suppress warnings from them. (pybind11's own include directory
# is already added as SYSTEM automatically by pybind11_add_module below,
# for slugPython.) This only takes effect for headers actually reached
# via this search list (e.g. an angle-bracket include) -- a relative
# "../extern/..." include resolves via the including file's own
# directory first and never consults this list, so it would NOT be
# treated as a system header despite living under a SYSTEM include path.
# This is why the toml.hpp, pybind11, and cubature.h includes throughout
# the codebase are written in angle-bracket form.
target_include_directories(slug PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slug SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugTestInterpolation PRIVATE src)
target_include_directories(slugTestInterpolation SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestExtinct PRIVATE src)
target_include_directories(slugTestExtinct SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestPhot PRIVATE src)
target_include_directories(slugTestPhot SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestPDFs PRIVATE src)
target_include_directories(slugTestPDFs SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)


target_include_directories(slugTestUtils PRIVATE src)
target_include_directories(slugTestUtils SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugTestTracks PRIVATE src)
target_include_directories(slugTestTracks SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestCore PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestCore SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugTestCoreFull PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestCoreFull SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugTestIo PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestIo SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugTestSpecsyn PRIVATE src)
target_include_directories(slugTestSpecsyn SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slugPython PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugPython SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus src/extern/cubature)

target_include_directories(slug SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestInterpolation SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestExtinct SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestPhot SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestPDFs SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestUtils SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestTracks SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestCore SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestCoreFull SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestIo SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestSpecsyn SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugPython SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})

# Link to GSL
target_link_libraries(slug PRIVATE GSL::gsl)
target_link_libraries(slugTestInterpolation PRIVATE GSL::gsl)
target_link_libraries(slugTestExtinct PRIVATE GSL::gsl)
target_link_libraries(slugTestPhot PRIVATE GSL::gsl)
target_link_libraries(slugTestPDFs PRIVATE GSL::gsl)
target_link_libraries(slugTestUtils PRIVATE GSL::gsl)
target_link_libraries(slugTestTracks PRIVATE GSL::gsl)
target_link_libraries(slugTestCore PRIVATE GSL::gsl)
target_link_libraries(slugTestCoreFull PRIVATE GSL::gsl)
target_link_libraries(slugTestIo PRIVATE GSL::gsl)
target_link_libraries(slugTestSpecsyn PRIVATE GSL::gsl)
target_link_libraries(slugPython PRIVATE GSL::gsl)

# Link to HDF5
target_link_libraries(slug PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestInterpolation PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestExtinct PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestPhot PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestPDFs PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestUtils PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestTracks PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestCore PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestCoreFull PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestIo PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugTestSpecsyn PRIVATE ${HDF5_CXX_LIBRARIES})
target_link_libraries(slugPython PRIVATE ${HDF5_CXX_LIBRARIES})

# Link to OpenMP, if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(slug PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestInterpolation PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestExtinct PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestPhot PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestPDFs PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestUtils PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestTracks PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestCore PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestCoreFull PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestIo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestSpecsyn PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugPython PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_Interpolation COMMAND slugTestInterpolation)
# test_Extinct loads the extinction curve registry (data/extinct/
# extinct.toml/.h5) by relative path, so it needs to run from the
# source tree root.
add_test(NAME test_Extinct COMMAND slugTestExtinct
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_Phot COMMAND slugTestPhot)
add_test(NAME test_PDFs COMMAND slugTestPDFs)
# test_Utils loads real IMF files (data/imfs/chabrier.toml,
# data/imfs/salpeter.toml) directly by relative path for
# testPDFIntegrator/testPDFIntegratorND, so it needs to run from the
# source tree root.
add_test(NAME test_Utils COMMAND slugTestUtils
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# test_Specsyn loads a real IMF file (data/imfs/chabrier.toml)
# directly by relative path for testSpecCts, so it needs to run from
# the source tree root.
add_test(NAME test_Specsyn COMMAND slugTestSpecsyn
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
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

# test_CoreFull is the slow, full-scale end-to-end counterpart to
# test_Core -- see testCoreFullAll.cpp and slugTestCoreFull's own
# comments above for why it's built and run as a separate executable/
# CTest entry rather than folded into test_Core. Needs the same
# SLUG_DIR as test_Core, for the same reason.
add_test(NAME test_CoreFull COMMAND slugTestCoreFull
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_CoreFull PROPERTIES
    ENVIRONMENT "SLUG_DIR=${CMAKE_SOURCE_DIR}")

# Python binding tests: run via pytest, using the same Python interpreter
# pybind11 built slugPython against, so that this always tests the module
# that was just built rather than whatever "python" happens to be on PATH.
# Requires pytest and numpy to be installed for that interpreter. Also
# needs SLUG_DIR set, same as test_Core/test_Io above, since the
# SimPhysics bindings exercise the same stars.IMF file-resolution path.
# PYTHONPATH is the repo root, not slugPython's own output directory,
# since the tests import the slugpy package (which re-exports slugPython's
# _slug.cpython-*.so from slugpy/__init__.py) rather than the compiled
# extension directly.
add_test(NAME test_PythonBindings
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/pybind -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_PythonBindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR};SLUG_DIR=${CMAKE_SOURCE_DIR}")

# slugpy tests: pure-Python, so unlike test_PythonBindings this doesn't
# need slugPython/_slug built first -- only the slugpy package itself,
# which is source, not a build artifact. Reads examples/clusterlib/
# clusterlib.h5, a small enough real output file to be committed to the
# repo directly, so (unlike test_CoreFull and its own real-data
# dependencies) this needs no data fetched separately and can stay a
# "quick" test. Requires h5py, tomlkit, and astropy for whichever
# interpreter CMake picked up.
add_test(NAME test_Slugpy
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/slugpy -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_Slugpy PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}")

# Tag every test "quick" or "slow", so `ctest -L quick` (or `ctest -LE
# slow`) runs only the fast suite, skipping anything that takes
# upwards of a minute to run -- currently just test_CoreFull, but
# expected to grow as more full-scale end-to-end tests are added (see
# testCoreFullAll.cpp). Plain `ctest`, with no -L/-LE filter, still
# runs everything, quick and slow alike. A new slow test belongs in
# its own executable (or an existing "*Full" one, if its own area
# already has one), tagged "slow" here the same way, precisely so
# CTest can select it independently of the fast tests living
# alongside it in the same source directory -- CTest's label filter
# operates per add_test entry, not per individual test function
# inside one, so bundling a slow test into an otherwise-fast
# executable would make it impossible to skip on its own.
set_tests_properties(
    test_Interpolation test_Extinct test_Phot test_PDFs test_Utils test_Specsyn test_Tracks
    test_Core test_Io test_PythonBindings test_Slugpy
    PROPERTIES LABELS "quick")
set_tests_properties(test_CoreFull PROPERTIES LABELS "slow")

# Add command to copy assets to build dir
add_custom_command(
    TARGET slugTestPDFs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pdfs/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets   
)
