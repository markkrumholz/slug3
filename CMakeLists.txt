cmake_minimum_required(VERSION 3.15)

# The version number lives in exactly one place, pyproject.toml's own
# [project] version, which is also what pip/scikit-build-core use for
# an installed package -- read it from there so a plain CMake build and
# a pip-installed one always agree. Only its leading X.Y.Z is used here
# (CMake's project() accepts nothing else), so a PEP 440 suffix such as
# the ".devN" the wheel-building workflow adds for TestPyPI dry runs
# (see .github/workflows/wheels.yml) is ignored.
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml SLUG_VERSION_LINE
    REGEX "^version[ \t]*=[ \t]*\"[0-9]+\\.[0-9]+\\.[0-9]+")
string(REGEX REPLACE "^[^\"]*\"([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" SLUG_VERSION "${SLUG_VERSION_LINE}")
if(NOT SLUG_VERSION)
    message(FATAL_ERROR "Could not read version from pyproject.toml")
endif()
project(slug VERSION ${SLUG_VERSION} LANGUAGES CXX C)

# Default to an optimized Release build when no build type is given,
# since that is what anyone building slug to run simulations wants.
# Developers who want assertions enabled (they are compiled out by
# Release's NDEBUG) should configure with -DCMAKE_BUILD_TYPE=Debug, as
# CI's test and clang-tidy jobs do. Multi-config generators (Xcode,
# Visual Studio, Ninja Multi-Config) choose the configuration at build
# time instead, so this does not apply to them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified; defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# SKBUILD is set by scikit-build-core, i.e. when this project is being
# built by `pip install .` (see pyproject.toml's [tool.scikit-build])
# rather than by a plain `cmake -B build`. A plain CMake build keeps
# the in-tree developer workflow described in AGENTS.md (slugPython
# built straight into slugpy/, data found via REPO_DIR); an SKBUILD
# build instead installs slugPython, the slug CLI, and the small,
# checked-in data files into the slugpy package itself -- see the
# install() rules at the end of this file.
if(SKBUILD)
    message(STATUS "Building slug ${SLUG_VERSION} via scikit-build-core (pip install)")
endif()

# Determine the current git commit hash, so it can be embedded in output
# files to record exactly which version of slug produced them. This is
# captured at configure time, so re-run cmake to refresh the embedded
# hash after committing. An sdist (see pyproject.toml) is not a git
# checkout, so when scikit-build-core builds one (SKBUILD_STATE "sdist",
# with sdist.cmake = true), the hash is recorded in .slug_git_hash at
# the source root, which the sdist then includes; a later build from
# that sdist, where git can't supply the hash, reads it back from there.
# Falls back to "unknown" if neither is available (e.g. a plain source
# archive of a non-git copy).
# git is only trusted if this source tree is itself the top level of
# the repository it finds -- an unpacked sdist that happens to sit
# somewhere inside some other, unrelated git repository would
# otherwise pick up that repository's own hash instead.
set(SLUG_GIT_HASH_FILE ${CMAKE_SOURCE_DIR}/.slug_git_hash)
set(SLUG_GIT_HASH "")
execute_process(
    COMMAND git rev-parse --show-toplevel
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SLUG_GIT_TOPLEVEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE SLUG_GIT_TOPLEVEL_RESULT
)
get_filename_component(SLUG_SOURCE_REALPATH "${CMAKE_SOURCE_DIR}" REALPATH)
if(SLUG_GIT_TOPLEVEL_RESULT EQUAL 0 AND SLUG_GIT_TOPLEVEL)
    get_filename_component(SLUG_GIT_TOPLEVEL "${SLUG_GIT_TOPLEVEL}" REALPATH)
endif()
if(SLUG_GIT_TOPLEVEL_RESULT EQUAL 0 AND SLUG_GIT_TOPLEVEL STREQUAL SLUG_SOURCE_REALPATH)
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE SLUG_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE SLUG_GIT_HASH_RESULT
    )
    if(NOT SLUG_GIT_HASH_RESULT EQUAL 0)
        set(SLUG_GIT_HASH "")
    endif()
endif()
if(SLUG_GIT_HASH)
    if(SKBUILD_STATE STREQUAL "sdist")
        file(WRITE ${SLUG_GIT_HASH_FILE} "${SLUG_GIT_HASH}\n")
    endif()
elseif(EXISTS ${SLUG_GIT_HASH_FILE})
    file(STRINGS ${SLUG_GIT_HASH_FILE} SLUG_GIT_HASH LIMIT_COUNT 1)
else()
    set(SLUG_GIT_HASH "unknown")
endif()
if(NOT SLUG_GIT_HASH)
    set(SLUG_GIT_HASH "unknown")
endif()

# Building an sdist only needs the hash recorded above, so stop here
# rather than going on to require GSL, HDF5, etc. just to package the
# source tree
if(SKBUILD_STATE STREQUAL "sdist")
    message(STATUS "Recorded git hash ${SLUG_GIT_HASH} in ${SLUG_GIT_HASH_FILE} for the sdist")
    return()
endif()

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

# Find HDF5. slug uses only HDF5's C API (hdf5.h), never its C++ API
# (H5Cpp.h), so only the C component is required -- many HDF5
# installations, e.g. parallel (MPI-IO) builds common on HPC systems,
# don't provide the C++ library at all. A static HDF5
# (HDF5_USE_STATIC_LIBRARIES, as used to build binary wheels -- see
# .github/wheels/build_deps.sh) built with zlib support lists ZLIB::ZLIB
# in its own exported link interface without finding it itself, so find
# it first in that case.
if(HDF5_USE_STATIC_LIBRARIES)
    set(ZLIB_USE_STATIC_LIBS ON)
    find_package(ZLIB REQUIRED)
endif()
find_package(HDF5 REQUIRED COMPONENTS C)

# Find OpenMP; threading is optional, so fall back to a single-threaded
# build with a warning if no OpenMP implementation is available. Code that
# depends on OpenMP must guard itself with #ifdef _OPENMP.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found: building without threading support")
endif()

# Find MPI; multi-process parallelism is optional, like OpenMP, and only
# ever needed by the slug CLI executable itself (not slugPython or the
# test suite -- see SLUG_MPI's own uses in src/core and src/io). Unlike
# _OPENMP, which the compiler defines automatically, there is no standard
# preprocessor macro for "MPI is available", so code that depends on it
# must guard itself with #ifdef SLUG_MPI, defined below only for the
# targets that actually link MPI. Only MPI's C API is used (see
# src/utils/MPIUtils.hpp), so only its C component is required -- not
# the deprecated C++ bindings (libmpi_cxx), which e.g. Open MPI 4.x
# doesn't build by default, and which a host MPI bound into the
# container image (see Dockerfile) therefore may not provide.
# Off by default for a pip-installed (SKBUILD) build, whose slug CLI is
# meant to be portable across machines, which an MPI build (tied to one
# specific MPI implementation's ABI) is not.
if(SKBUILD)
    option(SLUG_ENABLE_MPI "Build the slug CLI with MPI support, if MPI is found" OFF)
else()
    option(SLUG_ENABLE_MPI "Build the slug CLI with MPI support, if MPI is found" ON)
endif()
if(SLUG_ENABLE_MPI)
    find_package(MPI COMPONENTS C)
endif()
if(SLUG_ENABLE_MPI AND NOT MPI_C_FOUND)
    message(WARNING "MPI not found: building without multi-process support")
endif()

# Find (or build) an implementation of C++23's <mdspan>; see that file's
# own comment for the native/package/submodule priority this uses.
include(cmake/Mdspan.cmake)

# SLUG_GIT_HASH (see the top of this file) is embedded in output files
# via the generated SlugVersion.hpp header
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
# getFilePath is reachable from most of them. Empty for a pip-installed
# (SKBUILD) build, whose source tree is typically a temporary directory
# that no longer exists once installation finishes; getFilePath skips
# an empty REPO_DIR, and an installed slugpy instead finds its data via
# SLUG_DATA_PATH (see slugpy/_paths.py).
if(SKBUILD)
    add_compile_definitions(REPO_DIR="")
else()
    add_compile_definitions(REPO_DIR="${CMAKE_SOURCE_DIR}")
endif()

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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/tracks/TrackUtils.cpp
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/utils/ParseUtils.cpp
    src/elem/DecayChain.cpp
    src/elem/IsotopeTable.cpp
    src/utils/HDF5Utils.cpp
    src/yields/YieldChannel.cpp
    src/yields/Yields.cpp
    src/feedback/Winds.cpp
)

# Test executables
add_executable(slugTestInterpolation
    src/interpolation/Mesh2DGrid.cpp
    tests/interpolation/testInterpolationAll.cpp
    tests/interpolation/testMesh2DGrid.cpp
    tests/interpolation/testMesh2DInterpolator.cpp
    tests/interpolation/testMesh3DInterpolator.cpp
    src/utils/HDF5Utils.cpp
)
add_executable(slugTestElem
    src/elem/DecayChain.cpp
    src/elem/IsotopeTable.cpp
    tests/elem/testElemAll.cpp
    src/utils/HDF5Utils.cpp
)
add_executable(slugTestExtinct
    src/extinct/Extinct.cpp
    src/interpolation/Mesh2DGrid.cpp
    src/nebular/Nebular.cpp
    src/phot/FilterTabulated.cpp
    src/phot/FilterIdeal.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynBlackbody.cpp
    tests/extinct/testExtinctAll.cpp
    src/utils/HDF5Utils.cpp
)
add_executable(slugTestPhot
    src/phot/FilterCollection.cpp
    src/phot/VegaSpectrum.cpp
    src/phot/FilterIdeal.cpp
    src/phot/FilterTabulated.cpp
    tests/phot/testPhotAll.cpp
    src/utils/HDF5Utils.cpp
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
    tests/pdfs/testPDFReflect.cpp
    tests/pdfs/testPDFSegmentPowerlaw.cpp
    tests/pdfs/testPDFSegmentSchechter.cpp
    src/utils/HDF5Utils.cpp
)
add_executable(slugTestUtils
    tests/utils/testUtilsAll.cpp
    tests/utils/testPDFIntegrator.cpp
    src/pdfs/PDFFileParser.cpp
    src/pdfs/PDFSegment.cpp
    src/pdfs/PDFSegmentDelta.cpp
    src/pdfs/PDFSegmentExponential.cpp
    src/pdfs/PDFSegmentLognormal.cpp
    src/pdfs/PDFSegmentNormal.cpp
    src/pdfs/PDFSegmentPowerlaw.cpp
    src/pdfs/PDFSegmentSchechter.cpp
    src/utils/ParseUtils.cpp
    src/utils/HDF5Utils.cpp
    src/utils/GridBracket.cpp
)
add_executable(slugTestTracks
    src/interpolation/Mesh2DGrid.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    tests/tracks/testTracksAll.cpp
    src/utils/HDF5Utils.cpp
)
# slugTestTracksFull: slow counterpart to slugTestTracks, run against
# the real, gitignored track files in data/tracks rather than the
# small fixtures in tests/tracks/assets -- built and run as its own
# executable/CTest entry (LABELS "slow" below) for the same reason as
# slugTestCoreFull. See testTracksFullAll.cpp.
add_executable(slugTestTracksFull
    src/interpolation/Mesh2DGrid.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    tests/tracks/testTracksFullAll.cpp
    src/utils/HDF5Utils.cpp
)
add_executable(slugTestYields
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    tests/yields/testYieldsAll.cpp
    src/utils/HDF5Utils.cpp
    src/utils/GridBracket.cpp
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
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/core/testCluster.cpp
    tests/core/testCoreAll.cpp
    tests/core/testGalaxy.cpp
    tests/core/testSimCluster.cpp
    tests/core/testSimGalaxy.cpp
    tests/core/testSpecsynChain.cpp
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
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
    src/core/SimGalaxy.cpp
    src/io/OutputManager.cpp
    src/io/OutputManagerAscii.cpp
    src/io/OutputManagerH5.cpp
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/core/testClusterSpecsynFullCommon.cpp
    tests/core/testClusterSpecsynFull.cpp
    tests/core/testClusterSpecsynFullAFe.cpp
    tests/core/testClusterSpecsynFullNonStoch.cpp
    tests/core/testClusterSpecsynFullGrid.cpp
    tests/core/testClusterSpecsynFullNebular.cpp
    tests/core/testGalaxySpecsynFullCommon.cpp
    tests/core/testGalaxySpecsynFullNonStoch.cpp
    tests/core/testGalaxySpecsynFullFieldStars.cpp
    tests/core/testCoreFullAll.cpp
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
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
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/io/testIoAll.cpp
    tests/io/testOutputManager.cpp
    tests/io/testSimControls.cpp
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
)
add_executable(slugTestNebular
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/nebular/testNebularAll.cpp
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
)
# slugTestFeedback needs the same full SimControls dependency closure as
# slugTestNebular above -- Winds only stores a live const io::SimControls&
# (see Winds.hpp's own comment), but SimControls.hpp itself includes the
# complete definitions of Specsyn/Extinct/Nebular/Tracks2D/Tracks3D/
# FilterCollection/Yields (its own shared_ptr members need complete types
# at ~SimControls(), which is defined inline in the header), so any target
# that constructs or destroys a SimControls -- even a default-constructed
# one, as in tests/feedback/testWinds.cpp -- needs all of them linked in.
add_executable(slugTestFeedback
    src/feedback/Winds.cpp
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    tests/feedback/testFeedbackAll.cpp
    tests/feedback/testWinds.cpp
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
)
add_executable(slugTestSpecsyn
    src/specsyn/SpecsynBlackbody.cpp
    src/specsyn/Specsyn.cpp
    src/specsyn/SpecsynUtils.cpp
    src/specsyn/SpecsynLib.cpp
    src/utils/GridBracket.cpp
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
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/HDF5Utils.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/elem/DecayChain.cpp
    src/yields/Yields.cpp
)

# Python module. The CMake target is named slugPython (the "slug" target
# name is already taken by the main executable), but the output module
# file is named _slug to match the PYBIND11_MODULE name in Bindings.cpp.
# It is built directly into slugpy/, the top-level pure-Python package
# that wraps it (see slugpy/__init__.py), rather than into the build
# directory, so that package's compiled and pure-Python halves always
# live side by side and "import slugpy" works straight out of the
# source tree with no separate install step. (A pip-installed, SKBUILD
# build instead install()s it into the slugpy package -- see the end of
# this file.)
pybind11_add_module(slugPython
    src/pybind/Bindings.cpp
    src/pybind/BindInterpolator1D.cpp
    src/pybind/BindInterpolator1DScalar.cpp
    src/pybind/BindTracks2D.cpp
    src/pybind/BindTracks3D.cpp
    src/pybind/BindSimControls.cpp
    src/pybind/BindExtinct.cpp
    src/pybind/BindNebular.cpp
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
    src/pybind/BindYieldChannel.cpp
    src/pybind/BindYields.cpp
    src/pybind/BindWinds.cpp
    src/core/Cluster.cpp
    src/core/Galaxy.cpp
    src/core/SimCluster.cpp
    src/core/SimGalaxy.cpp
    src/extinct/Extinct.cpp
    src/nebular/Nebular.cpp
    src/io/SimControls.cpp
    src/feedback/Winds.cpp
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
    src/utils/GridBracket.cpp
    src/specsyn/SpecsynLibNoWind.cpp
    src/specsyn/SpecsynLib2D.cpp
    src/specsyn/SpecsynLibWD.cpp
    src/specsyn/SpecsynLibWR.cpp
    src/specsyn/SpecsynLibChained.cpp
    src/tracks/TrackUtils.cpp
    src/tracks/Tracks2D.cpp
    src/tracks/Tracks3D.cpp
    src/utils/ParseUtils.cpp
    src/utils/HDF5Utils.cpp
    src/elem/DecayChain.cpp
    src/elem/IsotopeTable.cpp
    src/yields/YieldChannel.cpp
    src/yields/Yields.cpp
)
set_target_properties(slugPython PROPERTIES OUTPUT_NAME _slug)
if(NOT SKBUILD)
    set_target_properties(slugPython PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/slugpy)
endif()

# Force optimized flags for slugPython, independent of CMAKE_BUILD_TYPE,
# for the same reason as slugTestCoreFull above: it drives the same
# interpolation-heavy production code paths, so building it at whatever
# (often unoptimized) CMAKE_BUILD_TYPE the rest of the project happens
# to be configured with would make every slugpy-based analysis
# needlessly slow.
target_compile_options(slugPython PRIVATE -O3)
target_compile_definitions(slugPython PRIVATE NDEBUG)

# Include directories: project headers as regular includes; pcg-cpp
# and tomlplusplus as system includes so clang-tidy and the compiler
# suppress warnings from them. (pybind11's own include directory is
# already added as SYSTEM automatically by pybind11_add_module below,
# for slugPython.) This only takes effect for headers actually reached
# via this search list (e.g. an angle-bracket include) -- a relative
# "../extern/..." include resolves via the including file's own
# directory first and never consults this list, so it would NOT be
# treated as a system header despite living under a SYSTEM include path.
# This is why the toml.hpp and pybind11 includes throughout the
# codebase are written in angle-bracket form.
target_include_directories(slug PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slug SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestInterpolation PRIVATE src)
target_include_directories(slugTestInterpolation SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestElem PRIVATE src)
target_include_directories(slugTestElem SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestExtinct PRIVATE src)
target_include_directories(slugTestExtinct SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestPhot PRIVATE src)
target_include_directories(slugTestPhot SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestPDFs PRIVATE src)
target_include_directories(slugTestPDFs SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)


target_include_directories(slugTestUtils PRIVATE src)
target_include_directories(slugTestUtils SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestTracks PRIVATE src)
target_include_directories(slugTestTracksFull PRIVATE src)
target_include_directories(slugTestTracks SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)
target_include_directories(slugTestTracksFull SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestYields PRIVATE src)
target_include_directories(slugTestYields SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestCore PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestCore SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestCoreFull PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestCoreFull SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestIo PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugTestIo SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestNebular PRIVATE src)
target_include_directories(slugTestNebular SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestFeedback PRIVATE src)
target_include_directories(slugTestFeedback SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugTestSpecsyn PRIVATE src)
target_include_directories(slugTestSpecsyn SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slugPython PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(slugPython SYSTEM PRIVATE src/extern/pcg-cpp/include src/extern/tomlplusplus)

target_include_directories(slug SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestInterpolation SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestElem SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestExtinct SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestPhot SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestPDFs SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestUtils SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestTracks SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestTracksFull SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestYields SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestCore SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestCoreFull SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestIo SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestNebular SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestFeedback SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugTestSpecsyn SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(slugPython SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})

# Eigen (header-only; vendored as a git submodule at src/extern/eigen,
# like pcg-cpp/tomlplusplus/pybind11/mdspan above, rather than requiring
# a system install -- see elem::DecayChain for the one place this
# project uses it, via <Eigen/Dense> and <unsupported/Eigen/
# MatrixFunctions>). Scoped to only the targets that actually compile a
# source transitively including DecayChain.hpp (directly, or via
# SimControls.hpp -> Yields.hpp), unlike pcg-cpp/tomlplusplus/HDF5
# above, which every target happens to need.
target_include_directories(slug SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestElem SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestExtinct SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestCore SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestCoreFull SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestIo SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestNebular SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestFeedback SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugTestSpecsyn SYSTEM PRIVATE src/extern/eigen)
target_include_directories(slugPython SYSTEM PRIVATE src/extern/eigen)

# Link to GSL
target_link_libraries(slug PRIVATE GSL::gsl)
target_link_libraries(slugTestInterpolation PRIVATE GSL::gsl)
target_link_libraries(slugTestElem PRIVATE GSL::gsl)
target_link_libraries(slugTestExtinct PRIVATE GSL::gsl)
target_link_libraries(slugTestPhot PRIVATE GSL::gsl)
target_link_libraries(slugTestPDFs PRIVATE GSL::gsl)
target_link_libraries(slugTestUtils PRIVATE GSL::gsl)
target_link_libraries(slugTestTracks PRIVATE GSL::gsl)
target_link_libraries(slugTestTracksFull PRIVATE GSL::gsl)
target_link_libraries(slugTestYields PRIVATE GSL::gsl)
target_link_libraries(slugTestCore PRIVATE GSL::gsl)
target_link_libraries(slugTestCoreFull PRIVATE GSL::gsl)
target_link_libraries(slugTestIo PRIVATE GSL::gsl)
target_link_libraries(slugTestNebular PRIVATE GSL::gsl)
target_link_libraries(slugTestFeedback PRIVATE GSL::gsl)
target_link_libraries(slugTestSpecsyn PRIVATE GSL::gsl)
target_link_libraries(slugPython PRIVATE GSL::gsl)

# Link to mdspan (see cmake/Mdspan.cmake)
target_link_libraries(slug PRIVATE slug_mdspan)
target_link_libraries(slugTestInterpolation PRIVATE slug_mdspan)
target_link_libraries(slugTestElem PRIVATE slug_mdspan)
target_link_libraries(slugTestExtinct PRIVATE slug_mdspan)
target_link_libraries(slugTestPhot PRIVATE slug_mdspan)
target_link_libraries(slugTestPDFs PRIVATE slug_mdspan)
target_link_libraries(slugTestUtils PRIVATE slug_mdspan)
target_link_libraries(slugTestTracks PRIVATE slug_mdspan)
target_link_libraries(slugTestTracksFull PRIVATE slug_mdspan)
target_link_libraries(slugTestYields PRIVATE slug_mdspan)
target_link_libraries(slugTestCore PRIVATE slug_mdspan)
target_link_libraries(slugTestCoreFull PRIVATE slug_mdspan)
target_link_libraries(slugTestIo PRIVATE slug_mdspan)
target_link_libraries(slugTestNebular PRIVATE slug_mdspan)
target_link_libraries(slugTestFeedback PRIVATE slug_mdspan)
target_link_libraries(slugTestSpecsyn PRIVATE slug_mdspan)
target_link_libraries(slugPython PRIVATE slug_mdspan)

# Link to HDF5
target_link_libraries(slug PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestInterpolation PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestElem PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestExtinct PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestPhot PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestPDFs PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestUtils PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestTracks PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestTracksFull PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestYields PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestCore PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestCoreFull PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestIo PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestNebular PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestFeedback PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugTestSpecsyn PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries(slugPython PRIVATE ${HDF5_C_LIBRARIES})

# Link to OpenMP, if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(slug PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestInterpolation PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestElem PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestExtinct PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestPhot PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestPDFs PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestUtils PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestTracks PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestTracksFull PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestYields PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestCore PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestCoreFull PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestIo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestNebular PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestFeedback PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugTestSpecsyn PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(slugPython PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link to MPI, if available -- only the slug CLI executable itself, not
# slugPython or any test target (see SLUG_MPI's own comment above and in
# src/core/SimCluster.cpp/SimGalaxy.cpp/src/io/OutputManagerH5.cpp): the
# shared sources those other targets also compile simply take the
# existing, unchanged single-rank code path when SLUG_MPI is not defined
# for them.
# OMPI_SKIP_MPICXX/MPICH_SKIP_MPICXX keep mpi.h, compiled as C++, from
# pulling in the C++ bindings' own headers (and with them a dependency on
# libmpi_cxx) -- see the find_package(MPI) comment above.
if(MPI_C_FOUND)
    target_compile_definitions(slug PRIVATE SLUG_MPI OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
    target_link_libraries(slug PRIVATE MPI::MPI_C)
endif()

# Optional: auto-regenerate a .pyi type stub for the compiled _slug
# extension (via pybind11-stubgen, https://github.com/sizmailov/pybind11-stubgen
# -- "pip install pybind11-stubgen") every time slugPython is rebuilt,
# so editors/pyright see real signatures and docstrings for _slug's
# otherwise type-opaque compiled classes instead of treating every
# attribute access as Any. Entirely optional -- slugPython builds and
# works identically without pybind11-stubgen installed; this step is
# just skipped with a message() note in that case. The generated
# slugpy/_slug.pyi is gitignored (see .gitignore): it's a build
# artifact, one-to-one with the _slug binary that produced it, not
# source to commit.
# Skipped for a pip-installed (SKBUILD) build, since it imports
# slugpy._slug from the source tree, where an SKBUILD build does not
# put it.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11_stubgen"
    RESULT_VARIABLE PYBIND11_STUBGEN_MISSING
    OUTPUT_QUIET ERROR_QUIET)
if(SKBUILD)
    # Nothing to do; see above
elseif(PYBIND11_STUBGEN_MISSING EQUAL 0)
    message(STATUS "pybind11-stubgen found: slugPython will regenerate slugpy/_slug.pyi after each build")
    # -o ${CMAKE_SOURCE_DIR}: pybind11-stubgen mirrors the dotted module
    # name ("slugpy._slug") under the output root, so this lands the
    # stub at ${CMAKE_SOURCE_DIR}/slugpy/_slug.pyi -- right next to the
    # _slug.cpython-*.so module it documents, where pyright/editors
    # expect to find it. --ignore-all-errors: a handful of SimControls
    # properties (extinct, filters, nebular, specsyn) embed raw C++
    # namespace-qualified types (e.g. "phot::FilterCollection") rather
    # than their pybind-registered Python names in their docstrings, so
    # pybind11-stubgen can't resolve those specific annotations; without
    # this flag it still generates the rest of the stub and falls back
    # to Any for just those, but prints ERROR-level noise on every
    # build. PYTHONPATH is set (rather than relying on the ambient
    # environment) since pybind11-stubgen imports the module by name
    # ("slugpy._slug"), the same way test_PythonBindings/test_Slugpy do.
    add_custom_command(TARGET slugPython POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${Python_EXECUTABLE} -m pybind11_stubgen slugpy._slug
            -o ${CMAKE_SOURCE_DIR} --ignore-all-errors
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Regenerating slugpy/_slug.pyi with pybind11-stubgen"
        VERBATIM)
else()
    message(STATUS "pybind11-stubgen not found: skipping slugpy/_slug.pyi stub generation (pip install pybind11-stubgen to enable)")
endif()

# Optional: build doc/sphinx (see its own conf.py) as part of the
# normal build -- "cmake --build build" (no explicit --target) builds
# every target added to ALL, and this "docs" target is added to ALL
# only when its own dependencies (sphinx, breathe, exhale for the
# Python interpreter CMake picked up; the doxygen executable, which
# Exhale drives to actually parse the C++ side) are all present.
# Entirely optional -- every other target builds and works identically
# without them installed; this step is just skipped with a message()
# note in that case, the same pattern as pybind11-stubgen above. The
# built site (doc/html/) is gitignored: it's a build artifact, not
# source to commit.
# Never built for a pip-installed (SKBUILD) build, which installs
# only the slug CLI and the slugpy package.
find_program(DOXYGEN_EXECUTABLE doxygen)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sphinx, breathe, exhale"
    RESULT_VARIABLE SLUG_DOCS_PYTHON_MISSING
    OUTPUT_QUIET ERROR_QUIET)
if(SKBUILD)
    # Nothing to do; see above
elseif(DOXYGEN_EXECUTABLE AND SLUG_DOCS_PYTHON_MISSING EQUAL 0)
    message(STATUS "sphinx/breathe/exhale/doxygen found: doc/html will be (re)built as part of the default build")
    # PATH: doxygen's own directory is prepended explicitly (rather
    # than relying on it already being on PATH for whatever invokes
    # the build, e.g. an IDE) since Exhale shells out to plain
    # "doxygen" by name, not the full path find_program() resolved
    # here. PYTHONPATH: same reason slugPython's own tests/pybind11-
    # stubgen step above need it -- conf.py imports slugpy for
    # autodoc, which needs the repo root on the path. WORKING_DIRECTORY
    # is doc/sphinx/ itself: conf.py's own Doxygen invocation (and the
    # relative paths inside it) are written assuming that cwd, exactly
    # as if `make html`/`sphinx-build` had been run from there by hand.
    get_filename_component(SLUG_DOXYGEN_DIR ${DOXYGEN_EXECUTABLE} DIRECTORY)
    if(WIN32)
        set(SLUG_PATH_SEP ";")
    else()
        set(SLUG_PATH_SEP ":")
    endif()
    add_custom_target(docs ALL
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=${SLUG_DOXYGEN_DIR}${SLUG_PATH_SEP}$ENV{PATH}"
            PYTHONPATH=${CMAKE_SOURCE_DIR}
            ${Python_EXECUTABLE} -m sphinx -b html -d _build/doctrees . ${CMAKE_SOURCE_DIR}/doc/html
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/sphinx
        DEPENDS slugPython
        COMMENT "Building Sphinx documentation into doc/html"
        VERBATIM)
else()
    message(STATUS "sphinx/breathe/exhale and/or doxygen not found: skipping doc/html build (pip install sphinx breathe exhale, and install doxygen, to enable)")
endif()

# Enable testing
enable_testing()

# Define tests
add_test(NAME test_Interpolation COMMAND slugTestInterpolation)
# test_Elem loads the isotope data table (data/elem/isotopes.h5) by
# relative path, so it needs to run from the source tree root.
add_test(NAME test_Elem COMMAND slugTestElem
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# test_Extinct loads the extinction curve registry (data/extinct/
# extinct.toml/.h5) by relative path, so it needs to run from the
# source tree root.
add_test(NAME test_Extinct COMMAND slugTestExtinct
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_Phot COMMAND slugTestPhot)
add_test(NAME test_PDFs COMMAND slugTestPDFs)
# test_Utils loads a real IMF file (data/imfs/chabrier.toml) directly
# by relative path for testPDFIntegrator, so it needs to run from the
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
# test_TracksFull reads data/tracks/tracks.toml by its default relative
# path, so like test_Tracks it runs from the source tree root.
add_test(NAME test_TracksFull COMMAND slugTestTracksFull
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_Yields COMMAND slugTestYields
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

# test_Nebular loads its own small nebular emission fixture (tests/
# nebular/assets/nebular_test.h5) and IMF/track files by relative
# path, so it needs to run from the source tree root, mirroring
# test_Specsyn/test_Tracks above.
add_test(NAME test_Nebular COMMAND slugTestNebular
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# test_Feedback builds a SimControls from an input deck (tests/core/
# assets/testCluster.in, with a chained spectral library from
# tests/specsyn/assets) to check Winds::vWind's Wolf-Rayet dispatch
# against a real SpecsynLibChained, so it needs the same
# WORKING_DIRECTORY and SLUG_DIR as test_Io.
add_test(NAME test_Feedback COMMAND slugTestFeedback
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_Feedback PROPERTIES
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

# MPI end-to-end test: only registered when this build actually links
# MPI (see the find_package(MPI) block above), since the SLUG_MPI-
# guarded code under test (per-rank UID striping, checkpoint sync,
# rank-aware restart lookup -- see src/utils/MPIUtils.hpp) is entirely
# absent from a non-MPI build, and mpiexec itself may not even be
# installed. Unlike test_PythonBindings/test_Slugpy above, this drives
# the slug CLI itself under mpiexec rather than the Python bindings,
# so it needs the slug executable's own path and FindMPI's own
# mpiexec/flag variables passed through as environment variables --
# see tests/mpi/test_mpi_e2e.py's own skipif for how it uses them.
if(MPI_C_FOUND)
    add_test(NAME test_MPI
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/mpi -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_MPI PROPERTIES
        ENVIRONMENT "SLUG_DIR=${CMAKE_SOURCE_DIR};SLUG_EXECUTABLE=$<TARGET_FILE:slug>;MPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE};MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        LABELS "quick")
endif()

# Tag every test "quick" or "slow", so `ctest -L quick` (or `ctest -LE
# slow`) runs only the fast suite, skipping anything that takes
# upwards of a minute to run -- currently test_CoreFull and
# test_TracksFull, but
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
    test_Interpolation test_Elem test_Extinct test_Phot test_PDFs test_Utils test_Specsyn test_Tracks
    test_Yields test_Core test_Io test_Nebular test_Feedback test_PythonBindings test_Slugpy
    PROPERTIES LABELS "quick")
set_tests_properties(test_CoreFull test_TracksFull PROPERTIES LABELS "slow")

# Add command to copy assets to build dir
add_custom_command(
    TARGET slugTestPDFs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pdfs/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets   
)

# Installation, for a pip-installed (SKBUILD) build only -- a plain
# CMake build is used in place, straight from the source/build trees,
# and has nothing to install. Everything lands inside the slugpy
# package directory (scikit-build-core's install prefix is the wheel's
# own root, so "slugpy/..." here means site-packages/slugpy/...):
#   - slugpy/_slug*.so: the compiled extension module.
#   - slugpy/_bin/slug: the CLI executable, launched by the "slug"
#     console script (slugpy/_cli.py) rather than installed onto PATH
#     directly, so that it can find the data below.
#   - slugpy/_share/...: the small data files checked into the
#     repository, laid out exactly as in the source tree (data/imfs/,
#     src/pybind/assets/, etc.), so every existing
#     utils::getFilePath(name, prefix) call resolves unchanged once
#     slugpy/_share is on SLUG_DATA_PATH (see slugpy/_paths.py). Files
#     are listed explicitly, or by a pattern matching only checked-in
#     files, so that large data a developer has downloaded into their
#     own checkout's data/ is never swept into the package.
if(SKBUILD)
    # Keep the link-time search path for shared libraries (e.g. a
    # Homebrew or system GSL/HDF5) in the installed binaries' RPATH,
    # so they still find those libraries once installed
    set_target_properties(slug slugPython PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE)

    # On Linux, keep the symbols of any static libraries linked into
    # _slug (as binary wheels do for GSL, HDF5, and zlib -- see
    # .github/wheels/build_deps.sh) out of its dynamic symbol table, so
    # they can never interpose with another copy of the same library
    # loaded into the same Python process (e.g. the libhdf5 bundled in
    # h5py's own wheel). A no-op for shared-library dependencies. macOS's
    # two-level namespace already keeps each image's symbol bindings
    # separate, so it needs no equivalent.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(slugPython PRIVATE "LINKER:--exclude-libs,ALL")
    endif()

    install(TARGETS slugPython LIBRARY DESTINATION slugpy)
    # slugpy's pure-Python half: *.py files only, so that build products
    # an in-tree build may have left in the source slugpy/ directory
    # (_slug*.so, _slug.pyi, __pycache__) are never picked up -- see
    # pyproject.toml's own wheel.packages comment
    install(DIRECTORY slugpy/ DESTINATION slugpy
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE)
    install(TARGETS slug RUNTIME DESTINATION slugpy/_bin)

    set(SLUG_SHARE slugpy/_share)
    install(FILES data/cloudy/cloudy.in_grid_template
        DESTINATION ${SLUG_SHARE}/data/cloudy)
    install(FILES data/elem/isotopes.h5
        DESTINATION ${SLUG_SHARE}/data/elem)
    install(FILES data/extinct/extinct.h5 data/extinct/extinct.toml
        DESTINATION ${SLUG_SHARE}/data/extinct)
    install(FILES data/filters/V_filter.h5 data/filters/V_filter.toml
        DESTINATION ${SLUG_SHARE}/data/filters)
    install(FILES data/spectra/vega.h5
        DESTINATION ${SLUG_SHARE}/data/spectra)
    install(DIRECTORY data/imfs/
        DESTINATION ${SLUG_SHARE}/data/imfs
        FILES_MATCHING PATTERN "*.toml")
    install(DIRECTORY data/yields/
        DESTINATION ${SLUG_SHARE}/data/yields
        FILES_MATCHING PATTERN "*.h5" PATTERN "yields.toml")
    install(FILES src/pybind/assets/PyDefaults.toml
        DESTINATION ${SLUG_SHARE}/src/pybind/assets)
endif()
