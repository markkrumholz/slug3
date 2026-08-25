# Mdspan.cmake
#
# Provides the slug_mdspan INTERFACE target, which every target that uses
# <mdspan> links against. Application code never needs to know or care
# which of the following actually ended up providing it -- it always just
# does #include <mdspan> and uses std::mdspan/std::extents/std::dextents.
#
# In priority order:
#
# 1. Native: the compiler's own standard library already ships C++23's
#    <mdspan> (true of GCC >= 16, or Clang built against a recent libc++;
#    see AGENTS.md). Detected with a real compile check rather than a
#    version-number guess, so this also transparently covers any future
#    toolchain, or one already patched/backported by a distro.
# 2. Package: an externally-installed mdspan CONFIG package (found via
#    find_package(mdspan)) -- e.g. one a cluster's module system or Spack
#    provides, potentially a vendor-tuned build that generates better SIMD
#    code than the reference implementation's own generic templates.
# 3. Submodule: the Kokkos reference implementation, vendored as a git
#    submodule at src/extern/mdspan, built from source. This is the same
#    implementation options 1 and 2 are themselves normally either a
#    native adoption of, or a package build of, so its own behavior is
#    the most reliable fallback when neither of those is available (e.g.
#    building on a cluster with an older GCC, with no network access to
#    fetch anything and no mdspan package installed).
#
# Override the automatic priority with -DSLUG_MDSPAN_PROVIDER=<value>,
# one of AUTO (default), NATIVE, PACKAGE, or SUBMODULE -- e.g. to force
# exercising the submodule fallback in CI even on a toolchain that does
# have native support, or to get a hard failure instead of a silent
# fallback if a specific provider is expected to be available.

set(SLUG_MDSPAN_PROVIDER "AUTO" CACHE STRING
    "Which mdspan implementation to use: AUTO (native, then an externally \
installed package, then the bundled src/extern/mdspan submodule, in that \
order), NATIVE (require the compiler's own <mdspan>), PACKAGE (require \
find_package(mdspan)), or SUBMODULE (always build the bundled reference \
implementation).")
set_property(CACHE SLUG_MDSPAN_PROVIDER PROPERTY STRINGS AUTO NATIVE PACKAGE SUBMODULE)

add_library(slug_mdspan INTERFACE)

set(_slug_mdspan_provider "")

if(SLUG_MDSPAN_PROVIDER MATCHES "^(AUTO|NATIVE)$")
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles([[
        #include <mdspan>
        int main() {
            double data[6] = {0, 1, 2, 3, 4, 5};
            const std::mdspan<double, std::extents<std::size_t, 2, 3>> m(data);
            return static_cast<int>(m[0, 0]);
        }
    ]] SLUG_HAVE_NATIVE_MDSPAN)
    if(SLUG_HAVE_NATIVE_MDSPAN)
        set(_slug_mdspan_provider "native compiler support")
    elseif(SLUG_MDSPAN_PROVIDER STREQUAL "NATIVE")
        message(FATAL_ERROR "mdspan: SLUG_MDSPAN_PROVIDER=NATIVE was requested, but the compiler "
            "has no native <mdspan> (check_cxx_source_compiles failed).")
    endif()
endif()

if(NOT _slug_mdspan_provider AND SLUG_MDSPAN_PROVIDER MATCHES "^(AUTO|PACKAGE)$")
    find_package(mdspan CONFIG QUIET)
    if(TARGET mdspan::mdspan)
        set(_slug_mdspan_provider "externally-installed package (found via find_package(mdspan))")
    elseif(SLUG_MDSPAN_PROVIDER STREQUAL "PACKAGE")
        message(FATAL_ERROR "mdspan: SLUG_MDSPAN_PROVIDER=PACKAGE was requested, but "
            "find_package(mdspan) did not find one.")
    endif()
endif()

if(NOT _slug_mdspan_provider)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/extern/mdspan/CMakeLists.txt")
        message(FATAL_ERROR "mdspan: no native or externally-installed implementation is available, "
            "and the src/extern/mdspan submodule isn't checked out -- run "
            "'git submodule update --init --recursive' and re-run cmake.")
    endif()
    # This project's own tests/examples/benchmarks are irrelevant here --
    # we only want its mdspan::mdspan INTERFACE target.
    set(MDSPAN_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(MDSPAN_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MDSPAN_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    add_subdirectory(src/extern/mdspan)
    set(_slug_mdspan_provider "bundled reference implementation (src/extern/mdspan submodule)")
endif()

if(TARGET mdspan::mdspan)
    # Neither the package nor the submodule provider gives us a literal
    # <mdspan> header in namespace std -- generate a one-line shim so
    # every translation unit throughout the codebase can keep writing the
    # standard #include <mdspan> / std::mdspan spelling completely
    # unconditionally, the same as it would against a native
    # implementation.
    #
    # The shim deliberately does NOT use this library's own legacy
    # <experimental/mdspan> front header with MDSPAN_IMPL_STANDARD_NAMESPACE
    # forced to std (that header actually defaults to exactly that): opening
    # literal namespace std to hold full class/function template definitions
    # collides with libc++'s own internal namespace machinery on some
    # platforms (observed on Apple Clang/libc++ -- unrelated <memory>
    # internals started resolving std::forward as std::std::forward and
    # failing to compile). Instead, it includes the newer, namespaced
    # <mdspan/mdspan.hpp> front header under a private, harmless namespace
    # name, then pulls only the handful of names slug actually uses into
    # std via ordinary `using` declarations -- much narrower, and safe
    # regardless of what libc++/libstdc++ itself is doing with std
    # internally.
    set(_slug_mdspan_shim_dir "${CMAKE_BINARY_DIR}/generated/mdspan_shim")
    file(MAKE_DIRECTORY "${_slug_mdspan_shim_dir}")
    file(WRITE "${_slug_mdspan_shim_dir}/mdspan" [[
#pragma once
// Auto-generated by cmake/Mdspan.cmake -- see that file for why this exists.
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE slug_mdspan_backport
#endif
#include <mdspan/mdspan.hpp>
namespace std {
using slug_mdspan_backport::mdspan;
using slug_mdspan_backport::extents;
using slug_mdspan_backport::dextents;
using slug_mdspan_backport::layout_left;
using slug_mdspan_backport::layout_right;
using slug_mdspan_backport::layout_stride;
using slug_mdspan_backport::default_accessor;
using slug_mdspan_backport::dynamic_extent;
} // namespace std
]])
    target_include_directories(slug_mdspan SYSTEM INTERFACE "${_slug_mdspan_shim_dir}")
    target_link_libraries(slug_mdspan INTERFACE mdspan::mdspan)
endif()

message(STATUS "mdspan: using ${_slug_mdspan_provider}")
