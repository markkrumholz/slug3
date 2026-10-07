#!/usr/bin/env bash
#
# Build slug's native dependencies (zlib, GSL, HDF5 with its C++ API) from
# source as static, position-independent libraries, installed into a single
# prefix, for building binary wheels (see .github/workflows/wheels.yml). On
# macOS, also build LLVM's OpenMP runtime (libomp), which Apple's compiler
# lacks, as a shared library that delocate then bundles into each wheel.
#
# Static libraries are linked straight into the _slug extension module and
# the slug executable, so a wheel carries no separate copies of these
# libraries -- in particular, no second libhdf5 shared library alongside
# the one h5py's own wheel bundles. zlib is needed because slug's data
# files are gzip-compressed HDF5.
#
# Usage: build_deps.sh [PREFIX]   (default: /tmp/slug-deps)
#
# Run by cibuildwheel's before-all step: inside the manylinux container on
# Linux, and directly on the runner on macOS, where MACOSX_DEPLOYMENT_TARGET
# (set in pyproject.toml) controls the minimum macOS version the libraries
# are built for. libomp is built from source rather than taken from
# Homebrew for exactly this reason: Homebrew's libomp targets whatever
# macOS version the runner itself runs, and delocate would then retag the
# wheel to require that version.
#
# :copyright: Copyright (c) 2026 Mark Krumholz

set -euo pipefail

PREFIX="${1:-/tmp/slug-deps}"
ZLIB_VERSION=1.3.1
GSL_VERSION=2.8
HDF5_VERSION=1.14.6
LLVM_VERSION=19.1.7   # libomp, macOS only

if [ -f "${PREFIX}/.complete" ]; then
    echo "build_deps.sh: ${PREFIX} already complete, nothing to do"
    exit 0
fi

if command -v nproc > /dev/null; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.ncpu)
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
cd "${WORK}"

export CFLAGS="${CFLAGS:-} -O2 -fPIC"
export CXXFLAGS="${CXXFLAGS:-} -O2 -fPIC"

# curl's --retry-all-errors (which also retries connection failures) is
# only in curl >= 7.71, newer than the curl in the manylinux_2_28 image
CURL_RETRY=(--retry 5 --retry-delay 5)
if curl --help all 2> /dev/null | grep -q -- --retry-all-errors; then
    CURL_RETRY+=(--retry-all-errors)
fi

fetch() {
    # fetch OUTPUT URL [URL...] -- tries each URL in turn, each with
    # retries, since a transient download failure would otherwise fail a
    # whole wheel build
    local out="$1"
    shift
    for url in "$@"; do
        if curl -fsSL "${CURL_RETRY[@]}" -o "${out}" "${url}"; then
            return 0
        fi
        echo "build_deps.sh: download from ${url} failed" >&2
    done
    return 1
}

echo "=== zlib ${ZLIB_VERSION}"
fetch zlib.tar.gz \
    "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz" \
    "https://zlib.net/fossils/zlib-${ZLIB_VERSION}.tar.gz"
tar xzf zlib.tar.gz
(cd "zlib-${ZLIB_VERSION}" && ./configure --static --prefix="${PREFIX}" && make -j"${JOBS}" && make install)

echo "=== GSL ${GSL_VERSION}"
fetch gsl.tar.gz \
    "https://ftpmirror.gnu.org/gsl/gsl-${GSL_VERSION}.tar.gz" \
    "https://ftp.gnu.org/gnu/gsl/gsl-${GSL_VERSION}.tar.gz" \
    "https://mirrors.kernel.org/gnu/gsl/gsl-${GSL_VERSION}.tar.gz"
tar xzf gsl.tar.gz
(cd "gsl-${GSL_VERSION}" && ./configure --prefix="${PREFIX}" --disable-shared --enable-static --with-pic \
    && make -j"${JOBS}" && make install)

echo "=== HDF5 ${HDF5_VERSION}"
fetch hdf5.tar.gz \
    "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz"
tar xzf hdf5.tar.gz
cmake -S "hdf5-${HDF5_VERSION}" -B hdf5-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DHDF5_BUILD_CPP_LIB=ON \
    -DHDF5_BUILD_HL_LIB=OFF \
    -DHDF5_BUILD_FORTRAN=OFF \
    -DHDF5_BUILD_JAVA=OFF \
    -DHDF5_BUILD_TOOLS=OFF \
    -DHDF5_BUILD_UTILS=OFF \
    -DHDF5_BUILD_EXAMPLES=OFF \
    -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
    -DZLIB_USE_EXTERNAL=OFF \
    -DZLIB_ROOT="${PREFIX}" \
    -DZLIB_USE_STATIC_LIBS=ON \
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
    -DHDF5_ALLOW_EXTERNAL_SUPPORT=NO
cmake --build hdf5-build --parallel "${JOBS}"
cmake --install hdf5-build

if [ "$(uname)" = "Darwin" ]; then
    echo "=== libomp (LLVM ${LLVM_VERSION})"
    # A standalone OpenMP runtime build needs LLVM's shared CMake modules
    # alongside it, in a sibling directory named "cmake"
    LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
    fetch openmp.tar.xz "${LLVM_URL}/openmp-${LLVM_VERSION}.src.tar.xz"
    fetch llvm-cmake.tar.xz "${LLVM_URL}/cmake-${LLVM_VERSION}.src.tar.xz"
    tar xf openmp.tar.xz
    tar xf llvm-cmake.tar.xz
    mv "cmake-${LLVM_VERSION}.src" cmake
    cmake -S "openmp-${LLVM_VERSION}.src" -B openmp-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DLIBOMP_INSTALL_ALIASES=OFF \
        -DLIBOMP_OMPD_SUPPORT=OFF \
        -DLIBOMP_USE_HWLOC=OFF \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DOPENMP_ENABLE_OMPT_TOOLS=OFF
    cmake --build openmp-build --parallel "${JOBS}"
    cmake --install openmp-build
fi

touch "${PREFIX}/.complete"
echo "build_deps.sh: installed into ${PREFIX}"
