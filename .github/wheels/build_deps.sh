#!/usr/bin/env bash
#
# Build slug's native dependencies (zlib, GSL, HDF5 with its C++ API) from
# source as static, position-independent libraries, installed into a single
# prefix, for building binary wheels (see .github/workflows/wheels.yml).
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
# (set by the workflow) controls the minimum macOS version the libraries
# are built for.
#
# :copyright: Copyright (c) 2026 Mark Krumholz

set -euo pipefail

PREFIX="${1:-/tmp/slug-deps}"
ZLIB_VERSION=1.3.1
GSL_VERSION=2.8
HDF5_VERSION=1.14.6

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

fetch() {
    # fetch OUTPUT URL [URL...] -- tries each URL in turn, each with
    # retries (including on connection errors), since a transient download
    # failure would otherwise fail a whole wheel build
    local out="$1"
    shift
    for url in "$@"; do
        if curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -o "${out}" "${url}"; then
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

touch "${PREFIX}/.complete"
echo "build_deps.sh: installed into ${PREFIX}"
