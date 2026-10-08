# Container image for slug: the slug CLI (with hybrid MPI + OpenMP
# support) and the slugpy Python package, built by .github/workflows/
# container.yml and published to ghcr.io/markkrumholz/slug3. Usable
# directly with Docker, or on HPC systems via Apptainer/Singularity
# (`apptainer pull docker://ghcr.io/markkrumholz/slug3:<version>`).
#
# slug is installed exactly as `pip install .` installs it (see
# pyproject.toml), into a virtual environment at /opt/slug, but with MPI
# enabled (SLUG_ENABLE_MPI, off by default for pip installs). MPI is Open
# MPI 4.1 (Ubuntu 24.04's), whose libmpi ABI (libmpi.so.40) is shared by
# every Open MPI 4.1.x release, so that on a cluster the container can be
# launched by the host's own Open MPI 4.1 mpirun (Apptainer's "hybrid"
# model) or have the host's libmpi bound in (its "bind" model).
#
# Large data (stellar tracks, spectral libraries, nebular tables) is not
# included: bind-mount a host directory at /data (SLUG_DIR, below), holding
# a data/ subdirectory as fetched by `python -m slugpy.download_data`.
#
# :copyright: Copyright (c) 2026 Mark Krumholz

ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates g++-14 cmake ninja-build \
        libgsl-dev libhdf5-dev libopenmpi-dev openmpi-bin \
        python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY . /src
WORKDIR /src

# The build context has no usable git metadata, so the commit hash to
# embed in output files (see CMakeLists.txt's own SLUG_GIT_HASH comment)
# is passed in as a build argument instead, via .slug_git_hash
ARG SLUG_GIT_HASH=
RUN if [ -n "${SLUG_GIT_HASH}" ]; then echo "${SLUG_GIT_HASH}" > .slug_git_hash; fi

RUN python3 -m venv /opt/slug \
    && CC=gcc-14 CXX=g++-14 CMAKE_GENERATOR=Ninja \
       CMAKE_ARGS="-DSLUG_ENABLE_MPI=ON -DCMAKE_CXX_SCAN_FOR_MODULES=OFF" \
       /opt/slug/bin/pip install --no-cache-dir -v .

# ---------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION}

LABEL org.opencontainers.image.source="https://github.com/markkrumholz/slug3"
LABEL org.opencontainers.image.description="slug: stochastic stellar population synthesis (slug CLI with MPI + OpenMP, and the slugpy Python package)"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
        libgsl27 libhdf5-103-1t64 libhdf5-cpp-103-1t64 \
        libopenmpi3t64 openmpi-bin \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/slug /opt/slug

# Fail the image build, rather than the first user's run, if any shared
# library the compiled pieces need is missing from the runtime stage
RUN for f in /opt/slug/lib/python3*/site-packages/slugpy/_bin/slug \
             /opt/slug/lib/python3*/site-packages/slugpy/_slug*.so; do \
        if ldd "$f" | grep "not found"; then echo "missing libraries for $f"; exit 1; fi; \
    done \
    && /opt/slug/bin/python -c "import slugpy; print('slugpy imports OK')"

ENV PATH=/opt/slug/bin:${PATH} \
    SLUG_DIR=/data
WORKDIR /work
CMD ["bash"]
