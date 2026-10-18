"""
Check that an installed macOS slugpy wheel shares the OpenMP runtime of a
conda-forge environment, rather than loading a second one.

conda-forge's numpy uses an OpenBLAS built with OpenMP, which loads
conda's llvm-openmp; the wheel bundles its own libomp. Two OpenMP
runtimes in one process abort ("OMP: Error #15") or crash, so the wheel
loads libomp as @rpath/libomp.dylib, sharing whichever is loaded first
(see share_libomp.py). This runs OpenMP code in both numpy and slugpy, in
the given order of loading their runtimes, and checks that exactly one
libomp is loaded.

Usage: conda_libomp_test.py {numpy-first,slugpy-first}

Run by .github/workflows/wheels.yml in a conda-forge environment with the
wheel installed. Fails (rather than passing vacuously) if numpy does not
load a libomp at all, i.e. if the environment no longer tests anything.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import ctypes
import importlib.util
import pathlib
import sys


def loaded_libomps() -> list[str]:
    """The paths of every libomp image loaded in this process."""
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libsystem._dyld_image_count.restype = ctypes.c_uint32
    libsystem._dyld_get_image_name.restype = ctypes.c_char_p
    libsystem._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = (libsystem._dyld_get_image_name(i) for i in range(libsystem._dyld_image_count()))
    return [n.decode() for n in names if n is not None and n.decode().endswith("/libomp.dylib")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("order", choices=["numpy-first", "slugpy-first"],
                        help="which package's OpenMP runtime to load first")
    order = parser.parse_args().order

    if order == "slugpy-first":
        # Load the wheel's bundled libomp, as importing _slug would, before
        # numpy (which slugpy itself imports first) gets the chance
        spec = importlib.util.find_spec("slugpy")
        assert spec is not None and spec.submodule_search_locations
        bundled = pathlib.Path(next(iter(spec.submodule_search_locations))) / ".dylibs" / "libomp.dylib"
        ctypes.CDLL(str(bundled), mode=ctypes.RTLD_GLOBAL)

    import numpy as np

    rng = np.random.default_rng(0)
    np.linalg.svd(rng.standard_normal((1000, 1000)))  # OpenMP threads in OpenBLAS
    if not loaded_libomps():
        sys.exit("conda_libomp_test.py: numpy loaded no libomp, so this tests nothing; "
                 "is OpenBLAS the OpenMP build?")

    from slugpy import KDTree

    KDTree(rng.standard_normal((1_000_000, 3)))  # OpenMP threads in slugpy
    np.linalg.svd(rng.standard_normal((1000, 1000)))

    libomps = loaded_libomps()
    print(f"{order}: loaded libomp images: {libomps}")
    if len(libomps) != 1:
        sys.exit(f"conda_libomp_test.py: expected one libomp loaded, found {len(libomps)}")


if __name__ == "__main__":
    main()
