"""
Unit tests for cluster_slug: the checks it makes on the library it
reads, and its inference settings.

Uses the committed example cluster library
(examples/clusterlib/clusterlib.h5: 2000 clusters, one photometry entry
each, extincted but no nebular photometry, with a PDFs group), so needs
no data fetched separately. Tests that need a library lacking
something, or with nebular photometry, edit a copy of it in tmp_path.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import pathlib
import shutil
from collections.abc import Callable

import h5py
import numpy as np
import pytest

from slugpy import cluster_slug

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO_ROOT / "examples" / "clusterlib" / "clusterlib.h5"
LIBRARY_SIZE = 2000


@pytest.fixture(scope="module")
def lib() -> cluster_slug:
    """
    A cluster_slug on the example library, with default settings.

    Returns
    -------
    cluster_slug
        The cluster_slug.
    """
    return cluster_slug(str(LIBRARY))


def _edited_copy(tmp_path: pathlib.Path, edit: Callable[[h5py.File], None]) -> str:
    """
    Copy the example library into tmp_path, and edit the copy.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to copy into.
    edit : callable
        Function applied to the copy, opened for writing.

    Returns
    -------
    str
        Path to the edited copy.
    """
    path = tmp_path / "library.h5"
    shutil.copy(LIBRARY, path)
    with h5py.File(path, "r+") as f:
        edit(f)
    return str(path)


def test_construction_defaults(lib: cluster_slug) -> None:
    """The example library is accepted, with the documented defaults."""
    assert lib.size == LIBRARY_SIZE
    assert lib.pobs is None and lib.priors is None
    assert lib.filtersets == []
    assert lib.inference_quantities == ["mass", "age", "A_V", "metallicity"]
    assert lib.abstol == 1e-8 and lib.reltol == 1e-2
    # The example has extincted, but not nebular, photometry
    assert lib.use_extinct is True and lib.use_nebular is False
    with pytest.raises(AttributeError):
        lib.size = 3  # pyright: ignore[reportAttributeAccessIssue]


def test_rejects_unsuitable_libraries(tmp_path: pathlib.Path) -> None:
    """Libraries missing a group, or with more photometry entries than clusters, are rejected."""
    def drop(name: str) -> Callable[[h5py.File], None]:
        def edit(f: h5py.File) -> None:
            del f[name]
        return edit

    def shorten_phot(f: h5py.File) -> None:
        group = f["cluster_phot"]
        assert isinstance(group, h5py.Group)
        for name in group:
            dset = group[name]
            assert isinstance(dset, h5py.Dataset)
            dset.resize(LIBRARY_SIZE - 1, axis=0)

    for edit, what in [(drop("clusters"), "clusters"), (drop("cluster_phot"), "cluster_phot"),
                       (drop("PDFs"), "PDFs"), (shorten_phot, "different numbers")]:
        with pytest.raises(ValueError, match=what):
            cluster_slug(_edited_copy(tmp_path, edit))


def test_constructor_keywords() -> None:
    """Every setting can be set on construction."""
    pobs = np.full(LIBRARY_SIZE, 0.5)
    lib = cluster_slug(str(LIBRARY), pobs=pobs, priors=len, filtersets=[["Lbol"]],
                       inference_quantities=["mass", "Q(HI)"],
                       abstol=0.0, reltol=1e-3, use_nebular=False, use_extinct=False)
    assert isinstance(lib.pobs, np.ndarray) and np.array_equal(lib.pobs, pobs)
    assert lib.priors is len
    assert lib.filtersets == [["Lbol"]]
    assert lib.inference_quantities == ["mass", "Q(HI)"]
    assert lib.abstol == 0.0 and lib.reltol == 1e-3
    assert lib.use_nebular is False and lib.use_extinct is False


def test_pobs_and_priors_validation(lib: cluster_slug) -> None:
    """pobs and priors accept None, arrays of the right size and range, and callables."""
    lib.pobs = np.linspace(0.0, 1.0, LIBRARY_SIZE)
    lib.pobs = lambda x: x  # pyright: ignore[reportUnknownLambdaType]
    lib.pobs = None
    lib.priors = np.linspace(0.0, 5.0, LIBRARY_SIZE)
    lib.priors = None
    for name, bad_values in [("pobs", [1.5, -0.1]), ("priors", [-0.1])]:
        for bad in bad_values:
            with pytest.raises(ValueError):
                setattr(lib, name, np.full(LIBRARY_SIZE, bad))
        with pytest.raises(ValueError):
            setattr(lib, name, np.full(LIBRARY_SIZE - 1, 0.5))
        with pytest.raises(ValueError):
            setattr(lib, name, np.full((LIBRARY_SIZE, 1), 0.5))
        with pytest.raises(ValueError):
            setattr(lib, name, np.full(LIBRARY_SIZE, np.nan))
        with pytest.raises(TypeError):
            setattr(lib, name, [0.5] * LIBRARY_SIZE)
        with pytest.raises(TypeError):
            setattr(lib, name, 0.5)
        assert getattr(lib, name) is None


def test_filtersets_and_tolerances_validation(lib: cluster_slug) -> None:
    """filtersets must be a list of non-empty lists of str; tolerances non-negative reals."""
    lib.filtersets = [["Lbol"], ["HST.WFC3_UVIS1.F555W", "HST.WFC3_UVIS1.F814W"]]
    sets = lib.filtersets
    sets[0].append("x")  # the property returns a copy
    assert lib.filtersets[0] == ["Lbol"]
    for bad in ["Lbol", ["Lbol"], [("Lbol",)], [[1]]]:
        with pytest.raises(TypeError):
            lib.filtersets = bad
    with pytest.raises(ValueError):
        lib.filtersets = [[]]
    lib.filtersets = []

    lib.abstol = 1
    assert lib.abstol == 1.0 and isinstance(lib.abstol, float)
    for bad, err in [(-1.0, ValueError), (np.inf, ValueError), ("1e-3", TypeError), (True, TypeError)]:
        with pytest.raises(err):
            lib.reltol = bad
    lib.abstol, lib.reltol = 1e-8, 1e-2


def test_nebular_and_extinct_availability(tmp_path: pathlib.Path, lib: cluster_slug) -> None:
    """use_nebular/use_extinct default to what the library has, and may not be True without it."""
    with pytest.raises(ValueError, match="nebular"):
        lib.use_nebular = True
    with pytest.raises(ValueError, match="nebular"):
        cluster_slug(str(LIBRARY), use_nebular=True)
    with pytest.raises(TypeError):
        lib.use_extinct = 1
    lib.use_extinct = False
    lib.use_extinct = True

    # A copy with nebular, but no extincted, photometry
    def neb_not_ext(f: h5py.File) -> None:
        group = f["cluster_phot"]
        assert isinstance(group, h5py.Group)
        group.copy("phot_extinct", "phot_neb")
        del group["phot_extinct"]

    neb = cluster_slug(_edited_copy(tmp_path, neb_not_ext))
    assert neb.use_nebular is True and neb.use_extinct is False
    with pytest.raises(ValueError, match="extincted"):
        neb.use_extinct = True
    neb.use_nebular = False


def test_inference_quantities(lib: cluster_slug) -> None:
    """inference_quantities takes physical quantities and library filters, disjoint from filtersets."""
    lib.inference_quantities = ["age", "Q(HI)", "HST.WFC3_UVIS1.F275W"]
    q = lib.inference_quantities
    q.append("mass")  # the property returns a copy
    assert lib.inference_quantities == ["age", "Q(HI)", "HST.WFC3_UVIS1.F275W"]
    for bad, err in [("mass", TypeError), ([1], TypeError), ([], ValueError),
                     (["mass", "mass"], ValueError), (["mass", "radius"], ValueError),
                     (["NOT.A.FILTER"], ValueError)]:
        with pytest.raises(err):
            lib.inference_quantities = bad

    # A filter cannot be both inferred and used for inference, whichever is set second
    with pytest.raises(ValueError, match="Q\\(HI\\)"):
        lib.filtersets = [["Lbol"], ["Q(HI)", "Lbol"]]
    lib.filtersets = [["Lbol"]]
    with pytest.raises(ValueError, match="Lbol"):
        lib.inference_quantities = ["mass", "Lbol"]
    with pytest.raises(ValueError, match="Lbol"):
        cluster_slug(str(LIBRARY), filtersets=[["Lbol"]], inference_quantities=["Lbol"])
    lib.filtersets = []
    lib.inference_quantities = ["mass", "age", "A_V", "metallicity"]
