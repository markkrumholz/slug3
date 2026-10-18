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
from typing import cast

import h5py
import numpy as np
import pytest
from astropy import units as u

from slugpy import Bandwidth, cluster_slug

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


def _values(data: object) -> np.ndarray:
    """
    The numerical values of a reader's dataset.

    Parameters
    ----------
    data : astropy.units.Quantity or numpy.ndarray
        The dataset.

    Returns
    -------
    numpy.ndarray
        Its values, without units.
    """
    return np.asarray(cast(u.Quantity, data).value if isinstance(data, u.Quantity) else data)


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
    assert len(lib.filtersets) == 0 and len(lib.kdes) == 0
    assert lib.inference_quantities == ["mass", "age", "A_V", "metallicity"]
    assert lib.abstol == 1e-8 and lib.reltol == 1e-2
    assert lib.bw_phot == 0.1 and lib.bw_phys == 0.1
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
    """Every setting can be set on construction; filtersets may be one list or a list of lists."""
    pobs = np.full(LIBRARY_SIZE, 0.5)
    priors = np.full(LIBRARY_SIZE, 2.0)
    lib = cluster_slug(str(LIBRARY), pobs=pobs, priors=priors, filtersets=["Lbol"],
                       inference_quantities=["mass", "Q(HI)"], bw_phot=0.2, bw_phys=0.05,
                       abstol=0.0, reltol=1e-3, use_nebular=False, use_extinct=False)
    assert isinstance(lib.pobs, np.ndarray) and np.array_equal(lib.pobs, pobs)
    assert isinstance(lib.priors, np.ndarray) and np.array_equal(lib.priors, priors)
    assert list(lib.filtersets) == [("Lbol",)]
    assert lib.inference_quantities == ["mass", "Q(HI)"]
    assert lib.abstol == 0.0 and lib.reltol == 1e-3
    assert lib.bw_phot == 0.2 and lib.bw_phys == 0.05
    assert lib.use_nebular is False and lib.use_extinct is False

    two = cluster_slug(str(LIBRARY), filtersets=[["Lbol"], ["HST.WFC3_UVIS1.F555W", "HST.WFC3_UVIS1.F814W"]])
    assert list(two.filtersets) == [("Lbol",), ("HST.WFC3_UVIS1.F555W", "HST.WFC3_UVIS1.F814W")]
    assert cluster_slug(str(LIBRARY), filtersets=[]).kdes == {}
    for bad in ["Lbol", [("Lbol",)], [[1]], [["Lbol"], "Lbol"]]:
        with pytest.raises(TypeError):
            cluster_slug(str(LIBRARY), filtersets=bad)  # pyright: ignore[reportArgumentType]


def test_pobs_and_priors_validation(lib: cluster_slug) -> None:
    """pobs and priors accept None, arrays of the right size and range, and callables."""
    lib.pobs = np.linspace(0.0, 1.0, LIBRARY_SIZE)
    lib.pobs = lambda m, a, av, feh: np.full(LIBRARY_SIZE, 0.5)  # pyright: ignore[reportUnknownLambdaType]
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


def test_tolerances_validation(lib: cluster_slug) -> None:
    """Tolerances must be non-negative reals."""
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
    """inference_quantities takes physical quantities and library filters, disjoint from filter sets."""
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
        lib.add_filterset(["Q(HI)", "Lbol"])
    lib.inference_quantities = ["mass", "age"]
    lib.add_filterset(["Lbol"])
    with pytest.raises(ValueError, match="Lbol"):
        lib.inference_quantities = ["mass", "Lbol"]
    with pytest.raises(ValueError, match="Lbol"):
        cluster_slug(str(LIBRARY), filtersets=[["Lbol"]], inference_quantities=["Lbol"])
    lib.remove_filterset(["Lbol"])
    lib.inference_quantities = ["mass", "age", "A_V", "metallicity"]


def test_singular_quantities(tmp_path: pathlib.Path) -> None:
    """Quantities drawn from an absent or delta PDF are left out by default, and cannot be requested."""
    def singular(f: h5py.File) -> None:
        pdfs = f["PDFs"]
        assert isinstance(pdfs, h5py.Group)
        pdfs.attrs["fehDist"] = pdfs.attrs["avNebFac"]  # a delta distribution
        del pdfs.attrs["outTimeDist"]                   # absent

    path = _edited_copy(tmp_path, singular)
    lib = cluster_slug(path)
    assert lib.pdfs["fehDist"].getMin() == lib.pdfs["fehDist"].getMax()
    assert lib.inference_quantities == ["mass", "A_V"]
    for bad in [["mass", "metallicity"], ["age"]]:
        with pytest.raises(ValueError, match="cannot infer"):
            lib.inference_quantities = bad
        with pytest.raises(ValueError, match="cannot infer"):
            cluster_slug(path, inference_quantities=bad)
    # Filters may still be inferred
    lib.inference_quantities = ["mass", "Q(HI)"]

    # Priors are divided only by the non-singular PDFs
    lib.inference_quantities = ["mass", "A_V"]
    lib.priors = np.ones(LIBRARY_SIZE)
    lib.add_filterset(["HST.WFC3_UVIS1.F555W"])
    kde = lib.kdes[("HST.WFC3_UVIS1.F555W",)]
    clusters = lib.clusters
    assert clusters is not None
    mass = _values(clusters["target_mass"])
    av = _values(clusters["A_V"])
    want = 1.0 / (lib.pdfs["cmf"](mass) * lib.pdfs["avDist"](av))
    want = want[kde.aux] / want[kde.aux].sum()
    assert np.allclose(kde.w, want)


def test_add_filterset_dataset() -> None:
    """add_filterset() builds the dataset as specified: transforms, fallback, row filtering, layout."""
    lib = cluster_slug(str(LIBRARY))
    filters = ["HST.WFC3_UVIS1.F555W", "Q(HI)", "Lbol"]
    lib.add_filterset(filters)
    kde = lib.kdes[tuple(filters)]
    assert kde.names == filters + ["mass", "age", "A_V", "metallicity"]
    phot, clusters = lib.cluster_phot, lib.clusters
    assert phot is not None and clusters is not None

    # Expected columns, in library order: extincted F555W (mag, unlogged);
    # Q(HI), whose extincted values are all zero, so plain, logged; Lbol,
    # which has no extincted variant, so plain, logged
    with np.errstate(divide="ignore"):
        expected = np.column_stack([
            _values(phot["HST.WFC3_UVIS1.F555W_ex"]),
            np.log10(_values(phot["Q(HI)"])),
            np.log10(_values(phot["Lbol"])),
            np.log10(_values(clusters["target_mass"])),
            np.log10(_values(phot["time"])),
            _values(clusters["A_V"]),
            np.asarray(clusters["feh"]),
        ])
    assert np.all(_values(phot["Q(HI)_ex"]) == 0.0)

    # Clusters with a non-finite value (here, the one with Q(HI) = 0) are left out
    valid = np.all(np.isfinite(expected), axis=1)
    assert np.sum(~valid) == 1
    assert kde.dataset.shape == (np.sum(valid), 7)
    assert sorted(kde.aux.tolist()) == np.nonzero(valid)[0].tolist()
    assert np.array_equal(kde.dataset, expected[kde.aux])
    assert kde.w.sum() == pytest.approx(1.0) and np.allclose(kde.w, kde.w[0])

    # The tree is never split along the inference quantities
    split_dims = {n.splitDim for n in kde.kernel_density.nodes()}
    assert split_dims <= {-1, 0, 1, 2}

    # Without extinction, the plain photometry
    lib.use_extinct = False
    lib.add_filterset(["HST.WFC3_UVIS1.F555W"])
    plain = lib.kdes[("HST.WFC3_UVIS1.F555W",)]
    assert np.array_equal(plain.dataset[:, 0], _values(phot["HST.WFC3_UVIS1.F555W"])[plain.aux])


def test_add_filterset_fallback(tmp_path: pathlib.Path) -> None:
    """Where the chosen variant is NaN, photometry falls back to the next variant, cluster by cluster."""
    nan_rows = np.arange(0, LIBRARY_SIZE, 7)

    def nan_extinct(f: h5py.File) -> None:
        dset = f["cluster_phot"]["phot_extinct"]
        assert isinstance(dset, h5py.Dataset)
        data = dset[()]
        data[nan_rows, 3] = np.nan  # HST.WFC3_UVIS1.F555W
        dset[...] = data

    lib = cluster_slug(_edited_copy(tmp_path, nan_extinct), filtersets=["HST.WFC3_UVIS1.F555W"])
    kde = lib.kdes[("HST.WFC3_UVIS1.F555W",)]
    phot = lib.cluster_phot
    assert phot is not None
    ext = _values(phot["HST.WFC3_UVIS1.F555W_ex"])
    plain = _values(phot["HST.WFC3_UVIS1.F555W"])
    want = np.where(np.isnan(ext), plain, ext)
    assert kde.dataset.shape[0] == LIBRARY_SIZE
    assert np.array_equal(kde.dataset[:, 0], want[kde.aux])
    assert np.any(np.isin(kde.aux, nan_rows))


def test_add_filterset_weights() -> None:
    """Weights combine pobs and priors over the sampling density; callables get astropy Quantities."""
    rng = np.random.default_rng(1)
    pobs = rng.uniform(0.0, 1.0, LIBRARY_SIZE)
    pobs[:100] = 0.0  # never observed: kept, with no weight, so that reweighting can restore them
    lib = cluster_slug(str(LIBRARY), pobs=pobs, filtersets=["HST.WFC3_UVIS1.F555W"])
    kde = lib.kdes[("HST.WFC3_UVIS1.F555W",)]
    assert kde.dataset.shape[0] == LIBRARY_SIZE
    assert np.allclose(kde.w, pobs[kde.aux] / pobs[kde.aux].sum())
    assert np.all(kde.w[kde.aux < 100] == 0.0)

    # Callable pobs and priors, with the expected argument types
    seen: list[tuple[type, ...]] = []

    def prior(mass: u.Quantity, age: u.Quantity, av: u.Quantity, feh: np.ndarray) -> np.ndarray:
        seen.append((type(mass), type(age), type(av), type(feh)))
        assert mass.unit == u.Msun and age.unit == u.yr and av.unit == u.mag
        return np.asarray(mass.value) ** -1.0

    lib.pobs = None
    lib.priors = prior
    lib.add_filterset(["HST.WFC3_UVIS1.F814W"])
    assert seen == [(u.Quantity, u.Quantity, u.Quantity, np.ndarray)]
    clusters, phot = lib.clusters, lib.cluster_phot
    assert clusters is not None and phot is not None
    mass = _values(clusters["target_mass"])
    sampling = (lib.pdfs["cmf"](mass) * lib.pdfs["outTimeDist"](_values(phot["time"]))
                * lib.pdfs["avDist"](_values(clusters["A_V"]))
                * lib.pdfs["fehDist"](np.asarray(clusters["feh"])))
    want = mass ** -1.0 / sampling
    kde = lib.kdes[("HST.WFC3_UVIS1.F814W",)]
    assert np.allclose(kde.w, want[kde.aux] / want[kde.aux].sum())


def test_kdes_and_removal() -> None:
    """kdes and filtersets are read-only; add_filterset() validates; remove_filterset() removes."""
    lib = cluster_slug(str(LIBRARY), filtersets=[["Lbol"], ["HST.WFC3_UVIS1.F555W"]])
    assert set(lib.kdes) == {("Lbol",), ("HST.WFC3_UVIS1.F555W",)}
    with pytest.raises(TypeError):
        lib.kdes[("x",)] = lib.kdes[("Lbol",)]  # pyright: ignore[reportIndexIssue]
    with pytest.raises(AttributeError):
        lib.filtersets = []  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        lib.kdes = {}  # pyright: ignore[reportAttributeAccessIssue]
    for bad, err in [("Lbol", TypeError), ([1], TypeError), ([], ValueError),
                     (["Lbol", "Lbol"], ValueError), (["NOT.A.FILTER"], ValueError),
                     (["mass"], ValueError)]:
        with pytest.raises(err):
            lib.add_filterset(bad)  # pyright: ignore[reportArgumentType]

    # An explicit bandwidth is used as given
    lib.add_filterset(["Lbol"], h2Inv=2.0)
    assert np.array_equal(lib.kdes[("Lbol",)].h2Inv, 2.0 * np.eye(5))

    lib.remove_filterset(["Lbol"])
    assert list(lib.filtersets) == [("HST.WFC3_UVIS1.F555W",)]
    with pytest.raises(KeyError):
        lib.remove_filterset(["Lbol"])


def test_default_bandwidth() -> None:
    """The default h2Inv is diagonal from bw_phot/bw_phys, and follows changes to them."""
    lib = cluster_slug(str(LIBRARY), inference_quantities=["mass", "age", "Q(HI)"],
                       bw_phot=0.2, bw_phys=0.05)
    filters = ["HST.WFC3_UVIS1.F555W", "Lbol"]
    lib.add_filterset(filters)
    lib.add_filterset(["HST.WFC3_UVIS1.F814W"], h2Inv=3.0)
    kde = lib.kdes[tuple(filters)]
    assert kde.names == filters + ["mass", "age", "Q(HI)"]

    def expected(bw_phot: float, bw_phys: float) -> np.ndarray:
        # F555W in mag; Lbol and Q(HI) stored as log10, so 0.4 bw_phot dex
        return np.diag(1.0 / np.square([bw_phot, 0.4 * bw_phot, bw_phys, bw_phys, 0.4 * bw_phot]))

    assert np.allclose(kde.h2Inv, expected(0.2, 0.05), rtol=1e-14)

    # Changing either bandwidth updates default-bandwidth estimates in
    # place, and their evaluation uses the new bandwidth; estimates with
    # an explicit h2Inv are left alone
    x = kde.dataset[7].copy()
    before = kde.pdf(x, relTol=0.0, absTol=0.0)
    lib.bw_phot = 0.1
    lib.bw_phys = 0.2
    assert np.allclose(kde.h2Inv, expected(0.1, 0.2), rtol=1e-14)
    assert np.array_equal(lib.kdes[("HST.WFC3_UVIS1.F814W",)].h2Inv, 3.0 * np.eye(4))
    after = kde.pdf(x, relTol=0.0, absTol=0.0)
    ref = cluster_slug(str(LIBRARY), inference_quantities=["mass", "age", "Q(HI)"],
                       bw_phot=0.1, bw_phys=0.2, filtersets=[filters])
    assert after == pytest.approx(ref.kdes[tuple(filters)].pdf(x, relTol=0.0, absTol=0.0), rel=1e-12)
    assert after != pytest.approx(before)

    # Re-adding with an explicit h2Inv stops it following the
    # bandwidths; re-adding with the default starts again
    lib.add_filterset(filters, h2Inv=5.0)
    lib.bw_phot = 0.3
    assert np.array_equal(lib.kdes[tuple(filters)].h2Inv, 5.0 * np.eye(5))
    lib.add_filterset(filters)
    assert np.allclose(lib.kdes[tuple(filters)].h2Inv, expected(0.3, 0.2), rtol=1e-14)
    lib.remove_filterset(filters)
    lib.bw_phys = 0.1

    for bad, err in [(0.0, ValueError), (-0.1, ValueError), (np.inf, ValueError),
                     (np.nan, ValueError), ("0.1", TypeError), (True, TypeError)]:
        for name in ["bw_phot", "bw_phys"]:
            with pytest.raises(err):
                setattr(lib, name, bad)
    assert lib.bw_phot == 0.3 and lib.bw_phys == 0.1


def test_sampling_density() -> None:
    """sampling_density is the product of the library's PDFs, cached and read-only."""
    lib = cluster_slug(str(LIBRARY))
    clusters, phot = lib.clusters, lib.cluster_phot
    assert clusters is not None and phot is not None
    want = (lib.pdfs["cmf"](_values(clusters["target_mass"]))
            * lib.pdfs["outTimeDist"](_values(phot["time"]))
            * lib.pdfs["avDist"](_values(clusters["A_V"]))
            * lib.pdfs["fehDist"](_values(clusters["feh"])))
    assert np.allclose(lib.sampling_density, want)
    with pytest.raises(ValueError):
        lib.sampling_density[0] = 1.0
    with pytest.raises(AttributeError):
        lib.sampling_density = want  # pyright: ignore[reportAttributeAccessIssue]


def test_reweighting() -> None:
    """Setting pobs or priors reweights every KDE in place, matching a KDE built with those weights."""
    filters = ["HST.WFC3_UVIS1.F555W", "HST.WFC3_UVIS1.F814W"]
    other = ["Lbol"]
    h2inv = 4.0 * np.eye(6)
    lib = cluster_slug(str(LIBRARY))
    lib.add_filterset(filters, h2Inv=h2inv)
    lib.add_filterset(other)
    kde = lib.kdes[tuple(filters)]
    rng = np.random.default_rng(2)
    q = kde.dataset[rng.choice(kde.dataset.shape[0], 5, replace=False)] + 0.05

    def fresh(**settings: object) -> np.ndarray:
        # The same KDE, built from scratch with the given pobs/priors
        ref = cluster_slug(str(LIBRARY), **settings)  # pyright: ignore[reportArgumentType]
        ref.add_filterset(filters, h2Inv=h2inv)
        return np.asarray(ref.kdes[tuple(filters)].pdf(q, 0.0, 0.0))

    # pobs as an array; zero for some clusters
    pobs = rng.uniform(0.0, 1.0, LIBRARY_SIZE)
    pobs[::3] = 0.0
    lib.pobs = pobs
    assert np.allclose(kde.w, pobs[kde.aux] / pobs[kde.aux].sum())
    assert np.allclose(np.asarray(kde.pdf(q, 0.0, 0.0)), fresh(pobs=pobs), rtol=1e-10)
    other_kde = lib.kdes[tuple(other)]
    assert np.allclose(other_kde.w, pobs[other_kde.aux] / pobs[other_kde.aux].sum())

    # Callable priors: called once on setting, and not again when pobs changes
    calls: list[int] = []

    def prior(mass: u.Quantity, age: u.Quantity, av: u.Quantity, feh: np.ndarray) -> np.ndarray:
        calls.append(1)
        return np.asarray(mass.value) ** -0.5

    lib.priors = prior
    assert len(calls) == 1
    mass = _values(lib.clusters["target_mass"]) if lib.clusters is not None else None
    assert mass is not None
    want = pobs * mass ** -0.5 / lib.sampling_density
    assert np.allclose(kde.w, want[kde.aux] / want[kde.aux].sum())
    assert np.allclose(np.asarray(kde.pdf(q, 0.0, 0.0)), fresh(pobs=pobs, priors=prior), rtol=1e-10)
    n_calls = len(calls)  # fresh() calls prior too, for its own cluster_slug
    lib.pobs = np.ones(LIBRARY_SIZE)  # restores the clusters pobs had zeroed
    assert len(calls) == n_calls
    want = mass ** -0.5 / lib.sampling_density
    assert np.allclose(kde.w, want[kde.aux] / want[kde.aux].sum())
    assert np.all(kde.w > 0.0)

    # Back to no weights: every cluster equal
    lib.pobs = None
    lib.priors = None
    assert np.allclose(kde.w, 1.0 / kde.w.size)
    assert np.allclose(np.asarray(kde.pdf(q, 0.0, 0.0)), fresh(), rtol=1e-10)

    # A change that would leave an estimate with no weight is refused, changing nothing
    before = kde.w.copy()
    with pytest.raises(ValueError, match="no weight"):
        lib.pobs = np.zeros(LIBRARY_SIZE)
    assert lib.pobs is None and np.array_equal(kde.w, before)

    # A callable's result is checked like an array
    with pytest.raises(ValueError, match="callable"):
        lib.pobs = lambda m, a, av, feh: np.full(LIBRARY_SIZE, 2.0)  # pyright: ignore[reportUnknownLambdaType]
    assert lib.pobs is None


def test_conditional_filterset_selection() -> None:
    """conditional() uses the only filter set, requires one when there are several, and adds new ones."""
    lib = cluster_slug(str(LIBRARY))
    x0 = np.full(5, np.nan)
    with pytest.raises(ValueError, match="filterset must be given"):
        lib.conditional(x0)
    lib.add_filterset(["Lbol"])
    kde = lib.kdes[("Lbol",)]
    x0[0] = float(np.median(kde.dataset[:, 0]))
    cond = lib.conditional(x0, massTol=0.0)
    direct = kde.conditional(x0, massTol=0.0)
    assert cond.names == direct.names == ["mass", "age", "A_V", "metallicity"]
    assert cond.marginal == pytest.approx(direct.marginal, rel=1e-12)
    assert np.array_equal(cond.h2Inv, direct.h2Inv)

    # A second filter set: now one must be named; naming a new one adds it
    lib.add_filterset(["HST.WFC3_UVIS1.F555W"])
    with pytest.raises(ValueError, match="filterset must be given"):
        lib.conditional(x0)
    assert lib.conditional(x0, massTol=0.0, filterset=["Lbol"]).marginal == pytest.approx(cond.marginal)
    phot = lib.cluster_phot
    assert phot is not None
    y0 = np.full(6, np.nan)
    y0[0] = float(np.median(_values(phot["HST.WFC3_UVIS1.F814W_ex"])))
    y0[1] = float(np.median(_values(phot["HST.WFC3_UVIS1.F555W_ex"])))
    lib.conditional(y0, filterset=["HST.WFC3_UVIS1.F814W", "HST.WFC3_UVIS1.F555W"])
    assert ("HST.WFC3_UVIS1.F814W", "HST.WFC3_UVIS1.F555W") in lib.filtersets
    with pytest.raises(TypeError):
        lib.conditional(x0, filterset="Lbol")  # pyright: ignore[reportArgumentType]


def test_conditional_errors() -> None:
    """Photometric errors convolve the bandwidth as (M^-1 + 2 C)^-1 before conditioning."""
    filters = ["HST.WFC3_UVIS1.F555W", "HST.WFC3_UVIS1.F814W"]
    lib = cluster_slug(str(LIBRARY), filtersets=filters)
    kde = lib.kdes[tuple(filters)]
    x0 = np.full(6, np.nan)
    x0[:2] = kde.dataset[17, :2] + 0.05
    m = kde.h2Inv

    def expected(c_phot: np.ndarray) -> tuple[np.ndarray, float]:
        c = np.zeros((6, 6))
        c[:2, :2] = c_phot
        mtot = np.linalg.inv(np.linalg.inv(m) + 2.0 * c)
        marginal = kde.pdf(x0, 0.0, 0.0, bandwidth=Bandwidth(0.5 * (mtot + mtot.T)))
        return mtot[2:, 2:], float(np.asarray(marginal))

    # Same error in every band, one per band, and the equivalent covariance
    cond = lib.conditional(x0, massTol=0.0, errors=0.1)
    m_rr, marginal = expected(np.diag([0.01, 0.01]))
    assert np.allclose(cond.h2Inv, m_rr) and cond.marginal == pytest.approx(marginal, rel=1e-10)
    per_band = lib.conditional(x0, massTol=0.0, errors=np.array([0.1, 0.2]))
    m_rr, marginal = expected(np.diag([0.01, 0.04]))
    assert np.allclose(per_band.h2Inv, m_rr) and per_band.marginal == pytest.approx(marginal, rel=1e-10)
    via_cov = lib.conditional(x0, massTol=0.0, cov=np.diag([0.01, 0.04]))
    assert np.allclose(via_cov.h2Inv, per_band.h2Inv) and via_cov.marginal == pytest.approx(per_band.marginal)

    # Just the photometry: the inference quantities are kept automatically
    short = lib.conditional(x0[:2], massTol=0.0, errors=0.1)
    assert short.names == ["mass", "age", "A_V", "metallicity"]
    assert np.array_equal(short.h2Inv, cond.h2Inv) and short.marginal == pytest.approx(cond.marginal, rel=1e-12)
    for bad_x0 in [x0[:1], x0[:3], np.zeros(7)]:
        with pytest.raises(ValueError, match="one element per band"):
            lib.conditional(bad_x0)

    # Correlated errors
    c_corr = np.array([[0.04, 0.03], [0.03, 0.04]])
    corr = lib.conditional(x0, massTol=0.0, cov=c_corr)
    m_rr, marginal = expected(c_corr)
    assert np.allclose(corr.h2Inv, m_rr) and corr.marginal == pytest.approx(marginal, rel=1e-10)

    # Units: magnitudes for these bands
    as_quantity = lib.conditional(x0, massTol=0.0, errors=0.1 * u.mag)
    assert as_quantity.marginal == pytest.approx(cond.marginal)
    lib.conditional(x0, massTol=0.0, cov=np.diag([0.01, 0.04]) * u.mag ** 2)
    for bad in [0.1 * u.dex, 0.1 * u.Lsun, 0.1 * u.ABmag]:
        with pytest.raises(ValueError, match="does not match"):
            lib.conditional(x0, errors=bad)
    with pytest.raises(ValueError, match="must be in"):
        lib.conditional(x0, cov=np.diag([0.01, 0.04]) * u.dex ** 2)

    # Shapes and values
    for bad in [np.array([0.1, 0.2, 0.3]), np.array([[0.1, 0.2]]), -0.1, np.nan]:
        with pytest.raises(ValueError):
            lib.conditional(x0, errors=bad)
    with pytest.raises(ValueError):
        lib.conditional(x0, cov=np.eye(3))
    with pytest.raises(ValueError, match="not both"):
        lib.conditional(x0, errors=0.1, cov=np.eye(2))
    with pytest.raises(ValueError):
        lib.conditional(x0, cov=-np.eye(2))


def test_conditional_error_units_for_linear_bands() -> None:
    """Errors for bands not in magnitudes are in dex, of the log10 photometry the KDE stores."""
    lib = cluster_slug(str(LIBRARY), filtersets=[["Lbol"], ["HST.WFC3_UVIS1.F555W", "Lbol"]])
    kde = lib.kdes[("Lbol",)]
    x0 = np.full(5, np.nan)
    x0[0] = float(np.median(kde.dataset[:, 0]))
    plain = lib.conditional(x0, massTol=0.0, filterset=["Lbol"], errors=0.1)
    for ok in [0.1 * u.dex, 0.1 * u.dex(u.Lsun), 0.1 * u.dex(u.erg / u.s)]:
        assert lib.conditional(x0, massTol=0.0, filterset=["Lbol"], errors=ok).marginal == pytest.approx(plain.marginal)
    lib.conditional(x0, filterset=["Lbol"], cov=np.array([[0.01]]) * u.dex ** 2)
    for bad in [0.1 * u.mag, 0.1 * u.Lsun, 0.1 * u.dex(u.erg / u.s / u.Hz)]:
        with pytest.raises(ValueError, match="does not match"):
            lib.conditional(x0, filterset=["Lbol"], errors=bad)

    # A filter set mixing magnitudes and other units: a Quantity cov is ambiguous
    mixed = lib.kdes[("HST.WFC3_UVIS1.F555W", "Lbol")]
    y0 = np.full(6, np.nan)
    y0[:2] = mixed.dataset[5, :2]
    filterset = ["HST.WFC3_UVIS1.F555W", "Lbol"]
    lib.conditional(y0, filterset=filterset, cov=np.diag([0.01, 0.01]))
    with pytest.raises(ValueError, match="mixes"):
        lib.conditional(y0, filterset=filterset, cov=np.diag([0.01, 0.01]) * u.mag ** 2)
    with pytest.raises(ValueError, match="does not match"):
        lib.conditional(y0, filterset=filterset, errors=0.1 * u.mag)
