"""
Unit tests for KDE, the Python wrapper around the compiled
KernelDensity class: bandwidth specification (explicit and by
Silverman's rule), the zero-copy handling of the data set and weights,
evaluation against brute force, in-place changes to the weights and
bandwidth, and conditional estimates.

Uses small random data sets made here, so needs no data fetched.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import numpy as np
import pytest

from slugpy import KDE


def make_points(n: int, ndim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Random points with correlated first two dimensions, and weights.

    Parameters
    ----------
    n : int
        Number of points.
    ndim : int
        Number of dimensions.
    seed : int
        Random seed.

    Returns
    -------
    x : numpy.ndarray, shape (n, ndim)
        The points.
    w : numpy.ndarray, shape (n,)
        Weights in [0.5, 2).
    """
    rng = np.random.default_rng(seed)
    common = 3.0 * rng.standard_normal(n)
    x = rng.standard_normal((n, ndim))
    x[:, :2] = common[:, None] + 0.2 * rng.standard_normal((n, 2))
    return np.ascontiguousarray(x), rng.uniform(0.5, 2.0, n)


def brute_pdf(x: np.ndarray, w: np.ndarray, m: np.ndarray, q: np.ndarray) -> float:
    """
    Normalized Gaussian KDE exp(-d^T M d) at a point, by brute force.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Kernel centers.
    w : numpy.ndarray, shape (n,)
        Weights.
    m : numpy.ndarray, shape (d, d)
        Inverse squared bandwidth matrix.
    q : numpy.ndarray, shape (d,)
        The point.

    Returns
    -------
    float
        The estimate at q.
    """
    d = x - q
    r2 = np.einsum("ij,jk,ik->i", d, m, d)
    norm = np.sqrt(np.linalg.det(m)) / np.pi ** (0.5 * x.shape[1])
    return float(norm * np.sum(w * np.exp(-r2)) / np.sum(w))


def test_bandwidth_forms() -> None:
    """h2Inv may be a float, a diagonal, or a matrix."""
    x, _ = make_points(200, 3, 1)
    assert np.array_equal(KDE(x.copy(), h2Inv=2.0).h2Inv, 2.0 * np.eye(3))
    assert np.array_equal(KDE(x.copy(), h2Inv=np.array([1.0, 2.0, 3.0])).h2Inv, np.diag([1.0, 2.0, 3.0]))
    m = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.array_equal(KDE(x.copy(), h2Inv=m).h2Inv, m)
    with pytest.raises(ValueError):
        KDE(x.copy(), h2Inv=np.ones(2))
    with pytest.raises(TypeError):
        KDE(x.copy(), h2Inv=[1.0, 2.0, 3.0])  # pyright: ignore[reportArgumentType]


def test_silverman_bandwidth() -> None:
    """The automatic bandwidth is Silverman's rule, with weights."""
    x, w = make_points(500, 3, 2)
    x0, w0 = x.copy(), w.copy()
    kde = KDE(x, w=w)
    neff = w0.sum() ** 2 / (w0 ** 2).sum()
    factor = (neff * 5.0 / 4.0) ** (-1.0 / 7.0)
    cov = np.cov(x0, rowvar=False, aweights=w0) * factor ** 2
    assert np.allclose(kde.h2Inv, np.linalg.inv(cov) / 2.0)
    # Tiny (but valid) weights, whose squares underflow, give the same bandwidth
    z = x0.copy()
    tiny = KDE(z, w=1e-200 * w0 / w0.max())
    assert np.allclose(tiny.h2Inv, np.linalg.inv(cov) / 2.0)
    with pytest.raises(ValueError):
        KDE(x0.copy(), w=np.zeros(500))

    # Unweighted: neff = N
    y, _ = make_points(300, 2, 3)
    y0 = y.copy()
    cov_u = np.cov(y0, rowvar=False) * ((300 * 4.0 / 4.0) ** (-1.0 / 6.0)) ** 2
    assert np.allclose(KDE(y).h2Inv, np.linalg.inv(cov_u) / 2.0)


def test_data_handling() -> None:
    """dataset is reordered in place and locked; w is normalized in place and editable."""
    x, w = make_points(400, 3, 4)
    x0, w0 = x.copy(), w.copy()
    kde = KDE(x, h2Inv=1.0, w=w, names=["a", "b", "c"])
    assert kde.dataset is x and kde.w is w
    assert np.array_equal(x, x0[kde.aux])
    assert np.allclose(w, (w0 / w0.sum())[kde.aux])
    assert kde.nDim == 3 and kde.names == ["a", "b", "c"] and kde.marginal is None
    with pytest.raises(ValueError):
        x[0, 0] = 1.0  # the data set is locked, whichever reference is used
    with pytest.raises(ValueError):
        kde.dataset[0, 0] = 1.0
    kde.w[0] = kde.w[0]  # weights are editable in place
    kde.h2Inv[0, 0] = kde.h2Inv[0, 0]
    kde.names[0] = "A"
    assert kde.names == ["A", "b", "c"]
    for attr in ["dataset", "w", "h2Inv", "names", "aux"]:
        with pytest.raises(AttributeError):
            setattr(kde, attr, None)

    # Argument checks
    with pytest.raises(ValueError):
        KDE(make_points(50, 3, 5)[0], h2Inv=1.0, names=["a"])
    with pytest.raises(TypeError):
        KDE(make_points(50, 3, 5)[0].astype(np.float32), h2Inv=1.0)
    with pytest.raises(ValueError):
        KDE(make_points(50, 3, 5)[0], h2Inv=1.0, w=np.ones(49))

    # Weights sharing memory with the data set (here, its only column)
    one_d = np.random.default_rng(5).uniform(1.0, 2.0, (50, 1))
    with pytest.raises(ValueError, match="share memory"):
        KDE(one_d, h2Inv=1.0, w=one_d[:, 0])


def test_pdf_matches_brute_force() -> None:
    """pdf() matches brute force, and follows in-place changes to w (after reweight()) and h2Inv."""
    x, w = make_points(1500, 3, 6)
    x0, w0 = x.copy(), w.copy()
    m = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.2], [0.0, 0.2, 1.5]])
    kde = KDE(x, h2Inv=m, w=w, leafSize=8)
    q = 1.5 * np.random.default_rng(7).standard_normal((8, 3))
    want = np.array([brute_pdf(x0, w0, m, qi) for qi in q])
    got = kde.pdf(q, 1e-3, 0.0)
    assert isinstance(got, np.ndarray)
    assert np.all(np.abs(got - want) <= 1e-3 * got * (1 + 1e-9))
    assert kde.pdf(q[0], 0.0, 0.0) == pytest.approx(want[0], rel=1e-12)
    default = kde.pdf(q[0])  # relTol = 1e-2, absTol = 1e-8
    assert isinstance(default, float) and abs(default - want[0]) <= 1e-2 * default

    # New weights, in place
    kde.w[:] = np.where(kde.aux % 2 == 0, 1.0, 0.0)
    kde.reweight()
    assert kde.w.sum() == pytest.approx(1.0)
    w_new = np.where(np.arange(1500) % 2 == 0, 1.0, 0.0)
    assert kde.pdf(q[0], 0.0, 0.0) == pytest.approx(brute_pdf(x0, w_new, m, q[0]), rel=1e-12)
    kde.w[:] = 0.0
    with pytest.raises(ValueError):
        kde.reweight()
    kde.w[:] = 1.0
    kde.reweight()

    # A new bandwidth, in place
    kde.h2Inv[:] = 3.0 * np.eye(3)
    assert kde.pdf(q[1], 0.0, 0.0) == pytest.approx(brute_pdf(x0, np.ones(1500), 3.0 * np.eye(3), q[1]), rel=1e-12)


def test_conditional() -> None:
    """conditional() returns a KDE over the kept dimensions that reproduces the full estimate."""
    x, w = make_points(1500, 4, 8)
    m = np.linalg.inv(0.2 * np.eye(4) + 0.8 * np.ones((4, 4)))
    kde = KDE(x, h2Inv=m, w=w, names=["mass", "age", "A_V", "Z"])
    x0 = np.array([0.4, np.nan, 0.9, np.nan])
    cond = kde.conditional(x0, 0.0)
    assert isinstance(cond, KDE)
    assert cond.names == ["age", "Z"] and cond.nDim == 2
    assert cond.dataset.shape[1] == 2 and cond.w.sum() == pytest.approx(1.0)
    marg = kde.pdf(x0, 0.0, 0.0)
    assert cond.marginal == pytest.approx(marg, rel=1e-10)
    y = np.random.default_rng(9).standard_normal((5, 2))
    full = kde.pdf(np.column_stack([np.full(5, 0.4), y[:, 0], np.full(5, 0.9), y[:, 1]]), 0.0, 0.0)
    cond_pdf = cond.pdf(y, 0.0, 0.0)
    assert cond.marginal is not None
    assert np.allclose(cond.marginal * np.asarray(cond_pdf), full, rtol=1e-10)
    with pytest.raises(ValueError):
        cond.dataset[0, 0] = 0.0

    # Marginalizing, pruning, and conditioning again
    pruned = kde.conditional(np.array([0.4, np.nan, 0.9, 0.0]), marginalDims=[3])  # massTol = 1e-2
    assert pruned.names == ["age"] and 0 < pruned.dataset.shape[0] < cond.dataset.shape[0]
    again = cond.conditional(np.array([np.nan, 0.2]), 0.0)
    assert again.names == ["age"] and again.nDim == 1

    # The conditional's weights can be changed in place too
    cond.w[:] = 1.0
    cond.reweight()
    assert cond.w.sum() == pytest.approx(1.0)
