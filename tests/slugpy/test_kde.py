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

from slugpy import KDE, Bandwidth


def _one(result: KDE | list[KDE]) -> KDE:
    """
    A single-point conditional estimate, narrowed from the batch type.

    Parameters
    ----------
    result : KDE or list of KDE
        What conditional() returned for one point.

    Returns
    -------
    KDE
        result, checked to be a single KDE.
    """
    assert isinstance(result, KDE)
    return result


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

    # Explicit aux labels are reordered with the points
    y, _ = make_points(100, 2, 10)
    y0 = y.copy()
    labels = np.arange(1000, 1100, dtype=np.int64)
    labeled = KDE(y, h2Inv=1.0, aux=labels)
    assert np.array_equal(labeled.aux, labels) and np.array_equal(y, y0[labels - 1000])

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
    cond = _one(kde.conditional(x0, 0.0))
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
    pruned = _one(kde.conditional(np.array([0.4, np.nan, 0.9, 0.0]), marginalDims=[3]))  # massTol = 1e-2
    assert pruned.names == ["age"] and 0 < pruned.dataset.shape[0] < cond.dataset.shape[0]
    again = _one(cond.conditional(np.array([np.nan, 0.2]), 0.0))
    assert again.names == ["age"] and again.nDim == 1

    # The conditional's weights can be changed in place too
    cond.w[:] = 1.0
    cond.reweight()
    assert cond.w.sum() == pytest.approx(1.0)


def test_batches_with_errors() -> None:
    """pdf() and conditional() pass per-point errors through; conditional() takes a batch of points."""
    x, w = make_points(1500, 4, 21)
    m = np.linalg.inv(0.2 * np.eye(4) + 0.8 * np.ones((4, 4)))
    kde = KDE(x, h2Inv=m, w=w, names=["mass", "age", "A_V", "Z"])
    rng = np.random.default_rng(22)
    x0 = np.column_stack([rng.normal(0.0, 1.0, 6), np.full(6, np.nan),
                          rng.normal(0.0, 1.0, 6), np.full(6, np.nan)])
    cov = np.stack([np.diag([0.01 * (i + 1), 0.02]) for i in range(6)])
    err_dims = [0, 2]

    # A batch of conditionals: a list of KDEs, each named and as for its point alone
    batch = kde.conditional(x0, 0.0)
    assert isinstance(batch, list) and len(batch) == 6
    for xi, cond in zip(x0, batch, strict=True):
        assert cond.names == ["age", "Z"]
        assert cond.marginal == pytest.approx(_one(kde.conditional(xi, 0.0)).marginal, rel=1e-12)

    # Errors: each point's marginal is pdf() at it with the same errors
    with_errors = kde.conditional(x0, 0.0, errDims=err_dims, errCov=cov)
    densities = np.asarray(kde.pdf(x0, 0.0, 0.0, errDims=err_dims, errCov=cov))
    assert isinstance(with_errors, list)
    for i, cond in enumerate(with_errors):
        assert cond.marginal == pytest.approx(densities[i], rel=1e-10)
    full = np.zeros((4, 4))
    full[np.ix_(err_dims, err_dims)] = cov[2]
    one = kde.pdf(x0[2], 0.0, 0.0, bandwidth=Bandwidth(kde.h2Inv).convolveCov(full))
    assert densities[2] == pytest.approx(float(np.asarray(one)), rel=1e-10)
    with pytest.raises(ValueError, match="without errCov"):
        kde.pdf(x0, errDims=err_dims)


def test_conditional_sample() -> None:
    """conditional_sample() gives conditional()'s points and weights, from which it can be rebuilt."""
    x, w = make_points(1500, 4, 31)
    m = np.linalg.inv(0.2 * np.eye(4) + 0.8 * np.ones((4, 4)))
    kde = KDE(x, h2Inv=m, w=w, names=["mass", "age", "A_V", "Z"])
    x0 = np.array([0.4, np.nan, 0.9, np.nan])
    cond = _one(kde.conditional(x0))
    sample = kde.conditional_sample(x0)
    assert isinstance(sample, tuple)
    pts, wts, aux = sample
    assert pts.shape == (cond.dataset.shape[0], 2) and wts.shape == (pts.shape[0],)
    assert np.array_equal(np.sort(aux), np.sort(cond.aux))
    rebuilt = KDE(pts.copy(), h2Inv=cond.h2Inv, w=wts.copy(), aux=aux.copy())
    y = np.random.default_rng(32).standard_normal((5, 2))
    assert np.allclose(np.asarray(rebuilt.pdf(y, 0.0, 0.0)), np.asarray(cond.pdf(y, 0.0, 0.0)), rtol=1e-12)

    batch = kde.conditional_sample(np.stack([x0, x0 + np.array([0.2, 0.0, -0.1, 0.0])]))
    assert isinstance(batch, list) and len(batch) == 2
    assert all(np.array_equal(a, b) for a, b in zip(batch[0], sample, strict=True))


def test_neighbors() -> None:
    """neighbors() finds the nearest points in the bandwidth's metric, indexing dataset."""
    x, w = make_points(1000, 3, 41)
    m = np.diag([4.0, 1.0, 9.0])
    kde = KDE(x, h2Inv=m, w=w)
    q = np.array([[0.1, -0.2, 0.3], [1.0, 0.5, -0.5]])
    idx, dist2 = kde.neighbors(q, 4)
    assert idx.shape == dist2.shape == (2, 4)
    for i in range(2):
        d = kde.dataset - q[i]
        want = np.sort(np.einsum("ij,jk,ik->i", d, m, d))[:4]
        assert np.allclose(dist2[i], want, rtol=1e-12)
        dd = kde.dataset[idx[i]] - q[i]
        assert np.allclose(np.einsum("ij,jk,ik->i", dd, m, dd), dist2[i], rtol=1e-12)
    idx_r, dist2_r = kde.neighbors(q, np.array([1, 3]), bandwidth=Bandwidth.identity(3))
    assert idx_r.shape == (2, 3) and np.all(idx_r[0, 1:] == -1) and np.all(np.isnan(dist2_r[0, 1:]))


def test_marginalize() -> None:
    """marginalize() returns a KDE over the remaining dimensions that reproduces the marginal density."""
    x, w = make_points(1500, 4, 51)
    m = np.linalg.inv(0.2 * np.eye(4) + 0.8 * np.ones((4, 4)))
    kde = KDE(x, h2Inv=m, w=w, names=["mass", "age", "A_V", "Z"])
    marg = kde.marginalize([2])
    assert marg.names == ["mass", "age", "Z"] and marg.nDim == 3 and marg.marginal is None
    assert marg.dataset.shape == (1500, 3) and marg.w.sum() == pytest.approx(1.0)
    want_metric = np.linalg.inv(np.linalg.inv(m)[np.ix_([0, 1, 3], [0, 1, 3])])
    assert np.allclose(marg.h2Inv, want_metric, rtol=1e-12)
    y = np.random.default_rng(52).standard_normal((4, 3))
    full = np.column_stack([y[:, 0], y[:, 1], np.full(4, np.nan), y[:, 2]])
    assert np.allclose(np.asarray(marg.pdf(y, 0.0, 0.0)), np.asarray(kde.pdf(full, 0.0, 0.0)), rtol=1e-10)
    # Marginalizing again, and over nothing
    again = marg.marginalize([0, 1])
    assert again.names == ["Z"]
    same = kde.marginalize([])
    assert np.allclose(np.asarray(same.pdf(x[:3], 0.0, 0.0)), np.asarray(kde.pdf(x[:3], 0.0, 0.0)), rtol=1e-12)
    with pytest.raises(ValueError):
        kde.marginalize([0, 1, 2, 3])
