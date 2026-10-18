"""
kde.py

Implements KDE, a Gaussian kernel density estimate over a numpy data set,
wrapping the compiled KernelDensity class.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from typing import Any, cast

import numpy as np

from ._slug import Bandwidth, KernelDensity, KernelDensityView


def _parse_h2inv(h2Inv: Any, ndim: int) -> np.ndarray:
    """
    Convert an inverse squared bandwidth specification to a matrix.

    Parameters
    ----------
    h2Inv : float or numpy.ndarray
        A float (the same value for every diagonal element), an array
        of ndim values (the diagonal), or an ndim x ndim matrix.
    ndim : int
        Number of dimensions.

    Returns
    -------
    numpy.ndarray
        The ndim x ndim matrix, as a new float64 array.

    Raises
    ------
    TypeError
        If h2Inv is neither a real number nor a numpy array.
    ValueError
        If an array has the wrong shape.
    """
    if isinstance(h2Inv, (int, float, np.integer, np.floating)) and not isinstance(h2Inv, bool):
        return float(h2Inv) * np.eye(ndim)
    if not isinstance(h2Inv, np.ndarray):
        raise TypeError("h2Inv must be None, a float, or a numpy array")
    arr = np.array(h2Inv, dtype=np.float64)
    if arr.shape == (ndim,):
        return np.diag(arr)
    if arr.shape == (ndim, ndim):
        return arr
    raise ValueError(f"h2Inv must be a float, or an array of shape ({ndim},) or ({ndim}, {ndim})")


def _silverman_h2inv(dataset: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute an inverse squared bandwidth matrix by Silverman's rule.

    Parameters
    ----------
    dataset : numpy.ndarray, shape (N, d)
        The points.
    w : numpy.ndarray, shape (N,)
        Their weights.

    Returns
    -------
    numpy.ndarray
        The d x d matrix M = Sigma^-1 / 2, where Sigma is the weighted
        covariance of the data times the square of Silverman's factor
        (n_eff (d + 2) / 4)^(-1 / (d + 4)), n_eff = (sum w)^2 / sum w^2;
        the factor scales the bandwidth, so its square scales the
        covariance. The 1/2 converts the covariance of the Gaussian
        exp(-d^T Sigma^-1 d / 2) to the kernel exp(-d^T M d).

    Raises
    ------
    ValueError
        If the weights are not finite and non-negative, or are all zero,
        or the kernel covariance is singular (e.g. too few points, or a
        dimension with no spread).
    """
    d = dataset.shape[1]
    # Scale the weights by the largest first, so that squaring tiny (but
    # valid) weights cannot underflow
    w_max = float(np.max(w)) if w.size > 0 else 0.0
    if not (np.all(np.isfinite(w)) and np.all(w >= 0.0) and w_max > 0.0):
        raise ValueError("weights must be finite and non-negative, and not all zero")
    ws = w / w_max
    cov = np.atleast_2d(np.cov(dataset, rowvar=False, aweights=ws))
    neff = float(np.sum(ws)) ** 2 / float(np.sum(ws ** 2))
    factor = (neff * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    kernel_cov = cov * factor ** 2
    try:
        h2inv = np.linalg.inv(kernel_cov) / 2.0
    except np.linalg.LinAlgError as err:
        raise ValueError("cannot compute a bandwidth: the data covariance is singular") from err
    return 0.5 * (h2inv + h2inv.T)


class KDE:
    """
    A Gaussian kernel density estimate over a numpy data set.

    Wraps the compiled KernelDensity class: the estimate at x is
    p(x) = sqrt(det M) / pi^(d / 2) sum_i w_i exp(-(x - x_i)^T M (x - x_i)),
    with the weights normalized to sum to one and M = h2Inv the inverse
    squared bandwidth matrix, evaluated to within a tolerance by a kd
    tree.

    The data set is not copied: building the estimate reorders dataset
    (and w) in place, and dataset is then made read-only, since the tree
    is built around it; aux maps the reordered points back to their
    original indices. The weights may be changed in place, after which
    reweight() must be called; h2Inv may also be changed in place, and
    takes effect on the next evaluation.

    Parameters
    ----------
    dataset : numpy.ndarray of float64, shape (N, nDim)
        The data set; C-contiguous and writeable, with every value
        finite. Reordered in place, then made read-only.
    h2Inv : float or numpy.ndarray, optional
        The inverse squared bandwidth matrix: a float (the same value
        for every diagonal element, with zero off-diagonal elements), an
        array of nDim values (the diagonal), or an nDim x nDim matrix.
        If None (the default), computed from the data by Silverman's
        rule (see Notes).
    w : numpy.ndarray of float64, shape (N,), optional
        The weight of each point: C-contiguous, writeable, finite,
        non-negative, and not all zero. Reordered in place along with
        dataset, and normalized in place to sum to one. If None (the
        default), equal weights.
    names : list of str, optional
        Names of the nDim quantities in the data set (default: none).
    aux : numpy.ndarray of int64, shape (N,), optional
        An integer label for each point (e.g. its row in a library on
        disk), C-contiguous and writeable; reordered in place along with
        dataset. If None (the default), each point's index in dataset as
        given.
    noSplitDims : list of int, optional
        Dimensions along which the kd tree is never split (default:
        none), as for KernelDensity.
    leafSize : int, optional
        Maximum number of points in a leaf of the kd tree (default 16).

    Raises
    ------
    TypeError
        If an argument has the wrong type (in particular, if dataset or
        w is not a float64 numpy array, which would need copying).
    ValueError
        If an argument has the wrong shape or invalid values, or the
        automatic bandwidth cannot be computed.

    Attributes
    ----------
    dataset : numpy.ndarray of float, shape (N, nDim)
        The data set, in tree order (read-only array).
    w : numpy.ndarray of float, shape (N,)
        The weights, in tree order, normalized to sum to one. Elements
        may be changed; call reweight() afterwards.
    h2Inv : numpy.ndarray of float, shape (nDim, nDim)
        The inverse squared bandwidth matrix. Elements may be changed;
        the change takes effect on the next evaluation.
    names : list of str
        Names of the quantities in the data set (empty if not given).
    aux : numpy.ndarray of int64, shape (N,)
        Each point's label, in tree order (read-only array): by default
        its index in the data set as originally given; for a KDE from
        conditional(), its parent's label.
    nDim : int
        Number of dimensions.
    marginal : float or None
        For a KDE from conditional(), the parent's marginal density over
        the fixed dimensions at the fixed values; otherwise None.
    kernel_density : KernelDensityView
        The underlying compiled kernel density estimate.

    Notes
    -----
    The automatic bandwidth follows Silverman's rule: the kernel
    covariance is the weighted covariance of the data times f^2, with
    f = (n_eff (d + 2) / 4)^(-1 / (d + 4)), d = nDim, and n_eff =
    (sum w)^2 / sum w^2 the effective number of points; then
    h2Inv = Sigma^-1 / 2, since the kernel is exp(-d^T h2Inv d). The
    kd tree's split dimensions are chosen with this same bandwidth.
    """

    def __init__(self, dataset: np.ndarray, h2Inv: float | np.ndarray | None = None,
                 w: np.ndarray | None = None, names: list[str] | None = None,
                 aux: np.ndarray | None = None, noSplitDims: list[int] | None = None,
                 leafSize: int = 16) -> None:
        if not isinstance(dataset, np.ndarray) or dataset.ndim != 2:
            raise TypeError("dataset must be a 2D numpy array")
        n, ndim = dataset.shape
        weights = np.ones(n) if w is None else w
        if not isinstance(weights, np.ndarray):
            raise TypeError("w must be a numpy array")
        if weights.shape != (n,):
            raise ValueError(f"w must have shape ({n},)")
        self._names: list[str] = self._check_names(names, ndim)
        self._h2Inv: np.ndarray = (_silverman_h2inv(dataset, weights) if h2Inv is None
                                   else _parse_h2inv(h2Inv, ndim))
        bandwidth = Bandwidth(self._h2Inv)
        self._kde: KernelDensityView = KernelDensity(
            dataset, weights, aux, leafSize, bandwidth, [] if noSplitDims is None else noSplitDims)
        self._bandwidth: Bandwidth = bandwidth
        self._bandwidth_h2Inv: np.ndarray = self._h2Inv.copy()
        dataset.setflags(write=False)
        self._dataset: np.ndarray = dataset
        self._w: np.ndarray = weights
        self._aux: np.ndarray = self._kde.aux
        self._marginal: float | None = None
        self._owner: Any = None

    @classmethod
    def _from_kernel_density(cls, kde: KernelDensityView, h2Inv: np.ndarray,
                             names: list[str], marginal: float | None, owner: Any) -> "KDE":
        """
        Wrap an existing compiled kernel density estimate.

        Parameters
        ----------
        kde : KernelDensityView
            The estimate.
        h2Inv : numpy.ndarray, shape (nDim, nDim)
            Its inverse squared bandwidth matrix.
        names : list of str
            Names of its dimensions (empty or nDim of them).
        marginal : float or None
            Value of the marginal property.
        owner : Any
            Object that owns kde's points, kept alive with the KDE.

        Returns
        -------
        KDE
            A KDE whose dataset, w, and aux view kde's own points.
        """
        self = cls.__new__(cls)
        ndim = kde.nDim()
        self._names = cls._check_names(names, ndim)
        self._h2Inv = np.array(h2Inv, dtype=np.float64)
        self._kde = kde
        self._bandwidth = Bandwidth(self._h2Inv)
        self._bandwidth_h2Inv = self._h2Inv.copy()
        self._dataset = kde.x
        self._w = kde.w
        self._aux = kde.aux
        self._marginal = marginal
        self._owner = owner
        return self

    @staticmethod
    def _check_names(names: Any, ndim: int) -> list[str]:
        """
        Check a list of dimension names.

        Parameters
        ----------
        names : list of str or None
            The names; None means none.
        ndim : int
            Number of dimensions.

        Returns
        -------
        list of str
            The names (an empty list if None).

        Raises
        ------
        TypeError
            If names is not a list of str.
        ValueError
            If names is neither empty nor of length ndim.
        """
        if names is None:
            return []
        if not isinstance(names, list) or not all(isinstance(n, str) for n in cast(list[Any], names)):
            raise TypeError("names must be a list of str")
        checked = cast(list[str], names)
        if len(checked) not in (0, ndim):
            raise ValueError(f"names must be empty or have one entry per dimension ({ndim})")
        return checked

    def _current_bandwidth(self) -> Bandwidth:
        """
        The Bandwidth for h2Inv as it now stands, rebuilt if it has been changed in place.

        Returns
        -------
        Bandwidth
            The bandwidth.
        """
        if not np.array_equal(self._h2Inv, self._bandwidth_h2Inv):
            self._bandwidth = Bandwidth(self._h2Inv)
            self._bandwidth_h2Inv = self._h2Inv.copy()
        return self._bandwidth

    @property
    def dataset(self) -> np.ndarray:
        """
        numpy.ndarray of float, shape (N, nDim) : the data set, in tree
        order (a read-only array).
        """
        return self._dataset

    @property
    def w(self) -> np.ndarray:
        """
        numpy.ndarray of float, shape (N,) : the weights, in tree order,
        normalized to sum to one. Elements may be changed in place; call
        reweight() afterwards.
        """
        return self._w

    @property
    def h2Inv(self) -> np.ndarray:
        """
        numpy.ndarray of float, shape (nDim, nDim) : the inverse squared
        bandwidth matrix. Elements may be changed in place; the change
        takes effect on the next evaluation.
        """
        return self._h2Inv

    @property
    def names(self) -> list[str]:
        """
        list of str : names of the quantities in the data set (empty if
        not given).
        """
        return self._names

    @property
    def aux(self) -> np.ndarray:
        """
        numpy.ndarray of int64, shape (N,) : each point's label, in tree
        order (a read-only array): by default its index in the data set
        as originally given; for a KDE from conditional(), its parent's
        label.
        """
        return self._aux

    @property
    def nDim(self) -> int:
        """
        int : number of dimensions.
        """
        return self._kde.nDim()

    @property
    def marginal(self) -> float | None:
        """
        float or None : for a KDE from conditional(), the parent's
        marginal density over the fixed dimensions at the fixed values
        (so the parent's density at (x0, y) is marginal times this KDE's
        density at y); otherwise None.
        """
        return self._marginal

    @property
    def kernel_density(self) -> KernelDensityView:
        """
        KernelDensityView : the underlying compiled kernel density
        estimate.
        """
        return self._kde

    def reweight(self) -> None:
        """
        Bring the estimate up to date after the weights are changed in place.

        Renormalizes w in place to sum to one, and recomputes the kd
        tree's node weight sums. Costs O(N), in parallel. Like changing
        the weights in place, this must not happen while the estimate is
        being evaluated (pdf() or conditional(), e.g. from another
        thread): reweighting is not synchronized with evaluations.

        Raises
        ------
        ValueError
            If the weights are not finite and non-negative, or are all
            zero.
        """
        self._kde.reweight()

    def pdf(self, x: Any, relTol: float = 1e-2, absTol: float = 1e-8,
            bandwidth: Bandwidth | None = None, errDims: list[int] | None = None,
            errCov: Any = None) -> float | np.ndarray:
        """
        Evaluate the kernel density estimate at one or more points.

        Parameters
        ----------
        x : array_like of float, shape (nDim,) or (M, nDim)
            One point, or M of them. Coordinates may be NaN, to evaluate
            the marginal density over the other dimensions.
        relTol : float, optional
            Relative tolerance: stop once the uncertainty is less than
            relTol times the estimate (default 1e-2).
        absTol : float, optional
            Absolute tolerance: stop once the estimate plus its
            uncertainty is less than absTol (default 1e-8).
        bandwidth : Bandwidth, optional
            The kernel's bandwidth, in place of h2Inv (e.g. h2Inv
            convolved with measurement errors; see
            Bandwidth.convolveCov()); if None (the default), h2Inv.
        errDims : list of int, optional
            Dimensions with measurement errors (default: none); each must
            be specified (not NaN) in every point. Requires errCov.
        errCov : array_like of float, shape (nErr, nErr) or (M, nErr, nErr), optional
            Covariance matrices of the measurement errors in errDims, in
            that order (nErr = len(errDims)): one shared by every point,
            or one per point. Each point's kernel is convolved with its
            own errors (see Bandwidth.convolveCov()), so that the result
            is the density of measuring that point with those errors.

        Returns
        -------
        float or numpy.ndarray of float
            The estimate, normalized to unit integral (over the
            specified dimensions, for a point with NaNs); a float for
            one point.

        Raises
        ------
        ValueError
            As for KernelDensity.pdf().

        Notes
        -----
        Several points are evaluated in parallel, one per thread at a
        time, without the GIL, each convolving its own errors in the
        thread that evaluates it.
        """
        bw = self._current_bandwidth() if bandwidth is None else bandwidth
        return self._kde.pdf(x, relTol, absTol, bw, [] if errDims is None else errDims, errCov)

    def conditional(self, x0: Any, massTol: float = 1e-2, marginalDims: list[int] | None = None,
                    leafSize: int = 16, bandwidth: Bandwidth | None = None,
                    errDims: list[int] | None = None, errCov: Any = None) -> "KDE | list[KDE]":
        """
        Condition the estimate on values of some dimensions, optionally marginalizing over others.

        Parameters
        ----------
        x0 : array_like of float, shape (nDim,) or (M, nDim)
            One point, or M of them. A finite value fixes that
            dimension; NaN keeps it as a dimension of the result. Values
            for dimensions in marginalDims are ignored.
        massTol : float, optional
            Largest fraction of the conditional estimate's total weight
            that may be dropped, in [0, 1) (default 1e-2); the result is
            within L1 distance 2 massTol of the exact conditional
            density.
        marginalDims : list of int, optional
            Dimensions to marginalize over (default: none).
        leafSize : int, optional
            Leaf size of the result's kd tree (default 16).
        bandwidth : Bandwidth, optional
            The kernel's bandwidth, in place of h2Inv (e.g. h2Inv
            convolved with measurement errors; see
            Bandwidth.convolveCov()); if None (the default), h2Inv.
        errDims : list of int, optional
            Dimensions with measurement errors (default: none); each must
            be fixed (finite, and not in marginalDims) in every point.
            Requires errCov.
        errCov : array_like of float, shape (nErr, nErr) or (M, nErr, nErr), optional
            Covariance matrices of the measurement errors in errDims, as
            for pdf(). Each point's kernel is convolved with its own
            errors before conditioning, so that the result is the
            posterior given a measurement of that point with those
            errors.

        Returns
        -------
        KDE or list of KDE
            The conditional estimate p(y | x0) over the kept dimensions,
            with its names taken from this KDE's, its marginal set to
            this KDE's marginal density at x0, and each point's aux label
            carried over from this KDE; a list of one per point for M
            points.

        Raises
        ------
        ValueError
            As for KernelDensity.conditional().

        Notes
        -----
        Several points are conditioned on in parallel, one per thread at
        a time, without the GIL.
        """
        bw = self._current_bandwidth() if bandwidth is None else bandwidth
        result = self._kde.conditional(x0, massTol, [] if marginalDims is None else marginalDims,
                                       bw, leafSize, [] if errDims is None else errDims, errCov)
        if isinstance(result, list):
            return [self._wrap_conditional(cd) for cd in cast(list[Any], result)]
        return self._wrap_conditional(result)

    def conditional_sample(self, x0: Any, massTol: float = 1e-2, marginalDims: list[int] | None = None,
                           bandwidth: Bandwidth | None = None, errDims: list[int] | None = None,
                           errCov: Any = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """
        Find the sample points, weights, and aux labels of a conditional estimate, without building it.

        The same points, weights, and aux labels conditional() keeps,
        e.g. for storing a conditional estimate and rebuilding it later:
        a KDE
        over them, with dataset the points, w the weights, and h2Inv the
        conditional kernel metric -- the block over the kept dimensions
        of the bandwidth reduced over marginalDims (or of the bandwidth
        convolved with the errors, if errCov is given) -- is the
        conditional estimate.

        Parameters
        ----------
        x0 : array_like of float, shape (nDim,) or (M, nDim)
            As for conditional().
        massTol : float, optional
            As for conditional() (default 1e-2).
        marginalDims : list of int, optional
            As for conditional() (default: none).
        bandwidth : Bandwidth, optional
            As for conditional(); if None (the default), h2Inv.
        errDims : list of int, optional
            As for conditional() (default: none).
        errCov : array_like of float, shape (nErr, nErr) or (M, nErr, nErr), optional
            As for conditional().

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray), or list of such tuples
            The points kept, shape (K, nR), in the kept dimensions (those
            NaN in x0 and not in marginalDims, in increasing order); their
            weights, shape (K,); and their aux labels, shape (K,), each
            that of the point in this KDE it came from (see aux); all in
            increasing order of weight. A list of one such tuple per
            point for M points. The weights are not normalized (see
            KernelDensity.conditionalSample()).

        Raises
        ------
        ValueError
            As for conditional().

        Notes
        -----
        Several points are handled in parallel, one per thread at a
        time, without the GIL.
        """
        bw = self._current_bandwidth() if bandwidth is None else bandwidth
        result = self._kde.conditionalSample(x0, massTol, [] if marginalDims is None else marginalDims, bw,
                                             [] if errDims is None else errDims, errCov)
        if isinstance(result, list):
            return [(x, w, aux) for x, w, aux in cast(list[tuple[np.ndarray, np.ndarray, np.ndarray]], result)]
        x, w, aux = cast(tuple[np.ndarray, np.ndarray, np.ndarray], result)
        return x, w, aux

    def _wrap_conditional(self, cd: Any) -> "KDE":
        """
        Wrap a ConditionalDensity as a KDE.

        Parameters
        ----------
        cd : ConditionalDensity
            The conditional density, from this KDE's kernel density.

        Returns
        -------
        KDE
            A KDE over cd's points (see conditional()), keeping cd alive.
        """
        dims = cd.dims()
        names = [self._names[d] for d in dims] if self._names else []
        return KDE._from_kernel_density(cd.kde(), cd.bandwidth().h2Inv(), names, cd.marginal(), cd)
