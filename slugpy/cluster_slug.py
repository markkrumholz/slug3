"""
cluster_slug.py

Implements cluster_slug, Bayesian inference of star cluster properties
from photometry, using a library of slug cluster simulations.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import math
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np

from .slug_reader import slug_reader

# A probability specification (pobs or priors): None, one value per
# library cluster, or a callable
ProbSpec = np.ndarray | Callable[..., Any] | None

# Physical quantities that can be inferred, besides photometry
PHYSICAL_QUANTITIES = ("mass", "age", "A_V", "metallicity")


def _check_prob_spec(value: Any, name: str, size: int, at_most_one: bool) -> ProbSpec:
    """
    Check that a value is a valid pobs or priors specification.

    Parameters
    ----------
    value : Any
        The value to check.
    name : str
        Its name ("pobs" or "priors"), for error messages.
    size : int
        Number of clusters in the library.
    at_most_one : bool
        Whether array values must also be at most one (as for a
        probability), rather than just non-negative.

    Returns
    -------
    ProbSpec
        value, unchanged if it is None or a callable, or as a float64
        array if it is an array.

    Raises
    ------
    TypeError
        If value is not None, a numpy array, or a callable.
    ValueError
        If value is an array that is not 1D with size elements, or has
        values that are not finite, negative, or (if at_most_one)
        greater than one.
    """
    if value is None or callable(value):
        return value
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be None, a 1D numpy array, or a callable")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(f"{name} must be a 1D array with one element per library cluster ({size})")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} values must be finite and non-negative")
    if at_most_one and np.any(arr > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return arr


def _check_tolerance(value: Any, name: str) -> float:
    """
    Check that a value is a valid tolerance.

    Parameters
    ----------
    value : Any
        The value to check.
    name : str
        Its name ("abstol" or "reltol"), for error messages.

    Returns
    -------
    float
        value, as a float.

    Raises
    ------
    TypeError
        If value is not a real number.
    ValueError
        If value is negative or not finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    tol = float(value)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return tol


class cluster_slug(slug_reader):
    """
    Bayesian inference of star cluster properties from photometry,
    using a library of slug cluster simulations.

    A slug_reader on a cluster library -- slug output with one
    photometry entry per cluster (i.e., one output time per trial) --
    plus the settings that control inference with it. Construction
    checks that the library is suitable: it must have clusters and
    cluster_phot groups with equal numbers of entries, and a PDFs
    group, since the sampling distributions the library was drawn from
    are needed to interpret it.

    Parameters
    ----------
    filename : str
        The library to read: as for slug_reader.
    pobs : None, numpy.ndarray, or callable, optional
        Initial value of the pobs property (default None).
    priors : None, numpy.ndarray, or callable, optional
        Initial value of the priors property (default None).
    filtersets : list of list of str, optional
        Initial value of the filtersets property (default: none).
    inference_quantities : list of str, optional
        Initial value of the inference_quantities property (default
        ["mass", "age", "A_V", "metallicity"]).
    abstol : float, optional
        Initial value of the abstol property (default 1e-8).
    reltol : float, optional
        Initial value of the reltol property (default 1e-2).
    use_nebular : bool, optional
        Initial value of the use_nebular property; if None (the
        default), True if the library has nebular photometry, False
        otherwise.
    use_extinct : bool, optional
        Initial value of the use_extinct property; if None (the
        default), True if the library has extincted photometry, False
        otherwise.

    Raises
    ------
    ValueError
        If the library has no clusters or cluster_phot group, they have
        different numbers of entries, or it has no PDFs group; or any
        other argument is invalid (see the corresponding property).
    TypeError
        If an argument has the wrong type (see the corresponding
        property).

    Attributes
    ----------
    size : int
        Number of clusters in the library (read-only).
    pobs : None, numpy.ndarray, or callable
        Probability that each library cluster would be observed: None
        (every cluster is observed), an array of one value in [0, 1]
        per library cluster, or a callable.
    priors : None, numpy.ndarray, or callable
        Prior probability of each library cluster's physical
        properties: None (the prior equals the sampling density from
        which the library was drawn), an array of one non-negative
        value per library cluster, or a callable.
    filtersets : list of list of str
        Combinations of filters to do inference with; each must name
        filters the library has photometry for, which is checked when
        the data for a filter set are loaded. No filter may also be in
        inference_quantities.
    inference_quantities : list of str
        Quantities to infer: any of "mass", "age", "A_V", and
        "metallicity", and the names of any of the library's filters
        (e.g. to infer Q(HI) from photometry in other filters). No
        filter named here may also be in any of filtersets.
    abstol : float
        Absolute tolerance for evaluating PDFs (non-negative).
    reltol : float
        Relative tolerance for evaluating PDFs (non-negative).
    use_nebular : bool
        Whether to use photometry including nebular emission. May be
        True only if the library has nebular photometry; filters for
        which it is unavailable (NaN) fall back to photometry without
        nebular emission.
    use_extinct : bool
        Whether to use extincted photometry. May be True only if the
        library has extincted photometry; filters for which it is
        unavailable (NaN, e.g. because the extinction curve does not
        cover the filter) fall back to unextincted photometry.
    """

    def __init__(self, filename: str, pobs: ProbSpec = None, priors: ProbSpec = None,
                 filtersets: list[list[str]] | None = None,
                 inference_quantities: list[str] | None = None, abstol: float = 1e-8,
                 reltol: float = 1e-2, use_nebular: bool | None = None,
                 use_extinct: bool | None = None) -> None:
        super().__init__(filename)

        # The library must have one photometry entry per cluster, and
        # record the distributions it was sampled from
        if self.clusters is None or self.cluster_phot is None:
            raise ValueError(f"{filename} is not a cluster library: it must have clusters "
                             "and cluster_phot groups")
        n_clusters = len(self.clusters)
        n_phot = len(self.cluster_phot)
        if n_clusters != n_phot:
            raise ValueError(f"{filename} is not a cluster library: its clusters and "
                             f"cluster_phot groups have different numbers of entries "
                             f"({n_clusters} and {n_phot}), so it does not have one "
                             "photometry entry per cluster")
        with h5py.File(self._last_checkpoint_file, "r") as f:
            if "PDFs" not in f:
                raise ValueError(f"{filename} has no PDFs group, so the distributions the "
                                 "library was sampled from are unknown")
        self._size: int = n_clusters
        phot_datasets = self.cluster_phot.keys()
        self._has_nebular: bool = "phot_neb" in phot_datasets
        self._has_extinct: bool = "phot_extinct" in phot_datasets

        # Settings, each checked by its own setter
        self._pobs: ProbSpec = None
        self._priors: ProbSpec = None
        self._filtersets: list[list[str]] = []
        self._inference_quantities: list[str] = []
        self._abstol: float = 1e-8
        self._reltol: float = 1e-2
        self._use_nebular: bool = False
        self._use_extinct: bool = False
        self.pobs = pobs
        self.priors = priors
        self.filtersets = [] if filtersets is None else filtersets
        self.inference_quantities = (list(PHYSICAL_QUANTITIES) if inference_quantities is None
                                     else inference_quantities)
        self.abstol = abstol
        self.reltol = reltol
        self.use_nebular = self._has_nebular if use_nebular is None else use_nebular
        self.use_extinct = self._has_extinct if use_extinct is None else use_extinct

    @property
    def size(self) -> int:
        """
        int : number of clusters in the library (read-only).
        """
        return self._size

    @size.setter
    def size(self, value: Any) -> None:
        raise AttributeError("size is read-only")

    @property
    def pobs(self) -> ProbSpec:
        """
        None, numpy.ndarray, or callable : probability that each library
        cluster would be observed. None means every cluster is
        observed; an array gives one value in [0, 1] per library
        cluster; a callable computes it.

        Raises
        ------
        TypeError
            On setting, if the value is not None, a numpy array, or a
            callable.
        ValueError
            On setting, if an array is not 1D with size elements, or has
            values outside [0, 1].
        """
        return self._pobs

    @pobs.setter
    def pobs(self, value: Any) -> None:
        self._pobs = _check_prob_spec(value, "pobs", self._size, at_most_one=True)

    @property
    def priors(self) -> ProbSpec:
        """
        None, numpy.ndarray, or callable : prior probability of each
        library cluster's physical properties. None means the prior
        equals the sampling density from which the library was drawn;
        an array gives one non-negative value per library cluster; a
        callable computes it.

        Raises
        ------
        TypeError
            On setting, if the value is not None, a numpy array, or a
            callable.
        ValueError
            On setting, if an array is not 1D with size elements, or has
            negative or non-finite values.
        """
        return self._priors

    @priors.setter
    def priors(self, value: Any) -> None:
        self._priors = _check_prob_spec(value, "priors", self._size, at_most_one=False)

    @property
    def filtersets(self) -> list[list[str]]:
        """
        list of list of str : combinations of filters to do inference
        with (a copy; assign to change it). Each filter must be one the
        library has photometry for, which is checked when the data for
        a filter set are loaded.

        Raises
        ------
        TypeError
            On setting, if the value is not a list of lists of str.
        ValueError
            On setting, if any filter set is empty, or names a filter
            that is also in inference_quantities.
        """
        return [list(fs) for fs in self._filtersets]

    @filtersets.setter
    def filtersets(self, value: Any) -> None:
        if not isinstance(value, list) or not all(
                isinstance(fs, list) and all(isinstance(name, str) for name in fs)
                for fs in value):
            raise TypeError("filtersets must be a list of lists of filter names")
        sets = [list(fs) for fs in value]
        if any(len(fs) == 0 for fs in sets):
            raise ValueError("filter sets must not be empty")
        self._check_disjoint(sets, self._inference_quantities)
        self._filtersets = sets

    @property
    def inference_quantities(self) -> list[str]:
        """
        list of str : quantities to infer (a copy; assign to change it):
        any of "mass", "age", "A_V", and "metallicity", and the names
        of any of the library's filters, so that, e.g., Q(HI) can be
        inferred from photometry in other filters. No filter named
        here may also be in any of filtersets.

        Raises
        ------
        TypeError
            On setting, if the value is not a list of str.
        ValueError
            On setting, if the value is empty, has repeated entries, has
            an entry that is neither a physical quantity nor one of the
            library's filters, or names a filter that is also in one of
            filtersets.
        """
        return list(self._inference_quantities)

    @inference_quantities.setter
    def inference_quantities(self, value: Any) -> None:
        if not isinstance(value, list) or not all(isinstance(q, str) for q in value):
            raise TypeError("inference_quantities must be a list of str")
        quantities = list(value)
        if len(quantities) == 0:
            raise ValueError("inference_quantities must not be empty")
        if len(set(quantities)) != len(quantities):
            raise ValueError("inference_quantities must not repeat a quantity")
        filters = self.filters or []
        unknown = [q for q in quantities if q not in PHYSICAL_QUANTITIES and q not in filters]
        if unknown:
            raise ValueError(f"unknown inference quantities {unknown}: each must be one of "
                             f"{list(PHYSICAL_QUANTITIES)} or one of the library's filters")
        self._check_disjoint(self._filtersets, quantities)
        self._inference_quantities = quantities

    @staticmethod
    def _check_disjoint(filtersets: list[list[str]], quantities: list[str]) -> None:
        """
        Check that no filter is both inferred and used to infer.

        Parameters
        ----------
        filtersets : list of list of str
            The filter sets.
        quantities : list of str
            The inference quantities.

        Raises
        ------
        ValueError
            If a quantity appears in any filter set.
        """
        both = sorted({name for fs in filtersets for name in fs if name in quantities})
        if both:
            raise ValueError(f"filters {both} are both inference quantities and in filter "
                             "sets; a filter cannot be inferred from itself")

    @property
    def abstol(self) -> float:
        """
        float : absolute tolerance for evaluating PDFs (non-negative).

        Raises
        ------
        TypeError
            On setting, if the value is not a real number.
        ValueError
            On setting, if the value is negative or not finite.
        """
        return self._abstol

    @abstol.setter
    def abstol(self, value: Any) -> None:
        self._abstol = _check_tolerance(value, "abstol")

    @property
    def reltol(self) -> float:
        """
        float : relative tolerance for evaluating PDFs (non-negative).

        Raises
        ------
        TypeError
            On setting, if the value is not a real number.
        ValueError
            On setting, if the value is negative or not finite.
        """
        return self._reltol

    @reltol.setter
    def reltol(self, value: Any) -> None:
        self._reltol = _check_tolerance(value, "reltol")

    @property
    def use_nebular(self) -> bool:
        """
        bool : whether to use photometry including nebular emission.
        Filters for which it is unavailable (NaN) fall back to
        photometry without nebular emission.

        Raises
        ------
        TypeError
            On setting, if the value is not a bool.
        ValueError
            On setting True, if the library has no nebular photometry.
        """
        return self._use_nebular

    @use_nebular.setter
    def use_nebular(self, value: Any) -> None:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("use_nebular must be a bool")
        if value and not self._has_nebular:
            raise ValueError("use_nebular cannot be True: the library has no nebular photometry")
        self._use_nebular = bool(value)

    @property
    def use_extinct(self) -> bool:
        """
        bool : whether to use extincted photometry. Filters for which it
        is unavailable (NaN, e.g. because the extinction curve does not
        cover the filter) fall back to unextincted photometry.

        Raises
        ------
        TypeError
            On setting, if the value is not a bool.
        ValueError
            On setting True, if the library has no extincted photometry.
        """
        return self._use_extinct

    @use_extinct.setter
    def use_extinct(self, value: Any) -> None:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("use_extinct must be a bool")
        if value and not self._has_extinct:
            raise ValueError("use_extinct cannot be True: the library has no extincted photometry")
        self._use_extinct = bool(value)
