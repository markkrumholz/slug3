"""
cluster_slug.py

Implements cluster_slug, Bayesian inference of star cluster properties
from photometry, using a library of slug cluster simulations.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import math
from collections.abc import Callable, KeysView, Mapping
from types import MappingProxyType
from typing import Any, cast

import h5py
import numexpr  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from astropy import units as u

from ._slug import Bandwidth
from .kde import KDE
from .slug_reader import slug_reader

# A probability specification (pobs or priors): None, one value per
# library cluster, or a callable
ProbSpec = np.ndarray | Callable[..., Any] | None

# Physical quantities that can be inferred, besides photometry
PHYSICAL_QUANTITIES = ("mass", "age", "A_V", "metallicity")

# The PDF (a key of slug_reader.pdfs) from which a library draws each
# physical quantity
QUANTITY_PDFS = {"mass": "cmf", "age": "outTimeDist", "A_V": "avDist", "metallicity": "fehDist"}

# Photometric variant suffixes (see slug_phot_reader) to try in turn, by
# (use_nebular, use_extinct): the requested variant first, then those
# without nebular emission and/or extinction
_PHOT_CHAINS = {
    (True, True): ("_neb_ex", "_neb", "_ex", ""),
    (True, False): ("_neb", ""),
    (False, True): ("_ex", ""),
    (False, False): ("",),
}


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
        value, unchanged if it is None or a callable, or, if it is an
        array, a read-only float64 copy of it (so that changing the
        caller's array later cannot desynchronize it from the weights
        computed from it).

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
    arr = np.array(value, dtype=np.float64)
    arr.setflags(write=False)
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


def _check_bandwidth(value: Any, name: str) -> float:
    """
    Check that a value is a valid bandwidth.

    Parameters
    ----------
    value : Any
        The value to check.
    name : str
        Its name ("bw_phot" or "bw_phys"), for error messages.

    Returns
    -------
    float
        value, as a float.

    Raises
    ------
    TypeError
        If value is not a real number.
    ValueError
        If value is not finite and positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    bw = float(value)
    if not math.isfinite(bw) or bw <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return bw


def _is_mag_unit(unit: u.UnitBase) -> bool:
    """
    Whether a photometric unit is a magnitude.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit.

    Returns
    -------
    bool
        True for mag and the magnitude units of photometric systems
        (e.g. mag(AB), mag(ST)).
    """
    return unit == u.mag or isinstance(unit, u.MagUnit)


def _error_unit_matches(unit: u.UnitBase, band: u.UnitBase) -> bool:
    """
    Whether a photometric error's unit suits a band.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The error's unit.
    band : astropy.units.UnitBase
        The band's photometric unit.

    Returns
    -------
    bool
        For a band in magnitudes, whether unit is mag or the band's own
        unit; otherwise, whether unit is dex or dex of a unit equivalent
        to the band's.
    """
    if _is_mag_unit(band):
        return unit == u.mag or unit == band
    if unit == u.dex:
        return True
    return isinstance(unit, u.DexUnit) and cast(u.DexUnit, unit).physical_unit.is_equivalent(band)


def _error_unit_rule(band: u.UnitBase) -> str:
    """
    The rule for a photometric error's unit, for error messages.

    Parameters
    ----------
    band : astropy.units.UnitBase
        The band's photometric unit.

    Returns
    -------
    str
        A description of the units an error in that band may have.
    """
    if _is_mag_unit(band):
        return f"errors in magnitudes must be in mag or {band}"
    return f"errors in a non-magnitude unit are in dex (of the log10 photometry), so must be in dex or dex({band})"


def _filterset_list(value: Any) -> list[list[str]]:
    """
    Interpret the constructor's filtersets argument.

    Parameters
    ----------
    value : Any
        None, a list of filter names (one filter set), or a list of such
        lists.

    Returns
    -------
    list of list of str
        The filter sets (none, if value is None or empty).

    Raises
    ------
    TypeError
        If value has any other form.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("filtersets must be a list of filter names, or a list of such lists")
    entries = cast(list[Any], value)
    if all(isinstance(name, str) for name in entries):
        return [cast(list[str], entries)] if entries else []
    if all(isinstance(fs, list) and all(isinstance(name, str) for name in cast(list[Any], fs))
           for fs in entries):
        return [list(cast(list[str], fs)) for fs in entries]
    raise TypeError("filtersets must be a list of filter names, or a list of such lists")


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
    filtersets : list of str or list of list of str, optional
        Filter sets to build kernel density estimates for, each passed
        to add_filterset() (with the default bandwidth) once the other
        settings are made: a list of filter names for one filter set,
        or a list of such lists (default: none).
    inference_quantities : list of str, optional
        Initial value of the inference_quantities property. If None (the
        default), those of "mass", "age", "A_V", and "metallicity" that
        are not singular in the library (see inference_quantities).
    bw_phot : float, optional
        Initial value of the bw_phot property (default 0.1).
    bw_phys : float, optional
        Initial value of the bw_phys property (default 0.1).
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
    sampling_density : numpy.ndarray of float
        The density from which the library drew each cluster's physical
        properties (read-only array; see add_filterset()).
    pobs : None, numpy.ndarray, or callable
        Probability that each library cluster would be observed: None
        (every cluster is observed), an array of one value in [0, 1]
        per library cluster, or a callable. Setting it reweights every
        kernel density estimate.
    priors : None, numpy.ndarray, or callable
        Prior probability of each library cluster's physical
        properties: None (the prior equals the sampling density from
        which the library was drawn), an array of one non-negative
        value per library cluster, or a callable. Setting it reweights
        every kernel density estimate.
    kdes : Mapping of tuple of str to KDE
        The kernel density estimate for each filter set, keyed by the
        filter set's names, as a tuple (read-only view; see
        add_filterset() and remove_filterset()).
    filtersets : KeysView of tuple of str
        The filter sets with kernel density estimates: kdes.keys()
        (read-only).
    inference_quantities : list of str
        Quantities to infer: any of "mass", "age", "A_V", and
        "metallicity" that are not singular in the library (drawn from a
        PDF that is absent or a delta distribution, so constant), and
        the names of any of the library's filters (e.g. to infer Q(HI)
        from photometry in other filters). No filter named here may also
        be in any filter set.
    bw_phot : float
        Photometric bandwidth of the default kernel, in mag (positive);
        see add_filterset(). Setting it updates every kernel density
        estimate built with the default bandwidth.
    bw_phys : float
        Bandwidth of the default kernel in the physical quantities, in
        dex for mass and age, mag for A_V, and dex for [Fe/H]
        (positive); see add_filterset(). Setting it updates every kernel
        density estimate built with the default bandwidth.
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

    Notes
    -----
    Changing pobs or priors reweights every kernel density estimate in
    place (see the pobs property), and changing bw_phot or bw_phys
    updates the bandwidth of every estimate built with the default
    bandwidth. The other settings are read when a filter set is added:
    changing inference_quantities, use_nebular, or use_extinct
    afterwards does not change estimates already built; remove and
    re-add a filter set to rebuild its estimate.
    """

    def __init__(self, filename: str, pobs: ProbSpec = None, priors: ProbSpec = None,
                 filtersets: list[str] | list[list[str]] | None = None,
                 inference_quantities: list[str] | None = None, bw_phot: float = 0.1,
                 bw_phys: float = 0.1, abstol: float = 1e-8, reltol: float = 1e-2,
                 use_nebular: bool | None = None, use_extinct: bool | None = None) -> None:
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

        # Physical quantities drawn from an absent or delta PDF are
        # constant across the library, so cannot be inferred: their
        # dimension of a kernel density estimate would have zero spread
        self._singular: frozenset[str] = frozenset(
            q for q, name in QUANTITY_PDFS.items()
            if name not in self.pdfs or self.pdfs[name].getMin() == self.pdfs[name].getMax())

        # Cached library quantities, which never change: every cluster's
        # physical properties, and the density the library drew them from
        mass, age, av, feh = self._read_physical()
        self._mass: u.Quantity = mass
        self._age: u.Quantity = age
        self._av: u.Quantity | None = av
        self._feh: np.ndarray = feh
        self._sampling_density: np.ndarray = self._compute_sampling_density()

        # Settings, each checked by its own setter
        self._kdes: dict[tuple[str, ...], KDE] = {}
        # The filter sets whose estimates use the default bandwidth, which
        # follows bw_phot and bw_phys
        self._default_bw: set[tuple[str, ...]] = set()
        self._bw_phot: float = 0.1
        self._bw_phys: float = 0.1
        self._pobs: ProbSpec = None
        self._priors: ProbSpec = None
        # pobs and priors as arrays (a callable's result, cached so that
        # changing one need not recompute the other), and the weights
        # they give each library cluster; None where there are none
        self._pobs_values: np.ndarray | None = None
        self._priors_values: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._inference_quantities: list[str] = []
        self._abstol: float = 1e-8
        self._reltol: float = 1e-2
        self._use_nebular: bool = False
        self._use_extinct: bool = False
        self.pobs = pobs
        self.priors = priors
        self.inference_quantities = (
            [q for q in PHYSICAL_QUANTITIES if q not in self._singular]
            if inference_quantities is None else inference_quantities)
        self.bw_phot = bw_phot
        self.bw_phys = bw_phys
        self.abstol = abstol
        self.reltol = reltol
        self.use_nebular = self._has_nebular if use_nebular is None else use_nebular
        self.use_extinct = self._has_extinct if use_extinct is None else use_extinct

        # Filter sets last, since building their estimates uses every
        # other setting
        for filters in _filterset_list(filtersets):
            self.add_filterset(filters)

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
    def sampling_density(self) -> np.ndarray:
        """
        numpy.ndarray of float : the density from which the library drew
        each cluster's physical properties (a read-only array): the
        product, over the physical quantities that are not singular, of
        the library's PDF for that quantity evaluated at the cluster's
        value of it (mass in Msun, age in yr, A_V in mag, [Fe/H]).
        Computed on construction.
        """
        return self._sampling_density

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
            On setting, if an array (or a callable's result, which must
            be one) is not 1D with size elements, or has values outside
            [0, 1]; or if the new weights would leave a kernel density
            estimate with no weight at all.

        Notes
        -----
        A callable is called once, on setting, as for add_filterset(),
        and its result cached. Setting pobs reweights every kernel
        density estimate in place: each cluster's weight becomes
        pobs * priors / sampling_density (see add_filterset()). If that
        fails, nothing is changed.
        """
        return self._pobs

    @pobs.setter
    def pobs(self, value: Any) -> None:
        spec = _check_prob_spec(value, "pobs", self._size, at_most_one=True)
        values = self._evaluate_prob_spec(spec, "pobs", at_most_one=True)
        self._update_weights(values, self._priors_values)
        self._pobs, self._pobs_values = spec, values

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
            On setting, if an array (or a callable's result, which must
            be one) is not 1D with size elements, or has negative or
            non-finite values; or if the new weights would leave a
            kernel density estimate with no weight at all.

        Notes
        -----
        As for pobs: a callable is called once, on setting, and its
        result cached, and setting priors reweights every kernel density
        estimate in place.
        """
        return self._priors

    @priors.setter
    def priors(self, value: Any) -> None:
        spec = _check_prob_spec(value, "priors", self._size, at_most_one=False)
        values = self._evaluate_prob_spec(spec, "priors", at_most_one=False)
        self._update_weights(self._pobs_values, values)
        self._priors, self._priors_values = spec, values

    @property
    def kdes(self) -> Mapping[tuple[str, ...], KDE]:
        """
        Mapping of tuple of str to KDE : the kernel density estimate for
        each filter set, keyed by the filter set's names as a tuple
        (read-only view: use add_filterset() and remove_filterset()).
        """
        return MappingProxyType(self._kdes)

    @property
    def filtersets(self) -> KeysView[tuple[str, ...]]:
        """
        KeysView of tuple of str : the filter sets with kernel density
        estimates, i.e. kdes.keys() (read-only: use add_filterset() and
        remove_filterset()).
        """
        return self._kdes.keys()

    @property
    def inference_quantities(self) -> list[str]:
        """
        list of str : quantities to infer (a copy; assign to change it):
        any of "mass", "age", "A_V", and "metallicity", and the names
        of any of the library's filters, so that, e.g., Q(HI) can be
        inferred from photometry in other filters. A physical quantity
        must not be singular in the library: drawn from a PDF that is
        absent (e.g. A_V without extinction) or a delta distribution
        (e.g. a single [Fe/H]), since it is then the same for every
        cluster. No filter named here may also be in any filter set.

        Raises
        ------
        TypeError
            On setting, if the value is not a list of str.
        ValueError
            On setting, if the value is empty, has repeated entries, has
            an entry that is neither a physical quantity nor one of the
            library's filters, names a singular physical quantity, or
            names a filter that is also in a filter set.
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
        singular = [q for q in quantities if q in self._singular]
        if singular:
            raise ValueError(f"cannot infer {singular}: the library draws them from a PDF that is "
                             "absent or a delta distribution, so they are the same for every cluster")
        self._check_disjoint([list(fs) for fs in self._kdes], quantities)
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
    def bw_phot(self) -> float:
        """
        float : photometric bandwidth of the default kernel, in mag
        (positive; see add_filterset()).

        Raises
        ------
        TypeError
            On setting, if the value is not a real number.
        ValueError
            On setting, if the value is not finite and positive.

        Notes
        -----
        Setting bw_phot updates the bandwidth of every kernel density
        estimate built with the default bandwidth (no rebuild is
        needed); estimates built with an explicit h2Inv are unchanged.
        """
        return self._bw_phot

    @bw_phot.setter
    def bw_phot(self, value: Any) -> None:
        self._bw_phot = _check_bandwidth(value, "bw_phot")
        self._update_default_bandwidths()

    @property
    def bw_phys(self) -> float:
        """
        float : bandwidth of the default kernel in the physical
        quantities (positive; see add_filterset()): in dex for mass and
        age, which the estimates store as log10, in mag for A_V, and in
        dex for [Fe/H].

        Raises
        ------
        TypeError
            On setting, if the value is not a real number.
        ValueError
            On setting, if the value is not finite and positive.

        Notes
        -----
        As for bw_phot, setting bw_phys updates the bandwidth of every
        kernel density estimate built with the default bandwidth.
        """
        return self._bw_phys

    @bw_phys.setter
    def bw_phys(self, value: Any) -> None:
        self._bw_phys = _check_bandwidth(value, "bw_phys")
        self._update_default_bandwidths()

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

    def add_filterset(self, filters: list[str], h2Inv: float | np.ndarray | None = None) -> None:
        """
        Build the kernel density estimate for a filter set.

        Builds a KDE over the library in the space of the filters'
        photometry and the inference quantities, and stores it in kdes
        under tuple(filters), replacing any estimate already there.

        Parameters
        ----------
        filters : list of str
            Names of the filters; each must be one of the library's
            filters, and none may be in inference_quantities.
        h2Inv : float or numpy.ndarray, optional
            The KDE's inverse squared bandwidth matrix, in any form KDE
            accepts. If None (the default), the default bandwidth, set
            by bw_phot and bw_phys (see Notes), which then follows any
            later change to them.

        Raises
        ------
        TypeError
            If filters is not a list of str.
        ValueError
            If filters is empty, repeats a filter, names a filter the
            library does not have, or names an inference quantity; if no
            library cluster has valid values for every quantity, or
            those that do all have zero weight; or if the KDE cannot be
            built (see KDE).

        Notes
        -----
        The estimate's dimensions are the filters, in order, then the
        inference quantities, in order (see kdes[...].names); its tree is
        never split along the inference quantities. Each dimension holds:

        - for a filter (as a filter or an inference quantity): the
          photometry chosen by use_nebular and use_extinct, falling back,
          cluster by cluster, to photometry without nebular emission
          and/or extinction where it is NaN, zero (for a flux or photon
          rate, e.g. an extincted ionizing photon rate when the
          extinction curve does not extend past 13.6 eV), or absent for
          that filter. Magnitudes are used as they are; any other unit
          (e.g. photon/s, Lsun) is converted to its log10.
        - for "mass" and "age": log10 of the target mass (Msun) and of
          the cluster's age (yr);
        - for "A_V" and "metallicity": A_V (mag) and [Fe/H], unlogged.

        The default bandwidth is diagonal: the kernel is exp(-d^T h2Inv
        d), with h2Inv's diagonal entry 1 / bw_phys^2 for a physical
        quantity, and, for a filter, 1 / bw_phot^2 if it is in
        magnitudes or 1 / (0.4 bw_phot)^2 if it is stored as log10, so
        that bw_phot is the same width in either unit.

        If pobs or priors is set, each cluster is weighted by pobs, and
        by priors divided by the density from which the library drew
        the cluster's physical properties (the product of the library's
        PDFs for those physical quantities that are not singular). A
        callable pobs or priors is called as f(mass, age, A_V, feh),
        with mass (Msun), age (yr), and A_V (mag) astropy Quantities (A_V
        None if the library has no extinction) and feh a numpy array.

        Clusters with a non-finite value in any dimension (e.g. the log
        of a photon rate that is zero in every variant), or with zero or
        non-finite sampling_density, are left out of the estimate.
        Clusters whose weight is zero are kept, contributing nothing, so
        that a later change to pobs or priors can give them weight. Each
        point's aux label is its cluster's row in the library.
        """
        if not isinstance(filters, list) or not all(isinstance(f, str) for f in cast(list[Any], filters)):
            raise TypeError("filters must be a list of filter names")
        names = list(filters)
        if not names:
            raise ValueError("a filter set must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("a filter set must not repeat a filter")
        available = self.filters or []
        unknown = [f for f in names if f not in available]
        if unknown:
            raise ValueError(f"the library has no photometry for filters {unknown}")
        self._check_disjoint([names], self._inference_quantities)

        # The data set: the filters, then the inference quantities
        columns = names + self._inference_quantities
        dataset = np.empty((self._size, len(columns)))
        for i, name in enumerate(columns):
            dataset[:, i] = (self._physical_column(name) if name in PHYSICAL_QUANTITIES
                             else self._phot_column(name))

        # Leave out clusters that can never be used: those with an
        # invalid value, or drawn with zero or non-finite density (so
        # their prior weight would be infinite). Those whose weight is
        # zero now are kept, since reweighting may give them weight.
        keep = np.all(np.isfinite(dataset), axis=1)
        keep &= np.isfinite(self._sampling_density) & (self._sampling_density > 0.0)
        if not np.any(keep):
            raise ValueError(f"no library cluster has valid values for every quantity in {columns}")
        rows = np.nonzero(keep)[0].astype(np.int64)
        kept = np.ascontiguousarray(dataset[keep])
        kept_weights = None
        if self._weights is not None:
            kept_weights = np.ascontiguousarray(self._weights[keep])
            if not np.any(kept_weights > 0.0):
                raise ValueError(f"the library clusters valid for {columns} all have zero weight")

        key = tuple(names)
        self._kdes[key] = KDE(
            kept, h2Inv=self._default_h2inv(columns) if h2Inv is None else h2Inv,
            w=kept_weights, names=columns, aux=rows,
            noSplitDims=list(range(len(names), len(columns))))
        if h2Inv is None:
            self._default_bw.add(key)
        else:
            self._default_bw.discard(key)

    def _default_h2inv(self, columns: list[str]) -> np.ndarray:
        """
        The default inverse squared bandwidth matrix for an estimate.

        Parameters
        ----------
        columns : list of str
            The estimate's dimensions: filters and physical quantities.

        Returns
        -------
        numpy.ndarray of float, shape (len(columns), len(columns))
            The diagonal matrix described in add_filterset().
        """
        filters = [c for c in columns if c not in PHYSICAL_QUANTITIES]
        units = dict(zip(filters, self._band_units(filters), strict=True))
        widths = [self._bw_phys if c in PHYSICAL_QUANTITIES
                  else self._bw_phot if _is_mag_unit(units[c])
                  else 0.4 * self._bw_phot
                  for c in columns]
        return np.diag(1.0 / np.square(widths))

    def _update_default_bandwidths(self) -> None:
        """
        Set every default-bandwidth estimate's h2Inv from bw_phot and bw_phys.
        """
        for key in self._default_bw:
            kde = self._kdes[key]
            kde.h2Inv[...] = self._default_h2inv(kde.names)

    def conditional(self, x0: Any, massTol: float = 1e-2, marginalDims: list[int] | None = None,
                    leafSize: int = 16, filterset: list[str] | None = None,
                    errors: Any = None, cov: Any = None) -> KDE | list[KDE]:
        """
        Condition a filter set's kernel density estimate on observed photometry.

        Parameters
        ----------
        x0 : array_like of float, shape (nband,), (nDim,), (M, nband), or (M, nDim)
            One point or M of them, each either the photometry in the
            filter set's nband bands, in which case the inference
            quantities are kept as the dimensions of the result (the
            usual case), or a full point in the estimate's space (see
            kdes[...].names), its filters then the inference quantities,
            in which, as for KDE.conditional(), a finite value fixes that
            dimension and NaN keeps it as a dimension of the result.
            Photometry is in the units the estimate stores: magnitudes as
            they are, and log10 of any other unit (see add_filterset()).
        massTol : float, optional
            As for KDE.conditional() (default 1e-2).
        marginalDims : list of int, optional
            As for KDE.conditional() (default: none).
        leafSize : int, optional
            As for KDE.conditional() (default 16).
        filterset : list of str, optional
            The filter set whose estimate to use; added with
            add_filterset() first if it is not already in kdes. May be
            None (the default) only if kdes holds a single filter set,
            which is then used.
        errors : float, array_like, or astropy.units.Quantity, optional
            The photometric errors: one value for every band, one per
            band of the filter set (shape (nband,)), or, for M points,
            one per band for each point (shape (M, nband)); the first two
            apply to every point. For a band in magnitudes, a plain
            number is in mag, and a Quantity must be in mag or the band's
            own magnitude unit (e.g. mag(AB)); for a band in any other
            unit (e.g. erg/(s Hz)), which the estimate stores as log10, a
            plain number is in dex, and a Quantity must be in dex or dex
            of a unit equivalent to the band's. Errors are taken to be
            independent; see cov for correlated ones.
        cov : array_like or astropy.units.Quantity, optional
            The covariance matrix of the photometric errors, nband x
            nband for every point, or M x nband x nband for each of M
            points, in the same units as errors, squared: a Quantity must
            be in mag^2 if every band is in magnitudes, or dex^2 if none
            is (a filter set with both needs a plain array, since no
            single astropy unit describes it). May not be given with
            errors.

        Returns
        -------
        KDE or list of KDE
            The conditional estimate over the kept dimensions (see
            KDE.conditional()), with its marginal property the filter
            set's estimate's marginal density at the fixed values; a list
            of one per point for M points. If errors or cov is given, the
            estimate's kernel is first convolved with each point's errors
            (see Bandwidth.convolveCov()), so that the result is the
            posterior given photometry with those errors.

        Raises
        ------
        TypeError
            If filterset, errors, or cov has the wrong type.
        ValueError
            If filterset is None and kdes does not hold exactly one filter
            set; if x0 is not 1D or 2D with nband or nDim elements per
            point; if both errors and cov are given; if errors or cov has
            the wrong shape (including per-point errors for a different
            number of points), invalid values, or units that do not match
            the bands' photometric systems; or as for add_filterset() and
            KDE.conditional().

        Notes
        -----
        Several points are conditioned on in parallel, one per thread at
        a time (see KDE.conditional()), each with its own errors.
        """
        filters = self._select_filterset(filterset)
        kde = self._kdes[tuple(filters)]
        points = self._full_points(x0, "x0", filters, kde)
        bandwidth, per_point = self._phot_errors(filters, kde, errors, cov,
                                                 1 if points.ndim == 1 else points.shape[0])
        if per_point is not None:
            return kde.conditional(points, massTol, marginalDims, leafSize,
                                   errDims=list(range(len(filters))),
                                   errCov=per_point[0] if points.ndim == 1 else per_point)
        return kde.conditional(points, massTol, marginalDims, leafSize, bandwidth=bandwidth)

    def conditional_sample(self, x0: Any, massTol: float = 1e-2, marginalDims: list[int] | None = None,
                           filterset: list[str] | None = None, errors: Any = None,
                           cov: Any = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """
        Find the sample points, weights, and library rows of the posterior given photometry, without building it.

        The same points, weights, and aux labels conditional() keeps,
        e.g. for storing posteriors and rebuilding them later: a KDE over
        them (dataset the points, w the weights, aux the labels), with
        h2Inv the conditional
        kernel metric, is the posterior that conditional() returns. With
        the default bandwidth (see add_filterset()), which is diagonal,
        and no photometric errors, that metric is just the bandwidth's
        block over the kept dimensions; with errors, it is that block of
        the bandwidth convolved with them (see Bandwidth.convolveCov()),
        reduced over marginalDims if any.

        Parameters
        ----------
        x0 : array_like of float, shape (nband,), (nDim,), (M, nband), or (M, nDim)
            As for conditional().
        massTol : float, optional
            As for conditional() (default 1e-2).
        marginalDims : list of int, optional
            As for conditional() (default: none).
        filterset : list of str, optional
            As for conditional().
        errors : float, array_like, or astropy.units.Quantity, optional
            As for conditional().
        cov : array_like or astropy.units.Quantity, optional
            As for conditional().

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray), or list of such tuples
            The points kept, shape (K, nR), in the kept dimensions (see
            conditional(); for photometry alone, the inference
            quantities); their weights, shape (K,); and their aux labels,
            shape (K,): the library row of the cluster each point came
            from. A list of one such tuple per point for M points (see
            KDE.conditional_sample()).

        Raises
        ------
        TypeError
            As for conditional().
        ValueError
            As for conditional().

        Notes
        -----
        Several points are handled in parallel, one per thread at a
        time, each with its own errors.
        """
        filters = self._select_filterset(filterset)
        kde = self._kdes[tuple(filters)]
        points = self._full_points(x0, "x0", filters, kde)
        bandwidth, per_point = self._phot_errors(filters, kde, errors, cov,
                                                 1 if points.ndim == 1 else points.shape[0])
        if per_point is not None:
            return kde.conditional_sample(points, massTol, marginalDims,
                                          errDims=list(range(len(filters))),
                                          errCov=per_point[0] if points.ndim == 1 else per_point)
        return kde.conditional_sample(points, massTol, marginalDims, bandwidth=bandwidth)

    def pdf(self, x: Any, relTol: float | None = None, absTol: float | None = None,
            filterset: list[str] | None = None, errors: Any = None,
            cov: Any = None) -> float | np.ndarray:
        """
        Evaluate a filter set's kernel density estimate at one or more points.

        Parameters
        ----------
        x : array_like of float, shape (nband,), (nDim,), (M, nband), or (M, nDim)
            One point or M of them, each either the photometry in the
            filter set's nband bands, in which case the inference
            quantities are marginalized over (giving the density of the
            photometry alone), or a full point in the estimate's space
            (see kdes[...].names), its filters then the inference
            quantities, in which, as for KDE.pdf(), NaN marginalizes
            over that dimension. Photometry is in the units the estimate
            stores, as for conditional().
        relTol : float, optional
            As for KDE.pdf(); if None (the default), reltol.
        absTol : float, optional
            As for KDE.pdf(); if None (the default), abstol.
        filterset : list of str, optional
            As for conditional().
        errors : float, array_like, or astropy.units.Quantity, optional
            As for conditional(): the photometric errors, the same for
            every point or (shape (M, nband)) one set per point.
        cov : array_like or astropy.units.Quantity, optional
            As for conditional(): one matrix for every point, or one per
            point.

        Returns
        -------
        float or numpy.ndarray of float
            The estimate at each point (see KDE.pdf()); a float for one
            point. If errors or cov is given, the estimate's kernel is
            first convolved with each point's errors (see
            Bandwidth.convolveCov()), so that the result is the density
            of photometry measured with those errors.

        Raises
        ------
        TypeError
            As for conditional().
        ValueError
            If filterset is None and kdes does not hold exactly one filter
            set; if x is not 1D or 2D with nband or nDim elements per
            point; or as for conditional()'s errors and cov, and for
            add_filterset().

        Notes
        -----
        Several points are evaluated in parallel, one per thread at a
        time (see KDE.pdf()), each with its own errors.
        """
        filters = self._select_filterset(filterset)
        kde = self._kdes[tuple(filters)]
        points = self._full_points(x, "x", filters, kde)
        bandwidth, per_point = self._phot_errors(filters, kde, errors, cov,
                                                 1 if points.ndim == 1 else points.shape[0])
        rel = self._reltol if relTol is None else relTol
        abs_ = self._abstol if absTol is None else absTol
        if per_point is not None:
            return kde.pdf(points, rel, abs_, errDims=list(range(len(filters))),
                           errCov=per_point[0] if points.ndim == 1 else per_point)
        return kde.pdf(points, rel, abs_, bandwidth=bandwidth)

    @staticmethod
    def _full_points(x: Any, name: str, filters: list[str], kde: KDE) -> np.ndarray:
        """
        Points in a filter set's estimate's space, from photometry alone or full points.

        Parameters
        ----------
        x : array_like of float
            One point or M of them, each with one element per band or
            per dimension of the estimate.
        name : str
            The argument's name, for error messages.
        filters : list of str
            The filter set.
        kde : KDE
            Its kernel density estimate.

        Returns
        -------
        numpy.ndarray of float, shape (nDim,) or (M, nDim)
            The points, with NaN for every inference quantity of a point
            given as photometry alone.

        Raises
        ------
        ValueError
            If x is not 1D or 2D, with nband or nDim elements per point.
        """
        points = np.asarray(x, dtype=np.float64)
        nband = len(filters)
        if points.ndim not in (1, 2) or points.shape[-1] not in (nband, kde.nDim):
            raise ValueError(f"{name} must have one element per band ({nband}) or per dimension of the "
                             f"estimate ({kde.nDim}) for each point")
        if points.shape[-1] == nband:
            pad = np.full(points.shape[:-1] + (kde.nDim - nband,), np.nan)
            points = np.concatenate([points, pad], axis=-1)
        return points

    def _phot_errors(self, filters: list[str], kde: KDE, errors: Any, cov: Any,
                     n_points: int) -> tuple[Bandwidth | None, np.ndarray | None]:
        """
        Photometric errors, as a convolved bandwidth or per-point covariances.

        Parameters
        ----------
        filters : list of str
            The filter set.
        kde : KDE
            Its kernel density estimate.
        errors : Any
            The errors, as for conditional(), or None.
        cov : Any
            Their covariance, as for conditional(), or None.
        n_points : int
            The number of points they are for.

        Returns
        -------
        bandwidth : Bandwidth or None
            For errors the same for every point, kde's bandwidth
            convolved with them; otherwise None.
        per_point : numpy.ndarray of float or None
            For errors given per point, their covariance matrices,
            n_points x nband x nband, for KDE's errCov (with errDims the
            filter set's bands); otherwise None.

        Raises
        ------
        TypeError
            If errors or cov has the wrong type.
        ValueError
            If both errors and cov are given, either is invalid (see
            conditional()), or per-point errors are not for n_points
            points.
        """
        if errors is not None and cov is not None:
            raise ValueError("give the photometric errors as errors or cov, not both")
        if errors is None and cov is None:
            return None, None
        phot_cov = (self._errors_cov(filters, errors) if errors is not None
                    else self._check_cov(filters, cov))
        if phot_cov.ndim == 3:
            if phot_cov.shape[0] != n_points:
                raise ValueError(f"per-point photometric errors are for {phot_cov.shape[0]} points, "
                                 f"but {n_points} were given")
            return None, np.ascontiguousarray(phot_cov)
        full = np.zeros((kde.nDim, kde.nDim))
        full[:len(filters), :len(filters)] = phot_cov
        return Bandwidth(kde.h2Inv).convolveCov(full), None

    def _select_filterset(self, filterset: Any) -> list[str]:
        """
        The filter set conditional() uses, added if need be.

        Parameters
        ----------
        filterset : list of str or None
            The filter set, or None for the only one in kdes.

        Returns
        -------
        list of str
            The filter set, now in kdes.

        Raises
        ------
        ValueError
            If filterset is None and kdes does not hold exactly one.
        """
        if filterset is None:
            if len(self._kdes) != 1:
                raise ValueError(f"filterset must be given, since there are {len(self._kdes)} "
                                 "filter sets (not exactly one)")
            return list(next(iter(self._kdes)))
        if not isinstance(filterset, list):
            raise TypeError("filterset must be a list of filter names")
        filters = cast(list[str], filterset)
        if tuple(filters) not in self._kdes:
            self.add_filterset(filters)
        return list(filters)

    def _band_units(self, filters: list[str]) -> list[u.UnitBase]:
        """
        Each filter's photometric unit.

        Parameters
        ----------
        filters : list of str
            The filters.

        Returns
        -------
        list of astropy.units.UnitBase
            Their units (e.g. mag, mag(AB), erg/(s Hz)).
        """
        phot = self.cluster_phot
        assert phot is not None
        return [u.Unit(phot.filter_units[phot.filters.index(f)]) for f in filters]

    def _errors_cov(self, filters: list[str], errors: Any) -> np.ndarray:
        """
        The covariance matrix of independent photometric errors.

        Parameters
        ----------
        filters : list of str
            The filter set.
        errors : float, array_like, or astropy.units.Quantity
            The errors (see conditional()).

        Returns
        -------
        numpy.ndarray of float
            diag(errors^2), nband x nband, in mag^2 or dex^2 by band; or,
            for an M x nband errors array, one such matrix per point,
            M x nband x nband.

        Raises
        ------
        ValueError
            If errors has the wrong shape, negative or non-finite
            values, or units that do not match a band's.
        """
        nband = len(filters)
        if isinstance(errors, u.Quantity):
            quantity = cast(u.Quantity, errors)
            for f, band in zip(filters, self._band_units(filters), strict=True):
                if not _error_unit_matches(quantity.unit, band):
                    raise ValueError(f"errors are in {quantity.unit}, which does not match filter {f}'s "
                                     f"photometric unit {band}: " + _error_unit_rule(band))
            values = np.asarray(quantity.value, dtype=np.float64)
        else:
            values = np.asarray(errors, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(nband, float(values))
        if values.shape[-1:] != (nband,) or values.ndim > 2:
            raise ValueError(f"errors must be a single value, or have one per band ({nband}), for "
                             "every point or (as an M x nband array) for each of M points")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("errors must be finite and non-negative")
        return (values ** 2)[..., :, np.newaxis] * np.eye(nband)

    def _check_cov(self, filters: list[str], cov: Any) -> np.ndarray:
        """
        Check a covariance matrix of photometric errors.

        Parameters
        ----------
        filters : list of str
            The filter set.
        cov : array_like or astropy.units.Quantity
            The covariance matrix (see conditional()).

        Returns
        -------
        numpy.ndarray of float
            cov, nband x nband (or M x nband x nband, one per point), in
            mag^2 or dex^2 by band.

        Raises
        ------
        ValueError
            If cov has the wrong shape, non-finite values, or units that
            do not match the bands' (or a Quantity is given for a filter
            set mixing magnitudes with other units).
        """
        nband = len(filters)
        if isinstance(cov, u.Quantity):
            quantity = cast(u.Quantity, cov)
            is_mag = [_is_mag_unit(band) for band in self._band_units(filters)]
            if all(is_mag):
                want = u.mag ** 2
            elif not any(is_mag):
                want = u.dex ** 2
            else:
                raise ValueError("cov cannot be a Quantity for a filter set that mixes magnitudes "
                                 "with other units: give it as a plain array, in mag^2 and dex^2")
            if quantity.unit != want:
                raise ValueError(f"cov is in {quantity.unit}, but must be in {want} for filters {filters}")
            values = np.asarray(quantity.value, dtype=np.float64)
        else:
            values = np.asarray(cov, dtype=np.float64)
        if values.shape[-2:] != (nband, nband) or values.ndim > 3:
            raise ValueError(f"cov must be {nband} x {nband}, one row and column per band, for every "
                             f"point, or M x {nband} x {nband}, for each of M points")
        if not np.all(np.isfinite(values)):
            raise ValueError("cov must be finite")
        return values

    def remove_filterset(self, filters: list[str]) -> None:
        """
        Remove a filter set and its kernel density estimate.

        Parameters
        ----------
        filters : list of str
            The filter set, as given to add_filterset().

        Raises
        ------
        KeyError
            If there is no such filter set.
        """
        key = tuple(filters)
        if key not in self._kdes:
            raise KeyError(f"no filter set {list(key)}")
        del self._kdes[key]
        self._default_bw.discard(key)

    def _phot_column(self, name: str) -> np.ndarray:
        """
        One filter's photometry for every library cluster, as stored in a kernel density estimate.

        Parameters
        ----------
        name : str
            The filter.

        Returns
        -------
        numpy.ndarray of float
            The photometry, chosen by use_nebular and use_extinct with
            fallback (see add_filterset()), as magnitudes or log10 of
            any other unit; -inf where a flux is zero in every variant,
            NaN where no variant has a value.
        """
        phot = self.cluster_phot
        assert phot is not None
        is_mag = _is_mag_unit(self._band_units([name])[0])
        result: np.ndarray | None = None
        for suffix in _PHOT_CHAINS[(self._use_nebular, self._use_extinct)]:
            # Read a fallback variant only if values are still missing
            missing = (None if result is None
                       else np.isnan(result) if is_mag else (np.isnan(result) | (result == 0.0)))
            if missing is not None and not np.any(missing):
                break
            try:
                data = phot[name + suffix]
            except (KeyError, IndexError):
                continue  # no such variant for this filter
            values = np.asarray(cast(u.Quantity, data).value if isinstance(data, u.Quantity) else data,
                                dtype=np.float64)
            if missing is None:
                result = values.copy()  # a copy, since phot caches values
            else:
                assert result is not None
                result[missing] = values[missing]
        if result is None:
            return np.full(self._size, np.nan)
        if is_mag:
            return result
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(result)

    def _physical_column(self, name: str) -> np.ndarray:
        """
        One physical quantity for every library cluster, as stored in a kernel density estimate.

        Parameters
        ----------
        name : str
            "mass", "age", "A_V", or "metallicity".

        Returns
        -------
        numpy.ndarray of float
            log10 of the target mass (Msun) or age (yr), or A_V (mag) or
            [Fe/H] as they are, from the cached library values.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            if name == "mass":
                return np.log10(self._mass.to_value(u.Msun))
            if name == "age":
                return np.log10(self._age.to_value(u.yr))
        if name == "A_V":
            assert self._av is not None  # A_V is singular, so never inferred, without extinction
            return np.asarray(self._av.to_value(u.mag), dtype=np.float64)
        return self._feh

    def _read_physical(self) -> tuple[u.Quantity, u.Quantity, u.Quantity | None, np.ndarray]:
        """
        Read every library cluster's physical properties from disk.

        Returns
        -------
        mass : astropy.units.Quantity
            Target masses (Msun).
        age : astropy.units.Quantity
            Ages (yr): the output times, since every cluster in a library
            forms at time zero.
        A_V : astropy.units.Quantity or None
            V-band extinctions (mag), or None if the library has no
            extinction.
        feh : numpy.ndarray
            [Fe/H].
        """
        clusters = self.clusters
        phot = self.cluster_phot
        assert clusters is not None and phot is not None
        mass = cast(u.Quantity, clusters["target_mass"])
        age = cast(u.Quantity, phot["time"])
        datasets = clusters.keys()
        av = cast(u.Quantity, clusters["A_V"]) if "A_V" in datasets else None
        feh = np.asarray(clusters["feh"], dtype=np.float64)
        return mass, age, av, feh

    def _compute_sampling_density(self) -> np.ndarray:
        """
        The density from which the library drew each cluster's physical properties.

        Returns
        -------
        numpy.ndarray of float
            The product, over the non-singular physical quantities, of
            the library's PDF for each at the cluster's value of it (a
            read-only array; all ones if every quantity is singular).
        """
        values = {"mass": self._mass.to_value(u.Msun), "age": self._age.to_value(u.yr),
                  "A_V": None if self._av is None else self._av.to_value(u.mag),
                  "metallicity": self._feh}
        density = np.ones(self._size)
        for q, pdf_name in QUANTITY_PDFS.items():
            if q in self._singular:
                continue  # a constant factor, which normalization removes
            density *= np.asarray(self.pdfs[pdf_name](values[q]), dtype=np.float64)
        density.setflags(write=False)
        return density

    def _evaluate_prob_spec(self, spec: ProbSpec, name: str, at_most_one: bool) -> np.ndarray | None:
        """
        A pobs or priors specification as an array of one value per library cluster.

        Parameters
        ----------
        spec : ProbSpec
            The (already checked) specification.
        name : str
            "pobs" or "priors", for error messages.
        at_most_one : bool
            Whether values must be at most one (pobs).

        Returns
        -------
        numpy.ndarray of float or None
            spec itself if it is an array, a callable's result (called
            with the cached mass, age, A_V, and [Fe/H]; see
            add_filterset()), or None if spec is None.

        Raises
        ------
        ValueError
            If a callable's result is not a valid array (see
            _check_prob_spec()).
        """
        if spec is None:
            return None
        if callable(spec):
            result = np.asarray(spec(self._mass, self._age, self._av, self._feh), dtype=np.float64)
            return cast(np.ndarray, _check_prob_spec(result, f"{name} (the callable's result)",
                                                     self._size, at_most_one))
        return spec

    def _library_weights(self, pobs: np.ndarray | None, priors: np.ndarray | None) -> np.ndarray | None:
        """
        Every library cluster's weight, from pobs and priors.

        Parameters
        ----------
        pobs : numpy.ndarray of float or None
            pobs, as an array (None for every cluster observed).
        priors : numpy.ndarray of float or None
            priors, as an array (None for the library's own sampling
            density).

        Returns
        -------
        numpy.ndarray of float or None
            pobs * priors / sampling_density, evaluated in parallel by
            numexpr (pobs, or priors / sampling_density, if the other is
            None), or None if both are None; zero where sampling_density
            is zero or non-finite (such clusters are never in an
            estimate).
        """
        if pobs is None and priors is None:
            return None
        if priors is None:
            return np.array(pobs, dtype=np.float64)
        operands = {"priors": priors, "sampling": self._sampling_density}
        expr = "priors / sampling"
        if pobs is not None:
            operands["pobs"] = pobs
            expr = "pobs * priors / sampling"
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = cast(np.ndarray, numexpr.evaluate(expr, local_dict=operands))  # pyright: ignore[reportUnknownMemberType]
        weights[~np.isfinite(weights)] = 0.0
        return weights

    def _update_weights(self, pobs: np.ndarray | None, priors: np.ndarray | None) -> None:
        """
        Recompute the library weights, and reweight every kernel density estimate in place.

        Parameters
        ----------
        pobs : numpy.ndarray of float or None
            The new pobs, as an array.
        priors : numpy.ndarray of float or None
            The new priors, as an array.

        Raises
        ------
        ValueError
            If the new weights would leave an estimate with no weight at
            all; nothing is changed in that case.
        """
        weights = self._library_weights(pobs, priors)
        if weights is not None:
            empty = [list(key) for key, kde in self._kdes.items() if not np.any(weights[kde.aux] > 0.0)]
            if empty:
                raise ValueError(f"the new weights leave filter sets {empty} with no weight at all")
        for kde in self._kdes.values():
            kde.w[:] = 1.0 if weights is None else weights[kde.aux]
            kde.reweight()
        self._weights = weights
