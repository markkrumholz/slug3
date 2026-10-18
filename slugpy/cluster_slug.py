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
    place (see the pobs property). The other settings are read when a
    filter set is added: changing inference_quantities, use_nebular, or
    use_extinct afterwards does not change estimates already built;
    remove and re-add a filter set to rebuild its estimate.
    """

    def __init__(self, filename: str, pobs: ProbSpec = None, priors: ProbSpec = None,
                 filtersets: list[str] | list[list[str]] | None = None,
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
            accepts; if None (the default), chosen by Silverman's rule.

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

        self._kdes[tuple(names)] = KDE(
            kept, h2Inv=h2Inv, w=kept_weights, names=columns, aux=rows,
            noSplitDims=list(range(len(names), len(columns))))

    def conditional(self, x0: Any, massTol: float = 1e-2, marginalDims: list[int] | None = None,
                    leafSize: int = 16, filterset: list[str] | None = None,
                    errors: Any = None, cov: Any = None) -> KDE:
        """
        Condition a filter set's kernel density estimate on observed photometry.

        Parameters
        ----------
        x0 : array_like of float, shape (nband,) or (nDim,)
            The photometry in the filter set's nband bands, in which case
            the inference quantities are kept as the dimensions of the
            result (the usual case); or a full point in the estimate's
            space (see kdes[...].names), its filters then the inference
            quantities, in which, as for KDE.conditional(), a finite
            value fixes that dimension and NaN keeps it as a dimension
            of the result. Photometry is in the units the estimate
            stores: magnitudes as they are, and log10 of any other unit
            (see add_filterset()).
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
            The photometric errors: one value for every band, or one per
            band of the filter set. For a band in magnitudes, a plain
            number is in mag, and a Quantity must be in mag or the band's
            own magnitude unit (e.g. mag(AB)); for a band in any other
            unit (e.g. erg/(s Hz)), which the estimate stores as log10, a
            plain number is in dex, and a Quantity must be in dex or dex
            of a unit equivalent to the band's. Errors are taken to be
            independent; see cov for correlated ones.
        cov : array_like or astropy.units.Quantity, optional
            The covariance matrix of the photometric errors, nband x
            nband, in the same units as errors, squared: a Quantity must
            be in mag^2 if every band is in magnitudes, or dex^2 if none
            is (a filter set with both needs a plain array, since no
            single astropy unit describes it). May not be given with
            errors.

        Returns
        -------
        KDE
            The conditional estimate over the kept dimensions (see
            KDE.conditional()), with its marginal property the filter
            set's estimate's marginal density at the fixed values. If
            errors or cov is given, the estimate's kernel is first
            convolved with the errors (see Bandwidth.convolveCov()), so
            that the result is the posterior given photometry with those
            errors.

        Raises
        ------
        TypeError
            If filterset, errors, or cov has the wrong type.
        ValueError
            If filterset is None and kdes does not hold exactly one filter
            set; if x0 has neither nband nor nDim elements; if both errors
            and cov are given; if errors or cov has
            the wrong shape, invalid values, or units that do not match
            the bands' photometric systems; or as for add_filterset() and
            KDE.conditional().
        """
        filters = self._select_filterset(filterset)
        kde = self._kdes[tuple(filters)]
        point = np.asarray(x0, dtype=np.float64)
        if point.shape == (len(filters),):
            # Just the photometry: keep every inference quantity
            point = np.concatenate([point, np.full(kde.nDim - len(filters), np.nan)])
        elif point.shape != (kde.nDim,):
            raise ValueError(f"x0 must have one element per band ({len(filters)}) or per dimension "
                             f"of the estimate ({kde.nDim})")
        if errors is not None and cov is not None:
            raise ValueError("give the photometric errors as errors or cov, not both")
        bandwidth = None
        if errors is not None or cov is not None:
            phot_cov = (self._errors_cov(filters, errors) if errors is not None
                        else self._check_cov(filters, cov))
            full = np.zeros((kde.nDim, kde.nDim))
            full[:len(filters), :len(filters)] = phot_cov
            bandwidth = Bandwidth(kde.h2Inv).convolveCov(full)
        return kde.conditional(point, massTol, marginalDims, leafSize, bandwidth=bandwidth)

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
            diag(errors^2), nband x nband, in mag^2 or dex^2 by band.

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
        if values.shape != (nband,):
            raise ValueError(f"errors must be a single value or have one per band ({nband})")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("errors must be finite and non-negative")
        return np.diag(values ** 2)

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
            cov, nband x nband, in mag^2 or dex^2 by band.

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
        if values.shape != (nband, nband):
            raise ValueError(f"cov must be {nband} x {nband}, one row and column per band")
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
        is_mag = phot.filter_units[phot.filters.index(name)].startswith("mag")
        result = np.full(self._size, np.nan)
        for suffix in _PHOT_CHAINS[(self._use_nebular, self._use_extinct)]:
            try:
                data = phot[name + suffix]
            except (KeyError, IndexError):
                continue  # no such variant for this filter
            values = np.asarray(cast(u.Quantity, data).value if isinstance(data, u.Quantity) else data,
                                dtype=np.float64)
            missing = np.isnan(result) if is_mag else (np.isnan(result) | (result == 0.0))
            result[missing] = values[missing]
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
