"""
slug_phot_reader.py

Implements slug_phot_reader, a lazy reader for the photometry
(cluster_phot) group of a slug HDF5 output file.
"""

from typing import Any, cast

import h5py
from astropy import units as u

from ._slug import Filter, FilterCollection, PhotSystem
from .phot_convert import _WL_OPTIONAL_PAIRS, _detect_phot_system
from .phot_convert import phot_convert as _phot_convert_value
from .slug_group_reader import Dataset, slug_group_reader


class slug_phot_reader(slug_group_reader):
    """
    A lazy reader for the cluster_phot group of a slug HDF5 output
    file. Extends slug_group_reader with per-filter access: indexing
    by a filter name (e.g. "HST.WFC3_UVIS1.F555W") returns that
    filter's photometry as an astropy Quantity, read from the phot
    dataset's column for that filter on first access and cached
    thereafter; appending "_ext" to the filter name (e.g.
    "HST.WFC3_UVIS1.F555W_ext") returns the same filter's extincted
    photometry from phot_extinct instead. Any other key falls through
    to slug_group_reader's own __getitem__ (e.g. "trial", "time",
    "uid").

    Parameters
    ----------
    file : h5py.File
        The open slug HDF5 output file containing this group.
    group_name : str
        Name of the group within file to read.
    registry_name : str, optional
        Name of the filter registry file to resolve tabulated filter
        names against (see get_filter()); if omitted, uses
        FilterCollection's own built-in default registry.

    Attributes
    ----------
    filters : list of str
        Names of the filters this group has photometry for, in the
        order used to index the phot/phot_extinct datasets
        (read-only).
    filter_units : list of str
        Units of each filter's photometry, in the same order as
        filters (read-only).
    filter_collection : FilterCollection
        A PhotSystem.Vega FilterCollection, initially empty, that
        get_filter() lazily adds filters to on request (read-only).
    """

    def __init__(self, file: h5py.File, group_name: str,
        registry_name: str | None = None) -> None:
        super().__init__(file, group_name)

        group = self._file[self._group_name]
        self._filters: list[str] = list(group.attrs["filters"])
        self._filter_units: list[str] = list(group["phot"].attrs["units"])

        self._phot: dict[str, Dataset | None] = {name: None for name in self._filters}
        self._phot_extinct: dict[str, Dataset | None] | None = None
        if "phot_extinct" in self._datasets:
            self._phot_extinct = {name: None for name in self._filters}

        self._registry_name = registry_name
        self._filter_collection: FilterCollection = (
            FilterCollection([], PhotSystem.Vega) if registry_name is None
            else FilterCollection([], PhotSystem.Vega, registry_name))

    @property
    def filters(self) -> list[str]:
        """
        list of str : names of the filters this group has photometry
        for, in the order used to index the phot/phot_extinct
        datasets (read-only).
        """
        return self._filters

    @property
    def filter_units(self) -> list[str]:
        """
        list of str : units of each filter's photometry, in the same
        order as filters (read-only).
        """
        return self._filter_units

    @property
    def filter_collection(self) -> FilterCollection:
        """
        FilterCollection : a PhotSystem.Vega FilterCollection, initially
        empty, that get_filter() lazily adds filters to on request
        (read-only).
        """
        return self._filter_collection

    @filter_collection.setter
    def filter_collection(self, value: Any) -> None:
        raise AttributeError("filter_collection is read-only")

    def __getitem__(self, key: str) -> Dataset:
        """
        Return one filter's photometry, or fall through to
        slug_group_reader.__getitem__ for a non-filter key.

        Parameters
        ----------
        key : str
            A filter name (e.g. "HST.WFC3_UVIS1.F555W"); that same
            name with "_ext" appended for its extincted counterpart
            (e.g. "HST.WFC3_UVIS1.F555W_ext"); or any dataset name
            from the underlying group (e.g. "trial", "time", "uid").

        Returns
        -------
        astropy.units.Quantity or numpy.ndarray
            The requested photometry or dataset, read from disk (and
            cached) on first access.

        Raises
        ------
        KeyError
            If key is neither a known filter name (with or without a
            trailing "_ext") nor a dataset name in this group.
        """
        extinct = key.endswith("_ext")
        name = key[:-len("_ext")] if extinct else key

        if name not in self._filters:
            return super().__getitem__(key)

        phot = self._phot_extinct if extinct else self._phot
        assert phot is not None
        cached = phot[name]
        if cached is not None:
            return cached

        index = self._filters.index(name)
        dataset_name = "phot_extinct" if extinct else "phot"
        arr = self._file[self._group_name][dataset_name][:, index]
        unit = self._filter_units[index]
        result: Dataset = arr if unit == "" else arr * u.Unit(unit)
        phot[name] = result
        return result

    def get_filter(self, filter_name: str) -> Filter:
        """
        Get a Filter object for one of this group's own filters,
        constructing (and caching, on filter_collection) it on first
        request.

        Parameters
        ----------
        filter_name : str
            Name of the filter to get; must be one of this group's own
            filters (see filters).

        Returns
        -------
        Filter
            The requested filter, as whichever concrete type
            (FilterIdeal or FilterTabulated) it actually is, from
            filter_collection.

        Raises
        ------
        KeyError
            If filter_name is not one of this group's own filters.
        RuntimeError
            If filter_name is one of this group's own filters (e.g.
            "Lbol") but does not itself match a recognized tabulated-
            or idealized-filter naming convention, so it cannot
            actually be constructed as a Filter.
        """
        if filter_name not in self._filters:
            raise KeyError(filter_name)

        names = self._filter_collection.filterNames()
        if filter_name in names:
            return self._filter_collection.getFilter(names.index(filter_name))

        if self._registry_name is None:
            self._filter_collection.addFilter(filter_name)
        else:
            self._filter_collection.addFilter(filter_name, self._registry_name)
        index = self._filter_collection.filterNames().index(filter_name)
        return self._filter_collection.getFilter(index)

    def _convert_phot_dict(self, phot_dict: dict[str, Dataset | None],
        suffix: str, phot_to: str) -> None:
        """Convert every convertible entry of phot_dict to phot_to, in place."""
        for filter_name in list(phot_dict.keys()):
            try:
                value = self[filter_name + suffix]
            except IndexError:
                # e.g. filter_name + "_ext" for a filter (Lbol) that
                # phot_extinct has no column for at all -- see
                # __getitem__'s own docstring
                continue

            if not isinstance(value, u.Quantity):
                continue
            # isinstance already guarantees this at runtime; cast for
            # the same reason as phot_convert.py's own wl handling --
            # astropy ships no type stubs, so pyright can't narrow
            # ndarray | Quantity against it on its own
            quantity = cast(u.Quantity, value)
            try:
                phot_from = _detect_phot_system(quantity.unit)
            except ValueError:
                continue
            if phot_from == phot_to:
                continue

            wl = None
            if (phot_from, phot_to) not in _WL_OPTIONAL_PAIRS:
                wl = self.get_filter(filter_name).wlPivot()

            filt = None
            if "Vega" in (phot_from, phot_to):
                filt = self.get_filter(filter_name)

            phot_dict[filter_name] = _phot_convert_value(
                quantity, phot_to, wl=wl, filt=filt)

    def phot_convert(self, phot_to: str) -> None:
        """
        Convert every cached filter's photometry to a new photometric
        system, in place.

        Parameters
        ----------
        phot_to : str
            The photometric system to convert to: one of "Flambda",
            "Fnu", "ST", "AB", or "Vega".

        Details
        -------
        Every entry of _phot and _phot_extinct is read (triggering its
        usual lazy load from disk, if not already cached) and, if its
        current unit is one phot_convert.phot_convert() recognizes,
        replaced with its own conversion to phot_to -- so subsequent
        __getitem__ calls for that filter return the converted value.
        Entries with a non-photometric unit (Lbol's Lsun, or an
        idealized filter's photon/s) are left untouched, as is any
        entry already in phot_to's own system. A conversion that needs
        a wavelength or a Vega zero point gets it from
        get_filter(filter_name) -- see its own docstring on how (and
        from where) that filter gets built.
        """
        self._convert_phot_dict(self._phot, "", phot_to)
        if self._phot_extinct is not None:
            self._convert_phot_dict(self._phot_extinct, "_ext", phot_to)
