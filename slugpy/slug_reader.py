"""
slug_reader.py

Implements slug_reader, a lazy reader for slug HDF5 output files.
"""

from typing import Any, cast

import h5py
import tomlkit

from .slug_group_reader import slug_group_reader
from .slug_phot_reader import slug_phot_reader

AnyGroupReader = slug_group_reader | slug_phot_reader


class slug_reader:
    """
    A lazy reader for a slug HDF5 output file. Metadata is read on
    construction; the output data itself (cluster properties, spectra,
    photometry, ...) is only read from disk when a caller actually
    asks for it.

    Parameters
    ----------
    filename : str
        Path to the slug HDF5 output file to open.

    Attributes
    ----------
    slug_hash : str
        Git commit hash of the slug build that produced this file.
    date : str
        Date the run that produced this file started, as YYYY-MM-DD.
    time : str
        Time of day the run that produced this file started.
    rng_state : str
        Serialized state of the random number generator at the start
        of the run, one space-separated pcg64 state per thread.
    input_deck : tomlkit.TOMLDocument
        The input deck used to produce this file, read from the
        input_deck group's toml dataset and parsed on first access
        (read-only).
    clusters : slug_group_reader or None
        Lazy reader for the clusters group's datasets (target_mass,
        birth_mass, ...), or None if this file has no clusters group
        (read-only).
    cluster_spectra : slug_group_reader or None
        Lazy reader for the cluster_spectra group's datasets (wl,
        spec, ...), or None if this file has no cluster_spectra group
        (read-only).
    cluster_phot : slug_phot_reader or None
        Lazy reader for the cluster_phot group's per-filter photometry
        (indexable by filter name, e.g. cluster_phot["Lbol"]), or None
        if this file has no cluster_phot group (read-only).
    filters : list of str or None
        Alias for cluster_phot.filters, or None if this file has no
        cluster_phot group (read-only).
    filter_units : list of str or None
        Alias for cluster_phot.filter_units, or None if this file has
        no cluster_phot group (read-only).
    """

    def __init__(self, filename: str) -> None:
        self._file: h5py.File = h5py.File(filename, "r")

        self.slug_hash: str = self._file.attrs["slug-hash"]
        self.date: str = self._file.attrs["date"]
        self.time: str = self._file.attrs["time"]
        self.rng_state: str = self._file.attrs["rng_state"]

        self._groups: dict[str, AnyGroupReader | None] = {name: None for name in self._file
            if isinstance(self._file[name], h5py.Group)}

        self._input_deck: tomlkit.TOMLDocument | None = None

    @property
    def input_deck(self) -> tomlkit.TOMLDocument:
        """
        tomlkit.TOMLDocument : the input deck used to produce this
        file, parsed from the input_deck group's toml dataset the
        first time this property is accessed, and cached thereafter.
        """
        if self._input_deck is not None:
            return self._input_deck
        toml_str = self._file["input_deck"]["toml"][()]
        self._input_deck = tomlkit.parse(toml_str)
        return self._input_deck

    @input_deck.setter
    def input_deck(self, value: Any) -> None:
        raise AttributeError("input_deck is read-only")

    @property
    def clusters(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the clusters
        group's datasets, built the first time this property is
        accessed and cached thereafter, or None if this file has no
        clusters group.
        """
        if "clusters" not in self._groups:
            return None
        if self._groups["clusters"] is None:
            self._groups["clusters"] = slug_group_reader(self._file, "clusters")
        return self._groups["clusters"]

    @clusters.setter
    def clusters(self, value: Any) -> None:
        raise AttributeError("clusters is read-only")

    @property
    def cluster_spectra(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the
        cluster_spectra group's datasets, built the first time this
        property is accessed and cached thereafter, or None if this
        file has no cluster_spectra group.
        """
        if "cluster_spectra" not in self._groups:
            return None
        if self._groups["cluster_spectra"] is None:
            self._groups["cluster_spectra"] = slug_group_reader(self._file, "cluster_spectra")
        return self._groups["cluster_spectra"]

    @cluster_spectra.setter
    def cluster_spectra(self, value: Any) -> None:
        raise AttributeError("cluster_spectra is read-only")

    @property
    def cluster_phot(self) -> slug_phot_reader | None:
        """
        slug_phot_reader or None : lazy reader for the cluster_phot
        group's per-filter photometry, built the first time this
        property is accessed and cached thereafter, or None if this
        file has no cluster_phot group.
        """
        if "cluster_phot" not in self._groups:
            return None
        if self._groups["cluster_phot"] is None:
            self._groups["cluster_phot"] = slug_phot_reader(self._file, "cluster_phot")
        return cast(slug_phot_reader, self._groups["cluster_phot"])

    @cluster_phot.setter
    def cluster_phot(self, value: Any) -> None:
        raise AttributeError("cluster_phot is read-only")

    @property
    def filters(self) -> list[str] | None:
        """
        list of str or None : alias for cluster_phot.filters, or None
        if this file has no cluster_phot group (read-only).
        """
        cluster_phot = self.cluster_phot
        if cluster_phot is None:
            return None
        return cluster_phot.filters

    @filters.setter
    def filters(self, value: Any) -> None:
        raise AttributeError("filters is read-only")

    @property
    def filter_units(self) -> list[str] | None:
        """
        list of str or None : alias for cluster_phot.filter_units, or
        None if this file has no cluster_phot group (read-only).
        """
        cluster_phot = self.cluster_phot
        if cluster_phot is None:
            return None
        return cluster_phot.filter_units

    @filter_units.setter
    def filter_units(self, value: Any) -> None:
        raise AttributeError("filter_units is read-only")
