"""
slug_reader.py

Implements slug_reader, a lazy reader for slug HDF5 output files.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from typing import Any, cast

import h5py
import numpy as np
import tomlkit

from ._slug import Cluster, Filter, SimControls
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
    galaxy : slug_group_reader or None
        Lazy reader for the galaxy group's datasets (target_mass,
        actual_mass, ...), or None if this file has no galaxy group --
        only a galaxy-type simulation ever has one (read-only).
    galaxy_spectra : slug_group_reader or None
        Lazy reader for the galaxy_spectra group's datasets (wl,
        spec, ...), or None if this file has no galaxy_spectra group
        (read-only).
    galaxy_phot : slug_phot_reader or None
        Lazy reader for the galaxy_phot group's per-filter photometry
        (indexable by filter name, e.g. galaxy_phot["Lbol"]), or None
        if this file has no galaxy_phot group (read-only).
    filters : list of str or None
        Alias for cluster_phot.filters if this file has a cluster_phot
        group, else for galaxy_phot.filters if this file has a
        galaxy_phot group, else None (read-only).
    filter_units : list of str or None
        Alias for cluster_phot.filter_units if this file has a
        cluster_phot group, else for galaxy_phot.filter_units if this
        file has a galaxy_phot group, else None (read-only).
    controls : SimControls
        Simulation controls parsed from this file's own input_deck,
        built the first time this property is accessed and cached
        thereafter; used by get_cluster() to reconstruct clusters
        (read-only).
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
        self._controls: SimControls | None = None

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
            registry_name = self.input_deck.get("phot", {}).get("registry")
            self._groups["cluster_phot"] = slug_phot_reader(
                self._file, "cluster_phot", registry_name=registry_name)
        return cast(slug_phot_reader, self._groups["cluster_phot"])

    @cluster_phot.setter
    def cluster_phot(self, value: Any) -> None:
        raise AttributeError("cluster_phot is read-only")

    @property
    def galaxy(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the galaxy
        group's datasets, built the first time this property is
        accessed and cached thereafter, or None if this file has no
        galaxy group (only a galaxy-type simulation ever has one).
        """
        if "galaxy" not in self._groups:
            return None
        if self._groups["galaxy"] is None:
            self._groups["galaxy"] = slug_group_reader(self._file, "galaxy")
        return self._groups["galaxy"]

    @galaxy.setter
    def galaxy(self, value: Any) -> None:
        raise AttributeError("galaxy is read-only")

    @property
    def galaxy_spectra(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the
        galaxy_spectra group's datasets, built the first time this
        property is accessed and cached thereafter, or None if this
        file has no galaxy_spectra group.
        """
        if "galaxy_spectra" not in self._groups:
            return None
        if self._groups["galaxy_spectra"] is None:
            self._groups["galaxy_spectra"] = slug_group_reader(self._file, "galaxy_spectra")
        return self._groups["galaxy_spectra"]

    @galaxy_spectra.setter
    def galaxy_spectra(self, value: Any) -> None:
        raise AttributeError("galaxy_spectra is read-only")

    @property
    def galaxy_phot(self) -> slug_phot_reader | None:
        """
        slug_phot_reader or None : lazy reader for the galaxy_phot
        group's per-filter photometry, built the first time this
        property is accessed and cached thereafter, or None if this
        file has no galaxy_phot group.
        """
        if "galaxy_phot" not in self._groups:
            return None
        if self._groups["galaxy_phot"] is None:
            registry_name = self.input_deck.get("phot", {}).get("registry")
            self._groups["galaxy_phot"] = slug_phot_reader(
                self._file, "galaxy_phot", registry_name=registry_name)
        return cast(slug_phot_reader, self._groups["galaxy_phot"])

    @galaxy_phot.setter
    def galaxy_phot(self, value: Any) -> None:
        raise AttributeError("galaxy_phot is read-only")

    @property
    def filters(self) -> list[str] | None:
        """
        list of str or None : alias for cluster_phot.filters if this
        file has a cluster_phot group, else for galaxy_phot.filters if
        this file has a galaxy_phot group, else None (read-only).
        """
        phot = self.cluster_phot if self.cluster_phot is not None else self.galaxy_phot
        if phot is None:
            return None
        return phot.filters

    @filters.setter
    def filters(self, value: Any) -> None:
        raise AttributeError("filters is read-only")

    @property
    def filter_units(self) -> list[str] | None:
        """
        list of str or None : alias for cluster_phot.filter_units if
        this file has a cluster_phot group, else for
        galaxy_phot.filter_units if this file has a galaxy_phot group,
        else None (read-only).
        """
        phot = self.cluster_phot if self.cluster_phot is not None else self.galaxy_phot
        if phot is None:
            return None
        return phot.filter_units

    @filter_units.setter
    def filter_units(self, value: Any) -> None:
        raise AttributeError("filter_units is read-only")

    @property
    def controls(self) -> SimControls:
        """
        SimControls : simulation controls parsed from this file's own
        input_deck, built the first time this property is accessed
        and cached thereafter.
        """
        if self._controls is None:
            self._controls = SimControls(tomlkit.dumps(self.input_deck))
        return self._controls

    @controls.setter
    def controls(self, value: Any) -> None:
        raise AttributeError("controls is read-only")

    def get_cluster(self, uid: int) -> Cluster:
        """
        Reconstruct one cluster from this file's clusters group.

        Parameters
        ----------
        uid : int
            Unique ID of the cluster to reconstruct (see
            clusters["uid"]).

        Returns
        -------
        Cluster
            A Cluster built from this cluster's own target_mass and
            rng_state, so its birth-time draws -- birthMass(),
            starMasses(), feH(), aV() -- are reproduced bit-for-bit;
            not yet advanced to any output time.

        Raises
        ------
        RuntimeError
            If this file has no clusters group.
        KeyError
            If uid is not one of this file's clusters.

        Details
        -------
        Built from this reader's own controls property -- see its own
        docstring on how (and how cheaply) that's constructed.
        """
        clusters = self.clusters
        if clusters is None:
            raise RuntimeError("get_cluster: this file has no clusters group")

        uids = clusters["uid"]
        matches = np.nonzero(uids == uid)[0]
        if len(matches) == 0:
            raise KeyError(uid)
        index = int(matches[0])

        rng_state = clusters["rng"][index]
        target_mass = float(clusters["target_mass"][index].value)

        return Cluster(target_mass, uid, 0.0, self.controls, rng_state)

    def get_filter(self, filter_name: str) -> Filter:
        """
        Get a Filter object for one of this file's own filters.

        Parameters
        ----------
        filter_name : str
            Name of the filter to get; must be one of this file's own
            filters (see filters).

        Returns
        -------
        Filter
            The requested filter -- see cluster_phot.get_filter()'s
            own docstring for how it's constructed and cached.

        Raises
        ------
        RuntimeError
            If this file has no cluster_phot group.
        KeyError
            If filter_name is not one of this file's own filters.
        """
        cluster_phot = self.cluster_phot
        if cluster_phot is None:
            raise RuntimeError("get_filter: this file has no cluster_phot group")
        return cluster_phot.get_filter(filter_name)

    def phot_convert(self, phot_to: str) -> None:
        """
        Convert every cached filter's photometry to a new photometric
        system, in place. A no-op if this file has no cluster_phot
        group.

        Parameters
        ----------
        phot_to : str
            The photometric system to convert to: one of "Flambda",
            "Fnu", "ST", "AB", or "Vega".

        Details
        -------
        Thin wrapper around cluster_phot.phot_convert() -- see its own
        docstring for exactly which entries get converted and how.
        """
        cluster_phot = self.cluster_phot
        if cluster_phot is not None:
            cluster_phot.phot_convert(phot_to)
