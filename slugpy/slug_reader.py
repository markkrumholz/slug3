"""
slug_reader.py

Implements slug_reader, a lazy reader for slug HDF5 output files.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import tomlkit
from astropy import units as u

from ._slug import Cluster, Filter, FilterIdeal, SimControls, parsePDFDescriptor
from .cloudy.cloudy_input import write_cloudy_input
from .cloudy.hiiregparam import hiiregparam
from .slug_group_reader import slug_group_reader
from .slug_phot_reader import slug_phot_reader

AnyGroupReader = slug_group_reader | slug_phot_reader

# The six hiiregparam nebular-condition keyword names, in the order
# run_cloudy's own signature lists them
_HII_REG_PARAMS = ("nII", "r0", "r1", "U", "U0", "Omega")


def _find_index(id_arr: np.ndarray, time_arr: u.Quantity | np.ndarray,
    id_val: int, time_val: float) -> int | None:
    """
    Find the row index in id_arr/time_arr matching (id_val, time_val).

    Parameters
    ----------
    id_arr : numpy.ndarray
        Array of uid or trial numbers.
    time_arr : astropy.units.Quantity or numpy.ndarray
        Array of output times, same length as id_arr.
    id_val : int
        The uid or trial number to match.
    time_val : float
        The output time to match, in the same units as time_arr (if
        time_arr is a Quantity, its own .value is compared against).

    Returns
    -------
    int or None
        The index of the first matching row, or None if there is no
        match.
    """
    time_val_arr = cast(u.Quantity, time_arr).value if isinstance(time_arr, u.Quantity) else time_arr
    matches = np.nonzero((id_arr == id_val) & (time_val_arr == time_val))[0]
    return int(matches[0]) if len(matches) > 0 else None


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

    def run_cloudy(self, spec_type: Literal["cluster", "galaxy"],
        uid: int | None = None, trial: int | None = None, time: float | u.Quantity | None = None,
        nII: u.Quantity | float | None = None, r0: u.Quantity | float | None = None,
        r1: u.Quantity | float | None = None, U: u.Quantity | float | None = None,
        U0: u.Quantity | float | None = None, Omega: u.Quantity | float | None = None,
        model_name: str | None = None, output_dir: str | Path = ".",
        template: str | Path | None = None, warn: bool = True,
        fix_quantity: Literal["nII", "r0", "r1", "U", "U0", "Omega"] | None = None,
        r0safety: float = 0.01) -> list[Path]:
        """
        Write cloudy input decks for one or more of this file's own spectra.

        Parameters
        ----------
        spec_type : {"cluster", "galaxy"}
            Which kind of spectrum to use.
        uid : int, optional
            Unique ID of the cluster spectrum to use; only meaningful
            if spec_type is "cluster". If omitted, every cluster
            matching time (or every cluster, if time is also omitted)
            is used.
        trial : int, optional
            Trial number of the galaxy spectrum to use; only
            meaningful if spec_type is "galaxy". If omitted, every
            trial matching time (or every trial, if time is also
            omitted) is used.
        time : float or astropy.units.Quantity, optional
            Output time to use, in yr if a bare float. If omitted,
            every output time matching uid/trial (or every output
            time, if those are also omitted) is used.
        nII, r0, r1, U, U0, Omega : astropy.units.Quantity or float, optional
            Two of these six must be given to fully specify the
            physical conditions of the HII region (see hiiregparam);
            if none are given, nII = 100 cm^-3 and U = 10^-2.5 are
            used as reasonable defaults.
        model_name : str, optional
            Base name for the written input deck(s); defaults to this
            file's own output.model_name.
        output_dir : str or pathlib.Path, default "."
            Directory to write the input deck(s) into.
        template : str or pathlib.Path, optional
            Path to the cloudy input template to use; see
            write_cloudy_input's own default.
        warn : bool, default True
            Passed through to hiiregparam.
        fix_quantity : {"nII", "r0", "r1", "U", "U0", "Omega"}, optional
            Passed through to hiiregparam.
        r0safety : float, default 0.01
            Passed through to hiiregparam.

        Returns
        -------
        list of pathlib.Path
            The cloudy input deck(s) written, one per matching
            spectrum.

        Raises
        ------
        ValueError
            If spec_type is "cluster" and trial is given, or spec_type
            is "galaxy" and uid is given; if no spectrum matches
            uid/trial/time; if spec_type is "cluster" and no matching
            cluster is found in the clusters group; or if this file's
            own stars.FeH entry (for spec_type "galaxy") is neither a
            number nor a PDF file path.

        Notes
        -----
        For each matching spectrum, the ionizing photon rate Q(HI) is
        taken directly from this file's own cluster_phot/galaxy_phot
        group if it has a "Q(HI)" filter; otherwise it is computed
        from the spectrum itself via FilterIdeal("Q(HI)").phot(). For
        spec_type "cluster", [Fe/H] is taken from the matching
        cluster's own entry in the clusters group. For spec_type
        "galaxy", [Fe/H] is instead the expectation value of this
        file's own stars.FeH distribution (or its fixed value, if
        stars.FeH is not a distribution) -- a single population-level
        value used for every matching galaxy spectrum, since there
        is no single [Fe/H] value for a galaxy the way there is for
        a cluster.
        """
        if spec_type == "cluster" and trial is not None:
            raise ValueError("run_cloudy: trial is only meaningful for spec_type='galaxy'")
        if spec_type == "galaxy" and uid is not None:
            raise ValueError("run_cloudy: uid is only meaningful for spec_type='cluster'")

        spectra = self.cluster_spectra if spec_type == "cluster" else self.galaxy_spectra
        if spectra is None:
            raise ValueError(f"run_cloudy: this file has no {spec_type}_spectra group")
        phot = self.cluster_phot if spec_type == "cluster" else self.galaxy_phot

        id_key = "uid" if spec_type == "cluster" else "trial"
        id_val = uid if spec_type == "cluster" else trial
        ids = cast(np.ndarray, spectra[id_key])
        times = cast(u.Quantity, spectra["time"])
        time_val = cast(u.Quantity, time).to_value(u.yr) if isinstance(time, u.Quantity) else time

        mask = np.ones(len(ids), dtype=bool)
        if id_val is not None:
            mask &= (ids == id_val)
        if time_val is not None:
            mask &= (times.value == time_val)
        indices = np.nonzero(mask)[0]
        if len(indices) == 0:
            raise ValueError(
                f"run_cloudy: no {spec_type} spectra match "
                f"{id_key} = {id_val!r}, time = {time!r}")

        nebular_kwargs = {k: v for k, v in
            zip(_HII_REG_PARAMS, (nII, r0, r1, U, U0, Omega), strict=True) if v is not None}
        if not nebular_kwargs:
            nebular_kwargs = {"nII": 100.0 / u.cm ** 3, "U": 10.0 ** -2.5}

        if model_name is None:
            model_name = str(self.input_deck.get("output", {}).get("model_name", "cloudy"))
        output_dir = Path(output_dir)

        # [Fe/H] for a galaxy spectrum is a single population-level
        # value (the expectation value of stars.FeH's own
        # distribution, or its fixed value), computed once here rather
        # than per matching spectrum
        galaxy_feh: float | None = None
        if spec_type == "galaxy":
            feh_entry = self.input_deck.get("stars", {}).get("FeH")
            if isinstance(feh_entry, (int, float)):
                galaxy_feh = float(feh_entry)
            elif isinstance(feh_entry, str):
                galaxy_feh = parsePDFDescriptor(feh_entry).expectationValue()
            else:
                raise ValueError(
                    "run_cloudy: this file's own stars.FeH entry is neither "
                    f"a number nor a PDF file path: {feh_entry!r}")

        wl = cast(u.Quantity, spectra["wl"])
        all_spec = cast(u.Quantity, spectra["spec"])

        written: list[Path] = []
        for idx in indices:
            this_id = int(ids[idx])
            this_time = float(times.value[idx])
            spec = all_spec[idx, :]

            if spec_type == "cluster":
                clusters = self.clusters
                if clusters is None:
                    raise ValueError("run_cloudy: this file has no clusters group")
                cluster_matches = np.nonzero(cast(np.ndarray, clusters["uid"]) == this_id)[0]
                if len(cluster_matches) == 0:
                    raise ValueError(f"run_cloudy: no cluster found with uid = {this_id}")
                feh = float(cast(np.ndarray, clusters["feh"])[int(cluster_matches[0])])
            else:
                feh = cast(float, galaxy_feh)

            qH0: u.Quantity | None = None
            if phot is not None and "Q(HI)" in phot.filters:
                phot_idx = _find_index(cast(np.ndarray, phot[id_key]), cast(u.Quantity, phot["time"]),
                    this_id, this_time)
                if phot_idx is not None:
                    qH0 = cast(u.Quantity, phot["Q(HI)"])[phot_idx]
            if qH0 is None:
                filt = FilterIdeal("Q(HI)")
                value = filt.phot(wl.to_value(u.AA).tolist(), spec.to_value(u.erg / u.s / u.AA).tolist())
                qH0 = value * u.photon / u.s

            hp = hiiregparam(qH0, warn=warn, fix_quantity=fix_quantity, r0safety=r0safety, **nebular_kwargs)

            suffix = f"uid{this_id:09d}" if spec_type == "cluster" else f"trial{this_id:05d}"
            output_path = output_dir / f"{model_name}_{suffix}_t{this_time:.6e}.in"
            written.append(write_cloudy_input(wl, spec, qH0, hp, feh, output_path, template=template))

        return written
