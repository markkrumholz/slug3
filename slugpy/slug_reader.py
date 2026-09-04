"""
slug_reader.py

Implements slug_reader, a lazy reader for slug HDF5 output files.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import concurrent.futures
import re
import shutil
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import tomlkit
from astropy import units as u

from ._slug import Cluster, Filter, FilterIdeal, SimControls, parsePDFDescriptor
from .cloudy.cloudy_continuum import read_cloudy_continuum
from .cloudy.cloudy_input import write_cloudy_input
from .cloudy.cloudy_lines import read_cloudy_linearr
from .cloudy.cloudy_output import (
    CloudyRunResult,
    delete_cloudy_h5_rows,
    write_cloudy_h5_results,
)
from .cloudy.cloudy_process import (
    cloudy_run_succeeded,
    find_cloudy_executable,
    run_cloudy_decks,
)
from .cloudy.hiiregparam import hiiregparam
from .slug_group_reader import slug_group_reader
from .slug_phot_reader import slug_phot_reader
from .slug_spectra_reader import slug_spectra_reader

AnyGroupReader = slug_group_reader | slug_phot_reader | slug_spectra_reader

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


def _find_matching_cloudy_row(group: slug_group_reader | None, id_key: str, id_val: int,
    time_val: float, hp: hiiregparam) -> int | None:
    """
    Find the row already stored in a cluster_cloudy/galaxy_cloudy
    group whose id/time and all six hiiregparam values exactly match.

    Parameters
    ----------
    group : slug_group_reader or None
        The cluster_cloudy or galaxy_cloudy group reader to search, or
        None if that group doesn't exist yet (e.g. no cloudy run has
        ever been written to this file).
    id_key : str
        Name of the id dataset: "uid" or "trial".
    id_val : int
        The uid or trial number to match.
    time_val : float
        The output time to match, in yr.
    hp : hiiregparam
        The physical conditions (nII, r0, r1, U, U0, Omega) a
        candidate row must also match exactly.

    Returns
    -------
    int or None
        The index of the first row whose id, time, and all six
        hiiregparam values all match, or None if group is None or no
        such row exists.
    """
    if group is None:
        return None
    ids = cast(np.ndarray, group[id_key])
    times = cast(u.Quantity, group["time"]).value
    mask = (ids == id_val) & (times == time_val)
    if not np.any(mask):
        return None
    for name, unit, hp_val in (
        ("nII", u.cm ** -3, hp.nII), ("r0", u.cm, hp.r0), ("r1", u.cm, hp.r1),
        ("U", u.dimensionless_unscaled, hp.U), ("U0", u.dimensionless_unscaled, hp.U0),
        ("Omega", u.dimensionless_unscaled, hp.Omega)):
        stored_raw = group[name]
        if isinstance(stored_raw, u.Quantity):
            stored = cast(Any, stored_raw).to_value(unit)
        else:
            stored = np.asarray(stored_raw)
        mask &= (stored == hp_val.to_value(unit))
    matches = np.nonzero(mask)[0]
    return int(matches[0]) if len(matches) > 0 else None


def _thread_files_in_dir(dir_path: Path) -> list[str]:
    """
    Sorted thread_NNNN.h5 files directly inside dir_path -- one
    OpenMP thread's own output file, in a checkpoint (or, without
    checkpointing, the whole run's own) directory that OutputManagerH5
    has not consolidated -- mirrors OutputManagerH5::consolidateFiles's
    own C++ file-matching rule (name starts with "thread_", suffix
    ".h5"), and its own sort-by-name (equivalent to numeric order,
    since the thread number is zero-padded to a fixed width).

    Raises
    ------
    FileNotFoundError
        If dir_path holds no matching files.
    """
    files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.name.startswith("thread_") and p.suffix == ".h5")
    if not files:
        raise FileNotFoundError(f"no thread_*.h5 files found in {dir_path}")
    return [str(p) for p in files]


def _checkpoint_files(prefix: Path, checkpoint_num: int) -> list[str]:
    """
    The file(s) belonging to one checkpoint of prefix (prefix.parent /
    (prefix.name + "_chkNNNNN")).

    Returns
    -------
    list of str
        [prefix.parent / (name + ".h5")] if that consolidated file
        exists, else every thread_NNNN.h5 file in prefix.parent /
        name -- mirrors OutputManagerH5::restartSetup()'s own
        precedence between the two shapes a checkpoint can be in (see
        its own comment).

    Raises
    ------
    FileNotFoundError
        If neither shape exists.
    """
    name = f"{prefix.name}_chk{checkpoint_num:05d}"
    h5_path = prefix.parent / f"{name}.h5"
    if h5_path.is_file():
        return [str(h5_path)]
    dir_path = prefix.parent / name
    if dir_path.is_dir():
        return _thread_files_in_dir(dir_path)
    raise FileNotFoundError(
        f"checkpoint {checkpoint_num} of {prefix} found, but neither "
        f"{h5_path} nor {dir_path} exists")


def _find_output_files(filename: str) -> tuple[list[str], list[str]]:
    """
    Resolve filename to the underlying HDF5 file(s) making up one
    slug_reader's own output.

    Parameters
    ----------
    filename : str
        Either a path to a single HDF5 file (if it ends in ".h5" or
        ".hdf5"), or a slug run's own model name -- really, outDir/
        model_name together, e.g. the same string SimControls.outDir()
        + "/" + SimControls.modelName() would give, with no extension
        -- in which case this searches outDir for every checkpoint
        this run left behind: either model_name_chkNNNNN.h5 (a
        consolidated checkpoint) or model_name_chkNNNNN/ (holding that
        checkpoint's own, not-yet-consolidated thread_NNNN.h5 files) --
        see OutputManagerH5::restartSetup()'s own comment, which this
        mirrors. If no checkpoints are found at all, falls back to a
        plain, non-checkpointed run's own output instead: model_name.h5
        if it exists, else model_name/ (holding that run's own
        thread_NNNN.h5 files, if it used h5divided output without
        checkpointing).

    Returns
    -------
    (all_files, last_checkpoint_files) : tuple of list of str
        all_files : every underlying HDF5 file, in ascending
        checkpoint order (or, within one checkpoint, in
        _checkpoint_files()'s own order) -- pass to the group readers
        for data aggregation; row order across files is not meaningful
        (see OutputManagerH5::consolidateFiles's own comment), so this
        order is only for determinism, not correctness.
        last_checkpoint_files : just the file(s) belonging to the
        most recent (highest-numbered) checkpoint -- the same as
        all_files if there is no checkpointing at all -- the files to
        read trials_completed/restart_uid from (see slug_reader's own
        docstring for why those, unlike slug-hash/date/time/rng_state,
        are not safe to read from just any one file).

    Raises
    ------
    FileNotFoundError
        If filename ends in ".h5"/".hdf5" but does not exist, or if it
        names a model with no matching output found at all.
    """
    # Resolved to an absolute path immediately, before anything below
    # ever gets stored: every path this returns is later reopened on
    # demand rather than held open (see slug_reader.__init__'s own
    # comment), so a relative path captured here would silently break
    # once the caller's own working directory ever changed, even
    # though the file itself never moved
    if filename.endswith((".h5", ".hdf5")):
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(filename)
        return [str(path)], [str(path)]

    prefix = Path(filename).resolve()
    chk_re = re.compile(re.escape(prefix.name) + r"_chk(\d{5})$")

    checkpoints: dict[int, list[str]] = {}
    if prefix.parent.is_dir():
        for entry in prefix.parent.iterdir():
            if entry.is_file() and entry.suffix == ".h5":
                candidate = entry.stem
            elif entry.is_dir():
                candidate = entry.name
            else:
                continue
            match = chk_re.fullmatch(candidate)
            if match is None:
                continue
            checkpoint_num = int(match.group(1))
            checkpoints[checkpoint_num] = _checkpoint_files(prefix, checkpoint_num)

    if checkpoints:
        all_files: list[str] = []
        for checkpoint_num in sorted(checkpoints):
            all_files.extend(checkpoints[checkpoint_num])
        return all_files, checkpoints[max(checkpoints)]

    plain_h5 = prefix.parent / f"{prefix.name}.h5"
    if plain_h5.is_file():
        return [str(plain_h5)], [str(plain_h5)]
    if prefix.is_dir():
        files = _thread_files_in_dir(prefix)
        return files, files

    raise FileNotFoundError(
        f"no slug output found matching model name {filename!r} -- looked "
        f"for {plain_h5}, {prefix} as a thread directory, and "
        f"{prefix.name}_chkNNNNN.h5/{prefix.name}_chkNNNNN/ checkpoints in "
        f"{prefix.parent}")


class slug_reader:
    """
    A lazy reader for slug HDF5 output, spread across one file or many
    (checkpointed and/or h5divided output -- see filename below).
    Metadata is read on construction; the output data itself (cluster
    properties, spectra, photometry, ...) is only read from disk when
    a caller actually asks for it, and only ever one underlying file
    at a time -- see slug_group_reader's own docstring.

    Parameters
    ----------
    filename : str
        Either a path to a single slug HDF5 output file (if it ends in
        ".h5" or ".hdf5"), or a run's own model name (really, outDir/
        model_name together, with no extension) to search for and
        aggregate every checkpoint (and, within each, every OpenMP
        thread's own file, if not yet consolidated) that run left
        behind -- see _find_output_files's own docstring for the exact
        search rule, which mirrors OutputManagerH5::restartSetup()'s
        own.

    Attributes
    ----------
    slug_hash : str
        Git commit hash of the slug build that produced this output.
    date : str
        Date the run that produced this output started, as YYYY-MM-DD.
    time : str
        Time of day the run that produced this output started.
    rng_state : str
        Serialized state of the random number generator at the start
        of the run, one space-separated pcg64 state per thread.
    trials_completed : int or None
        Number of trials the run that produced this output had
        completed as of its own most recent checkpoint (see
        OutputManagerH5::closeOutputFile()'s own "trials_completed"
        attribute) -- unlike slug_hash/date/time/rng_state above, this
        (and restart_uid below) is read from the *last* checkpoint
        specifically, not just any one file, since -- unlike those --
        it differs from one checkpoint to the next. None if this
        output predates checkpointing (no such attribute at all), or
        if its own most recent checkpoint was still open (being
        written to) when this was read, rather than already closed
        (read-only).
    restart_uid : int or None
        The utils::uniqueID() value a restart of this run would resume
        from, as of the same checkpoint trials_completed was read from
        (see OutputManagerH5::closeOutputFile()'s own "restart_uid"
        attribute) -- same None cases as trials_completed (read-only).
    input_deck : tomlkit.TOMLDocument
        The input deck used to produce this file, read from the
        input_deck group's toml dataset and parsed on first access
        (read-only).
    clusters : slug_group_reader or None
        Lazy reader for the clusters group's datasets (target_mass,
        birth_mass, ...), or None if this file has no clusters group
        (read-only).
    cluster_spectra : slug_spectra_reader or None
        Lazy reader for the cluster_spectra group's datasets (wl,
        spec, ...), including nebular emission data (spec_neb,
        line_labels, neb_lines, ...) if this file's own simulation
        requested it -- see slug_spectra_reader's own docstring -- or
        None if this file has no cluster_spectra group (read-only).
    cluster_phot : slug_phot_reader or None
        Lazy reader for the cluster_phot group's per-filter photometry
        (indexable by filter name, e.g. cluster_phot["Lbol"]; "_ex",
        "_neb", and "_neb_ex" suffixes select extincted, nebular-
        inclusive, and nebular-inclusive-and-extincted variants --
        see slug_phot_reader's own docstring), or None if this file
        has no cluster_phot group (read-only).
    galaxy : slug_group_reader or None
        Lazy reader for the galaxy group's datasets (target_mass,
        actual_mass, ...), or None if this file has no galaxy group --
        only a galaxy-type simulation ever has one (read-only).
    galaxy_spectra : slug_spectra_reader or None
        Lazy reader for the galaxy_spectra group's datasets (wl,
        spec, ...), including nebular emission data (spec_neb,
        line_labels, neb_lines, ...) if this file's own simulation
        requested it -- see slug_spectra_reader's own docstring -- or
        None if this file has no galaxy_spectra group (read-only).
    galaxy_phot : slug_phot_reader or None
        Lazy reader for the galaxy_phot group's per-filter photometry
        (indexable by filter name, e.g. galaxy_phot["Lbol"]; "_ex",
        "_neb", and "_neb_ex" suffixes select extincted, nebular-
        inclusive, and nebular-inclusive-and-extincted variants --
        see slug_phot_reader's own docstring), or None if this file
        has no galaxy_phot group (read-only).
    cluster_cloudy : slug_group_reader or None
        Lazy reader for the cluster_cloudy group's datasets (uid,
        time, nII, ...), written by run_cloudy(spec_type="cluster"),
        or None if this file has no cluster_cloudy group (read-only).
    galaxy_cloudy : slug_group_reader or None
        Lazy reader for the galaxy_cloudy group's datasets (trial,
        time, nII, ...), written by run_cloudy(spec_type="galaxy"), or
        None if this file has no galaxy_cloudy group (read-only).
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
        # self._file is always a list of underlying file paths, even
        # for the ordinary, single-file case -- never a list of open
        # handles: every read below (and in slug_group_reader/
        # slug_phot_reader/slug_spectra_reader, which share this same
        # list) opens whichever file(s) it actually needs just for the
        # duration of that one read, so this reader never holds more
        # than one file open at a time, no matter how many files it
        # spans (a checkpointed and/or h5divided run can easily span
        # thousands).
        self._file: list[str]
        self._file, last_checkpoint_files = _find_output_files(filename)

        with h5py.File(self._file[0], "r") as f:
            self.slug_hash: str = f.attrs["slug-hash"]
            self.date: str = f.attrs["date"]
            self.time: str = f.attrs["time"]
            self.rng_state: str = f.attrs["rng_state"]
            self._groups: dict[str, AnyGroupReader | None] = {
                name: None for name in f if isinstance(f[name], h5py.Group)}

        # trials_completed/restart_uid differ from one checkpoint to
        # the next (see this class's own docstring), so -- unlike the
        # four attributes above -- these come from the *last*
        # checkpoint specifically, not just self._file[0]; .attrs.get()
        # (not simple indexing) so a still-open last checkpoint (no
        # such attribute written yet -- see
        # OutputManagerH5::closeOutputFile()'s own comment) reads back
        # as None rather than raising
        with h5py.File(last_checkpoint_files[0], "r") as f:
            raw_trials_completed = f.attrs.get("trials_completed")
            raw_restart_uid = f.attrs.get("restart_uid")
            self.trials_completed: int | None = (
                None if raw_trials_completed is None else int(raw_trials_completed))
            self.restart_uid: int | None = (
                None if raw_restart_uid is None else int(raw_restart_uid))

        self._input_deck: tomlkit.TOMLDocument | None = None
        self._controls: SimControls | None = None

    @property
    def input_deck(self) -> tomlkit.TOMLDocument:
        """
        tomlkit.TOMLDocument : the input deck used to produce this
        output, parsed from the first file's own input_deck group's
        toml dataset the first time this property is accessed, and
        cached thereafter.
        """
        if self._input_deck is not None:
            return self._input_deck
        with h5py.File(self._file[0], "r") as f:
            toml_str = f["input_deck"]["toml"][()]
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
    def cluster_spectra(self) -> slug_spectra_reader | None:
        """
        slug_spectra_reader or None : lazy reader for the
        cluster_spectra group's datasets (including nebular emission
        data -- spec_neb, line_labels, neb_lines, ... -- if this
        file's own simulation requested it; see slug_spectra_reader's
        own docstring), built the first time this property is
        accessed and cached thereafter, or None if this file has no
        cluster_spectra group.
        """
        if "cluster_spectra" not in self._groups:
            return None
        if self._groups["cluster_spectra"] is None:
            self._groups["cluster_spectra"] = slug_spectra_reader(self._file, "cluster_spectra")
        return cast(slug_spectra_reader, self._groups["cluster_spectra"])

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
    def galaxy_spectra(self) -> slug_spectra_reader | None:
        """
        slug_spectra_reader or None : lazy reader for the
        galaxy_spectra group's datasets (including nebular emission
        data -- spec_neb, line_labels, neb_lines, ... -- if this
        file's own simulation requested it; see slug_spectra_reader's
        own docstring), built the first time this property is
        accessed and cached thereafter, or None if this file has no
        galaxy_spectra group.
        """
        if "galaxy_spectra" not in self._groups:
            return None
        if self._groups["galaxy_spectra"] is None:
            self._groups["galaxy_spectra"] = slug_spectra_reader(self._file, "galaxy_spectra")
        return cast(slug_spectra_reader, self._groups["galaxy_spectra"])

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
    def cluster_cloudy(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the cluster_cloudy
        group's datasets (uid, time, nII, ..., and -- where available
        -- the continuum and line data run_cloudy wrote), built the
        first time this property is accessed and cached thereafter, or
        None if this file has no cluster_cloudy group (e.g. run_cloudy
        has never been called with spec_type="cluster" on this file).
        """
        if "cluster_cloudy" not in self._groups:
            return None
        if self._groups["cluster_cloudy"] is None:
            self._groups["cluster_cloudy"] = slug_group_reader(self._file, "cluster_cloudy")
        return self._groups["cluster_cloudy"]

    @cluster_cloudy.setter
    def cluster_cloudy(self, value: Any) -> None:
        raise AttributeError("cluster_cloudy is read-only")

    @property
    def galaxy_cloudy(self) -> slug_group_reader | None:
        """
        slug_group_reader or None : lazy reader for the galaxy_cloudy
        group's datasets (trial, time, nII, ..., and -- where
        available -- the continuum and line data run_cloudy wrote),
        built the first time this property is accessed and cached
        thereafter, or None if this file has no galaxy_cloudy group
        (e.g. run_cloudy has never been called with spec_type="galaxy"
        on this file).
        """
        if "galaxy_cloudy" not in self._groups:
            return None
        if self._groups["galaxy_cloudy"] is None:
            self._groups["galaxy_cloudy"] = slug_group_reader(self._file, "galaxy_cloudy")
        return self._groups["galaxy_cloudy"]

    @galaxy_cloudy.setter
    def galaxy_cloudy(self, value: Any) -> None:
        raise AttributeError("galaxy_cloudy is read-only")

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
        uid: int | list[int] | None = None, trial: int | list[int] | None = None,
        time: float | u.Quantity | list[float | u.Quantity] | None = None,
        nII: u.Quantity | float | None = None, r0: u.Quantity | float | None = None,
        r1: u.Quantity | float | None = None, U: u.Quantity | float | None = None,
        U0: u.Quantity | float | None = None, Omega: u.Quantity | float | None = None,
        model_name: str | None = None, temp_dir: str | Path = "cloudy_tmp",
        template: str | Path | None = None, warn: bool = True,
        fix_quantity: Literal["nII", "r0", "r1", "U", "U0", "Omega"] | None = None,
        r0safety: float = 0.01, cloudy_path: str | Path | None = None,
        max_workers: int | None = None, overwrite: bool = False,
        save_temp: bool = False, progress: bool = True,
        executor: concurrent.futures.Executor | None = None) -> bool | list[bool]:
        """
        Write cloudy input decks for one or more of this file's own
        spectra, run cloudy on each of them, and save the results of
        every successful run into this file's own cluster_cloudy or
        galaxy_cloudy group.

        Parameters
        ----------
        spec_type : {"cluster", "galaxy"}
            Which kind of spectrum to use.
        uid : int or list of int, optional
            Unique ID(s) of the cluster spectrum/spectra to use; only
            meaningful if spec_type is "cluster". If omitted, every
            cluster matching time (or every cluster, if time is also
            omitted) is used.
        trial : int or list of int, optional
            Trial number(s) of the galaxy spectrum/spectra to use;
            only meaningful if spec_type is "galaxy". If omitted,
            every trial matching time (or every trial, if time is also
            omitted) is used.
        time : float, astropy.units.Quantity, or list of either, optional
            Output time(s) to use, in yr if a bare float. If omitted,
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
        temp_dir : str or pathlib.Path, default "cloudy_tmp"
            Directory to write the input deck(s) and cloudy's own
            output into. A relative path (the default) is resolved
            against this file's own directory, not the current working
            directory, so cloudy's scratch files never land alongside
            (or get confused with) the HDF5 output itself; pass an
            absolute path to place it somewhere else entirely. See
            save_temp below for whether this directory is kept.
        template : str or pathlib.Path, optional
            Path to the cloudy input template to use; see
            write_cloudy_input's own default.
        warn : bool, default True
            Passed through to hiiregparam.
        fix_quantity : {"nII", "r0", "r1", "U", "U0", "Omega"}, optional
            Passed through to hiiregparam.
        r0safety : float, default 0.01
            Passed through to hiiregparam.
        cloudy_path : str or pathlib.Path, optional
            Path to the cloudy executable. If omitted, it is looked up
            as $CLOUDY_DIR/cloudy.exe (see find_cloudy_executable).
        max_workers : int, optional
            Maximum number of cloudy processes to run at once.
            Defaults to the number of available CPUs; pass a smaller
            value to leave some cores free. Ignored if executor is
            given.
        overwrite : bool, default False
            By default, a matching spectrum whose id/time and all six
            hiiregparam values (nII, r0, r1, U, U0, Omega) exactly
            match an already-stored cluster_cloudy/galaxy_cloudy row
            is skipped -- no deck is written and cloudy is not run for
            it -- and counted as a success, since its data are already
            stored. Pass True to instead always rerun cloudy for every
            matching spectrum, replacing any such already-stored row
            with the new run's results.
        save_temp : bool, default False
            By default, temp_dir (input decks, and every file cloudy
            itself wrote into it) is deleted once every matching
            spectrum has been processed and its results (if any)
            written to this file, regardless of whether any individual
            run succeeded or failed. Pass True to keep temp_dir
            instead, e.g. to inspect a failed run's own cloudy output.
        progress : bool, default True
            Whether to show a progress bar (see slugpy.progress)
            tracking decks completed. A no-op if every matching
            spectrum is skipped as an already-stored duplicate (see
            overwrite above).
        executor : concurrent.futures.Executor, optional
            An already-running executor to submit this call's own
            cloudy runs into, instead of a private max_workers-sized
            pool of this call's own (see run_cloudy_decks). Pass this
            to share one pool's own worker threads across multiple
            run_cloudy calls -- e.g. concurrent calls on different
            slug_reader instances for different files, each run from
            its own driver thread -- so an idle worker immediately
            picks up whichever file's work is next, rather than each
            call's own private pool leaving workers idle once its own
            matching spectra run out while other files' work is still
            queued elsewhere. This call still blocks until every
            matching spectrum here has completed; executor itself is
            left running for the caller to manage.

        Returns
        -------
        bool or list of bool
            Whether each matching spectrum's cloudy run succeeded: a
            skipped duplicate (see overwrite above) counts as a
            success without cloudy having been run at all, and
            everything else is judged by whether its own captured
            stdout's last non-blank line contains "Cloudy exited OK"
            (see cloudy_run_succeeded). A single bool if exactly one
            spectrum matched, otherwise a list of bool in the same
            order as the matched spectra were processed. This call
            blocks until every matching spectrum's cloudy run (skipped
            duplicates aside) has completed.

        Raises
        ------
        ValueError
            If spec_type is "cluster" and trial is given, or spec_type
            is "galaxy" and uid is given; if no spectrum matches
            uid/trial/time; if spec_type is "cluster" and no matching
            cluster is found in the clusters group; or if this file's
            own stars.FeH entry (for spec_type "galaxy") is neither a
            number nor a PDF file path.
        RuntimeError
            If the cloudy executable cannot be located (see
            find_cloudy_executable), or if this reader spans more than
            one underlying HDF5 file (checkpointed and/or h5divided
            output -- see this class's own docstring): there is no
            single file to write new cluster_cloudy/galaxy_cloudy rows
            into in that case, so this refuses up front rather than
            guessing one.

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
        a cluster. Cloudy input decks for every matching spectrum are
        written before any of them are run, and the cloudy executable
        is located before any decks are written, so a missing
        executable is reported without writing partial output.

        Every successful run's own physical conditions (nII, r0, r1,
        U, U0, Omega) and, where its own deck produced one, emergent
        continuum (wavelength grid plus incident/transmitted/emitted/
        transmitted+emitted luminosity) and emission lines (wavelength,
        4-character label, and observable luminosity of every line
        with emergent luminosity above 1e10 erg/s -- see
        read_cloudy_linearr) are appended as new rows to this file's
        own cluster_cloudy (spec_type "cluster") or galaxy_cloudy
        (spec_type "galaxy") group, creating it on first use (this
        reader's own self._groups is refreshed afterward, so a newly-
        created group is discoverable through cluster_cloudy/
        galaxy_cloudy right away); this only happens if at least one
        run in this call succeeded, so a call where every run fails
        leaves the file untouched.

        Duplicate detection (and, with overwrite=True, replacement) is
        keyed on an exact match of id/time and all six hiiregparam
        values against what's already stored -- not just id/time alone
        -- so the same spectrum rerun with different physical
        conditions is always treated as a new, independent run rather
        than a duplicate. With overwrite=True, a matched row is only
        actually replaced once its own rerun succeeds: a failed rerun
        leaves the previously-stored row untouched.

        temp_dir is only ever deleted if this call actually wrote at
        least one deck into it -- a call where every match was skipped
        as an already-stored duplicate never creates or touches
        temp_dir at all.
        """
        if len(self._file) > 1:
            raise RuntimeError(
                "run_cloudy: this reader spans more than one underlying "
                "HDF5 file (checkpointed and/or h5divided output); there "
                "is no single file to write new cluster_cloudy/"
                "galaxy_cloudy rows into")
        if spec_type == "cluster" and trial is not None:
            raise ValueError("run_cloudy: trial is only meaningful for spec_type='galaxy'")
        if spec_type == "galaxy" and uid is not None:
            raise ValueError("run_cloudy: uid is only meaningful for spec_type='cluster'")

        # Locate the cloudy executable before doing any other work, so
        # a missing executable is reported before we bother writing
        # any input decks
        cloudy_exe = find_cloudy_executable(cloudy_path)

        spectra = self.cluster_spectra if spec_type == "cluster" else self.galaxy_spectra
        if spectra is None:
            raise ValueError(f"run_cloudy: this file has no {spec_type}_spectra group")
        phot = self.cluster_phot if spec_type == "cluster" else self.galaxy_phot

        id_key = "uid" if spec_type == "cluster" else "trial"
        id_val = uid if spec_type == "cluster" else trial
        ids = cast(np.ndarray, spectra[id_key])
        times = cast(u.Quantity, spectra["time"])
        if isinstance(time, list):
            time_val = [cast(u.Quantity, t).to_value(u.yr) if isinstance(t, u.Quantity) else t for t in time]
        else:
            time_val = cast(u.Quantity, time).to_value(u.yr) if isinstance(time, u.Quantity) else time

        mask = np.ones(len(ids), dtype=bool)
        if id_val is not None:
            mask &= np.isin(ids, id_val) if isinstance(id_val, list) else (ids == id_val)
        if time_val is not None:
            mask &= np.isin(times.value, time_val) if isinstance(time_val, list) else (times.value == time_val)
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
        temp_dir_path = Path(temp_dir)
        if not temp_dir_path.is_absolute():
            temp_dir_path = Path(self._file[0]).resolve().parent / temp_dir_path

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
        existing_group = self.cluster_cloudy if spec_type == "cluster" else self.galaxy_cloudy

        written: list[Path] = []
        run_ids: list[int] = []
        run_times: list[float] = []
        run_hps: list[hiiregparam] = []
        run_overwrite_rows: list[int | None] = []
        # Aligned with indices: True for a spectrum skipped as an
        # already-stored duplicate (see overwrite above), None for one
        # that got a deck written and was actually run
        skipped: list[bool | None] = []
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

            existing_row = _find_matching_cloudy_row(existing_group, id_key, this_id, this_time, hp)
            if existing_row is not None and not overwrite:
                skipped.append(True)
                continue
            skipped.append(None)

            suffix = f"uid{this_id:09d}" if spec_type == "cluster" else f"trial{this_id:05d}"
            output_path = temp_dir_path / f"{model_name}_{suffix}_t{this_time:.6e}.in"
            written.append(write_cloudy_input(wl, spec, qH0, hp, feh, output_path, template=template))
            run_ids.append(this_id)
            run_times.append(this_time)
            run_hps.append(hp)
            run_overwrite_rows.append(existing_row)

        out_paths = run_cloudy_decks(
            written, cloudy_exe, max_workers=max_workers, progress=progress,
            executor=executor) if written else []
        ran_successes = [cloudy_run_succeeded(p) for p in out_paths]

        results: list[CloudyRunResult] = []
        rows_to_delete: set[int] = set()
        for deck, this_id, this_time, hp, overwrite_row, success in zip(
            written, run_ids, run_times, run_hps, run_overwrite_rows, ran_successes, strict=True):
            if not success:
                continue
            continuum = None
            con_path = deck.with_suffix(".con")
            if con_path.is_file() and con_path.stat().st_size > 0:
                continuum = read_cloudy_continuum(con_path)
            lines = None
            linearr_path = deck.with_suffix(".linearr")
            if linearr_path.is_file() and linearr_path.stat().st_size > 0:
                lines = read_cloudy_linearr(linearr_path)
            results.append(CloudyRunResult(
                this_id, this_time, hp.nII, hp.r0, hp.r1, hp.U, hp.U0, hp.Omega, continuum, lines))
            if overwrite_row is not None:
                rows_to_delete.add(overwrite_row)

        if results:
            cloudy_group = "cluster_cloudy" if spec_type == "cluster" else "galaxy_cloudy"
            filename = self._file[0]  # the guard at the top of this method guarantees len(self._file) == 1
            if rows_to_delete:
                delete_cloudy_h5_rows(filename, cloudy_group, id_key, sorted(rows_to_delete))
            write_cloudy_h5_results(filename, cloudy_group, id_key, results)
            # Rebuild the group-name set (not just reset cached values):
            # this call may have just created cluster_cloudy/
            # galaxy_cloudy for the first time, and those wouldn't be
            # discoverable via their own properties otherwise, since
            # self._groups's key set was previously fixed at __init__.
            # No need to close/reopen anything first, the way this used
            # to -- self._file is just a path, never an open handle
            # held between reads (see __init__'s own comment), so
            # write_cloudy_h5_results()/delete_cloudy_h5_rows() opening
            # and closing filename themselves for writing, above, never
            # conflicted with a handle of this reader's own to begin
            # with.
            with h5py.File(filename, "r") as f:
                self._groups = {name: None for name in f if isinstance(f[name], h5py.Group)}

        if written and not save_temp:
            shutil.rmtree(temp_dir_path, ignore_errors=True)

        # Merge the skipped-duplicate successes (True, no cloudy run)
        # back in among the ones actually run, preserving indices' own
        # order
        ran_iter = iter(ran_successes)
        successes = [flag if flag is not None else next(ran_iter) for flag in skipped]

        return successes[0] if len(successes) == 1 else successes
