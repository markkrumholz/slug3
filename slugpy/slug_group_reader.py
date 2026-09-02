"""
slug_group_reader.py

Implements slug_group_reader, a lazy reader for a single group spread
across one or more slug HDF5 output files.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from collections.abc import KeysView

import h5py
import numpy as np
from astropy import units as u

Dataset = np.ndarray | u.Quantity


class slug_group_reader:
    """
    A lazy reader for one group, spread across one or more slug HDF5
    output files (see slug_reader's own docstring for when a group
    reader spans more than one file: checkpointed and/or h5divided
    output). Dataset names are read from the first file on
    construction; each dataset itself is only read from disk the first
    time it is requested via __getitem__, and attached to its "units"
    attribute as an astropy Quantity -- unless that attribute is an
    empty string (the dataset has no physical unit, e.g. it's a
    dimensionless count, or non-numeric data like a serialized RNG
    state) or is not a single string at all (e.g. cluster_phot's
    phot/phot_extinct datasets, whose "units" attribute is one string
    per column, since each filter can have its own unit), in which
    case it is returned as a plain numpy array instead.

    Every file is opened only for the duration of each individual read
    (never held open between calls), so this never holds more than one
    file open at a time regardless of how many files it spans -- see
    slug_reader's own docstring for why.

    A dataset that spans more than one file is read from every file
    and concatenated along its own first axis (one row per trial/time,
    so this is what "every trial from every file, pooled together"
    means) -- but only if the dataset is *extensible* (an unlimited
    maximum extent on some axis; see isExtensibleDataset() on the C++
    side, which this mirrors exactly): a handful of datasets (e.g.
    "wl", "line_wl", "line_label") are instead fixed, shared metadata,
    physically identical in every file a given run produced, and so
    are read from the first file only, not concatenated -- reading a
    non-extensible dataset from every file and pasting the (identical)
    copies together would silently produce nonsense (e.g. an N-times
    duplicated wavelength grid, for N constituent files).

    Parameters
    ----------
    file_paths : list of str
        Path(s) to the open slug HDF5 output file(s) containing this
        group; every one is assumed to share the identical group/
        dataset layout (each was built by the same simulation's own
        openOutputFile() call, once per checkpoint/thread -- see
        OutputManagerH5's own header comment).
    group_name : str
        Name of the group within each file to read.
    """

    def __init__(self, file_paths: list[str], group_name: str) -> None:
        self._file_paths: list[str] = file_paths
        self._group_name: str = group_name

        self._datasets: dict[str, Dataset | None] = {}
        self._extensible: dict[str, bool] = {}
        with h5py.File(file_paths[0], "r") as f:
            group = f[group_name]
            for name in group:
                if not isinstance(group[name], h5py.Dataset):
                    continue
                self._datasets[name] = None
                self._extensible[name] = any(
                    dim is None for dim in group[name].maxshape)

    def __getitem__(self, key: str) -> Dataset:
        """
        Return one dataset from this group, as an astropy Quantity.

        Parameters
        ----------
        key : str
            Name of the dataset to return.

        Returns
        -------
        astropy.units.Quantity or numpy.ndarray
            The requested dataset, read from disk (and cached) on
            first access: a Quantity if it has a single physical unit,
            or a plain array if its "units" attribute is absent, empty,
            or not a single string. Pooled across every one of this
            reader's own files if the dataset is extensible (see this
            class's own docstring); otherwise read from the first file
            alone.

        Raises
        ------
        KeyError
            If key is not one of this group's dataset names.
        """
        if key not in self._datasets:
            raise KeyError(key)
        cached = self._datasets[key]
        if cached is not None:
            return cached

        paths = self._file_paths if self._extensible[key] else self._file_paths[:1]
        arrs = []
        unit: str | list[str] | None = None
        for path in paths:
            with h5py.File(path, "r") as f:
                dset = f[self._group_name][key]
                arrs.append(dset[()])
                if unit is None:
                    unit = dset.attrs.get("units")
        arr = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=0)

        has_unit = isinstance(unit, str) and unit != ""
        result: Dataset = arr * u.Unit(unit) if has_unit else arr
        self._datasets[key] = result
        return result

    def keys(self) -> KeysView[str]:
        """
        Return the names of the datasets available in this group.

        Returns
        -------
        KeysView of str
            Names of every dataset in this group, whether or not it
            has been read from disk yet.
        """
        return self._datasets.keys()
