"""
slug_group_reader.py

Implements slug_group_reader, a lazy reader for a single group within
a slug HDF5 output file.
"""

import h5py
from astropy import units as u


class slug_group_reader:
    """
    A lazy reader for one group within a slug HDF5 output file.
    Dataset names are read on construction; each dataset itself is
    only read from disk the first time it is requested via
    __getitem__, and attached to its "units" attribute as an astropy
    Quantity -- unless that attribute is an empty string, in which
    case the dataset has no physical unit (e.g. it's a dimensionless
    count, or non-numeric data like a serialized RNG state) and is
    returned as a plain numpy array instead.

    Parameters
    ----------
    file : h5py.File
        The open slug HDF5 output file containing this group.
    group_name : str
        Name of the group within file to read.
    """

    def __init__(self, file, group_name):
        self._file = file
        self._group_name = group_name

        group = self._file[self._group_name]
        self._datasets = {name: None for name in group
            if isinstance(group[name], h5py.Dataset)}

    def __getitem__(self, key):
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
            first access: a Quantity if it has a physical unit, or a
            plain array if its "units" attribute is empty.

        Raises
        ------
        KeyError
            If key is not one of this group's dataset names.
        """
        if key not in self._datasets:
            raise KeyError(key)
        if self._datasets[key] is not None:
            return self._datasets[key]

        dset = self._file[self._group_name][key]
        arr = dset[()]
        unit = dset.attrs["units"]
        self._datasets[key] = arr if unit == "" else arr * u.Unit(unit)
        return self._datasets[key]
