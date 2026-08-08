"""
slug_phot_reader.py

Implements slug_phot_reader, a lazy reader for the photometry
(cluster_phot) group of a slug HDF5 output file.
"""

from astropy import units as u

from .slug_group_reader import slug_group_reader


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

    Attributes
    ----------
    filters : list of str
        Names of the filters this group has photometry for, in the
        order used to index the phot/phot_extinct datasets
        (read-only).
    filter_units : list of str
        Units of each filter's photometry, in the same order as
        filters (read-only).
    """

    def __init__(self, file, group_name):
        super().__init__(file, group_name)

        group = self._file[self._group_name]
        self._filters = list(group.attrs["filters"])
        self._filter_units = list(group["phot"].attrs["units"])

        self._phot = {name: None for name in self._filters}
        self._phot_extinct = None
        if "phot_extinct" in self._datasets:
            self._phot_extinct = {name: None for name in self._filters}

    @property
    def filters(self):
        """
        list of str : names of the filters this group has photometry
        for, in the order used to index the phot/phot_extinct
        datasets (read-only).
        """
        return self._filters

    @property
    def filter_units(self):
        """
        list of str : units of each filter's photometry, in the same
        order as filters (read-only).
        """
        return self._filter_units

    def __getitem__(self, key):
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
        if phot[name] is not None:
            return phot[name]

        index = self._filters.index(name)
        dataset_name = "phot_extinct" if extinct else "phot"
        arr = self._file[self._group_name][dataset_name][:, index]
        unit = self._filter_units[index]
        phot[name] = arr if unit == "" else arr * u.Unit(unit)
        return phot[name]
