"""
slug_spectra_reader.py

Implements slug_spectra_reader, a lazy reader for the cluster_spectra/
galaxy_spectra group of a slug HDF5 output file.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from typing import Any, cast

import h5py
import numpy as np
from astropy import units as u

from .slug_group_reader import Dataset, slug_group_reader


class slug_spectra_reader(slug_group_reader):
    """
    A lazy reader for the cluster_spectra/galaxy_spectra group of a
    slug HDF5 output file. Extends slug_group_reader with two
    differences. First, the "line_label" dataset (present only if this
    file's own simulation requested nebular emission) is decoded from
    raw HDF5 bytes into plain Python str, both via the line_labels
    property and __getitem__("line_label"), rather than the raw
    byte-string numpy array slug_group_reader's own generic
    __getitem__ would otherwise return. Second, indexing by a (label,
    wl) pair (e.g. spectra["HI", 6563.0]) returns that one specific
    nebular emission line's own luminosity from neb_lines, read on
    first access and cached thereafter, exactly mirroring
    slug_phot_reader's own per-filter access; append "_ex" to label
    (e.g. spectra["HI_ex", 6563.0]) for the same line's extincted
    luminosity, from neb_lines_extinct, instead -- see line_index for
    why label and wl together, not label alone, identify one specific
    line. Every plain string key (wl, spec, spec_neb, spec_extinct,
    spec_neb_extinct, line_wl, neb_lines, neb_lines_extinct, trial,
    time, uid, ...) falls through to slug_group_reader's own
    __getitem__ unchanged, letting a caller read every line's
    luminosity at once (e.g. spectra["neb_lines"]) just as easily as
    one line at a time.

    Parameters
    ----------
    file : h5py.File
        The open slug HDF5 output file containing this group.
    group_name : str
        Name of the group within file to read.

    Attributes
    ----------
    line_labels : list of str or None
        Label of each of this group's own nebular emission lines (see
        neb_lines/neb_lines_extinct), in the same order as line_wl's
        own entries and neb_lines/neb_lines_extinct's own second axis,
        decoded to plain Python str. None if this file's own
        simulation did not request nebular emission (no line_label
        dataset at all) (read-only). A label identifies the species
        producing a line (e.g. "HI"), not one specific line -- a
        single species produces many lines at different wavelengths,
        so several entries can share one label; the (label, wl) pair
        is unique, which is exactly what line_index(), line_luminosity(),
        and __getitem__'s own (label, wl)-tuple indexing key on.
    """

    def __init__(self, file: h5py.File, group_name: str) -> None:
        super().__init__(file, group_name)

        self._line_labels: list[str] | None = None
        if "line_label" in self._datasets:
            raw = self._file[self._group_name]["line_label"][()]
            self._line_labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in raw]

        # Per-line luminosity caches for __getitem__'s own (label, wl)
        # indexing, keyed by line_index()'s own integer result rather
        # than the raw (label, wl) key itself -- mirrors
        # slug_phot_reader's own _phot/_phot_extinct caching, but keyed
        # by index rather than name since, unlike a filter name, a
        # (label, wl) pair can be spelled more than one way (a bare
        # float vs. an equivalent Quantity in a different unit) and
        # should still hit the same cache entry
        self._line_lum: dict[int, Dataset] | None = None
        self._line_lum_extinct: dict[int, Dataset] | None = None
        if self._line_labels is not None:
            self._line_lum = {}
            if "neb_lines_extinct" in self._datasets:
                self._line_lum_extinct = {}

    @property
    def line_labels(self) -> list[str] | None:
        """
        list of str or None : label of each of this group's own
        nebular emission lines, decoded to plain Python str, or None
        if this file's own simulation did not request nebular emission
        (read-only).
        """
        return self._line_labels

    @line_labels.setter
    def line_labels(self, value: Any) -> None:
        raise AttributeError("line_labels is read-only")

    def __getitem__(self, key: str | tuple[str, float | u.Quantity]) -> Dataset:
        """
        Return one dataset from this group, one specific nebular
        emission line's own luminosity, or the decoded "line_label"
        array.

        Parameters
        ----------
        key : str, or tuple of (str, float or astropy.units.Quantity)
            A (label, wl) pair (e.g. ("HI", 6563.0), or equivalently
            spectra["HI", 6563.0] via Python's own multi-subscript
            sugar) identifying one specific nebular emission line (see
            line_index) -- label with "_ex" appended (e.g.
            ("HI_ex", 6563.0)) selects that same line's extincted
            luminosity instead. Otherwise, a dataset name: "line_label"
            (decoded, see line_labels), or any other name from the
            underlying group (e.g. "wl", "spec", "neb_lines", "trial").

        Returns
        -------
        astropy.units.Quantity or numpy.ndarray
            A (label, wl) key returns that line's own luminosity
            across every stored trial/time, from neb_lines (or
            neb_lines_extinct, for a label ending "_ex"), read on
            first access and cached thereafter -- see
            line_luminosity's own identical return value. "line_label"
            returns a numpy array of the same decoded strings as
            line_labels. Every other string key falls through to
            slug_group_reader's own __getitem__.

        Raises
        ------
        KeyError
            For a (label, wl) key: see line_index's own docstring. For
            a string key: if key is not one of this group's dataset
            names.
        """
        if isinstance(key, tuple):
            label, wl = key
            extinct = label.endswith("_ex")
            base_label = label[:-len("_ex")] if extinct else label
            index = self.line_index(base_label, wl)

            cache = self._line_lum_extinct if extinct else self._line_lum
            if cache is None:
                raise KeyError((label, wl))
            if index in cache:
                return cache[index]

            dataset_name = "neb_lines_extinct" if extinct else "neb_lines"
            result = cast(u.Quantity, super().__getitem__(dataset_name))[:, index]
            cache[index] = result
            return result

        if key == "line_label" and self._line_labels is not None:
            return np.array(self._line_labels, dtype=object)
        return super().__getitem__(key)

    def line_index(self, label: str, wl: float | u.Quantity) -> int:
        """
        Find the position of one specific nebular emission line.

        Parameters
        ----------
        label : str
            Species label producing this line (see line_labels) --
            not unique on its own, since one species produces many
            lines at different wavelengths.
        wl : float or astropy.units.Quantity
            Wavelength of this line, in Angstrom if a bare float,
            matched against line_wl to within a relative tolerance of
            1e-9 (see this class's own line_labels docstring for why
            label and wl together, not label alone, identify one
            specific line; the tolerance accommodates float rounding
            from converting a Quantity given in a different unit, not
            genuine ambiguity between two close-but-distinct lines).

        Returns
        -------
        int
            Index into line_wl/line_labels, and the second axis of
            neb_lines/neb_lines_extinct, for the line matching both
            label and wl.

        Raises
        ------
        KeyError
            If this group has no nebular line data at all (line_labels
            is None), or no line matches both label and wl.
        """
        if self._line_labels is None:
            raise KeyError((label, wl))

        wl_aa = cast(u.Quantity, wl).to_value(u.AA) if isinstance(wl, u.Quantity) else wl
        line_wl = cast(u.Quantity, self["line_wl"]).to_value(u.AA)
        for index, (this_label, this_wl) in enumerate(zip(self._line_labels, line_wl, strict=True)):
            if this_label == label and np.isclose(this_wl, wl_aa, rtol=1e-9, atol=0.0):
                return index
        raise KeyError((label, wl))

    def line_luminosity(self, label: str, wl: float | u.Quantity, extinct: bool = False) -> Dataset:
        """
        Get one nebular emission line's own luminosity, across every stored trial/time.

        A named, more discoverable alternative to indexing directly
        with a (label, wl) tuple (see __getitem__) -- both call the
        same underlying lookup, so e.g.
        spectra.line_luminosity("HI", 6563.0) and
        spectra["HI", 6563.0] are exactly equivalent.

        Parameters
        ----------
        label : str
            Species label producing this line (see line_index).
        wl : float or astropy.units.Quantity
            Wavelength of this line, in Angstrom if a bare float (see
            line_index).
        extinct : bool, default False
            If True, return the extincted luminosity from
            neb_lines_extinct instead of neb_lines.

        Returns
        -------
        astropy.units.Quantity
            neb_lines[:, i] (or neb_lines_extinct[:, i] if extinct is
            True), where i is line_index(label, wl).

        Raises
        ------
        KeyError
            See line_index's own docstring. Also raised if extinct is
            True but this group has no neb_lines_extinct dataset (no
            extinction curve was requested for this simulation).
        """
        return self[(label + "_ex", wl) if extinct else (label, wl)]
