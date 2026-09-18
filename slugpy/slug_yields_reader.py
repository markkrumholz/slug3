"""
slug_yields_reader.py

Implements slug_yields_reader, a lazy reader for the nucleosynthetic
yield (cluster_yields/galaxy_yields) group of a slug HDF5 output file.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from collections.abc import Sequence
from typing import Any, cast

import h5py
import numpy as np
from astropy import units as u

from .slug_group_reader import Dataset, slug_group_reader

#: The key type __getitem__ accepts: either a bare isotope name (e.g.
#: "Fe56"), or, when decomposed is True, a (channel, model, isotope)
#: triple of strings, each of channel/model optionally blank ("") to
#: match every channel/model -- see __getitem__'s own docstring.
YieldsKey = str | Sequence[str]


class slug_yields_reader(slug_group_reader):
    """
    A lazy reader for the cluster_yields/galaxy_yields group of a slug
    HDF5 output file. Extends slug_group_reader with per-isotope
    access: indexing by an isotope name (e.g. "Fe56", matched case-
    insensitively and with or without a space before the mass number
    -- see __getitem__'s own docstring) returns that isotope's yield,
    read from the yields dataset's own column(s) for that isotope on
    first access and cached thereafter. If this group's own decomposed
    attribute is True, indexing can instead take a (channel, model,
    isotope) triple, letting a caller select just the channel(s)
    and/or model(s) it wants rather than every channel the isotope was
    tabulated for.

    Parameters
    ----------
    file_paths : list of str
        Path(s) to the open slug HDF5 output file(s) containing this
        group -- see slug_group_reader's own docstring.
    group_name : str
        Name of the group within each file to read.
    registry_name : str, optional
        Accepted for signature parity with slug_phot_reader; currently
        unused, since nothing here resolves a name against a yield
        registry the way slug_phot_reader.get_filter() does against a
        filter registry.

    Attributes
    ----------
    isotopes : list of str
        Names of the isotopes this group has yields for (e.g. "H1",
        "Fe56"), in the order used to index the yields dataset's own
        isotope axis (read-only).
    channels : list of str
        Which nucleosynthetic channel (e.g. "ccsn") each entry of
        models below came from, in the same order as the yields
        dataset's own channel axis -- meaningful only if decomposed is
        True (read-only).
    models : list of str
        Which yield model (e.g. "sukhbold16") was used for each
        channel entry above, same order (read-only).
    decomposed : bool
        Whether the yields dataset's own columns are broken out one
        (channel, isotope) pair at a time (True) or summed over every
        channel, one column per isotope (False) -- see __getitem__'s
        own docstring for how this changes what a key means
        (read-only).
    """

    def __init__(self, file_paths: list[str], group_name: str,
        registry_name: str | None = None) -> None:
        super().__init__(file_paths, group_name)

        with h5py.File(file_paths[0], "r") as f:
            group = f[self._group_name]
            self._isotopes: list[str] = list(group.attrs["isotopes"])
            self._channels: list[str] = list(group.attrs["channels"])
            self._models: list[str] = list(group.attrs["models"])
            self._decomposed: bool = bool(group.attrs["decomposed"])

        self._registry_name = registry_name
        self._yields: dict[str, u.Quantity] = {}

    @property
    def isotopes(self) -> list[str]:
        """
        list of str : names of the isotopes this group has yields for,
        in the order used to index the yields dataset's own isotope
        axis (read-only).
        """
        return self._isotopes

    @isotopes.setter
    def isotopes(self, value: Any) -> None:
        raise AttributeError("isotopes is read-only")

    @property
    def channels(self) -> list[str]:
        """
        list of str : which nucleosynthetic channel each entry of
        models came from, same order (read-only).
        """
        return self._channels

    @channels.setter
    def channels(self, value: Any) -> None:
        raise AttributeError("channels is read-only")

    @property
    def models(self) -> list[str]:
        """
        list of str : which yield model was used for each channel
        entry, same order as channels (read-only).
        """
        return self._models

    @models.setter
    def models(self, value: Any) -> None:
        raise AttributeError("models is read-only")

    @property
    def decomposed(self) -> bool:
        """
        bool : whether the yields dataset's own columns are broken out
        one (channel, isotope) pair at a time (True) or summed over
        every channel (False) -- see __getitem__'s own docstring
        (read-only).
        """
        return self._decomposed

    @decomposed.setter
    def decomposed(self, value: Any) -> None:
        raise AttributeError("decomposed is read-only")

    def _resolve_isotope(self, name: str) -> str:
        """
        Resolve an isotope name to this group's own canonical spelling.

        Parameters
        ----------
        name : str
            An isotope name, e.g. "Fe56", "fe56", "Fe 56", or "fe 56"
            -- matched against isotopes case-insensitively, and with or
            without a space between the element symbol and the mass
            number.

        Returns
        -------
        str
            The matching entry of isotopes, in its own canonical
            spelling (e.g. "Fe56") -- used as this isotope's own key
            into _yields, so every spelling of the same isotope shares
            one cache entry.

        Raises
        ------
        KeyError
            If name does not match any entry of isotopes.
        """
        normalized = name.replace(" ", "").upper()
        for isotope in self._isotopes:
            if isotope.upper() == normalized:
                return isotope
        raise KeyError(name)

    def _load_isotope(self, isotope: str) -> u.Quantity:
        """
        Load one isotope's own yield column(s) from disk, cache it in
        _yields, and return it.

        Parameters
        ----------
        isotope : str
            One of this group's own canonical isotopes entries (see
            _resolve_isotope) -- not re-validated here.

        Returns
        -------
        astropy.units.Quantity
            _yields[isotope]: a 1D array (one entry per trial/time) if
            decomposed is False, or a 2D array (one row per trial/
            time, one column per channel, in the same order as
            channels/models) if decomposed is True -- read from disk
            (and cached) on first access.
        """
        cached = self._yields.get(isotope)
        if cached is not None:
            return cached

        n_iso = len(self._isotopes)
        iso_index = self._isotopes.index(isotope)
        if self._decomposed:
            col_indices = [iso_index + c * n_iso for c in range(len(self._channels))]
        else:
            col_indices = [iso_index]

        paths = self._file_paths if self._extensible["yields"] else self._file_paths[:1]
        arrs = []
        for path in paths:
            with h5py.File(path, "r") as f:
                arrs.append(f[self._group_name]["yields"][:, col_indices])
        arr = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=0)
        if not self._decomposed:
            arr = arr[:, 0]

        result = arr * u.Msun
        self._yields[isotope] = result
        return result

    def __getitem__(self, key: YieldsKey) -> Dataset:
        """
        Return one isotope's yield, selecting a subset of channels/
        models if decomposed and key names them.

        Parameters
        ----------
        key : str or sequence of 3 str
            If decomposed is False, must be a bare isotope name (see
            _resolve_isotope for the accepted spellings); the returned
            Quantity is 1D, one entry per trial/time, summed over
            every channel.

            If decomposed is True, key can either be:

            - A bare isotope name, in which case the full 2D array
              (one row per trial/time, one column per channel, in the
              same order as channels/models) is returned.
            - A (channel, model, isotope) triple of strings, in which
              case channel/model are matched against this group's own
              channels/models -- an empty string ("") for either
              matches every channel/model -- and only the matching
              columns are returned: a 1D array if exactly one channel
              matches, otherwise a 2D array (one row per trial/time,
              one column per match, in channels/models order).

        Returns
        -------
        astropy.units.Quantity
            The requested yield, in Msun, read from disk (and cached,
            keyed by isotope) on first access.

        Raises
        ------
        TypeError
            If decomposed is False and key is not a string, or if key
            is neither a string nor a sequence of exactly 3 strings.
        KeyError
            If key (or key[2], for a triple) does not match any entry
            of isotopes (see _resolve_isotope); or key is a triple
            whose (channel, model) does not match any entry of
            channels/models.
        """
        if isinstance(key, str):
            isotope = self._resolve_isotope(key)
            return self._load_isotope(isotope)

        if not self._decomposed:
            raise TypeError(
                "slug_yields_reader.__getitem__: key must be a bare isotope "
                "name when decomposed is False")

        if (not isinstance(key, Sequence) or len(key) != 3 or
            not all(isinstance(k, str) for k in key)):
            raise TypeError(
                "slug_yields_reader.__getitem__: key must be a string or a "
                "sequence of 3 strings (channel, model, isotope)")
        channel_query, model_query, isotope_query = key

        isotope = self._resolve_isotope(isotope_query)
        full = cast(u.Quantity, self._load_isotope(isotope))

        matched = [j for j, (channel, model) in
            enumerate(zip(self._channels, self._models, strict=True))
            if (channel_query == "" or channel == channel_query) and
                (model_query == "" or model == model_query)]
        if not matched:
            raise KeyError((channel_query, model_query, isotope_query))
        if len(matched) == 1:
            return full[:, matched[0]]
        return full[:, matched]
