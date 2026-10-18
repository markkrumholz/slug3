"""
Unit tests for the physical units slug attaches to its HDF5 datasets:
every "units" attribute in the committed example outputs and in
data/spectra/vega.h5 must parse in astropy without even a warning (so
readers can attach it as an astropy unit), and the legacy "magnitudes"
written for A_V/A_V_neb through 3.0.1 must still read, as "mag".

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import pathlib
import shutil
import warnings
from typing import cast

import h5py
import numpy as np
import pytest
from astropy import units as u

from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKED_FILES = sorted((REPO_ROOT / "examples").glob("*/*.h5")) + [REPO_ROOT / "data" / "spectra" / "vega.h5"]
CLUSTERLIB = REPO_ROOT / "examples" / "clusterlib" / "clusterlib.h5"


def _units_attrs(path: pathlib.Path) -> list[tuple[str, str]]:
    """
    Every units string attached to a dataset in an HDF5 file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    list of (str, str)
        (dataset name, units string) pairs; a dataset with one units
        string per column contributes one pair per column.
    """
    found: list[tuple[str, str]] = []

    def visit(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        dset = cast(h5py.Dataset, obj)
        if "units" not in dset.attrs:
            return
        value = dset.attrs["units"]
        values = [value] if isinstance(value, (str, bytes)) else list(np.atleast_1d(value))
        for v in values:
            found.append((name, v.decode() if isinstance(v, bytes) else str(v)))

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found


@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: p.parent.name + "/" + p.name)
def test_units_parse_cleanly(path: pathlib.Path) -> None:
    """Every units string parses in astropy, with warnings (e.g. multiple slashes) treated as errors."""
    attrs = _units_attrs(path)
    assert attrs
    for name, unit in attrs:
        if unit == "":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                u.Unit(unit)
            except (ValueError, u.UnitsWarning) as err:
                pytest.fail(f"{path.name}:{name}: units {unit!r} do not parse cleanly: {err}")


def test_av_reads_in_magnitudes() -> None:
    """A_V reads as an astropy Quantity in magnitudes."""
    clusters = slug_reader(str(CLUSTERLIB)).clusters
    assert clusters is not None
    av = clusters["A_V"]
    assert isinstance(av, u.Quantity)
    assert cast(u.Quantity, av).unit == u.mag


def test_legacy_magnitudes_units(tmp_path: pathlib.Path) -> None:
    """Files written through 3.0.1, with A_V in "magnitudes", still read, as mag."""
    path = tmp_path / "legacy.h5"
    shutil.copy(CLUSTERLIB, path)
    with h5py.File(path, "r+") as f:
        dset = f["clusters"]["A_V"]
        assert isinstance(dset, h5py.Dataset)
        dset.attrs.modify("units", "magnitudes")
    legacy = slug_reader(str(path)).clusters
    current = slug_reader(str(CLUSTERLIB)).clusters
    assert legacy is not None and current is not None
    av = legacy["A_V"]
    assert isinstance(av, u.Quantity)
    assert cast(u.Quantity, av).unit == u.mag
    assert np.array_equal(av, current["A_V"])
