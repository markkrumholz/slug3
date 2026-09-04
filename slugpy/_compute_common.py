"""
_compute_common.py

Shared helpers between compute_isochrones and compute_tracks: both
resolve a SimControls and an optional FilterCollection the same way,
and run the same "evolutionary state -> spectrum -> photometry" step
for each star. Not part of slugpy's own public API.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import numpy as np
from astropy import units as u

from ._slug import FilterCollection, PhotSystem, SimControls, Specsyn

# Names and units of the quantities Interpolator1D.__call__() returns
# for a track/isochrone segment, in the order tracks::FieldIdx/
# fieldStr list them in src/tracks/TrackCommons.hpp. That array isn't
# exposed to Python, so this has to be kept in sync with it by hand.
TRACK_FIELDS: tuple[tuple[str, u.UnitBase], ...] = (
    ("mass", u.Msun),
    ("mdot", u.Msun / u.yr),
    ("log_L", u.dex(u.Lsun)),
    ("log_Teff", u.dex(u.K)),
    ("h_surf", u.dimensionless_unscaled),
    ("he_surf", u.dimensionless_unscaled),
    ("c_surf", u.dimensionless_unscaled),
    ("n_surf", u.dimensionless_unscaled),
    ("o_surf", u.dimensionless_unscaled),
)

# The PhotSystem compute_isochrones/compute_tracks build a
# FilterCollection with when filters is given as name(s) rather than
# an existing FilterCollection and phot_system is left at its own
# default of None.
DEFAULT_PHOT_SYSTEM = PhotSystem.Flambda


def resolve_controls(
    simcontrols: SimControls | None, simcontrols_kwargs: dict[str, object],
) -> SimControls:
    """simcontrols itself, or a SimControls built from simcontrols_kwargs if it's None."""
    return simcontrols if simcontrols is not None else SimControls(**simcontrols_kwargs)


def resolve_filter_collection(
    filters: str | list[str] | FilterCollection | None,
    phot_system: PhotSystem | None,
) -> FilterCollection | None:
    """filters itself if it's None or already a FilterCollection; built from name(s) otherwise."""
    if filters is None:
        return None
    if isinstance(filters, FilterCollection):
        return filters
    names = [filters] if isinstance(filters, str) else list(filters)
    ps = DEFAULT_PHOT_SYSTEM if phot_system is None else phot_system
    return FilterCollection(names, ps)


def resolve_fixed_feh(controls: SimControls) -> float | None:
    """simcontrols.feH's own single value if it's a delta distribution, else None."""
    feh_dist = controls.feH
    return feh_dist.getMin() if feh_dist.getMin() == feh_dist.getMax() else None


def empty_data(
    shape: tuple[int, int], filter_names: list[str], filter_units: list[str],
) -> dict[str, u.Quantity]:
    """A NaN-filled, unit-tagged dict of the given shape: one key per track field, plus one per filter."""
    data: dict[str, u.Quantity] = {
        name: np.full(shape, np.nan) * unit for name, unit in TRACK_FIELDS}
    for name, unit_str in zip(filter_names, filter_units, strict=True):
        data[name] = np.full(shape, np.nan) * (
            u.dimensionless_unscaled if unit_str == "" else u.Unit(unit_str))
    return data


def fill_star(
    data: dict[str, u.Quantity], row: int, col: int, props: list[float],
    filter_collection: FilterCollection | None,
    filter_names: list[str], filter_units: list[str],
    specsyn: Specsyn, wl: list[float], feh: float,
) -> None:
    """Fill data[*][row, col] from props (and, if filter_collection is given, that spectrum's photometry)."""
    for (name, unit), value in zip(TRACK_FIELDS, props, strict=True):
        data[name][row, col] = value * unit

    if filter_collection is None:
        return
    spec = specsyn.spec(props, feh)
    phot = filter_collection.phot(wl, spec)
    for name, unit_str, value in zip(filter_names, filter_units, phot, strict=True):
        unit = u.dimensionless_unscaled if unit_str == "" else u.Unit(unit_str)
        data[name][row, col] = value * unit
