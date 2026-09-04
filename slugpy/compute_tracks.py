"""
compute_tracks.py

Implements compute_tracks, an end-to-end convenience function that
loads a set of stellar tracks, evaluates the track for each of a set
of stellar masses, passes the resulting stars through spectral
synthesis, and (optionally) passes the spectra through a set of
filters to get photometry.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import numpy as np
from astropy import units as u

from ._compute_common import (
    empty_data,
    fill_star,
    resolve_controls,
    resolve_filter_collection,
    resolve_fixed_feh,
)
from ._slug import FilterCollection, PhotSystem, SimControls


def compute_tracks(
    masses: list[float] | np.ndarray | u.Quantity,
    time: float | list[float] | np.ndarray | u.Quantity | None = None,
    simcontrols: SimControls | None = None,
    filters: str | list[str] | FilterCollection | None = None,
    ntimes: int = 100,
    phot_system: PhotSystem | None = None,
    feh: float = 0.0,
    **simcontrols_kwargs: object,
) -> tuple[u.Quantity, dict[str, u.Quantity]]:
    """
    Compute stellar evolution tracks end to end: tracks -> spectra ->
    photometry.

    Parameters
    ----------
    masses : array_like of float or astropy.units.Quantity
        Stellar masses to evaluate a track for. An array_like is
        interpreted as being in Msun; a Quantity array may carry any
        unit of mass.
    time : float, array_like of float, or astropy.units.Quantity, optional
        Time(s) at which to evaluate each track. A plain float or
        array_like is interpreted as being in years; an astropy
        Quantity (scalar or array) may carry any unit of time. If None
        (the default), set to ntimes values, log-spaced from 1e5 yr to
        the maximum age spanned by simcontrols.tracks (its own
        logTMax()).
    simcontrols : SimControls, optional
        Simulation controls providing the [Fe/H] distribution, tracks,
        and spectral synthesizer to use. If None (the default), one is
        constructed from simcontrols_kwargs (or default-constructed,
        if none are given).
    filters : str, list of str, or FilterCollection, optional
        The filter(s) to compute photometry with. A str or list of str
        is used to construct a FilterCollection (see phot_system
        below); a FilterCollection is used directly. If None (the
        default), no photometry is computed.
    ntimes : int, default 100
        Number of times to use, log-spaced between 1e5 yr and
        simcontrols.tracks's own maximum age. Ignored if time is given.
    phot_system : PhotSystem, optional
        The photometric system to pass to the FilterCollection
        constructor when filters is a str or list of str; ignored
        otherwise. Defaults to PhotSystem.Flambda.
    feh : float, default 0.0
        [Fe/H] to compute the track at. Used only if simcontrols.feH is
        not a delta distribution (i.e. its getMin() and getMax()
        differ); if it is a delta distribution, that single value is
        used instead and feh is ignored.
    **simcontrols_kwargs
        Additional keywords, passed on to the SimControls constructor.
        Ignored if simcontrols is given.

    Returns
    -------
    astropy.units.Quantity
        The times each track was evaluated at: time itself (converted
        to yr), or, if time was not given, the ntimes log-spaced values
        this function constructed instead. A 1D Quantity array of
        length ntimes.
    dict of str to astropy.units.Quantity
        One key per stellar property returned by the track ("mass",
        "mdot", "log_L", "log_Teff", "h_surf", "he_surf", "c_surf",
        "n_surf", "o_surf"), plus one key per filter name (if filters
        is not None). Each value is a 2D Quantity array of shape
        (nmass, ntimes), indexed the same way as masses and the first
        return value. Entries for a mass outside simcontrols.tracks's
        own mass range, or a time outside a given mass's own track
        range, are NaN.

    Raises
    ------
    RuntimeError
        If simcontrols.specsyn is None (i.e. no spectral synthesizer
        was configured).
    """
    controls = resolve_controls(simcontrols, simcontrols_kwargs)
    tracks = controls.tracks

    mass_val = u.Quantity(masses, u.Msun).to_value(u.Msun)
    nmass = len(mass_val)

    if time is None:
        time_yr = np.geomspace(1e5, 10 ** tracks.logTMax(), ntimes)
    else:
        time_yr = np.atleast_1d(u.Quantity(time, u.yr).to_value(u.yr))
    ntimes = len(time_yr)
    log_time = np.log10(time_yr)

    filter_collection = resolve_filter_collection(filters, phot_system)

    specsyn = controls.specsyn
    if specsyn is None:
        raise RuntimeError(
            "compute_tracks: simcontrols.specsyn is None -- set "
            "spectra.model in the input deck, or pass specsyn=... "
            "through to SimControls, before computing spectra")
    wl = specsyn.wl()

    fixed_feh = resolve_fixed_feh(controls)
    this_feh = fixed_feh if fixed_feh is not None else feh

    filter_names = filter_collection.filterNames() if filter_collection is not None else []
    filter_units = filter_collection.filterUnits() if filter_collection is not None else []

    data = empty_data((nmass, ntimes), filter_names, filter_units)

    for i, m in enumerate(mass_val):
        if m < tracks.mMin() or m > tracks.mMax():
            continue
        track = tracks.getTrack(m, this_feh)
        for j, logt in enumerate(log_time):
            if logt < track.xMin() or logt > track.xMax():
                continue
            fill_star(
                data, i, j, track(logt), filter_collection, filter_names, filter_units,
                specsyn, wl, this_feh)

    return time_yr * u.yr, data
