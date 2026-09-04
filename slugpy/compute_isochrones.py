"""
compute_isochrones.py

Implements compute_isochrones, an end-to-end convenience function that
loads a set of stellar tracks, builds isochrones from them, passes the
resulting stars through spectral synthesis, and (optionally) passes
the spectra through a set of filters to get photometry.

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


def compute_isochrones(
    time: float | list[float] | np.ndarray | u.Quantity,
    simcontrols: SimControls | None = None,
    filters: str | list[str] | FilterCollection | None = None,
    masses: list[float] | np.ndarray | u.Quantity | None = None,
    nmass: int = 100,
    phot_system: PhotSystem | None = None,
    feh: float = 0.0,
    **simcontrols_kwargs: object,
) -> tuple[u.Quantity, dict[str, u.Quantity]]:
    """
    Compute isochrones end to end: tracks -> isochrone -> spectra ->
    photometry.

    Parameters
    ----------
    time : float, array_like of float, or astropy.units.Quantity
        Time(s) at which to compute isochrones. A plain float or
        array_like is interpreted as being in years; an astropy
        Quantity (scalar or array) may carry any unit of time.
    simcontrols : SimControls, optional
        Simulation controls providing the IMF, [Fe/H] distribution,
        tracks, and spectral synthesizer to use. If None (the
        default), one is constructed from simcontrols_kwargs (or
        default-constructed, if none are given).
    filters : str, list of str, or FilterCollection, optional
        The filter(s) to compute photometry with. A str or list of str
        is used to construct a FilterCollection (see phot_system
        below); a FilterCollection is used directly. If None (the
        default), no photometry is computed.
    masses : array_like of float or astropy.units.Quantity, optional
        Stellar masses at which to evaluate each isochrone. An
        array_like is interpreted as being in Msun; a Quantity array
        may carry any unit of mass. If None (the default), set to
        nmass values, log-spaced from the minimum to the maximum mass
        of simcontrols.imf.
    nmass : int, default 100
        Number of stellar masses to use, log-spaced between
        simcontrols.imf's own minimum and maximum mass. Ignored if
        masses is given.
    phot_system : PhotSystem, optional
        The photometric system to pass to the FilterCollection
        constructor when filters is a str or list of str; ignored
        otherwise. Defaults to PhotSystem.Flambda.
    feh : float, default 0.0
        [Fe/H] to compute the isochrone at. Used only if
        simcontrols.feH is not a delta distribution (i.e. its getMin()
        and getMax() differ); if it is a delta distribution, that
        single value is used instead and feh is ignored.
    **simcontrols_kwargs
        Additional keywords, passed on to the SimControls constructor.
        Ignored if simcontrols is given.

    Returns
    -------
    astropy.units.Quantity
        The stellar masses each isochrone was evaluated at: masses
        itself (converted to Msun), or, if masses was not given, the
        nmass log-spaced values this function constructed instead. A
        1D Quantity array of length nmass.
    dict of str to astropy.units.Quantity
        One key per stellar property returned by the isochrone
        ("mass", "mdot", "log_L", "log_Teff", "h_surf", "he_surf",
        "c_surf", "n_surf", "o_surf"), plus one key per filter name (if
        filters is not None). Each value is a 2D Quantity array of
        shape (ntimes, nmass), indexed the same way as time and the
        first return value. Entries for a mass outside the isochrone's
        valid range at a given time are NaN.

    Raises
    ------
    RuntimeError
        If simcontrols.specsyn is None (i.e. no spectral synthesizer
        was configured).
    """
    controls = resolve_controls(simcontrols, simcontrols_kwargs)

    if masses is None:
        imf = controls.imf
        mass_val = np.geomspace(imf.getMin(), imf.getMax(), nmass)
    else:
        mass_val = u.Quantity(masses, u.Msun).to_value(u.Msun)
    nmass = len(mass_val)

    filter_collection = resolve_filter_collection(filters, phot_system)

    time_yr = np.atleast_1d(u.Quantity(time, u.yr).to_value(u.yr))
    ntimes = len(time_yr)

    specsyn = controls.specsyn
    if specsyn is None:
        raise RuntimeError(
            "compute_isochrones: simcontrols.specsyn is None -- set "
            "spectra.model in the input deck, or pass specsyn=... "
            "through to SimControls, before computing spectra")
    wl = specsyn.wl()

    fixed_feh = resolve_fixed_feh(controls)

    filter_names = filter_collection.filterNames() if filter_collection is not None else []
    filter_units = filter_collection.filterUnits() if filter_collection is not None else []

    data = empty_data((ntimes, nmass), filter_names, filter_units)

    tracks = controls.tracks
    for i, t in enumerate(time_yr):
        this_feh = fixed_feh if fixed_feh is not None else feh
        isochrone = tracks.getIsochrone(np.log10(t), this_feh)
        for j, m in enumerate(mass_val):
            seg = next((s for s in isochrone if s.xMin() <= m <= s.xMax()), None)
            if seg is None:
                continue
            fill_star(
                data, i, j, seg(m), filter_collection, filter_names, filter_units,
                specsyn, wl, this_feh)

    return mass_val * u.Msun, data
