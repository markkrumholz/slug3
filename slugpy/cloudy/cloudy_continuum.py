"""
cloudy_continuum.py

Implements parsing of cloudy's own "save last continuum" ASCII output.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from pathlib import Path

import numpy as np
from astropy import units as u


def read_cloudy_continuum(path: str | Path) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Read a cloudy continuum output, produced by a "save last continuum"
    command in the input deck.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the continuum output file.

    Returns
    -------
    wl : astropy.units.Quantity
        Wavelength, in Angstrom, ascending.
    inc, trans, emit, trans_emit : astropy.units.Quantity
        Incident, transmitted, emitted, and transmitted+emitted
        continuum luminosity, in erg/s/Angstrom, each the same length
        and in the same order as wl.

    Details
    -------
    Cloudy's own continuum file has one header line (starting with
    "#", automatically skipped, along with any other comment lines)
    followed by one row per grid point: photon energy in Rydberg,
    then incident/transmitted/DiffOut/net-transmitted as nu*L_nu
    (erg/s), in that column order, plus additional columns this
    function ignores. Energy increases monotonically down the file,
    i.e. wavelength decreases, so rows are reversed to return an
    ascending-wavelength grid. nu*L_nu is converted to L_lambda via
    L_lambda = (nu*L_nu) / wl, which is exact once wl is already
    expressed in the same length unit (Angstrom) the result is wanted
    in.
    """
    data = np.atleast_2d(np.loadtxt(path, comments="#", usecols=(0, 1, 2, 3, 4)))
    energy = data[:, 0] * u.Ry
    wl = energy.to(u.AA, equivalencies=u.spectral())[::-1]
    nu_L_nu = (data[:, 1:5] * (u.erg / u.s))[::-1, :]
    L_lambda = nu_L_nu / wl[:, np.newaxis]
    return wl, L_lambda[:, 0], L_lambda[:, 1], L_lambda[:, 2], L_lambda[:, 3]
