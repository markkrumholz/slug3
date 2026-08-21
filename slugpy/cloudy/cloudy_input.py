"""
cloudy_input.py

Implements write_cloudy_input, which writes a cloudy input deck for a
single stellar spectrum, given the physical conditions of the
surrounding HII region (an hiiregparam instance).

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from pathlib import Path

import numpy as np
from astropy import constants as const
from astropy import units as u

from .hiiregparam import hiiregparam

# The bundled default template, alongside data/{imfs,filters,spectra,
# tracks,extinct}/'s own default-resource files -- resolved relative
# to this package's own location (slugpy/cloudy/cloudy_input.py is
# three directories below the repo root: slugpy/cloudy -> slugpy ->
# repo root), so this works regardless of the caller's own working
# directory, unlike the C++ side's SLUG_DIR/REPO_DIR-based resolution.
DEFAULT_TEMPLATE: Path = Path(__file__).resolve().parents[2] / "data" / "cloudy" / "cloudy.in_grid_template"

_c: float = const.c.cgs.value  # Speed of light, cm/s


def _to_rate(q: u.Quantity) -> float:
    """
    Convert an ionizing photon rate Quantity to a bare value in 1/s.

    Parameters
    ----------
    q : astropy.units.Quantity
        A photon rate, either explicitly tagged (e.g. photon/s) or a
        bare rate (1/s) -- astropy's photon unit does not decompose
        against plain SI units, so it must be handled specially (see
        hiiregparam.py's own identical helper).

    Returns
    -------
    float
        The same rate, in bare s^-1.
    """
    try:
        return q.to_value(u.photon / u.s)
    except u.UnitConversionError:
        return q.to_value(1 / u.s)


def write_cloudy_input(
    wl: u.Quantity,
    spec: u.Quantity,
    qH0: u.Quantity,
    hp: hiiregparam,
    feh: float,
    output_path: str | Path,
    template: str | Path | None = None,
) -> Path:
    """
    Write a cloudy input deck for a single spectrum.

    Parameters
    ----------
    wl : astropy.units.Quantity
        Wavelength grid of the spectrum, convertible to Angstrom.
    spec : astropy.units.Quantity
        The spectrum itself, convertible to erg/s/Angstrom, on the
        same grid as wl.
    qH0 : astropy.units.Quantity
        Ionizing photon rate of the spectrum, in photons/s (or any
        equivalent rate) -- normally the same value used to construct
        hp.
    hp : hiiregparam
        The physical conditions of the HII region -- only nII and r0
        are actually used (as the "hden"/"radius" commands).
    feh : float
        [Fe/H] of the stellar population, used to set the nebular
        metallicity via cloudy's own "metals and grains" command
        (which wants a metallicity relative to Solar, not a log
        value, so this writes 10**feh).
    output_path : str or pathlib.Path
        Path to write the cloudy input deck to. cloudy's own output
        files (named by substituting $OUTPUT_FILENAME in the template)
        use output_path's own stem (no directory, no extension), so
        cloudy should be run with its working directory set to
        output_path's own parent.
    template : str or pathlib.Path, optional
        Path to the cloudy input template to use; defaults to
        DEFAULT_TEMPLATE (data/cloudy/cloudy.in_grid_template).

    Returns
    -------
    pathlib.Path
        output_path, for convenience chaining.

    Notes
    -----
    Follows examples/cloudy_slug/cloudy_slug.py's own historical
    approach (from slug2) for writing the physical-condition commands
    and the spectral shape: the template is copied through verbatim
    (substituting $OUTPUT_FILENAME), then hden/radius/metals and
    grains/Q(H) commands are appended, followed by the spectrum itself
    as an "interpolate"/"continue" table in log10(frequency [Hz]),
    log10(L_nu [erg/s/Hz]) pairs, padded at both ends with a flux
    floor so cloudy treats everything outside the spectrum's own
    wavelength range as negligible rather than undefined. The floor is
    set per padding frequency so that nu*L_nu there -- the actual
    characteristic energy at that frequency, not L_nu alone -- is 4
    dex below the real spectrum's own minimum nu*L_nu; a
    frequency-independent L_nu floor would leave nu*L_nu at the
    padding points (~6 dex in frequency from the real data) large
    enough to dominate the total energy budget.
    """
    template_path = Path(template) if template is not None else DEFAULT_TEMPLATE
    template_lines = template_path.read_text().splitlines()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_filename = output_path.stem

    wl_aa = wl.to_value(u.AA)
    spec_flambda = spec.to_value(u.erg / u.s / u.AA)
    freq = _c / (wl_aa * 1e-8)  # Hz
    logfreq = np.log10(freq)

    specclean = np.copy(spec_flambda)
    positive = specclean[specclean > 0]
    if positive.size == 0:
        raise ValueError("write_cloudy_input: spectrum is all zero or negative")
    specclean[specclean <= 0] = np.amin(positive) * 1e-4
    logL_nu = np.log10(specclean * _c / freq ** 2)

    # Padding points sit ~6 dex in frequency away from the real
    # spectrum's own edges, so a frequency-independent L_nu floor
    # leaves nu*L_nu (the actual characteristic energy at frequency
    # nu, not L_nu alone) at those padding points many orders of
    # magnitude above the real spectrum's own minimum nu*L_nu -- large
    # enough to dominate cloudy's total energy budget. Instead, floor
    # each padding point so that nu*L_nu there equals 1e-4 times the
    # real spectrum's own minimum nu*L_nu, i.e. floor(nu) = 1e-4 *
    # min(nu*L_nu) / nu.
    log_nuLnu_floor = np.amin(logfreq + logL_nu) - 4.0

    def floor(logfreq_pad: float) -> float:
        return log_nuLnu_floor - logfreq_pad

    lines: list[str] = []
    for line in template_lines:
        if "$OUTPUT_FILENAME" in line:
            lines.append(line.replace("$OUTPUT_FILENAME", output_filename))
        else:
            lines.append(line)

    nII_cgs = hp.nII.to_value(1 / u.cm ** 3)
    r0_cgs = hp.r0.to_value(u.cm)
    qH0_cgs = _to_rate(qH0)
    lines.append(f"hden {np.log10(nII_cgs):f}")
    lines.append(f"radius {np.log10(r0_cgs):f}")
    lines.append(f"metals and grains {10.0 ** feh:f}")
    lines.append(f"Q(H) = {np.log10(qH0_cgs):f}")

    spec_line = "interpolate"
    spec_line += f" ({7.51:15.12f} {floor(7.51):f})"
    spec_line += f" ({logfreq[-1] - 0.01:15.12f} {floor(logfreq[-1] - 0.01):f})"
    n = len(logL_nu)
    for i in range(n):
        if i % 4 == 0:
            spec_line += "\ncontinue"
        if logfreq[-i - 1] == logfreq[-(i - 1) - 1]:
            continue
        spec_line += f" ({logfreq[-i - 1]:15.12f} {logL_nu[-i - 1]:f})"
    spec_line += f"\ncontinue ({logfreq[0] + 0.01:15.12f} {floor(logfreq[0] + 0.01):f})"
    spec_line += f" ({22.4:f} {floor(22.4):f})"
    lines.append(spec_line)

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
