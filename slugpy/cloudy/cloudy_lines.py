"""
cloudy_lines.py

Implements parsing of cloudy's own "save last line array" ASCII output.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from pathlib import Path

from astropy import units as u

_LABEL_WIDTH = 9
_MNEMONIC_WIDTH = 4
_LUMINOSITY_THRESHOLD = 1e10  # erg/s


def read_cloudy_linearr(path: str | Path) -> tuple[u.Quantity, list[str], u.Quantity]:
    """
    Read a cloudy line array output, produced by a "save last line
    array" command in the input deck.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the line array output file.

    Returns
    -------
    line_wl : astropy.units.Quantity
        Wavelength of each kept line, in Angstrom.
    line_label : list of str
        4-character mnemonic identifying each kept line (e.g. "H  1").
    line_lum : astropy.units.Quantity
        Emergent (observable) luminosity of each kept line, in erg/s.

    Details
    -------
    Cloudy's own line array file is tab-separated, with one header
    line (skipped) followed by one row per entry: wavelength
    [Angstrom], a combined field whose first 9 characters are a
    fixed-width line label (a 4-character mnemonic followed by 5
    characters that are blank for the line's own total, or otherwise
    identify which physical process this row is one contribution to),
    log10(intrinsic luminosity) [erg/s], log10(emergent luminosity)
    [erg/s], and a (largely undocumented) type code this function
    ignores.

    An entry is kept only if (1) its label's last 5 characters are all
    blank -- i.e. it's the line's own total, not one physical
    process's contribution to it -- and (2) its emergent luminosity
    exceeds 1e10 erg/s, which excludes both physically negligible
    lines and the file's own non-line entries (broadband/continuum
    aggregates cloudy also reports in the same file).
    """
    line_wl: list[float] = []
    line_label: list[str] = []
    line_lum: list[float] = []

    with open(path) as f:
        next(f)  # header line
        for line in f:
            fields = line.split("\t")
            if len(fields) < 5:
                continue
            label9 = fields[1][:_LABEL_WIDTH]
            if label9[_MNEMONIC_WIDTH:_LABEL_WIDTH].strip() != "":
                continue
            emergent = 10.0 ** float(fields[3])
            if emergent <= _LUMINOSITY_THRESHOLD:
                continue
            line_wl.append(float(fields[0]))
            line_label.append(label9[:_MNEMONIC_WIDTH])
            line_lum.append(emergent)

    return line_wl * u.AA, line_label, line_lum * u.erg / u.s
