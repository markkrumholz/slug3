"""
cloudy_lines.py

Implements parsing of cloudy's own "save last line array" ASCII output.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import re
from pathlib import Path

from astropy import units as u

_LABEL_WIDTH = 9
_MNEMONIC_WIDTH = 4
_LUMINOSITY_THRESHOLD = 1e10  # erg/s

# A true atomic-line mnemonic is "XXNN": a 1- or 2-letter element
# symbol (space-padded on the right to 2 characters for a 1-letter
# element, e.g. "H "), followed by a 2-digit ionization stage,
# space-padded on the left for a single-digit stage (e.g. " 1"). Rows
# whose mnemonic doesn't match this form are cloudy's own pseudo-lines
# (band/flux aggregates it also reports in the same file), not real
# atomic transitions.
_MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z ][ 0-9][0-9]$")

_ROMAN_NUMERALS: tuple[tuple[int, str], ...] = (
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
)

# The widest human-readable label this module can produce: a 2-letter
# element symbol (e.g. "Fe") plus the longest Roman numeral among
# ionization stages 1-50 -- 38, "XXXVIII", 7 characters -- so 9 in
# total. write_cloudy_h5_results's own line_label dataset must be at
# least this wide to store every label without truncation.
MAX_LINE_LABEL_LENGTH = 9


def _to_roman(n: int) -> str:
    """
    Convert a positive integer to its Roman numeral representation.

    Parameters
    ----------
    n : int
        The integer to convert; assumed to be in [1, 50] (cloudy's own
        ionization-stage mnemonics never exceed this).

    Returns
    -------
    str
        n's Roman numeral representation, e.g. 4 -> "IV".
    """
    result = ""
    for value, symbol in _ROMAN_NUMERALS:
        count, n = divmod(n, value)
        result += symbol * count
    return result


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
        Human-readable element-symbol-plus-Roman-numeral label
        identifying each kept line, translated from cloudy's own
        4-character mnemonic (e.g. "H  1" -> "HI", "S  3" -> "SIII")
        -- see MAX_LINE_LABEL_LENGTH for the longest such label this
        can produce.
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
    process's contribution to it -- (2) its 4-character mnemonic
    matches the true atomic-line form "XXNN" (element symbol followed
    by ionization stage; see _MNEMONIC_RE), excluding cloudy's own
    pseudo-lines (band/flux aggregates it also reports in the same
    file, under labels that don't take this form), and (3) its
    emergent luminosity exceeds 1e10 erg/s, which excludes physically
    negligible lines. A kept row's mnemonic is then translated into a
    human-readable label: its first two characters (the element
    symbol, stripped of any padding space) unchanged, followed
    directly (no separator) by its last two characters (the
    ionization stage) converted to a Roman numeral -- e.g. cloudy's
    "H  1" becomes "HI", and "S  3" becomes "SIII".
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
            mnemonic = label9[:_MNEMONIC_WIDTH]
            if not _MNEMONIC_RE.match(mnemonic):
                continue
            emergent = 10.0 ** float(fields[3])
            if emergent <= _LUMINOSITY_THRESHOLD:
                continue
            line_wl.append(float(fields[0]))
            line_label.append(mnemonic[:2].strip() + _to_roman(int(mnemonic[2:4])))
            line_lum.append(emergent)

    return line_wl * u.AA, line_label, line_lum * u.erg / u.s
