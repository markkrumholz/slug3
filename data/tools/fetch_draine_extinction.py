#!/usr/bin/env python3
"""Fetch B. Draine's Milky Way dust extinction curves (R_V = 3.1, 4.0,
5.5) from https://www.astro.princeton.edu/~draine/dust/dustmix.html
and add them to data/extinct/extinct.h5 and extinct.toml, using the
same add_extinction_curve.py machinery the manually-added curves went
through (see that script's own docstring).

Each source file (e.g.
https://www.astro.princeton.edu/~draine/dust/extcurvs/kext_albedo_WD_MW_3.1_60_D03.all)
is a header of free-text metadata, followed by a dashes-only separator
line, followed by a whitespace-delimited table with columns lambda
(vacuum wavelength, micron), albedo, <cos>, C_ext/H (extinction cross
section per H nucleon, cm^2/H -- what we want, converted to Angstrom
along with lambda; the absolute units don't matter for slug's purposes,
only the curve's shape, so C_ext/H is used as-is as "kappa"), K_abs,
<cos^2>, and an optional trailing free-text comment (e.g. a named
filter or spectral line the row was inserted at). The table is nearly
perfectly wavelength-decreasing by construction, but each file has a
handful of exceptions deep in the X-ray region (far outside any
wavelength slug would realistically evaluate extinction at): a
verified typo (a missing leading zero, confirmed by cross-checking the
comment's own eV energy label against hc/lambda), a small non-monotonic
wiggle, and an exact duplicate row. parse_curve() drops exactly these
few rows using a simple, general monotonic filter (walking the table in
its own native decreasing order and keeping only rows strictly less
than the last kept wavelength) -- no hardcoded wavelengths or
thresholds -- rather than trying to correct or special-case them.

Run from the repository root: python3 data/tools/fetch_draine_extinction.py
"""
import re
import sys
import pathlib

import numpy as np
import urllib3

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import add_extinction_curve as aec  # noqa: E402

BASE_URL = "https://www.astro.princeton.edu/~draine/dust/extcurvs/"

# (curve name, source file, R_V value for the description)
CURVES = [
    ("Draine_MW_RV3.1", "kext_albedo_WD_MW_3.1_60_D03.all", "3.1"),
    ("Draine_MW_RV4.0", "kext_albedo_WD_MW_4.0A_40_D03.all", "4.0"),
    ("Draine_MW_RV5.5", "kext_albedo_WD_MW_5.5A_30_D03.all", "5.5"),
]

REFERENCE = [
    "Weingartner, B., & Draine, B. 2001, ApJ, 548, 296",
    "Li, A., & Draine, B. 2001, ApJ, 554, 778",
    "Draine, B. 2003, ARA&A, 41, 241",
    "Draine, B., 2003, ApJ, 598, 1017",
    "Draine, B. 2003, ApJ, 598, 1026",
]
REFERENCE_URL = [
    "https://ui.adsabs.harvard.edu/abs/2001ApJ...548..296W/abstract",
    "https://ui.adsabs.harvard.edu/abs/2001ApJ...554..778L/abstract",
    "https://ui.adsabs.harvard.edu/abs/2003ARA%26A..41..241D/abstract",
    "https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract",
    "https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1026D/abstract",
]

MICRON_TO_ANGSTROM = 1e4

# The table header/separator line is a run of dashes and whitespace,
# e.g. "----------- ------  ------ --------- --------- ------- --------";
# unique in these files (verified against all three), so used as-is to
# locate where the data table itself begins.
_SEPARATOR_RE = re.compile(r"^[-\s]+$")


def fetch_text(url: str) -> str:
    http = urllib3.PoolManager()
    resp = http.request("GET", url, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.status}")
    return resp.data.decode("ascii")


def parse_curve(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one Draine .all file's text into (wavelength, kappa)
    arrays, in ascending-wavelength Angstrom -- see this module's own
    docstring for the file format and the monotonic-filter row-drop
    rule."""
    in_data = False
    wavelength_micron = []
    kappa = []
    last = None
    for line in text.splitlines():
        if not in_data:
            if _SEPARATOR_RE.match(line) and "-" in line:
                in_data = True
            continue
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            lam = float(tokens[0])
            c_ext_h = float(tokens[3])
        except ValueError:
            continue
        if last is not None and not (lam < last):
            continue  # drop a row that breaks the table's own strict decrease
        wavelength_micron.append(lam)
        kappa.append(c_ext_h)
        last = lam

    # The table itself runs wavelength-decreasing; reverse to the
    # ascending order the rest of data/extinct/ uses.
    wavelength = np.array(wavelength_micron[::-1]) * MICRON_TO_ANGSTROM
    kappa = np.array(kappa[::-1])
    return wavelength, kappa


def main() -> None:
    for name, filename, r_v in CURVES:
        url = BASE_URL + filename
        text = fetch_text(url)
        wavelength, kappa = parse_curve(text)

        aec.write_h5(aec.H5_PATH, name, wavelength, kappa, REFERENCE, REFERENCE_URL)
        print(f"wrote curve '{name}' ({len(wavelength)} points) to {aec.H5_PATH}")

        aec.update_registry(aec.REGISTRY_PATH, aec.H5_PATH, name,
            REFERENCE, REFERENCE_URL,
            f"Draine (2003) Milky Way extinction curve, R_V = {r_v}")
        print(f"updated {aec.REGISTRY_PATH}")


if __name__ == "__main__":
    main()
