"""
process_cloudy_grid.py

Third and final stage of the three-stage pipeline to build a "quick
and dirty" nebular emission lookup table -- see make_slug_grid.py's
own module docstring for the full pipeline description, and
run_cloudy_grid.py for the second stage this one consumes. This
script's own piece: post-process every cloudy run run_cloudy_grid.py
produced into a single lookup table, cloudy_table.h5, mirroring
slug2's own cloudy_slug/process_grid.py in spirit (see that file's own
line_filter/line_filter_interp, reused here nearly verbatim) but
restructured around slug3's per-file cluster_cloudy/galaxy_cloudy
groups instead of slug2's separate cloudy output files.

For every (uid or trial, output time, U) row stored in a processed
file's own cluster_cloudy/galaxy_cloudy group, this script:

- Strips emission lines out of the emergent continuum (cloudy's own
  "save last continuum" output can have line flux leaking into nearby
  continuum bins at its own grid resolution -- see line_filter) and
  interpolates the result onto a single common wavelength grid shared
  by the whole table.
- Normalizes both that continuum and the row's own line luminosities
  by the ionizing photon rate Q(HI) at that row's own output time, so
  every stored quantity is a per-ionizing-photon yield: erg/Angstrom/
  photon for the continuum, erg/photon for each line.

The common wavelength grid and line list are each built the same way
run_cloudy itself builds them within a single file (see
slugpy.cloudy.cloudy_output's own _grow_wavelength_grid/_append_lines):
the wavelength grid is the longest "wl" grid found across every
processed file (cloudy's own grid always shares the same maximum
wavelength and spacing across runs, so a shorter grid is always a
right-aligned subset of a longer one); the line list is the union of
every distinct line wavelength seen anywhere, sorted ascending -- but
first restricted, row by row, to the --nline brightest lines that row
itself reported (default 100; see _row_top_lines), since a full,
unrestricted line list runs to ~10,000 entries and makes the table
much larger than it needs to be. Because the brightest lines differ
somewhat from one row to the next, the table's own overall line count
can still end up well above --nline once every row's own top-nline
set is unioned together -- just far short of the unrestricted total.

Output layout (cloudy_table.h5, default name):

- Top-level datasets: wl (Angstrom, the common continuum grid),
  line_wl/line_label (Angstrom / cloudy's own line label, the common
  line list), and time (yr, every distinct cluster output time found
  -- the same 0.25-10 Myr, 0.25 Myr grid make_slug_grid.py's cluster
  decks use; consumers should treat any t < 0.25 Myr as equal to the
  t = 0.25 Myr row -- see make_slug_grid.py's own module docstring for
  why the grid starts there instead of at t = 0).
- One group per track set found (e.g. "MIST"; attrs["track"] records
  the name), containing one group per [Fe/H] value found within it
  (e.g. "FeH+0.0000"; attrs["FeH"] records the value), containing one
  group per v/vcrit value found within that (e.g. "vvcrit0.00";
  attrs["v_vcrit"] records the value), containing one group per
  log10(U) value found within that (e.g. "logU-2.50"; attrs["logU"]
  records the value; see _assign_nominal_log_u for how each row is
  matched back to the nominal --log-u value it was actually requested
  at, which its own U may not be closest to once hiiregparam has
  adjusted it).
- Within each logU group: a "galaxy" group (if that combination has
  galaxy data) with 1-D "spec" (erg/Angstrom/photon, on the common
  wavelength grid, zero where that grid extends past what this
  particular run covered) and "line_lum" (erg/photon, on the common
  line list, zero for lines this run didn't report); and/or a
  "cluster" group (if that combination has cluster data) with the same
  two datasets but 2-D, one row per entry in the top-level time
  dataset, zero for any output time this combination has no cloudy run
  for (including times excluded by run_cloudy_grid.py's own
  --qhi-fraction cut) -- except for an isolated missing output time
  with a real cloudy run immediately before and after it in this same
  combination's own rows (see _fill_bounded_gaps): that row is instead
  filled in by averaging its two neighbors' own continua outright, and
  averaging their line luminosities for whichever lines both neighbors
  report (any line only one of them reports, or neither does, is left
  at 0 for the filled row, same as an ordinary missing line). This
  covers the case of a single cloudy run genuinely failing (as opposed
  to being skipped for falling under --qhi-fraction, which run_cloudy_
  grid.py's own report distinguishes from a failure, but which this
  script has no way to tell apart after the fact) -- run_cloudy_grid.py
  does not stop the pipeline over such a failure, precisely so this
  gap-fill gets a chance to paper over it.

By default, --output is merged into rather than replaced: if a file
already exists there, it is read back first (see read_existing_table),
and the new table this call writes carries forward every (track,
[Fe/H], v/vcrit) combination it already had that --work-dir's own
entries don't actually produce fresh cloudy results for, remapped onto
whatever this call's own merged wl/line/time grids turn out to be (see
compute_global_grids, _remap_combo_sim) -- a combination --work-dir
does produce fresh results for is instead written from those results,
entirely replacing (not merging into) whatever that same combination
already had, on the assumption that producing fresh results at all
means it was deliberately (re)run. A combination --work-dir merely
*covers* (i.e. its own slug output file(s) are there) but that yields
no cloudy results at all -- every cloudy run for it failed, most
commonly -- is not "fresh" by this definition, so whatever it already
had in --output survives untouched instead of being silently dropped.
This is what lets --work-dir be reused a track set (or [Fe/H]/v/vcrit
slice) at a time -- e.g. running the full pipeline once for MIST/
Stromlo/PARSEC_comp, discarding that --work-dir once done, then later
running it again from scratch for a newly-available track set like
Geneva -- without the second run wiping out the first's own results:
make_slug_grid.py's own --output-table already skips writing decks for
combinations already complete there, so a --work-dir populated this
way only ever contains combinations genuinely missing from --output,
and this script's own merge just needs to add those in without
disturbing anything else.
Pass --overwrite to restore the old behavior instead: build --output
from --work-dir alone, discarding whatever it already had.

Like the two earlier stages, this is cheap to run compared to the
cloudy runs it post-processes, but only meaningful once run against a
work-dir run_cloudy_grid.py has actually populated -- verify locally
against the same narrow --feh-min/--feh-max slice make_slug_grid.py
and run_cloudy_grid.py were run with.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

# Imports
import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple, cast

import astropy.units as u
import h5py
import numpy as np

# slugpy is a sibling package three directories up from this script
# (data/tools/cloudy -> data/tools -> data -> repo root), and is not
# installed, so it has to be reached via sys.path directly; this
# script's own directory is also added so run_cloudy_grid (a sibling
# script, not a package) can be imported directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from slugpy.cloudy.cloudy_lines import MAX_LINE_LABEL_LENGTH  # noqa: E402 -- see sys.path setup above
from slugpy.slug_group_reader import slug_group_reader  # noqa: E402
from slugpy.slug_phot_reader import slug_phot_reader  # noqa: E402
from slugpy.slug_reader import slug_reader  # noqa: E402

from run_cloudy_grid import LOG_U_VALUES, find_h5_files  # noqa: E402 -- sibling script, see sys.path setup above

DEFAULT_WORK_DIR = _SCRIPT_DIR / "slug_grid_work"
DEFAULT_OUTPUT = _SCRIPT_DIR / "cloudy_table.h5"
DEFAULT_NLINE = 100

_LINE_FILTER_PASSES = 3
_LINE_FILTER_THRESH = 0.01

_QHI_UNIT = u.photon / u.s
_FLUX_UNIT = u.erg / u.s / u.AA
_LINE_UNIT = u.erg / u.s
_SPEC_NORM_UNIT = u.erg / u.AA / u.photon
_LINE_NORM_UNIT = u.erg / u.photon


def line_filter(wl: np.ndarray, spec: np.ndarray, thresh: float = _LINE_FILTER_THRESH) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove one pass of narrow upward spikes from a spectrum.

    A direct translation of slug2's own cloudy_slug/process_grid.py
    line_filter: a point is removed if it is more than a factor
    (1 + thresh) above both of its immediate neighbors, or above both
    of its next-nearest neighbors with the intervening point also
    removed (catching two-point-wide spikes as well as single-point
    ones).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths, ascending.
    spec : numpy.ndarray
        Spectrum values at each wavelength in wl.
    thresh : float, default 0.01
        Fractional excess above both neighbors required to flag a
        point as a spike.

    Returns
    -------
    wl_filt, spec_filt : numpy.ndarray
        wl and spec with every flagged point removed.
    """
    flagged = np.zeros(wl.shape, dtype=bool)
    lfilt = spec[1:-1] > (1 + thresh) * spec[:-2]
    rfilt = spec[1:-1] > (1 + thresh) * spec[2:]
    flagged[1:-1] = lfilt & rfilt
    flagged[1:-2] = flagged[1:-2] | (lfilt[:-1] & rfilt[1:])
    flagged[2:-1] = flagged[2:-1] | (lfilt[:-1] & rfilt[1:])
    keep = ~flagged
    return wl[keep], spec[keep]


def line_filter_interp(wl: np.ndarray, spec: np.ndarray, wl_grid: np.ndarray,
    npass: int = _LINE_FILTER_PASSES) -> np.ndarray:
    """
    Strip emission lines from a spectrum and interpolate it onto a target wavelength grid.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelengths spec is defined on, ascending.
    spec : numpy.ndarray
        Spectrum values at each wavelength in wl.
    wl_grid : numpy.ndarray
        Wavelength grid to interpolate the line-stripped spectrum onto.
    npass : int, default 3
        Number of times to apply line_filter; matches slug2's own
        cloudy_slug/process_grid.py line_filter_interp default, enough
        to also remove spikes line_filter's own neighbor-based check
        only catches after an adjacent spike has already been removed.

    Returns
    -------
    numpy.ndarray
        The line-stripped spectrum, linearly interpolated onto
        wl_grid, with 0 outside wl's own range.
    """
    wl_pass, spec_pass = wl, spec
    for _ in range(npass):
        wl_pass, spec_pass = line_filter(wl_pass, spec_pass)
    return np.interp(wl_grid, wl_pass, spec_pass, left=0.0, right=0.0)


def _assign_nominal_log_u(row_time: np.ndarray, log_u_values: list[float]) -> list[float | None]:
    """
    Recover the nominal log10(U) value each row was actually requested at, from row order alone.

    A row's own stored U is the *achieved* value -- hiiregparam can
    adjust the requested U to keep it physically consistent with the
    run's own density and ionizing luminosity (see its own "U too
    large for input value of nII" warning), so it need not equal, or
    even be closest to, 10**(the log_u_values entry that was actually
    requested). Matching rows back to a nominal log10(U) by nearest
    achieved value is therefore unsound: a heavily-lowered run
    requested at one nominal value can end up nearer a neighboring
    nominal value than the row genuinely run at that neighbor, and
    silently overwrite it.

    Instead, this relies on run_cloudy_grid.py's own process_cluster_
    file/process_galaxy_file, each of which loops over log_u_values in
    order and, for each entry, fully appends one row per qualifying
    output time (cluster) or one row (galaxy) before moving on to the
    next entry -- so the k-th time a given output time recurs in row
    order is always that time's own k-th log_u_values entry's run,
    regardless of what U that run actually achieved.

    Parameters
    ----------
    row_time : numpy.ndarray
        Output time of each row, in yr, in on-disk row order (see
        _GroupArrays.row_time).
    log_u_values : list of float
        The nominal log10(U) grid cloudy was requested to run at (see
        run_cloudy_grid.py's own --log-u), in the same order it was
        passed there.

    Returns
    -------
    list of float or None
        log_u_values[k] for the k-th occurrence (in row order) of that
        row's own output time; None for a (len(log_u_values)+1)-th or
        later occurrence, which should not happen -- see callers for
        how this is handled.
    """
    seen: dict[float, int] = {}
    assigned: list[float | None] = []
    for t in row_time:
        k = seen.get(t, 0)
        seen[t] = k + 1
        assigned.append(log_u_values[k] if k < len(log_u_values) else None)
    return assigned


def _row_top_lines(wl: np.ndarray, label: list[str], lum_row: np.ndarray,
    nline: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Get one row's own reported lines, restricted to the nline brightest.

    Parameters
    ----------
    wl : numpy.ndarray
        The shared line-wavelength list lum_row is aligned with, in
        Angstrom (typically a group's own "line_wl" dataset -- a union
        across every row in that group, so most entries are 0 for any
        given row).
    label : list of str
        Cloudy's own label for each wavelength in wl, same order.
    lum_row : numpy.ndarray
        One row's own line luminosity at each wavelength in wl -- 0
        wherever that row's own cloudy run didn't report a line.
    nline : int
        Maximum number of lines to keep, ranked by luminosity
        descending. A value <= 0 means "keep every line this row
        reported" (no brightness cut at all).

    Returns
    -------
    wl, label, lum : as above, restricted to this row's own nonzero
        entries, then (if nline > 0) further restricted to the nline
        brightest of those. Since which lines are brightest generally
        differs from one row to the next, applying this same cut to
        every row and then taking the union (as compute_global_grids
        and process_cluster_file/process_galaxy_file both do) can
        still end up with more than nline distinct lines overall.
    """
    nz = lum_row > 0
    row_wl = wl[nz]
    row_label = [label[i] for i in np.nonzero(nz)[0]]
    row_lum = lum_row[nz]
    if nline <= 0 or len(row_wl) <= nline:
        return row_wl, row_label, row_lum
    order = np.argsort(row_lum)[::-1][:nline]
    return row_wl[order], [row_label[i] for i in order], row_lum[order]


class FileEntry(NamedTuple):
    """One discovered slug HDF5 output file, classified by simulation type, track set, [Fe/H], and v/vcrit."""

    path: Path
    sim_type: str
    track: str
    feh: float
    vvcrit: float


def discover_files(work_dir: Path) -> list[FileEntry]:
    """
    Classify every slug HDF5 output file in a working directory by simulation type, track set, [Fe/H], and v/vcrit.

    Parameters
    ----------
    work_dir : pathlib.Path
        Directory to search (see find_h5_files); should be the same
        --work-dir make_slug_grid.py and run_cloudy_grid.py were run
        with.

    Returns
    -------
    list of FileEntry
        One entry per "*.h5" file found, with track/feh/vvcrit read
        from that file's own input_deck stars.tracks/FeH/v_vcrit
        entries.

    Raises
    ------
    ValueError
        If any file's own input deck is missing sim_type, stars.tracks,
        stars.FeH, or stars.v_vcrit, or has a sim_type other than
        "cluster"/"galaxy".
    """
    entries: list[FileEntry] = []
    for path in find_h5_files(work_dir):
        deck = slug_reader(str(path)).input_deck
        sim_type = deck.get("sim_type")
        track = deck.get("stars", {}).get("tracks")
        feh = deck.get("stars", {}).get("FeH")
        vvcrit = deck.get("stars", {}).get("v_vcrit")
        if sim_type not in ("cluster", "galaxy") or track is None or feh is None or vvcrit is None:
            raise ValueError(
                f"discover_files: {path} has an unexpected input deck "
                f"(sim_type={sim_type!r}, stars.tracks={track!r}, "
                f"stars.FeH={feh!r}, stars.v_vcrit={vvcrit!r})")
        entries.append(FileEntry(path, cast(str, sim_type), str(track), float(feh), float(vvcrit)))
    return entries


class ExistingTable(NamedTuple):
    """
    An earlier build_table call's own output, read back for merging into a new one.

    Attributes
    ----------
    wl : numpy.ndarray
        This table's own top-level continuum wavelength grid, in
        Angstrom.
    line_wl : numpy.ndarray
        This table's own top-level line wavelength list, in Angstrom.
    line_label : list of str
        Cloudy's own label for each wavelength in line_wl, same order.
    time : numpy.ndarray
        This table's own top-level cluster output-time list, in yr.
    combos : dict mapping (str, float, float) to dict
        {(track, [Fe/H], v/vcrit): {log10(U): {"cluster" or "galaxy":
        {"spec": ..., "line_lum": ...}}}} -- every group this table
        has, with every array still aligned to this same object's own
        wl/line_wl/time (i.e. not yet remapped onto a new build's own
        merged grids; see _remap_combo_sim).
    """

    wl: np.ndarray
    line_wl: np.ndarray
    line_label: list[str]
    time: np.ndarray
    combos: dict[tuple[str, float, float], dict[float, dict[str, dict[str, np.ndarray]]]]


def read_existing_table(path: Path) -> ExistingTable | None:
    """
    Read back an existing build_table output for merging into a new one.

    Parameters
    ----------
    path : pathlib.Path
        Path to a previously-written table (see build_table's own
        output layout); need not exist.

    Returns
    -------
    ExistingTable, optional
        None if path does not exist yet. Otherwise every dataset this
        table has -- its own top-level wl/line_wl/line_label/time and
        every group's own spec/line_lum -- is read fully into plain
        in-memory numpy arrays, not left as lazy h5py datasets: by
        default build_table reopens path itself in "w" (i.e.
        truncating) mode immediately after calling this, since the
        normal case is --output pointing at the very file being merged
        from (rerunning this script against the same --output it wrote
        last time), so nothing here can still be reading from path by
        the time that happens.
    """
    if not path.exists():
        return None

    combos: dict[tuple[str, float, float], dict[float, dict[str, dict[str, np.ndarray]]]] = {}
    with h5py.File(path, "r") as f:
        wl = f["wl"][()]
        line_wl = f["line_wl"][()]
        line_label = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in f["line_label"][()]]
        time = f["time"][()]

        for track_name, track_group in f.items():
            if not isinstance(track_group, h5py.Group):
                continue  # skip the top-level wl/line_wl/line_label/time datasets themselves
            for feh_group in track_group.values():
                feh = float(feh_group.attrs["FeH"])
                for vvcrit_group in feh_group.values():
                    vvcrit = float(vvcrit_group.attrs["v_vcrit"])
                    logu_data: dict[float, dict[str, dict[str, np.ndarray]]] = {}
                    for logu_group in vvcrit_group.values():
                        log_u = float(logu_group.attrs["logU"])
                        logu_data[log_u] = {
                            sim_type: {"spec": logu_group[sim_type]["spec"][()],
                                "line_lum": logu_group[sim_type]["line_lum"][()]}
                            for sim_type in ("cluster", "galaxy") if sim_type in logu_group
                        }
                    combos[(track_name, feh, vvcrit)] = logu_data

    return ExistingTable(wl=wl, line_wl=line_wl, line_label=line_label, time=time, combos=combos)


def compute_global_grids(
    entries: list[FileEntry], nline: int, existing: ExistingTable | None = None,
) -> tuple[u.Quantity, np.ndarray, list[str], np.ndarray]:
    """
    Build the common wavelength grid, line list, and cluster output-time list shared by the whole table.

    Parameters
    ----------
    entries : list of FileEntry
        As returned by discover_files.
    nline : int
        Maximum number of lines to keep per row, ranked by luminosity
        (see _row_top_lines); a value <= 0 keeps every line. The union
        of every row's own top-nline lines, across every processed
        file, can still exceed nline overall, since the brightest
        lines generally differ from one row to the next.
    existing : ExistingTable, optional
        A previously-written table being merged into (see
        read_existing_table); its own wl/line_wl/line_label/time are
        folded into the result alongside whatever entries itself
        contributes, so a combination being carried forward unchanged
        from existing (see build_table) never needs remapping onto a
        smaller grid than its own data already lives on.

    Returns
    -------
    global_wl : astropy.units.Quantity
        The longest "wl" continuum grid found across every processed
        file's own cluster_cloudy/galaxy_cloudy group, and existing's
        own wl if given (see this module's own docstring for why the
        longest grid is always a superset of every shorter one).
    global_line_wl : numpy.ndarray
        Ascending-sorted union of every row's own top-nline distinct
        line wavelengths (in Angstrom) found across every processed
        file, and every wavelength in existing.line_wl if given.
    global_line_label : list of str
        Cloudy's own label for each wavelength in global_line_wl, in
        the same order.
    global_time : numpy.ndarray
        Ascending-sorted union of every distinct cluster output time
        (in yr) found across every processed cluster-type file's own
        cluster_spectra group, and every time in existing.time if
        given.

    Raises
    ------
    ValueError
        If existing is None and none of the processed files have any
        cloudy continuum data yet (i.e. run_cloudy_grid.py has not
        been run against work_dir, or every run in it failed).
    """
    global_wl: u.Quantity | None = None
    line_table: dict[float, str] = {}
    time_set: set[float] = set()

    if existing is not None:
        global_wl = existing.wl * u.AA
        line_table.update(zip(existing.line_wl.tolist(), existing.line_label, strict=True))
        time_set.update(existing.time.tolist())

    for path, sim_type, _track, _feh, _vvcrit in entries:
        reader = slug_reader(str(path))
        group = reader.cluster_cloudy if sim_type == "cluster" else reader.galaxy_cloudy
        if group is not None:
            if "wl" in group.keys():
                wl = cast(u.Quantity, group["wl"])
                if global_wl is None or len(wl) > len(global_wl):
                    global_wl = wl
            if "line_wl" in group.keys():
                line_wl = cast(u.Quantity, group["line_wl"]).to_value(u.AA)
                line_label = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in group["line_label"]]
                line_lum = cast(u.Quantity, group["line_lum"]).to_value(_LINE_UNIT)
                for row in range(line_lum.shape[0]):
                    top_wl, top_label, _ = _row_top_lines(line_wl, line_label, line_lum[row, :], nline)
                    for w, lbl in zip(top_wl, top_label, strict=True):
                        line_table.setdefault(w, lbl)

        if sim_type == "cluster":
            spectra = reader.cluster_spectra
            if spectra is not None:
                time_set.update(cast(u.Quantity, spectra["time"]).to_value(u.yr).tolist())

    if global_wl is None:
        raise ValueError(
            "compute_global_grids: none of the processed files have any cloudy continuum "
            "data yet -- has run_cloudy_grid.py been run against this work-dir?")

    global_line_wl = np.array(sorted(line_table), dtype=np.float64)
    global_line_label = [line_table[w] for w in global_line_wl]
    global_time = np.array(sorted(time_set), dtype=np.float64)
    return global_wl, global_line_wl, global_line_label, global_time


class _GroupArrays:
    """
    A cluster_cloudy/galaxy_cloudy group's own datasets, read once as plain arrays for repeated per-row use.

    Attributes
    ----------
    row_time : numpy.ndarray
        Output time of each row, in yr.
    row_u : numpy.ndarray
        Actual (dimensionless) ionization parameter of each row.
    file_wl : numpy.ndarray or None
        This group's own continuum wavelength grid, in Angstrom, or
        None if the group has no continuum data at all.
    file_line_wl : numpy.ndarray or None
        This group's own line wavelength list, in Angstrom, or None if
        the group has no line data at all.
    file_line_label : list of str or None
        Cloudy's own label for each wavelength in file_line_wl, same
        order; None iff file_line_wl is None.
    spec_emit : numpy.ndarray or None
        This group's own emitted continuum, one row per entry in
        row_time/row_u, in erg/s/Angstrom, aligned with file_wl; None
        iff file_wl is None.
    line_lum : numpy.ndarray or None
        This group's own line luminosities, one row per entry in
        row_time/row_u, in erg/s, aligned with file_line_wl; None iff
        file_line_wl is None.
    """

    def __init__(self, group: slug_group_reader) -> None:
        has_spec = "wl" in group.keys()
        has_lines = "line_wl" in group.keys()
        self.row_time = cast(u.Quantity, group["time"]).to_value(u.yr)
        # U is stored dimensionless (units="", see cloudy_output.py's own
        # _append_scalar call for it), so slug_group_reader returns it as
        # a plain ndarray rather than a Quantity.
        self.row_u = cast(np.ndarray, group["U"])
        self.file_wl = cast(u.Quantity, group["wl"]).to_value(u.AA) if has_spec else None
        self.file_line_wl = cast(u.Quantity, group["line_wl"]).to_value(u.AA) if has_lines else None
        self.file_line_label = ([lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in group["line_label"]]
            if has_lines else None)
        self.spec_emit = cast(u.Quantity, group["spec_emit"]).to_value(_FLUX_UNIT) if has_spec else None
        self.line_lum = cast(u.Quantity, group["line_lum"]).to_value(_LINE_UNIT) if has_lines else None


def _qhi_by_time(phot: slug_phot_reader) -> dict[float, float]:
    """Build a {output time in yr: Q(HI) in photon/s} lookup from a cluster_phot/galaxy_phot group."""
    times = cast(u.Quantity, phot["time"]).to_value(u.yr)
    qhi = cast(u.Quantity, phot["Q(HI)"]).to_value(_QHI_UNIT)
    return dict(zip(times, qhi, strict=True))


def _normalize_row(arrays: _GroupArrays, row: int, qhi: float, global_wl: np.ndarray,
    line_index: dict[float, int], n_line_total: int, nline: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one row's own line-stripped continuum and line luminosities, normalized by Q(HI).

    Parameters
    ----------
    arrays : _GroupArrays
        The source group's own arrays.
    row : int
        Which row of arrays to normalize.
    qhi : float
        Q(HI), in photon/s, at this row's own output time.
    global_wl : numpy.ndarray
        The table's own common continuum wavelength grid, in Angstrom.
    line_index : dict mapping float to int
        wavelength (Angstrom) -> column index into the table's own
        common line list.
    n_line_total : int
        Number of entries in the table's own common line list.
    nline : int
        Maximum number of this row's own lines to keep, ranked by
        luminosity (see _row_top_lines); a value <= 0 keeps every line
        this row reported. Must match the value compute_global_grids
        was called with, so every line this row keeps is guaranteed to
        already be in line_index.

    Returns
    -------
    spec : numpy.ndarray
        Line-stripped continuum on global_wl, in erg/Angstrom/photon
        (0 where global_wl extends past arrays.file_wl's own range, or
        if arrays has no continuum data at all).
    line_lum : numpy.ndarray
        This row's own top-nline line luminosities, on the table's own
        common line list, in erg/photon (0 for any line not among
        those, or if arrays has no line data at all).
    """
    spec = np.zeros(len(global_wl))
    if arrays.file_wl is not None and arrays.spec_emit is not None:
        spec[:] = line_filter_interp(arrays.file_wl, arrays.spec_emit[row, :], global_wl) / qhi

    line_lum = np.zeros(n_line_total)
    if arrays.file_line_wl is not None and arrays.file_line_label is not None and arrays.line_lum is not None:
        top_wl, _, top_lum = _row_top_lines(arrays.file_line_wl, arrays.file_line_label,
            arrays.line_lum[row, :], nline)
        for w, lum in zip(top_wl, top_lum, strict=True):
            line_lum[line_index[w]] = lum / qhi

    return spec, line_lum


def process_cluster_file(path: Path, global_wl: np.ndarray, line_index: dict[float, int],
    time_index: dict[float, int], log_u_values: list[float], nline: int) -> dict[float, dict[str, np.ndarray]]:
    """
    Normalize every row of one cluster-type file's own cluster_cloudy group.

    Parameters
    ----------
    path : pathlib.Path
        Path to the cluster-type slug HDF5 output file.
    global_wl : numpy.ndarray
        The table's own common continuum wavelength grid, in Angstrom.
    line_index : dict mapping float to int
        wavelength (Angstrom) -> column index into the table's own
        common line list.
    time_index : dict mapping float to int
        output time (yr) -> row index into the table's own common
        cluster output-time list.
    log_u_values : list of float
        Nominal log10(U) grid to sort rows into (see
        _assign_nominal_log_u), in the same order passed to
        run_cloudy_grid.py's own --log-u.
    nline : int
        Maximum number of lines to keep per row (see _row_top_lines);
        must match the value compute_global_grids built line_index
        with.

    Returns
    -------
    dict mapping float to dict
        {log10(U): {"spec": (ntime, nwl) array, "line_lum": (ntime, nline)
        array, "has_data": (ntime,) bool array}}, one entry per log10(U)
        value actually found. "spec"/"line_lum" are zero in any row for
        an output time this combination has no cloudy run for;
        "has_data" is True at exactly the rows that aren't -- see
        _fill_bounded_gaps, which callers should run over each bucket
        before "has_data" is discarded and "spec"/"line_lum" written
        out. Empty if this file has no cluster_cloudy group, or no
        matching cluster_phot Q(HI) photometry.
    """
    reader = slug_reader(str(path))
    group = reader.cluster_cloudy
    phot = reader.cluster_phot
    if group is None:
        print(f"  {path.name}: no cluster_cloudy group; skipping", file=sys.stderr)
        return {}
    if phot is None or "Q(HI)" not in phot.filters:
        print(f"  {path.name}: no cluster_phot Q(HI) photometry; skipping", file=sys.stderr)
        return {}

    arrays = _GroupArrays(group)
    qhi_by_time = _qhi_by_time(phot)
    ntime, nwl, n_line_total = len(time_index), len(global_wl), len(line_index)
    nominal_log_u = _assign_nominal_log_u(arrays.row_time, log_u_values)

    result: dict[float, dict[str, np.ndarray]] = {}
    for i in range(len(arrays.row_time)):
        t = arrays.row_time[i]
        qhi = qhi_by_time.get(t)
        t_idx = time_index.get(t)
        if qhi is None or qhi <= 0 or t_idx is None:
            continue

        log_u = nominal_log_u[i]
        if log_u is None:
            print(f"  {path.name}: t={t:.0f} yr has more cloudy runs than --log-u values "
                f"({len(log_u_values)}); skipping the extra one", file=sys.stderr)
            continue
        bucket = result.setdefault(log_u, {
            "spec": np.zeros((ntime, nwl)),
            "line_lum": np.zeros((ntime, n_line_total)),
            "has_data": np.zeros(ntime, dtype=bool),
        })
        spec, line_lum = _normalize_row(arrays, i, qhi, global_wl, line_index, n_line_total, nline)
        bucket["spec"][t_idx, :] = spec
        bucket["line_lum"][t_idx, :] = line_lum
        bucket["has_data"][t_idx] = True

    return result


def _fill_bounded_gaps(bucket: dict[str, np.ndarray]) -> None:
    """
    Fill in an isolated missing output time bounded by real cloudy runs on both sides.

    One cluster/log10(U) combination's own "spec"/"line_lum" rows (see
    process_cluster_file) are zero for any output time index this
    combination has no cloudy run for -- most commonly a trailing block
    of the latest output times, once Q(HI) has fallen under run_cloudy_
    grid.py's own --qhi-fraction cut, but occasionally a single time
    sandwiched between two others where cloudy itself simply failed to
    converge (see run_cloudy_grid.py's own module docstring). This
    function papers over the latter case -- a row with no data of its
    own, but with the immediately preceding and following row (by
    output-time index, i.e. this combination's own nearest earlier/
    later entries in the table's shared time grid, not necessarily
    adjacent in real time if this combination is missing other rows
    too) both already having data -- by filling it in from those two
    neighbors: their continua averaged outright, and their line
    luminosities averaged wherever both report a nonzero value for the
    same line (any line only one of them reports, or neither does, is
    left at 0, exactly like an ordinary missing line elsewhere in the
    table).

    A run of two or more consecutive missing rows is left untouched --
    only a single missing row has both an immediately preceding and
    immediately following row with data -- as is a missing row at
    either end of the time grid, which can only ever have a neighbor on
    one side.

    Parameters
    ----------
    bucket : dict
        One log10(U) entry of process_cluster_file's own return value,
        with "spec" ((ntime, nwl) array), "line_lum" ((ntime, nline)
        array), and "has_data" ((ntime,) bool array) all aligned by
        output-time row index. Modifies "spec" and "line_lum" in place;
        "has_data" is read but left as found, so a just-filled row is
        never itself treated as a bound for another gap.
    """
    has_data = bucket["has_data"]
    spec = bucket["spec"]
    line_lum = bucket["line_lum"]
    for i in range(1, len(has_data) - 1):
        if has_data[i] or not has_data[i - 1] or not has_data[i + 1]:
            continue
        spec[i, :] = 0.5 * (spec[i - 1, :] + spec[i + 1, :])
        both_nonzero = (line_lum[i - 1, :] != 0) & (line_lum[i + 1, :] != 0)
        line_lum[i, both_nonzero] = 0.5 * (line_lum[i - 1, both_nonzero] + line_lum[i + 1, both_nonzero])


def process_galaxy_file(path: Path, global_wl: np.ndarray, line_index: dict[float, int],
    log_u_values: list[float], nline: int) -> dict[float, dict[str, np.ndarray]]:
    """
    Normalize every row of one galaxy-type file's own galaxy_cloudy group.

    Parameters
    ----------
    path : pathlib.Path
        Path to the galaxy-type slug HDF5 output file.
    global_wl : numpy.ndarray
        The table's own common continuum wavelength grid, in Angstrom.
    line_index : dict mapping float to int
        wavelength (Angstrom) -> column index into the table's own
        common line list.
    log_u_values : list of float
        Nominal log10(U) grid to sort rows into (see
        _assign_nominal_log_u), in the same order passed to
        run_cloudy_grid.py's own --log-u.
    nline : int
        Maximum number of lines to keep per row (see _row_top_lines);
        must match the value compute_global_grids built line_index
        with.

    Returns
    -------
    dict mapping float to dict
        {log10(U): {"spec": (nwl,) array, "line_lum": (nline,) array}},
        one entry per log10(U) value actually found. Empty if this
        file has no galaxy_cloudy group, or no matching galaxy_phot
        Q(HI) photometry.
    """
    reader = slug_reader(str(path))
    group = reader.galaxy_cloudy
    phot = reader.galaxy_phot
    if group is None:
        print(f"  {path.name}: no galaxy_cloudy group; skipping", file=sys.stderr)
        return {}
    if phot is None or "Q(HI)" not in phot.filters:
        print(f"  {path.name}: no galaxy_phot Q(HI) photometry; skipping", file=sys.stderr)
        return {}

    arrays = _GroupArrays(group)
    qhi_by_time = _qhi_by_time(phot)
    n_line_total = len(line_index)
    nominal_log_u = _assign_nominal_log_u(arrays.row_time, log_u_values)

    result: dict[float, dict[str, np.ndarray]] = {}
    for i in range(len(arrays.row_time)):
        qhi = qhi_by_time.get(arrays.row_time[i])
        if qhi is None or qhi <= 0:
            continue
        log_u = nominal_log_u[i]
        if log_u is None:
            print(f"  {path.name}: t={arrays.row_time[i]:.0f} yr has more cloudy runs than "
                f"--log-u values ({len(log_u_values)}); skipping the extra one", file=sys.stderr)
            continue
        spec, line_lum = _normalize_row(arrays, i, qhi, global_wl, line_index, n_line_total, nline)
        result[log_u] = {"spec": spec, "line_lum": line_lum}

    return result


def _write_top_level(fout: h5py.File, global_wl: u.Quantity, global_line_wl: np.ndarray,
    global_line_label: list[str], global_time: np.ndarray) -> None:
    """Write the table's own top-level wl/line_wl/line_label/time datasets."""
    wl_dset = fout.create_dataset("wl", data=global_wl.to_value(u.AA))
    wl_dset.attrs["units"] = str(u.AA)

    line_wl_dset = fout.create_dataset("line_wl", data=global_line_wl)
    line_wl_dset.attrs["units"] = str(u.AA)

    label_dset = fout.create_dataset("line_label", shape=(len(global_line_label),),
        dtype=h5py.string_dtype(encoding="ascii", length=MAX_LINE_LABEL_LENGTH))
    label_dset[:] = global_line_label

    time_dset = fout.create_dataset("time", data=global_time)
    time_dset.attrs["units"] = str(u.yr)


def _write_normalized_group(parent: h5py.Group, name: str, data: dict[str, np.ndarray]) -> None:
    """Write one normalized "galaxy" or "cluster" group's own spec/line_lum datasets."""
    grp = parent.require_group(name)
    spec_dset = grp.create_dataset("spec", data=data["spec"])
    spec_dset.attrs["units"] = str(_SPEC_NORM_UNIT)
    line_dset = grp.create_dataset("line_lum", data=data["line_lum"])
    line_dset.attrs["units"] = str(_LINE_NORM_UNIT)


def _remap_wl(old_wl: np.ndarray, values: np.ndarray, new_wl: np.ndarray) -> np.ndarray:
    """
    Re-grid an already-normalized spectrum (or a stack of them) onto a new common wavelength grid.

    Parameters
    ----------
    old_wl : numpy.ndarray
        The wavelength grid values is aligned with along its last axis
        (Angstrom) -- an existing table's own top-level "wl" dataset.
        Unlike line_filter_interp, no line filtering is applied here:
        values is already line-stripped/Q(HI)-normalized, so this is
        purely a re-gridding.
    values : numpy.ndarray
        1-D (galaxy) or 2-D (cluster, one row per output time)
        spectra.
    new_wl : numpy.ndarray
        The table's own new (possibly longer) merged wavelength grid.

    Returns
    -------
    numpy.ndarray
        values re-interpolated onto new_wl, 0 outside old_wl's own
        range. If new_wl equals old_wl -- the expected common case,
        since cloudy's own grid is stable across runs at fixed nII/U
        (see this module's own docstring) -- this is an exact copy.
    """
    if old_wl.shape == new_wl.shape and np.array_equal(old_wl, new_wl):
        return values.copy()
    return np.apply_along_axis(lambda row: np.interp(new_wl, old_wl, row, left=0.0, right=0.0), -1, values)


def _remap_lines(old_line_wl: np.ndarray, values: np.ndarray, new_line_index: dict[float, int]) -> np.ndarray:
    """
    Re-index an already-normalized line-luminosity array (or stack of them) onto a new common line list.

    Parameters
    ----------
    old_line_wl : numpy.ndarray
        The line wavelength list values is aligned with along its last
        axis (Angstrom) -- an existing table's own top-level "line_wl"
        dataset.
    values : numpy.ndarray
        1-D (galaxy) or 2-D (cluster) line luminosities.
    new_line_index : dict mapping float to int
        wavelength (Angstrom) -> column index into the table's own new
        merged line list; always a superset of old_line_wl, since
        compute_global_grids folds an existing table's own line_wl
        into the merged line list unconditionally.

    Returns
    -------
    numpy.ndarray
        values re-indexed onto the new merged line list, 0 for any new
        line old_line_wl never had (there are no lines old_line_wl has
        that new_line_index lacks, by the same superset guarantee).
    """
    new_values = np.zeros(values.shape[:-1] + (len(new_line_index),))
    for old_i, w in enumerate(old_line_wl):
        new_values[..., new_line_index[w]] = values[..., old_i]
    return new_values


def _remap_time(old_time: np.ndarray, values: np.ndarray, time_index: dict[float, int]) -> np.ndarray:
    """
    Re-index a cluster-type (ntime_old, ...) array onto the table's own new merged output-time grid.

    Parameters
    ----------
    old_time : numpy.ndarray
        The output times (yr) values's own first axis is aligned with
        -- an existing table's own top-level "time" dataset.
    values : numpy.ndarray
        A cluster-type "spec" or "line_lum" array, one row per entry
        in old_time.
    time_index : dict mapping float to int
        output time (yr) -> row index into the table's own new merged
        cluster output-time list; always a superset of old_time, for
        the same reason as _remap_lines's own new_line_index.

    Returns
    -------
    numpy.ndarray
        values re-indexed onto the new merged time grid, 0 for any
        output time old_time never had.
    """
    new_values = np.zeros((len(time_index),) + values.shape[1:])
    for old_i, t in enumerate(old_time):
        new_values[time_index[t], ...] = values[old_i, ...]
    return new_values


def _remap_combo_sim(existing: ExistingTable, data: dict[str, np.ndarray], sim_type: str,
    global_wl: np.ndarray, line_index: dict[float, int], time_index: dict[float, int]) -> dict[str, np.ndarray]:
    """
    Remap one existing "cluster" or "galaxy" group's own spec/line_lum onto a new build's own merged grids.

    Parameters
    ----------
    existing : ExistingTable
        The table data was read from -- carries the old wl/line_wl/
        time grids data's own arrays are still aligned to.
    data : dict
        {"spec": ..., "line_lum": ...} for one (combo, log10(U),
        sim_type), as stored in ExistingTable.combos.
    sim_type : {"cluster", "galaxy"}
        Which kind of group this is -- cluster's own arrays carry an
        extra leading output-time axis that also needs remapping
        (galaxy has none: a galaxy-type file has only ever had a
        single output time, so there is nothing to index on).
    global_wl, line_index, time_index
        This build_table call's own merged grids -- see _remap_wl,
        _remap_lines, _remap_time.

    Returns
    -------
    dict
        {"spec": ..., "line_lum": ...}, remapped onto global_wl/
        line_index (and, for a cluster group, time_index too).
    """
    spec = _remap_wl(existing.wl, data["spec"], global_wl)
    line_lum = _remap_lines(existing.line_wl, data["line_lum"], line_index)
    if sim_type == "cluster":
        spec = _remap_time(existing.time, spec, time_index)
        line_lum = _remap_time(existing.time, line_lum, time_index)
    return {"spec": spec, "line_lum": line_lum}


def _write_remapped_combo(fout: h5py.File, combo: tuple[str, float, float],
    logu_data: dict[float, dict[str, dict[str, np.ndarray]]], existing: ExistingTable,
    global_wl: np.ndarray, line_index: dict[float, int], time_index: dict[float, int]) -> None:
    """
    Write one (track, [Fe/H], v/vcrit) combination carried forward unchanged from an existing table.

    Parameters
    ----------
    fout : h5py.File
        The new table currently being written.
    combo : tuple of (str, float, float)
        (track, [Fe/H], v/vcrit) this combination's own data belongs
        to.
    logu_data : dict
        This combo's own entry in existing.combos -- {log10(U):
        {"cluster" or "galaxy": {"spec": ..., "line_lum": ...}}}, still
        aligned to existing's own wl/line_wl/time.
    existing : ExistingTable
        The table logu_data was read from (see _remap_combo_sim).
    global_wl, line_index, time_index
        This build_table call's own merged grids (see
        _remap_combo_sim).
    """
    track, feh, vvcrit = combo
    track_group = fout.require_group(track)
    track_group.attrs["track"] = track
    feh_group = track_group.require_group(f"FeH{feh:+.4f}")
    feh_group.attrs["FeH"] = feh
    vvcrit_group = feh_group.require_group(f"vvcrit{vvcrit:.2f}")
    vvcrit_group.attrs["v_vcrit"] = vvcrit

    for log_u, sim_data in sorted(logu_data.items()):
        logu_group = vvcrit_group.require_group(f"logU{log_u:+.2f}")
        logu_group.attrs["logU"] = log_u
        for sim_type, data in sim_data.items():
            remapped = _remap_combo_sim(existing, data, sim_type, global_wl, line_index, time_index)
            _write_normalized_group(logu_group, sim_type, remapped)


class _OutputTableLock:
    """
    An exclusive, cross-process lock on one output_path, held for build_table's entire read/merge/write.

    Two build_table calls racing against the same output_path (e.g. two
    concurrent pipeline runs sharing one --output) would otherwise be
    unsafe in a way this module's own pre-merge behavior never was: the
    later call's own read_existing_table (see build_table) can capture
    output_path's on-disk state from *before* the earlier call's own
    h5py.File(output_path, "w") lands, so the later call's write then
    completely replaces the file with a merge based on that stale
    snapshot -- silently discarding every combination the earlier call
    had just added. Even with --overwrite (no read step at all), two
    processes opening the same path "w" concurrently is still unsafe at
    the HDF5/filesystem level.

    Implemented via a sidecar output_path-plus-".lock" file created with
    os.O_CREAT | os.O_EXCL -- atomic on every filesystem this pipeline
    runs on, including the Lustre scratch filesystem --output normally
    lives on -- deliberately *not* flock/fcntl locking, which
    run_grid_pipeline.pbs's own HDF5_USE_FILE_LOCKING=FALSE setting
    documents as unreliable on that same Lustre filesystem. This lock
    fails fast rather than blocking-and-waiting: a long-running batch
    pipeline stuck waiting on a lock (rather than erroring loudly) risks
    quietly burning an entire job's own walltime/core-hour allocation
    for nothing, which is worse than just telling the second invocation
    to go away and rerun once the first one finishes.
    """

    def __init__(self, output_path: Path) -> None:
        self._lock_path = output_path.with_name(output_path.name + ".lock")

    def __enter__(self) -> "_OutputTableLock":
        try:
            fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"build_table: {self._lock_path} already exists -- either another process is "
                f"currently building/merging {self._lock_path.with_name(self._lock_path.name.removesuffix('.lock'))} "
                "right now, or a previous call crashed while holding this lock; if the latter, "
                "delete the lock file by hand and retry.") from None
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock_path.unlink(missing_ok=True)


def build_table(entries: list[FileEntry], output_path: Path, log_u_values: list[float], nline: int,
    overwrite: bool = False) -> None:
    """
    Build the full nebular emission lookup table from every discovered slug output file.

    Parameters
    ----------
    entries : list of FileEntry
        As returned by discover_files.
    output_path : pathlib.Path
        Path to write the table to.
    log_u_values : list of float
        Nominal log10(U) grid to sort rows into (see
        _assign_nominal_log_u); must match run_cloudy_grid.py's own
        --log-u, in the same order it was passed there.
    nline : int
        Maximum number of lines to keep per row, ranked by luminosity
        (see _row_top_lines); a value <= 0 keeps every line.
    overwrite : bool, default False
        If True, build output_path from entries alone, discarding
        whatever it already had (the old, pre-merge behavior). If
        False (the default), read back an existing file at output_path
        first (see read_existing_table) and merge: every (track,
        [Fe/H], v/vcrit) combination it already had is carried forward
        into the new output_path, remapped onto this call's own merged
        wl/line/time grids (see _write_remapped_combo), *except* a
        combination entries actually produces fresh cloudy results
        for -- that one is instead (re)computed fresh from entries,
        entirely replacing (not merging into) whatever it already had,
        on the assumption that producing fresh results at all means it
        was deliberately (re)run. A combination entries *covers* but
        that yields no cloudy results at all (e.g. every cloudy run
        for it failed) is not "fresh" by this definition, so whatever
        it already had in output_path survives untouched rather than
        being dropped. Has no effect if output_path does not exist
        yet.

    Raises
    ------
    RuntimeError
        If another build_table call already holds output_path's own
        lock file (see _OutputTableLock) -- either a genuinely
        concurrent call against the same output_path, or one left
        behind by an earlier call that crashed mid-build.
    """
    with _OutputTableLock(output_path):
        existing = None if overwrite else read_existing_table(output_path)

        global_wl_q, global_line_wl, global_line_label, global_time = compute_global_grids(
            entries, nline, existing)
        global_wl = global_wl_q.to_value(u.AA)
        line_index = {w: i for i, w in enumerate(global_line_wl)}
        time_index = {t: i for i, t in enumerate(global_time)}

        by_track_feh_vvcrit: dict[tuple[str, float, float], dict[str, Path]] = {}
        for path, sim_type, track, feh, vvcrit in entries:
            by_track_feh_vvcrit.setdefault((track, feh, vvcrit), {})[sim_type] = path

        with h5py.File(output_path, "w") as fout:
            _write_top_level(fout, global_wl_q, global_line_wl, global_line_label, global_time)

            # Fresh combinations are written first, tracking which ones
            # actually produced cloudy results, so the existing.combos
            # carry-forward loop below can tell "entries covers this combo
            # but every cloudy run for it failed" (existing data must
            # survive) apart from "entries produced fresh data for this
            # combo" (existing data is superseded) -- by_track_feh_vvcrit
            # alone conflates the two, since a combo with zero results
            # still has an entry there.
            written: set[tuple[str, float, float]] = set()
            for (track, feh, vvcrit), paths in sorted(by_track_feh_vvcrit.items()):
                cluster_path = paths.get("cluster")
                galaxy_path = paths.get("galaxy")

                cluster_data = (process_cluster_file(cluster_path, global_wl, line_index, time_index,
                    log_u_values, nline) if cluster_path is not None else {})
                for bucket in cluster_data.values():
                    _fill_bounded_gaps(bucket)
                galaxy_data = (process_galaxy_file(galaxy_path, global_wl, line_index, log_u_values, nline)
                    if galaxy_path is not None else {})

                log_us = sorted(set(cluster_data) | set(galaxy_data))
                if not log_us:
                    print(f"  {track} FeH={feh:+.4f} vvcrit={vvcrit:.2f}: no cloudy results found; "
                        "skipping", file=sys.stderr)
                    continue

                written.add((track, feh, vvcrit))
                track_group = fout.require_group(track)
                track_group.attrs["track"] = track
                feh_group = track_group.require_group(f"FeH{feh:+.4f}")
                feh_group.attrs["FeH"] = feh
                vvcrit_group = feh_group.require_group(f"vvcrit{vvcrit:.2f}")
                vvcrit_group.attrs["v_vcrit"] = vvcrit

                for log_u in log_us:
                    logu_group = vvcrit_group.require_group(f"logU{log_u:+.2f}")
                    logu_group.attrs["logU"] = log_u
                    if log_u in galaxy_data:
                        _write_normalized_group(logu_group, "galaxy", galaxy_data[log_u])
                    if log_u in cluster_data:
                        _write_normalized_group(logu_group, "cluster", cluster_data[log_u])

            if existing is not None:
                for combo, logu_data in sorted(existing.combos.items()):
                    if combo in written:
                        continue  # entries produced fresh data for this combo instead
                    _write_remapped_combo(fout, combo, logu_data, existing, global_wl, line_index, time_index)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process every cloudy run run_cloudy_grid.py produced in a working "
            "directory into a single nebular emission lookup table.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
        help=f"Directory to look for slug *.h5 output files in (default: {DEFAULT_WORK_DIR}); "
            "should be the same --work-dir make_slug_grid.py and run_cloudy_grid.py were run with.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Path to write the lookup table to (default: {DEFAULT_OUTPUT}). By default, if this "
            "already exists, it is merged into rather than replaced -- see this module's own "
            "docstring, and --overwrite to disable that.")
    parser.add_argument("--overwrite", action="store_true",
        help="Build --output from --work-dir alone, discarding whatever it already had, instead "
            "of merging into it (the default -- see this module's own docstring).")
    parser.add_argument("--log-u", type=float, nargs="+", default=list(LOG_U_VALUES),
        help=f"Nominal log10(U) grid to sort rows into (default: {list(LOG_U_VALUES)}); "
            "should match run_cloudy_grid.py's own --log-u.")
    parser.add_argument("--nline", type=int, default=DEFAULT_NLINE,
        help="Maximum number of lines to keep per row, ranked by luminosity (default: "
            f"{DEFAULT_NLINE}); a value <= 0 keeps every line. Since the brightest lines "
            "generally differ from one row to the next, the table's own overall line count can "
            "still exceed this (see _row_top_lines).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = discover_files(args.work_dir)
    if not entries:
        print(f"No .h5 files found in {args.work_dir}")
        return

    build_table(entries, args.output, args.log_u, args.nline, overwrite=args.overwrite)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
