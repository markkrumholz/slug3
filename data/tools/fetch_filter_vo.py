"""
Script to fetch photometric filter transmission curves from the
Spanish Virtual Observatory (SVO) Filter Profile Service, and write
them into a gzip'ed HDF5 file (data/filters/filters.h5 by default)
that slug can read, along with a human-readable TOML registry
(data/filters/filters.toml by default) describing what was fetched.
See http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php for the
service itself.

The service organizes filters in a three-level hierarchy -- facility
(e.g. "HST"), instrument (e.g. "ACS_WFC"), and filter (e.g. "F435W")
-- though not every facility actually has a distinct instrument level;
single-instrument facilities (e.g. "GALEX") go straight from facility
to filter. This script's --mode query lets a user browse that
hierarchy: with no --facility/--instrument/--filter given at all, it
lists every facility; with --facility given, it lists that facility's
instruments (or, for a single-instrument facility, its filters
directly, since there is no real instrument level to list); with
--facility and --instrument given, it lists that instrument's filters.

--mode fetch downloads the actual filter data -- metadata and
transmission curve -- for every filter matching the given (possibly
partial) facility/instrument/filter, writing each into --output
(data/filters/filters.h5 by default) under a group hierarchy that
mirrors the FPS's own facility/instrument/filter nesting, and
rebuilding --registry (data/filters/filters.toml by default) to
reflect --output's complete current contents afterwards. An absent
facility/instrument/filter matches every value at that level -- e.g.
--facility HST --instrument ACS_WFC (no --filter) fetches every
ACS_WFC filter; omitting --facility entirely fetches every filter
from every facility, which is a very large, slow download.
"""

# Imports
import argparse
import h5py
import numpy as np
import re
import shutil
import tomlkit
import urllib3
import xml.etree.ElementTree as ET

# Magic strings
FILTER_VO_URL = "http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php"
# The FPS's documented VOTable API for one specific filter's own metadata
# and transmission curve -- distinct from FILTER_VO_URL, which is the
# (undocumented) HTML "browse" interface list_facilities/list_instruments/
# list_filters below walk to discover what filters actually exist
FILTER_VO_VOTABLE_URL = "http://svo2.cab.inta-csic.es/theory/fps/fps.php"
# The filter-level metadata fields fetched from each filter's VOTable and
# stored as like-named HDF5 group attributes -- see fetch_filter_votable
# and write_filter. Split by value type (string/float/int) since each
# needs different casting when read back from the VOTable's own PARAM
# elements (all of which are themselves just XML attribute strings).
FILTER_VO_STR_ATTRS = ["Description", "MagSys", "ZeroPointUnit", "ZeroPointType"]
FILTER_VO_NUM_ATTRS = ["WavelengthRef", "WavelengthMin", "WavelengthMax", "ZeroPoint"]
# DetectorType: 0 (energy counter) or 1 (photon counter) -- an integer
# code, unlike FILTER_VO_NUM_ATTRS' continuous quantities, so cast and
# stored separately from them
FILTER_VO_INT_ATTRS = ["DetectorType"]
FILTER_VO_references = [
    "Rodrigo, C., Cruz, P., Aguilar, J.F., et al. 2024, A&A, 689, 93",
    "Rodrigo, C., & Solano, E. 2020, XIV.0 Scientific Meeting of the Spanish Astronomical Society",
    "Rodrigo, C., Solano, E., & Bayo, A., 2012, IVOA Tech Rept, 15 October",
]
FILTER_VO_reference_urls = [
    "https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..93R/abstract",
    "https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract",
    "https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract",
]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Fetch photometric filter transmission curves from the SVO Filter Profile Service")
parser.add_argument("mode", choices=["query", "fetch"],
                    help="'query' to list available facilities/instruments/filters; "
                         "'fetch' to download filter data")
parser.add_argument("--facility", default=None,
                    help="Facility to query/fetch (e.g. 'HST'); if unspecified, "
                         "matches every facility")
parser.add_argument("--instrument", default=None,
                    help="Instrument to query/fetch (e.g. 'ACS_WFC'); requires --facility. "
                         "If unspecified, matches every instrument of --facility. Not every "
                         "facility has a distinct instrument level -- see this script's own "
                         "module docstring")
parser.add_argument("--filter", default=None,
                    help="Filter to query/fetch (e.g. 'F435W'); requires --instrument. "
                         "If unspecified, matches every filter of --instrument")
parser.add_argument("--output",
                    default=shutil.os.path.join("..", "filters", "filters.h5"),
                    help="Output file for the HDF5 filter data (--mode fetch only; "
                         "default: %(default)s)")
parser.add_argument("--registry",
                    default=shutil.os.path.join("..", "filters", "filters.toml"),
                    help="Filter registry TOML file (--mode fetch only; "
                         "default: %(default)s)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite filters already present in --output "
                         "(--mode fetch only)")
args = parser.parse_args()

if args.instrument and not args.facility:
    parser.error("--instrument requires --facility to also be specified")
if args.filter and not args.instrument:
    parser.error("--filter requires --instrument (and --facility) to also be specified")

# ---------------------------------------------------------------------------
# Filter Profile Service "browse" page fetching/parsing
# ---------------------------------------------------------------------------
#
# The FPS has no documented machine-readable API for browsing the
# facility/instrument/filter hierarchy itself (as opposed to fetching
# one specific filter's own data, which does have a documented VOTable
# API -- see --mode fetch, not yet implemented). Instead, this walks
# the same HTML "browse" page a human would (index.php?mode=browse,
# scoped via &gname=<facility>&gname2=<instrument>), extracting the
# facility/instrument/filter names it links to via regexes tied to
# that page's actual (undocumented, and so possibly fragile to a
# future redesign) link structure -- confirmed by hand against a
# representative sample of facilities, including several edge cases
# (a facility with no distinct instrument level at all, e.g. GALEX; a
# facility whose sole instrument isn't named after the facility
# itself, e.g. Astrosat/UVIT; and a facility with no filters in the
# FPS at all, e.g. Chandra).


def fetch_browse_page(facility: str | None = None, instrument: str | None = None) -> str:
    """Fetch one FPS "browse" HTML page, scoped to facility/instrument if given.
    """
    fields = {"mode": "browse", "asttype": ""}
    if facility:
        fields["gname"] = facility
    if instrument:
        fields["gname2"] = instrument
    http = urllib3.PoolManager()
    resp = http.request("GET", FILTER_VO_URL, fields=fields)
    if resp.status != 200:
        raise RuntimeError(f"Failed to fetch {FILTER_VO_URL} (fields={fields}): "
                           f"HTTP {resp.status}")
    # The page declares charset=ISO-8859-1; decoding as such (rather than the
    # more usual utf-8) avoids spurious decode errors on any non-ASCII byte
    # in, e.g., a facility's free-text description -- none of the tokens
    # this module actually extracts (facility/instrument/filter names) are
    # ever non-ASCII themselves.
    return resp.data.decode("iso-8859-1")


def list_facilities(html: str) -> list[str]:
    """Parse the top-level (no facility/instrument) browse page for every facility name.
    """
    return sorted(set(re.findall(
        r'index\.php\?mode=browse&gname=([^&"]+)&asttype=', html)))


def list_instruments(html: str, facility: str) -> list[str]:
    """Parse a facility's browse page for its distinct instrument names, if any.

    A facility with more than one instrument has an explicit selector
    link for each one, of the form
    "...mode=browse&gname=<facility>&gname2=<instrument>&asttype=...".
    A facility with only one instrument has no such selector (there is
    nothing to select between) -- that instrument's name instead only
    ever appears embedded in each of its own filters' own links, of
    the form "...mode=browse&gname=<facility>&gname2=<instrument>#filter".
    Matching up to whichever of "&" or "#" comes first, rather than
    requiring the "&asttype=" suffix specifically, catches both forms
    uniformly.
    """
    pattern = r'mode=browse&gname=' + re.escape(facility) + r'&gname2=([^&"#]+)'
    return sorted(set(re.findall(pattern, html)))


def list_filters(html: str) -> list[str]:
    """Parse a facility[/instrument]-scoped browse page for every filter ID it links to.

    Returns each ID with its leading "facility/" stripped (e.g.
    "ACS_WFC.F435W" rather than "HST/ACS_WFC.F435W"), since the
    facility is already implied by which page was fetched.
    """
    ids = set(re.findall(r'index\.php\?id=([^&"]+)&&mode=browse', html))
    return sorted(filter_id.split("/", 1)[1] for filter_id in ids if "/" in filter_id)


def query(facility: str | None, instrument: str | None, filter_name: str | None) -> None:
    """Print whatever is one level beyond the most specific of facility/instrument/filter given.
    """
    if not facility:
        print("Facilities = " + ", ".join(list_facilities(fetch_browse_page())))
        return

    facility_html = fetch_browse_page(facility=facility)
    instruments = list_instruments(facility_html, facility)

    if not instrument:
        # More than one distinct instrument: that's the next level to show.
        # Otherwise -- zero (e.g. Chandra, which the FPS carries with no
        # filters at all) or one (e.g. GALEX, Astrosat) -- there is no real
        # instrument level to stop at, so go straight to filters, scoped by
        # this same facility-only page (which, in both of those cases,
        # already links to nothing but that single implicit instrument's
        # own filters, if it has any).
        if len(instruments) > 1:
            print("Instruments = " + ", ".join(instruments))
        else:
            filters = list_filters(facility_html)
            if filters:
                print("Filters = " + ", ".join(filters))
            else:
                print(f"No filters found for facility '{facility}'.")
        return

    if instrument not in instruments:
        available = ", ".join(instruments) if instruments else "(none)"
        raise RuntimeError(
            f"'{instrument}' is not a valid instrument for facility '{facility}'; "
            f"available instruments are: {available}")

    filters = list_filters(fetch_browse_page(facility=facility, instrument=instrument))

    if not filter_name:
        if filters:
            print("Filters = " + ", ".join(filters))
        else:
            print(f"No filters found for facility '{facility}', instrument '{instrument}'.")
        return

    # Accept either the bare filter code (e.g. "F435W", matching --instrument
    # in spirit) or the full "instrument.filter" form list_filters() itself
    # returns (e.g. "ACS_WFC.F435W", matching what a user copies straight
    # out of this same script's own --instrument-only output) -- whichever
    # of the two actually matches.
    target = filter_name if filter_name in filters else f"{instrument}.{filter_name}"
    if target not in filters:
        available = ", ".join(filters) if filters else "(none)"
        raise RuntimeError(
            f"'{filter_name}' is not a valid filter for facility '{facility}', instrument "
            f"'{instrument}'; available filters are: {available}")
    print(f"Filter = {target}")


# ---------------------------------------------------------------------------
# Filter data fetching (--mode fetch)
# ---------------------------------------------------------------------------


def iter_matching_filters(facility: str | None, instrument: str | None, filter_name: str | None):
    """Yield (facility, instrument, filter) for every filter matching the given (possibly partial) query.

    An absent facility/instrument/filter matches every value at that
    level -- see this module's own docstring. Reuses list_facilities/
    list_instruments/list_filters (the same "browse" page parsing
    query() itself uses) to discover what actually exists, so this
    only ever yields real, fetchable filters.
    """
    facilities = [facility] if facility else list_facilities(fetch_browse_page())
    for fac in facilities:
        fac_html = fetch_browse_page(facility=fac)
        real_instruments = list_instruments(fac_html, fac)

        if instrument:
            if instrument not in real_instruments:
                available = ", ".join(real_instruments) if real_instruments else "(none)"
                raise RuntimeError(
                    f"'{instrument}' is not a valid instrument for facility '{fac}'; "
                    f"available instruments are: {available}")
            instruments = [instrument]
        else:
            instruments = real_instruments

        for instr in instruments:
            filters = list_filters(fetch_browse_page(facility=fac, instrument=instr))

            if filter_name:
                target = filter_name if filter_name in filters else f"{instr}.{filter_name}"
                if target not in filters:
                    available = ", ".join(filters) if filters else "(none)"
                    raise RuntimeError(
                        f"'{filter_name}' is not a valid filter for facility '{fac}', "
                        f"instrument '{instr}'; available filters are: {available}")
                matching = [target]
            else:
                matching = filters

            for full_id in matching:
                yield fac, instr, full_id.split(".", 1)[1]


def fetch_filter_votable(facility: str, instrument: str, filter_code: str):
    """Fetch and parse one filter's VOTable: its metadata PARAMs and (wavelength, transmission) columns.

    Returns a (params, wavelength, transmission) tuple: params is a
    dict of every PARAM element's raw string value, keyed by name
    (only FILTER_VO_STR_ATTRS/FILTER_VO_NUM_ATTRS are actually used by
    write_filter/update_registry below, but every PARAM is returned
    here regardless, in case a future caller needs one that isn't);
    wavelength and transmission are equal-length float arrays read
    from the VOTable's TABLEDATA rows, in the units documented in this
    module's own docstring (Angstrom, dimensionless).
    """
    filter_id = f"{facility}/{instrument}.{filter_code}"
    http = urllib3.PoolManager()
    resp = http.request("GET", FILTER_VO_VOTABLE_URL, fields={"ID": filter_id})
    if resp.status != 200:
        raise RuntimeError(f"Failed to fetch VOTable for {filter_id}: HTTP {resp.status}")

    root = ET.fromstring(resp.data)
    status = root.find(".//INFO[@name='QUERY_STATUS']")
    if status is not None and status.get("value") != "OK":
        raise RuntimeError(f"FPS query for {filter_id} did not succeed: "
                           f"QUERY_STATUS = {status.get('value')}")

    params = {param.get("name"): param.get("value") for param in root.iter("PARAM")}

    field_names = [field.get("name") for field in root.iter("FIELD")]
    wavelength = []
    transmission = []
    for row in root.iter("TR"):
        values = dict(zip(field_names, (td.text for td in row.iter("TD"))))
        wavelength.append(float(values["Wavelength"]))
        transmission.append(float(values["Transmission"]))

    return params, np.array(wavelength), np.array(transmission)


def write_filter(h5file: h5py.File, facility: str, instrument: str, filter_code: str,
                 overwrite: bool) -> None:
    """Fetch one filter's data and write it into h5file[facility][instrument][filter_code].

    Skips (rather than re-fetching) a filter already present in
    h5file unless overwrite is True, matching the other fetch_*.py
    scripts' own --overwrite convention. Always prints which filter
    it's working on (unlike the other fetch_*.py scripts' own
    progress messages, which are opt-in via --verbose) since fetching
    is one individual network request per filter -- potentially many
    of them, for a broad wildcard query -- rather than one bulk
    download followed by local processing, so a silent run here could
    otherwise sit looking hung for a long time.
    """
    instr_grp = h5file.require_group(facility).require_group(instrument)
    if filter_code in instr_grp and not overwrite:
        print(f"  Skipping {facility}/{instrument}.{filter_code}: already present.")
        return

    print(f"  Fetching {facility}/{instrument}.{filter_code} ...")
    params, wavelength, transmission = fetch_filter_votable(facility, instrument, filter_code)

    if filter_code in instr_grp:
        del instr_grp[filter_code]
    filt_grp = instr_grp.create_group(filter_code)

    # Metadata fields not present in this particular filter's VOTable (e.g.
    # ZeroPointType, which the FPS only supplies for some filters) are
    # simply left unset here, rather than written as some placeholder value
    for key in FILTER_VO_STR_ATTRS:
        if key in params:
            filt_grp.attrs[key] = params[key]
    for key in FILTER_VO_NUM_ATTRS:
        if key in params:
            filt_grp.attrs[key] = float(params[key])
    for key in FILTER_VO_INT_ATTRS:
        if key in params:
            filt_grp.attrs[key] = int(params[key])
    filt_grp.attrs["references"] = FILTER_VO_references
    filt_grp.attrs["reference_urls"] = FILTER_VO_reference_urls
    filt_grp.attrs["source"] = FILTER_VO_URL

    filt_grp.create_dataset("wavelength", data=wavelength, compression="gzip")
    filt_grp.create_dataset("transmission", data=transmission, compression="gzip")


def update_registry(output: str, registry: str) -> None:
    """Rebuild registry from scratch to reflect output's complete current contents.

    Unlike spectra.toml (shared across every fetch_*.py spectral
    library script, each of which only ever rewrites its own single
    top-level entry), filters.toml exists only for this script, and
    its entire structure -- not just one entry within it -- mirrors
    output's own facility/instrument/filter group hierarchy. So there
    is nothing else in filters.toml worth preserving across a rebuild;
    this always regenerates it wholesale from output's current
    contents, rather than reading back and patching any previous
    version of the file.
    """
    doc = tomlkit.document()
    doc["name"] = "Registry of filters"
    doc["file"] = output

    with h5py.File(output, "r") as h5file:
        facilities = sorted(h5file.keys())
        doc["Facilities"] = facilities

        for fac in facilities:
            fac_grp = h5file[fac]
            instruments = sorted(fac_grp.keys())
            fac_table = tomlkit.table()
            fac_table["instruments"] = instruments

            for instr in instruments:
                instr_grp = fac_grp[instr]
                filters = sorted(instr_grp.keys())
                instr_table = tomlkit.table()
                instr_table["filters"] = filters

                for filt in filters:
                    filt_attrs = instr_grp[filt].attrs
                    filt_table = tomlkit.table()
                    filt_table["description"] = filt_attrs.get("Description", "")
                    filt_table["wl_ref"] = float(filt_attrs.get("WavelengthRef", float("nan")))
                    filt_table["source"] = FILTER_VO_URL
                    instr_table[filt] = filt_table

                fac_table[instr] = instr_table

            doc[fac] = fac_table

    with open(registry, "w") as f:
        f.write(tomlkit.dumps(doc))


def fetch(facility: str | None, instrument: str | None, filter_name: str | None,
         output: str, registry: str, overwrite: bool) -> None:
    """Fetch every filter matching the given query into output, then rebuild registry.
    """
    # Unlike data/spectra/ (which every fetch_*.py script targets, and which
    # already exists in the repository), data/filters/ is new -- output's
    # and registry's own default parent directory won't exist yet on a
    # fresh checkout, so create both (a no-op if they're already there)
    # before trying to write either file.
    shutil.os.makedirs(shutil.os.path.dirname(output) or ".", exist_ok=True)
    shutil.os.makedirs(shutil.os.path.dirname(registry) or ".", exist_ok=True)

    with h5py.File(output, "a") as h5file:
        for fac, instr, filt in iter_matching_filters(facility, instrument, filter_name):
            write_filter(h5file, fac, instr, filt, overwrite)

    update_registry(output, registry)
    print(f"Updated registry at {registry}.")


if args.mode == "query":
    query(args.facility, args.instrument, args.filter)
else:
    fetch(args.facility, args.instrument, args.filter,
          args.output, args.registry, args.overwrite)
