#!/usr/bin/env python3
"""Import a directory of per-mass nucleosynthetic yield tables into
data/yields/<name>.h5 and data/yields/yields.toml.

Like data/tools/extinct/add_extinction_curve.py, most yield data slug
uses does not exist in a publicly-fetchable digital form -- it is
added by hand from files supplied via private communication, each
documented with its own literature reference. This script's own input
format is one specific such source layout: a directory containing one
plain-text file per progenitor mass, named ``s<mass>.yield_table``
(e.g. ``s18.2.yield_table`` for an 18.2 Msun progenitor), each holding
a whitespace-delimited table with a header row

    [isotope]    [ejecta]     [wind]

(the ``[ejecta]`` column is entirely absent for a failed supernova
that never explodes -- see below) followed by one row per isotope,
formatted as a one- or two-letter element symbol immediately followed
by its mass number with no separator (e.g. ``fe56``), then the mass
(Msun) of that isotope returned over the star's life via each channel.
Not every isotope appears in every file; a combination of isotope,
channel, and mass this script never actually sees a row for is written
as a yield of exactly zero, not left out -- including every isotope's
own ejecta yield for a mass whose own file has no ``[ejecta]`` column
at all (a failed supernova, which returns nothing via that channel by
construction, as opposed to simply not having been measured). The wind
column is used as-is for the ``--wind-channel`` (e.g.
massive_star_winds), and the ejecta column for ``--ejecta-channel``
(e.g. ccsn) -- see this project's own documentation for why the actual
timing of wind mass return within the star's life is approximated as
happening all at once, when the star dies.

The output HDF5 file holds top-level ``reference``/``reference_url``
attributes, then one group per channel (``--ejecta-channel``,
``--wind-channel``), each with its own ``masses`` (Msun, ascending),
``isotope_z`` (atomic number), and ``isotope_a`` (mass number) 1D
datasets -- shared across every Fe/H this channel's group holds, and
identical between the two channel groups here, since both are read off
the very same per-mass files -- plus one ``feh_<value>`` subgroup per
distinct ``--feh`` value the script has ever been run with for this
channel (only one, here, since this particular data set is Solar
metallicity only), holding its own ``Fe_H`` attribute and a 2D
``yield`` dataset (n_isotopes x n_masses, Msun) for that channel at
that metallicity. yields.toml is updated to match: a top-level
``channels`` list, a ``models`` list under each channel table naming
every model available for it, and a ``[<channel>.<name>]`` table per
channel/model pair giving its own reference/reference_url/file/Fe_H/
masses. As with extinct.toml (see add_extinction_curve.py's own
docstring), only the channel(s)/model this run actually touches are
ever modified -- everything else already in yields.toml/<name>.h5 is
left exactly as it was.

Run from the repository root, e.g. to import the Sukhbold et al. (2016)
core-collapse yields this script was first written for:

    python3 data/tools/yields/import_yield_tables.py \\
        --input-dir /path/to/SNII_Sukhbold16 \\
        --name sukhbold16 \\
        --reference "Sukhbold, T., Ertl, T., Woosley, S. E., Brown, J. M., Janka, H.-T. 2016, ApJ, 821, 38" \\
        --reference-url "https://ui.adsabs.harvard.edu/abs/2016ApJ...821...38S/abstract" \\
        --feh 0.0

:copyright: Copyright (c) 2026 Mark Krumholz
"""
import argparse
import pathlib
import re

import h5py
import numpy as np
import tomlkit

H5_DIR = "data/yields"
REGISTRY_PATH = "data/yields/yields.toml"

# Atomic symbols for all 118 elements, ordered by atomic number -- see
# src/elem/ElemCommons.hpp's own identical Symbols enum, which this
# mirrors so that an isotope's Z here always matches what the C++ side
# would assign the same element symbol.
ELEMENT_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
SYMBOL_TO_Z = {symbol.lower(): z for z, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}

# e.g. "fe56" -> ("fe", "56"); "h1" -> ("h", "1")
_ISOTOPE_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")
# e.g. "s18.2.yield_table" -> "18.2"; "s100.yield_table" -> "100"
_FILENAME_RE = re.compile(r"^s(\d+\.?\d*)\.yield_table$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
        help="Directory containing s<mass>.yield_table files")
    p.add_argument("--name", required=True,
        help="Name of this yield model, e.g. sukhbold16")
    p.add_argument("--reference", required=True, nargs="+",
        help="One or more literature references, e.g. "
             "'Sukhbold, T., et al. 2016, ApJ, 821, 38'; written as a "
             "plain string if exactly one is given, or a TOML array of "
             "strings otherwise")
    p.add_argument("--reference-url", nargs="*", default=[],
        help="URL(s) for --reference, e.g. ADS abstract link(s); same "
             "scalar-or-array rule as --reference, and independently "
             "sized -- a reference with no URL (e.g. a personal "
             "communication) is simply omitted here")
    p.add_argument("--feh", type=float, required=True,
        help="The [Fe/H] value every file in --input-dir corresponds "
             "to (this source format has no way to encode more than "
             "one [Fe/H] per directory; run this script again, once "
             "per [Fe/H], for a source that has several)")
    p.add_argument("--ejecta-channel", default="ccsn",
        help="Nucleosynthetic channel name for the ejecta column "
             "(default: ccsn)")
    p.add_argument("--wind-channel", default="massive_star_winds",
        help="Nucleosynthetic channel name for the wind column "
             "(default: massive_star_winds)")
    p.add_argument("--h5-dir", default=H5_DIR,
        help=f"Directory to write <name>.h5 into (default: {H5_DIR})")
    p.add_argument("--registry", default=REGISTRY_PATH,
        help=f"Path to the yields registry TOML file (default: {REGISTRY_PATH})")
    return p.parse_args()


def scalar_or_list(values: list[str]) -> str | list[str]:
    """A single value collapses to a plain string; more than one stays
    a list, written as a TOML/HDF5-attribute array."""
    return values[0] if len(values) == 1 else list(values)


def parse_mass_from_filename(path: pathlib.Path) -> float:
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"{path}: filename does not match s<mass>.yield_table")
    return float(match.group(1))


def parse_yield_table(path: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Parse one s<mass>.yield_table file.

    Returns
    -------
    dict mapping each isotope symbol (as given in the file, e.g.
    "fe56") to (ejecta, wind) in Msun -- ejecta is 0.0 for every
    isotope if this file has no [ejecta] column at all (a failed
    supernova).
    """
    with open(path) as f:
        lines = f.readlines()
    header = lines[0].split()
    if header == ["[isotope]", "[ejecta]", "[wind]"]:
        has_ejecta = True
    elif header == ["[isotope]", "[wind]"]:
        has_ejecta = False
    else:
        raise ValueError(
            f"{path}: unrecognized header {header!r}; expected "
            "['[isotope]', '[ejecta]', '[wind]'] or ['[isotope]', '[wind]']")

    table: dict[str, tuple[float, float]] = {}
    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        isotope = tokens[0]
        if has_ejecta:
            ejecta, wind = float(tokens[1]), float(tokens[2])
        else:
            ejecta, wind = 0.0, float(tokens[1])
        table[isotope] = (ejecta, wind)
    return table


def isotope_z_a(isotope: str) -> tuple[int, int]:
    """Resolve an isotope symbol like "fe56" to (Z, A)."""
    match = _ISOTOPE_RE.match(isotope)
    if match is None:
        raise ValueError(f"unrecognized isotope symbol: {isotope!r}")
    symbol, mass_number = match.groups()
    z = SYMBOL_TO_Z.get(symbol.lower())
    if z is None:
        raise ValueError(f"unknown element symbol {symbol!r} (isotope {isotope!r})")
    return z, int(mass_number)


def import_directory(input_dir: str) -> dict:
    """Parse every s<mass>.yield_table file in input_dir.

    Returns
    -------
    dict with "masses" (ascending, Msun), "isotope_z"/"isotope_a"
    (sorted by (Z, A)), and "ejecta_yield"/"wind_yield" (each
    n_isotopes x n_masses, Msun) -- the four arrays needed to populate
    both channel groups this script writes.
    """
    files = sorted(pathlib.Path(input_dir).glob("s*.yield_table"))
    if not files:
        raise ValueError(f"no s*.yield_table files found in {input_dir}")

    tables_by_mass: dict[float, dict[str, tuple[float, float]]] = {}
    all_isotopes: set[str] = set()
    for path in files:
        mass = parse_mass_from_filename(path)
        table = parse_yield_table(path)
        tables_by_mass[mass] = table
        all_isotopes.update(table.keys())

    masses = sorted(tables_by_mass.keys())
    isotopes = sorted(all_isotopes, key=isotope_z_a)
    # float64, not an integer dtype -- matching data/elem/isotopes.h5's
    # own Z/A datasets (see data/tools/elem/build_isotope_table.py), so
    # the C++ reader can use the same utils::readDataset1D() every
    # other float dataset in this project's own HDF5 files already
    # goes through, rather than needing a separate integer-dataset
    # reader just for this one case.
    isotope_z = np.array([isotope_z_a(iso)[0] for iso in isotopes], dtype=np.float64)
    isotope_a = np.array([isotope_z_a(iso)[1] for iso in isotopes], dtype=np.float64)
    isotope_index = {iso: i for i, iso in enumerate(isotopes)}
    mass_index = {m: j for j, m in enumerate(masses)}

    ejecta_yield = np.zeros((len(isotopes), len(masses)))
    wind_yield = np.zeros((len(isotopes), len(masses)))
    for mass, table in tables_by_mass.items():
        j = mass_index[mass]
        for iso, (ejecta, wind) in table.items():
            i = isotope_index[iso]
            ejecta_yield[i, j] = ejecta
            wind_yield[i, j] = wind

    return {
        "masses": np.array(masses),
        "isotope_z": isotope_z,
        "isotope_a": isotope_a,
        "ejecta_yield": ejecta_yield,
        "wind_yield": wind_yield,
    }


def write_channel_group(h5file: h5py.File, channel: str, masses: np.ndarray,
        isotope_z: np.ndarray, isotope_a: np.ndarray, feh: float,
        yield_: np.ndarray) -> None:
    """Write (or update) one channel's group: masses/isotope_z/isotope_a
    (replaced outright -- this run's own mass and isotope grid is
    authoritative for this channel) plus a feh_<value> subgroup for
    this specific --feh (replaced if this exact value already exists;
    every other feh_<value> subgroup already in this channel's group,
    if any, is left untouched).

    masses/isotope_z/isotope_a are shared across every feh_<value>
    subgroup in this channel: re-running this script for a second
    --feh against a source whose own mass/isotope grid differs from
    the first would otherwise silently leave that first --feh's own
    "yield" dataset shaped against a grid these three datasets no
    longer describe, so if this channel's group already holds at
    least one feh_<value> subgroup, the incoming grid is required to
    exactly match its existing masses/isotope_z/isotope_a -- a
    ValueError is raised rather than overwriting them out from under
    that other [Fe/H]'s own data. A channel with no feh_<value>
    subgroup yet (a first import, or a prior run that only got as far
    as raising this same error) has no existing grid to conflict with,
    so this run's own grid is written unconditionally in that case.
    """
    grp = h5file.require_group(channel)
    existing_feh_groups = [name for name in grp if name.startswith("feh_")]
    if existing_feh_groups:
        mismatched = [
            dset_name for dset_name, data in
            (("masses", masses), ("isotope_z", isotope_z), ("isotope_a", isotope_a))
            if not np.array_equal(grp[dset_name][()], data)
        ]
        if mismatched:
            raise ValueError(
                f"channel '{channel}' already holds {existing_feh_groups} on a "
                f"different {'/'.join(mismatched)} grid; this script does not "
                "support merging different grids for the same channel")

    for dset_name, data in (("masses", masses), ("isotope_z", isotope_z),
            ("isotope_a", isotope_a)):
        if dset_name in grp:
            del grp[dset_name]
        grp.create_dataset(dset_name, data=data, compression="gzip")

    feh_name = f"feh_{feh:g}"
    if feh_name in grp:
        del grp[feh_name]
    feh_grp = grp.create_group(feh_name)
    feh_grp.attrs["Fe_H"] = feh
    feh_grp.create_dataset("yield", data=yield_, compression="gzip")


def write_h5(h5_path: str, reference: list[str], reference_url: list[str],
        ejecta_channel: str, wind_channel: str, feh: float, imported: dict) -> None:
    pathlib.Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as h5file:
        h5file.attrs["reference"] = scalar_or_list(reference)
        if reference_url:
            h5file.attrs["reference_url"] = scalar_or_list(reference_url)

        write_channel_group(h5file, ejecta_channel, imported["masses"],
            imported["isotope_z"], imported["isotope_a"], feh, imported["ejecta_yield"])
        write_channel_group(h5file, wind_channel, imported["masses"],
            imported["isotope_z"], imported["isotope_a"], feh, imported["wind_yield"])


def update_registry(registry_path: str, h5_filename: str, name: str,
        reference: list[str], reference_url: list[str],
        channels: list[str], feh: float, masses: np.ndarray) -> None:
    reg_file = pathlib.Path(registry_path)
    if reg_file.exists():
        doc = tomlkit.parse(reg_file.read_text())
    else:
        doc = tomlkit.document()
        doc["name"] = "Registry of nucleosynthetic yield models"
        doc["channels"] = []

    channel_list = list(doc["channels"])
    for channel in channels:
        if channel not in channel_list:
            channel_list.append(channel)
    doc["channels"] = channel_list

    for channel in channels:
        channel_table = doc[channel] if channel in doc else tomlkit.table()
        models = list(channel_table["models"]) if "models" in channel_table else []
        # Read back before models/Fe_H are touched below, so a second
        # run adding a new [Fe/H] to an already-registered model sees
        # that model's own existing Fe_H list to merge into, rather
        # than the one this run is about to write over it with.
        existing_model = channel_table.get(name)

        if name not in models:
            models.append(name)
            models.sort()
        channel_table["models"] = models

        feh_list = (sorted(set(existing_model["Fe_H"]) | {feh})
            if existing_model is not None else [feh])

        model_table = tomlkit.table()
        model_table["reference"] = scalar_or_list(reference)
        if reference_url:
            model_table["reference_url"] = scalar_or_list(reference_url)
        model_table["file"] = h5_filename
        model_table["Fe_H"] = feh_list
        model_table["masses"] = [float(m) for m in masses]
        channel_table[name] = model_table
        doc[channel] = channel_table

    reg_file.write_text(tomlkit.dumps(doc))


def main() -> None:
    args = parse_args()

    imported = import_directory(args.input_dir)
    n_iso, n_mass = imported["ejecta_yield"].shape
    print(f"parsed {n_mass} masses, {n_iso} isotopes from {args.input_dir}")

    h5_filename = f"{args.name}.h5"
    h5_path = str(pathlib.Path(args.h5_dir) / h5_filename)
    write_h5(h5_path, args.reference, args.reference_url,
        args.ejecta_channel, args.wind_channel, args.feh, imported)
    print(f"wrote channels '{args.ejecta_channel}'/'{args.wind_channel}' to {h5_path}")

    update_registry(args.registry, h5_filename, args.name,
        args.reference, args.reference_url,
        [args.ejecta_channel, args.wind_channel], args.feh, imported["masses"])
    print(f"updated {args.registry}")


if __name__ == "__main__":
    main()
