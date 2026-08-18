"""
Unit tests for slugpy's HDF5 output readers (slug_reader,
slug_group_reader, slug_phot_reader), exercised against a real
committed output file, examples/clusterlib/clusterlib.h5, rather than
a small synthetic fixture: the reader classes have no logic of their
own to react to a particular deck, so a real file is both simpler to
obtain (it's already committed, for exactly this kind of use) and a
more representative check that the classes agree with the actual
layout OutputManagerH5 writes.

This file is run via pytest, invoked as a CTest test from
CMakeLists.txt (see the test_Slugpy target), with WORKING_DIRECTORY
set to the repo root, so CLUSTERLIB_H5 below resolves without needing
SLUG_DIR or any other environment variable.
"""

import pathlib
from typing import cast

import h5py
import numpy as np
import pytest
from astropy import units as u

from slugpy._slug import Filter, FilterCollection
from slugpy.slug_group_reader import slug_group_reader
from slugpy.slug_phot_reader import slug_phot_reader
from slugpy.slug_reader import slug_reader

REPO_ROOT = pathlib.Path.cwd()
CLUSTERLIB_DIR = REPO_ROOT / "examples" / "clusterlib"
CLUSTERLIB_H5 = "examples/clusterlib/clusterlib.h5"

# clusterlib.h5's own input_deck requests the real MIST tracks and the
# full "default" chained spectral library (see clusterlib.toml), so
# building a real SimControls from it -- as slug_reader.controls (and
# so get_cluster()) does -- needs every one of these files. They're
# fetched data (see data/tools/*/fetch_*.py), not committed to the repo,
# so are unavailable in CI -- mirrors tests/core/
# testClusterSpecsynFullCommon.cpp's own requiredDataFiles/
# requiredPhotDataFiles and allRequiredDataFilesExist(), which guard
# the C++ side's own equivalent full-chain tests the same way.
_REQUIRED_FULL_DATA_FILES = (
    "data/tracks/tracks.toml",
    "data/tracks/mist.h5",
    "data/spectra/spectra.toml",
    "data/spectra/powr_wc.h5",
    "data/spectra/powr_wne.h5",
    "data/spectra/powr_wnl_h20.h5",
    "data/spectra/powr_wnl_h40.h5",
    "data/spectra/powr_wnl_h60.h5",
    "data/spectra/tlusty_o.h5",
    "data/spectra/tlusty_b.h5",
    "data/spectra/bosz.h5",
    "data/spectra/ck04.h5",
    "data/spectra/tremblay_da.h5",
    "data/spectra/tremblay_elm.h5",
    "data/spectra/rauch.h5",
    "data/spectra/rauch_h07.h5",
    "data/filters/filters.toml",
    "data/filters/filters.h5",
)
_FULL_DATA_AVAILABLE = all((REPO_ROOT / p).exists() for p in _REQUIRED_FULL_DATA_FILES)
requires_full_data = pytest.mark.skipif(
    not _FULL_DATA_AVAILABLE,
    reason="requires the real MIST tracks + full spectral library chain + "
        "filter registry (fetched data, not committed to the repo)")


@pytest.fixture(scope="module")
def reader():
    """A slug_reader opened on the committed clusterlib.h5 example output."""
    return slug_reader(CLUSTERLIB_H5)


@pytest.fixture(scope="module")
def clusters(reader):
    """reader.clusters, the clusters group's slug_group_reader."""
    return reader.clusters


@pytest.fixture(scope="module")
def cluster_spectra(reader):
    """reader.cluster_spectra, the cluster_spectra group's slug_group_reader."""
    return reader.cluster_spectra


@pytest.fixture(scope="module")
def cluster_phot(reader):
    """reader.cluster_phot, the cluster_phot group's slug_phot_reader."""
    return reader.cluster_phot


# ---------------------------------------------------------------------
# slug_reader: top-level metadata
# ---------------------------------------------------------------------

def test_top_level_metadata(reader):
    """slug_hash/date/time/rng_state are all non-empty strings."""
    assert isinstance(reader.slug_hash, str) and len(reader.slug_hash) > 0
    assert isinstance(reader.date, str) and len(reader.date) > 0
    assert isinstance(reader.time, str) and len(reader.time) > 0
    assert isinstance(reader.rng_state, str) and len(reader.rng_state) > 0


# ---------------------------------------------------------------------
# slug_reader.input_deck
# ---------------------------------------------------------------------

def test_input_deck_contents(reader):
    """input_deck parses to a mapping with this example's own known values."""
    deck = reader.input_deck
    assert deck["sim_type"] == "cluster"
    assert deck["n_trial"] == 2000
    assert deck["output"]["model_name"] == "clusterlib"


def test_input_deck_cached(reader):
    """input_deck is read once and the same object returned thereafter."""
    assert reader.input_deck is reader.input_deck


def test_input_deck_readonly(reader):
    """Assigning to input_deck raises AttributeError."""
    with pytest.raises(AttributeError):
        reader.input_deck = None


# ---------------------------------------------------------------------
# slug_reader.clusters / cluster_spectra / cluster_phot: presence,
# laziness, caching, and read-only-ness, common to all three
# ---------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["clusters", "cluster_spectra", "cluster_phot"])
def test_group_property_present_and_cached(reader, attr):
    """Each group property returns a non-None reader, cached across accesses."""
    first = getattr(reader, attr)
    assert first is not None
    assert getattr(reader, attr) is first


@pytest.mark.parametrize("attr", ["clusters", "cluster_spectra", "cluster_phot"])
def test_group_property_readonly(reader, attr):
    """Assigning to any of the group properties raises AttributeError."""
    with pytest.raises(AttributeError):
        setattr(reader, attr, None)


def test_clusters_is_group_reader(clusters):
    """clusters is a plain slug_group_reader (not the phot subclass)."""
    assert isinstance(clusters, slug_group_reader)
    assert not isinstance(clusters, slug_phot_reader)


def test_cluster_spectra_is_group_reader(cluster_spectra):
    """cluster_spectra is a plain slug_group_reader."""
    assert isinstance(cluster_spectra, slug_group_reader)


def test_cluster_phot_is_phot_reader(cluster_phot):
    """cluster_phot is specifically a slug_phot_reader."""
    assert isinstance(cluster_phot, slug_phot_reader)


# ---------------------------------------------------------------------
# slug_reader.galaxy / galaxy_spectra / galaxy_phot
#
# clusterlib.h5 is a cluster-type simulation's output, so it has none
# of these groups; the "properties are None" case is checked directly
# against it below. There is no committed galaxy-type output file to
# mirror clusterlib.h5's own real-data role for the "properties are
# populated" case, so that case instead uses a small synthetic fixture
# built by hand, mirroring _make_phot_convert_test_file's own pattern
# elsewhere in this file -- the underlying slug_group_reader/
# slug_phot_reader classes these three properties build on are
# entirely unchanged (already exhaustively tested above, via clusters/
# cluster_spectra/cluster_phot), so this only needs to check the new
# accessor plumbing itself, not re-derive that same coverage.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["galaxy", "galaxy_spectra", "galaxy_phot"])
def test_galaxy_group_property_none_for_cluster_sim(reader, attr):
    """galaxy/galaxy_spectra/galaxy_phot are all None for a cluster-type file."""
    assert getattr(reader, attr) is None


def _make_galaxy_test_file(tmp_path):
    """A minimal galaxy-type output file, with galaxy, galaxy_spectra, and galaxy_phot groups."""
    path = tmp_path / "galaxy_test.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset("toml", data="n_trial = 1\nsim_type = \"galaxy\"\n")

        g = f.create_group("galaxy")
        trial = g.create_dataset("trial", data=np.array([0], dtype="u8"))
        trial.attrs["units"] = ""
        target_mass = g.create_dataset("target_mass", data=np.array([1e4]))
        target_mass.attrs["units"] = "Msun"

        gs = f.create_group("galaxy_spectra")
        wl = gs.create_dataset("wl", data=np.array([1000.0, 2000.0]))
        wl.attrs["units"] = "Angstrom"
        spec = gs.create_dataset("spec", data=np.array([[1.0, 2.0]]))
        spec.attrs["units"] = "erg/s/Angstrom"

        gp = f.create_group("galaxy_phot")
        gp.attrs["filters"] = ["Q(HI)"]
        phot = gp.create_dataset("phot", data=np.array([[1e47]]))
        phot.attrs["units"] = ["photon/s"]
        gp_trial = gp.create_dataset("trial", data=np.array([0], dtype="u8"))
        gp_trial.attrs["units"] = ""
    return path


def test_galaxy_group_property_present_and_cached(tmp_path):
    """galaxy is a slug_group_reader (not the phot subclass), cached across accesses."""
    reader = slug_reader(str(_make_galaxy_test_file(tmp_path)))
    first = reader.galaxy
    assert first is not None
    assert isinstance(first, slug_group_reader)
    assert not isinstance(first, slug_phot_reader)
    assert reader.galaxy is first
    assert cast(u.Quantity, first["target_mass"]).value == pytest.approx([1e4])


def test_galaxy_spectra_property_present_and_cached(tmp_path):
    """galaxy_spectra is a slug_group_reader, cached across accesses."""
    reader = slug_reader(str(_make_galaxy_test_file(tmp_path)))
    first = reader.galaxy_spectra
    assert first is not None
    assert isinstance(first, slug_group_reader)
    assert reader.galaxy_spectra is first
    assert cast(u.Quantity, first["wl"]).unit == u.AA


def test_galaxy_phot_property_present_and_cached(tmp_path):
    """galaxy_phot is specifically a slug_phot_reader, cached across accesses."""
    reader = slug_reader(str(_make_galaxy_test_file(tmp_path)))
    first = reader.galaxy_phot
    assert first is not None
    assert isinstance(first, slug_phot_reader)
    assert reader.galaxy_phot is first
    assert first.filters == ["Q(HI)"]
    phot = cast(u.Quantity, first["Q(HI)"])
    assert phot.unit.is_equivalent(u.photon / u.s)


@pytest.mark.parametrize("attr", ["galaxy", "galaxy_spectra", "galaxy_phot"])
def test_galaxy_group_property_readonly(tmp_path, attr):
    """Assigning to galaxy/galaxy_spectra/galaxy_phot raises AttributeError."""
    reader = slug_reader(str(_make_galaxy_test_file(tmp_path)))
    with pytest.raises(AttributeError):
        setattr(reader, attr, None)


# ---------------------------------------------------------------------
# slug_group_reader, via clusters
# ---------------------------------------------------------------------

def test_clusters_keys(clusters):
    """keys() lists every dataset actually written to the clusters group."""
    expected = {"trial", "uid", "target_mass", "birth_mass", "form_time", "feh", "rng", "A_V"}
    assert set(clusters.keys()) == expected


def test_clusters_unitful_dataset(clusters):
    """A dataset with a real physical unit comes back as a positive-valued Quantity."""
    masses = clusters["target_mass"]
    assert isinstance(masses, u.Quantity)
    assert masses.unit == u.Msun
    assert len(masses) == 2000
    assert np.all(masses > 0)


def test_clusters_empty_unit_dataset(clusters):
    """A dataset with an empty units attribute comes back as a plain array."""
    feh = clusters["feh"]
    assert isinstance(feh, np.ndarray)
    assert not isinstance(feh, u.Quantity)
    assert len(feh) == 2000


def test_clusters_non_numeric_dataset(clusters):
    """The non-numeric, empty-unit rng dataset also comes back as a plain array."""
    rng = clusters["rng"]
    assert isinstance(rng, np.ndarray)
    assert not isinstance(rng, u.Quantity)
    assert rng.dtype.kind == "S"


def test_clusters_getitem_cached(clusters):
    """Repeated access to the same dataset returns the identical cached object."""
    assert clusters["target_mass"] is clusters["target_mass"]


def test_clusters_getitem_unknown_key(clusters):
    """An unknown dataset name raises KeyError."""
    with pytest.raises(KeyError):
        clusters["not_a_real_dataset"]


# ---------------------------------------------------------------------
# slug_group_reader, via cluster_spectra (2D datasets)
# ---------------------------------------------------------------------

def test_cluster_spectra_keys(cluster_spectra):
    """keys() lists every dataset actually written to the cluster_spectra group.

    clusterlib.toml enables extinction, so wl_extinct is written
    alongside spec_extinct (see OutputManagerH5::openClusterSpectraGroup())."""
    expected = {"trial", "time", "uid", "wl", "wl_extinct", "spec", "spec_extinct"}
    assert set(cluster_spectra.keys()) == expected


def test_cluster_spectra_wl(cluster_spectra):
    """wl is a 1D wavelength Quantity in Angstrom."""
    wl = cluster_spectra["wl"]
    assert isinstance(wl, u.Quantity)
    assert wl.unit == u.AA
    assert len(wl) == 1024
    assert np.all(np.diff(wl.value) > 0)


def test_cluster_spectra_spec_2d(cluster_spectra, reader):
    """spec is a 2D flux-density Quantity, one row per trial, one column per wl point."""
    spec = cluster_spectra["spec"]
    wl = cluster_spectra["wl"]
    assert isinstance(spec, u.Quantity)
    assert spec.unit.is_equivalent(u.erg / u.s / u.AA)
    assert spec.shape == (reader.input_deck["n_trial"], len(wl))
    assert np.all(spec.value >= 0)
    assert not np.any(np.isnan(spec.value))


# ---------------------------------------------------------------------
# slug_phot_reader, via cluster_phot
# ---------------------------------------------------------------------

def test_cluster_phot_filters_match_input_deck(cluster_phot, reader):
    """filters/filter_units match this deck's own phot.filters, in order."""
    assert cluster_phot.filters == list(reader.input_deck["phot"]["filters"])
    assert len(cluster_phot.filter_units) == len(cluster_phot.filters)


# ---------------------------------------------------------------------
# slug_reader.filters / filter_units: aliases for cluster_phot's own,
# falling back to galaxy_phot's own when there is no cluster_phot
# ---------------------------------------------------------------------

def test_reader_filters_alias(reader, cluster_phot):
    """reader.filters is the identical object as reader.cluster_phot.filters."""
    assert reader.filters is cluster_phot.filters


def test_reader_filter_units_alias(reader, cluster_phot):
    """reader.filter_units is the identical object as reader.cluster_phot.filter_units."""
    assert reader.filter_units is cluster_phot.filter_units


@pytest.mark.parametrize("attr", ["filters", "filter_units"])
def test_reader_filters_readonly(reader, attr):
    """Assigning to reader.filters/filter_units raises AttributeError."""
    with pytest.raises(AttributeError):
        setattr(reader, attr, None)


@pytest.mark.parametrize("attr", ["filters", "filter_units"])
def test_reader_filters_falls_back_to_galaxy_phot(tmp_path, attr):
    """When a file has no cluster_phot group (as for a galaxy-type sim
    with cluster output disabled), reader.filters/filter_units fall
    back to galaxy_phot's own, rather than returning None."""
    galaxy_reader = slug_reader(str(_make_galaxy_test_file(tmp_path)))
    assert galaxy_reader.cluster_phot is None
    galaxy_phot = galaxy_reader.galaxy_phot
    assert galaxy_phot is not None
    assert getattr(galaxy_reader, attr) is getattr(galaxy_phot, attr)


def test_reader_filters_none_without_any_phot_group(tmp_path):
    """reader.filters/filter_units are None when a file has neither a
    cluster_phot nor a galaxy_phot group."""
    path = tmp_path / "no_phot.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset("toml", data="n_trial = 1\nsim_type = \"galaxy\"\n")
    no_phot_reader = slug_reader(str(path))
    assert no_phot_reader.filters is None
    assert no_phot_reader.filter_units is None


def test_cluster_phot_magnitude_filter(cluster_phot):
    """A Vega-system real filter comes back as a magnitude Quantity."""
    phot = cluster_phot["HST.WFC3_UVIS1.F555W"]
    assert isinstance(phot, u.Quantity)
    assert phot.unit == u.mag
    assert len(phot) == 2000


def test_cluster_phot_photon_count_filter(cluster_phot):
    """An idealized photon-count filter comes back in photon/s."""
    phot = cluster_phot["Q(HI)"]
    assert isinstance(phot, u.Quantity)
    assert phot.unit.is_equivalent(u.photon / u.s)
    assert np.all(phot.value >= 0)


def test_cluster_phot_lbol_filter(cluster_phot):
    """Lbol comes back as a luminosity Quantity."""
    lbol = cluster_phot["Lbol"]
    assert isinstance(lbol, u.Quantity)
    assert lbol.unit.is_equivalent(u.Lsun)
    assert np.all(lbol.value > 0)


def test_cluster_phot_ext_suffix(cluster_phot):
    """Appending _ex to a real filter name returns its extincted counterpart."""
    plain = cluster_phot["HST.WFC3_UVIS1.F555W"]
    extinct = cluster_phot["HST.WFC3_UVIS1.F555W_ex"]
    assert isinstance(extinct, u.Quantity)
    assert extinct.unit == plain.unit
    assert len(extinct) == len(plain)
    assert not np.array_equal(extinct.value, plain.value)


def test_cluster_phot_lbol_has_no_extincted_counterpart(cluster_phot):
    """Lbol_ex isn't a real column (phot_extinct excludes Lbol) and fails."""
    with pytest.raises(Exception):
        cluster_phot["Lbol_ex"]


def test_cluster_phot_getitem_cached(cluster_phot):
    """Repeated per-filter access returns the identical cached object."""
    assert cluster_phot["Lbol"] is cluster_phot["Lbol"]


def test_cluster_phot_fallthrough_to_group_reader(cluster_phot):
    """A non-filter key (e.g. trial) falls through to slug_group_reader.__getitem__."""
    trial = cluster_phot["trial"]
    assert isinstance(trial, np.ndarray)
    assert not isinstance(trial, u.Quantity)
    assert len(trial) == 2000


def test_cluster_phot_bulk_raw_access(cluster_phot):
    """The raw phot/phot_extinct datasets (per-column units) come back as plain arrays."""
    phot = cluster_phot["phot"]
    phot_extinct = cluster_phot["phot_extinct"]
    assert isinstance(phot, np.ndarray) and not isinstance(phot, u.Quantity)
    assert isinstance(phot_extinct, np.ndarray) and not isinstance(phot_extinct, u.Quantity)
    assert phot.shape == (2000, len(cluster_phot.filters))
    # phot_extinct excludes Lbol, the only filter with no extincted counterpart
    assert phot_extinct.shape == (2000, len(cluster_phot.filters) - 1)


def test_cluster_phot_unknown_key(cluster_phot):
    """A key that's neither a filter name nor a dataset name raises KeyError."""
    with pytest.raises(KeyError):
        cluster_phot["not_a_filter_or_dataset"]


# ---------------------------------------------------------------------
# slug_reader.get_cluster
#
# clusterlib.toml's output.output_times points at dists/times.toml,
# a path relative to examples/clusterlib/ (see that deck's own leading
# comment); get_cluster() reconstructs SimControls purely from
# input_deck's own text, with no memory of which directory the
# original deck lived in, so that reconstruction only resolves such
# relative paths correctly when the current working directory happens
# to match -- exactly the same constraint the original slug run itself
# had. Every test below chdir's into examples/clusterlib for this
# reason (monkeypatch.chdir, so it's undone automatically after each
# test), even though the reader fixture itself was opened from the
# repo root.
# ---------------------------------------------------------------------

@requires_full_data
def test_get_cluster_reconstructs_correctly(reader, clusters, monkeypatch):
    """get_cluster(uid) returns a Cluster with that uid's own uid/target_mass."""
    uid = int(clusters["uid"][0])
    target_mass = float(clusters["target_mass"][0].value)

    monkeypatch.chdir(CLUSTERLIB_DIR)
    cl = reader.get_cluster(uid)

    assert cl.uid() == uid
    assert cl.targetMass() == target_mass


@requires_full_data
def test_get_cluster_bitwise_reproducible(reader, clusters, monkeypatch):
    """Two independent get_cluster() calls for the same uid agree bit-for-bit."""
    uid = int(clusters["uid"][1])

    monkeypatch.chdir(CLUSTERLIB_DIR)
    cl_a = reader.get_cluster(uid)
    cl_b = reader.get_cluster(uid)

    assert cl_a.starMasses() == cl_b.starMasses()
    assert cl_a.birthMass() == cl_b.birthMass()
    assert cl_a.feH() == cl_b.feH()


@requires_full_data
def test_get_cluster_caches_controls(monkeypatch):
    """The SimControls built for get_cluster() is cached and reused across calls."""
    fresh_reader = slug_reader(CLUSTERLIB_H5)
    monkeypatch.chdir(CLUSTERLIB_DIR)
    clusters = fresh_reader.clusters
    assert clusters is not None

    assert fresh_reader._controls is None
    fresh_reader.get_cluster(int(clusters["uid"][0]))
    controls = fresh_reader._controls
    assert controls is not None
    fresh_reader.get_cluster(int(clusters["uid"][1]))
    assert fresh_reader._controls is controls


def test_get_cluster_unknown_uid(reader, monkeypatch):
    """An unrecognized uid raises KeyError."""
    monkeypatch.chdir(CLUSTERLIB_DIR)
    with pytest.raises(KeyError):
        reader.get_cluster(-1)


def test_get_cluster_no_clusters_group(tmp_path, monkeypatch):
    """get_cluster() raises RuntimeError when this file has no clusters group."""
    path = tmp_path / "no_clusters.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset("toml", data="n_trial = 1")

    monkeypatch.chdir(CLUSTERLIB_DIR)
    empty_reader = slug_reader(str(path))
    with pytest.raises(RuntimeError):
        empty_reader.get_cluster(1)


# ---------------------------------------------------------------------
# slug_phot_reader.filter_collection / get_filter, and
# slug_reader.get_filter
#
# Only idealized filters (Q(HI), Q(HeI), ...) are exercised against
# clusterlib.h5 itself: its real HST filters resolve through the
# default filter registry (data/filters/filters.h5/.toml), which,
# unlike data/filters/V_filter.h5/.toml, is fetched data rather than
# committed to the repo, so a test depending on it wouldn't run in CI.
# The registry-propagation test below builds its own tiny fixture
# instead, pointing phot.registry at the committed V_filter registry.
# ---------------------------------------------------------------------

def test_filter_collection_starts_empty(cluster_phot):
    """filter_collection is a PhotSystem.Vega FilterCollection, initially empty."""
    assert isinstance(cluster_phot.filter_collection, FilterCollection)
    assert cluster_phot.filter_collection.filterNames() == []


def test_filter_collection_readonly(cluster_phot):
    """Assigning to filter_collection raises AttributeError."""
    with pytest.raises(AttributeError):
        cluster_phot.filter_collection = None


def test_get_filter_idealized(cluster_phot):
    """get_filter() on an idealized filter needs no registry at all."""
    filt = cluster_phot.get_filter("Q(HI)")
    assert isinstance(filt, Filter)
    assert filt.name() == "Q(HI)"
    assert "Q(HI)" in cluster_phot.filter_collection.filterNames()


def test_get_filter_cached(cluster_phot):
    """Repeated get_filter() calls for the same name return the identical object."""
    assert cluster_phot.get_filter("Q(HeI)") is cluster_phot.get_filter("Q(HeI)")


def test_get_filter_via_reader(reader):
    """slug_reader.get_filter() delegates to cluster_phot.get_filter()."""
    assert reader.get_filter("Q(OII)").name() == "Q(OII)"


def test_get_filter_unknown_name(cluster_phot):
    """A name that isn't one of this file's own filters raises KeyError."""
    with pytest.raises(KeyError):
        cluster_phot.get_filter("not_one_of_the_filters")


def test_get_filter_unparseable_name(cluster_phot):
    """A filter name in this file that isn't a real filter (Lbol) raises RuntimeError."""
    with pytest.raises(RuntimeError):
        cluster_phot.get_filter("Lbol")


def test_get_filter_no_cluster_phot_group(tmp_path):
    """slug_reader.get_filter() raises RuntimeError when this file has no cluster_phot group."""
    path = tmp_path / "no_cluster_phot.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset("toml", data="n_trial = 1")

    empty_reader = slug_reader(str(path))
    with pytest.raises(RuntimeError):
        empty_reader.get_filter("Q(HI)")


def test_get_filter_registry_propagation(tmp_path):
    """phot.registry in input_deck is passed through to slug_phot_reader."""
    path = tmp_path / "registry_test.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        deck = (
            "n_trial = 1\n"
            "sim_type = \"cluster\"\n\n"
            "[phot]\n"
            "registry = \"data/filters/V_filter.toml\"\n")
        f.create_group("input_deck").create_dataset("toml", data=deck)
        g = f.create_group("cluster_phot")
        g.attrs["filters"] = ["Generic.Johnson.V"]
        phot = g.create_dataset("phot", data=np.array([[15.0]]))
        phot.attrs["units"] = ["mag"]
        trial = g.create_dataset("trial", data=np.array([0], dtype="u8"))
        trial.attrs["units"] = ""

    registry_reader = slug_reader(str(path))
    filt = registry_reader.get_filter("Generic.Johnson.V")
    assert filt.name() == "Generic.Johnson.V"
    assert abs(filt.wlPivot() - 5501.402599255) < 1e-6


# ---------------------------------------------------------------------
# slug_phot_reader.phot_convert / slug_reader.phot_convert
#
# Like test_get_filter_registry_propagation above, uses its own tiny
# fixture pointing phot.registry at the committed V_filter registry,
# rather than clusterlib.h5's own real HST filters, which resolve
# through the default filter registry (fetched data, unavailable in
# CI) -- see that test's own comment.
# ---------------------------------------------------------------------

def _make_phot_convert_test_file(tmp_path):
    """A cluster_phot fixture with one real (Vega-system) filter and one idealized one."""
    path = tmp_path / "phot_convert_test.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        deck = (
            "n_trial = 1\n"
            "sim_type = \"cluster\"\n\n"
            "[phot]\n"
            "registry = \"data/filters/V_filter.toml\"\n")
        f.create_group("input_deck").create_dataset("toml", data=deck)
        g = f.create_group("cluster_phot")
        g.attrs["filters"] = ["Generic.Johnson.V", "Q(HI)"]
        phot = g.create_dataset("phot", data=np.array([[15.0, 1e47]]))
        phot.attrs["units"] = ["mag", "photon/s"]
        # Only one column -- Q(HI) has no extincted counterpart here,
        # mirroring clusterlib.h5's own Lbol/phot_extinct gap.
        phot_extinct = g.create_dataset("phot_extinct", data=np.array([[15.5]]))
        phot_extinct.attrs["units"] = ["mag"]
        trial = g.create_dataset("trial", data=np.array([0], dtype="u8"))
        trial.attrs["units"] = ""
    return path


def test_phot_convert_converts_real_filter(tmp_path):
    """phot_convert() converts a real filter's photometry to the target system."""
    reader = slug_reader(str(_make_phot_convert_test_file(tmp_path)))
    cp = reader.cluster_phot
    assert cp is not None
    orig = cast(u.Quantity, cp["Generic.Johnson.V"])
    assert orig.unit == u.mag

    reader.phot_convert("AB")
    converted = cast(u.Quantity, cp["Generic.Johnson.V"])
    assert converted.unit == u.ABmag
    assert not np.allclose(converted.value, orig.value)


def test_phot_convert_leaves_non_convertible_units_untouched(tmp_path):
    """phot_convert() leaves a photon-count filter's entry alone."""
    reader = slug_reader(str(_make_phot_convert_test_file(tmp_path)))
    cp = reader.cluster_phot
    assert cp is not None
    orig = cast(u.Quantity, cp["Q(HI)"])

    reader.phot_convert("AB")
    after = cast(u.Quantity, cp["Q(HI)"])
    assert after.unit == orig.unit
    assert np.allclose(after.value, orig.value)


def test_phot_convert_extinct_dict_and_missing_column(tmp_path):
    """phot_convert() also converts phot_extinct, skipping a filter missing there."""
    reader = slug_reader(str(_make_phot_convert_test_file(tmp_path)))
    cp = reader.cluster_phot
    assert cp is not None

    reader.phot_convert("AB")  # must not raise, despite Q(HI) having no extinct column
    ext = cast(u.Quantity, cp["Generic.Johnson.V_ex"])
    assert ext.unit == u.ABmag
    with pytest.raises(IndexError):
        cp["Q(HI)_ex"]


def test_phot_convert_roundtrip(tmp_path):
    """Converting Vega -> AB -> Vega recovers the original value."""
    reader = slug_reader(str(_make_phot_convert_test_file(tmp_path)))
    cp = reader.cluster_phot
    assert cp is not None
    orig = cast(u.Quantity, cp["Generic.Johnson.V"]).value

    reader.phot_convert("AB")
    reader.phot_convert("Vega")
    back = cast(u.Quantity, cp["Generic.Johnson.V"])
    assert np.allclose(back.value, orig, atol=1e-9)


def test_phot_convert_via_reader_delegates(tmp_path):
    """slug_reader.phot_convert() delegates to cluster_phot.phot_convert()."""
    reader = slug_reader(str(_make_phot_convert_test_file(tmp_path)))
    reader.phot_convert("AB")
    cp = reader.cluster_phot
    assert cp is not None
    converted = cast(u.Quantity, cp["Generic.Johnson.V"])
    assert converted.unit == u.ABmag


def test_phot_convert_noop_when_no_cluster_phot(tmp_path):
    """slug_reader.phot_convert() is a silent no-op when there's no cluster_phot group."""
    path = tmp_path / "no_cluster_phot_convert.h5"
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset("toml", data="n_trial = 1")

    reader = slug_reader(str(path))
    reader.phot_convert("AB")
