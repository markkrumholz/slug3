"""
Unit tests for slug_reader.run_cloudy.

Uses small, hand-built synthetic HDF5 fixtures (rather than a real
slug output file) so every code path -- multiple matches, the
Q(HI)-from-phot vs. Q(HI)-from-spectrum fallback, cluster vs. galaxy
[Fe/H] lookup (fixed value and PDF-distribution cases) -- can be
exercised and checked exactly, independent of what any particular
committed example happens to contain.
"""

import re

import astropy.units as u
import h5py
import numpy as np
import pytest
from slugpy._slug import FilterIdeal, parsePDFDescriptor

from slugpy.slug_reader import slug_reader

# A spectrum with real flux shortward of the H ionization edge
# (~911.7 Angstrom), so the Q(HI)-from-spectrum fallback path has
# something nonzero to compute
_WL = [200.0, 500.0, 900.0, 1000.0, 3000.0]
_SPEC = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]


def _extract_float(text: str, pattern: str) -> float:
    """Search text for pattern (which must have one capture group) and return it as a float."""
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"pattern {pattern!r} not found"
    return float(match.group(1))


def _write_common_attrs(f: h5py.File, sim_type: str, model_name: str = "testmodel") -> None:
    f.attrs["slug-hash"] = "deadbeef"
    f.attrs["date"] = "2026-01-01"
    f.attrs["time"] = "00:00:00"
    f.attrs["rng_state"] = "x"
    f.create_group("input_deck").create_dataset(
        "toml", data=f'sim_type = "{sim_type}"\nn_trial = 1\n\n[output]\nmodel_name = "{model_name}"\n')


def _make_cluster_test_file(tmp_path, name: str, qhi_in_phot: bool) -> str:
    """
    A minimal cluster-type output file with 2 clusters, 3 (uid, time)
    spectra (uid 1 has two output times, uid 2 has one), and either
    Q(HI) directly in cluster_phot's own filters, or a different
    filter only (so run_cloudy must fall back to computing Q(HI) from
    the spectrum itself).
    """
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        _write_common_attrs(f, "cluster")

        c = f.create_group("clusters")
        uid = c.create_dataset("uid", data=np.array([1, 2], dtype="u8"))
        uid.attrs["units"] = ""
        feh = c.create_dataset("feh", data=np.array([0.1, -0.2]))
        feh.attrs["units"] = ""

        cs = f.create_group("cluster_spectra")
        cs_uid = cs.create_dataset("uid", data=np.array([1, 1, 2], dtype="u8"))
        cs_uid.attrs["units"] = ""
        cs_time = cs.create_dataset("time", data=np.array([1e6, 2e6, 1e6]))
        cs_time.attrs["units"] = "yr"
        cs_wl = cs.create_dataset("wl", data=np.array(_WL))
        cs_wl.attrs["units"] = "Angstrom"
        cs_spec = cs.create_dataset("spec", data=np.array([_SPEC, _SPEC, _SPEC]))
        cs_spec.attrs["units"] = "erg/(s Angstrom)"

        cp = f.create_group("cluster_phot")
        cp_uid = cp.create_dataset("uid", data=np.array([1, 1, 2], dtype="u8"))
        cp_uid.attrs["units"] = ""
        cp_time = cp.create_dataset("time", data=np.array([1e6, 2e6, 1e6]))
        cp_time.attrs["units"] = "yr"
        if qhi_in_phot:
            cp.attrs["filters"] = ["Q(HI)"]
            cp_phot = cp.create_dataset("phot", data=np.array([[1e49], [5e47], [2e49]]))
            cp_phot.attrs["units"] = ["photon/s"]
        else:
            cp.attrs["filters"] = ["Lbol"]
            cp_phot = cp.create_dataset("phot", data=np.array([[1.0], [1.0], [1.0]]))
            cp_phot.attrs["units"] = ["Lsun"]
    return str(path)


def _make_galaxy_test_file(tmp_path, name: str, feh_entry: str) -> str:
    """A minimal galaxy-type output file with 2 trials, each a single output time, Q(HI) in galaxy_phot."""
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.attrs["slug-hash"] = "deadbeef"
        f.attrs["date"] = "2026-01-01"
        f.attrs["time"] = "00:00:00"
        f.attrs["rng_state"] = "x"
        f.create_group("input_deck").create_dataset(
            "toml", data=f'sim_type = "galaxy"\nn_trial = 1\n\n[output]\nmodel_name = "testmodel"\n'
                f'\n[stars]\nFeH = {feh_entry}\n')

        gs = f.create_group("galaxy_spectra")
        gs_trial = gs.create_dataset("trial", data=np.array([0, 1], dtype="u8"))
        gs_trial.attrs["units"] = ""
        gs_time = gs.create_dataset("time", data=np.array([1e6, 1e6]))
        gs_time.attrs["units"] = "yr"
        gs_wl = gs.create_dataset("wl", data=np.array(_WL))
        gs_wl.attrs["units"] = "Angstrom"
        gs_spec = gs.create_dataset("spec", data=np.array([_SPEC, _SPEC]))
        gs_spec.attrs["units"] = "erg/(s Angstrom)"

        gp = f.create_group("galaxy_phot")
        gp_trial = gp.create_dataset("trial", data=np.array([0, 1], dtype="u8"))
        gp_trial.attrs["units"] = ""
        gp_time = gp.create_dataset("time", data=np.array([1e6, 1e6]))
        gp_time.attrs["units"] = "yr"
        gp.attrs["filters"] = ["Q(HI)"]
        gp_phot = gp.create_dataset("phot", data=np.array([[3e49], [4e49]]))
        gp_phot.attrs["units"] = ["photon/s"]
    return str(path)


def _qh_value(text: str) -> float:
    return _extract_float(text, r"^Q\(H\) = (\S+)$")


# ---------------------------------------------------------------------
# Cluster mode
# ---------------------------------------------------------------------

def test_cluster_single_match(tmp_path):
    """uid + time together select exactly one spectrum."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    paths = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    assert len(paths) == 1
    assert paths[0].is_file()


def test_cluster_uid_only_matches_all_times(tmp_path):
    """uid alone (time omitted) matches every output time for that uid."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    paths = reader.run_cloudy("cluster", uid=1, output_dir=tmp_path / "out")
    assert len(paths) == 2


def test_cluster_no_selector_matches_everything(tmp_path):
    """Neither uid nor time given matches every (uid, time) spectrum in the file."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    paths = reader.run_cloudy("cluster", output_dir=tmp_path / "out")
    assert len(paths) == 3


def test_cluster_no_match_raises(tmp_path):
    """A uid that doesn't exist raises ValueError."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    with pytest.raises(ValueError, match="no cluster spectra match"):
        reader.run_cloudy("cluster", uid=999, output_dir=tmp_path / "out")


def test_cluster_trial_argument_raises(tmp_path):
    """Passing trial for spec_type='cluster' is a usage error."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    with pytest.raises(ValueError, match="trial is only meaningful"):
        reader.run_cloudy("cluster", trial=0, output_dir=tmp_path / "out")


def test_cluster_feh_looked_up_by_uid(tmp_path):
    """Each cluster's own feh (not some default) is used, per its own uid."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    path_uid1 = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")[0]
    path_uid2 = reader.run_cloudy("cluster", uid=2, time=1e6, output_dir=tmp_path / "out2")[0]

    metals1 = _extract_float(path_uid1.read_text(), r"^metals and grains (\S+)$")
    metals2 = _extract_float(path_uid2.read_text(), r"^metals and grains (\S+)$")
    assert metals1 == pytest.approx(10.0 ** 0.1)
    assert metals2 == pytest.approx(10.0 ** -0.2)


def test_cluster_qhi_taken_from_phot_when_available(tmp_path):
    """Q(HI) matches cluster_phot's own value exactly when the file has one."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    path = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")[0]
    assert _qh_value(path.read_text()) == pytest.approx(np.log10(1e49), rel=1e-6)


def test_cluster_qhi_fallback_to_filterideal(tmp_path):
    """Without Q(HI) in cluster_phot, Q(HI) is instead computed from the spectrum via FilterIdeal."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c2.h5", qhi_in_phot=False))
    path = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")[0]

    expected = FilterIdeal("Q(HI)").phot(_WL, _SPEC)
    assert _qh_value(path.read_text()) == pytest.approx(np.log10(expected), rel=1e-6)


def test_cluster_missing_group_raises(tmp_path):
    """A file with no cluster_spectra group raises ValueError."""
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        _write_common_attrs(f, "cluster")
    reader = slug_reader(str(path))
    with pytest.raises(ValueError, match="no cluster_spectra group"):
        reader.run_cloudy("cluster", output_dir=tmp_path / "out")


# ---------------------------------------------------------------------
# Galaxy mode
# ---------------------------------------------------------------------

def test_galaxy_fixed_feh(tmp_path):
    """A fixed (non-distribution) stars.FeH value is used directly."""
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "-0.4"))
    path = reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")[0]
    metals = _extract_float(path.read_text(), r"^metals and grains (\S+)$")
    assert metals == pytest.approx(10.0 ** -0.4)


def test_galaxy_pdf_feh_uses_expectation_value(tmp_path):
    """A stars.FeH PDF file path resolves to that distribution's own expectation value."""
    feh_path = tmp_path / "feh_dist.toml"
    feh_path.write_text('breakpoints = [-1.0, 1.0]\n\n[segment1]\ntype = "powerlaw"\nslope = 0.0\n')
    expected_feh = parsePDFDescriptor(str(feh_path)).expectationValue()

    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g2.h5", f'"{feh_path}"'))
    path = reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")[0]
    metals = _extract_float(path.read_text(), r"^metals and grains (\S+)$")
    assert metals == pytest.approx(10.0 ** expected_feh)
    assert expected_feh == pytest.approx(0.0)  # symmetric range, sanity check


def test_galaxy_feh_same_for_every_matching_trial(tmp_path):
    """The same population-level feh is used for every matched galaxy spectrum, not looked up per row."""
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.2"))
    paths = reader.run_cloudy("galaxy", output_dir=tmp_path / "out")
    assert len(paths) == 2
    metals = [_extract_float(p.read_text(), r"^metals and grains (\S+)$") for p in paths]
    assert metals[0] == pytest.approx(metals[1])
    assert metals[0] == pytest.approx(10.0 ** 0.2)


def test_galaxy_uid_argument_raises(tmp_path):
    """Passing uid for spec_type='galaxy' is a usage error."""
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.0"))
    with pytest.raises(ValueError, match="uid is only meaningful"):
        reader.run_cloudy("galaxy", uid=1, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------
# Nebular-condition (hiiregparam) kwargs
# ---------------------------------------------------------------------

def test_default_nebular_conditions(tmp_path):
    """With no nII/r0/r1/U/U0/Omega given, defaults to nII=100 cm^-3, U=10**-2.5."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    path = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")[0]
    hden = _extract_float(path.read_text(), r"^hden (\S+)$")
    assert hden == pytest.approx(np.log10(100.0))


def test_explicit_nebular_conditions(tmp_path):
    """User-supplied nII/r0/etc. are used instead of the defaults."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    path = reader.run_cloudy(
        "cluster", uid=1, time=1e6, nII=500.0 / u.cm ** 3, U=1e-3, output_dir=tmp_path / "out")[0]
    hden = _extract_float(path.read_text(), r"^hden (\S+)$")
    assert hden == pytest.approx(np.log10(500.0))
