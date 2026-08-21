"""
Unit tests for slug_reader.run_cloudy.

Uses small, hand-built synthetic HDF5 fixtures (rather than a real
slug output file) so every code path -- multiple matches, the
Q(HI)-from-phot vs. Q(HI)-from-spectrum fallback, cluster vs. galaxy
[Fe/H] lookup (fixed value and PDF-distribution cases), success vs.
failure tracking, and the cluster_cloudy/galaxy_cloudy HDF5 group
being written -- can be exercised and checked exactly, independent of
what any particular committed example happens to contain.
"""

import re
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest
from slugpy._slug import FilterIdeal, parsePDFDescriptor

from slugpy.slug_group_reader import slug_group_reader
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


def _decks_in(out_dir: Path) -> list[Path]:
    """Every .in deck written into out_dir, in a deterministic order (run_cloudy no longer returns deck paths directly)."""
    return sorted(out_dir.glob("*.in"))


@pytest.fixture(scope="session")
def _fake_cloudy_dir(tmp_path_factory) -> Path:
    """A directory containing a tiny stand-in for cloudy.exe: it echoes a marker, copies its own stdin to stdout, then ends with cloudy's own real success message, so run_cloudy sees every run as successful (see _failing_cloudy_exe/_mixed_cloudy_exe for the other cases)."""
    d = tmp_path_factory.mktemp("fake_cloudy")
    script = d / "cloudy.exe"
    script.write_text(
        '#!/bin/sh\n'
        'sleep "${FAKE_CLOUDY_SLEEP:-0}"\n'
        'echo FAKE_CLOUDY_RAN\n'
        'cat\n'
        'echo "[Stop in cdMain at maincl.cpp:590, Cloudy exited OK]"\n'
    )
    script.chmod(0o755)
    return d


@pytest.fixture(autouse=True)
def _cloudy_dir_env(monkeypatch, _fake_cloudy_dir):
    """Point every test's own run_cloudy() calls at the fake cloudy executable, so tests don't depend on a real cloudy installation."""
    monkeypatch.setenv("CLOUDY_DIR", str(_fake_cloudy_dir))


@pytest.fixture(scope="session")
def _failing_cloudy_exe(tmp_path_factory) -> Path:
    """A fake cloudy.exe that runs but never prints cloudy's own success message, simulating a crashed/incomplete run."""
    d = tmp_path_factory.mktemp("fake_cloudy_fail")
    script = d / "cloudy.exe"
    script.write_text('#!/bin/sh\necho FAKE_CLOUDY_RAN\necho "cloudy did not finish"\n')
    script.chmod(0o755)
    return script


@pytest.fixture(scope="session")
def _mixed_cloudy_exe(tmp_path_factory) -> Path:
    """A fake cloudy.exe that fails for uid 2's own deck and succeeds for everything else (matched by uid000000002 appearing in the deck's own "save" filenames), to exercise per-run success/failure tracking across a batch."""
    d = tmp_path_factory.mktemp("fake_cloudy_mixed")
    script = d / "cloudy.exe"
    script.write_text(
        '#!/bin/sh\n'
        'DECK=$(cat)\n'
        'if echo "$DECK" | grep -q "uid000000002"; then\n'
        '  echo FAKE_CLOUDY_RAN_FAIL\n'
        '  echo "cloudy did not finish"\n'
        'else\n'
        '  echo FAKE_CLOUDY_RAN_OK\n'
        '  echo "[Stop in cdMain at maincl.cpp:590, Cloudy exited OK]"\n'
        'fi\n'
    )
    script.chmod(0o755)
    return script


@pytest.fixture(scope="session")
def _continuum_cloudy_exe(tmp_path_factory) -> Path:
    """A fake cloudy.exe that also writes small canned "save last continuum"- and "save last line array"-format files at the names the deck itself requests, so both output-reading paths can be exercised end-to-end without a real cloudy run."""
    d = tmp_path_factory.mktemp("fake_cloudy_continuum")
    script = d / "cloudy.exe"
    script.write_text(
        '#!/usr/bin/env python3\n'
        'import re, sys\n'
        'deck = sys.stdin.read()\n'
        'm = re.search(r\'save last continuum "([^"]+)"\', deck)\n'
        'if m:\n'
        '    with open(m.group(1), "w") as f:\n'
        '        f.write("#Cont\\tnu\\tincident\\ttrans\\tDiffOut\\tnet trans\\n")\n'
        '        f.write("1.0000e-02\\t1.000e+30\\t0.000e+00\\t1.000e+30\\t1.000e+30\\n")\n'
        '        f.write("5.0000e-02\\t2.000e+30\\t0.000e+00\\t2.000e+30\\t2.000e+30\\n")\n'
        '        f.write("1.0000e-01\\t3.000e+30\\t0.000e+00\\t3.000e+30\\t3.000e+30\\n")\n'
        'm2 = re.search(r\'save last line array units Angstrom "([^"]+)"\', deck)\n'
        'if m2:\n'
        '    with open(m2.group(1), "w") as f:\n'
        '        f.write("#enr\\tID\\tI(intrinsic)\\tI(emergent)\\ttype\\n")\n'
        '        f.write(" 1.00000e+03\\tH  1                1.00000A \\t  40.000\\t  41.000 \\ti\\n")\n'
        '        f.write(" 2.00000e+03\\tH  1 Coll           2.00000A \\t  40.000\\t  41.000 \\ti\\n")\n'
        '        f.write(" 3.00000e+03\\tHe 2                3.00000A \\t   4.000\\t   4.000 \\ti\\n")\n'
        'print("FAKE_CLOUDY_RAN")\n'
        'print("[Stop in cdMain at maincl.cpp:590, Cloudy exited OK]")\n'
    )
    script.chmod(0o755)
    return script


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
    """uid + time together select exactly one spectrum; a single match returns a bare bool."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    assert result is True
    assert len(_decks_in(tmp_path / "out")) == 1


def test_cluster_uid_only_matches_all_times(tmp_path):
    """uid alone (time omitted) matches every output time for that uid; multiple matches return a list of bool."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", uid=1, output_dir=tmp_path / "out")
    assert result == [True, True]
    assert len(_decks_in(tmp_path / "out")) == 2


def test_cluster_uid_list_matches_every_listed_uid(tmp_path):
    """uid as a list matches every output time for every uid in the list."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", uid=[2, 1], output_dir=tmp_path / "out")
    assert result == [True, True, True]
    assert len(_decks_in(tmp_path / "out")) == 3


def test_cluster_uid_list_excludes_unlisted_uids(tmp_path):
    """uid as a list only matches the listed uids, not every uid in the file."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", uid=[2], output_dir=tmp_path / "out")
    assert result is True
    assert len(_decks_in(tmp_path / "out")) == 1


def test_cluster_no_selector_matches_everything(tmp_path):
    """Neither uid nor time given matches every (uid, time) spectrum in the file."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", output_dir=tmp_path / "out")
    assert result == [True, True, True]
    assert len(_decks_in(tmp_path / "out")) == 3


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
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    reader.run_cloudy("cluster", uid=2, time=1e6, output_dir=tmp_path / "out2")

    path_uid1 = _decks_in(tmp_path / "out1")[0]
    path_uid2 = _decks_in(tmp_path / "out2")[0]
    metals1 = _extract_float(path_uid1.read_text(), r"^metals and grains (\S+)$")
    metals2 = _extract_float(path_uid2.read_text(), r"^metals and grains (\S+)$")
    assert metals1 == pytest.approx(10.0 ** 0.1)
    assert metals2 == pytest.approx(10.0 ** -0.2)


def test_cluster_qhi_taken_from_phot_when_available(tmp_path):
    """Q(HI) matches cluster_phot's own value exactly when the file has one."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    assert _qh_value(path.read_text()) == pytest.approx(np.log10(1e49), rel=1e-6)


def test_cluster_qhi_fallback_to_filterideal(tmp_path):
    """Without Q(HI) in cluster_phot, Q(HI) is instead computed from the spectrum via FilterIdeal."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c2.h5", qhi_in_phot=False))
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]

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
    reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    metals = _extract_float(path.read_text(), r"^metals and grains (\S+)$")
    assert metals == pytest.approx(10.0 ** -0.4)


def test_galaxy_pdf_feh_uses_expectation_value(tmp_path):
    """A stars.FeH PDF file path resolves to that distribution's own expectation value."""
    feh_path = tmp_path / "feh_dist.toml"
    feh_path.write_text('breakpoints = [-1.0, 1.0]\n\n[segment1]\ntype = "powerlaw"\nslope = 0.0\n')
    expected_feh = parsePDFDescriptor(str(feh_path)).expectationValue()

    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g2.h5", f'"{feh_path}"'))
    reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    metals = _extract_float(path.read_text(), r"^metals and grains (\S+)$")
    assert metals == pytest.approx(10.0 ** expected_feh)
    assert expected_feh == pytest.approx(0.0)  # symmetric range, sanity check


def test_galaxy_trial_list_matches_every_listed_trial(tmp_path):
    """trial as a list matches every listed trial, same as uid does for cluster mode."""
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.0"))
    result = reader.run_cloudy("galaxy", trial=[1, 0], output_dir=tmp_path / "out")
    assert result == [True, True]
    assert len(_decks_in(tmp_path / "out")) == 2


def test_galaxy_feh_same_for_every_matching_trial(tmp_path):
    """The same population-level feh is used for every matched galaxy spectrum, not looked up per row."""
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.2"))
    result = reader.run_cloudy("galaxy", output_dir=tmp_path / "out")
    assert result == [True, True]
    paths = _decks_in(tmp_path / "out")
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
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    hden = _extract_float(path.read_text(), r"^hden (\S+)$")
    assert hden == pytest.approx(np.log10(100.0))


def test_explicit_nebular_conditions(tmp_path):
    """User-supplied nII/r0/etc. are used instead of the defaults."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    reader.run_cloudy(
        "cluster", uid=1, time=1e6, nII=500.0 / u.cm ** 3, U=1e-3, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    hden = _extract_float(path.read_text(), r"^hden (\S+)$")
    assert hden == pytest.approx(np.log10(500.0))


# ---------------------------------------------------------------------
# Actually running cloudy
# ---------------------------------------------------------------------

def test_cloudy_is_run_on_every_deck(tmp_path):
    """Every written deck gets its own .out file, with the fake cloudy's own stdout marker in it -- proving cloudy was actually launched, not just that the deck was written."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", output_dir=tmp_path / "out")
    assert result == [True, True, True]
    for deck in _decks_in(tmp_path / "out"):
        out_file = deck.with_suffix(".out")
        assert out_file.is_file()
        assert "FAKE_CLOUDY_RAN" in out_file.read_text()


def test_cloudy_deck_content_piped_to_stdin(tmp_path):
    """The fake cloudy echoes its own stdin back after the marker line; check the deck's own content shows up there, confirming the right file was piped in."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    path = _decks_in(tmp_path / "out")[0]
    out_text = path.with_suffix(".out").read_text()
    assert path.read_text() in out_text


def test_cloudy_path_override_takes_precedence(tmp_path, monkeypatch):
    """An explicit cloudy_path is used even if CLOUDY_DIR points somewhere else."""
    other_dir = tmp_path / "other_cloudy"
    other_dir.mkdir()
    other_exe = other_dir / "cloudy.exe"
    other_exe.write_text('#!/bin/sh\necho OTHER_CLOUDY_RAN\necho "[Stop, Cloudy exited OK]"\n')
    other_exe.chmod(0o755)

    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    reader.run_cloudy(
        "cluster", uid=1, time=1e6, output_dir=tmp_path / "out", cloudy_path=other_exe)
    path = _decks_in(tmp_path / "out")[0]
    out_text = path.with_suffix(".out").read_text()
    assert "OTHER_CLOUDY_RAN" in out_text
    assert "FAKE_CLOUDY_RAN" not in out_text


def test_missing_cloudy_executable_raises(tmp_path, monkeypatch):
    """If neither cloudy_path nor CLOUDY_DIR resolve to a real executable, run_cloudy raises before writing anything."""
    monkeypatch.delenv("CLOUDY_DIR", raising=False)
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="cloudy"):
        reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=out_dir)
    assert not out_dir.exists()


def test_max_workers_is_passed_through(tmp_path):
    """max_workers is accepted and doesn't break a multi-deck run (concurrency itself is exercised more directly in test_cloudy_process.py)."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy("cluster", output_dir=tmp_path / "out", max_workers=1)
    assert result == [True, True, True]
    for deck in _decks_in(tmp_path / "out"):
        assert deck.with_suffix(".out").is_file()


# ---------------------------------------------------------------------
# Success/failure detection
# ---------------------------------------------------------------------

def test_single_failure_returns_false(tmp_path, _failing_cloudy_exe):
    """A cloudy run whose own log never says "Cloudy exited OK" is reported as a failure."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy(
        "cluster", uid=1, time=1e6, output_dir=tmp_path / "out", cloudy_path=_failing_cloudy_exe)
    assert result is False


def test_mixed_success_and_failure_returns_matching_bool_list(tmp_path, _mixed_cloudy_exe):
    """Per-run success/failure is tracked independently across a batch, in the same order the spectra were processed."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    result = reader.run_cloudy(
        "cluster", output_dir=tmp_path / "out", cloudy_path=_mixed_cloudy_exe)
    # Processing order follows the underlying arrays: (uid=1,t=1e6),
    # (uid=1,t=2e6), (uid=2,t=1e6) -- only the uid=2 entry fails
    assert result == [True, True, False]


# ---------------------------------------------------------------------
# Writing results into cluster_cloudy/galaxy_cloudy
# ---------------------------------------------------------------------

def test_successful_run_recorded_in_cluster_cloudy_group(tmp_path):
    """A successful run's own uid/time/nII/etc. are appended as a new row in cluster_cloudy."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    assert result is True

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1]
        assert g["time"][()] == pytest.approx([1e6])
        assert g["nII"][()] == pytest.approx([100.0])  # default nII


def test_failed_run_not_recorded(tmp_path, _mixed_cloudy_exe):
    """A failed run leaves the HDF5 file untouched -- no cluster_cloudy group at all if every run in the call failed."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy(
        "cluster", uid=2, time=1e6, output_dir=tmp_path / "out", cloudy_path=_mixed_cloudy_exe)
    assert result is False
    with h5py.File(h5_path, "r") as f:
        assert "cluster_cloudy" not in f


def test_mixed_batch_records_only_successes(tmp_path, _mixed_cloudy_exe):
    """In a batch with both successes and failures, only the successful runs get a row."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy("cluster", output_dir=tmp_path / "out", cloudy_path=_mixed_cloudy_exe)
    assert result == [True, True, False]
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1, 1]
        assert g["time"][()] == pytest.approx([1e6, 2e6])


def test_repeated_calls_append_rows_and_reader_stays_usable(tmp_path):
    """A second run_cloudy() call appends more rows rather than overwriting, and this reader's own cached group readers keep working afterward (they were rebuilt against the reopened file)."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    reader.run_cloudy("cluster", uid=2, time=1e6, output_dir=tmp_path / "out2")

    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [1, 2]

    clusters = reader.clusters
    assert clusters is not None
    assert clusters["uid"].tolist() == [1, 2]


# ---------------------------------------------------------------------
# Duplicate detection and overwrite
# ---------------------------------------------------------------------

def test_repeated_call_with_same_conditions_is_skipped_by_default(tmp_path):
    """Calling run_cloudy again for the same uid/time with the same (default) nebular conditions skips the actual cloudy run and reports success, without writing a deck or adding a duplicate row."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    result = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out2")
    assert result is True
    assert len(_decks_in(tmp_path / "out2")) == 0
    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [1]


def test_batch_with_one_duplicate_only_reruns_the_new_one(tmp_path):
    """In a uid list spanning one already-stored spectrum and one new one, only the new one is actually run, but both report success in the right order."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    result = reader.run_cloudy("cluster", uid=[1, 2], time=1e6, output_dir=tmp_path / "out2")
    assert result == [True, True]
    assert len(_decks_in(tmp_path / "out2")) == 1  # only uid 2 actually ran
    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [1, 2]


def test_repeated_call_with_different_conditions_is_not_a_duplicate(tmp_path):
    """A rerun with different nebular conditions is a genuinely new run, not a duplicate -- it's actually run and appended as its own new row."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    result = reader.run_cloudy(
        "cluster", uid=1, time=1e6, nII=1.0 / u.cm ** 3, U=1e-2, output_dir=tmp_path / "out2")
    assert result is True
    assert len(_decks_in(tmp_path / "out2")) == 1
    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["uid"][()].tolist() == [1, 1]
        assert g["nII"][()] == pytest.approx([100.0, 1.0])


def test_overwrite_reruns_and_replaces_the_stored_row(tmp_path):
    """overwrite=True reruns what would otherwise be a skipped duplicate, and replaces its stored row instead of skipping or duplicating it."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out1")
    result = reader.run_cloudy(
        "cluster", uid=1, time=1e6, output_dir=tmp_path / "out2", overwrite=True)
    assert result is True
    assert len(_decks_in(tmp_path / "out2")) == 1  # actually reran, not skipped
    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [1]  # replaced, not duplicated


def test_overwrite_without_an_existing_match_just_appends(tmp_path):
    """overwrite=True has no special effect when there's nothing already stored to overwrite -- it's a plain new run and append."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out", overwrite=True)
    assert result is True
    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [1]


def test_overwrite_does_not_delete_on_a_failed_rerun(tmp_path, _mixed_cloudy_exe):
    """If overwrite=True but the rerun itself fails, the previously-stored row is left untouched rather than being deleted."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    reader.run_cloudy("cluster", uid=2, time=1e6, output_dir=tmp_path / "out1")
    result = reader.run_cloudy(
        "cluster", uid=2, time=1e6, output_dir=tmp_path / "out2",
        cloudy_path=_mixed_cloudy_exe, overwrite=True)
    assert result is False
    with h5py.File(h5_path, "r") as f:
        assert f["cluster_cloudy"]["uid"][()].tolist() == [2]  # still there, untouched


def test_galaxy_run_recorded_in_galaxy_cloudy_group(tmp_path):
    """Galaxy-mode runs go into galaxy_cloudy (keyed by trial), not cluster_cloudy."""
    h5_path = _make_galaxy_test_file(tmp_path, "g1.h5", "0.0")
    reader = slug_reader(h5_path)
    result = reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")
    assert result is True
    with h5py.File(h5_path, "r") as f:
        assert "cluster_cloudy" not in f
        assert f["galaxy_cloudy"]["trial"][()].tolist() == [0]


def test_continuum_written_when_available(tmp_path, _continuum_cloudy_exe):
    """When a run's own deck produces a "save last continuum" file, its wavelength grid and spectra are written into cluster_cloudy too."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy(
        "cluster", uid=1, time=1e6, output_dir=tmp_path / "out", cloudy_path=_continuum_cloudy_exe)
    assert result is True

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["wl"].shape[0] == 3
        assert g["spec_inc"].shape == (1, 3)
        assert np.all(g["spec_inc"][0] > 0)
        assert g["spec_trans"].shape == (1, 3)
        assert np.all(g["spec_trans"][0] == pytest.approx(0.0))


def test_lines_written_when_available(tmp_path, _continuum_cloudy_exe):
    """When a run's own deck produces a "save last line array" file, only the lines passing both filters (blank label suffix, emergent luminosity > 1e10 erg/s) are written into cluster_cloudy."""
    h5_path = _make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True)
    reader = slug_reader(h5_path)
    result = reader.run_cloudy(
        "cluster", uid=1, time=1e6, output_dir=tmp_path / "out", cloudy_path=_continuum_cloudy_exe)
    assert result is True

    with h5py.File(h5_path, "r") as f:
        g = f["cluster_cloudy"]
        assert g["line_wl"][()] == pytest.approx([1000.0])
        assert [x.decode() for x in g["line_label"][()]] == ["H  1"]
        assert g["line_lum"][()] == pytest.approx(np.array([[10.0 ** 41.0]]))


# ---------------------------------------------------------------------
# cluster_cloudy / galaxy_cloudy reader properties
# ---------------------------------------------------------------------

def test_cluster_cloudy_none_before_run_cloudy(tmp_path):
    """A freshly-opened reader has no cluster_cloudy group until run_cloudy has been called."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    assert reader.cluster_cloudy is None


def test_cluster_cloudy_populated_after_run_cloudy(tmp_path):
    """After run_cloudy creates the group on this same reader, cluster_cloudy becomes a working slug_group_reader over it -- exercising the _groups key-set refresh on reopen, not just the property itself."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    assert reader.cluster_cloudy is None

    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")

    cluster_cloudy = reader.cluster_cloudy
    assert isinstance(cluster_cloudy, slug_group_reader)
    assert cluster_cloudy["uid"].tolist() == [1]
    assert cluster_cloudy["time"].to_value(u.yr) == pytest.approx([1e6])
    # cached: the same object is returned on a second access
    assert reader.cluster_cloudy is cluster_cloudy


def test_galaxy_cloudy_none_before_run_cloudy(tmp_path):
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.0"))
    assert reader.galaxy_cloudy is None


def test_galaxy_cloudy_populated_after_run_cloudy(tmp_path):
    reader = slug_reader(_make_galaxy_test_file(tmp_path, "g1.h5", "0.0"))
    reader.run_cloudy("galaxy", trial=0, output_dir=tmp_path / "out")

    galaxy_cloudy = reader.galaxy_cloudy
    assert isinstance(galaxy_cloudy, slug_group_reader)
    assert galaxy_cloudy["trial"].tolist() == [0]


def test_cluster_cloudy_does_not_leak_into_galaxy_cloudy(tmp_path):
    """A cluster-mode run populates cluster_cloudy but not galaxy_cloudy."""
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    reader.run_cloudy("cluster", uid=1, time=1e6, output_dir=tmp_path / "out")
    assert reader.cluster_cloudy is not None
    assert reader.galaxy_cloudy is None


def test_cluster_cloudy_is_read_only(tmp_path):
    reader = slug_reader(_make_cluster_test_file(tmp_path, "c1.h5", qhi_in_phot=True))
    with pytest.raises(AttributeError):
        reader.cluster_cloudy = None
