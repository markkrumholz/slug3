"""
Unit tests for slugpy.compute_isochrones, an end-to-end convenience
function that builds isochrones from a set of stellar tracks, runs
spectral synthesis on the resulting stars, and (optionally) computes
filter photometry from their spectra.

Uses the same small, committed MIST_test tracks fixture (tests/tracks/
assets/tracks.toml + MIST_test.h5) and blackbody spectral synthesizer
test_run_sim.py itself uses, plus the pre-existing
testClusterFeHDist.toml PDF fixture for the non-fixed-[Fe/H] tests, so
this needs no data fetched separately and can stay a "quick" test. Run
via pytest with WORKING_DIRECTORY set to the repo root, matching
test_readers.py/test_run_sim.py's own convention, so the relative
track_registry/FeH paths below resolve correctly.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import numpy as np
import pytest
from astropy import units as u
from slugpy._slug import FilterCollection, PhotSystem, SimControls

from slugpy import compute_isochrones

# A cluster deck using the small MIST_test tracks and a blackbody
# spectral synthesizer (fast, no spectral library data needed), with a
# fixed (delta-function) [Fe/H] = 0.0 -- one of the values actually
# present in MIST_test.h5's own Fe_H grid.
FIXED_FEH_DECK = """
sim_type = "cluster"
n_trial = 1

[stars]
IMF = "chabrier.toml"
track_registry = "tests/tracks/assets/tracks.toml"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[output]
output_times = [0.0]

[nebular]
compute_neb = false
"""

# Otherwise identical to FIXED_FEH_DECK, but with a non-delta [Fe/H]
# distribution (uniform between -0.5 and 0.5, both of which are also
# present in MIST_test.h5's own Fe_H grid) -- so compute_isochrones's
# own feh keyword actually takes effect, rather than being overridden
# by a single fixed value.
VAR_FEH_DECK = """
sim_type = "cluster"
n_trial = 1

[stars]
IMF = "chabrier.toml"
track_registry = "tests/tracks/assets/tracks.toml"
tracks = "MIST_test"
FeH = "tests/core/assets/testClusterFeHDist.toml"
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[output]
output_times = [0.0]

[nebular]
compute_neb = false
"""

_TRACK_FIELD_NAMES = (
    "mass", "mdot", "log_L", "log_Teff",
    "h_surf", "he_surf", "c_surf", "n_surf", "o_surf")


@pytest.fixture
def fixed_feh_controls():
    """A SimControls built from FIXED_FEH_DECK."""
    return SimControls(FIXED_FEH_DECK)


@pytest.fixture
def var_feh_controls():
    """A SimControls built from VAR_FEH_DECK."""
    return SimControls(VAR_FEH_DECK)


def test_default_masses_are_log_spaced_within_imf_range(fixed_feh_controls):
    """With masses left as None, the returned masses are nmass values
    log-spaced between the IMF's own min and max mass, and the result
    dict has nmass columns and every track-quantity key, and no filter
    keys (filters defaults to None)."""
    imf = fixed_feh_controls.imf
    masses, data = compute_isochrones(1e7, simcontrols=fixed_feh_controls, nmass=5)

    assert masses.unit == u.Msun
    assert masses.shape == (5,)
    assert np.allclose(
        masses.value, np.geomspace(imf.getMin(), imf.getMax(), 5))

    assert set(data.keys()) == set(_TRACK_FIELD_NAMES)
    for name in _TRACK_FIELD_NAMES:
        assert data[name].shape == (1, 5)


def test_explicit_masses_are_returned_unchanged(fixed_feh_controls):
    """masses, when given, comes back as the first return value,
    converted to Msun."""
    masses, _ = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0, 20.0] * u.Msun)

    assert masses.unit == u.Msun
    assert np.allclose(masses.value, [1.0, 5.0, 20.0])


def test_explicit_masses_plain_array_and_quantity_agree(fixed_feh_controls):
    """masses as a plain array (assumed Msun) and as an equivalent
    astropy Quantity array give the same result."""
    masses_plain, from_plain = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0, 20.0])
    masses_quantity, from_quantity = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0, 20.0] * u.Msun)

    assert np.allclose(masses_plain.value, masses_quantity.value)
    for name in _TRACK_FIELD_NAMES:
        assert np.allclose(from_plain[name].value, from_quantity[name].value, equal_nan=True)


def test_mass_outside_track_range_is_nan(fixed_feh_controls):
    """A mass beyond every isochrone segment's own range (MIST_test's
    tracks top out at 300 Msun) is NaN in every field."""
    _, data = compute_isochrones(1e7, simcontrols=fixed_feh_controls, masses=[500.0])

    for name in _TRACK_FIELD_NAMES:
        assert np.isnan(data[name].value).all()


def test_time_float_years_and_quantity_agree(fixed_feh_controls):
    """A plain float time (assumed yr) and an equivalent Quantity in a
    different time unit give the same result."""
    _, from_float = compute_isochrones(1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0])
    _, from_myr = compute_isochrones(10 * u.Myr, simcontrols=fixed_feh_controls, masses=[1.0, 5.0])

    for name in _TRACK_FIELD_NAMES:
        assert np.allclose(from_float[name].value, from_myr[name].value, equal_nan=True)


def test_filters_none_skips_photometry(fixed_feh_controls):
    """filters=None (the default) adds no filter keys to the result."""
    _, data = compute_isochrones(1e7, simcontrols=fixed_feh_controls, masses=[1.0])
    assert set(data.keys()) == set(_TRACK_FIELD_NAMES)


def test_filters_string_and_collection_agree(fixed_feh_controls):
    """Passing a filter name builds the same FilterCollection (same
    default PhotSystem.Flambda) as passing one explicitly."""
    fc = FilterCollection(["ideal_energy_1000_2000"], PhotSystem.Flambda)

    _, from_name = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0], filters="ideal_energy_1000_2000")
    _, from_collection = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0], filters=fc)

    assert "ideal_energy_1000_2000" in from_name
    assert np.allclose(
        from_name["ideal_energy_1000_2000"].value,
        from_collection["ideal_energy_1000_2000"].value, equal_nan=True)
    assert from_name["ideal_energy_1000_2000"].unit == from_collection["ideal_energy_1000_2000"].unit


def test_phot_system_changes_filter_units(fixed_feh_controls):
    """phot_system (passed through to the FilterCollection built from a
    filter name) changes the unit an energy-flux filter's photometry
    comes back in."""
    _, data_flambda = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0],
        filters="ideal_energy_1000_2000", phot_system=PhotSystem.Flambda)
    _, data_ab = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0],
        filters="ideal_energy_1000_2000", phot_system=PhotSystem.AB)

    assert data_flambda["ideal_energy_1000_2000"].unit == u.erg / u.s / u.AA
    assert data_ab["ideal_energy_1000_2000"].unit == u.ABmag


def test_fixed_feh_ignores_feh_kwarg(fixed_feh_controls):
    """When simcontrols.feH is a delta distribution, the feh keyword is
    ignored -- results are identical regardless of what it's set to."""
    _, default_feh = compute_isochrones(1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0])
    _, other_feh = compute_isochrones(
        1e7, simcontrols=fixed_feh_controls, masses=[1.0, 5.0], feh=-1.0)

    for name in _TRACK_FIELD_NAMES:
        assert np.allclose(default_feh[name].value, other_feh[name].value, equal_nan=True)


def test_var_feh_uses_feh_kwarg(var_feh_controls):
    """When simcontrols.feH is not a delta distribution, the feh keyword
    actually selects which [Fe/H] slice of the tracks is used."""
    _, data_low = compute_isochrones(1e7, simcontrols=var_feh_controls, masses=[1.0], feh=-0.5)
    _, data_high = compute_isochrones(1e7, simcontrols=var_feh_controls, masses=[1.0], feh=0.5)

    assert not np.allclose(
        data_low["log_Teff"].value, data_high["log_Teff"].value, equal_nan=True)


def test_no_specsyn_raises():
    """simcontrols.specsyn is None (no spectra.model in the deck) raises
    RuntimeError rather than failing deeper in with a confusing error."""
    deck = """
sim_type = "cluster"
n_trial = 1

[stars]
IMF = "chabrier.toml"
track_registry = "tests/tracks/assets/tracks.toml"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[clusters]
CMF = 1e3

[output]
output_times = [0.0]

[nebular]
compute_neb = false
"""
    controls = SimControls(deck)
    assert controls.specsyn is None

    with pytest.raises(RuntimeError, match="specsyn"):
        compute_isochrones(1e7, simcontrols=controls, masses=[1.0])


def test_simcontrols_kwargs_build_a_default_simcontrols():
    """With simcontrols left as None, compute_isochrones builds one from
    its own extra keywords, exactly as SimControls(**kwargs) would."""
    masses, data = compute_isochrones(
        1e7, path=FIXED_FEH_DECK, masses=[1.0, 5.0])
    assert masses.unit == u.Msun
    assert set(data.keys()) == set(_TRACK_FIELD_NAMES)
    assert data["mass"].shape == (1, 2)
