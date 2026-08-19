"""
Unit tests for slugpy.phot_convert, a thin astropy-Quantity-aware
wrapper around the compiled PhotConvert binding.

Uses the committed data/filters/V_filter.h5/.toml fixture (a
standalone, single-filter copy of the real Generic.Johnson.V filter --
see make_v_filter_fixture.py) as a real filter, needed for the Vega-
system conversions, so this needs no data fetched separately and can
stay a "quick" test alongside test_readers.py -- see the test_Slugpy
CMakeLists.txt target, which runs this file too. Like test_readers.py,
run via pytest with WORKING_DIRECTORY set to the repo root, so
V_FILTER_REGISTRY below resolves without needing SLUG_DIR.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import numpy as np
import pytest
from astropy import units as u

from slugpy._slug import FilterTabulated, PhotConvert
from slugpy.phot_convert import phot_convert

V_FILTER_REGISTRY = "data/filters/V_filter.toml"


@pytest.fixture(scope="module")
def v_filter():
    """The real Generic.Johnson.V filter, from the committed V_filter fixture."""
    return FilterTabulated("Generic", "Johnson", "V", registry=V_FILTER_REGISTRY)


def test_flambda_to_fnu_matches_raw_photconvert(v_filter):
    """Flambda -> Fnu agrees with calling the compiled binding directly."""
    wl = v_filter.wlPivot()
    flambda = np.array([1e-10, 2e-10, 5e-11]) * u.erg / u.s / u.AA

    result = phot_convert(flambda, "Fnu", wl=wl)
    expected = PhotConvert("Flambda", "Fnu", flambda.value, wl)

    assert result.unit == u.Jy
    assert np.allclose(result.value, expected)


def test_flambda_fnu_roundtrip(v_filter):
    """Flambda -> Fnu -> Flambda recovers the original value."""
    wl = v_filter.wlPivot()
    flambda = np.array([1e-10, 2e-10]) * u.erg / u.s / u.AA

    fnu = phot_convert(flambda, "Fnu", wl=wl)
    back = phot_convert(fnu, "Flambda", wl=wl)

    assert back.unit == u.erg / u.s / u.AA
    assert np.allclose(back.value, flambda.value)


def test_wl_as_quantity_matches_wl_as_float(v_filter):
    """Passing wl as an astropy Quantity gives the same result as a plain float."""
    wl = v_filter.wlPivot()
    flambda = np.array([1e-10, 2e-10]) * u.erg / u.s / u.AA

    from_float = phot_convert(flambda, "Fnu", wl=wl)
    from_quantity = phot_convert(flambda, "Fnu", wl=wl * u.AA)

    assert np.allclose(from_float.value, from_quantity.value)


def test_fnu_ab_wl_independent(v_filter):
    """Fnu <-> AB works without a wavelength at all."""
    fnu = np.array([100.0, 200.0]) * u.Jy
    ab = phot_convert(fnu, "AB")
    assert ab.unit == u.ABmag

    back = phot_convert(ab, "Fnu")
    assert np.allclose(back.value, fnu.value)


def test_flambda_st_wl_independent(v_filter):
    """Flambda <-> ST works without a wavelength at all."""
    flambda = np.array([1e-10, 2e-10]) * u.erg / u.s / u.AA
    st = phot_convert(flambda, "ST")
    assert st.unit == u.STmag

    back = phot_convert(st, "Flambda")
    assert np.allclose(back.value, flambda.value)


def test_missing_wl_raises(v_filter):
    """A wavelength-dependent pair without wl raises ValueError."""
    flambda = np.array([1e-10]) * u.erg / u.s / u.AA
    with pytest.raises(ValueError):
        phot_convert(flambda, "AB")


def test_vega_requires_filter(v_filter):
    """Converting to/from Vega without a filter raises ValueError."""
    flambda = np.array([1e-10]) * u.erg / u.s / u.AA
    wl = v_filter.wlPivot()
    with pytest.raises(ValueError):
        phot_convert(flambda, "Vega", wl=wl)


def test_vega_conversion_with_filter(v_filter):
    """Flambda -> Vega -> Flambda recovers the original value, given a filter."""
    flambda = np.array([1e-10, 2e-10]) * u.erg / u.s / u.AA
    wl = v_filter.wlPivot()

    vega = phot_convert(flambda, "Vega", wl=wl, filt=v_filter)
    assert vega.unit == u.mag

    back = phot_convert(vega, "Flambda", wl=wl, filt=v_filter)
    assert np.allclose(back.value, flambda.value)


def test_identity_is_noop(v_filter):
    """Converting to the system a Quantity is already in returns it unchanged, with no wl/filter needed."""
    flambda = np.array([1e-10, 2e-10]) * u.erg / u.s / u.AA
    assert phot_convert(flambda, "Flambda") is flambda


def test_unrecognized_unit_raises():
    """A unit that isn't a photometric system (e.g. Lsun) raises ValueError."""
    lbol = np.array([1.0, 2.0]) * u.Lsun
    with pytest.raises(ValueError):
        phot_convert(lbol, "AB")


def test_photon_count_unit_raises():
    """photon/s (an idealized photon-count filter) also raises ValueError."""
    qhi = np.array([1e47, 2e47]) * u.photon / u.s
    with pytest.raises(ValueError):
        phot_convert(qhi, "AB")
