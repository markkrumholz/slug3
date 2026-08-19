"""
Unit tests for slugpy.cloudy.hiiregparam.

The central test here (test_cross_consistency) is a systematic
cross-check: starting from an independently-derived ("hand-verified,"
not using hiiregparam itself) full physical state for several seed
(nII, r0) pairs, it checks that every one of the 13 valid two-parameter
combinations reproduces that same state when fed back into a fresh
hiiregparam. This is the regression guard against the exact class of
bug that motivated hiiregparam's internal design (solving every
combination through a single verified (nII, Omega) core, rather than
13 separate closed-form shortcuts) -- an earlier, more direct port
reproduced this way would have failed 9 of the 13 combinations.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import itertools
import warnings

import astropy.units as u
import numpy as np
import pytest

from slugpy.cloudy.hiiregparam import hiiregparam

QH0 = 1e49 * u.photon / u.s

# Reference physical constants (matching hiiregparam's own private
# module constants), used only to build the independent reference
# calculation below -- deliberately not imported from hiiregparam
# itself, so this check doesn't just validate the module against its
# own constants.
_ALPHA_B = 2.59e-13   # cm^3/s
_FE = 1.1
_C = 2.99792458e10    # cm/s

_NAMES = ["nII", "r0", "r1", "U", "U0", "Omega"]
_DISALLOWED = {frozenset(["r0", "U"]), frozenset(["r1", "U0"])}
_UNITS: dict[str, u.UnitBase] = {
    "nII": 1 / u.cm ** 3, "r0": u.cm, "r1": u.cm,
    "U": u.dimensionless_unscaled, "U0": u.dimensionless_unscaled,
    "Omega": u.dimensionless_unscaled,
}


def _wind_fac_reference(omega: float) -> float:
    """The same wind factor W(omega), reimplemented independently for the reference calculation."""
    if omega < 1.0e6:
        return (1.0 + omega) ** (4.0 / 3.0) - omega ** (1.0 / 3.0) * (4.0 / 3.0 + omega)
    return (2.0 / 9.0 * omega ** (-2.0 / 3.0) - 4.0 / 81.0 * omega ** (-5.0 / 3.0)
        + 5.0 / 243.0 * omega ** (-8.0 / 3.0))


def _reference_state(qH0: float, nII: float, r0: float) -> dict[str, float]:
    """
    Independently derive a full (nII, r0, r1, U, U0, Omega) state.

    Built only from the recombination-equilibrium and ionization-
    parameter definitions themselves (rs^3 = r1^3 - r0^3, Omega =
    r0^3/rs^3, U = U(nII, Omega), U0 = qH0/(4 pi r0^2 fe nII c)), not
    from hiiregparam's own code, so it serves as an independent ground
    truth for test_cross_consistency below.
    """
    rs = (3.0 * qH0 / (4.0 * np.pi * _ALPHA_B * _FE * nII ** 2)) ** (1.0 / 3.0)
    omega = (r0 / rs) ** 3
    r1 = rs * (1.0 + omega) ** (1.0 / 3.0)
    K1 = (81.0 * _ALPHA_B ** 2 * qH0 / (256.0 * np.pi * _C ** 3 * _FE)) ** (1.0 / 3.0)
    U = K1 * nII ** (1.0 / 3.0) * _wind_fac_reference(omega)
    U0 = qH0 / (4.0 * np.pi * r0 ** 2 * _FE * nII * _C)
    return {"nII": nII, "r0": r0, "r1": r1, "U": U, "U0": U0, "Omega": omega}


_SEEDS = [(100.0, 1e18), (100.0, 5e18), (100.0, 9e18), (10.0, 1e17), (1000.0, 3e17)]

_ALL_COMBO_CASES = [
    (seed, pair)
    for seed in _SEEDS
    for pair in itertools.combinations(_NAMES, 2)
    if frozenset(pair) not in _DISALLOWED
]


@pytest.mark.parametrize(
    "seed,pair", _ALL_COMBO_CASES,
    ids=[f"nII0={s[0]:g},r00={s[1]:g}-{p[0]}+{p[1]}" for s, p in _ALL_COMBO_CASES])
def test_cross_consistency(seed: tuple[float, float], pair: tuple[str, str]) -> None:
    """Every valid 2-parameter combination reproduces the independent reference state."""
    nII0, r00 = seed
    truth = _reference_state(QH0.to_value(u.photon / u.s), nII0, r00)

    a, b = pair
    kwargs = {a: truth[a] * _UNITS[a], b: truth[b] * _UNITS[b]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hp = hiiregparam(QH0, **kwargs)

    got = {
        "nII": hp.nII.to_value(1 / u.cm ** 3), "r0": hp.r0.to_value(u.cm),
        "r1": hp.r1.to_value(u.cm), "U": hp.U.value, "U0": hp.U0.value,
        "Omega": hp.Omega.value,
    }
    for name in _NAMES:
        assert got[name] == pytest.approx(truth[name], rel=1e-6)


def test_disallowed_r0_U() -> None:
    """The (r0, U) combination raises ValueError -- it has no unique solution."""
    with pytest.raises(ValueError, match="non-unique"):
        hiiregparam(QH0, r0=1e18 * u.cm, U=1e-3)


def test_disallowed_r1_U0() -> None:
    """The (r1, U0) combination raises ValueError -- it has no unique solution."""
    with pytest.raises(ValueError, match="non-unique"):
        hiiregparam(QH0, r1=1e19 * u.cm, U0=1e-2)


@pytest.mark.parametrize("kwargs", [
    {"nII": 100.0 / u.cm ** 3},
    {"nII": 100.0 / u.cm ** 3, "r0": 1e18 * u.cm, "r1": 1e19 * u.cm},
])
def test_wrong_parameter_count_raises(kwargs: dict[str, u.Quantity]) -> None:
    """Anything other than exactly 2 of the 6 parameters raises ValueError."""
    with pytest.raises(ValueError, match="need exactly 2"):
        hiiregparam(QH0, **kwargs)


def test_dynamic_mode_matches_rKM() -> None:
    """(t, n0, Omega) mode derives r1 via rKM(), and the result is self-consistent."""
    t = 1.0 * u.Myr
    n0 = 100.0 / u.cm ** 3
    hp = hiiregparam(QH0, Omega=0.5, n0=n0, t=t)

    assert hp.Omega.value == pytest.approx(0.5)
    assert hp.r1.to_value(u.cm) == pytest.approx(hp.rKM(t, n0).to_value(u.cm))


def test_photon_and_bare_rate_give_same_result() -> None:
    """qH0 in explicit photon/s and in bare 1/s give identical results."""
    hp_photon = hiiregparam(1e49 * u.photon / u.s, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
    hp_bare = hiiregparam(1e49 / u.s, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
    assert hp_photon.Omega.value == pytest.approx(hp_bare.Omega.value)
    assert hp_photon.U.value == pytest.approx(hp_bare.U.value)


def test_outputs_are_dimensionless_quantities() -> None:
    """U, U0, and Omega come back as plain dimensionless Quantities, with no residual unit tag."""
    hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
    for q in (hp.U, hp.U0, hp.Omega):
        assert q.unit == u.dimensionless_unscaled


class TestROrdering:
    """r0 >= r1 is corrected (warn=True) or raised (warn=False)."""

    def test_warn_false_raises(self) -> None:
        with pytest.raises(ValueError, match="r1 must be >= r0"):
            hiiregparam(QH0, r0=1e19 * u.cm, r1=1e19 * u.cm, warn=False)

    def test_warn_true_corrects_r1(self) -> None:
        with pytest.warns(UserWarning, match="r1 must be >= r0"):
            hp = hiiregparam(QH0, r0=1e19 * u.cm, r1=1e19 * u.cm)
        assert hp.r1 > hp.r0

    def test_fix_quantity_r0_corrects_r0(self) -> None:
        with pytest.warns(UserWarning, match="r1 must be >= r0"):
            hp = hiiregparam(QH0, r0=1e19 * u.cm, r1=1e19 * u.cm, fix_quantity="r0")
        assert hp.r1 > hp.r0
        assert hp.r1.to_value(u.cm) == pytest.approx(1e19)


class TestNIIUBoundary:
    """U cannot exceed the Omega=0 maximum for a given nII."""

    def test_warn_false_raises(self) -> None:
        with pytest.raises(ValueError, match="U too large"):
            hiiregparam(QH0, nII=100.0 / u.cm ** 3, U=1.0, warn=False)

    def test_warn_true_lowers_U(self) -> None:
        with pytest.warns(UserWarning, match="U too large"):
            hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, U=1.0)
        assert hp.Omega.value == pytest.approx(0.0, abs=1e-6)
        # Round-trips: the corrected state is self-consistent
        hp2 = hiiregparam(QH0, nII=hp.nII, U=hp.U)
        assert hp2.Omega.value == pytest.approx(hp.Omega.value, rel=1e-6)

    def test_fix_quantity_nII_raises_density(self) -> None:
        with pytest.warns(UserWarning, match="nII too small"):
            hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, U=1e-2, fix_quantity="nII")
        assert hp.U.value == pytest.approx(1e-2)
        assert hp.nII.to_value(1 / u.cm ** 3) > 100.0
        hp2 = hiiregparam(QH0, nII=hp.nII, U=hp.U)
        assert hp2.nII.to_value(1 / u.cm ** 3) == pytest.approx(hp.nII.to_value(1 / u.cm ** 3), rel=1e-6)


class TestNIIR1Boundary:
    """r1 cannot be smaller than the windless Stromgren radius for a given nII."""

    def test_warn_false_raises(self) -> None:
        with pytest.raises(ValueError, match="nII too small"):
            hiiregparam(QH0, nII=100.0 / u.cm ** 3, r1=1e17 * u.cm, warn=False)

    def test_warn_true_raises_nII(self) -> None:
        with pytest.warns(UserWarning, match="nII too small"):
            hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r1=1e17 * u.cm)
        # r0safety=0.01 (default) means the corrected nII lands just
        # above (not exactly at) the Omega=0 boundary
        assert 0.0 < hp.Omega.value < 0.05
        assert hp.nII.to_value(1 / u.cm ** 3) > 100.0

    def test_fix_quantity_r1_raises_r1(self) -> None:
        with pytest.warns(UserWarning, match="r1 too small"):
            hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r1=1e17 * u.cm, fix_quantity="r1")
        assert hp.nII.to_value(1 / u.cm ** 3) == pytest.approx(100.0)
        assert hp.r1.to_value(u.cm) > 1e17
        hp2 = hiiregparam(QH0, nII=hp.nII, r1=hp.r1)
        assert hp2.Omega.value == pytest.approx(hp.Omega.value, rel=1e-6)

    def test_valid_combination_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r1=1e19 * u.cm)
        assert hp.Omega.value > 0.0


class TestR1UBoundary:
    """U cannot exceed the Omega=0 maximum for a given r1."""

    def test_warn_false_raises(self) -> None:
        with pytest.raises(ValueError, match="U too large"):
            hiiregparam(QH0, r1=1e19 * u.cm, U=1e-2, warn=False)

    def test_warn_true_lowers_U(self) -> None:
        with pytest.warns(UserWarning, match="U too large"):
            hp = hiiregparam(QH0, r1=1e19 * u.cm, U=1e-2)
        assert hp.r1.to_value(u.cm) == pytest.approx(1e19)
        assert hp.Omega.value == pytest.approx(0.0, abs=1e-3)

    def test_fix_quantity_r1_lowers_r1(self) -> None:
        with pytest.warns(UserWarning, match="r1 too large"):
            hp = hiiregparam(QH0, r1=1e19 * u.cm, U=1e-2, fix_quantity="r1")
        assert hp.U.value == pytest.approx(1e-2)
        assert hp.r1.to_value(u.cm) < 1e19
        hp2 = hiiregparam(QH0, nII=hp.nII, r1=hp.r1)
        assert hp2.U.value == pytest.approx(hp.U.value, rel=1e-6)


class TestUU0Boundary:
    """U0 cannot be smaller than 2U (the Omega -> infinity asymptote)."""

    def test_warn_false_raises(self) -> None:
        with pytest.raises(ValueError, match="maximum value of U is U0/2"):
            hiiregparam(QH0, U=1e-2, U0=1.5e-2, warn=False)

    def test_warn_true_lowers_U(self) -> None:
        with pytest.warns(UserWarning, match="maximum value of U is U0/2"):
            hp = hiiregparam(QH0, U=1e-2, U0=1.5e-2)
        assert hp.U0.value == pytest.approx(1.5e-2)
        assert hp.U.value < 0.75e-2

    def test_valid_combination_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hp = hiiregparam(QH0, U=1e-3, U0=1.0)
        assert hp.Omega.value > 0.0


class TestSetters:
    """Property setters re-solve the state, and refuse to set a never-given parameter."""

    def test_setter_updates_and_resolves(self) -> None:
        hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
        hp.r0 = 2e18 * u.cm
        assert hp.r0.to_value(u.cm) == pytest.approx(2e18)
        assert hp.nII.to_value(1 / u.cm ** 3) == pytest.approx(100.0)

    @pytest.mark.parametrize("attr,val", [
        ("r1", 1e19 * u.cm), ("U", 1e-3), ("U0", 1e-2), ("Omega", 0.1),
    ])
    def test_setter_on_never_given_parameter_raises(self, attr: str, val: object) -> None:
        hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
        with pytest.raises(ValueError, match="cannot add a parameter"):
            setattr(hp, attr, val)


def test_large_omega_uses_series_branch() -> None:
    """A very large Omega (exercising _wind_fac's large-omega series branch) stays finite and positive."""
    hp = hiiregparam(QH0, U0=1e-8, Omega=1e8)
    assert np.isfinite(hp.nII.value)
    assert hp.nII.value > 0.0
    assert np.isfinite(hp.U.value)
    assert hp.U.value > 0.0


def test_rS_matches_r1_at_zero_omega() -> None:
    """rS() (the windless Stromgren radius) equals r1 when Omega is essentially zero."""
    hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, Omega=1e-12)
    assert hp.rS().to_value(u.cm) == pytest.approx(hp.r1.to_value(u.cm), rel=1e-6)


def test_zeta_equals_rch_over_r1() -> None:
    """zeta() is exactly rch()/r1, by definition."""
    hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
    assert hp.zeta().value == pytest.approx((hp.rch() / hp.r1).decompose().value)


def test_rKM_increases_with_time() -> None:
    """The Krumholz & Matzner (2009) expansion radius grows monotonically with age."""
    hp = hiiregparam(QH0, nII=100.0 / u.cm ** 3, r0=1e18 * u.cm)
    n0 = 100.0 / u.cm ** 3
    r_early = hp.rKM(0.1 * u.Myr, n0)
    r_late = hp.rKM(10.0 * u.Myr, n0)
    assert r_late > r_early
