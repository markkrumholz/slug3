"""
hiiregparam.py

Implements hiiregparam, a utility class that derives the physical
parameters of an HII region (density, inner/outer radius, ionization
parameter, wind parameter) from any two of one another.

The wind-driven HII region model itself (the relations among nII, r0,
r1, U, U0, and Omega) follows Krumholz et al. (2015, MNRAS, 452, 1447):
http://adsabs.harvard.edu/abs/2015MNRAS.452.1447K.
The characteristic radius/timescale used by rch(), zeta(), and rKM()
follows Krumholz & Matzner (2009, ApJ, 703, 1352):
https://ui.adsabs.harvard.edu/abs/2009ApJ...703.1352K/abstract.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import warnings
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.optimize import brentq

# Physical constants, derived from astropy's own constants/units
# rather than hand-converted from SI
_c: u.Quantity = const.c.cgs                     # Speed of light
_kB: u.Quantity = const.k_B.cgs                   # Boltzmann constant
_mH: u.Quantity = (const.m_e + const.m_p).cgs     # Hydrogen atom mass
_eps0: u.Quantity = (1.0 * u.Ry).cgs              # H ionization potential
_TII: u.Quantity = 1.0e4 * u.K                    # Fiducial HII region temperature

# Physical parameters specific to this HII region model -- fitted or
# assumed values, not fundamental constants astropy can supply
_alphaB: u.Quantity = 2.59e-13 * u.cm ** 3 / u.s   # Case B recombination coefficient at T = 1e4 K
_muH: float = 1.4            # Mean mass per H nucleus for standard cosmic composition
_fe: float = 1.1             # Electrons per H nucleus
_phi: float = 0.73           # Krumholz & Matzner (2009) dust absorption parameter
_psi: float = 3.2            # Krumholz & Matzner (2009) psi parameter
_ftrap: float = 2.0          # Krumholz & Matzner (2009) trapping factor

FixQuantity = Literal["nII", "r0", "r1", "U", "U0", "Omega"]


def _to_rate(q: u.Quantity) -> u.Quantity:
    """
    Convert an ionizing photon rate to bare 1/s.

    Parameters
    ----------
    q : astropy.units.Quantity
        A photon rate, either explicitly tagged (e.g. photon/s) or a
        bare rate (1/s) -- astropy's photon unit does not decompose
        against plain SI units, so it must be handled specially.

    Returns
    -------
    astropy.units.Quantity
        The same rate, in bare s^-1.
    """
    try:
        value = q.to_value(u.photon / u.s)
    except u.UnitConversionError:
        value = q.to_value(1 / u.s)
    return value / u.s


def _wind_fac(omega: float) -> float:
    """
    Evaluate the Krumholz et al. (2015) wind factor W(omega).

    Parameters
    ----------
    omega : float
        The (dimensionless) wind parameter Omega.

    Returns
    -------
    float
        W(omega) = (1 + omega)^(4/3) - omega^(1/3) (4/3 + omega), the
        factor relating the volume-averaged ionization parameter to
        the density and ionizing luminosity of a wind-driven HII
        region.
    """
    if omega < 1.0e6:
        return (1.0 + omega) ** (4.0 / 3.0) - omega ** (1.0 / 3.0) * (4.0 / 3.0 + omega)
    # Series representation to avoid catastrophic cancellation (and
    # spurious negative values from floating-point error) at large omega
    return (2.0 / 9.0 * omega ** (-2.0 / 3.0)
        - 4.0 / 81.0 * omega ** (-5.0 / 3.0)
        + 5.0 / 243.0 * omega ** (-8.0 / 3.0))


def _rs(qH0: u.Quantity, nII: u.Quantity) -> u.Quantity:
    """
    Compute the windless Stromgren radius for a given density.

    Parameters
    ----------
    qH0 : astropy.units.Quantity
        Ionizing photon rate, convertible to 1/s.
    nII : astropy.units.Quantity
        Number density of H nuclei, convertible to cm^-3.

    Returns
    -------
    astropy.units.Quantity
        The radius, in cm, a pure (wind-free) Stromgren sphere of
        density nII would have around a source emitting qH0.
    """
    return (3.0 * qH0 / (4.0 * np.pi * _alphaB * _fe * nII ** 2)) ** (1.0 / 3.0)


def _U_from_nII_omega(qH0: u.Quantity, nII: u.Quantity, omega: float) -> u.Quantity:
    """
    Compute the volume-averaged ionization parameter from density and wind parameter.

    Parameters
    ----------
    qH0 : astropy.units.Quantity
        Ionizing photon rate, convertible to 1/s.
    nII : astropy.units.Quantity
        Number density of H nuclei, convertible to cm^-3.
    omega : float
        The wind parameter Omega.

    Returns
    -------
    astropy.units.Quantity
        The (dimensionless) volume-averaged ionization parameter U.
    """
    K1 = (81.0 * _alphaB ** 2 * qH0 / (256.0 * np.pi * _c ** 3 * _fe)) ** (1.0 / 3.0)
    return (K1 * nII ** (1.0 / 3.0) * _wind_fac(omega)).decompose()


def _U0_from_r0_nII(qH0: u.Quantity, r0: u.Quantity, nII: u.Quantity) -> u.Quantity:
    """
    Compute the inner-radius ionization parameter from r0 and density.

    Parameters
    ----------
    qH0 : astropy.units.Quantity
        Ionizing photon rate, convertible to 1/s.
    r0 : astropy.units.Quantity
        Inner radius, convertible to cm.
    nII : astropy.units.Quantity
        Number density of H nuclei, convertible to cm^-3.

    Returns
    -------
    astropy.units.Quantity
        The (dimensionless) ionization parameter U0 at r0.
    """
    return (qH0 / (4.0 * np.pi * r0 ** 2 * _fe * nII * _c)).decompose()


def _nII_from_r1_omega(qH0: u.Quantity, r1: u.Quantity, omega: float) -> u.Quantity:
    """
    Compute the density implied by a given outer radius and wind parameter.

    Parameters
    ----------
    qH0 : astropy.units.Quantity
        Ionizing photon rate, convertible to 1/s.
    r1 : astropy.units.Quantity
        Outer radius, convertible to cm.
    omega : float
        The wind parameter Omega.

    Returns
    -------
    astropy.units.Quantity
        Number density of H nuclei, in cm^-3, such that an HII region
        of outer radius r1 has the given wind parameter.
    """
    return np.sqrt(3.0 * qH0 * (1.0 + omega) / (4.0 * np.pi * _alphaB * _fe * r1 ** 3))


def _brentq_log_omega(residual: Callable[[float], float], lo: float = -20.0, hi: float = 20.0) -> float:
    """
    Solve for omega via brentq over log10(omega).

    Parameters
    ----------
    residual : callable
        A function of log10(omega) that is monotonic and changes sign
        between lo and hi.
    lo : float, default -20.0
        Lower bound of the log10(omega) search bracket.
    hi : float, default 20.0
        Upper bound of the log10(omega) search bracket.

    Returns
    -------
    float
        The solution omega = 10**log10(omega).

    Raises
    ------
    ValueError
        If no solution is bracketed in [10^lo, 10^hi].
    """
    log_omega = brentq(residual, lo, hi)
    return float(10.0 ** cast(float, log_omega))


class hiiregparam:
    """
    Derive the physical parameters of an HII region from any two of them.

    Parameters
    ----------
    qH0 : astropy.units.Quantity
        Ionizing luminosity of the exciting stars, in photons/s (or
        any equivalent rate).
    nII : astropy.units.Quantity, optional
        Number density of H nuclei, convertible to cm^-3.
    r0 : astropy.units.Quantity, optional
        Inner radius, convertible to cm.
    r1 : astropy.units.Quantity, optional
        Outer radius, convertible to cm.
    U : astropy.units.Quantity or float, optional
        Volume-averaged ionization parameter (dimensionless).
    U0 : astropy.units.Quantity or float, optional
        Ionization parameter at the inner radius (dimensionless).
    Omega : astropy.units.Quantity or float, optional
        Wind parameter (dimensionless).
    t : astropy.units.Quantity, optional
        HII region age, convertible to Myr; if set, n0 and Omega must
        be set also, and r1 is derived from the Krumholz & Matzner
        (2009) expansion solution (see rKM()) rather than being one of
        the two parameters below.
    n0 : astropy.units.Quantity, optional
        Number density of H nuclei in the medium the HII region is
        expanding into, convertible to cm^-3; if set, t and Omega must
        be set also.
    warn : bool, default True
        If True, warn and correct if the input parameter values are
        outside the physically allowed range; if False, raise
        ValueError instead.
    fix_quantity : {"nII", "r0", "r1", "U", "U0", "Omega"}, optional
        If warn is True, which quantity to adjust to bring the inputs
        into the allowed range; only used for the pairs where an
        out-of-range combination is actually possible (nII+U, r1+nII,
        r1+U, U+U0, and the r0 >= r1 ordering check).
    r0safety : float, default 0.01
        If nonzero, a fractional margin of safety of this size is
        added when correcting quantities, to avoid landing exactly at
        the physical boundary (e.g. r0 = 0 or Omega = 0).

    Attributes
    ----------
    nII : astropy.units.Quantity
        Number density of H nuclei, in cm^-3.
    r0 : astropy.units.Quantity
        Inner radius, in cm.
    r1 : astropy.units.Quantity
        Outer radius, in cm.
    U : astropy.units.Quantity
        Volume-averaged ionization parameter (dimensionless).
    U0 : astropy.units.Quantity
        Ionization parameter at the inner radius (dimensionless).
    Omega : astropy.units.Quantity
        Wind parameter (dimensionless).

    Notes
    -----
    Upon instantiation, the caller must set either (1) exactly two of
    the six parameters (nII, r0, r1, U, U0, Omega) -- all pairwise
    combinations are allowed except (r0, U) and (r1, U0), neither of
    which defines a unique solution -- or (2) the combination
    (t, n0, Omega).

    Internally, every combination is solved by first deriving the
    canonical pair (nII, Omega); every other quantity is then computed
    from that pair via a small set of directly-verified formulas. This
    keeps every one of the 13 valid combinations mutually consistent
    with each other by construction, rather than relying on a
    separate closed-form shortcut for each pair.

    References
    ----------
    The wind-driven HII region model (the relations among nII, r0, r1,
    U, U0, and Omega) is from Krumholz et al. (2015, MNRAS, 452, 1447):
    http://adsabs.harvard.edu/abs/2015MNRAS.452.1447K.

    The characteristic radius/timescale used by rch(), zeta(), and
    rKM() is from Krumholz & Matzner (2009, ApJ, 703, 1352):
    https://ui.adsabs.harvard.edu/abs/2009ApJ...703.1352K/abstract.
    """

    def __init__(self, qH0: u.Quantity, nII: u.Quantity | None = None,
        r0: u.Quantity | None = None, r1: u.Quantity | None = None,
        U: u.Quantity | float | None = None, U0: u.Quantity | float | None = None,
        Omega: u.Quantity | float | None = None, t: u.Quantity | None = None,
        n0: u.Quantity | None = None, warn: bool = True,
        fix_quantity: FixQuantity | None = None,
        r0safety: float = 0.01) -> None:

        # Store the raw inputs -- exactly two of these six (or, in
        # dynamic mode, r1 derived from t/n0 plus Omega) must be set.
        # self._qH0_rate is qH0 stripped to a bare rate (1/s) -- used
        # for every internal computation, since astropy's photon unit
        # does not decompose against plain SI units (see _to_rate()) --
        # while self.qH0 keeps whatever unit the caller actually passed
        self.qH0: u.Quantity = qH0
        self._qH0_rate: u.Quantity = _to_rate(qH0)
        self.nII_: u.Quantity | None = nII
        self.r0_: u.Quantity | None = r0
        self.r1_: u.Quantity | None = r1
        self.U_: u.Quantity | None = None if U is None else u.Quantity(U, u.dimensionless_unscaled)
        self.U0_: u.Quantity | None = None if U0 is None else u.Quantity(U0, u.dimensionless_unscaled)
        self.Omega_: u.Quantity | None = None if Omega is None else u.Quantity(Omega, u.dimensionless_unscaled)
        self.warn: bool = warn
        self.fix_quantity: FixQuantity | None = fix_quantity
        self.r0safety: float = r0safety

        # If t and n0 are set, determine r1
        if t is not None and n0 is not None:
            self.r1_ = self.rKM(t, n0)

        # Solve for the canonical (nII, Omega) pair, validating and
        # correcting the inputs as needed along the way
        self._nII_solved: u.Quantity
        self._omega_solved: float
        self._nII_solved, self._omega_solved = self._solve()

    ##################################################################
    # Solving for the canonical (nII, Omega) pair
    ##################################################################

    def _solve(self) -> tuple[u.Quantity, float]:
        """
        Solve for the canonical (nII, Omega) pair from whichever two inputs were given.

        Returns
        -------
        tuple of (astropy.units.Quantity, float)
            (nII, Omega).

        Raises
        ------
        ValueError
            If something other than exactly two of nII, r0, r1, U, U0,
            Omega are set; if the combination is (r0, U) or (r1, U0),
            neither of which defines a unique solution; or if warn is
            False and the given combination is numerically invalid
            (e.g. r1 < r0, or U too large for the given nII).
        """
        qH0 = self._qH0_rate
        nparam = ((self.nII_ is not None) + (self.r0_ is not None)
            + (self.r1_ is not None) + (self.U_ is not None)
            + (self.Omega_ is not None) + (self.U0_ is not None))
        if nparam != 2:
            raise ValueError(
                "hiiregparam: need exactly 2 of the "
                "following: nII, r0, r1, U, U0, Omega")
        if self.U_ is not None and self.r0_ is not None:
            raise ValueError(
                "hiiregparam: combination U and r0 "
                "not allowed because solution is non-unique")
        if self.U0_ is not None and self.r1_ is not None:
            raise ValueError(
                "hiiregparam: combination U0 and r1 "
                "not allowed because solution is non-unique")

        if self.r0_ is not None and self.r1_ is not None:
            self._fix_r0_r1_ordering()
            nII = np.sqrt(3.0 * qH0 / (4.0 * np.pi * _alphaB * _fe * (self.r1_ ** 3 - self.r0_ ** 3)))
            omega = float((self.r0_ ** 3 / (self.r1_ ** 3 - self.r0_ ** 3)).decompose().value)
            return nII, omega
        if self.nII_ is not None and self.r0_ is not None:
            omega = float((self.r0_ / _rs(qH0, self.nII_)).decompose().value ** 3)
            return self.nII_, omega
        if self.nII_ is not None and self.r1_ is not None:
            self._check_nII_r1()
            omega = float((self.r1_ / _rs(qH0, self.nII_)).decompose().value ** 3 - 1.0)
            return self.nII_, omega
        if self.nII_ is not None and self.Omega_ is not None:
            return self.nII_, float(self.Omega_.value)
        if self.r1_ is not None and self.Omega_ is not None:
            nII = _nII_from_r1_omega(qH0, self.r1_, float(self.Omega_.value))
            return nII, float(self.Omega_.value)
        if self.r0_ is not None and self.Omega_ is not None:
            rs = self.r0_ / float(self.Omega_.value) ** (1.0 / 3.0)
            nII = np.sqrt(3.0 * qH0 / (4.0 * np.pi * _alphaB * _fe * rs ** 3))
            return nII, float(self.Omega_.value)
        if self.r0_ is not None and self.U0_ is not None:
            nII = qH0 / (4.0 * np.pi * self.r0_ ** 2 * _fe * _c * self.U0_)
            omega = float((self.r0_ / _rs(qH0, nII)).decompose().value ** 3)
            return nII, omega
        if self.U_ is not None and self.Omega_ is not None:
            self._check_nII_U_omega(self.Omega_.value)
            nII = (self.U_ / _U_from_nII_omega(qH0, 1.0 / u.cm ** 3, float(self.Omega_.value))) ** 3 / u.cm ** 3
            return nII, float(self.Omega_.value)
        if self.U0_ is not None and self.Omega_ is not None:
            r0_at_nII1 = _rs(qH0, 1.0 / u.cm ** 3) * float(self.Omega_.value) ** (1.0 / 3.0)
            U0_at_nII1 = _U0_from_r0_nII(qH0, r0_at_nII1, 1.0 / u.cm ** 3)
            nII = (self.U0_ / U0_at_nII1) ** 3 / u.cm ** 3
            return nII, float(self.Omega_.value)

        if self.nII_ is not None and self.U_ is not None:
            self._check_nII_U(self.nII_)
            omega = self._solve_omega_for_nII_U(self.nII_, self.U_)
            return self.nII_, omega
        if self.nII_ is not None and self.U0_ is not None:
            r0 = np.sqrt(qH0 / (4.0 * np.pi * _fe * self.nII_ * _c * self.U0_))
            omega = float((r0 / _rs(qH0, self.nII_)).decompose().value ** 3)
            return self.nII_, omega
        if self.r1_ is not None and self.U_ is not None:
            nII_lim = _nII_from_r1_omega(qH0, self.r1_, 0.0)
            self._check_r1_U(self.r1_, nII_lim)
            omega = self._solve_omega_for_r1_U(self.r1_, self.U_)
            nII = _nII_from_r1_omega(qH0, self.r1_, omega)
            return nII, omega
        if self.U_ is not None and self.U0_ is not None:
            self._check_U_U0()
            omega = self._solve_omega_for_U_U0(self.U_, self.U0_)
            r0_at_nII1 = _rs(qH0, 1.0 / u.cm ** 3) * omega ** (1.0 / 3.0)
            U0_at_nII1 = _U0_from_r0_nII(qH0, r0_at_nII1, 1.0 / u.cm ** 3)
            nII = (self.U0_ / U0_at_nII1) ** 3 / u.cm ** 3
            return nII, omega

        raise RuntimeError("hiiregparam: unreachable -- no valid parameter combination")

    def _solve_omega_for_nII_U(self, nII: u.Quantity, U: u.Quantity) -> float:
        """Find omega such that _U_from_nII_omega(qH0, nII, omega) == U."""
        target = float(U.value)

        def residual(log_omega: float) -> float:
            got = float(_U_from_nII_omega(self._qH0_rate, nII, 10.0 ** log_omega).value)
            return float(np.log(got / target))

        try:
            return _brentq_log_omega(residual)
        except ValueError as ve:
            raise ValueError(
                "hiiregparam: failed to find solution for Omega at "
                f"(nII = {nII!r}, U = {U!r}, qH0 = {self.qH0!r})") from ve

    def _solve_omega_for_r1_U(self, r1: u.Quantity, U: u.Quantity) -> float:
        """Find omega such that U(nII(r1, omega), omega) == U, at fixed r1."""
        target = float(U.value)

        def residual(log_omega: float) -> float:
            omega = 10.0 ** log_omega
            nII = _nII_from_r1_omega(self._qH0_rate, r1, omega)
            got = float(_U_from_nII_omega(self._qH0_rate, nII, omega).value)
            return float(np.log(got / target))

        try:
            return _brentq_log_omega(residual)
        except ValueError as ve:
            raise ValueError(
                "hiiregparam: failed to find solution for Omega at "
                f"(r1 = {r1!r}, U = {U!r}, qH0 = {self.qH0!r})") from ve

    def _solve_omega_for_U_U0(self, U: u.Quantity, U0: u.Quantity) -> float:
        """Find omega such that U(nII=1, omega)/U0(nII=1, omega) == U/U0 (nII-independent ratio)."""
        target = float((U / U0).decompose().value)

        def residual(log_omega: float) -> float:
            omega = 10.0 ** log_omega
            nII1 = 1.0 / u.cm ** 3
            U_at_1 = _U_from_nII_omega(self._qH0_rate, nII1, omega)
            r0_at_1 = _rs(self._qH0_rate, nII1) * omega ** (1.0 / 3.0)
            U0_at_1 = _U0_from_r0_nII(self._qH0_rate, r0_at_1, nII1)
            got = float((U_at_1 / U0_at_1).decompose().value)
            return float(np.log(got / target))

        try:
            return _brentq_log_omega(residual)
        except ValueError as ve:
            raise ValueError(
                "hiiregparam: failed to find solution for Omega at "
                f"(U0 = {U0!r}, U = {U!r})") from ve

    ##################################################################
    # Validity checks -- these are the only combinations that can
    # yield an unphysical (Omega < 0) result; every other combination
    # is valid for any positive input values
    ##################################################################

    def _fix_r0_r1_ordering(self) -> None:
        """Ensure r1 > r0, correcting or raising per self.warn/fix_quantity."""
        assert self.r0_ is not None and self.r1_ is not None
        if self.r0_ < self.r1_:
            return
        if not self.warn:
            raise ValueError("hiiregparam: r1 must be >= r0")
        if self.fix_quantity != "r0":
            r1new = 1.0001 * self.r0_
            warnings.warn(
                "hiiregparam: r1 must be >= r0; increasing "
                f"r1 from {self.r1_!r} to {r1new!r}")
            self.r1_ = r1new
        else:
            r0new = 0.9999 * self.r0_
            warnings.warn(
                "hiiregparam: r1 must be >= r0; decreasing "
                f"r0 from {self.r0_!r} to {r0new!r}")
            self.r0_ = r0new

    def _check_nII_U(self, nII: u.Quantity) -> None:
        """Ensure U does not exceed the maximum (Omega=0) value for nII, correcting or raising."""
        assert self.U_ is not None
        Ulim = _U_from_nII_omega(self._qH0_rate, nII, 0.0)
        if self.U_ <= Ulim:
            return
        if not self.warn:
            raise ValueError(
                "hiiregparam: U too large for input value of nII; "
                f"for nII = {nII!r}, maximum U is {Ulim!r}")
        if self.fix_quantity != "nII":
            Unew = (1.0 - self.r0safety) * Ulim
            warnings.warn(
                "hiiregparam: U too large for input value of nII; "
                f"lowering from {self.U_!r} to {Unew!r}")
            self.U_ = Unew
        else:
            nIInew = nII * ((self.U_ / Ulim) ** 3 * (1.0 + self.r0safety))
            warnings.warn(
                "hiiregparam: nII too small for input value of U; "
                f"increasing from {nII!r} to {nIInew!r}")
            self.nII_ = nIInew

    def _check_nII_U_omega(self, omega: float) -> None:
        """Ensure a directly-given Omega is non-negative."""
        if omega < 0.0:
            raise ValueError(f"hiiregparam: Omega must be >= 0, got {omega!r}")

    def _check_nII_r1(self) -> None:
        """Ensure r1 >= rs(nII) (Omega >= 0), correcting or raising."""
        assert self.nII_ is not None and self.r1_ is not None
        rs = _rs(self._qH0_rate, self.nII_)
        if self.r1_ >= rs:
            return
        if not self.warn:
            nIIlim = np.sqrt(3.0 * self._qH0_rate / (4.0 * np.pi * _alphaB * _fe * self.r1_ ** 3))
            raise ValueError(
                "hiiregparam: nII too small for input value of r1; "
                f"for r1 = {self.r1_!r}, minimum nII is {nIIlim!r}")
        if self.fix_quantity != "r1":
            nIIlim = np.sqrt(3.0 * self._qH0_rate / (4.0 * np.pi * _alphaB * _fe * self.r1_ ** 3))
            nIInew = (1.0 + self.r0safety) * nIIlim
            warnings.warn(
                "hiiregparam: nII too small for input value of r1; "
                f"raising from {self.nII_!r} to {nIInew!r}")
            self.nII_ = nIInew
        else:
            r1new = (1.0 + self.r0safety) * rs
            warnings.warn(
                "hiiregparam: r1 too small for input value of nII; "
                f"raising from {self.r1_!r} to {r1new!r}")
            self.r1_ = r1new

    def _check_r1_U(self, r1: u.Quantity, nII_lim: u.Quantity) -> None:
        """Ensure U does not exceed the maximum (Omega=0) value for r1, correcting or raising."""
        assert self.U_ is not None
        Ulim = _U_from_nII_omega(self._qH0_rate, nII_lim, 0.0)
        if self.U_ <= Ulim:
            return
        if not self.warn:
            raise ValueError(
                "hiiregparam: U too large for input value of r1; "
                f"for r1 = {r1!r}, maximum U is {Ulim!r}")
        if self.fix_quantity != "r1":
            Unew = (1.0 - self.r0safety) * Ulim
            warnings.warn(
                "hiiregparam: U too large for input value of r1; "
                f"lowering from {self.U_!r} to {Unew!r}")
            self.U_ = Unew
        else:
            # Ulim(r1) is proportional to r1**(-1/2) (since it scales
            # as nII_lim(r1)**(1/3) and nII_lim(r1) itself scales as
            # r1**(-3/2)), so shrinking r1 raises Ulim to accommodate
            # the given (too-large) U
            target = self.U_ / (1.0 - self.r0safety)
            r1new = r1 * (Ulim / target).decompose().value ** 2
            warnings.warn(
                "hiiregparam: r1 too large for input value of U; "
                f"lowering from {r1!r} to {r1new!r}")
            self.r1_ = r1new

    def _check_U_U0(self) -> None:
        """Ensure U0 >= 2U (the Omega -> infinity asymptote), correcting or raising."""
        assert self.U_ is not None and self.U0_ is not None
        if self.U0_ >= 2.0 * self.U_:
            return
        if not self.warn:
            raise ValueError("hiiregparam: maximum value of U is U0/2")
        Ulim = (1.0 - self.r0safety) * 0.5 * self.U0_
        warnings.warn(
            "hiiregparam: U too large for input U0; maximum value of "
            f"U is U0/2; lowering U to {Ulim!r}")
        self.U_ = Ulim

    ##################################################################
    # Properties -- all derived from the canonical (nII, Omega) pair
    ##################################################################

    @property
    def nII(self) -> u.Quantity:
        """astropy.units.Quantity : number density of H nuclei, in cm^-3."""
        return self._nII_solved

    @nII.setter
    def nII(self, val: u.Quantity) -> None:
        if self.nII_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.nII_ = val
        self._nII_solved, self._omega_solved = self._solve()

    @property
    def r0(self) -> u.Quantity:
        """astropy.units.Quantity : inner radius, in cm."""
        return _rs(self._qH0_rate, self._nII_solved) * self._omega_solved ** (1.0 / 3.0)

    @r0.setter
    def r0(self, val: u.Quantity) -> None:
        if self.r0_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.r0_ = val
        self._nII_solved, self._omega_solved = self._solve()

    @property
    def r1(self) -> u.Quantity:
        """astropy.units.Quantity : outer radius, in cm."""
        return _rs(self._qH0_rate, self._nII_solved) * (1.0 + self._omega_solved) ** (1.0 / 3.0)

    @r1.setter
    def r1(self, val: u.Quantity) -> None:
        if self.r1_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.r1_ = val
        self._nII_solved, self._omega_solved = self._solve()

    @property
    def U(self) -> u.Quantity:
        """astropy.units.Quantity : volume-averaged ionization parameter (dimensionless)."""
        return _U_from_nII_omega(self._qH0_rate, self._nII_solved, self._omega_solved)

    @U.setter
    def U(self, val: u.Quantity | float) -> None:
        if self.U_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.U_ = u.Quantity(val, u.dimensionless_unscaled)
        self._nII_solved, self._omega_solved = self._solve()

    @property
    def Omega(self) -> u.Quantity:
        """astropy.units.Quantity : wind parameter (dimensionless)."""
        return self._omega_solved * u.dimensionless_unscaled

    @Omega.setter
    def Omega(self, val: u.Quantity | float) -> None:
        if self.Omega_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.Omega_ = u.Quantity(val, u.dimensionless_unscaled)
        self._nII_solved, self._omega_solved = self._solve()

    @property
    def U0(self) -> u.Quantity:
        """astropy.units.Quantity : ionization parameter at the inner radius (dimensionless)."""
        return _U0_from_r0_nII(self._qH0_rate, self.r0, self._nII_solved)

    @U0.setter
    def U0(self, val: u.Quantity | float) -> None:
        if self.U0_ is None:
            raise ValueError("hiiregparam: cannot add a parameter after instantiation")
        self.U0_ = u.Quantity(val, u.dimensionless_unscaled)
        self._nII_solved, self._omega_solved = self._solve()

    ##################################################################
    # Derived quantities
    ##################################################################

    def rS(self) -> u.Quantity:
        """
        Compute the Stromgren radius.

        Returns
        -------
        astropy.units.Quantity
            The Stromgren radius, in cm.
        """
        return _rs(self._qH0_rate, self.nII)

    def rch(self) -> u.Quantity:
        """
        Compute the Krumholz & Matzner (2009) characteristic radius.

        Returns
        -------
        astropy.units.Quantity
            r_ch, in cm -- the radius inside which radiation pressure
            is dynamically significant.
        """
        return (_alphaB / (12.0 * np.pi * _phi) * (_eps0 / (2.0 * _fe * _kB * _TII)) ** 2
            * _ftrap ** 2 * _psi ** 2 * self._qH0_rate / _c ** 2).to(u.cm)

    def zeta(self) -> u.Quantity:
        """
        Compute the Krumholz & Matzner (2009) zeta parameter.

        Returns
        -------
        astropy.units.Quantity
            zeta = r_ch / r1 (dimensionless). Values of zeta >> 1
            imply that radiation pressure is dynamically significant.
        """
        return (self.rch() / self.r1).decompose()

    def rKM(self, t: u.Quantity, n: u.Quantity) -> u.Quantity:
        """
        Compute the Krumholz & Matzner (2009) HII region expansion radius.

        Parameters
        ----------
        t : astropy.units.Quantity
            HII region age, convertible to Myr.
        n : astropy.units.Quantity
            Number density of the medium the HII region is expanding
            into, convertible to cm^-3.

        Returns
        -------
        astropy.units.Quantity
            The radius predicted by Krumholz & Matzner (2009) for an
            HII region of age t expanding into a medium of density n,
            in cm.
        """
        rch = self.rch()
        tch = np.sqrt(4.0 * np.pi * _muH * _mH * n * _c * rch ** 4
            / (3.0 * _ftrap * self._qH0_rate * _psi * _eps0))
        tau = (t / tch).decompose()
        xrad = (2.0 * tau ** 2) ** 0.25
        xgas = (49.0 * tau ** 2 / 36.0) ** (2.0 / 7.0)
        return (rch * (xrad ** 3.5 + xgas ** 3.5) ** (2.0 / 7.0)).to(u.cm)
