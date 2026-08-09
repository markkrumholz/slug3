"""
phot_convert.py

Implements phot_convert, a convenience wrapper around the compiled
PhotConvert binding that deduces the source photometric system from
an astropy Quantity's own unit.
"""

from typing import cast

from astropy import units as u

from ._slug import Filter, PhotConvert

# The unit phot_convert re-attaches to its result for each photometric
# system, and (read in reverse) the units phot_convert recognizes when
# deducing a Quantity's own system.
_PHOT_SYSTEM_UNITS: dict[str, u.UnitBase] = {
    "Flambda": u.erg / u.s / u.AA,
    "Fnu": u.Jy,
    "ST": u.STmag,
    "AB": u.ABmag,
    "Vega": u.mag,
}

# (phot_from, phot_to) pairs whose underlying PhotConvert<From, To>
# specialization doesn't actually use its wl argument (Fnu<->AB and
# Flambda<->ST have fixed, wavelength-independent zero points;
# Flambda<->Vega uses fluxVega instead) -- see PhotCommons.hpp's own
# comments on each specialization for the physics. Every other pair
# genuinely depends on wavelength.
_WL_OPTIONAL_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("Fnu", "AB"), ("AB", "Fnu"),
    ("Flambda", "ST"), ("ST", "Flambda"),
    ("Flambda", "Vega"), ("Vega", "Flambda"),
})


def _detect_phot_system(unit: u.UnitBase) -> str:
    """
    Deduce a phot_convert phot_from/phot_to string from a unit.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit to deduce a photometric system from: mag(ST),
        mag(AB), a bare magnitude, a spectral flux density like Jy, or
        a specific luminosity like erg/(s Angstrom).

    Returns
    -------
    str
        One of "Flambda", "Fnu", "ST", "AB", or "Vega".

    Raises
    ------
    ValueError
        If unit doesn't match any recognized photometric system --
        e.g. Lsun (Lbol) or photon/s (an idealized photon-count
        filter), neither of which is a convertible photometric
        quantity at all.
    """
    if isinstance(unit, u.MagUnit):
        physical = unit.physical_unit
        if physical == u.Unit("ST"):
            return "ST"
        if physical == u.Unit("AB"):
            return "AB"
    elif unit.is_equivalent(u.mag):
        return "Vega"
    elif unit.is_equivalent(u.Jy):
        return "Fnu"
    elif unit.is_equivalent(_PHOT_SYSTEM_UNITS["Flambda"]):
        return "Flambda"

    raise ValueError(
        f"phot_convert: cannot deduce a photometric system from unit "
        f"{unit!r} (expected mag(ST), mag(AB), a bare magnitude, a "
        "spectral flux density like Jy, or a specific luminosity like "
        "erg/(s Angstrom))")


def phot_convert(phot: u.Quantity, phot_to: str,
    wl: u.Quantity | float | None = None, filt: Filter | None = None) -> u.Quantity:
    """
    Convert a photometric Quantity from its own system to another.

    Parameters
    ----------
    phot : astropy.units.Quantity
        The photometry to convert; its own unit determines the source
        photometric system (see _detect_phot_system()).
    phot_to : str
        The photometric system to convert phot to: one of "Flambda",
        "Fnu", "ST", "AB", or "Vega".
    wl : astropy.units.Quantity or float, optional
        The wavelength(s) at which phot is evaluated (a plain float is
        interpreted as Angstrom); broadcast against phot, so e.g. a
        whole spectrum can be converted in one call by passing
        matching phot and wl arrays. Required only if the (phot_from,
        phot_to) pair actually depends on wavelength -- every pair
        except Fnu<->AB, Flambda<->ST, and Flambda<->Vega.
    filt : Filter, optional
        A filter whose fluxVega() gives the Vega zero point to use;
        required if phot's own system or phot_to is "Vega", ignored
        otherwise.

    Returns
    -------
    astropy.units.Quantity
        phot converted to phot_to, with phot_to's own unit attached.
        Returns phot unchanged if it is already in phot_to's system.

    Raises
    ------
    ValueError
        If phot's unit doesn't match a recognized photometric system,
        if wl is required (see above) but not given, or if filt is
        required (phot's own system or phot_to is "Vega") but not
        given.
    RuntimeError
        If phot_to is not a recognized photometric system.
    """
    phot_from = _detect_phot_system(phot.unit)
    if phot_from == phot_to:
        return phot

    if (phot_from, phot_to) not in _WL_OPTIONAL_PAIRS and wl is None:
        raise ValueError(
            f"phot_convert: converting {phot_from} to {phot_to} requires a "
            "wavelength (wl) -- this pair depends on wavelength, unlike "
            "Fnu<->AB, Flambda<->ST, or Flambda<->Vega")
    if wl is None:
        wl_value = 0.0
    elif isinstance(wl, u.Quantity):
        # isinstance already guarantees this at runtime; cast because
        # astropy ships no type stubs, so pyright can't narrow a
        # Quantity | float union against it on its own (see
        # pyrightconfig.json's own comment on useLibraryCodeForTypes)
        wl_value = cast(u.Quantity, wl).to(u.AA).value
    else:
        wl_value = wl

    if "Vega" in (phot_from, phot_to) and filt is None:
        raise ValueError(
            f"phot_convert: converting {phot_from} to {phot_to} requires a "
            "filter (filt), to look up its own fluxVega()")

    result = PhotConvert(phot_from, phot_to, phot.value, wl_value, filt)
    return result * _PHOT_SYSTEM_UNITS[phot_to]
