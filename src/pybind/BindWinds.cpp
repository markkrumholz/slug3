/**
 * @file BindWinds.cpp
 * @author Mark Krumholz
 * @brief Python bindings for feedback::Winds
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../feedback/Winds.hpp"
#include "../io/SimControls.hpp"
#include "../specsyn/Specsyn.hpp"
#include <algorithm>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view classDocstring = R"doc(A calculator for stellar wind feedback quantities.

Reads the choice of wind model live from the SimControls it was built
against (SimControls.wrWindModel and SimControls.obWindModel), on every
call, so changing that choice takes effect immediately without
rebuilding this object.)doc";

static constexpr std::string_view constructorDocstring = R"doc(Construct a Winds.

Parameters
----------
controls : SimControls
    Simulation controls this Winds reads its wind model choices from,
    live, for the rest of its lifetime. Kept alive at least as long as
    this Winds.)doc";

static constexpr std::string_view vWindWRDocstring = R"doc(Compute the terminal wind velocity of a Wolf-Rayet star.

Parameters
----------
props : list of float
    Stellar properties, in track order: mass (Msun), mdot (Msun/yr),
    logL (log10 Lsun), logTe (log10 K), h_surf, he_surf, c_surf,
    n_surf, o_surf -- exactly 9 elements, the same layout Specsyn.spec()
    takes. The star is assumed to already be a Wolf-Rayet star; this is
    not checked.

Returns
-------
vwind : float
    Terminal wind velocity, in cm/s, from whichever model the
    SimControls this Winds was built against selects via its
    wrWindModel property: "none" gives exactly 0; "l_over_c" gives
    L / (mdot c); "nugis_lamers_00" gives the Nugis & Lamers (2000)
    prescription, clamped to 740-5500 km/s.

Raises
------
RuntimeError
    If props does not have exactly 9 elements.)doc";

static constexpr std::string_view vWindOBDocstring = R"doc(Compute the terminal wind velocity of an O or B star.

Parameters
----------
props : list of float
    Stellar properties, in the same 9-element layout vWindWR() takes.
    The star is assumed to already be an O or B star; this is not
    checked.
feh : float
    The star's metallicity [Fe/H], used as a proxy for log(Z/Zsun).

Returns
-------
vwind : float
    Terminal wind velocity, in cm/s, from whichever model the
    SimControls this Winds was built against selects via its
    obWindModel property: "none" gives exactly 0; "vink_01" gives
    1.3 or 2.6 times the effective escape speed
    sqrt(2 G M (1 - Gamma_e) / R) (below or above the Vink et al. 2001
    bistability jump temperature, respectively), times (10^feh)^0.13,
    or exactly 0 if the electron-scattering Eddington factor
    Gamma_e >= 1; "vink_sander_21" gives the Vink & Sander
    (2021) fits, again switching at the same jump temperature.

Raises
------
RuntimeError
    If props does not have exactly 9 elements.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)

// Convert props list to StarData, checking length -- see
// BindSpecsyn.cpp's own identical helper, duplicated here for the same
// reason BindYields.cpp duplicates BindYieldChannel.cpp's
// toIsotopeList(): a small, private implementation detail of each
// translation unit's own bindings. method names the calling method,
// for the error message
static auto toStarData(const std::vector<double>& props, const std::string_view method) -> specsyn::Specsyn::StarData
{
    if (props.size() != nQty)
    {
        throw std::runtime_error(
            "Winds." + std::string(method) + ": props must have exactly " + std::to_string(nQty) +
            " elements (mass, mdot, logL, logTe, h_surf, he_surf, c_surf, n_surf, o_surf)");
    }
    specsyn::Specsyn::StarData starData{};
    std::ranges::copy(props, starData.begin());
    return starData;
}

void bindWinds(py::module_& m)
{
    py::class_<feedback::Winds, py::smart_holder>(m, "Winds", classDocstring.data())
        .def(py::init<const io::SimControls&>(),
                constructorDocstring.data(),
                py::arg("controls"),
                // Keep controls (index 2: 1 = self) alive at least as
                // long as this Winds, which stores a live reference to
                // it -- see Yields's own identical py::keep_alive<1, 2>
                // in BindYields.cpp
                py::keep_alive<1, 2>())
        .def("vWindWR",
                [](const feedback::Winds& self, const std::vector<double>& props) -> double
                { return self.vWindWR(toStarData(props, "vWindWR")); },
                vWindWRDocstring.data(), py::arg("props"))
        .def("vWindOB",
                [](const feedback::Winds& self, const std::vector<double>& props, const double feh) -> double
                { return self.vWindOB(toStarData(props, "vWindOB"), feh); },
                vWindOBDocstring.data(), py::arg("props"), py::arg("feh"));
}
// NOLINTEND(misc-include-cleaner)
