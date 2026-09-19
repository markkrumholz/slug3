/**
 * @file BindDecayChain.cpp
 * @author Mark Krumholz
 * @brief Python bindings for elem::DecayChain
 * @date 2026-09-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../elem/DecayChain.hpp"
#include "../elem/IsotopeData.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for vector conversions
#include <string_view>
#include <vector>

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view classDocstring =
    R"doc(The radioactive decay network of one isotope, and its Bateman equation solution.

Built once, from a single starting isotope, by following each
isotope's own decay daughters recursively until every branch reaches a
stable isotope -- products() then lists every isotope reachable this
way (including the starting one, at index 0), and yield_(t) gives each
one's own abundance, in atoms per atom of the starting isotope
initially present, after time t has elapsed.

The underlying math is the standard closed-form Bateman equation
solution for a linear decay chain, generalized only by reading the
branching ratio for each step directly off the relevant isotope's own
decay data (1 for a simple, non-branching decay). This assumes every
isotope in the chain has a distinct decay rate -- in particular, that
the starting isotope's own decay network does not branch into more
than one distinct stable end product, since two different stable
isotopes in the same chain would both have rate 0. Every isotope
actually used for nucleosynthetic yields in this codebase decays along
a single, non-reconverging path, so this is not a practical concern
here, but it is not checked for either.)doc";

static constexpr std::string_view constructorDocstring =
    R"doc(Build the decay chain for one isotope.

Parameters
----------
isotope : IsotopeData
    The isotope to build the decay chain for.

Throws
------
RuntimeError
    If a daughter named by isotope's own decay data (or that of any of
    its descendants) is not found in the global isotope table.)doc";

static constexpr std::string_view productsDocstring =
    R"doc(Get every isotope in this decay chain.

Returns
-------
products : list of IsotopeData
    The starting isotope (index 0) followed by every isotope reachable
    from it by following decay daughters recursively, in the same
    order (and of the same length) as yield_(t)'s own returned list --
    products()[j] is the isotope yield_(t)[j] gives the abundance of.)doc";

static constexpr std::string_view yieldDocstring =
    R"doc(Get each product isotope's own abundance after time t.

Parameters
----------
t : float
    Elapsed time since one atom of products()[0] (the starting
    isotope) was present, and no atoms of any other product yet, in
    the same time units as every isotope's own lifetime().

Returns
-------
yields : list of float
    One abundance per entry of products(), in the same order:
    yields[j] is the number of atoms of products()[j] expected to be
    present at time t, per atom of products()[0] present at t=0.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindDecayChain(py::module_& m)
{
    py::class_<elem::DecayChain, py::smart_holder>(m, "DecayChain", classDocstring.data())
        .def(py::init<const elem::IsotopeData&>(),
                constructorDocstring.data(), py::arg("isotope"))
        .def("products",
                [](const elem::DecayChain& self) -> std::vector<const elem::IsotopeData*>
                {
                    std::vector<const elem::IsotopeData*> result;
                    result.reserve(self.products().size());
                    for (const auto& iso : self.products()) { result.push_back(&iso.get()); }
                    return result;
                },
                productsDocstring.data(), py::return_value_policy::reference)
        .def("yield_", &elem::DecayChain::yield, yieldDocstring.data(), py::arg("t"));
}
// NOLINTEND(misc-include-cleaner)
