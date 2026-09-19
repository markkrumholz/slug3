/**
 * @file BindYieldChannel.cpp
 * @author Mark Krumholz
 * @brief Python bindings for elem::IsotopeData, elem::isotopeTable(z, a),
 *   yields::Channel, yields::YieldChannelDescriptor, and yields::YieldChannel
 * @date 2026-09-18
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../elem/ElemCommons.hpp"
#include "../elem/IsotopeData.hpp"
#include "../elem/IsotopeTable.hpp"
#include "../yields/YieldChannel.hpp"
#include "../yields/YieldCommons.hpp"
#include <cstddef>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector/optional conversions
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Convert a Python-facing list of IsotopeData pointers (each one
// resolved, via isotopeTable(z, a) or an existing YieldChannel's/
// Yields's own isotopes()/isotopesOrig(), to a reference into the
// single, global elem::isotopeTable() -- so a raw pointer is always
// safe to dereference, with no lifetime tie to anything else) into the
// elem::IsotopeList rebuildYieldGrid() itself takes. Shared by
// YieldChannel::rebuildYieldGrid()'s and Yields::rebuildYieldGrid()'s
// own bindings (see BindYields.cpp for the latter).
static auto toIsotopeList(const std::vector<const elem::IsotopeData*>& isotopes) -> elem::IsotopeList
{
    elem::IsotopeList result;
    result.reserve(isotopes.size());
    for (const auto* iso : isotopes)
    {
        if (iso == nullptr) { throw py::value_error("isotopes must not contain None"); } // NOLINT(misc-include-cleaner) -- py::value_error is provided by pybind11.h (already included above); clang-tidy's IWYU mapping just doesn't know that
        result.emplace_back(*iso);
    }
    return result;
}

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view isotopeDataClassDocstring =
    R"doc(One isotope's identity and radioactive decay data.

Every instance returned anywhere in this module (isotopeTable(),
YieldChannel.isotopes()/isotopesOrig(), Yields.isotopes()) is a
reference into the single, global isotope table, resolved once at
program startup -- so two calls naming the same (Z, A) always return
objects comparing equal by identity from C++'s own perspective, and
neither this module nor a caller ever constructs one directly (there
is no exposed constructor).)doc";

static constexpr std::string_view zDocstring = R"doc(Get the atomic number.

Returns
-------
Z : int
    The atomic number.)doc";

static constexpr std::string_view aDocstring = R"doc(Get the mass number.

Returns
-------
A : int
    The mass number.)doc";

static constexpr std::string_view labelDocstring = R"doc(Get this isotope formatted as "<symbol><A>".

Returns
-------
label : str
    The element symbol immediately followed by the mass number, e.g.
    "Fe56", "H1", "Na22".)doc";

static constexpr std::string_view lifetimeDocstring = R"doc(Get the radioactive decay lifetime.

Returns
-------
lifetime : float
    Radioactive decay lifetime in years; 0 indicates a stable
    isotope.)doc";

static constexpr std::string_view stableDocstring = R"doc(Check whether this isotope is stable.

Returns
-------
stable : bool
    True if lifetime() == 0.)doc";

static constexpr std::string_view isotopeTableDocstring = R"doc(Look up a single isotope by (Z, A) in the global isotope table.

Parameters
----------
z : int
    Atomic number.
a : int
    Mass number.

Returns
-------
isotope : IsotopeData
    The requested isotope's data.

Throws
------
KeyError
    If no isotope with this (Z, A) exists in the table.

Details
-------
The returned IsotopeData is a reference into the single, global isotope
table, resolved once at program startup -- see its own class docstring.
The most common use for this function is building a list of isotopes to
pass to YieldChannel.rebuildYieldGrid()/Yields.rebuildYieldGrid(), to
restrict which isotopes end up tabulated.)doc";

static constexpr std::string_view yieldChannelTypeDocstring = R"doc(A known nucleosynthetic yield channel.

More channels will be added here as more yield sources are imported.)doc";

static constexpr std::string_view descriptorClassDocstring =
    R"doc(Everything needed to select and size one YieldChannel.

Members have the same meaning as the like-named parameters of
YieldChannel's own constructor -- see its own docstring for each.)doc";

static constexpr std::string_view descriptorConstructorDocstring =
    R"doc(Construct a YieldChannelDescriptor.

Parameters
----------
channel : YieldChannelType
    Which nucleosynthetic channel to load (e.g. YieldChannelType.ccsn).
model_name : str
    Name of the yield model (e.g. "sukhbold16").
m_min : float, optional
    Minimum stellar mass this channel should cover; None (the default)
    means the model's own native minimum.
m_max : float, optional
    Maximum stellar mass this channel should cover; None (the default)
    means the model's own native maximum.)doc";

static constexpr std::string_view yieldChannelClassDocstring =
    R"doc(Nucleosynthetic yield data for one channel (e.g. ccsn), one model (e.g. sukhbold16), over a range of [Fe/H].)doc";

static constexpr std::string_view yieldChannelConstructorDocstring =
    R"doc(Construct a YieldChannel from a yield model on disk.

Parameters
----------
descriptor : YieldChannelDescriptor
    Which channel/model to load, and the mass range it should cover --
    descriptor.m_min/m_max mean exactly what rebuildYieldGrid()'s own
    m_min/m_max parameters do.
feh_min : float
    Minimum [Fe/H] value.
feh_max : float
    Maximum [Fe/H] value.
registry_name : str, optional
    Name of the yield registry file.

Throws
------
RuntimeError
    If descriptor.channel/model_name is not found in the registry, if
    feh_min or feh_max lies outside the [Fe/H] range actually available
    for that channel/model, or if the underlying HDF5 file cannot be
    read or is malformed.

Details
-------
Does not itself populate masses()/isotopes()/build a usable yield grid
-- rebuildYieldGrid() must be called at least once (with whatever
m_min/m_max/isotopes are wanted) before yield_() can be called.)doc";

static constexpr std::string_view rebuildYieldGridDocstring =
    R"doc(Rebuild masses()/isotopes() (and the underlying yield grid) from this channel's own native data.

Parameters
----------
m_min : float, optional
    Minimum stellar mass this channel should cover; None (the default)
    means massesOrig()[0].
m_max : float, optional
    Maximum stellar mass this channel should cover; None (the default)
    means massesOrig()[-1].
isotopes : list of IsotopeData, optional
    The isotope list the yield grid's own isotope axis should be built
    over; an empty list (the default) means "use isotopesOrig() itself,
    unchanged". A non-empty list lets a caller reorder, subset, or
    extend the isotope list this channel's own data is built over --
    e.g. to match another channel's own isotopes, before chaining both
    together (see the Yields class).

Throws
------
ValueError
    If the resolved m_min or m_max is not finite and strictly positive,
    or if the resolved m_min is not strictly less than the resolved
    m_max.

Details
-------
Must be called at least once before yield_() can be called. Public so
an already-constructed YieldChannel can be rebuilt (e.g. after changing
which mass range it should cover) without building a new one.)doc";

static constexpr std::string_view channelDocstring = R"doc(Get which nucleosynthetic channel this object holds.

Returns
-------
channel : YieldChannelType
    descriptor().channel -- the channel named by the descriptor passed
    to the constructor.)doc";

static constexpr std::string_view descriptorDocstring =
    R"doc(Get a copy of the descriptor this channel was built from.

Returns
-------
descriptor : YieldChannelDescriptor
    A copy of the descriptor passed to the constructor, except that
    m_min/m_max instead reflect whichever mass range was actually last
    requested -- either still the constructor's own descriptor.m_min/
    m_max, if rebuildYieldGrid() has never been called with an explicit
    m_min/m_max of its own, or whatever it was called with most
    recently otherwise. A copy, not a live view: it does not update if
    this channel's own descriptor changes on a later rebuildYieldGrid()
    call.)doc";

static constexpr std::string_view massesDocstring = R"doc(Get the stellar masses this channel's yields are tabulated at.

Returns
-------
masses : list of float
    Stellar masses (Msun), ascending -- this is the grid yield_()/
    hasYield() actually use, which can extend below/above
    massesOrig()'s own range (extrapolated) or be a narrower sub-range
    of it. Empty until rebuildYieldGrid() is called at least once.)doc";

static constexpr std::string_view massesOrigDocstring =
    R"doc(Get the stellar masses natively tabulated by the underlying model.

Returns
-------
masses_orig : list of float
    Stellar masses (Msun), ascending, exactly as read from the model's
    own HDF5 file -- unlike masses(), never affected by
    rebuildYieldGrid()'s own m_min/m_max, and already populated once
    the constructor returns.)doc";

static constexpr std::string_view isotopesDocstring =
    R"doc(Get the isotopes this channel's yield grid is actually tabulated for.

Returns
-------
isotopes : list of IsotopeData
    The isotopes, in the same order as yield_()'s own return value --
    either isotopesOrig() itself, or whatever isotope list was last
    passed to rebuildYieldGrid(). Empty until rebuildYieldGrid() is
    called at least once.)doc";

static constexpr std::string_view isotopesOrigDocstring =
    R"doc(Get the isotopes natively tabulated by the underlying model.

Returns
-------
isotopes_orig : list of IsotopeData
    The isotopes, exactly as read from the model's own HDF5 file --
    unlike isotopes(), never affected by rebuildYieldGrid(), and
    already populated once the constructor returns.)doc";

static constexpr std::string_view fehDocstring = R"doc(Get the [Fe/H] values this channel's yields are tabulated at.

Returns
-------
feh : list of float
    The [Fe/H] values, ascending.)doc";

static constexpr std::string_view hasYieldDocstring =
    R"doc(Check whether a stellar mass falls within this channel's mass grid.

Parameters
----------
mass : float
    Stellar mass to check (Msun).

Returns
-------
has_yield : bool
    True if mass lies within [masses()[0], masses()[-1]]; always False
    before rebuildYieldGrid() has been called at least once.)doc";

static constexpr std::string_view yieldDocstring =
    R"doc(Get the yield of every isotope for a star of given mass and [Fe/H].

Parameters
----------
mass : float
    Stellar mass (Msun); must satisfy hasYield(mass).
feh : float
    [Fe/H]; must lie within [feH()[0], feH()[-1]].

Returns
-------
yields : list of float
    One yield (Msun) per entry of isotopes(), in the same order,
    bilinearly interpolated in (feh, mass).

Throws
------
RuntimeError
    If rebuildYieldGrid() has never been called.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindYieldChannel(py::module_& m)
{
    py::class_<elem::IsotopeData, py::smart_holder>(m, "IsotopeData", isotopeDataClassDocstring.data())
        .def("Z", &elem::IsotopeData::Z, zDocstring.data())
        .def("A", &elem::IsotopeData::A, aDocstring.data())
        .def("label", &elem::IsotopeData::label, labelDocstring.data())
        .def("lifetime", &elem::IsotopeData::lifetime, lifetimeDocstring.data())
        .def("stable", &elem::IsotopeData::stable, stableDocstring.data());

    m.def("isotopeTable",
            [](unsigned int z, unsigned int a) -> const elem::IsotopeData&
            {
                try
                {
                    return elem::isotopeTable(z, a);
                }
                catch (const std::out_of_range&)
                {
                    throw py::key_error(
                        "isotopeTable: no isotope with Z = " + std::to_string(z) +
                        ", A = " + std::to_string(a));
                }
            },
            isotopeTableDocstring.data(),
            py::arg("z"), py::arg("a"), py::return_value_policy::reference);

    py::enum_<yields::Channel>(m, "YieldChannelType", yieldChannelTypeDocstring.data())
        .value("ccsn", yields::Channel::ccsn_, "Core-collapse supernova ejecta")
        .value("massive_star_winds", yields::Channel::massiveStarWinds_,
                "Massive star winds (pre-supernova mass loss)");

    py::class_<yields::YieldChannelDescriptor, py::smart_holder>(
            m, "YieldChannelDescriptor", descriptorClassDocstring.data())
        .def(py::init(
                [](yields::Channel channel, std::string modelName,
                   std::optional<double> mMin, std::optional<double> mMax)
                {
                    return yields::YieldChannelDescriptor{
                        channel, std::move(modelName), mMin, mMax};
                }),
                descriptorConstructorDocstring.data(),
                py::arg("channel"), py::arg("model_name"),
                py::arg("m_min") = std::nullopt, py::arg("m_max") = std::nullopt)
        .def_readwrite("channel", &yields::YieldChannelDescriptor::channel_)
        .def_readwrite("model_name", &yields::YieldChannelDescriptor::modelName_)
        .def_readwrite("m_min", &yields::YieldChannelDescriptor::mMin_)
        .def_readwrite("m_max", &yields::YieldChannelDescriptor::mMax_);

    py::class_<yields::YieldChannel, py::smart_holder>(m, "YieldChannel", yieldChannelClassDocstring.data())
        .def(py::init<const yields::YieldChannelDescriptor&, double, double, const std::string&>(),
                yieldChannelConstructorDocstring.data(),
                py::arg("descriptor"), py::arg("feh_min"), py::arg("feh_max"),
                py::arg("registry_name") = yields::defaultRegistry)
        .def("rebuildYieldGrid",
                [](yields::YieldChannel& self, std::optional<double> mMin, std::optional<double> mMax,
                   const std::vector<const elem::IsotopeData*>& isotopes)
                {
                    self.rebuildYieldGrid(mMin, mMax, toIsotopeList(isotopes));
                },
                rebuildYieldGridDocstring.data(),
                py::arg("m_min") = std::nullopt, py::arg("m_max") = std::nullopt,
                py::arg("isotopes") = std::vector<const elem::IsotopeData*>{})
        .def("channel", &yields::YieldChannel::channel, channelDocstring.data())
        .def("descriptor", &yields::YieldChannel::descriptor, descriptorDocstring.data())
        .def("masses", &yields::YieldChannel::masses, massesDocstring.data())
        .def("massesOrig", &yields::YieldChannel::massesOrig, massesOrigDocstring.data())
        .def("isotopes",
                [](const yields::YieldChannel& self) -> std::vector<const elem::IsotopeData*>
                {
                    std::vector<const elem::IsotopeData*> result;
                    result.reserve(self.isotopes().size());
                    for (const auto& iso : self.isotopes()) { result.push_back(&iso.get()); }
                    return result;
                },
                isotopesDocstring.data(), py::return_value_policy::reference)
        .def("isotopesOrig",
                [](const yields::YieldChannel& self) -> std::vector<const elem::IsotopeData*>
                {
                    std::vector<const elem::IsotopeData*> result;
                    result.reserve(self.isotopesOrig().size());
                    for (const auto& iso : self.isotopesOrig()) { result.push_back(&iso.get()); }
                    return result;
                },
                isotopesOrigDocstring.data(), py::return_value_policy::reference)
        .def("feH", &yields::YieldChannel::feH, fehDocstring.data())
        .def("hasYield", &yields::YieldChannel::hasYield, hasYieldDocstring.data(), py::arg("mass"))
        .def("yield_", &yields::YieldChannel::yield, yieldDocstring.data(),
                py::arg("mass"), py::arg("feh"));
}
// NOLINTEND(misc-include-cleaner)
