/**
 * @file BindYields.cpp
 * @author Mark Krumholz
 * @brief Python bindings for yields::Yields
 * @date 2026-09-18
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../elem/IsotopeData.hpp"
#include "../io/SimControls.hpp"
#include "../yields/YieldChannel.hpp"
#include "../yields/YieldCommons.hpp"
#include "../yields/Yields.hpp"
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <string>
#include <string_view>
#include <vector>

// See BindYieldChannel.cpp's own identical helper -- duplicated here
// rather than shared via Bindings.hpp, since it's a small, private
// implementation detail of these two translation units' own bindings,
// not part of the cross-file infrastructure Bindings.hpp otherwise
// holds (resolveControls() and friends).
static auto toIsotopeList(const std::vector<const elem::IsotopeData*>& isotopes) -> yields::IsotopeList
{
    yields::IsotopeList result;
    result.reserve(isotopes.size());
    for (const auto* iso : isotopes) { result.emplace_back(*iso); }
    return result;
}

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view classDocstring =
    R"doc(Chains together the nucleosynthetic yield channels a SimControls requests.

Built from controls.yieldChannels(): one YieldChannel per descriptor,
added via addChannel(). The constructor also calls rebuildYieldGrid()
once, which collects every loaded channel's own isotopesOrig() into
one deduplicated, sorted isotopes() (the union of every isotope any
requested channel covers), then pushes that same isotopes() back down
into every channel's own rebuildYieldGrid() -- so every YieldChannel
this Yields owns ends up synchronized onto the same isotope list, in
the same order.)doc";

static constexpr std::string_view constructorDocstring =
    R"doc(Construct a Yields from a SimControls's own yieldChannels().

Parameters
----------
controls : SimControls
    Simulation controls this Yields reads its yield-channel descriptors
    (controls.yieldChannels()) and [Fe/H] range (controls.fehDist())
    from, live, for the rest of its lifetime. Must outlive this Yields.
registry_name : str, optional
    Name of the yield registry file.

Throws
------
RuntimeError
    If any descriptor in controls.yieldChannels() names a channel/model
    not found in registry_name, or whose [Fe/H] range doesn't cover
    controls.fehDist().

Details
-------
Prints a warning (to stdout, not thrown) if controls.yieldChannels()
requests the same channel more than once -- a suspicious pattern that's
easy to hit by accident, but not forbidden outright, since there are
legitimate reasons for it too (e.g. comparing two different yield
tables for the same channel).)doc";

static constexpr std::string_view addChannelDocstring =
    R"doc(Add one YieldChannel, built from a descriptor, to yieldChannels().

Parameters
----------
descriptor : YieldChannelDescriptor
    Which channel/model to load, and the mass range it should cover.

Throws
------
RuntimeError
    If descriptor.channel/model_name is not found in the registry
    named by registryName(), or if this Yields's own SimControls's
    [Fe/H] range lies outside the range actually available for that
    channel/model.)doc";

static constexpr std::string_view rebuildYieldGridDocstring =
    R"doc(Rebuild isotopes() from yieldChannels(), then push it back into every channel.

Parameters
----------
isotopes : list of IsotopeData, optional
    Restricts isotopes() to its own intersection with this list; an
    empty list (the default) means "keep every isotope any loaded
    channel tabulates".

Throws
------
RuntimeError
    If isotopes is non-empty but matches nothing any loaded channel
    tabulates, leaving isotopes() empty.

Details
-------
Called once by the constructor, right after every requested channel
has been added; also callable directly, e.g. after addChannel() adds a
further channel to an already-built Yields, to re-synchronize every
channel -- including ones added earlier -- onto the new, larger union
of isotopes, or to change which isotopes subset is kept.)doc";

static constexpr std::string_view registryNameDocstring =
    R"doc(Get the name of the yield registry this Yields reads from.

Returns
-------
registry_name : str)doc";

static constexpr std::string_view yieldChannelsDocstring =
    R"doc(Get the yield channels loaded so far.

Returns
-------
yield_channels : list of YieldChannel
    In the same order as controls().yieldChannels().)doc";

static constexpr std::string_view isotopesDocstring =
    R"doc(Get the union of every loaded channel's own isotopesOrig().

Returns
-------
isotopes : list of IsotopeData
    Every isotope that appears in at least one of yieldChannels()'s own
    isotopesOrig() lists, deduplicated and sorted -- once
    rebuildYieldGrid() has run (always true after the constructor
    itself returns), this is also exactly what every yieldChannels()
    entry's own isotopes() equals, in the same order.)doc";

static constexpr std::string_view yieldDocstring =
    R"doc(Get every channel's own yield, as one (n_channels, len(isotopes())) array.

Parameters
----------
mass : float
    Stellar mass (Msun); need not satisfy every yieldChannels() entry's
    own hasYield(mass).
feh : float
    [Fe/H]; need not lie within every yieldChannels() entry's own
    feH() range.

Returns
-------
yields : list of list of float
    yields[i][j] is the yield (Msun) of isotopes()[j] from
    yieldChannels()[i], for a star of the given mass/feh. Row i is left
    all zero if yieldChannels()[i] doesn't cover mass/feh.

Throws
------
RuntimeError
    If any yieldChannels() entry's own yield_() throws (e.g.
    rebuildYieldGrid() was never called on that particular channel).)doc";

static constexpr std::string_view yieldSumDocstring =
    R"doc(Get the sum, over every channel, of yield_(mass, feh).

Parameters
----------
mass : float
    Stellar mass (Msun); see yield_()'s own mass parameter.
feh : float
    [Fe/H]; see yield_()'s own feh parameter.

Returns
-------
yield_sum : list of float
    One yield (Msun) per entry of isotopes(), in the same order:
    result[j] = sum over i of yield_(mass, feh)[i][j].

Throws
------
RuntimeError
    Under the same conditions as yield_().)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindYields(py::module_& m)
{
    py::class_<yields::Yields, py::smart_holder>(m, "Yields", classDocstring.data())
        .def(py::init<const io::SimControls&, std::string>(),
                constructorDocstring.data(),
                py::arg("controls"), py::arg("registry_name") = yields::defaultRegistry,
                // Keep controls (index 2: 1 = self) alive at least as
                // long as this Yields, which stores a live reference
                // to it rather than copying its yieldChannels()/
                // fehDist() out -- see Yields's own controls_ member,
                // and Extinct's own identical py::keep_alive<1, 3> in
                // BindExtinct.cpp for the same rationale.
                py::keep_alive<1, 2>())
        .def("addChannel", &yields::Yields::addChannel,
                addChannelDocstring.data(), py::arg("descriptor"))
        .def("rebuildYieldGrid",
                [](yields::Yields& self, const std::vector<const elem::IsotopeData*>& isotopes)
                {
                    self.rebuildYieldGrid(toIsotopeList(isotopes));
                },
                rebuildYieldGridDocstring.data(),
                py::arg("isotopes") = std::vector<const elem::IsotopeData*>{})
        .def("registryName", &yields::Yields::registryName, registryNameDocstring.data())
        .def("yieldChannels",
                [](const yields::Yields& self) -> std::vector<const yields::YieldChannel*>
                {
                    std::vector<const yields::YieldChannel*> result;
                    result.reserve(self.yieldChannels().size());
                    for (const auto& channel : self.yieldChannels()) { result.push_back(channel.get()); }
                    return result;
                },
                yieldChannelsDocstring.data(), py::return_value_policy::reference_internal)
        .def("isotopes",
                [](const yields::Yields& self) -> std::vector<const elem::IsotopeData*>
                {
                    std::vector<const elem::IsotopeData*> result;
                    result.reserve(self.isotopes().size());
                    for (const auto& iso : self.isotopes()) { result.push_back(&iso.get()); }
                    return result;
                },
                isotopesDocstring.data(), py::return_value_policy::reference)
        .def("yield_",
                [](const yields::Yields& self, double mass, double feH) -> std::vector<std::vector<double>>
                {
                    const auto [view, data] = self.yield(mass, feH);
                    std::vector<std::vector<double>> result(
                        view.extent(0), std::vector<double>(view.extent(1)));
                    for (std::size_t i = 0; i < view.extent(0); ++i)
                    {
                        for (std::size_t j = 0; j < view.extent(1); ++j) { result[i][j] = view[i, j]; }
                    }
                    return result;
                },
                yieldDocstring.data(), py::arg("mass"), py::arg("feh"))
        .def("yieldSum", &yields::Yields::yieldSum,
                yieldSumDocstring.data(), py::arg("mass"), py::arg("feh"));
}
// NOLINTEND(misc-include-cleaner)
