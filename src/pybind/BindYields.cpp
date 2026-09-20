/**
 * @file BindYields.cpp
 * @author Mark Krumholz
 * @brief Python bindings for yields::Yields
 * @date 2026-09-18
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "Bindings.hpp"
#include "../elem/ElemCommons.hpp"
#include "../elem/IsotopeData.hpp"
#include "../io/SimControls.hpp"
#include "../yields/YieldChannel.hpp"
#include "../yields/YieldCommons.hpp"
#include "../yields/Yields.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// See BindYieldChannel.cpp's own identical helper -- duplicated here
// rather than shared via Bindings.hpp, since it's a small, private
// implementation detail of these two translation units' own bindings,
// not part of the cross-file infrastructure Bindings.hpp otherwise
// holds (resolveControls() and friends).
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
    channel/model.

Details
-------
Also calls rebuildYieldGrid() afterward, so isotopes() (and every
channel this Yields owns, including the new one) stays synchronized.)doc";

static constexpr std::string_view addChannelPointerDocstring =
    R"doc(Add one already-built YieldChannel to yieldChannels().

Parameters
----------
channel : YieldChannel
    The channel to add; ownership is transferred to this Yields, so
    channel is no longer usable from Python after this call. Must not
    be None.

Throws
------
ValueError
    If channel is None.

Details
-------
Unlike addChannel(), which builds a fresh YieldChannel from a
descriptor, this overload moves an already-built channel into place.
Also calls rebuildYieldGrid(), so isotopes() and every channel this
Yields owns (including channel itself) stay synchronized afterward.)doc";

static constexpr std::string_view deleteChannelDocstring =
    R"doc(Remove one entry from yieldChannels().

Parameters
----------
index : int
    Index, into yieldChannels(), of the channel to remove.

Throws
------
IndexError
    If index is out of range for yieldChannels().

Details
-------
Also calls rebuildYieldGrid() afterward, so isotopes() (and every
remaining channel) stays synchronized once the removed channel's own
isotopes no longer count toward the union. A YieldChannel object
previously read back from yieldChannels() stays fully valid even if it
happens to be the entry index removes -- it's simply no longer one of
the entries yieldChannels() itself returns afterward.)doc";

static constexpr std::string_view setChannelsPointersDocstring =
    R"doc(Replace yieldChannels() wholesale with a list of already-built channels.

Parameters
----------
channels : list of YieldChannel
    The channels to install, in order; ownership of each is
    transferred to this Yields, so none of them are usable from Python
    after this call. None of the entries may be None.

Throws
------
ValueError
    If any entry of channels is None.

Details
-------
yieldChannels() is discarded and replaced by channels, then
rebuildYieldGrid() is called, synchronizing isotopes() (and every
installed channel) onto the union of their own isotopesOrig(). A
YieldChannel object previously read back from yieldChannels() stays
fully valid afterward, even though it's no longer one of the entries
yieldChannels() itself returns -- see deleteChannel()'s own docstring
for the same behavior.)doc";

static constexpr std::string_view setChannelsDescriptorsDocstring =
    R"doc(Replace yieldChannels() wholesale with channels built from a list of descriptors.

Parameters
----------
descriptors : list of YieldChannelDescriptor
    The descriptors to build fresh channels from, in order -- see
    addChannel()'s own docstring for what each one means and how it's
    loaded.

Throws
------
RuntimeError
    If any descriptor fails to load (e.g. an unknown channel/model, or
    a [Fe/H] range mismatch). yieldChannels() is left completely
    unchanged in this case -- every replacement channel is built into
    a temporary list first, and yieldChannels() only ever gets
    replaced wholesale once every descriptor has succeeded.

Details
-------
Unlike the YieldChannel-list overload, this builds every channel fresh
from disk rather than moving in already-built ones. Also calls
rebuildYieldGrid() afterward. As with the YieldChannel-list overload, a
YieldChannel object read back from yieldChannels() before this call
stays fully valid afterward -- see deleteChannel()'s own docstring.)doc";

static constexpr std::string_view rebuildYieldGridDocstring =
    R"doc(Rebuild isotopes() from yieldChannels(), then push it back into every channel.

Parameters
----------
isotopes : list of IsotopeData, optional
    The isotopes wanted; isotopes() is narrowed to these plus whatever
    decay-chain context they need (see Details). An empty list (the
    default) means "keep every isotope any loaded channel tabulates".

Throws
------
RuntimeError
    If isotopes is non-empty but matches nothing any loaded channel
    tabulates (or that is a decay product of something one
    tabulates), leaving isotopes() empty.

Details
-------
A non-empty isotopes list is expanded, not just intersected with what
the loaded channels tabulate. isotopes() keeps every requested isotope
that a channel tabulates or that is a decay product of one that is,
plus every isotope that decays, directly or through intermediates,
into one of those (so requesting Fe56 also keeps Ni56 and Co56, whose
decay contributes to it), plus every decay product of anything kept
(so requesting Ni56 also keeps Co56 and Fe56, and the decay network
always stays complete). An emitted proton or alpha (H1 or He4) does not
count as a link when looking for parents, so requesting H1 or He4 does
not pull in every proton or alpha emitter; but a kept emitter does keep
its own H1 or He4 decay product. Entries matching nothing are ignored.

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
    R"doc(The yield channels loaded so far.

Reading returns the channels in the same order as
controls().yieldChannels(); each one stays fully valid even after being
removed/replaced by a later deleteChannel()/setChannels() call -- see
deleteChannel()'s own docstring. Assigning a list transfers ownership of
each element to this Yields via setChannels() -- see its own
docstring for the two accepted element types (YieldChannel or
YieldChannelDescriptor) -- then calls rebuildYieldGrid(), so isotopes()
(and every channel) stays synchronized onto the new list.)doc";

static constexpr std::string_view isotopesDocstring =
    R"doc(The isotopes this Yields' yield grid is tabulated for.

Reading returns the union of every loaded channel's own
isotopesOrig(): every isotope that appears in at least one of
yieldChannels()'s own isotopesOrig() lists, deduplicated and sorted --
once rebuildYieldGrid() has run (always true after the constructor
itself returns), this is also exactly what every yieldChannels() entry's
own isotopes() equals, in the same order.

Assigning a list calls rebuildYieldGrid() with it, narrowing isotopes()
down to the assigned isotopes plus their decay-chain context (or, for an
empty list, resetting isotopes() back to the full union above) -- see
rebuildYieldGrid()'s own docstring for exactly which isotopes that
keeps, and for the RuntimeError raised if the assigned list is
non-empty but matches nothing any loaded channel tabulates.)doc";

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
dt_decay : float, optional
    Elapsed time, in years (the same units as every isotope's own
    lifetime()), over which to apply radioactive decay to the raw
    per-channel yields before returning them; 0 (the default) applies
    no decay at all. Ignored entirely if the SimControls this Yields
    was built from has noDecay set to True, in which case every
    returned value is the cumulative amount of each isotope ever
    produced, regardless of any radioactive decay since.

Returns
-------
yields : list of list of float
    yields[i][j] is the yield (Msun) of isotopes()[j] from
    yieldChannels()[i], for a star of the given mass/feh, after dt_decay
    has elapsed. Row i is left all zero if yieldChannels()[i] doesn't
    cover mass/feh.

Throws
------
RuntimeError
    If any yieldChannels() entry's own yield_() throws (e.g.
    rebuildYieldGrid() was never called on that particular channel).)doc";

static constexpr std::string_view yieldSumDocstring =
    R"doc(Get the sum, over every channel, of yield_(mass, feh), decayed by dt_decay.

Parameters
----------
mass : float
    Stellar mass (Msun); see yield_()'s own mass parameter.
feh : float
    [Fe/H]; see yield_()'s own feh parameter.
dt_decay : float, optional
    Elapsed time, in years, over which to apply radioactive decay --
    see yield_()'s own dt_decay parameter for exactly what this means
    and when it's ignored. Applied once, to the summed total, rather
    than once per channel -- numerically identical either way, since
    decay is linear in each isotope's own mass.

Returns
-------
yield_sum : list of float
    One yield (Msun) per entry of isotopes(), in the same order:
    result[j] = sum over i of yield_(mass, feh, 0)[i][j], decayed by
    dt_decay.

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
        .def("addChannel",
                [](yields::Yields& self, const yields::YieldChannelDescriptor& descriptor)
                {
                    self.addChannel(descriptor);
                    self.rebuildYieldGrid();
                },
                addChannelDocstring.data(), py::arg("descriptor"))
        .def("addChannel",
                [](yields::Yields& self, std::unique_ptr<yields::YieldChannel> channel)
                {
                    self.addChannel(std::move(channel));
                    self.rebuildYieldGrid();
                },
                addChannelPointerDocstring.data(), py::arg("channel"))
        .def("deleteChannel",
                [](yields::Yields& self, std::size_t index)
                {
                    try
                    {
                        self.deleteChannel(index);
                    }
                    catch (const std::out_of_range& e)
                    {
                        throw py::index_error(e.what());
                    }
                    self.rebuildYieldGrid();
                },
                deleteChannelDocstring.data(), py::arg("index"))
        .def("setChannels",
                [](yields::Yields& self, std::vector<std::unique_ptr<yields::YieldChannel>> channels)
                {
                    self.setChannels(std::move(channels));
                    self.rebuildYieldGrid();
                },
                setChannelsPointersDocstring.data(), py::arg("channels"))
        .def("setChannels",
                [](yields::Yields& self, const std::vector<yields::YieldChannelDescriptor>& descriptors)
                {
                    self.setChannels(descriptors);
                    self.rebuildYieldGrid();
                },
                setChannelsDescriptorsDocstring.data(), py::arg("descriptors"))
        .def("rebuildYieldGrid",
                [](yields::Yields& self, const std::vector<const elem::IsotopeData*>& isotopes)
                {
                    self.rebuildYieldGrid(toIsotopeList(isotopes));
                },
                rebuildYieldGridDocstring.data(),
                py::arg("isotopes") = std::vector<const elem::IsotopeData*>{})
        .def("registryName", &yields::Yields::registryName, registryNameDocstring.data())
        .def_property("yieldChannels",
                [](const yields::Yields& self) -> std::vector<std::shared_ptr<const yields::YieldChannel>>
                {
                    return { self.yieldChannels().begin(), self.yieldChannels().end() };
                },
                [](yields::Yields& self, const py::sequence& channels)
                {
                    // def_property only supports one setter signature,
                    // unlike setChannels()'s own two overloaded .def()
                    // bindings above -- so dispatch by inspecting every
                    // element's type up front (via py::isinstance,
                    // which only inspects -- it never casts/consumes/
                    // releases ownership of anything) before committing
                    // to either ownership-transferring cast below.
                    // Checking every element, not just the first, is
                    // what matters here: a mixed list (e.g. one
                    // YieldChannel followed by one
                    // YieldChannelDescriptor) must be rejected outright,
                    // rather than picking a branch from the first
                    // element and letting py::cast's own per-element
                    // vector conversion fail partway through the
                    // second overload's own cast -- by then, any
                    // YieldChannel elements before the mismatched one
                    // would already have had their own Python-side
                    // ownership released, even though the overall
                    // assignment ends up throwing and yieldChannels_
                    // itself is left unchanged.
                    bool allDescriptors = true;
                    bool allChannels = true;
                    for (const auto& element : channels)
                    {
                        allDescriptors = allDescriptors &&
                            py::isinstance<yields::YieldChannelDescriptor>(element);
                        allChannels = allChannels && py::isinstance<yields::YieldChannel>(element);
                    }
                    if (allDescriptors)
                    {
                        self.setChannels(py::cast<std::vector<yields::YieldChannelDescriptor>>(channels));
                    }
                    else if (allChannels)
                    {
                        self.setChannels(
                            py::cast<std::vector<std::unique_ptr<yields::YieldChannel>>>(channels));
                    }
                    else
                    {
                        throw py::type_error(
                            "yieldChannels must be assigned a list of only YieldChannel or "
                            "only YieldChannelDescriptor, not a mix of the two (or any other type)");
                    }
                    self.rebuildYieldGrid();
                },
                yieldChannelsDocstring.data(), py::return_value_policy::reference_internal)
        .def_property("isotopes",
                [](const yields::Yields& self) -> std::vector<const elem::IsotopeData*>
                {
                    std::vector<const elem::IsotopeData*> result;
                    result.reserve(self.isotopes().size());
                    for (const auto& iso : self.isotopes()) { result.push_back(&iso.get()); }
                    return result;
                },
                [](yields::Yields& self, const std::vector<const elem::IsotopeData*>& isotopes)
                {
                    self.rebuildYieldGrid(toIsotopeList(isotopes));
                },
                isotopesDocstring.data(), py::return_value_policy::reference)
        .def("yield_",
                [](const yields::Yields& self, double mass, double feH, double dtDecay)
                    -> std::vector<std::vector<double>>
                {
                    const auto [view, data] = self.yield(mass, feH, dtDecay);
                    std::vector<std::vector<double>> result(
                        view.extent(0), std::vector<double>(view.extent(1)));
                    for (std::size_t i = 0; i < view.extent(0); ++i)
                    {
                        for (std::size_t j = 0; j < view.extent(1); ++j) { result[i][j] = view[i, j]; }
                    }
                    return result;
                },
                yieldDocstring.data(), py::arg("mass"), py::arg("feh"), py::arg("dt_decay") = 0.0)
        .def("yieldSum", &yields::Yields::yieldSum,
                yieldSumDocstring.data(), py::arg("mass"), py::arg("feh"), py::arg("dt_decay") = 0.0);
}
// NOLINTEND(misc-include-cleaner)
