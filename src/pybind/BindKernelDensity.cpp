/**
 * @file BindKernelDensity.cpp
 * @author Mark Krumholz
 * @brief Python bindings for kden::KernelDensity and kden::ConditionalDensity
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * kden::KernelDensity itself is bound as KernelDensityView, with all
 * its methods: it is what ConditionalDensity.kde() returns, an
 * estimate over points the ConditionalDensity owns. The Python
 * KernelDensity (PyKernelDensity below) derives from it, adding a
 * constructor that views numpy arrays owned by Python and reorders
 * them in place (see BindKden.hpp).
 */

#include "BindKden.hpp"
#include "../kden/Bandwidth.hpp"
#include "../kden/KernelDensity.hpp"
#include "Bindings.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <string_view>
#include <utility>
#include <vector>

namespace {

    /**
     * @struct PyKDEArrays
     * @brief The numpy arrays a Python KernelDensity views, and its C++ view of them
     */
    struct PyKDEArrays
    {
        py::array_t<double> x_;             /**< Positions, N x nDim, owned by Python */
        py::array_t<double> w_;             /**< Weights, N, owned by Python */
        py::array_t<kdenbind::Aux> aux_;    /**< Auxiliary data, N, owned by Python */
        kdenbind::KDEData view_;            /**< C++ view of x_, w_, and aux_ */
    };

    /**
     * @class PyKernelDensity
     * @brief A KernelDensity over numpy arrays, holding references to them
     * @details
     * The arrays are a base, so they (and the view of them) exist
     * before the KernelDensity base is built over the view.
     */
    class PyKernelDensity : public PyKDEArrays, public kdenbind::KDE
    {
    public:

        /**
         * @brief Build the estimate over a set of arrays
         * @param arrays The arrays and the view of them
         * @param leafSize As for KernelDensity
         * @param splitBandwidth As for KernelDensity
         * @param noSplitDims As for KernelDensity
         */
        PyKernelDensity(PyKDEArrays arrays, const std::size_t leafSize, const kden::Bandwidth& splitBandwidth,
            const std::vector<std::size_t>& noSplitDims) :
            PyKDEArrays(std::move(arrays)),
            kdenbind::KDE(this->view_, leafSize, splitBandwidth, noSplitDims)
        {}
    };

} // namespace

static constexpr std::string_view viewClassDocstring = R"doc(A kernel density estimate over points it does not own.

The base of KernelDensity, with all its methods; on its own, it is
what ConditionalDensity.kde() returns, an estimate over points the
ConditionalDensity owns. See KernelDensity for the estimate itself.)doc";

static constexpr std::string_view classDocstring = R"doc(A kernel density estimate built on a kd tree, over points held in numpy arrays.

The estimate at x is p(x) = A sum_i w_i exp(-(x - x_i)^T M (x - x_i)),
with M the inverse squared bandwidth matrix (a Bandwidth, passed to
each evaluation) and A = sqrt(det M) / pi^(nDim / 2), and the weights
w_i normalized to sum to one, so that p integrates to one.

Like KDTree, it views the arrays it is given rather than copying them:
they must be C-contiguous and writeable, of exactly the right type,
and building the estimate reorders them in place (and normalizes the
weights in place). It keeps the arrays alive; changing their contents
afterwards invalidates it.)doc";

static constexpr std::string_view constructorDocstring = R"doc(Build a kernel density estimate over a set of weighted points, reordering them in place.

Parameters
----------
x : numpy.ndarray of float64, shape (N, nDim)
    The kernel centers; C-contiguous and writeable. Every coordinate
    must be finite. Reordered in place.
w : numpy.ndarray of float64, shape (N,), optional
    The weights; C-contiguous and writeable, finite, non-negative, and
    not all zero. Reordered in place along with x, and rescaled in
    place to sum to one. If None (the default), a new array of equal
    weights is made.
aux : numpy.ndarray of int64, shape (N,), optional
    Auxiliary data, as for KDTree; if None (the default), a new array
    holding 0, ..., N - 1.
leafSize : int, optional
    Maximum number of points in a leaf (default 16).
splitBandwidth : Bandwidth, optional
    Metric used to choose split dimensions (default: Euclidean).
noSplitDims : list of int, optional
    Dimensions along which no node is split (default: none), e.g.
    dimensions that evaluations will always leave unspecified.

Raises
------
TypeError
    If x, w, or aux is not a numpy array of the right type.
ValueError
    If an array has the wrong shape, is not C-contiguous or writeable,
    or holds invalid values; or leafSize, splitBandwidth, or
    noSplitDims is invalid.)doc";

static constexpr std::string_view nodeWDocstring = R"doc(Sum of the weights of a node's points.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
w : float
    The sum (the weights sum to one over the whole tree).

Raises
------
IndexError
    If idx is out of range.)doc";

static constexpr std::string_view wDocstring = R"doc(The weights, in tree order.

Returns
-------
w : numpy.ndarray of float, shape (N,)
    A read-only view of the (reordered, normalized) weight array.)doc";

static constexpr std::string_view pdfDocstring = R"doc(Evaluate the kernel density estimate at one or more points, to within a tolerance.

Parameters
----------
x : array_like of float, shape (nDim,) or (M, nDim)
    One point, or M of them. Coordinates may be NaN, to evaluate the
    marginal density over the other dimensions.
relTol : float
    Relative tolerance: stop once the uncertainty is less than relTol
    times the estimate.
absTol : float
    Absolute tolerance: stop once the estimate plus its uncertainty is
    less than absTol.
bandwidth : Bandwidth or sequence of Bandwidth, optional
    The kernel's inverse squared bandwidth matrix: None (the default)
    for the identity, one Bandwidth for every point, or (for M points)
    a sequence of M. A bandwidth not reduced for a point's NaNs is
    reduced automatically (once for the whole batch, if every point
    shares one pattern).

Returns
-------
pdf : float or numpy.ndarray of float
    The estimate, normalized to unit integral (over the specified
    dimensions, for a point with NaNs); a float for one point.

Raises
------
ValueError
    If x or a bandwidth has the wrong number of dimensions, a bandwidth
    was reduced for a different pattern of NaNs, or a bandwidth is
    singular over the specified dimensions.

Notes
-----
A batch runs in parallel, without the GIL.)doc";

static constexpr std::string_view conditionalDocstring = R"doc(Condition the estimate on values of some dimensions, optionally marginalizing over others.

Parameters
----------
x0 : array_like of float, shape (nDim,)
    A finite value fixes that dimension; NaN keeps it as a dimension
    of the result. Values for dimensions in marginalDims are ignored.
massTol : float
    Largest fraction of the conditional estimate's total weight that
    may be dropped, in [0, 1); 0 keeps every point with non-zero
    weight. The result is within L1 distance 2 massTol of the exact
    conditional density.
marginalDims : list of int, optional
    Dimensions to marginalize over (default: none).
bandwidth : Bandwidth, optional
    The kernel's inverse squared bandwidth matrix (default: identity);
    must not be reduced.
leafSize : int, optional
    Leaf size of the result's tree (default 16).

Returns
-------
conditional : ConditionalDensity
    The conditional estimate p(y | x0) over the kept dimensions.

Raises
------
ValueError
    If x0 or bandwidth has the wrong number of dimensions, bandwidth
    is reduced, marginalDims is invalid, x0 fixes no dimension or
    every dimension, massTol is not in [0, 1), the bandwidth's blocks
    are not positive definite, or every point's conditional weight
    underflows to zero.)doc";

static constexpr std::string_view condClassDocstring = R"doc(A kernel density estimate conditioned on some dimensions' values.

Made by KernelDensity.conditional(). Owns its points: shifted,
reweighted, and pruned copies of the original estimate's points, with
their auxiliary data. Its pdf() is the conditional density p(y | x0)
over the kept dimensions (see dims()), and marginal() the original
estimate's marginal density at the fixed values, so that the original
estimate at (x0, y) is marginal() * pdf(y).)doc";

static constexpr std::string_view condPdfDocstring = R"doc(Evaluate the conditional density at one or more points.

Parameters
----------
y : array_like of float, shape (nKept,) or (M, nKept)
    One point in the kept dimensions (see dims()), or M of them;
    coordinates may be NaN, to marginalize over them too.
relTol : float
    Relative tolerance, as for KernelDensity.pdf().
absTol : float
    Absolute tolerance, as for KernelDensity.pdf().

Returns
-------
pdf : float or numpy.ndarray of float
    p(y | x0), normalized to unit integral over the kept dimensions; a
    float for one point.

Notes
-----
A batch runs in parallel, without the GIL.)doc";

static constexpr std::string_view condDimsDocstring = R"doc(The kept dimensions.

Returns
-------
dims : list of int
    Their indices in the original estimate, in order: dimension i of
    this estimate is dims()[i] there.)doc";

static constexpr std::string_view condMarginalDocstring = R"doc(The original estimate's marginal density over the fixed dimensions, at x0.

Returns
-------
marginal : float
    p(x0), estimated from the points kept, so low by at most a
    fraction massTol.)doc";

static constexpr std::string_view condBandwidthDocstring = R"doc(The kernel metric of the conditional estimate.

Returns
-------
bandwidth : Bandwidth
    The block of the (marginalized) bandwidth over the kept dimensions.)doc";

static constexpr std::string_view condKdeDocstring = R"doc(The conditional estimate itself.

Returns
-------
kde : KernelDensityView
    Its kernel density estimate, to be evaluated with bandwidth(). It
    keeps this ConditionalDensity alive.)doc";

static constexpr std::string_view condXDocstring = R"doc(The conditional estimate's points, in tree order.

Returns
-------
x : numpy.ndarray of float, shape (K, nKept)
    A read-only view of the positions.)doc";

static constexpr std::string_view condWDocstring = R"doc(The conditional estimate's weights, in tree order.

Returns
-------
w : numpy.ndarray of float, shape (K,)
    A read-only view of the weights, normalized to sum to one.)doc";

static constexpr std::string_view condAuxDocstring = R"doc(The conditional estimate's auxiliary data, in tree order.

Returns
-------
aux : numpy.ndarray of int64, shape (K,)
    A read-only view of the kept points' auxiliary data, carried over
    from the original estimate.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindKernelDensity(py::module_& m)
{
    using kdenbind::KDE;
    using kdenbind::Conditional;

    // kden::KernelDensity itself, as the base of the Python KernelDensity
    py::class_<KDE, py::smart_holder> view(m, "KernelDensityView", viewClassDocstring.data());
    kdenbind::addTreeMethods<KDE>(view, kdTreeMethodDocstrings());
    view.def("nodeW", [](const KDE& kde, const std::size_t idx) -> double
            {
                if (idx >= kde.nodes().size()) { throw py::index_error("node index out of range"); }
                return kde.nodeW(idx);
            }, nodeWDocstring.data(), py::arg("idx"))
        .def_property_readonly("w", [](const py::object& self) -> py::array_t<double>
            {
                const auto& kde = py::cast<const KDE&>(self);
                return kdenbind::readOnlyView(kde.data().w_, {static_cast<py::ssize_t>(kde.data().size())}, self);
            }, wDocstring.data())
        .def("pdf", [](const KDE& kde, const kdenbind::QueryArray& x, const double relTol,
                const double absTol, const py::object& bandwidth) -> py::object
            {
                return kdenbind::pdf(kde, x, relTol, absTol, bandwidth, kden::Bandwidth::identity(kde.nDim()));
            }, pdfDocstring.data(), py::arg("x"), py::arg("relTol"), py::arg("absTol"),
            py::arg("bandwidth") = py::none())
        .def("conditional", [](const KDE& kde, const std::vector<double>& x0, const double massTol,
                const std::vector<std::size_t>& marginalDims, const py::object& bandwidth,
                const std::size_t leafSize) -> Conditional
            {
                const kden::Bandwidth bw = bandwidth.is_none()
                    ? kden::Bandwidth::identity(kde.nDim())
                    : py::cast<kden::Bandwidth>(bandwidth);
                const py::gil_scoped_release release;
                return kde.conditional(x0, marginalDims, massTol, bw, leafSize);
            }, conditionalDocstring.data(), py::arg("x0"), py::arg("massTol"),
            py::arg("marginalDims") = std::vector<std::size_t>{}, py::arg("bandwidth") = py::none(),
            py::arg("leafSize") = 16);
    view.attr("nullNode") = kden::nullNode;
    view.attr("rootNode") = kden::rootNode;

    // The Python KernelDensity: the same, over numpy arrays
    // multiple_inheritance(): KDE is PyKernelDensity's second C++ base
    // (after the arrays), so is not at offset zero; without this,
    // pybind11 would cast a KernelDensity to its registered base KDE
    // as if it were
    py::class_<PyKernelDensity, KDE, py::smart_holder>(m, "KernelDensity", classDocstring.data(),
            py::multiple_inheritance())
        .def(py::init([](const py::object& x, const py::object& w, const py::object& aux,
                const std::size_t leafSize, const py::object& splitBandwidth,
                const std::vector<std::size_t>& noSplitDims) -> std::unique_ptr<PyKernelDensity>
            {
                PyKDEArrays arrays;
                arrays.x_ = kdenbind::viewableArray<double>(x, "x", 2);
                const auto n = static_cast<std::size_t>(arrays.x_.shape(0));
                const auto nDim = static_cast<std::size_t>(arrays.x_.shape(1));
                if (w.is_none())
                {
                    arrays.w_ = py::array_t<double>(static_cast<py::ssize_t>(n));
                    std::fill_n(arrays.w_.mutable_data(), n, 1.0);
                }
                else
                {
                    arrays.w_ = kdenbind::viewableArray<double>(w, "w", 1);
                    if (static_cast<std::size_t>(arrays.w_.size()) != n)
                    {
                        throw py::value_error("w must have one element per point");
                    }
                }
                arrays.aux_ = kdenbind::auxArray(aux, n);
                arrays.view_.nDim_ = nDim;
                arrays.view_.x_ = kdenbind::spanOf(arrays.x_);
                arrays.view_.w_ = kdenbind::spanOf(arrays.w_);
                arrays.view_.aux_ = kdenbind::spanOf(arrays.aux_);
                const kden::Bandwidth split = splitBandwidth.is_none()
                    ? kden::Bandwidth::identity(std::max<std::size_t>(nDim, 1))
                    : py::cast<kden::Bandwidth>(splitBandwidth);
                return std::make_unique<PyKernelDensity>(std::move(arrays), leafSize, split, noSplitDims);
            }),
            constructorDocstring.data(),
            py::arg("x"), py::arg("w") = py::none(), py::arg("aux") = py::none(),
            py::arg("leafSize") = 16, py::arg("splitBandwidth") = py::none(),
            py::arg("noSplitDims") = std::vector<std::size_t>{});

    // ConditionalDensity, which owns its points
    py::class_<Conditional, py::smart_holder>(m, "ConditionalDensity", condClassDocstring.data())
        .def("pdf", [](const Conditional& cd, const kdenbind::QueryArray& y, const double relTol,
                const double absTol) -> py::object
            {
                auto [pts, single] = kdenbind::queryPoints(y, cd.dims().size());
                const auto out = kdenbind::evalPoints(pts, [&](const std::vector<double>& p,
                    const std::size_t) -> double { return cd.pdf(p, relTol, absTol); });
                if (single) { return py::float_(out.front()); }
                return kdenbind::toArray(out);
            }, condPdfDocstring.data(), py::arg("y"), py::arg("relTol"), py::arg("absTol"))
        .def("dims", &Conditional::dims, condDimsDocstring.data())
        .def("marginal", &Conditional::marginal, condMarginalDocstring.data())
        .def("bandwidth", &Conditional::bandwidth, condBandwidthDocstring.data(),
            py::return_value_policy::reference_internal)
        .def("kde", &Conditional::kde, condKdeDocstring.data(),
            py::return_value_policy::reference_internal)
        .def_property_readonly("x", [](const py::object& self) -> py::array_t<double>
            {
                const auto& cd = py::cast<const Conditional&>(self);
                return kdenbind::readOnlyView(cd.data().x_, {static_cast<py::ssize_t>(cd.data().size()),
                    static_cast<py::ssize_t>(cd.dims().size())}, self);
            }, condXDocstring.data())
        .def_property_readonly("w", [](const py::object& self) -> py::array_t<double>
            {
                const auto& cd = py::cast<const Conditional&>(self);
                return kdenbind::readOnlyView(cd.data().w_, {static_cast<py::ssize_t>(cd.data().size())}, self);
            }, condWDocstring.data())
        .def_property_readonly("aux", [](const py::object& self) -> py::array_t<kdenbind::Aux>
            {
                const auto& cd = py::cast<const Conditional&>(self);
                return kdenbind::readOnlyView(cd.data().aux_, {static_cast<py::ssize_t>(cd.data().size())}, self);
            }, condAuxDocstring.data());
}
// NOLINTEND(misc-include-cleaner)
