/**
 * @file BindKDTree.cpp
 * @author Mark Krumholz
 * @brief Python bindings for kden::KDTree and kden::KDTreeNode
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * The Python KDTree (PyKDTree below) views numpy arrays owned by
 * Python, and reorders them in place as it is built; see
 * BindKden.hpp.
 */

#include "BindKden.hpp"
#include "../kden/Bandwidth.hpp"
#include "../kden/KDTree.hpp"
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
     * @struct PyKDTreeArrays
     * @brief The numpy arrays a Python KDTree views, and its C++ view of them
     */
    struct PyKDTreeArrays
    {
        py::array_t<double> x_;                 /**< Positions, N x nDim, owned by Python */
        py::array_t<kdenbind::Aux> aux_;        /**< Auxiliary data, N, owned by Python */
        kdenbind::TreeData view_;               /**< C++ view of x_ and aux_ */
    };

    /**
     * @class PyKDTree
     * @brief A KDTree over numpy arrays, holding references to them
     * @details
     * The arrays are a base, so they (and the view of them) exist
     * before the KDTree base is built over the view.
     */
    class PyKDTree : public PyKDTreeArrays, public kdenbind::Tree
    {
    public:

        /**
         * @brief Build the tree over a set of arrays
         * @param arrays The arrays and the view of them
         * @param leafSize As for KDTree
         * @param splitBandwidth As for KDTree
         * @param noSplitDims As for KDTree
         */
        PyKDTree(PyKDTreeArrays arrays, const std::size_t leafSize, const kden::Bandwidth& splitBandwidth,
            const std::vector<std::size_t>& noSplitDims) :
            PyKDTreeArrays(std::move(arrays)),
            kdenbind::Tree(this->view_, leafSize, splitBandwidth, noSplitDims)
        {}
    };

} // namespace

static constexpr std::string_view nodeClassDocstring = R"doc(A single node of a kd tree.

A node covers the count points starting at index start of the tree's
(reordered) points. An interior node is split along dimension
splitDim into children lchild and rchild; a leaf has splitDim = -1
and both children equal to KDTree.nullNode.)doc";

static constexpr std::string_view classDocstring = R"doc(A balanced kd tree over points held in numpy arrays.

The tree views the arrays it is given rather than copying them: they
must be C-contiguous and writeable, of exactly the right type, and
building the tree reorders them in place, so that every node's points
are contiguous. The tree keeps the arrays alive; changing their
contents afterwards invalidates it. Results (e.g. neighbors()) index
the reordered arrays; aux, which by default holds each point's
original index, maps them back.

Distances are measured in a metric given by a Bandwidth, passed to
each search, so the same tree serves any number of metrics; the
metric given to the constructor only chooses the split dimensions.)doc";

static constexpr std::string_view constructorDocstring = R"doc(Build a kd tree over a set of points, reordering them in place.

Parameters
----------
x : numpy.ndarray of float64, shape (N, nDim)
    The positions; C-contiguous and writeable. Every coordinate must
    be finite. Reordered in place.
aux : numpy.ndarray of int64, shape (N,), optional
    Auxiliary data, one per point (e.g. each point's index in a
    library on disk); C-contiguous and writeable, reordered in place
    along with x. If None (the default), a new array holding 0, ...,
    N - 1 is made, so that aux maps the reordered points back to their
    original indices.
leafSize : int, optional
    Maximum number of points in a leaf (default 16).
splitBandwidth : Bandwidth, optional
    Metric used to choose split dimensions (default: Euclidean).
noSplitDims : list of int, optional
    Dimensions along which no node is split (default: none), e.g.
    dimensions that searches will always leave unspecified.

Raises
------
TypeError
    If x or aux is not a numpy array of the right type.
ValueError
    If an array has the wrong shape, is not C-contiguous or writeable,
    or holds non-finite positions; or leafSize, splitBandwidth, or
    noSplitDims is invalid.)doc";

static constexpr std::string_view nDimDocstring = R"doc(Number of dimensions.

Returns
-------
nDim : int
    The number of dimensions of the points.)doc";

static constexpr std::string_view leafSizeDocstring = R"doc(Maximum number of points in a leaf.

Returns
-------
leafSize : int
    The leaf size.)doc";

static constexpr std::string_view nLevelsDocstring = R"doc(Number of levels allocated for the tree.

Returns
-------
nLevels : int
    The number of levels (an upper bound on the depth).)doc";

static constexpr std::string_view nLeavesDocstring = R"doc(Number of leaf slots on the tree's last level.

Returns
-------
nLeaves : int
    An upper bound on the number of actual leaves.)doc";

static constexpr std::string_view nodesDocstring = R"doc(The tree's nodes, in index order (root first).

Returns
-------
nodes : list of KDTreeNode
    The nodes; node 0 is the root, and the children of node i are 2i + 1
    and 2i + 2. Slots below a leaf are unused, and hold default nodes.)doc";

static constexpr std::string_view nodeLoDocstring = R"doc(Lower corner of a node's bounding box.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
lo : numpy.ndarray of float
    The minimum of the node's points' positions in each dimension.

Raises
------
IndexError
    If idx is out of range.)doc";

static constexpr std::string_view nodeHiDocstring = R"doc(Upper corner of a node's bounding box.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
hi : numpy.ndarray of float
    The maximum of the node's points' positions in each dimension.

Raises
------
IndexError
    If idx is out of range.)doc";

static constexpr std::string_view neighborsDocstring = R"doc(Find the nearest neighbors of one or more points.

Parameters
----------
x : array_like of float, shape (nDim,) or (M, nDim)
    One point, or M of them. Coordinates may be NaN, to judge nearness
    only in the other dimensions.
nNeighbor : int
    Number of neighbors to find for each point.
bandwidth : Bandwidth or sequence of Bandwidth, optional
    The metric: None (the default) for the Euclidean one, one
    Bandwidth for every point, or (for M points) a sequence of M. A
    bandwidth not reduced for a point's NaNs is reduced automatically
    (once for the whole batch, if every point shares one pattern).

Returns
-------
idx : numpy.ndarray of int64, shape (k,) or (M, k)
    Indices of the neighbors in the tree's (reordered) points, nearest
    first, with k = min(nNeighbor, N); aux[idx] gives their auxiliary
    data.
dist2 : numpy.ndarray of float, shape (k,) or (M, k)
    Their squared distances.

Notes
-----
An exact search; a batch runs in parallel, without the GIL.)doc";

static constexpr std::string_view leftDocstring = R"doc(Index of a node's left child.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
left : int
    2 idx + 1.)doc";

static constexpr std::string_view rightDocstring = R"doc(Index of a node's right child.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
right : int
    2 idx + 2.)doc";

static constexpr std::string_view parentDocstring = R"doc(Index of a node's parent.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
parent : int
    (idx - 1) // 2, or KDTree.nullNode for the root.)doc";

static constexpr std::string_view siblingDocstring = R"doc(Index of a node's sibling.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
sibling : int
    The other child of the node's parent, or KDTree.nullNode for the
    root.)doc";

static constexpr std::string_view nextDocstring = R"doc(Index of the next node to visit after a node's whole subtree.

Parameters
----------
idx : int
    Index of the node.

Returns
-------
next : int
    The first node after idx and all its descendants in a depth-first,
    parent-before-children traversal, or KDTree.nullNode if there is
    none.)doc";

static constexpr std::string_view xDocstring = R"doc(The positions, in tree order.

Returns
-------
x : numpy.ndarray of float, shape (N, nDim)
    A read-only view of the (reordered) position array.)doc";

static constexpr std::string_view auxDocstring = R"doc(The auxiliary data, in tree order.

Returns
-------
aux : numpy.ndarray of int64, shape (N,)
    A read-only view of the (reordered) auxiliary data array.)doc";

/**
 * @brief The tree-method docstrings, in the order addTreeMethods() takes them
 * @return The docstrings
 */
auto kdTreeMethodDocstrings() -> std::vector<const char*>
{
    return {nDimDocstring.data(), leafSizeDocstring.data(), nLevelsDocstring.data(),
        nLeavesDocstring.data(), nodesDocstring.data(), nodeLoDocstring.data(),
        nodeHiDocstring.data(), neighborsDocstring.data(), leftDocstring.data(),
        rightDocstring.data(), parentDocstring.data(), siblingDocstring.data(),
        nextDocstring.data(), xDocstring.data(), auxDocstring.data()};
}

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindKDTree(py::module_& m)
{
    py::class_<kden::KDTreeNode, py::smart_holder>(m, "KDTreeNode", nodeClassDocstring.data())
        .def_readonly("start", &kden::KDTreeNode::start_, "Index of the node's first point")
        .def_readonly("count", &kden::KDTreeNode::count_, "Number of points in the node")
        .def_readonly("lchild", &kden::KDTreeNode::lchild_, "Index of the left child, or KDTree.nullNode for a leaf")
        .def_readonly("rchild", &kden::KDTreeNode::rchild_, "Index of the right child, or KDTree.nullNode for a leaf")
        .def_readonly("parent", &kden::KDTreeNode::parent_, "Index of the parent, or KDTree.nullNode for the root")
        .def_readonly("splitDim", &kden::KDTreeNode::splitDim_, "Dimension the node is split along, or -1 for a leaf");

    py::class_<PyKDTree, py::smart_holder> cls(m, "KDTree", classDocstring.data());
    cls.def(py::init([](const py::object& x, const py::object& aux, const std::size_t leafSize,
                const py::object& splitBandwidth, const std::vector<std::size_t>& noSplitDims)
                -> std::unique_ptr<PyKDTree>
            {
                PyKDTreeArrays arrays;
                arrays.x_ = kdenbind::viewableArray<double>(x, "x", 2);
                const auto n = static_cast<std::size_t>(arrays.x_.shape(0));
                const auto nDim = static_cast<std::size_t>(arrays.x_.shape(1));
                arrays.aux_ = kdenbind::auxArray(aux, n);
                arrays.view_.nDim_ = nDim;
                arrays.view_.x_ = kdenbind::spanOf(arrays.x_);
                arrays.view_.aux_ = kdenbind::spanOf(arrays.aux_);
                const kden::Bandwidth split = splitBandwidth.is_none()
                    ? kden::Bandwidth::identity(std::max<std::size_t>(nDim, 1))
                    : py::cast<kden::Bandwidth>(splitBandwidth);
                return std::make_unique<PyKDTree>(std::move(arrays), leafSize, split, noSplitDims);
            }),
            constructorDocstring.data(),
            py::arg("x"), py::arg("aux") = py::none(), py::arg("leafSize") = 16,
            py::arg("splitBandwidth") = py::none(),
            py::arg("noSplitDims") = std::vector<std::size_t>{});
    kdenbind::addTreeMethods<kdenbind::Tree>(cls, kdTreeMethodDocstrings());
    cls.attr("nullNode") = kden::nullNode;
    cls.attr("rootNode") = kden::rootNode;
}
// NOLINTEND(misc-include-cleaner)
