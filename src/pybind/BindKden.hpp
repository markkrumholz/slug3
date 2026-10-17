/**
 * @file BindKden.hpp
 * @author Mark Krumholz
 * @brief Shared helpers for the Python bindings of src/kden
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * The kd tree classes are templates, and are non-owning views of the
 * points they index, which they reorder in place. Python binds one
 * instantiation of each, with 64-bit integer auxiliary data (meant to
 * hold each point's index in some original data set, e.g. a library
 * on disk, so that the reordering can always be undone). The Python
 * KDTree and KernelDensity classes view numpy arrays owned by Python:
 * they hold references to the arrays, so the arrays outlive the tree,
 * and building the tree reorders them in place, with no copying.
 *
 * This header holds what BindKDTree.cpp and BindKernelDensity.cpp
 * share: the array checks, the conversion of query points, the
 * (OpenMP-parallel, GIL-free) evaluation of a batch of queries, and
 * the methods every tree type gets.
 */

#ifndef BINDKDEN_HPP
#define BINDKDEN_HPP

#include "../kden/Bandwidth.hpp"
#include "../kden/KDTree.hpp"
#include "../kden/KernelDensity.hpp"
#include "Bindings.hpp" // NOLINT(misc-include-cleaner); provides the py namespace alias used throughout
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h> // NOLINT(misc-include-cleaner); used throughout, via the py alias
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <span>
#include <string>
#include <utility>
#include <vector>

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
/**
 * @brief Docstrings for the methods every bound tree type shares
 * @return Them, in the order addTreeMethods() takes them (defined in BindKDTree.cpp)
 */
auto kdTreeMethodDocstrings() -> std::vector<const char*>;

namespace kdenbind {

    using Aux = std::int64_t;                         /**< Auxiliary data type of the bound instantiations */
    using TreeData = kden::KDTreeData<Aux>;           /**< Points of a bound KDTree */
    using KDEData = kden::KernelDensityData<Aux>;     /**< Points of a bound KernelDensity */
    using Tree = kden::KDTree<TreeData>;              /**< The bound KDTree */
    using KDE = kden::KernelDensity<Aux>;             /**< The bound KernelDensity */
    using Conditional = kden::ConditionalDensity<Aux>; /**< The bound ConditionalDensity */

    /** A numpy array of queries: converted to C-contiguous float64 if need be */
    using QueryArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    /**
     * @brief Check that an array can be viewed, and reordered in place, by C++
     * @tparam T Element type the array must have
     * @param obj The array
     * @param name Its name, for error messages
     * @param ndim Number of dimensions it must have
     * @return obj, as a py::array_t<T>, sharing its memory
     * @throws py::type_error if obj is not a numpy array of exactly
     *   type T
     * @throws py::value_error if it does not have ndim dimensions, is
     *   not C-contiguous, or is not writeable
     * @details
     * Never converts or copies: C++ must see, and reorder, the
     * caller's own memory.
     */
    template <typename T>
    inline auto viewableArray(const py::object& obj, const std::string& name, const py::ssize_t ndim)
        -> py::array_t<T>
    {
        if (!py::isinstance<py::array>(obj) ||
            !py::reinterpret_borrow<py::array>(obj).dtype().is(py::dtype::of<T>()))
        {
            throw py::type_error(name + " must be a numpy array of " +
                std::string(py::str(py::dtype::of<T>())) + " (it is reordered in place, so is never converted)");
        }
        auto arr = py::reinterpret_borrow<py::array_t<T>>(obj);
        if (arr.ndim() != ndim)
        {
            throw py::value_error(name + " must have " + std::to_string(ndim) + " dimension(s)");
        }
        if ((arr.flags() & py::array::c_style) == 0 || !arr.writeable())
        {
            throw py::value_error(name + " must be C-contiguous and writeable (it is reordered in place)");
        }
        return arr;
    }

    /**
     * @brief A span over a numpy array's memory
     * @tparam T Element type
     * @param arr The array
     * @return A span over all its elements
     */
    template <typename T>
    inline auto spanOf(py::array_t<T>& arr) -> std::span<T>
    {
        return {arr.mutable_data(), static_cast<std::size_t>(arr.size())};
    }

    /**
     * @brief The auxiliary data array for a set of points
     * @param aux An int64 array of one element per point, or None
     * @param n Number of points
     * @return aux (checked as for viewableArray()), or, if aux is None,
     *   a new array holding 0, ..., n - 1
     * @throws py::value_error if aux does not have n elements
     */
    inline auto auxArray(const py::object& aux, const std::size_t n) -> py::array_t<Aux>
    {
        if (aux.is_none())
        {
            py::array_t<Aux> idx(static_cast<py::ssize_t>(n));
            auto* p = idx.mutable_data();
            for (std::size_t i = 0; i < n; ++i) { p[i] = static_cast<Aux>(i); } // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            return idx;
        }
        auto arr = viewableArray<Aux>(aux, "aux", 1);
        if (static_cast<std::size_t>(arr.size()) != n)
        {
            throw py::value_error("aux must have one element per point");
        }
        return arr;
    }

    /**
     * @brief Query points from a numpy array
     * @param x One point (nDim elements) or several (an M x nDim array)
     * @param nDim Number of dimensions
     * @return The points, and whether x was a single point
     * @throws py::value_error if x has the wrong shape
     */
    inline auto queryPoints(const QueryArray& x, const std::size_t nDim)
        -> std::pair<std::vector<std::vector<double>>, bool>
    {
        const bool single = x.ndim() == 1;
        if ((x.ndim() != 1 && x.ndim() != 2) || static_cast<std::size_t>(x.shape(x.ndim() - 1)) != nDim)
        {
            throw py::value_error("x must have shape (nDim,) or (M, nDim), with nDim = " + std::to_string(nDim));
        }
        const std::size_t m = single ? 1 : static_cast<std::size_t>(x.shape(0));
        std::vector<std::vector<double>> pts(m);
        const double* p = x.data();
        for (std::size_t i = 0; i < m; ++i)
        {
            pts[i].assign(p + (i * nDim), p + ((i + 1) * nDim)); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        }
        return {std::move(pts), single};
    }

    /**
     * @brief Evaluate a function at each of a set of points, in parallel
     * @tparam Eval Callable taking (point, index) and returning a double
     * @param pts The points
     * @param eval The function
     * @return Its value at each point
     * @details
     * Runs without the GIL, and in parallel (dynamically scheduled) if
     * built with OpenMP; the first exception thrown, if any, is
     * rethrown afterwards, since one must not escape an OpenMP region.
     */
    template <typename Eval>
    inline auto evalPoints(const std::vector<std::vector<double>>& pts, const Eval& eval) -> std::vector<double>
    {
        std::vector<double> out(pts.size());
        std::exception_ptr error = nullptr;
        {
            const py::gil_scoped_release release;
#ifdef _OPENMP
#           pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t i = 0; i < pts.size(); ++i)
            {
                try
                {
                    out[i] = eval(pts[i], i);
                }
                catch (...)
                {
#ifdef _OPENMP
#                   pragma omp critical(kdenbindEvalError)
#endif
                    {
                        if (!error) { error = std::current_exception(); }
                    }
                }
            }
        }
        if (error) { std::rethrow_exception(error); }
        return out;
    }

    /**
     * @brief A 1D numpy array copied from a vector
     * @tparam T Element type
     * @param v The values
     * @return A new array holding them
     */
    template <typename T>
    inline auto toArray(const std::vector<T>& v) -> py::array_t<T>
    {
        py::array_t<T> arr(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), arr.mutable_data());
        return arr;
    }

    /**
     * @brief A numpy view of a span, kept alive by an owner
     * @tparam T Element type
     * @param s The span
     * @param shape Shape of the view
     * @param owner Python object whose lifetime the memory shares
     * @param writeable Whether the view may be written through
     * @return The view
     */
    template <typename T>
    inline auto spanView(const std::span<T> s, const std::vector<py::ssize_t>& shape, const py::handle owner,
        const bool writeable) -> py::array_t<T>
    {
        py::array_t<T> arr(shape, s.data(), owner);
        if (!writeable) { arr.attr("setflags")(py::arg("write") = false); }
        return arr;
    }

    /**
     * @brief A read-only numpy view of a span, kept alive by an owner
     * @tparam T Element type
     * @param s The span
     * @param shape Shape of the view
     * @param owner Python object whose lifetime the memory shares
     * @return The view
     */
    template <typename T>
    inline auto readOnlyView(const std::span<T> s, const std::vector<py::ssize_t>& shape, const py::handle owner)
        -> py::array_t<T>
    {
        return spanView(s, shape, owner, false);
    }

    /**
     * @brief The bandwidths a batch of queries uses
     * @param bandwidth None (the Euclidean metric, taken from euclidean),
     *   one Bandwidth for every query, or a sequence of one per query
     * @param m Number of queries
     * @param euclidean The Euclidean metric
     * @return One bandwidth per query (all equal to the shared one, if
     *   only one was given), and whether a single one is shared
     * @throws py::value_error if a sequence has the wrong length
     */
    inline auto batchBandwidths(const py::object& bandwidth, const std::size_t m, const kden::Bandwidth& euclidean)
        -> std::pair<std::vector<kden::Bandwidth>, bool>
    {
        if (bandwidth.is_none()) { return {{euclidean}, true}; }
        if (py::isinstance<kden::Bandwidth>(bandwidth)) { return {{py::cast<kden::Bandwidth>(bandwidth)}, true}; }
        auto bws = py::cast<std::vector<kden::Bandwidth>>(bandwidth);
        if (bws.size() != m)
        {
            throw py::value_error("bandwidth must be None, a Bandwidth, or a sequence of one Bandwidth per point");
        }
        return {std::move(bws), false};
    }

    /**
     * @brief A shared bandwidth, reduced once for a batch of queries if they all share a pattern of NaNs
     * @param bandwidth The shared bandwidth
     * @param pts The query points
     * @return bandwidth reduced for the points' common pattern of NaNs,
     *   if it is not reduced already, they all share one, and it has a
     *   NaN; otherwise bandwidth itself (each query then reduces it as
     *   needed)
     */
    inline auto reduceShared(const kden::Bandwidth& bandwidth, const std::vector<std::vector<double>>& pts)
        -> kden::Bandwidth
    {
        if (pts.empty() || bandwidth.isReduced() || pts.front().size() != bandwidth.nDim() ||
            bandwidth.matches(pts.front()))
        {
            return bandwidth;
        }
        const std::vector<bool> pattern = kden::Bandwidth::nanPattern(pts.front());
        for (const auto& p : pts)
        {
            if (kden::Bandwidth::nanPattern(p) != pattern) { return bandwidth; }
        }
        return bandwidth.reduce(pattern);
    }

    /**
     * @brief Find nearest neighbors for one point or a batch, for Python
     * @tparam TreeT The tree type
     * @param tree The tree
     * @param x One point or an M x nDim array
     * @param nNeighbor Number of neighbors per point
     * @param bandwidth None, a Bandwidth, or (for a batch) a sequence of one per point
     * @param euclidean The Euclidean metric
     * @return (idx, dist2): int64 and float64 arrays of shape (k,) for
     *   one point, or (M, k) for a batch, k = min(nNeighbor, number of points)
     */
    template <typename TreeT>
    inline auto neighbors(const TreeT& tree, const QueryArray& x, const std::size_t nNeighbor,
        const py::object& bandwidth, const kden::Bandwidth& euclidean) -> py::tuple
    {
        auto [pts, single] = queryPoints(x, tree.nDim());
        auto [bws, shared] = batchBandwidths(bandwidth, pts.size(), euclidean);
        std::vector<std::vector<kden::Neighbor>> res;
        {
            const py::gil_scoped_release release;
            const std::vector<std::size_t> ks(pts.size(), nNeighbor);
            if (shared) { res = tree.neighbors(pts, ks, bws.front()); }
            else { res = tree.neighbors(pts, ks, bws); }
        }
        const std::size_t k = res.empty() ? 0 : res.front().size();
        std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(k)};
        if (!single) { shape.insert(shape.begin(), static_cast<py::ssize_t>(res.size())); }
        py::array_t<Aux> idx(shape);
        py::array_t<double> dist2(shape);
        Aux* pi = idx.mutable_data();
        double* pd = dist2.mutable_data();
        for (std::size_t q = 0; q < res.size(); ++q)
        {
            for (std::size_t j = 0; j < k; ++j)
            {
                pi[(q * k) + j] = static_cast<Aux>(res[q][j].idx_); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                pd[(q * k) + j] = res[q][j].dist2_; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            }
        }
        return py::make_tuple(idx, dist2);
    }

    /**
     * @brief Evaluate a kernel density estimate at one point or a batch, for Python
     * @param kde The estimate
     * @param x One point or an M x nDim array
     * @param relTol Relative tolerance
     * @param absTol Absolute tolerance
     * @param bandwidth None, a Bandwidth, or (for a batch) a sequence of one per point
     * @param euclidean The Euclidean metric
     * @return A float for one point, or an array of M for a batch
     */
    inline auto pdf(const KDE& kde, const QueryArray& x, const double relTol, const double absTol,
        const py::object& bandwidth, const kden::Bandwidth& euclidean) -> py::object
    {
        // Not structured bindings: the lambdas below, which run inside
        // an OpenMP region, capture these, and clang does not support
        // capturing a structured binding there
        const auto query = queryPoints(x, kde.nDim());
        const auto& pts = query.first;
        const auto batch = batchBandwidths(bandwidth, pts.size(), euclidean);
        const auto& bws = batch.first;
        std::vector<double> out;
        if (batch.second)
        {
            const kden::Bandwidth bw = reduceShared(bws.front(), pts);
            out = evalPoints(pts, [&](const std::vector<double>& p, const std::size_t) -> double
                { return kde.pdf(p, relTol, absTol, bw); });
        }
        else
        {
            out = evalPoints(pts, [&](const std::vector<double>& p, const std::size_t i) -> double
                { return kde.pdf(p, relTol, absTol, bws[i]); }); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        }
        if (query.second) { return py::float_(out.front()); }
        return toArray(out);
    }

    /**
     * @brief Add the methods every bound tree type shares
     * @tparam PyClass The py::class_ type
     * @tparam TreeT The C++ tree type (or a base of the bound class)
     * @param cls The class being bound
     * @param docs Docstrings, in order: nDim, leafSize, nLevels,
     *   nLeaves, nodes, nodeLo, nodeHi, neighbors, left, right, parent,
     *   sibling, next, x, aux
     */
    template <typename TreeT, typename PyClass>
    inline void addTreeMethods(PyClass& cls, const std::vector<const char*>& docs)
    {
        using Bound = typename PyClass::type;
        auto checkNode = [](const TreeT& tree, const std::size_t idx) -> void
        {
            if (idx >= tree.nodes().size()) { throw py::index_error("node index out of range"); }
        };
        cls.def("nDim", [](const Bound& t) -> std::size_t { return t.nDim(); }, docs.at(0))
            .def("leafSize", [](const Bound& t) -> std::size_t { return t.leafSize(); }, docs.at(1))
            .def("nLevels", [](const Bound& t) -> std::size_t { return t.nLevels(); }, docs.at(2))
            .def("nLeaves", [](const Bound& t) -> std::size_t { return t.nLeaves(); }, docs.at(3))
            .def("nodes", [](const Bound& t) -> std::vector<kden::KDTreeNode> { return t.nodes(); }, docs.at(4))
            .def("nodeLo", [checkNode](const Bound& t, const std::size_t idx) -> py::array_t<double>
                {
                    checkNode(t, idx);
                    const auto lo = t.nodeLo(idx);
                    return toArray(std::vector<double>(lo.begin(), lo.end()));
                }, docs.at(5), py::arg("idx"))
            .def("nodeHi", [checkNode](const Bound& t, const std::size_t idx) -> py::array_t<double>
                {
                    checkNode(t, idx);
                    const auto hi = t.nodeHi(idx);
                    return toArray(std::vector<double>(hi.begin(), hi.end()));
                }, docs.at(6), py::arg("idx"))
            .def("neighbors", [](const Bound& t, const QueryArray& x, const std::size_t nNeighbor,
                    const py::object& bandwidth) -> py::tuple
                {
                    return neighbors(t, x, nNeighbor, bandwidth,
                        kden::Bandwidth::identity(t.nDim()));
                }, docs.at(7), py::arg("x"), py::arg("nNeighbor"), py::arg("bandwidth") = py::none())
            .def_static("left", &TreeT::left, docs.at(8), py::arg("idx"))
            .def_static("right", &TreeT::right, docs.at(9), py::arg("idx"))
            .def_static("parent", &TreeT::parent, docs.at(10), py::arg("idx"))
            .def_static("sibling", &TreeT::sibling, docs.at(11), py::arg("idx"))
            .def_static("next", &TreeT::next, docs.at(12), py::arg("idx"))
            .def_property_readonly("x", [](const py::object& self) -> py::array_t<double>
                {
                    const auto& t = py::cast<const Bound&>(self);
                    return readOnlyView(t.data().x_, {static_cast<py::ssize_t>(t.data().size()),
                        static_cast<py::ssize_t>(t.nDim())}, self);
                }, docs.at(13))
            .def_property_readonly("aux", [](const py::object& self) -> py::array_t<Aux>
                {
                    const auto& t = py::cast<const Bound&>(self);
                    return readOnlyView(t.data().aux_, {static_cast<py::ssize_t>(t.data().size())}, self);
                }, docs.at(14));
    }

} // namespace kdenbind
// NOLINTEND(misc-include-cleaner)

#endif // BINDKDEN_HPP
