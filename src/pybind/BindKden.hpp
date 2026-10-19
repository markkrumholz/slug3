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
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h> // NOLINT(misc-include-cleaner); used throughout, via the py alias
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <span>
#include <string>
#include <tuple>
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
     * @brief Check that arrays a tree will reorder in place do not share memory
     * @param arrays The arrays (C-contiguous, so each occupies
     *   [data(), data() + nbytes()))
     * @param names Their names, for the error message
     * @throws py::value_error if any two overlap
     * @details
     * Points' positions, weights, and auxiliary data are reordered (and
     * the weights normalized) in place, so overlapping arrays would
     * corrupt one another -- e.g. weights given as a column of the
     * positions.
     */
    inline void checkNoOverlap(const std::vector<py::array>& arrays, const std::vector<std::string>& names)
    {
        for (std::size_t i = 0; i < arrays.size(); ++i)
        {
            const auto* pi = static_cast<const char*>(arrays[i].data());
            const auto ni = static_cast<std::size_t>(arrays[i].nbytes());
            for (std::size_t j = i + 1; j < arrays.size(); ++j)
            {
                const auto* pj = static_cast<const char*>(arrays[j].data());
                const auto nj = static_cast<std::size_t>(arrays[j].nbytes());
                if (ni > 0 && nj > 0 && pi < pj + nj && pj < pi + ni) // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                {
                    throw py::value_error(names[i] + " and " + names[j] + " must not share memory "
                        "(they are reordered in place)");
                }
            }
        }
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
     * @brief Per-point measurement error covariances, from Python
     * @param errCov An (nErr, nErr) array, shared by every point, or an
     *   (M, nErr, nErr) array of one per point
     * @param m Number of points, M
     * @param nErr Number of dimensions with errors
     * @return One nErr x nErr matrix per point
     * @throws py::value_error if errCov has neither shape
     */
    inline auto batchErrors(const py::object& errCov, const std::size_t m, const std::size_t nErr)
        -> std::vector<Eigen::MatrixXd>
    {
        const auto arr = py::cast<QueryArray>(errCov);
        const auto k = static_cast<py::ssize_t>(nErr);
        const bool shared = arr.ndim() == 2 && arr.shape(0) == k && arr.shape(1) == k;
        const bool each = arr.ndim() == 3 && static_cast<std::size_t>(arr.shape(0)) == m &&
            arr.shape(1) == k && arr.shape(2) == k;
        if (!shared && !each)
        {
            throw py::value_error("errCov must have shape (nErr, nErr), shared by every point, or (M, nErr, "
                "nErr), one per point, with nErr = len(errDims) = " + std::to_string(nErr));
        }
        std::vector<Eigen::MatrixXd> out(m, Eigen::MatrixXd(k, k));
        const double* p = arr.data();
        const std::size_t kk = nErr * nErr;
        for (std::size_t i = 0; i < m; ++i)
        {
            const double* src = shared ? p : p + (i * kk); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            for (std::size_t a = 0; a < nErr; ++a)
            {
                for (std::size_t b = 0; b < nErr; ++b)
                {
                    out[i](static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = src[(a * nErr) + b]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,cppcoreguidelines-pro-bounds-pointer-arithmetic)
                }
            }
        }
        return out;
    }

    /**
     * @brief The single bandwidth a query with measurement errors uses
     * @param bandwidth None (the Euclidean metric) or one Bandwidth
     * @param euclidean The Euclidean metric
     * @return The bandwidth
     * @throws py::value_error if bandwidth is a sequence
     */
    inline auto errorsBandwidth(const py::object& bandwidth, const kden::Bandwidth& euclidean) -> kden::Bandwidth
    {
        if (bandwidth.is_none()) { return euclidean; }
        if (!py::isinstance<kden::Bandwidth>(bandwidth))
        {
            throw py::value_error("with errCov, bandwidth must be None or a single Bandwidth");
        }
        return py::cast<kden::Bandwidth>(bandwidth);
    }

    /**
     * @brief The number of neighbors to find for each of a batch of searches, from Python
     * @param nNeighbor An int, the same for every search, or a sequence
     *   of one non-negative int per search
     * @param m Number of searches
     * @return One count per search
     * @throws py::value_error if a sequence has the wrong length or a
     *   negative element
     */
    inline auto batchCounts(const py::object& nNeighbor, const std::size_t m) -> std::vector<std::size_t>
    {
        if (py::isinstance<py::int_>(nNeighbor) || py::isinstance(nNeighbor, py::module_::import("numpy").attr("integer")))
        {
            const auto k = py::cast<py::ssize_t>(nNeighbor);
            if (k < 0) { throw py::value_error("nNeighbor must be non-negative"); }
            std::vector<std::size_t> counts(m, static_cast<std::size_t>(k));
            return counts;
        }
        const auto ks = py::cast<std::vector<py::ssize_t>>(nNeighbor);
        if (ks.size() != m)
        {
            throw py::value_error("nNeighbor must be an int, or a sequence of one int per point");
        }
        std::vector<std::size_t> out;
        out.reserve(m);
        for (const auto k : ks)
        {
            if (k < 0) { throw py::value_error("nNeighbor must be non-negative"); }
            out.push_back(static_cast<std::size_t>(k));
        }
        return out;
    }

    /**
     * @brief Find nearest neighbors for one point or a batch, for Python
     * @tparam TreeT The bound tree type
     * @param tree The tree
     * @param x One point or an M x nDim array
     * @param nNeighbor Number of neighbors per point: an int, or (for a
     *   batch) a sequence of one per point (see batchCounts())
     * @param bandwidth None, a Bandwidth, or (without errCov, for a
     *   batch) a sequence of one per point
     * @param errDims Dimensions with measurement errors (empty for none)
     * @param errCov None, or their covariances (see batchErrors())
     * @param euclidean The Euclidean metric
     * @return (idx, dist2): arrays of shape (k,) for one point, or
     *   (M, k) for a batch, with k the most neighbors any search found;
     *   searches that found fewer are padded with idx -1 and dist2 NaN
     * @throws py::value_error if nNeighbor, errDims, errCov, or
     *   bandwidth is invalid
     * @details
     * Calls the tree's batch neighbors(), which runs in parallel,
     * without the GIL.
     */
    template <typename TreeT>
    inline auto neighbors(const TreeT& tree, const QueryArray& x, const py::object& nNeighbor,
        const py::object& bandwidth, const std::vector<std::size_t>& errDims, const py::object& errCov,
        const kden::Bandwidth& euclidean) -> py::tuple
    {
        const auto [pts, single] = queryPoints(x, tree.nDim());
        const std::vector<std::size_t> ks = batchCounts(nNeighbor, pts.size());
        std::vector<std::vector<kden::Neighbor>> res;
        if (!errCov.is_none())
        {
            const kden::Bandwidth bw = errorsBandwidth(bandwidth, euclidean);
            const auto errs = batchErrors(errCov, pts.size(), errDims.size());
            const py::gil_scoped_release release;
            res = tree.neighbors(pts, ks, errDims, errs, bw);
        }
        else
        {
            if (!errDims.empty()) { throw py::value_error("errDims was given without errCov"); }
            const auto [bws, shared] = batchBandwidths(bandwidth, pts.size(), euclidean);
            const py::gil_scoped_release release;
            res = shared ? tree.neighbors(pts, ks, bws.front()) : tree.neighbors(pts, ks, bws);
        }
        std::size_t k = 0;
        for (const auto& r : res) { k = std::max(k, r.size()); }
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
                const bool found = j < res[q].size(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                pi[(q * k) + j] = found ? static_cast<Aux>(res[q][j].idx_) : Aux{-1}; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                pd[(q * k) + j] = found ? res[q][j].dist2_ : std::numeric_limits<double>::quiet_NaN(); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
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
     * @param bandwidth None, a Bandwidth, or (without errCov, for a
     *   batch) a sequence of one per point
     * @param errDims Dimensions with measurement errors (empty for none)
     * @param errCov None, or their covariances (see batchErrors())
     * @param euclidean The Euclidean metric
     * @return A float for one point, or an array of M for a batch
     * @throws py::value_error if errDims is given without errCov, or
     *   errCov or bandwidth is invalid
     * @details
     * Calls KernelDensity's batch pdf(), which runs in parallel, without
     * the GIL.
     */
    inline auto pdf(const KDE& kde, const QueryArray& x, const double relTol, const double absTol,
        const py::object& bandwidth, const std::vector<std::size_t>& errDims, const py::object& errCov,
        const kden::Bandwidth& euclidean) -> py::object
    {
        const auto [pts, single] = queryPoints(x, kde.nDim());
        std::vector<double> out;
        if (!errCov.is_none())
        {
            const kden::Bandwidth bw = errorsBandwidth(bandwidth, euclidean);
            const auto errs = batchErrors(errCov, pts.size(), errDims.size());
            const py::gil_scoped_release release;
            out = kde.pdf(pts, errDims, errs, relTol, absTol, bw);
        }
        else
        {
            if (!errDims.empty()) { throw py::value_error("errDims was given without errCov"); }
            const auto [bws, shared] = batchBandwidths(bandwidth, pts.size(), euclidean);
            const py::gil_scoped_release release;
            out = shared ? kde.pdf(pts, relTol, absTol, bws.front()) : kde.pdf(pts, relTol, absTol, bws);
        }
        if (single) { return py::float_(out.front()); }
        return toArray(out);
    }

    /**
     * @brief Condition a kernel density estimate on one point or a batch, for Python
     * @param kde The estimate
     * @param x0 One point or an M x nDim array
     * @param massTol As for KernelDensity::conditional()
     * @param marginalDims As for KernelDensity::conditional()
     * @param bandwidth None or a Bandwidth
     * @param leafSize As for KernelDensity::conditional()
     * @param errDims Dimensions with measurement errors (empty for none)
     * @param errCov None, or their covariances (see batchErrors())
     * @param euclidean The Euclidean metric
     * @return A ConditionalDensity for one point, or a list of M for a batch
     * @throws py::value_error if errDims is given without errCov, or
     *   errCov or bandwidth is invalid
     * @details
     * Calls KernelDensity's batch conditional(), which runs in
     * parallel, without the GIL.
     */
    inline auto conditional(const KDE& kde, const QueryArray& x0, const double massTol,
        const std::vector<std::size_t>& marginalDims, const py::object& bandwidth, const std::size_t leafSize,
        const std::vector<std::size_t>& errDims, const py::object& errCov,
        const kden::Bandwidth& euclidean) -> py::object
    {
        const auto [pts, single] = queryPoints(x0, kde.nDim());
        if (errCov.is_none() && !errDims.empty()) { throw py::value_error("errDims was given without errCov"); }
        const kden::Bandwidth bw = errorsBandwidth(bandwidth, euclidean);
        std::vector<Conditional> cds;
        if (!errCov.is_none())
        {
            const auto errs = batchErrors(errCov, pts.size(), errDims.size());
            const py::gil_scoped_release release;
            cds = kde.conditional(pts, errDims, errs, marginalDims, massTol, bw, leafSize);
        }
        else
        {
            const py::gil_scoped_release release;
            cds = kde.conditional(pts, marginalDims, massTol, bw, leafSize);
        }
        if (single) { return py::cast(std::move(cds.front())); }
        py::list out;
        for (auto& cd : cds) { out.append(py::cast(std::move(cd))); }
        return out;
    }

    /**
     * @brief Find the sample points, weights, and auxiliary data of conditional estimates at one point or a batch, for Python
     * @param kde The estimate
     * @param x0 One point or an M x nDim array
     * @param massTol As for KernelDensity::conditionalSample()
     * @param marginalDims As for KernelDensity::conditionalSample()
     * @param bandwidth None or a Bandwidth
     * @param errDims Dimensions with measurement errors (empty for none)
     * @param errCov None, or their covariances (see batchErrors())
     * @param euclidean The Euclidean metric
     * @return For one point, a tuple of the points (K x nR array),
     *   their weights (K array), and their auxiliary data (K array);
     *   for a batch, a list of one such tuple per point
     * @throws py::value_error if errDims is given without errCov, or
     *   errCov or bandwidth is invalid
     * @details
     * Calls KernelDensity's batch conditionalSample(), which runs in
     * parallel, without the GIL.
     */
    inline auto conditionalSample(const KDE& kde, const QueryArray& x0, const double massTol,
        const std::vector<std::size_t>& marginalDims, const py::object& bandwidth,
        const std::vector<std::size_t>& errDims, const py::object& errCov,
        const kden::Bandwidth& euclidean) -> py::object
    {
        const auto query = queryPoints(x0, kde.nDim());
        const auto& pts = query.first;
        if (errCov.is_none() && !errDims.empty()) { throw py::value_error("errDims was given without errCov"); }
        const kden::Bandwidth bw = errorsBandwidth(bandwidth, euclidean);
        std::vector<KDE::Sample> samples;
        if (!errCov.is_none())
        {
            const auto errs = batchErrors(errCov, pts.size(), errDims.size());
            const py::gil_scoped_release release;
            samples = kde.conditionalSample(pts, errDims, errs, marginalDims, massTol, bw);
        }
        else
        {
            const py::gil_scoped_release release;
            samples = kde.conditionalSample(pts, marginalDims, massTol, bw);
        }
        // A sample as (points, weights, aux); it always has at least one point
        auto toPython = [](const KDE::Sample& sample) -> py::tuple
        {
            const auto& x = std::get<0>(sample);
            const auto& w = std::get<1>(sample);
            const auto nKept = static_cast<py::ssize_t>(w.size());
            const auto nR = static_cast<py::ssize_t>(x.size() / w.size());
            py::array_t<double> xArr({nKept, nR});
            std::ranges::copy(x, xArr.mutable_data());
            return py::make_tuple(xArr, toArray(w), toArray(std::get<2>(sample)));
        };
        if (query.second) { return toPython(samples.front()); }
        py::list out;
        for (const auto& sample : samples) { out.append(toPython(sample)); }
        return out;
    }

    /**
     * @brief Evaluate one or several percentiles, for Python
     * @tparam Eval Callable taking a vector of percentiles and returning
     *   one value per percentile (a batch percentile())
     * @param pct A float, or an array_like of floats (any shape)
     * @param eval The batch evaluation
     * @return A float for a float pct, or an array of pct's shape
     * @details
     * Runs eval without the GIL.
     */
    template <typename Eval>
    inline auto percentiles(const py::object& pct, const Eval& eval) -> py::object
    {
        const auto arr = py::cast<QueryArray>(pct);
        const std::vector<double> ps(arr.data(), arr.data() + arr.size()); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::vector<double> out;
        {
            const py::gil_scoped_release release;
            out = eval(ps);
        }
        if (arr.ndim() == 0) { return py::float_(out.front()); }
        const std::vector<py::ssize_t> shape(arr.shape(), arr.shape() + arr.ndim()); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        py::array_t<double> result(shape);
        std::ranges::copy(out, result.mutable_data());
        return result;
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
            .def("neighbors", [](const Bound& t, const QueryArray& x, const py::object& nNeighbor,
                    const py::object& bandwidth, const std::vector<std::size_t>& errDims,
                    const py::object& errCov) -> py::tuple
                {
                    return neighbors(t, x, nNeighbor, bandwidth, errDims, errCov,
                        kden::Bandwidth::identity(t.nDim()));
                }, docs.at(7), py::arg("x"), py::arg("nNeighbor"), py::arg("bandwidth") = py::none(),
                py::arg("errDims") = std::vector<std::size_t>{}, py::arg("errCov") = py::none())
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
