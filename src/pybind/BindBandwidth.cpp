/**
 * @file BindBandwidth.cpp
 * @author Mark Krumholz
 * @brief Python bindings for kden::Bandwidth and kden::Bandwidth::BoxBounder
 * @date 2026-10-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "BindKden.hpp"
#include "../kden/Bandwidth.hpp"
#include "Bindings.hpp"
#include <algorithm>
#include <cstddef>
#include <pybind11/eigen.h> // NOLINT(misc-include-cleaner); needed for Eigen::MatrixXd <-> numpy conversions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); needed for list/vector conversions
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static constexpr std::string_view classDocstring = R"doc(The metric of a kernel: an inverse squared bandwidth matrix.

Distances are measured as d^T M d, with M the inverse squared
bandwidth matrix (for a diagonal bandwidth h_i, M = diag(1 / h_i^2)),
which may be a full (non-diagonal) matrix. Dimensions can be left
unspecified, by giving their coordinates as NaN: distances are then
measured in the other dimensions, with the matrix reduced to the
Schur complement over them (see reduce()), which is the inverse of
the specified block of the covariance, i.e. the metric of the
marginal of a Gaussian kernel.)doc";

static constexpr std::string_view constructorDocstring = R"doc(Construct a Bandwidth from an inverse squared bandwidth matrix.

Parameters
----------
h2Inv : numpy.ndarray of float, shape (n, n)
    The matrix M; must be symmetric and positive semi-definite.

Raises
------
ValueError
    If h2Inv is not square, is empty, is not symmetric, or is not
    positive semi-definite.)doc";

static constexpr std::string_view identityDocstring = R"doc(The Euclidean metric.

Parameters
----------
nDim : int
    Number of dimensions.

Returns
-------
bandwidth : Bandwidth
    A Bandwidth holding the nDim x nDim identity matrix.)doc";

static constexpr std::string_view nanPatternDocstring = R"doc(Which coordinates of a point are unspecified (NaN).

Parameters
----------
x : array_like of float
    The point.

Returns
-------
pattern : list of bool
    True for each NaN coordinate.)doc";

static constexpr std::string_view nDimDocstring = R"doc(Number of dimensions.

Returns
-------
nDim : int
    The number of dimensions n.)doc";

static constexpr std::string_view h2InvDocstring = R"doc(The inverse squared bandwidth matrix.

Returns
-------
h2Inv : numpy.ndarray of float, shape (n, n)
    A copy of the matrix; for a reduced Bandwidth, its rows and
    columns for unspecified dimensions are zero.)doc";

static constexpr std::string_view unspecifiedDocstring = R"doc(Which dimensions are unspecified.

Returns
-------
unspecified : list of bool
    True for each unspecified dimension (all False unless this
    Bandwidth came from reduce()).)doc";

static constexpr std::string_view isReducedDocstring = R"doc(Whether any dimension is unspecified.

Returns
-------
reduced : bool
    True if this Bandwidth came from reduce() with at least one
    unspecified dimension.)doc";

static constexpr std::string_view isDiagonalDocstring = R"doc(Whether the matrix is diagonal over the specified dimensions.

Returns
-------
diagonal : bool
    True if distances and bounds use the fast diagonal path.)doc";

static constexpr std::string_view nSpecifiedDocstring = R"doc(Number of specified dimensions.

Returns
-------
nSpecified : int
    The number of dimensions not flagged in unspecified().)doc";

static constexpr std::string_view detDocstring = R"doc(Determinant of the matrix over the specified dimensions.

Returns
-------
det : float
    det M_s, with M_s the matrix's block over the specified dimensions
    (the Schur complement, for a reduced Bandwidth); 1 if no dimension
    is specified. The integral of exp(-d^T M_s d) over the specified
    dimensions is pi^(nSpecified / 2) / sqrt(det).)doc";

static constexpr std::string_view reduceDocstring = R"doc(Reduce the metric for points with some coordinates unspecified.

Parameters
----------
unspecified : list of bool
    Which dimensions will be unspecified; must have n elements, and
    include every dimension this Bandwidth already leaves unspecified.

Returns
-------
reduced : Bandwidth
    A Bandwidth whose matrix, over the dimensions s that remain
    specified, is the Schur complement M_ss - M_sf M_ff^-1 M_fs of this
    one's, and is zero elsewhere.

Raises
------
ValueError
    If unspecified has the wrong size or omits a dimension this
    Bandwidth already leaves unspecified, or the block of the matrix
    over the newly unspecified dimensions is not positive definite.)doc";

static constexpr std::string_view convolveDocstring = R"doc(Convolve with another bandwidth.

Parameters
----------
other : Bandwidth
    The other Bandwidth; must have the same number of dimensions and
    the same unspecified dimensions.

Returns
-------
convolved : Bandwidth
    The Bandwidth of the convolution of the two kernels, whose matrix
    over the specified dimensions is (M1^-1 + M2^-1)^-1 (convolution
    adds covariances). Computed as M1 (M1 + M2)^-1 M2, so neither
    matrix need be invertible.

Raises
------
ValueError
    If other has a different number of dimensions or different
    unspecified dimensions, or M1 + M2 is not positive definite.)doc";

static constexpr std::string_view matchesDocstring = R"doc(Whether a point's pattern of NaNs matches this Bandwidth's unspecified dimensions.

Parameters
----------
x : array_like of float
    The point; must have n elements.

Returns
-------
matches : bool
    True if x is NaN in exactly the unspecified dimensions.)doc";

static constexpr std::string_view dist2Docstring = R"doc(Squared distance between points.

Parameters
----------
x1, x2 : array_like of float, shape (n,) or (M, n)
    The points. A single point is paired with every row of the other
    argument; two arrays of points are paired row by row. A NaN in a
    coordinate of either point drops that dimension (the zero
    separation of a point from a line or plane); if this Bandwidth is
    not diagonal, it should have been reduced for the pattern of NaNs
    (see reduce()).

Returns
-------
dist2 : float or numpy.ndarray of float
    d^T M d, over the dimensions specified in both points; a float if
    both arguments are single points.

Raises
------
ValueError
    If the points do not have n coordinates, or two arrays of points
    have different numbers of rows.)doc";

static constexpr std::string_view boxBounderDocstring = R"doc(Make a box bounder for a point.

Parameters
----------
x : array_like of float
    The point; must match this Bandwidth (see matches()).

Returns
-------
bounder : BoxBounder
    Bounds on the squared distance from x to axis-aligned boxes. It
    keeps this Bandwidth alive.)doc";

static constexpr std::string_view boxBounderClassDocstring = R"doc(Rigorous bounds on the squared distance from one point to axis-aligned boxes.

Made by Bandwidth.boxBounder(). The bounds are exact for a diagonal
metric, and rigorous (computed along the rows of an LDL^T factor of
the matrix) for a full one. Holds scratch space, so must not be used
from several threads at once.)doc";

static constexpr std::string_view lowerDocstring = R"doc(Lower bound on the squared distance from the point to a box.

Parameters
----------
lo, hi : array_like of float
    Lower and upper corners of the box (n elements each).

Returns
-------
lower : float
    A value no larger than the squared distance (as computed by
    Bandwidth.dist2()) from the point to any point in the box.)doc";

static constexpr std::string_view upperDocstring = R"doc(Upper bound on the squared distance from the point to a box.

Parameters
----------
lo, hi : array_like of float
    Lower and upper corners of the box (n elements each).

Returns
-------
upper : float
    A value no smaller than the squared distance (as computed by
    Bandwidth.dist2()) from the point to any point in the box.)doc";

static constexpr std::string_view boundsDocstring = R"doc(Lower and upper bounds together.

Parameters
----------
lo, hi : array_like of float
    Lower and upper corners of the box (n elements each).

Returns
-------
bounds : tuple of float
    (lower(lo, hi), upper(lo, hi)), computed in a single pass.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
namespace {

    /**
     * @brief Check a point has a Bandwidth's number of dimensions
     * @param bw The Bandwidth
     * @param x The point
     * @param name Its name, for the error message
     * @throws py::value_error if it does not
     */
    void checkSize(const kden::Bandwidth& bw, const std::vector<double>& x, const char* name)
    {
        if (x.size() != bw.nDim())
        {
            throw py::value_error(std::string(name) + " must have nDim elements");
        }
    }

    /**
     * @brief Check a box has a BoxBounder's number of dimensions
     * @param b The BoxBounder
     * @param lo Lower corner of the box
     * @param hi Upper corner of the box
     * @throws py::value_error if lo or hi has the wrong size
     */
    void checkBox(const kden::Bandwidth::BoxBounder& b, const std::vector<double>& lo,
        const std::vector<double>& hi)
    {
        if (lo.size() != b.nDim() || hi.size() != b.nDim())
        {
            throw py::value_error("lo and hi must have nDim elements");
        }
    }

} // namespace

void bindBandwidth(py::module_& m)
{
    using kden::Bandwidth;
    py::class_<Bandwidth::BoxBounder, py::smart_holder>(m, "BoxBounder", boxBounderClassDocstring.data())
        .def("lower", [](const Bandwidth::BoxBounder& b, const std::vector<double>& lo,
                const std::vector<double>& hi) -> double
                {
                    checkBox(b, lo, hi);
                    return b.lower(lo, hi);
                },
                lowerDocstring.data(), py::arg("lo"), py::arg("hi"))
        .def("upper", [](const Bandwidth::BoxBounder& b, const std::vector<double>& lo,
                const std::vector<double>& hi) -> double
                {
                    checkBox(b, lo, hi);
                    return b.upper(lo, hi);
                },
                upperDocstring.data(), py::arg("lo"), py::arg("hi"))
        .def("bounds", [](const Bandwidth::BoxBounder& b, const std::vector<double>& lo,
                const std::vector<double>& hi) -> std::pair<double, double>
                {
                    checkBox(b, lo, hi);
                    return b.bounds(lo, hi);
                },
                boundsDocstring.data(), py::arg("lo"), py::arg("hi"));

    py::class_<Bandwidth, py::smart_holder>(m, "Bandwidth", classDocstring.data())
        .def(py::init<Eigen::MatrixXd>(), constructorDocstring.data(), py::arg("h2Inv"))
        .def_static("identity", &Bandwidth::identity, identityDocstring.data(), py::arg("nDim"))
        .def_static("nanPattern", [](const std::vector<double>& x) -> std::vector<bool>
                { return Bandwidth::nanPattern(x); },
                nanPatternDocstring.data(), py::arg("x"))
        .def("nDim", &Bandwidth::nDim, nDimDocstring.data())
        .def("h2Inv", &Bandwidth::h2Inv, h2InvDocstring.data(), py::return_value_policy::copy)
        .def("unspecified", &Bandwidth::unspecified, unspecifiedDocstring.data())
        .def("isReduced", &Bandwidth::isReduced, isReducedDocstring.data())
        .def("isDiagonal", &Bandwidth::isDiagonal, isDiagonalDocstring.data())
        .def("nSpecified", &Bandwidth::nSpecified, nSpecifiedDocstring.data())
        .def("det", &Bandwidth::det, detDocstring.data())
        .def("reduce", &Bandwidth::reduce, reduceDocstring.data(), py::arg("unspecified"))
        .def("convolve", &Bandwidth::convolve, convolveDocstring.data(), py::arg("other"))
        .def("matches", [](const Bandwidth& bw, const std::vector<double>& x) -> bool
                {
                    checkSize(bw, x, "x");
                    return bw.matches(x);
                }, matchesDocstring.data(), py::arg("x"))
        .def("dist2", [](const Bandwidth& bw, const kdenbind::QueryArray& x1,
                const kdenbind::QueryArray& x2) -> py::object
                {
                    auto [p1, single1] = kdenbind::queryPoints(x1, bw.nDim());
                    auto [p2, single2] = kdenbind::queryPoints(x2, bw.nDim());
                    if (!single1 && !single2 && p1.size() != p2.size())
                    {
                        throw py::value_error("x1 and x2 must have the same number of rows");
                    }
                    const std::size_t n = std::max(p1.size(), p2.size());
                    std::vector<double> out(n);
                    std::vector<double> scratch;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        out[i] = bw.dist2(p1[single1 ? 0 : i], p2[single2 ? 0 : i], scratch);
                    }
                    if (single1 && single2) { return py::float_(out.front()); }
                    return kdenbind::toArray(out);
                }, dist2Docstring.data(), py::arg("x1"), py::arg("x2"))
        .def("boxBounder", [](const Bandwidth& bw, const std::vector<double>& x) -> Bandwidth::BoxBounder
                {
                    checkSize(bw, x, "x");
                    if (!bw.matches(x))
                    {
                        throw py::value_error("x's pattern of NaNs must match the bandwidth's unspecified "
                            "dimensions (see reduce())");
                    }
                    return bw.boxBounder(x);
                }, boxBounderDocstring.data(), py::arg("x"),
                // The bounder refers to the Bandwidth
                py::keep_alive<0, 1>());
}
// NOLINTEND(misc-include-cleaner)
