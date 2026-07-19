/**
 * @file BindInterpolator1D.cpp
 * @author Mark Krumholz
 * @brief Python bindings for interp::Interpolator1D
 * @date 2026-07-20
 */

#include "Bindings.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include "../tracks/TrackCommons.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The one explicit instantiation definition of Interpolator1D<nQty>
// -- see the comment on the extern template declaration in
// Bindings.hpp
template class interp::Interpolator1D<nQty>;

// Numpy-style docstrings for the Python bindings below. Documents
// only the (x, f) constructor overload exposed to Python; the
// interpType parameter is not exposed there (it always defaults to
// gsl_interp_steffen).
static constexpr std::string_view constructorDocstring = R"doc(Construct an Interpolator1D.

Parameters
----------
x : list of float
    Locations of the sample points, in strictly increasing order.
f : list of list of float
    Values of the sample points to interpolate: a sequence containing
    exactly as many arrays as the number of quantities this
    interpolator was built for, each the same length as x.

Throws
------
RuntimeError
    If x is not sorted in increasing order, if x and f are not the
    same length, or if too few distinct points remain (after removing
    duplicates) to build an interpolator.)doc";

static constexpr std::string_view xMinDocstring = R"doc(Get the minimum allowed value of x.

Returns
-------
xmin : float
    The minimum value of x for which interpolation is valid.)doc";

static constexpr std::string_view xMaxDocstring = R"doc(Get the maximum allowed value of x.

Returns
-------
xmax : float
    The maximum value of x for which interpolation is valid.)doc";

static constexpr std::string_view xRangeDocstring = R"doc(Get the allowed range of x.

Returns
-------
xrange : tuple of float
    A 2-element tuple (xmin, xmax) giving the allowed range of x.)doc";

// Docstring for the single-point __call__(x) overload, returning
// every quantity at once
static constexpr std::string_view callDocstring = R"doc(Interpolate every quantity to a given point.

Parameters
----------
x : float
    The point at which to interpolate.

Returns
-------
values : list of float
    The interpolated value of every quantity at x, in the same order
    supplied to the constructor.

Throws
------
RuntimeError
    If x is outside the range [xMin(), xMax()].)doc";

// Docstring for the vectorized, index-selected-quantity __call__(x, idx) overload
static constexpr std::string_view callIdxDocstring = R"doc(Interpolate a single quantity, selected by index, to a given point.

Parameters
----------
x : float or array_like of float
    The point(s) at which to interpolate.
idx : int or array_like of int
    Index of the quantity to interpolate. x and idx are broadcast
    against each other, so e.g. an array of x values together with
    idx = range(n) returns every quantity at every x.

Returns
-------
value : float or numpy.ndarray of float
    The interpolated value(s), with the shape resulting from
    broadcasting x and idx together.

Throws
------
RuntimeError
    If any requested x is outside the range [xMin(), xMax()].)doc";

// Docstring for the vectorized, name-selected-quantity __call__(x, name)
// overload; this overload has no corresponding C++ method of its own --
// it is implemented directly in the Python bindings below as a
// name-to-index lookup wrapping operator()(x, idx)
static constexpr std::string_view callNameDocstring = R"doc(Interpolate a single quantity, selected by name, to a given point.

Parameters
----------
x : float or array_like of float
    The point(s) at which to interpolate.
name : str
    Name of the quantity to interpolate, as recognized by the field
    names this interpolator was built to recognize.

Returns
-------
value : float or numpy.ndarray of float
    The interpolated value(s), with the same shape as x.

Throws
------
RuntimeError
    If name is not a recognized field name, or if any requested x is
    outside the range [xMin(), xMax()].)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindInterpolator1D(py::module_& m)
{
    py::class_<Interp1D, py::smart_holder>(m, "Interpolator1D")
        .def(py::init<
                const std::vector<double>&,                   // x
                const std::array<std::vector<double>, nQty>&  // f
                >(),
                constructorDocstring.data(),
                py::arg("x"),
                py::arg("f")
        )
        .def("xMin", &Interp1D::xMin,
                xMinDocstring.data())
        .def("xMax", &Interp1D::xMax,
                xMaxDocstring.data())
        .def("xRange", &Interp1D::xRange,
                xRangeDocstring.data())
        .def("__call__",
                [](const Interp1D& self, const double x) -> std::array<double, nQty>
                {
                    if (x < self.xMin() || x > self.xMax())
                    {
                        throw std::runtime_error(
                            "Interpolator1D: x = " + std::to_string(x) +
                            " is outside the allowed range [" +
                            std::to_string(self.xMin()) + ", " +
                            std::to_string(self.xMax()) + "]");
                    }
                    return self(x);
                },
                callDocstring.data(),
                py::arg("x"))
        .def("__call__",
                py::vectorize(
                    [](const Interp1D* self, const double x, const std::size_t idx) -> double
                    {
                        if (x < self->xMin() || x > self->xMax())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: x = " + std::to_string(x) +
                                " is outside the allowed range [" +
                                std::to_string(self->xMin()) + ", " +
                                std::to_string(self->xMax()) + "]");
                        }
                        return (*self)(x, idx);
                    }),
                callIdxDocstring.data(),
                py::arg("x"), py::arg("idx"))
        .def("__call__",
                py::vectorize(
                    [](const Interp1D* self, const double x, const std::string name) -> double // NOLINT(performance-unnecessary-value-param); this has to be a value rather than a reference due to pybind11 limitations
                    {
                        const auto *const it = std::ranges::find(tracks::fieldStr, name);
                        if (it == tracks::fieldStr.end())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: unrecognized field name " + name);
                        }
                        if (x < self->xMin() || x > self->xMax())
                        {
                            throw std::runtime_error(
                                "Interpolator1D: x = " + std::to_string(x) +
                                " is outside the allowed range [" +
                                std::to_string(self->xMin()) + ", " +
                                std::to_string(self->xMax()) + "]");
                        }
                        const auto idx = static_cast<std::size_t>(
                            std::distance(tracks::fieldStr.begin(), it));
                        return (*self)(x, idx);
                    }),
                callNameDocstring.data(),
                py::arg("x"), py::arg("name"));
}
// NOLINTEND(misc-include-cleaner)
