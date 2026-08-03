/**
 * @file BindInterpolator1DScalar.cpp
 * @author Mark Krumholz
 * @brief Python bindings for interp::Interpolator1D<1>
 * @date 2026-08-02
 */

#include "Bindings.hpp"
#include "../interpolation/Interpolator1D.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>

// The one explicit instantiation definition of Interpolator1D<1> --
// see the comment on the extern template declaration in Bindings.hpp
template class interp::Interpolator1D<1>;

// Numpy-style docstrings for the Python bindings below. No constructor
// is exposed here: an Interpolator1DScalar is only ever obtained from
// Python by way of another bound object that owns one (e.g.
// FilterTabulated.response()), not built directly.
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

static constexpr std::string_view callDocstring = R"doc(Interpolate to a given point.

Parameters
----------
x : float or array_like of float
    The point(s) at which to interpolate.

Returns
-------
value : float or numpy.ndarray of float
    The interpolated value(s) at x.

Throws
------
RuntimeError
    If any requested x is outside the range [xMin(), xMax()].)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindInterpolator1DScalar(py::module_& m)
{
    py::class_<Interp1DScalar, py::smart_holder>(m, "Interpolator1DScalar")
        .def("xMin", &Interp1DScalar::xMin,
                xMinDocstring.data())
        .def("xMax", &Interp1DScalar::xMax,
                xMaxDocstring.data())
        .def("xRange", &Interp1DScalar::xRange,
                xRangeDocstring.data())
        .def("__call__",
                py::vectorize(
                    [](const Interp1DScalar* self, const double x) -> double
                    {
                        if (x < self->xMin() || x > self->xMax())
                        {
                            throw std::runtime_error(
                                "Interpolator1DScalar: x = " + std::to_string(x) +
                                " is outside the allowed range [" +
                                std::to_string(self->xMin()) + ", " +
                                std::to_string(self->xMax()) + "]");
                        }
                        return (*self)(x);
                    }),
                callDocstring.data(),
                py::arg("x"));
}
// NOLINTEND(misc-include-cleaner)
