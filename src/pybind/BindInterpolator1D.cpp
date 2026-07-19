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
#include <vector>

// The one explicit instantiation definition of Interpolator1D<nQty>
// -- see the comment on the extern template declaration in
// Bindings.hpp
template class interp::Interpolator1D<nQty>;

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
                Interp1D::constructorDocstring.data(),
                py::arg("x"),
                py::arg("f")
        )
        .def("xMin", &Interp1D::xMin,
                Interp1D::xMinDocstring.data())
        .def("xMax", &Interp1D::xMax,
                Interp1D::xMaxDocstring.data())
        .def("xRange", &Interp1D::xRange,
                Interp1D::xRangeDocstring.data())
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
                Interp1D::callDocstring.data(),
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
                Interp1D::callIdxDocstring.data(),
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
                Interp1D::callNameDocstring.data(),
                py::arg("x"), py::arg("name"));
}
// NOLINTEND(misc-include-cleaner)
