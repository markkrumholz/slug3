/**
 * @file BindPDF.cpp
 * @author Mark Krumholz
 * @brief Python bindings for pdfs::PDF
 * @date 2026-08-02
 * @details
 * No constructor is exposed on PDF itself: a PDF is only ever obtained
 * from Python either by way of another bound object that owns one
 * (e.g. SimControls.imf) or via the module-level
 * parsePDFDescriptor() function, mirroring pdfs::parsePDFDescriptor().
 */

#include "Bindings.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFCommons.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include <limits>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner); this is needed for correct Python binding, even if clang-tidy can't recognize it
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    // Map a sampling-method string to pdfs::SamplingMethods, using the
    // same vocabulary pdfs::parseMethod() (PDFFileParser.cpp) itself
    // recognizes when reading a "sampling" line from a PDF descriptor
    // file -- "none" (the internal invalid/dummy sentinel a
    // default-constructed PDF starts with) is deliberately not
    // accepted here either, matching the file parser's own behavior
    auto samplingMethodFromString(const std::string& method) -> pdfs::SamplingMethods
    {
        if (method == "stop_nearest") { return pdfs::SamplingMethods::stopNearest; }
        if (method == "stop_before") { return pdfs::SamplingMethods::stopBefore; }
        if (method == "stop_after") { return pdfs::SamplingMethods::stopAfter; }
        if (method == "stop_50") { return pdfs::SamplingMethods::stop50; }
        if (method == "number") { return pdfs::SamplingMethods::number; }
        if (method == "poisson") { return pdfs::SamplingMethods::poisson; }
        if (method == "sorted_sampling") { return pdfs::SamplingMethods::sorted; }
        throw std::runtime_error(
            "PDF: '" + method + "' is not a recognized sampling method "
            "(expected stop_nearest, stop_before, stop_after, stop_50, "
            "number, poisson, or sorted_sampling)");
    }

    // The inverse of samplingMethodFromString(), including "none" (see
    // its own comment above) so the sampling property can still report
    // it for a PDF that has never had its sampling method set
    auto samplingMethodToString(const pdfs::SamplingMethods method) -> std::string
    {
        switch (method)
        {
            case pdfs::SamplingMethods::stopNearest: return "stop_nearest";
            case pdfs::SamplingMethods::stopBefore:  return "stop_before";
            case pdfs::SamplingMethods::stopAfter:   return "stop_after";
            case pdfs::SamplingMethods::stop50:      return "stop_50";
            case pdfs::SamplingMethods::number:      return "number";
            case pdfs::SamplingMethods::poisson:     return "poisson";
            case pdfs::SamplingMethods::sorted:      return "sorted_sampling";
            case pdfs::SamplingMethods::none:        return "none";
        }
        throw std::runtime_error("PDF: unrecognized sampling method"); // unreachable; exhaustive switch above
    }
} // namespace

// Numpy-style docstrings for the Python bindings below
static constexpr std::string_view validDocstring = R"doc(Check whether this PDF is initialized and usable.

Returns
-------
valid : bool)doc";

static constexpr std::string_view getMinDocstring = R"doc(Get the lower limit of this PDF.

Returns
-------
min : float)doc";

static constexpr std::string_view getMaxDocstring = R"doc(Get the upper limit of this PDF.

Returns
-------
max : float)doc";

static constexpr std::string_view getWeightsDocstring = R"doc(Get the weights of this PDF's segments.

Returns
-------
weights : list of float)doc";

static constexpr std::string_view samplingDocstring = R"doc(This PDF's sampling method.

One of "stop_nearest", "stop_before", "stop_after", "stop_50",
"number", "poisson", or "sorted_sampling" -- the same vocabulary a PDF
descriptor file's own "sampling" line uses. May also read back as
"none" for a PDF whose sampling method was never set, but "none"
itself cannot be assigned.

Throws
------
RuntimeError
    On assignment, if the value is not one of the recognized sampling
    method strings above.)doc";

static constexpr std::string_view normalizePDFDocstring = R"doc(Normalize this PDF, so its segment weights sum to unity.)doc";

static constexpr std::string_view normalizedDocstring = R"doc(Check whether this PDF is normalized.

Returns
-------
normalized : bool)doc";

static constexpr std::string_view callDocstring = R"doc(Evaluate this PDF at a point.

Parameters
----------
x : float or array_like of float
    The point(s) at which to evaluate this PDF.

Returns
-------
value : float or numpy.ndarray of float)doc";

static constexpr std::string_view expectationValueDocstring = R"doc(Get the expectation value of this PDF over its entire range.

Returns
-------
value : float)doc";

static constexpr std::string_view expectationValueRangeDocstring = R"doc(Get the expectation value of this PDF over a given range.

Parameters
----------
a : float
    Lower limit of the range; clamped to this PDF's own lower limit if
    below it.
b : float
    Upper limit of the range; clamped to this PDF's own upper limit if
    above it.

Returns
-------
value : float)doc";

static constexpr std::string_view integralDocstring = R"doc(Get the integral of this PDF over its entire range.

Returns
-------
value : float)doc";

static constexpr std::string_view integralRangeDocstring = R"doc(Get the integral of this PDF over a given range.

Parameters
----------
a : float
    Lower limit of the range; clamped to this PDF's own lower limit if
    below it.
b : float
    Upper limit of the range; clamped to this PDF's own upper limit if
    above it.

Returns
-------
value : float)doc";

static constexpr std::string_view drawDocstring = R"doc(Draw a single random value from this PDF.

Parameters
----------
a : float, optional
    Lower limit of the sampling range; defaults to this PDF's own
    lower limit.
b : float, optional
    Upper limit of the sampling range; defaults to this PDF's own
    upper limit.

Returns
-------
value : float)doc";

static constexpr std::string_view drawNDocstring = R"doc(Draw multiple random values from this PDF.

Parameters
----------
n_draw : int
    Number of samples to draw.
a : float, optional
    Lower limit of the sampling range; defaults to this PDF's own
    lower limit.
b : float, optional
    Upper limit of the sampling range; defaults to this PDF's own
    upper limit.

Returns
-------
values : list of float)doc";

static constexpr std::string_view drawTargetDocstring = R"doc(Draw random values from this PDF targeting a fixed total.

Parameters
----------
target : float
    Target total to draw.
a : float, optional
    Lower limit of the sampling range; defaults to this PDF's own
    lower limit.
b : float, optional
    Upper limit of the sampling range; defaults to this PDF's own
    upper limit.

Returns
-------
values : list of float)doc";

static constexpr std::string_view parsePDFDescriptorDocstring = R"doc(Construct a PDF from a descriptor file.

Parameters
----------
file_name : str
    Name of the PDF descriptor file.

Returns
-------
pdf : PDF

Throws
------
RuntimeError
    If the file cannot be found or parsed.)doc";

// Disable linting for includes -- the pybind macro magic seems to confuse
// the linter
// NOLINTBEGIN(misc-include-cleaner)
void bindPDF(py::module_& m)
{
    py::class_<pdfs::PDF, py::smart_holder>(m, "PDF")
        .def("valid", &pdfs::PDF::valid,
                validDocstring.data())
        .def("getMin", &pdfs::PDF::getMin,
                getMinDocstring.data())
        .def("getMax", &pdfs::PDF::getMax,
                getMaxDocstring.data())
        .def("getWeights",
                [](const pdfs::PDF& self) -> std::vector<double>
                {
                    const auto& w = self.getWeights();
                    return {std::begin(w), std::end(w)};
                },
                getWeightsDocstring.data())
        .def_property("sampling",
                [](const pdfs::PDF& self) -> std::string
                {
                    return samplingMethodToString(self.getSampling());
                },
                [](pdfs::PDF& self, const std::string& method)
                {
                    self.setSampling(samplingMethodFromString(method));
                },
                samplingDocstring.data())
        .def("normalizePDF", &pdfs::PDF::normalizePDF,
                normalizePDFDocstring.data())
        .def("normalized", &pdfs::PDF::normalized,
                normalizedDocstring.data())
        .def("__call__",
                py::vectorize(
                    [](const pdfs::PDF* self, const double x) -> double
                    { return (*self)(x); }),
                callDocstring.data(), py::arg("x"))
        .def("expectationValue",
                [](const pdfs::PDF& self) -> double { return self.expectationValue(); },
                expectationValueDocstring.data())
        .def("expectationValue",
                [](const pdfs::PDF& self, double a, double b) -> double
                { return self.expectationValue(a, b); },
                expectationValueRangeDocstring.data(),
                py::arg("a"), py::arg("b"))
        .def("integral",
                [](const pdfs::PDF& self) -> double { return self.integral(); },
                integralDocstring.data())
        .def("integral",
                [](const pdfs::PDF& self, double a, double b) -> double
                { return self.integral(a, b); },
                integralRangeDocstring.data(),
                py::arg("a"), py::arg("b"))
        .def("draw",
                [](const pdfs::PDF& self, double a, double b) -> double
                { return self.draw(a, b); },
                drawDocstring.data(),
                py::arg("a") = std::numeric_limits<double>::lowest(),
                py::arg("b") = std::numeric_limits<double>::max())
        .def("draw",
                [](const pdfs::PDF& self, unsigned int nDraw, double a, double b)
                    -> std::vector<double>
                { return self.draw(nDraw, a, b); },
                drawNDocstring.data(),
                py::arg("n_draw"),
                py::arg("a") = std::numeric_limits<double>::lowest(),
                py::arg("b") = std::numeric_limits<double>::max())
        .def("drawTarget",
                [](const pdfs::PDF& self, double target, double a, double b)
                    -> std::vector<double>
                { return self.drawTarget(target, a, b); },
                drawTargetDocstring.data(),
                py::arg("target"),
                py::arg("a") = std::numeric_limits<double>::lowest(),
                py::arg("b") = std::numeric_limits<double>::max());

    m.def("parsePDFDescriptor",
            [](const std::string& fileName) -> std::unique_ptr<pdfs::PDF>
            {
                return std::make_unique<pdfs::PDF>(pdfs::parsePDFDescriptor(fileName));
            },
            parsePDFDescriptorDocstring.data(),
            py::arg("file_name"));
}
// NOLINTEND(misc-include-cleaner)
