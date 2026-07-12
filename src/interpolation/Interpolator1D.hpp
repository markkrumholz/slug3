/*
 * @file Interpolator1D.hpp
 * @author Mark Krumholz
 * @brief Provides a class to represent a single 1D interpolator
 * @date 2024-06-27
*/

#ifndef INTERPOLATOR1D_HPP
#define INTERPOLATOR1D_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

// Disable linting for array bounds checking in this
// file, since the overhead associated with enforcing
// such checks severely interferes with performance
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

namespace interp
{

    /**
     * @class Interpolator1D
     * @tparam NF Number of quantities to interpolate
     * @brief A class to handle 1D interpolation
     */
    template <size_t NF = 1>
    class Interpolator1D
    {
    public:

        // Constructors and destructor

        /**
         * @brief Numpy-style docstring for the Python constructor binding
         * @details
         * Documents only the (x, f) overload exposed to Python; the
         * interpType parameter is not exposed there (it always
         * defaults to gsl_interp_steffen).
         */
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

        /**
         * @brief Construct an Interpolator1D
         * @param x Locations of sample points to be interpolated
         * @param f Values of sample points to be interpolated
         * @param interpType The type of interpolation to use
        */
        Interpolator1D(
            const std::vector<double>& x,
            const std::array<std::vector<double>, NF>& f,
            const gsl_interp_type* interpType = gsl_interp_steffen
        ) : 
        x_(x),
        f_(f),
        interp_({nullptr}),
        acc_(nullptr)
        {
            checkSorted();
            cleanDuplicates();
            checkMinSize(interpType);
            interpInit(interpType);
        }

        Interpolator1D(
            const std::vector<double>& x,
            const std::vector<double>& f,
            const gsl_interp_type* interpType = gsl_interp_steffen
        ) requires (NF == 1)
        : 
        Interpolator1D(x, 
            std::array<std::vector<double>, 1>({f}),
            interpType)
        { }

        virtual ~Interpolator1D()
        {
            // Free gsl data
            for (auto& i : interp_) { gsl_interp_free(i); }
            gsl_interp_accel_free(acc_);
        }

        // Disallow move and copy constructors, since these would make
        // copies of gsl opaque objects that we can't manage with
        // smart pointers, resulting in potential memory corruption
        // when two Interpolator1D objects pointing to the same gsl
        // opaque objects pass out of scope and try to free the same
        // memory. We manage this by only ever dealing with unique_ptr's
        // to Interpolator1D objects elsewhere in the code, to ensure
        // that the destructor is invoked once and only once.
        Interpolator1D(const Interpolator1D&) = delete;
        auto operator=(const Interpolator1D&) -> Interpolator1D& = delete;
        Interpolator1D(Interpolator1D&&) = delete;
        auto operator=(Interpolator1D&&) -> Interpolator1D& = delete;

        /**
         * @brief Numpy-style docstring for the Python xMin binding
         */
        static constexpr std::string_view xMinDocstring = R"doc(Get the minimum allowed value of x.

Returns
-------
xmin : float
    The minimum value of x for which interpolation is valid.)doc";

        /**
         * @brief Get minimum allowed x
         * @return Minimum allowed x
         */
        [[nodiscard]] auto xMin() const { return x_.front(); }

        /**
         * @brief Numpy-style docstring for the Python xMax binding
         */
        static constexpr std::string_view xMaxDocstring = R"doc(Get the maximum allowed value of x.

Returns
-------
xmax : float
    The maximum value of x for which interpolation is valid.)doc";

        /**
         * @brief Get maximum allowed x
         * @return Maximum allowed x
         */
        [[nodiscard]] auto xMax() const { return x_.back(); }

        /**
         * @brief Numpy-style docstring for the Python xRange binding
         */
        static constexpr std::string_view xRangeDocstring = R"doc(Get the allowed range of x.

Returns
-------
xrange : tuple of float
    A 2-element tuple (xmin, xmax) giving the allowed range of x.)doc";

        /**
         * @brief Get allowed range in x
         * @return Allowed range in x
         */
        [[nodiscard]] auto xRange() const { return std::make_pair(xMin(), xMax()); }

        /**
         * @brief Numpy-style docstring for the Python __call__(x) binding
         * (the single-point overload returning every quantity at once)
         */
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

        /**
         * @brief Numpy-style docstring for the Python __call__(x, idx)
         * binding (the vectorized, index-selected-quantity overload)
         */
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

        /**
         * @brief Numpy-style docstring for the Python __call__(x, name)
         * binding (the vectorized, name-selected-quantity overload;
         * this overload has no corresponding C++ method of its own --
         * it is implemented directly in the Python bindings as a
         * name-to-index lookup wrapping operator()(x, idx) -- but is
         * documented here since it is exposed as part of
         * Interpolator1D's Python __call__ family)
         */
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

        /**
         * @brief Interpolate to a given point
         * @param x The point to which to interpolate
         * @return The interpolated value(s)
         * @details
         * Return type is a double if NF = 1, and a std::array otherwise.
         */
        [[nodiscard]] auto operator()(double x) const
        {
            assert(x >= x_.front() && x <= x_.back());
            std::array<double, NF> result;
            for (size_t i = 0; i < NF; ++i)
            {
                result[i] = gsl_interp_eval(interp_[i], x_.data(),
                    f_[i].data(), x, acc_);
            }
            if constexpr(NF > 1) { return result; }
            else { return result[0]; }
        }

        /**
         * @brief Interpolate a single quantity to a given point
         * @param x The point to which to interpolate
         * @param idx Quantity to interpolate; must be <= NF
         * @return The interpolated value
        */
        [[nodiscard]] auto operator()(double x, size_t idx) const
        {
            assert(x >= x_.front() && x <= x_.back());
            assert(idx < NF);
            return gsl_interp_eval(interp_[idx], x_.data(), f_[idx].data(), x, acc_);
        }

    private:

        /**
         * @brief Check that x is sorted
         * @details
         * Must run before cleanDuplicates(), on the raw, as-supplied
         * x_, so that a genuine ordering violation (as opposed to a
         * run of duplicate values, which cleanDuplicates() collapses)
         * is always caught here.
         */
        void checkSorted()
        {
            static_assert(NF > 0, "Interpolator1D: number of quantities to interpolate must be >= 1");
            if (!std::ranges::is_sorted(x_))
            {
                throw std::runtime_error("Interpolator1D: x must be strictly increasing");
            }
        }

        /**
         * @brief Check that there are enough points left for interpType
         * @details
         * Must run after cleanDuplicates(), so that this reflects the
         * true, final number of distinct points rather than the
         * pre-deduplication count; checking beforehand can let a
         * too-small post-deduplication array reach interpInit(),
         * which hands it to GSL's own (uncatchable, abort-on-failure)
         * size check instead of this catchable exception.
         */
        void checkMinSize(const gsl_interp_type *interpType)
        {
            if (x_.size() < gsl_interp_type_min_size(interpType))
            {
                std::stringstream ss;
                ss << "Interpolator1D: interpolation type requires minimum size "
                    << gsl_interp_type_min_size(interpType)
                    << ", but only " << x_.size() << " points provided";
                throw std::runtime_error(ss.str());
            }
        }

        /**
         * @brief Remove duplicate values in x and f
         * @details
         * Since x_ is sorted, duplicates always appear in contiguous
         * runs; this compacts each such run down to its first entry,
         * carrying the corresponding f_ entries along, and shrinks x_
         * and f_ to the resulting deduplicated size.
         */
        void cleanDuplicates()
        {
            if (x_.size() < 2) { return; }
            size_t w = 0;
            for (size_t r = 1; r < x_.size(); ++r)
            {
                if (x_[r] == x_[w]) { continue; }
                ++w;
                x_[w] = x_[r];
                for (auto &fi : f_) { fi[w] = fi[r]; }
            }
            const size_t newSize = w + 1;
            x_.resize(newSize);
            for (auto &fi : f_) { fi.resize(newSize); }
        }

        /**
         * @brief Initialize the gsl interpolation machinery
         */
        void interpInit(const gsl_interp_type *interpType)
        {
            for (auto&& [i, d] : std::views::zip(interp_, f_))
            {
                if (x_.size() != d.size())
                {
                    throw std::runtime_error("Interpolator1D: x and f must be of same length");
                }
                if (x_.size() > gsl_interp_type_min_size(interpType))
                {
                    i = gsl_interp_alloc(interpType, x_.size());
                }
                else
                {
                    i = gsl_interp_alloc(gsl_interp_linear, x_.size());
                }
                gsl_interp_init(i, x_.data(), d.data(), x_.size());
            }
            acc_ = gsl_interp_accel_alloc();
        }

        // Internal storage
        std::vector<double> x_;                 /**< Independent variables */
        std::array<std::vector<double>, NF> f_; /**< Dependent variables */
        std::array<gsl_interp *, NF> interp_;   /**< GSL interpolator */
        gsl_interp_accel *acc_;                 /**< Interpolation accelerator */
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // INTERPOLATOR1D_HPP