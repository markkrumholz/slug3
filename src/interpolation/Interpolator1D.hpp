/*
 * @file Interpolator1D.hpp
 * @author Mark Krumholz
 * @brief Provides a class to represent a single 1D interpolator
 * @date 2024-06-27
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
*/

#ifndef INTERPOLATOR1D_HPP
#define INTERPOLATOR1D_HPP

#include "../utils/ThreadVec.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <ranges>
#include <sstream>
#include <stdexcept>
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
        interp_({nullptr})
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

        // NOLINTNEXTLINE(portability-template-virtual-member-function) -- only reachable through unique_ptr<Interpolator1D<NF>> (e.g. specsyn::Specsyn::Isochrone), never a raw new/delete pair the compiler could implicitly instantiate differently; this project also always builds a given binary with a single, consistent compiler, so cross-compiler ABI divergence in the implicit-instantiation rules for this destructor is not a real risk here
        virtual ~Interpolator1D()
        {
            // Free gsl data
            for (auto& i : interp_) { gsl_interp_free(i); }
            for (auto& a : acc_) { gsl_interp_accel_free(a); }
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
         * @brief Get minimum allowed x
         * @return Minimum allowed x
         */
        [[nodiscard]] auto xMin() const { return x_.front(); }

        /**
         * @brief Get maximum allowed x
         * @return Maximum allowed x
         */
        [[nodiscard]] auto xMax() const { return x_.back(); }

        /**
         * @brief Get allowed range in x
         * @return Allowed range in x
         */
        [[nodiscard]] auto xRange() const { return std::make_pair(xMin(), xMax()); }

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
            std::array<double, NF> result = {};
            for (size_t i = 0; i < NF; ++i)
            {
                result[i] = gsl_interp_eval(interp_[i], x_.data(), // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is a loop index bounded by compile-time constant NF
                    f_[i].data(), x, acc_()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is a loop index bounded by compile-time constant NF
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
            return gsl_interp_eval(interp_[idx], x_.data(), f_[idx].data(), x, acc_()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- idx is asserted < NF just above
        }

        /**
         * @brief Integrate the interpolant over [x0, x1]
         * @param x0 Lower limit of integration
         * @param x1 Upper limit of integration
         * @return The integral value(s)
         * @details
         * Return type is a double if NF = 1, and a std::array
         * otherwise -- mirrors operator()(double) exactly, except
         * that each quantity's interpolant is integrated over
         * [x0, x1] (via gsl_interp_eval_integ) rather than evaluated
         * at a single point.
         */
        [[nodiscard]] auto integ(double x0, double x1) const
        {
            assert(x0 >= x_.front() && x0 <= x_.back());
            assert(x1 >= x_.front() && x1 <= x_.back());
            assert(x0 <= x1);
            std::array<double, NF> result = {};
            for (size_t i = 0; i < NF; ++i)
            {
                result[i] = gsl_interp_eval_integ(interp_[i], x_.data(), // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is a loop index bounded by compile-time constant NF
                    f_[i].data(), x0, x1, acc_()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is a loop index bounded by compile-time constant NF
            }
            if constexpr(NF > 1) { return result; }
            else { return result[0]; }
        }

        /**
         * @brief Integrate a single quantity's interpolant over [x0, x1]
         * @param x0 Lower limit of integration
         * @param x1 Upper limit of integration
         * @param idx Quantity to integrate; must be <= NF
         * @return The integral value
        */
        [[nodiscard]] auto integ(double x0, double x1, size_t idx) const
        {
            assert(x0 >= x_.front() && x0 <= x_.back());
            assert(x1 >= x_.front() && x1 <= x_.back());
            assert(x0 <= x1);
            assert(idx < NF);
            return gsl_interp_eval_integ(interp_[idx], x_.data(), f_[idx].data(), x0, x1, acc_()); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- idx is asserted < NF just above
        }

        /**
         * @brief Integrate the product f(x)*g(x) over [x0, x1] for a single field
         * @param g A second Interpolator1D; must have the same NF as this one
         * @param x0 Lower limit of integration
         * @param x1 Upper limit of integration
         * @param idx Field index to integrate; must be < NF
         * @return The integral value as a double
         * @details
         * Evaluates \f$\int_{x_0}^{x_1} f_\mathrm{idx}(x)\, g_\mathrm{idx}(x)\, dx\f$
         * where \f$f(x)\f$ is this Interpolator1D and \f$g(x)\f$ is a second
         * Interpolator1D. Both Interpolator1D objects are required to have the same
         * NF -- this is guaranteed at compile time by the template signature.
         * The integration range is the intersection of [x0, x1] with both
         * interpolators' domains; returns 0 if that intersection is empty.
         */
        [[nodiscard]] auto integ(
            const Interpolator1D& g, double x0, double x1, size_t idx) const -> double
        {
            // NF equality is enforced at compile time: both are Interpolator1D<NF>
            assert(idx < NF);

            const double xLo = std::max({x_.front(), g.x_.front(), x0});
            const double xHi = std::min({x_.back(),  g.x_.back(),  x1});
            if (xLo >= xHi) { return 0.0; }

            // Merged, deduplicated x-grid: xLo, all internal points from either
            // grid that fall strictly inside (xLo, xHi), then xHi
            std::vector<double> xGrid;
            xGrid.reserve(x_.size() + g.x_.size() + 2);
            xGrid.push_back(xLo);
            for (const double xi : x_)   { if (xi > xLo && xi < xHi) { xGrid.push_back(xi); } }
            for (const double xi : g.x_) { if (xi > xLo && xi < xHi) { xGrid.push_back(xi); } }
            xGrid.push_back(xHi);
            std::ranges::sort(xGrid);
            auto toErase = std::ranges::unique(xGrid);
            xGrid.erase(toErase.begin(), toErase.end());

            // Evaluate f_idx(x) * g_idx(x) at each grid point
            std::vector<double> fgGrid(xGrid.size());
            for (size_t j = 0; j < xGrid.size(); ++j)
            {
                fgGrid[j] = (*this)(xGrid[j], idx) * g(xGrid[j], idx);
            }

            // Integrate the product interpolant
            const Interpolator1D<1> fgInterp(xGrid, fgGrid, gsl_interp_linear);
            return fgInterp.integ(xLo, xHi);
        }

        /**
         * @brief Integrate the product f(x)*g(x) over [x0, x1] for all fields
         * @param g A second Interpolator1D; must have the same NF as this one
         * @param x0 Lower limit of integration
         * @param x1 Upper limit of integration
         * @return The integral value(s)
         * @details
         * Evaluates \f$\int_{x_0}^{x_1} f(x)\, g(x)\, dx\f$ where \f$f(x)\f$ is
         * this Interpolator1D and \f$g(x)\f$ is a second Interpolator1D. Both
         * Interpolator1D objects are required to have the same NF -- this is
         * guaranteed at compile time by the template signature.
         * The integration range is the intersection of [x0, x1] with both
         * interpolators' domains; returns 0 (or an array of zeros) if that
         * intersection is empty.
         * Return type mirrors integ(double, double): a double if NF = 1,
         * a std::array<double, NF> otherwise.
         */
        [[nodiscard]] auto integ(
            const Interpolator1D& g, double x0, double x1) const
        {
            // NF equality is enforced at compile time: both are Interpolator1D<NF>
            const double xLo = std::max({x_.front(), g.x_.front(), x0});
            const double xHi = std::min({x_.back(),  g.x_.back(),  x1});
            if (xLo >= xHi)
            {
                if constexpr (NF > 1) { return std::array<double, NF>{}; }
                else { return 0.0; }
            }

            // Merged, deduplicated x-grid: xLo, all internal points from either
            // grid that fall strictly inside (xLo, xHi), then xHi
            std::vector<double> xGrid;
            xGrid.reserve(x_.size() + g.x_.size() + 2);
            xGrid.push_back(xLo);
            for (const double xi : x_)   { if (xi > xLo && xi < xHi) { xGrid.push_back(xi); } }
            for (const double xi : g.x_) { if (xi > xLo && xi < xHi) { xGrid.push_back(xi); } }
            xGrid.push_back(xHi);
            std::ranges::sort(xGrid);
            auto toErase = std::ranges::unique(xGrid);
            xGrid.erase(toErase.begin(), toErase.end());

            // Evaluate f_i(x) * g_i(x) at each grid point for every field
            std::array<std::vector<double>, NF> fgGrid;
            for (auto& v : fgGrid) { v.resize(xGrid.size()); }
            for (size_t j = 0; j < xGrid.size(); ++j)
            {
                for (size_t i = 0; i < NF; ++i)
                {
                    fgGrid[i][j] = (*this)(xGrid[j], i) * g(xGrid[j], i); // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- i is bounded by compile-time constant NF
                }
            }

            // Integrate the product interpolant
            const Interpolator1D<NF> fgInterp(xGrid, fgGrid, gsl_interp_linear);
            return fgInterp.integ(xLo, xHi);
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
            for (auto& a : acc_) { a = gsl_interp_accel_alloc(); }
        }

        // Internal storage
        std::vector<double> x_;                 /**< Independent variables */
        std::array<std::vector<double>, NF> f_; /**< Dependent variables */
        std::array<gsl_interp *, NF> interp_;   /**< GSL interpolator */
        utils::ThreadVec<gsl_interp_accel *> acc_; /**< Interpolation accelerator, one per thread */
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // INTERPOLATOR1D_HPP