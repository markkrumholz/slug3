/*
 * @file Interpolator1D.hpp
 * @author Mark Krumholz
 * @brief Provides a class to represent a single 1D interpolator
 * @date 2024-06-27
*/

#ifndef RSINTERPOLATOR_HPP
#define RSINTERPOLATOR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <gsl/gsl_interp.h>
#include <ranges>
#include <sstream>
#include <vector>

namespace interp
{

    /**
     * @class Interpolator1D
     * @tparam nF Number of quantities to interpolate
     * @brief A class to handle 1D interpolation
     */
    template <size_t nF = 1>
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
            const std::array<std::vector<double>, nF>& f,
            const gsl_interp_type* interpType = gsl_interp_steffen
        ) : 
        x_(x),
        f_(f),
        interp_({nullptr}),
        acc_(nullptr)
        {
            safetyCheck(interpType);
            cleanDuplicates();
            interpInit(interpType);
        }

        Interpolator1D(
            const std::vector<double>& x,
            const std::vector<double>& f,
            const gsl_interp_type* interpType = gsl_interp_steffen
        ) requires (nF == 1)
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

        // Standard move constructors
        Interpolator1D(Interpolator1D&&) = default;
        auto operator=(Interpolator1D&&) -> Interpolator1D& = default;

        // Disallow copy constructors, since these would make
        // copies of gsl opaque objects that we can't manage with
        // smart pointers, resulting in potential memory corruption
        // when two Interpolator1D objects pointing to the same gsl
        // opaque objects pass out of scope and try to free the same
        // memory
        Interpolator1D(const Interpolator1D&) = delete;
        auto operator=(const Interpolator1D&) -> Interpolator1D& = delete;

        /**
         * @brief Interpolate to a given point
         * @param x The point to which to interpolate
         * @return The interpolated value(s)
         * @details
         * Return type is a double if nF = 1, and a std::array otherwise.
         */
        [[nodiscard]] auto operator()(double x) const
        {
            assert(x >= x_.front() && x <= x_.back());
            std::array<double, nF> result;
            for (size_t i = 0; i < nF; ++i)
            {
                result[i] = gsl_interp_eval(interp_[i], x_.data(), 
                    f_[i].data(), x, acc_);
            }
            if constexpr(nF > 1) { return result; }
            else { return result[0]; }
        }

        /**
         * @brief Interpolate a single quantity to a given point
         * @param x The point to which to interpolate
         * @param idx Quantity to interpolate; must be <= nF
         * @return The interpolated value
        */
        [[nodiscard]] auto operator()(double x, size_t idx) const
        {
            assert(x >= x_.front() && x <= x_.back());
            assert(idx < nF);
            return gsl_interp_eval(interp_[idx], x_.data(), f_[idx].data(), x, acc_);
        }

    private:

        /**
         * @brief Run safety check on inputs
         */
        void safetyCheck(const gsl_interp_type *interpType)
        {
            static_assert(nF > 0, "Interpolator1D: number of quantities to interpolate must be >= 1");
            if (!std::ranges::is_sorted(x_))
            {
                throw std::runtime_error("Interpolator1D: x must be strictly increasing");
            }
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
         */
        void cleanDuplicates()
        {
            // Loop through x values
            size_t i = 0;
            while (i < x_.size() - 2)
            {
                // Find sequences of identical values
                size_t count = 0;
                while (x_[i] == x_[i+count+1]) { ++count; }

                // Move values to remove duplicates
                if (count > 0)
                {
                    for (size_t j = i; j < x_.size() - count; ++j)
                    {
                        x_[j] = x_[j+count];
                        for (auto &fi : f_) { fi[j] = fi[j+count]; }
                    }
                }

                // Move to next element
                i += count + 1;
            }
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
                i = gsl_interp_alloc(interpType, x_.size());
                gsl_interp_init(i, x_.data(), d.data(), x_.size());
            }
            acc_ = gsl_interp_accel_alloc();
        }

        // Internal storage
        std::vector<double> x_;                 /**< Independent variables */
        std::array<std::vector<double>, nF> f_; /**< Dependent variables */
        std::array<gsl_interp *, nF> interp_;   /**< GSL interpolator */
        gsl_interp_accel *acc_;                 /**< Interpolation accelerator */
    };

} // namespace interp

#endif // INTERPOLATOR1D_HPP