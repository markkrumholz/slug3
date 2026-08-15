/**
 * @file GKIntegrator.hpp
 * @author Mark Krumholz
 * @brief Adaptive Gauss-Kronrod integration of vector-valued integrands
 * @date 2026-08-15
 */

#ifndef GKINTEGRATOR_HPP
#define GKINTEGRATOR_HPP

#include <cstddef>
#include <utility>

namespace utils
{

    /**
     * @class GKIntegrator
     * @brief Adaptively integrate a vector-valued function of one variable
     * @tparam F Type of the callable integrand: must be invocable as
     *   f(x, ...), where x is the (scalar) integration variable and
     *   ... is any number of additional arguments forwarded from
     *   integrate(); must return a contiguous container of nInt
     *   doubles (e.g. std::vector<double> or std::array<double, nInt>)
     *   -- the value of the integrand at x.
     * @details
     * An adaptation of the GSL's gsl_integration_qag() to vector-valued
     * integrands, using C++ templating rather than gsl_function's
     * pointer-to-void mechanism to pass any extra arguments f itself
     * requires beyond the integration variable.
     */
    template <class F>
    class GKIntegrator
    {
    public:

        /**
         * @brief Construct a GKIntegrator
         * @param f The integrand; see this class's own @tparam F
         * @param nInt Number of quantities f returns per point
         * @param maxEval Maximum number of integrand evaluations
         *   (0 = unlimited)
         * @param absTol Required absolute error
         * @param relTol Required relative error
         */
        GKIntegrator(
            F f,
            std::size_t nInt,
            std::size_t maxEval = 0,
            double absTol = 0.0,
            double relTol = 1e-6
        ) :
            f_(std::move(f)),
            nInt_(nInt),
            maxEval_(maxEval),
            absTol_(absTol),
            relTol_(relTol)
        { }

    private:

        F f_;                  /**< The integrand */
        std::size_t nInt_;     /**< Number of quantities f returns per point */
        std::size_t maxEval_;  /**< Maximum number of integrand evaluations */
        double absTol_;        /**< Required absolute error */
        double relTol_;        /**< Required relative error */
    };

} // namespace utils

#endif // GKINTEGRATOR_HPP
