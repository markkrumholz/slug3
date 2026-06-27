/**
 * @file Mesh2DInterpolator.hpp
 * @author Mark Krumholz
 * @brief Machinery to do interpolation on a semi-tensor mesh
 * @date 2024-06-19
 * @details
 * A class to inteprolate on a 2D tensor mesh
 */

#ifndef MESH2DINTERPOLATOR_HPP
#define MESH2DINTERPOLATOR_HPP

#include "Interpolator1D.hpp"
#include "Mesh2DGrid.hpp"
#include <gsl/gsl_interp.h>
#include <mdspan>
#include <utility>
#include <vector>

// Disable linting for array bounds checking in this
// file, since the overhead associated with enforcing
// such checks severely interferes with performance
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)


namespace interp
{

    /**
     * @class Mesh2DInterpolator
     * @tparam nF Number of quantities to inteprolate
     * @brief A class to perform interpolation on a 2D semi-tensor grid
     * @details
     * This class performs interpolation on a grid where the points in one
     * direction (the y direction) are arranged in a tensor grid
     * (i.e., in an Nx x Ny grid, there is one y value for each
     * of the Nx rows), but not in the other direction (i.e., the
     * values of x can be different from one row of y values to
     * to the next). The number of quantities to be interpolated is
     * determined by the template parameter nF.
     */
    template <size_t nF = 1>
    class Mesh2DInterpolator
    {
    public:

        // Shorten array types
        using Array3D = std::mdspan<double, std::dextents<size_t, 3>>;
        using Array2D = std::mdspan<double, std::dextents<size_t, 2>>;
        using Array1D = std::mdspan<double, std::dextents<size_t, 1>>;

        // Constructors and destructor
        /**
         * @brief Construct a Mesh2DInterpolator
         * @param x A 2d array giving the x coordinates of the mesh points
         * @param y A 1d array giving the y coordinates of the mesh points
         * @param f A 3d array giving the value of the function to be interpolated
         * @param interpType The type of interpolation to use
         * @param monotonic Enforce that interpolation is monotonicity-preserving
         * @details
         * If enforceMonotonic is set to true, the interpolation type
         * will always be gsl_interp_linear regardless of the value of interpType.
         */
        Mesh2DInterpolator(
            const Array2D& x,
            const Array1D& y,
            const Array3D& f,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) :
        monotonic_(monotonic),
        interpType_(monotonic ? gsl_interp_linear : interpType),
        mesh_(x, y, true)
        {
            // Safety check
            if (f.extent(0) != x.extent(0) || 
                f.extent(1) != x.extent(1) ||
                f.extent(2) != nF)
            {
                throw std::runtime_error(
                    "Mesh2DInterpolator: x and f must have same leading "
                    "two dimensions, and the third dimension of f must "
                    "match the number of interpolated quantities");
            }

            // Build rib interpolators
            for (size_t j = 0; j < ny(); ++j)
            {
                std::vector<double> xRib(nx());
                std::array<std::vector<double>, nF> fRib;
                for (size_t k = 0; k < nF; k++) fRib[k].resize(nx());
                for (size_t i = 0; i < nx(); ++i)
                {
                    xRib[i] = x[i,j];
                    for (size_t k = 0; k < nF; ++k) { fRib[k][i] = f[i,j,k]; }
                }
                auto ri = std::make_unique<Interpolator1D<nF>>(xRib, fRib, interpType_);
                ribInterp.push_back(std::move(ri));
            }

            // Build the spine interpolators
            for (size_t i = 0; i < nx(); ++i)
            {
                std::vector<double> xSpine(ny());
                std::array<std::vector<double>, nF> fSpine;
                for (size_t k = 0; k < nF; k++) fSpine[k].resize(ny());
                for (size_t j = 0; j < ny(); ++j)
                {
                    if (!monotonic_) { xSpine[j] = s()[i,j]; }
                    else { xSpine[j] = y[j]; }
                    for (size_t k = 0; k < nF; ++k) { fSpine[k][j] = f[i,j,k]; }
                }
                auto si = std::make_unique<Interpolator1D<nF>>(xSpine, fSpine, interpType_);
                spineInterp.push_back(std::move(si));
            }
        }

        /**
         * @brief Construct a Mesh2DInterpolator
         * @param x A 2d array giving the x coordinates of the mesh points
         * @param y A 1d array giving the y coordinates of the mesh points
         * @param f A 2d array giving the value of the function to be interpolated
         * @param interpType The type of interpolation to use
         * @param monotonic Enforce that interpolation is monotonicity-preserving
         * @details
         * This constructor is valid only if nF == 1, and it just wraps a 2D
         * array in a 3D view before invoking the alternative form of the constructor.
         */
        Mesh2DInterpolator(
            const Array2D& x,
            const Array1D& y,
            const Array2D& f,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) requires (nF == 1) :
        Mesh2DInterpolator(x, y, 
            std::mdspan(f.data_handle(), f.extent(0), f.extent(1), 1),
            interpType,
            monotonic)
        { }

        virtual ~Mesh2DInterpolator() = default;
        Mesh2DInterpolator(const Mesh2DInterpolator&) = default;
        Mesh2DInterpolator(Mesh2DInterpolator&&) = default;
        auto operator=(const Mesh2DInterpolator&) -> Mesh2DInterpolator& = delete;
        auto operator=(Mesh2DInterpolator&&) -> Mesh2DInterpolator& = delete;

        // Observers
        /**
         * @brief Return size of Mesh2DInterpolator
         * @returns Size of mesh
         */
        auto size() const { return mesh_.size(); }

        /**
         * @brief Return x dimension of Mesh2DInterpolator
         * @returns Number of mesh points in x direction
         */
        auto nx() const { return mesh_.nx(); }

        /**
         * @brief Return x dimension of Mesh2DInterpolator
         * @returns Number of mesh points in x direction
         */
        auto ny() const { return mesh_.ny(); }

        /**
         * @brief Return if the mesh is convex
         * @returns True if the mesh is convex
         */
        auto convex() const { return mesh_.convex(); }

        /**
         * @brief Return a const reference to the x array
         * @returns A const reference to the x array
         */
        const auto& x() const { return mesh_.xData(); }

        /**
         * @brief Return a const reference to the y array
         * @returns A const reference to the y array
         */
        const auto& y() const { return mesh_.yData(); }

        /**
         * @brief Return a const reference to the s array
         * @returns A const reference to the s array
         */
        const auto& s() const { return mesh_.sData(); }
        
    
    private:

        // Control parameters
        const gsl_interp_type *interpType_; /**< Type of interpolation used */
        const bool monotonic_;        /**< True if interpolation is monotonicity-preserving */

        // Internal storage
        Mesh2DGrid mesh_;            /**< Object to represent the (x,y) mesh */
        std::vector<
            std::unique_ptr<Interpolator1D<nF>>
        > ribInterp;                 /**< Interpolators for ribs */
        std::vector<
            std::unique_ptr<Interpolator1D<nF>>
        > spineInterp;               /**< Interpolators for spines */
    };

} // namespace interp

// NOLINEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // MESH2DINTERPOLATOR_HPP