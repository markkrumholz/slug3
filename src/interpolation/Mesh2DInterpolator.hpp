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
#include <array>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <mdspan>
#include <memory>
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
     * @class Mesh2DInterpolator
     * @tparam NF Number of quantities to inteprolate
     * @brief A class to perform interpolation on a 2D semi-tensor grid
     * @details
     * This class performs interpolation on a grid where the points in one
     * direction (the y direction) are arranged in a tensor grid
     * (i.e., in an Nx x Ny grid, there is one y value for each
     * of the Nx rows), but not in the other direction (i.e., the
     * values of x can be different from one row of y values to
     * to the next). The number of quantities to be interpolated is
     * determined by the template parameter NF.
     */
    template <size_t NF = 1>
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
        interpType_(monotonic ? gsl_interp_linear : interpType),
        monotonic_(monotonic),
        mesh_(x, y, true)
        {
            // Safety check
            if (f.extent(0) != x.extent(0) || 
                f.extent(1) != x.extent(1) ||
                f.extent(2) != NF)
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
                std::array<std::vector<double>, NF> fRib;
                for (size_t k = 0; k < NF; k++) { fRib[k].resize(nx()); }
                for (size_t i = 0; i < nx(); ++i)
                {
                    xRib[i] = x[i,j];
                    for (size_t k = 0; k < NF; ++k) { fRib[k][i] = f[i,j,k]; }
                }
                auto ri = std::make_unique<Interpolator1D<NF>>(xRib, fRib, interpType_);
                ribInterp_.push_back(std::move(ri));
            }

            // Build the spine interpolators
            for (size_t i = 0; i < nx(); ++i)
            {
                std::vector<double> xSpine(ny());
                std::array<std::vector<double>, NF> fSpine;
                for (size_t k = 0; k < NF; k++) { fSpine[k].resize(ny()); }
                for (size_t j = 0; j < ny(); ++j)
                {
                    if (!monotonic_) { xSpine[j] = s()[i,j]; }
                    else { xSpine[j] = y[j]; }
                    for (size_t k = 0; k < NF; ++k) { fSpine[k][j] = f[i,j,k]; }
                }
                auto si = std::make_unique<Interpolator1D<NF>>(xSpine, fSpine, interpType_);
                spineInterp_.push_back(std::move(si));
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
         * This constructor is valid only if NF == 1, and it just wraps a 2D
         * array in a 3D view before invoking the alternative form of the constructor.
         */
        Mesh2DInterpolator(
            const Array2D& x,
            const Array1D& y,
            const Array2D& f,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) requires (NF == 1) :
        Mesh2DInterpolator(x, y, 
            std::mdspan(f.data_handle(), f.extent(0), f.extent(1), 1),
            interpType,
            monotonic)
        { }

        virtual ~Mesh2DInterpolator() = default;

        // Disallow copy constructor due to the unique_ptrs we use
        // internally, but allow move constructor
        Mesh2DInterpolator(const Mesh2DInterpolator&) = delete;
        Mesh2DInterpolator(Mesh2DInterpolator&&) = default;
        auto operator=(const Mesh2DInterpolator&) -> Mesh2DInterpolator& = delete;
        auto operator=(Mesh2DInterpolator&&) noexcept -> Mesh2DInterpolator& = default;

        // Observers
        /**
         * @brief Return size of Mesh2DInterpolator
         * @returns Size of mesh
         */
        [[nodiscard]] auto size() const { return mesh_.size(); }

        /**
         * @brief Return x dimension of Mesh2DInterpolator
         * @returns Number of mesh points in x direction
         */
        [[nodiscard]] auto nx() const { return mesh_.nx(); }

        /**
         * @brief Return x dimension of Mesh2DInterpolator
         * @returns Number of mesh points in x direction
         */
        [[nodiscard]] auto ny() const { return mesh_.ny(); }

        /**
         * @brief Return if the mesh is convex
         * @returns True if the mesh is convex
         */
        [[nodiscard]] auto convex() const { return mesh_.convex(); }

        /**
         * @brief Return a const reference to the x array
         * @returns A const reference to the x array
         */
        [[nodiscard]] auto x() const -> const auto& { return mesh_.xData(); }

        /**
         * @brief Return a const reference to the y array
         * @returns A const reference to the y array
         */
        [[nodiscard]] auto y() const -> const auto& { return mesh_.yData(); }

        /**
         * @brief Return a const reference to the s array
         * @returns A const reference to the s array
         */
        [[nodiscard]] auto s() const -> const auto& { return mesh_.sData(); }

        /**
         * @brief Minimum x value in mesh
         * @returns Minimum x value in mesh
         */
        [[nodiscard]] auto xMin() const { return mesh_.xMin(); }

        /**
         * @brief Find minimum x value at given y
         * @returns Minimum x value at given y
         */
        [[nodiscard]] auto xMin(const double y) const { return mesh_.xMin(y); }

        /**
         * @brief Maximum x value in mesh
         * @returns Maximum x value in mesh
         */
        [[nodiscard]] auto xMax() const { return mesh_.xMax(); }

        /**
         * @brief Find maximum x value at given y
         * @returns Maximum x value at given y
         */
        [[nodiscard]] auto xMax(const double y) const { return mesh_.xMax(y); }

        /**
         * @brief Minimum y value in mesh
         * @returns Minimum y value in mesh
         */
       [[nodiscard]] auto yMin() const { return mesh_.yMin(); }

        /**
         * @brief Maximum y value in mesh
         * @returns Maximum y value in mesh
         */
        [[nodiscard]] auto yMax() const { return mesh_.yMax(); }

        /**
         * @brief Find minimum and maximum y position at a given x in the mesh
         * @param x x position
         * @returns Minimum and maximum y values
         * @details
         * The return value for this function is a vectors of pairs;
         * each pair contains the minimum and maximum y values where a line
         * segment at a given x value intersects the mesh. For a convex
         * mesh there is at most one minimum and one maximum, so this vector
         * will be of length 0 (if the mesh is missed entirely) or 1, but
         * for a non-convex mesh a line at fixed x can intersect the edge
         * an arbitrary number of times, so the vector can be longer. 
         */
        [[nodiscard]] auto yLim(double x) const { return mesh_.yLim(x); }


        // Interpolators
        /**
         * @brief Function to create a 1D interpolator at constant x
         * @param x x coordinate
         * @returns A vector of Interpolator1D's that interpolates in y at the given x
         * @details
         * This function returns a vector of Interpolator1D objeects rather than a
         * single one because, for a non-convex mesh, the intersection of a line of
         * constant x with the mesh may consist of multiple disconnected sgments. In
         * such cases, the vector will contain one Interpolator1D for each segment.
         * If the value of x provided is outside the mesh entirely, so there are no
         * intersections, the returned vector will be empty. A segment consisting of
         * a single point (e.g. a line of constant x that is tangent to the mesh,
         * touching it at only one mass) carries no interpolable information and is
         * likewise dropped, exactly as if that portion of the mesh had been missed.
         */
        [[nodiscard]] auto interpConstX(double x) const -> 
        std::vector<std::unique_ptr<Interpolator1D<NF>>>
        {
            // Grab list of intersection points
            const auto intersect = mesh_.xIntersect(x);

            // Output holders
            std::vector<std::unique_ptr<Interpolator1D<NF>>> result;
            std::vector<double> y;
            std::array<std::vector<double>, NF> f;

            // Build an interpolator from the accumulated segment and
            // push it onto result, unless the segment has too few
            // points to interpolate at all (e.g. a single isolated
            // tangent point, where the line of constant x touches the
            // mesh at only one mass); such a segment carries no
            // interpolable information, so it is silently dropped
            // rather than attempted, exactly as if that portion of
            // the mesh had been missed entirely
            const auto flushSegment = [&]() -> void
            {
                if (y.size() < 2) { return; }
                auto newInterp =
                    std::make_unique<Interpolator1D<NF>>(y, f, interpType_);
                result.push_back(std::move(newInterp));
            };

            // Loop over intersection points
            for (const auto& pt : intersect)
            {

                // Add y to accumulator
                y.push_back(pt.y);

                // Get point at which to evaluate spline
                const double s =
                    (pt.t == Mesh2DGrid::IntersectionType::spine &&
                    monotonic_) ? pt.y : pt.xs;

                // Evaluate spline and save
                const auto fInterp = pt.t == Mesh2DGrid::IntersectionType::rib ?
                    (*(ribInterp_[pt.idx]))(s) :
                    (*(spineInterp_[pt.idx]))(s);
                if constexpr (NF == 1) { f[0].push_back(fInterp); }
                else
                {
                    for (size_t k = 0; k < NF; k++) { f[k].push_back(fInterp[k]); }
                }

                // If this point is a mesh exit, construct new
                // interpolator here and push onto output list, then
                // empty the accumulators
                if (pt.meshExit)
                {
                    flushSegment();
                    y.clear();
                    for (auto &fi : f) { fi.clear(); }
                }
            }

            // If any points remain in accumulator, create an interpolator from them
            if (!y.empty())
            {
                flushSegment();
            }

            // Return result
            return result;
        }
        
        /**
         * @brief Function to create a 1D interpolator at constant y
         * @param y y coordinate
         * @returns An Interpolator1D that interpolates in x and the given y
         * @details
         * If the value of y provided is outside the mesh entirely (i.e., below
         * yMin or above yMax), this routine returns a null pointer.
         */
        [[nodiscard]] auto interpConstY(double y) const -> 
        std::unique_ptr<Interpolator1D<NF>>
        {
            // Grab list of intersection points
            auto intersect = mesh_.yIntersect(y);

            // Catch case where mesh is missed entirely, so we just return
            // a null pointer
            if (intersect.empty()) { return nullptr; }

            // Accumulators
            std::vector<double> x(intersect.size());
            std::array<std::vector<double>, NF> f;
            for (auto& fi : f) { fi.resize(x.size()); }

            // Loop over intersection points, evaluating function at each
            for (size_t i = 0; i < x.size(); i++)
            {
                const auto& pt = intersect[i];
                x[i] = pt.x;
                auto fInterp = (*(spineInterp_[pt.idx]))(pt.s);
                if constexpr (NF == 1) { f[0][i] = fInterp; }
                else
                { 
                    for (size_t k = 0; k < NF; k++) { f[k][i] = fInterp[k]; }
                }                
            }

            // Build final interpolation object to return
            return std::make_unique<Interpolator1D<NF>>(x, f, interpType_);
        }

        
    
    private:

        // Control parameters
        const gsl_interp_type *interpType_; /**< Type of interpolation used */
        bool monotonic_;                    /**< True if interpolation is monotonicity-preserving */

        // Internal storage; note that we handle the rib and spine
        // interpolators as vectors of unique_ptr rather than vectors
        // of Interpolator1D because we need to be sure to invoke the
        // destructor of Interpolator1D objects once only to avoid memory
        // issue with gsl opaque objects
        Mesh2DGrid mesh_;            /**< Object to represent the (x,y) mesh */
        std::vector<
            std::unique_ptr<Interpolator1D<NF>>
        > ribInterp_;                 /**< Interpolators for ribs */
        std::vector<
            std::unique_ptr<Interpolator1D<NF>>
        > spineInterp_;               /**< Interpolators for spines */
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // MESH2DINTERPOLATOR_HPP