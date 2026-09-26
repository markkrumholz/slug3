/**
 * @file Mesh2DInterpolator.hpp
 * @author Mark Krumholz
 * @brief Machinery to do interpolation on a semi-tensor mesh
 * @date 2024-06-19
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 * @details
 * A class to inteprolate on a 2D tensor mesh
 */

#ifndef MESH2DINTERPOLATOR_HPP
#define MESH2DINTERPOLATOR_HPP

#include "Interpolator1D.hpp"
#include "Mesh2DGrid.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <mdspan> // NOLINT(misc-include-cleaner)
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
     * @class LazyValueSource
     * @brief Supplies the function values at the points of a Mesh2DInterpolator's mesh on demand
     * @tparam NF Number of quantities interpolated
     * @details
     * Lets a Mesh2DInterpolator represent a mesh whose function
     * values are never stored in full -- e.g. a slice of a
     * Mesh3DInterpolator at an arbitrary z, whose values are
     * themselves interpolated in z only at the points a query
     * actually needs (see Mesh3DInterpolator::lazySliceConstZ()).
     * Implementations must be safe to call concurrently from multiple
     * threads.
     */
    template <size_t NF>
    class LazyValueSource
    {
    public:
        LazyValueSource() = default;
        virtual ~LazyValueSource() = default;
        LazyValueSource(const LazyValueSource&) = default;
        LazyValueSource(LazyValueSource&&) = default;
        auto operator=(const LazyValueSource&) -> LazyValueSource& = default;
        auto operator=(LazyValueSource&&) -> LazyValueSource& = default;

        /**
         * @brief Return the function values at one mesh point
         * @param i Index of the point in the x direction
         * @param j Index of the point in the y direction
         * @returns The NF function values at mesh point (i,j)
         */
        [[nodiscard]] virtual auto value(size_t i, size_t j) const -> std::array<double, NF> = 0;
    };

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
        using Array3D = std::mdspan<double, std::dextents<size_t, 3>>; // NOLINT(misc-include-cleaner)
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
                for (size_t k = 0; k < NF; k++) { fRib[k].resize(nx()); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
                for (size_t i = 0; i < nx(); ++i)
                {
                    xRib[i] = x[i,j];
                    for (size_t k = 0; k < NF; ++k) { fRib[k][i] = f[i,j,k]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
                }
                auto ri = std::make_unique<Interpolator1D<NF>>(xRib, fRib, interpType_);
                ribInterp_.push_back(std::move(ri));
            }

            // Build the spine interpolators
            for (size_t i = 0; i < nx(); ++i)
            {
                std::vector<double> xSpine(ny());
                std::array<std::vector<double>, NF> fSpine;
                for (size_t k = 0; k < NF; k++) { fSpine[k].resize(ny()); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
                for (size_t j = 0; j < ny(); ++j)
                {
                    if (!monotonic_) { xSpine[j] = s()[i,j]; }
                    else { xSpine[j] = y[j]; }
                    for (size_t k = 0; k < NF; ++k) { fSpine[k][j] = f[i,j,k]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
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
            std::mdspan(f.data_handle(), f.extent(0), f.extent(1), 1), // NOLINT(misc-include-cleaner)
            interpType,
            monotonic)
        { }

        /**
         * @brief Construct a Mesh2DInterpolator whose mesh coordinates and function values are evaluated lazily
         * @param x The x coordinates of the mesh points (see
         *   Mesh2DGrid::XView); not copied
         * @param y A 1d array giving the y coordinates of the mesh points
         * @param source Supplies the function value at each mesh point
         *   on demand
         * @param interpType The type of interpolation to use
         * @param monotonic Enforce that interpolation is monotonicity-preserving
         * @details
         * Unlike the other constructors, this builds no rib or spine
         * interpolators, and stores no function values: construction
         * costs O(Ny) (see Mesh2DGrid's own XView constructor), and
         * each query instead evaluates source only at the few mesh
         * points it needs, building a small local interpolator over
         * just those points (see evalRib()/evalSpine()). Whatever
         * storage x refers to, and source, must outlive this object.
         * Query results agree with those of an eagerly-constructed
         * Mesh2DInterpolator over the same values to within
         * floating-point rounding. That requires a local
         * interpolation scheme, whose value within a cell depends only
         * on a few neighbouring points, so interpType must be
         * gsl_interp_linear, gsl_interp_steffen, or gsl_interp_akima;
         * any other type throws std::runtime_error.
         */
        Mesh2DInterpolator(
            const Mesh2DGrid::XView& x,
            const Array1D& y,
            std::shared_ptr<const LazyValueSource<NF>> source,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) :
        interpType_(monotonic ? gsl_interp_linear : interpType),
        monotonic_(monotonic),
        mesh_(x, y),
        lazySource_(std::move(source))
        {
            if (interpType_ != gsl_interp_linear &&
                interpType_ != gsl_interp_steffen &&
                interpType_ != gsl_interp_akima)
            {
                throw std::runtime_error(
                    "Mesh2DInterpolator: lazily-evaluated meshes support "
                    "only local interpolation types (linear, steffen, "
                    "akima)");
            }
        }

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

        /**
         * @brief Find y and dy/dx at every point where a line of constant x intersects one mesh edge
         * @param x x position
         * @param leftEdge True to find intersections with the mesh's
         *   left edge (column 0), false for its right edge (column
         *   nx()-1)
         * @returns One (y, dy/dx) pair per intersection with the
         *   requested edge, in ascending y order
         * @details
         * Thin wrapper around Mesh2DGrid::yEdgeSlope() -- see its own
         * comment.
         */
        [[nodiscard]] auto yEdgeSlope(double x, bool leftEdge) const
        { return mesh_.yEdgeSlope(x, leftEdge); }

        /**
         * @brief Check whether a point is contained in the mesh
         * @param x x position
         * @param y y position
         * @returns True if the point is within the mesh
         */
        [[nodiscard]] auto contains(const double x, const double y) const
        { return mesh_.contains(x, y); }


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
                    evalRib(pt.idx, s) : evalSpine(pt.idx, s);
                if constexpr (NF == 1) { f[0].push_back(fInterp); }
                else
                {
                    for (size_t k = 0; k < NF; k++) { f[k].push_back(fInterp[k]); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
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
                auto fInterp = evalSpine(pt.idx, pt.s);
                if constexpr (NF == 1) { f[0][i] = fInterp; }
                else
                { 
                    for (size_t k = 0; k < NF; k++) { f[k][i] = fInterp[k]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
                }
            }

            // Build final interpolation object to return
            return std::make_unique<Interpolator1D<NF>>(x, f, interpType_);
        }

        /**
         * @brief Interpolate to a single point
         * @param x x coordinate
         * @param y y coordinate
         * @param linear If true, force linear interpolation instead of
         * this object's default interpolation type
         * @returns The interpolated value(s)
         * @details
         * Return type is a double if NF = 1, and a std::array otherwise.
         * This routine builds a local Interpolator1D from just enough
         * points around (x, y) to support the requested interpolation
         * type, then evaluates it at y; unlike interpConstX, it does
         * not need to handle multiple disconnected segments, since
         * xIntersectN only ever searches out from a single starting
         * point. If xIntersectN cannot find enough points to support
         * the requested type but still finds at least 2, this falls
         * back to linear interpolation instead, mirroring interpConstX's
         * handling of undersized segments. If it finds only a single
         * point, (x, y) must be an isolated tangent point of the mesh,
         * so there is nothing to interpolate and that point's own value
         * is returned directly.
         */
        [[nodiscard]] auto operator()(double x, double y, bool linear = false) const
        {
            // Number of points needed to support the requested
            // interpolation type
            const size_t n = linear ?
                gsl_interp_type_min_size(gsl_interp_linear) :
                gsl_interp_type_min_size(interpType_);

            // Grab list of intersection points
            const auto intersect = mesh_.xIntersectN(x, y, n);

            // A single point means (x, y) is an isolated tangent point
            // of the mesh; there is nothing to interpolate, so just
            // evaluate the appropriate rib or spine interpolator there
            // directly and return
            if (intersect.size() == 1)
            {
                const auto& pt = intersect.front();
                const double s =
                    (pt.t == Mesh2DGrid::IntersectionType::spine &&
                    monotonic_) ? pt.y : pt.xs;
                return pt.t == Mesh2DGrid::IntersectionType::rib ?
                    evalRib(pt.idx, s) : evalSpine(pt.idx, s);
            }

            // Accumulators
            std::vector<double> yPts;
            std::array<std::vector<double>, NF> f;

            // Loop over intersection points
            for (const auto& pt : intersect)
            {

                // Add y to accumulator
                yPts.push_back(pt.y);

                // Get point at which to evaluate spline
                const double s =
                    (pt.t == Mesh2DGrid::IntersectionType::spine &&
                    monotonic_) ? pt.y : pt.xs;

                // Evaluate spline and save
                const auto fInterp = pt.t == Mesh2DGrid::IntersectionType::rib ?
                    evalRib(pt.idx, s) : evalSpine(pt.idx, s);
                if constexpr (NF == 1) { f[0].push_back(fInterp); }
                else
                {
                    for (size_t k = 0; k < NF; k++) { f[k].push_back(fInterp[k]); } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- k is a loop index bounded by compile-time constant NF
                }
            }

            // If fewer points were found than the requested type
            // needs, fall back to linear interpolation
            const gsl_interp_type* useType = interpType_;
            if (linear || intersect.size() < gsl_interp_type_min_size(interpType_))
            {
                useType = gsl_interp_linear;
            }

            // Build interpolator and evaluate at y
            const Interpolator1D<NF> pointInterp(yPts, f, useType);
            return pointInterp(y);
        }



    private:

        /**
         * @brief Number of mesh points on either side of a query's own cell used by a lazily-built local interpolator
         * @details
         * The steffen interpolant within a cell [c, c+1] depends only on
         * the points c-1 through c+2, and the akima one on c-2 through
         * c+3, so a local interpolator including stencilHalfWidth
         * distinct coordinates on each side of the cell (counting c
         * itself on the lower side) reproduces the full rib/spine
         * interpolator there.
         */
        static constexpr size_t stencilHalfWidth = 3;

        /**
         * @brief Evaluate a local interpolator over part of one rib or spine
         * @param coord Coordinate of each point along the rib/spine
         *   (callable taking a point index)
         * @param value Function values at each point (callable taking a
         *   point index)
         * @param n Number of points along the rib/spine
         * @param q Query coordinate
         * @returns The interpolated value(s) at q
         * @details
         * Finds the cell [c, c+1] containing q by binary search over
         * coord (coordinates are non-decreasing along a rib or spine),
         * then builds an Interpolator1D over the points from the one
         * giving stencilHalfWidth distinct coordinates in [.., c] to
         * the one giving stencilHalfWidth distinct coordinates in
         * [c+1, ..] (clipped to [0, n-1]), and evaluates it at q. The
         * lower end is extended to the start of any run of duplicate
         * coordinates it falls in, since Interpolator1D keeps the first
         * point of such a run, so the local interpolator sees the same
         * deduplicated points as the full rib/spine one.
         */
        [[nodiscard]] auto evalLocal(const auto& coord, const auto& value,
            const size_t n, const double q) const
        {
            size_t lo = 0;
            size_t hi = n - 1;
            while (hi - lo > 1)
            {
                const size_t mid = (lo + hi) / 2;
                if (coord(mid) <= q) { lo = mid; }
                else { hi = mid; }
            }
            size_t first = lo;
            for (size_t nDistinct = 1; first > 0 && nDistinct < stencilHalfWidth; --first)
            {
                if (coord(first - 1) != coord(first)) { ++nDistinct; }
            }
            while (first > 0 && coord(first - 1) == coord(first)) { --first; }
            size_t last = lo;
            for (size_t nDistinct = 0; last < n - 1 && nDistinct < stencilHalfWidth; ++last)
            {
                if (coord(last + 1) != coord(last)) { ++nDistinct; }
            }
            std::vector<double> c(last - first + 1);
            std::array<std::vector<double>, NF> f;
            for (auto& fk : f) { fk.resize(c.size()); }
            for (size_t k = 0; k < c.size(); ++k)
            {
                c[k] = coord(first + k);
                const auto v = value(first + k);
                for (size_t n2 = 0; n2 < NF; ++n2) { f[n2][k] = v[n2]; } // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) -- n2 is a loop index bounded by compile-time constant NF
            }
            const Interpolator1D<NF> local(c, f, interpType_);
            return local(q);
        }

        /**
         * @brief Evaluate the interpolator along one rib at a given x
         * @param j Index of the rib
         * @param x Query x coordinate
         * @returns The interpolated value(s)
         * @details
         * Uses the prebuilt rib interpolator if there is one, and
         * otherwise (for a lazily-evaluated mesh) a local one built by
         * evalLocal().
         */
        [[nodiscard]] auto evalRib(const size_t j, const double x) const
        {
            if (!lazySource_) { return (*(ribInterp_[j]))(x); }
            const auto& xv = mesh_.xData();
            return evalLocal(
                [&xv, j](const size_t i) -> double { return xv[i, j]; },
                [this, j](const size_t i) { return lazySource_->value(i, j); },
                mesh_.nx(), x);
        }

        /**
         * @brief Evaluate the interpolator along one spine at a given coordinate
         * @param i Index of the spine
         * @param s Query coordinate along the spine -- the cumulative
         *   length along it, or y if monotonic_
         * @returns The interpolated value(s)
         * @details
         * As evalRib(), for a spine. For a lazily-evaluated mesh, the
         * spine's own coordinates are accumulated once, in O(Ny), in
         * the same order as Mesh2DGrid::SView.
         */
        [[nodiscard]] auto evalSpine(const size_t i, const double s) const
        {
            if (!lazySource_) { return (*(spineInterp_[i]))(s); }
            const size_t nyLoc = mesh_.ny();
            std::vector<double> coord(nyLoc);
            if (monotonic_)
            {
                for (size_t j = 0; j < nyLoc; ++j) { coord[j] = mesh_.yData()[j]; }
            }
            else
            {
                const auto& xv = mesh_.xData();
                const auto& yv = mesh_.yData();
                coord[0] = 0.0;
                for (size_t j = 1; j < nyLoc; ++j)
                {
                    coord[j] = coord[j-1] + std::sqrt(
                        std::pow(xv[i,j] - xv[i,j-1], 2) +
                        std::pow(yv[j] - yv[j-1], 2));
                }
            }
            return evalLocal(
                [&coord](const size_t j) -> double { return coord[j]; },
                [this, i](const size_t j) { return lazySource_->value(i, j); },
                nyLoc, s);
        }

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
        std::shared_ptr<const LazyValueSource<NF>>
            lazySource_;              /**< Supplies function values on demand, for a lazily-evaluated mesh; null otherwise */
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // MESH2DINTERPOLATOR_HPP