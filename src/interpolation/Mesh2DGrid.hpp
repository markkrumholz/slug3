/**
 * @file Mesh2DGrid.hpp
 * @author Mark Krumholz
 * @brief Machinery to do interpolation on a semi-tensor mesh
 * @date 2024-06-19
 * @details
 * The classes in this module and a few others solve a generic
 * mathematical problem. We have a function whose value is specified
 * at a set of data points that lie on a semi-tensor grid of the form
 *
 * (x_{ij}, y_j), i = 0 ... N-1, j = 0 ... M-1
 *
 * The points are non-decreasing, so x_{i,j} <= x_{i+1,j} and y_j <=
 * y_{j+1}, but need not be strictly increasing. There also need not
 * be any particular ordering of x_{i,j} and x_{i,j+1}, which implies
 * that the outer edge of the mesh need not be convex, and that edges
 * can be degenerate, i.e., (x_{ij}, y_j) = (x_{i+1,j}, y_j) =
 * (x_{i,j+1}, y_{j+1}), so the ij - i+1,j and ij - i,j+1 edges are
 * identical. Nor is the degeneracy guaranteed to be the same in all
 * rows, i.e., we may have x_{i,j} == x_{i+1,j} but 
 * x_{i,j+1} != x_{i+1,j+1}.
 *
 * Given a set of function values f_{ij} defined at the grid points,
 * we wish to define an interpolating function f(x,y) that will give
 * us the value of f at any point in the interior of the semi-regular
 * grid. We wish to extract this in two forms. First, we want to be
 * able to specify a value (x,y) that lies in the interior of the grid
 * and get back a value of f. Second, we want to be able to specify a
 * value x and get back a function f_x(y) that interpolates in the y
 * direction at fixed x, or specify a value of y and get back a
 * function f_y(x) that interpolates as fixed y.
 *
 * In the context of stellar evolution, the grid consists of a set of
 * evolutionary tracks. Each track is defined by an initial mass y_j =
 * log m_j, and by a series of evolutionary points, each characterized
 * by a time x_ij = log t_ij. The times t_ij are not the same for
 * different initial masses, so in general x_{ij} != x_{i+1,j}. For
 * each point in the evolutionary tracks, we have one or more stellar
 * properties f_{ij}, and we want to be able to interpolate these
 * properties, either by specifying a mass and an age directly, or by
 * specifying an age and getting back an isochrone -- a function that
 * returns stellar properties as a function of mass at fixed age.
 *
 * To solve this problem, we note that the points (x_ij, y_j) define a
 * series of quadrilaterals in the (x, y) plane. These quadrilaterals
 * have parallel edges in the x direction, but non-parallel edges in
 * the y direction. Graphically, the structure is something like this:
 *
 *  (x_{i,j+1}, y_{j+1})   (x_{i+1}, y_{i+1,j+1})
 *          o----------------------o
 *         /                        \
 *        /                          \
 *       /                            \
 *      o------------------------------o
 *  (x_{ij}, y_j)              (x_{i+1,j}, y_j})
 *
 * Note that "degenerate" quadrilaterals are allowed, whereby
 * either the top or bottom edge has length zero, and thus the
 * shape is actually a triangle.
 * 
 * For convenience we will refer to the edges at constant y as
 * "ribs", and the edges that are not at constant y as
 * "spines". Let index (i,j) refer to the spine that lies
 * above point (x_{ij}, y_j). The slope of this spine is
 *
 * m_{ij} = (y_{j+1} - y_j) / (x_{i,j+1} - x_{ij}).
 *
 * We can also define a distance coordinate along the spines.
 * There are two possible ways to parmaeterize the coordinates
 * in this direction. One is just by the y coordinate, but another
 * choice is to define the distances to the grid points along the
 * cell edges the define the spines recursively by
 *
 * s_{i0} = 0
 * s_{i,j+1} = s_{i,j} + sqrt[ (x_{i,j+1} - x_{ij})^2 +
 *                             (y_{j+1} - y_j)^2 ].
 *
 * This can be generalized naturally to give a coordinate s for an
 * arbitrary point (x, y) that lies on one of the spines:
 *
 * s(x,y) = s_{i,jmax} + sqrt[ (x - x_{i,jmax})^2 +
 *                             (y - y_{jmax})^2 ]
 *
 * where jmax = max(j) s.t. y > y_j and the condition that (x, y) be
 * on the spine is equivalent to requiring that
 *
 * y - y_{jmax} = m_{i,jmax} (x - x_{i,jmax})
 *
 * for some i.
 *
 * Next, we define spline fits along both the horizontal and vertical
 * edges. Specifically, consider the set of points along a rib
 *
 * p_j = (x_{0j}, f_{0j}), (x_{1,j}, f_{1,j}), ...
 *       (x_{N-1,j}, f_{N-1,j})
 *
 * These points p_j can be used to define a spline function P_j(x)
 * that accepts as an argument any x in [ x_{0j}, x_{N-1,j} ], and
 * returns a function value at x. The order and form of the spline is
 * arbitrary, but we will use a true spline, i.e., one that is
 * constrained to go through each data point, rather than a basis
 * spline.
 *
 * We can similarly define the set of points in along the ribs.
 * If we use the variable s defined above as our coordinate
 * along the spines, this is:
 *
 * q_i = (s_{i0}, f_{i0}), (s_{i1}, f_{i1}), ...
 *       (s_{i,M-1}, f_{i,M-1})
 *
 * and their corresponding spine spline functions Q_i(s(y)), where
 *
 * s(y) = s(x(y), y)
 * x(y) = x_{i,jmax} + (y - y_{jmax}) / m_{i,jmax}
 *
 * Alternatively, if we use the y position as our coordinate across
 * the spines the results are the same, except that we simply set
 * s = y everywhere.
 *
 * With these definitions in place, we can now define the procedures
 * for interpolating to a single point or alone a line of fixed
 * x. First consider generating an interpolating function along a line
 * of fixed x. We find all locations where a line of fixed x
 * intersects the mesh; these locations can be alone either spines
 * or ribs. Let y_k be the set of intersection points,
 * ordered so that y_{k+1} > y_k. For each intersection point, we
 * define a corresponding interpolated function value f_k by
 * evaluating either P(x) (if the point corresponds to intersecting a
 * horizontal cell edge) or Q(s(y)) (if it corresponds to intersecting
 * a vertical cell edge). Given the set (y_k, f_k), we can define a
 * new interpolating function F(y) that allows us to evaluate the
 * function at any y.
 *
 * Again, depending on our choice of coordinate along the spines, we
 * can also just set s = y in the above. The reason for using s = y
 * as an option is that this option, together with an overall linear
 * interpolation scheme, makes it possible to enforce strict
 * monotonicity, in the sense that if some quantity is monotonic in
 * x along all tracks, and we wish all interpolants to be similarly
 * monotonic, then we can guarantee that this will be so by setting
 * s = y and using linear interpolation. Any other choice does not
 * strictly guarantee monotonic behavior. However, the price of this
 * choice is an overall reduction in the level of accuracy in the
 * interpolation.
 *
 * The case of interpolating to a single point is analogous, except
 * that we need not find all the intersection points y_k. We just find
 * as many as needed on each side to have enough for the type of
 * interpolating function we wish to use.
 *
 * Finally, the case of creating a function that interpolates in x at
 * fixed y is the same as well, except that the intersection points
 * are all vertical.
 *
 * The code to do all of this is broken up into two classes:
 *
 * Mesh2DGrid -- this handles all the geometry of defining a
 *    grid, finding intersections of lines through it, etc. It does
 *    not know anything about interpolation, and just handles
 *    geometry.
 * Mesh2dInterpolator -- this class wraps Mesh2DGrid, and
 *    handles all the interpolation machinery.
 */

#ifndef MESH2DGRID_HPP
#define MESH2DGRID_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mdspan>
#include <tuple>
#include <utility>
#include <vector>

// Disable linting for array bounds checking in this
// file, since the overhead associated with enforcing
// such checks severely interferes with performance
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

/**
 * @brief A namespace to hold interpolation machinery
 */
namespace interp
{

    /**
     * @class Mesh2DGrid
     * @brief A class representing a semi-tensor 2D grid
     * @details
     * This class represents a grid where the points in one
     * direction (the y direction) are arranged in a tensor grid
     * (i.e., in an Nx x Ny grid, there is one y value for each
     * of the Nx rows), but not in the other direction (i.e., the
     * values of x can be different from one row of y values to
     * to the next).
     */
    class Mesh2DGrid
    {
    public:

        // Shorten array types
        using Array2D = std::mdspan<double, std::dextents<size_t, 2>>;
        using Array1D = std::mdspan<double, std::dextents<size_t, 1>>;

        // Shorthand for numeric limits
        static constexpr double bigNum = std::numeric_limits<double>::max(); /**< Shorthand for big number */

        /**
         * @brief An enum to hold intesection types
         */
        enum class IntersectionType : std::uint8_t {
            spine, /**< Intersection with a spine */
            rib,   /**< Intersection with a rib */
            none   /**< Dummy flag value */
        };

        /**
         * @brief A struct to hold an intersection point
         * @details
         * This struct describes the results of traversing
         * the mesh at constant x, which are distinct from
         * the information needed to describe a traversal
         * at constant y.
         */
        // NOLINTBEGIN(readability-identifier-naming)
        struct xIntersectionDescriptor {
            double y;  /**< y coordinate of intersection */
            double xs; /**< x or s coordinate of intersection, depending on type */
            IntersectionType t; /**< Type of intersection -- spine or rib */
            size_t idx; /**< Index of spine or rib at intersection point */
            bool meshExit; /**< True if the line exits the mesh at this intersection point */
        };
        // NOLINTEND(readability-identifier-naming)

        /**
         * @brief A struct to hold an intersection point
         * @details
         * This struct describes the results of traversing
         * the mesh at constant y, which are distinct from
         * the information needed to describe a traversal
         * at constant x.
         */
        // NOLINTBEGIN(readability-identifier-naming)
        struct yIntersectionDescriptor {
            double x;  /**< x coordinate of intersection */
            double s; /**< s coordinate of intersection */
            size_t idx; /**< Index of spine at intersection point */
        };
        // NOLINTEND(readability-identifier-naming)
        
        // Constructors and destructor
        /**
         * @brief Construct a Mesh2DGrid
         * @param x A 2d array giving the x coordinates of the mesh points
         * @param y A 1d array giving the y coordinates of the mesh points
         * @param copyData If true, copy the data in x and y
         */
        Mesh2DGrid(const Array2D& x,
            const Array1D& y,
            bool copyData = true);
        virtual ~Mesh2DGrid() = default;
        Mesh2DGrid(const Mesh2DGrid&) = default;
        Mesh2DGrid(Mesh2DGrid&&) = default; 
        auto operator=(const Mesh2DGrid&) -> Mesh2DGrid& = delete;
        auto operator=(Mesh2DGrid&&) -> Mesh2DGrid& = delete; 

        // Observers
        /**
         * @brief Return size of Mesh2DGrid
         * @returns Size of mesh
         */
        auto size() const { return x_.size(); }

        /**
         * @brief Return x dimension of Mesh2DGrid
         * @returns Number of mesh points in x direction
         */
        auto nx() const { return x_.extent(0); }

        /**
         * @brief Return x dimension of Mesh2DGrid
         * @returns Number of mesh points in x direction
         */
        auto ny() const { return x_.extent(1); }

        /**
         * @brief Returns in the mesh is convex
         * @returns True if the mesh is convex
         */
        auto convex() const { return convex_; }

        /**
         * @brief Return a const reference to the x array
         * @returns A const references to the x array
         */
        const auto& xData() const { return x_; }

        /**
         * @brief Return a const reference to the y array
         * @returns A const references to the y array
         */
        const auto& yData() const { return y_; }
        
        /**
         * @brief Return a const reference to the s array
         * @returns A const references to the s array
         */
        const auto& sData() const { return s_; }
        
        /**
         * @brief Minimum x value in mesh
         * @returns Minimum x value in mesh
         */
        auto xMin() const { return xMin_; }

        /**
         * @brief Find minimum x value at given y
         * @returns Minimum x value at given y
         */
        auto xMin(const double y) const
        {
            assert(y >= yMin_ && y <= yMax_);
            if (y == yMin_) { return x_[0,0]; }
            if (y == yMax_) { return x_[0,ny()-1]; }
            auto j = yIdx(y);
            return x_[0,j] + ((y - y_[j]) / m_[0,j]);
        }

        /**
         * @brief Maximum x value in mesh
         * @returns Maximum x value in mesh
         */
        auto xMax() const { return xMax_; }

        /**
         * @brief Find maximum x value at given y
         * @returns Maximum x value at given y
         */
        auto xMax(const double y) const
        {
            assert(y >= yMin_ && y <= yMax_);
            if (y == yMin_) { return x_[nx()-1,0]; }
            if (y == yMax_) { return x_[nx()-1,ny()-1]; }
            auto j = yIdx(y);
            return x_[nx()-1,j] + ((y - y_[j]) / m_[nx()-1,j]);
        }

        /**
         * @brief Minimum y value in mesh
         * @returns Minimum y value in mesh
         */
        auto yMin() const { return yMin_; }

        /**
         * @brief Maximum y value in mesh
         * @returns Maximum y value in mesh
         */
        auto yMax() const { return yMax_; }

        /**
         * @brief Find minimum and maximum y position at a given x
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
        auto yLim(double x)
        const -> std::vector<std::pair<double, double>>;

        /**
         * @brief Check if a point is contained in the mesh
         * @param x x position
         * @param y y position
         * @returns True if point is within mesh
         */
        auto contains(const double x, const double y) const
        {
            if (y < yMin_ || y > yMax_) { return false; }
            return (x >= xMin(y) && x <= xMax(y));
        }

        // Mesh indexing functions
        /**
         * @brief Find index of a point in the x direction along a specified rib
         * @param x x coordinate of point
         * @param j Index of rib
         * @return Index of point along rib
         * @details
         * It is an error if the the input point is < x[0,j] or
         * > x[nx()-1,j].
         */
        auto xIdx(const double x, const size_t j) const -> size_t
        {
            jSave_ = j;  // Update cached position
            assert((x >= x_[0,j] && x <= x_[nx()-1,j]));  // Safety check
            if (x < x_[iSave_,j]) { // Check if we need to update cached index
                iSave_ = xSearch(x, 0, j, 0, iSave_); // Below cached position
            } else if (x >= x_[iSave_+1,j]) {
                // Above cached position, or exactly on its upper edge; in
                // the latter case iSave_ must still advance, since leaving
                // it unchanged would compute the offset within this cell
                // as the full cell width instead of zero, which produces
                // spurious floating-point error in downstream traversals
                iSave_ = xSearch(x, 0, j, iSave_, nx()-1);
            }
            if (iSave_ == nx()-1) { --iSave_; } // Handle edge case
            return iSave_;
        }

        /**
         * @brief Find index of point in y direction
         * @returns Index of point in y direction
         * @details
         * It is an error if the input y is < yMin_
         * or > yMax_.
         */
        auto yIdx(const double y) const -> size_t
        {
            assert(y >= yMin_ && y <= yMax_);  // Safety check
            if (y < y_[jSave_]) { // Check if we need to update cached position
                jSave_ = ySearch(y, 0, jSave_);   // Below cached position, search left
            } else if (y >= y_[jSave_ + 1]) {
                // Above cached position, or exactly on its upper edge; in
                // the latter case jSave_ must still advance, since leaving
                // it unchanged would compute the offset within this cell
                // as the full cell width instead of zero, which produces
                // spurious floating-point error in downstream traversals
                jSave_ = ySearch(y, jSave_, ny()-1);
            }
            if (jSave_ == ny()-1) { --jSave_; } // Handle edge case
            return jSave_;
        }

        /**
         * @brief Find index of a given point in the mesh
         * @param x The x position
         * @param y The y position
         * @returns The indices (i,j) of the lower left corner of the cell containing the point
         * @details
         * It is an error if the input point is outside the mesh
         */
        auto xyIdx(const double x, const double y) const
        {
            assert(contains(x,y));  // Safety assertion
            auto j = yIdx(y);       // Get y index
            const double dy = y - y_[j];  // Offset in y
            size_t i = 0;  // Output holder
            if (x < x_[iSave_,j] + (dy/m_[iSave_,j])) { // Check if we need to update cached index
                iSave_ = xSearch(x, dy, j, 0, iSave_);  // Too low
                i = iSave_;
            } else if (x >= x_[iSave_+1,j] + (dy/m_[iSave_+1,j])) {
                iSave_ = xSearch(x, dy, j, iSave_, nx()-1); // Too high
                i = iSave_;
            } else {
                i = iSave_; // Cache hit
            }
            if (i == nx()-1) { --iSave_; } // Handle dge case
            return std::pair(i,j);
        }

        /**
         * @brief Check if a given point is exactly on the mesh
         * @param x x poistion
         * @param y y position
         * @returns True or false, and intersection descriptor
         * @details
         * The return value is a pair consisting of a boolean and
         * an xIntersectionDescriptor object. If the point is
         * exactly on a rib or spine, the boolean is set to true
         * and the xIntersectionDescriptor contains the details of the
         * point on the rib or spine; if the point is not on a
         * rib or spine, the boolean is false and the
         * xIntersectionDescriptor is uninitialized and meaningless.
         */
        auto onMesh(double x, double y)
        const -> std::pair<bool, xIntersectionDescriptor>;

        // Mesh traversal functions
        /**
         * @brief Compute intersection of mesh with line of constant x
         * @param x The x coordinate
         * @param yLo Lower y limit of traversal
         * @param yHi Upper y limit of traversal
         * @returns List of points where line crosses mesh
         * @details
         * This routine returns the set of points at which a
         * line of constant x starting at yLo and ending at yHi
         * intersects the mesh. The returned points are of type
         * xIntersectionDescriptor.
         */
        auto xIntersect(
            double x,
            double yLo = std::numeric_limits<double>::lowest(),
            double yHi = std::numeric_limits<double>::max()
        ) const -> std::vector<xIntersectionDescriptor>;

        /**
         * @brief Compute intersection of mesh with line of constant y
         * @param y The y coordinate
         * @param xLo Lower x limit of traversal
         * @param xHi Upper x limit of traversal
         * @returns List of points where the line crosses mesh
         * @details
         * This routine returns the set of points at which a
         * line of constant y starting at xLo and ending at xHi
         * intersects the mesh. The returned points are of type
         * yIntersectionDescriptor.
         */
        auto yIntersect(
            double y,
            double xLo = std::numeric_limits<double>::lowest(),
            double xHi = std::numeric_limits<double>::max()
        ) const -> std::vector<yIntersectionDescriptor>;
        
    private:

        // Construction helpers
        /**
         * @brief Safety checks on construction inputs
         * @param x A 2d array giving the x coordinates of the mesh points
         * @param y A 1d array giving the y coordinates of the mesh points
         */
        static void safetyCheckInputs(const Array2D& x, 
            const Array1D& y);
        /**
         * @brief Compute slopes and spine lengths
         */
        void computeSlopesAndLengths();
        /**
         * @brief Set the convexity of the mesh
         */
        void setConvexity();

        // Indexing / search helpers
        /**
         * @brief Search for index of y position
         * @param y Point for which to search
         * @param idxLo Lower index for search
         * @param idxHi Upper index for search
         * @details
         * This routine carries out a binary search for the
         * index at which position y can be found, between
         * idxLo and idxHi. This is a helper function for
         * other functions, so no checking is done to ensure
         * that y is in fact between these indices; the calling
         * routine is responsible for checking that.
         */
        auto ySearch(const double y,
            const size_t idxLo,
            const size_t idxHi
        ) const -> size_t
        {
            auto jLo = idxLo;
            auto jHi = idxHi;
            while (jHi > jLo + 1) {
                auto j = (jLo + jHi) / 2;
                if (y > y_[j]) {
                    jLo = j;
                } else {
                    jHi = j;
                }
            }
            if (y == y_[jHi]) { return jHi; }
            return jLo;
        }
        /* @brief Search for index of x position at given y
         * @param x x position for which to search
         * @param dy Distance from y[j]
         * @param j Index of y position
         * @param idxLo Lower index for search
         * @param idxHi Upper index for search
         * @details
         * This routine carries out a binary search for the
         * index at which position y can be found, between
         * idxLo and idxHi. This is a helper function for
         * other functions, so no checking is done to ensure
         * that y is in fact between these indices; the calling
         * routine is responsible for checking that.
         */
        auto xSearch(
            const double x,
            const double dy,
            const size_t j,
            const size_t idxLo,
            const size_t idxHi
        ) const -> size_t
        {
            size_t iLo = idxLo;
            size_t iHi = idxHi;
            while (iHi > iLo + 1) {
                const auto i = (iLo + iHi) / 2;
                if (x > x_[i,j] + (dy/m_[i,j])) {
                    iLo = i;
                } else {
                    iHi = i;
                }
            }
            if (x == x_[iHi,j] + (dy/m_[iHi,j])) {
                return iHi; // Edge case
            }
            return iLo;
        }

        // Search helpers
        /**
         * @brief Traverse mesh to find limits on y at fixed x
         * @param x x coordinate at which to traverse
         * @param yL Limits on y being filled
         * @returns True if traversal should continue, false if it should stop
        */
        auto yLimTraverse(double x, std::vector<double>& yL) const -> bool;

        /**
         * @brief Find the list of points where a line of constant x intersects mesh ribs and spines
         * @param x x coordinate
         * @param yMin Minimum y of segment
         * @param yMax Maximum y of segment
         * @param startInterior True if segment begins in mesh interior rather than at edge
         * @param endInterior True if segment ends in mesh interior rather than at edge
         * @details
         * This method is a helper fucntion for xIntersect. It differs from
         * xIntersect in that yMin and yMax must be guaranteed to be within
         * a range where the mesh is convex, and that it already knows if the
         * starting points and ending points of a segment on at the edge of a
         * mesh or in its interior.
         */
        auto xIntersectSeg(
            double x, 
            double yMin,
            double yMax,
            bool startInterior,
            bool endInterior
        ) const -> std::vector<xIntersectionDescriptor>;

        /**
         * @brief Helper to xIntersectSeg to start traversals on mesh interior
         * @param x x coordinate
         * @param y y coordinate
         * @param intList List of mesh intersections to populate
         * @param lastIntersectLeft True if the last intersection point was on the left edge of a cell
         * @param lastIntersectRight True if the last intersection point was on the right edge of a cell
         */
        void xIntersectSegStartInterior(
            double x,
            double& y,
            std::vector<xIntersectionDescriptor>& intList,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const;

        /**
         * @brief Helper to xIntersectSeg to start traversals on mesh edge
         * @param x x coordinate
         * @param y y coordinate
         * @param yMin Minimum y value of segment
         * @param intList List of mesh intersections to populate
         * @param lastIntersectLeft True if the last intersection point was on the left edge of a cell
         * @param lastIntersectRight True if the last intersection point was on the right edge of a cell
         * @returns True if starting point is tangent to mesh, and thus search should end immediately
         */
        auto xIntersectSegStartEdge(
            double x,
            double& y,
            double yMin,
            std::vector<xIntersectionDescriptor>& intList,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> bool;
        
        /**
         * @brief Find the next intersection of a line of constant x with the mesh
         * @param x x coordinate
         * @param y Starting y coordinate
         * @param yStop y Coordinate at which to stop
         * @param endInterior True if (x,yStop) is in the mesh interior and not on an edge
         * @param lastIntersectLeft True if the last intersection point was on the left edge of a cell
         * @param lastIntersectRight True if the last intersection point was on the right edge of a cell
         * @returns Outcome of the search
         * @details
         * The return value is a pair consisting if a boolean and an xIntersectionDescriptor.
         * The boolean is true if the search stopped because an intersection was found, and
         * false if the search stopped because yStop was reached. The xIntersectionDescriptor
         * contains the details of the intersection point found; if no valid intersection point
         * was found because the search terminated, the returned point will have a y
         * value of NAN. On return, the values of y, lastIntersectLeft, and
         * lastIntersectRight are modified to contain the position and
         * intersection type of the new intersection point that has been found.
         */
        auto findNextIntersect(
            double x,
            double& y,
            double yStop,
            bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>;

        /**
         * @brief Get distances from given position to cell edges moving at constant x
         * @param x Current x position
         * @param y Current y position
         * @param searchUp True if the search should be in the direction of increasing y
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns Distances to next right, to left spine, and to right spine
         * @details
         * Return values are set to bigNum if the path does not intersect that segment.
         * The values of lastIntersectLeft and lastIntersectRight are used to
         * exclude certain directions from consideration, avoiding numerical issues.
         */
        auto findIntersectDistance(
            double x,
            double y,
            bool searchUp,
            bool lastIntersectLeft,
            bool lastIntersectRight
        ) const -> std::tuple<double, double, double>;

        /**
         * @brief Handle case where next intersection is on the rib of a cell above starting position
         * @param y Starting y position
         * @param x x coordinate
         * @param yStop y Coordinate at which to stop
         * @param endInterior True if (x,yStop) is in the mesh interior and not on an edge
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns A pair of a boolean and an xIntersectionDescriptor
         * @details
         * The return value is a pair consisting if a boolean and an xIntersectionDescriptor.
         * The boolean is true if the search stopped because an intersection was found, and
         * false if the search stopped because yStop was reached. The xIntersectionDescriptor
         * contains the details of the intersection point found; if no valid intersection point
         * was found because the search terminated, the returned point will have a y
         * value of NAN. On return, the values of y, lastIntersectLeft, and
         * lastIntersectRight are modified to contain the position and
         * intersection type of the new intersection point that has been found.
         * On return from this function, y, lastIntersectLeft and
         * lastIntersectRight will be modified to reflact the new position.
         */
        auto handleNextIntersectTopRib(
            double& y,
            double x,
            double yStop,
            bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>;

        /**
         * @brief Handle case where next intersection is on the rib of a cell below starting position
         * @param y Starting y position
         * @param x x coordinate
         * @param yStop y Coordinate at which to stop
         * @param endInterior True if (x,yStop) is in the mesh interior and not on an edge
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns A pair of a boolean and an xIntersectionDescriptor
         * @details
         * The return value is a pair consisting if a boolean and an xIntersectionDescriptor.
         * The boolean is true if the search stopped because an intersection was found, and
         * false if the search stopped because yStop was reached. The xIntersectionDescriptor
         * contains the details of the intersection point found; if no valid intersection point
         * was found because the search terminated, the returned point will have a y
         * value of NAN. On return, the values of y, lastIntersectLeft, and
         * lastIntersectRight are modified to contain the position and
         * intersection type of the new intersection point that has been found.
         * On return from this function, y, lastIntersectLeft and
         * lastIntersectRight will be modified to reflact the new position.
         */
        auto handleNextIntersectBottomRib(
            double& y,
            double x,
            double yStop,
            bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>;

        /**
         * @brief Handle case where next intersection is on the left spine of a cell
         * @param y Starting y position
         * @param x x coordinate
         * @param yStop y coordinate at which to stop
         * @param dy y distance to the intersection point
         * @param endInterior True if (x,yStop) is in the mesh interior and not on an edge
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns A pair of a boolean and an xIntersectionDescriptor
         * @details
         * The return value is a pair consisting if a boolean and an xIntersectionDescriptor.
         * The boolean is true if the search stopped because an intersection was found, and
         * false if the search stopped because yStop was reached. The xIntersectionDescriptor
         * contains the details of the intersection point found; if no valid intersection point
         * was found because the search terminated, the returned point will have a y
         * value of NAN. On return, the values of y, lastIntersectLeft, and
         * lastIntersectRight are modified to contain the position and
         * intersection type of the new intersection point that has been found.
         * On return from this function, y, lastIntersectLeft and
         * lastIntersectRight will be modified to reflact the new position.
         */
        auto handleNextIntersectLeftSpine(
            double& y,
            double x,
            double yStop,
            double dy,
            bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>;

        /**
         * @brief Handle case where next intersection is on the left spine of a cell
         * @param y Starting y position
         * @param x x coordinate
         * @param yStop y coordinate at which to stop
         * @param dy y distance to the intersection point
         * @param endInterior True if (x,yStop) is in the mesh interior and not on an edge
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns A pair of a boolean and an xIntersectionDescriptor
         * @details
         * The return value is a pair consisting if a boolean and an xIntersectionDescriptor.
         * The boolean is true if the search stopped because an intersection was found, and
         * false if the search stopped because yStop was reached. The xIntersectionDescriptor
         * contains the details of the intersection point found; if no valid intersection point
         * was found because the search terminated, the returned point will have a y
         * value of NAN. On return, the values of y, lastIntersectLeft, and
         * lastIntersectRight are modified to contain the position and
         * intersection type of the new intersection point that has been found.
         * On return from this function, y, lastIntersectLeft and
         * lastIntersectRight will be modified to reflact the new position.
         */
        auto handleNextIntersectRightSpine(
            double& y,
            double x,
            double yStop,
            double dy,
            bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>;

        /**
         * @brief Handle corner cases for intersections with top ribs
         * @param x x coorindate
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns True if the corner case check found that we have exited the mesh
         */
        auto handleTopRibCornerCases(
            double x,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> bool;
 
        /**
         * @brief Handle corner cases for intersections with bottom ribs
         * @param x x coorindate
         * @param lastIntersectLeft True if last intersection found was on left of cell
         * @param lastIntersectRight True if last intersection found was on right of cell
         * @returns True if the corner case check found that we have exited the mesh
         */
        auto handleBottomRibCornerCases(
            double x,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> bool;
        
        // Input data
        Array2D x_{};                  /**< A 2d array giving the x coordinates of the mesh points */
        Array1D y_{};                  /**< A 1d array giving the y coordinates of the mesh points */
        std::vector<double> xData_{};  /**< Data holder for x_ */
        std::vector<double> yData_{};  /**< Data holder for y_ */

        // Descriptors
        double xMin_;                 /**< Lower limit of mesh in x direction */
        double xMax_;                 /**< Upper limit of mesh in x direction */
        double yMin_;                 /**< Lower limit of mesh in y direction */
        double yMax_;                 /**< Upper limit of mesh in y direction */
        bool convex_;                 /**< True if mesh is convex */

        // Derived data
        Array2D m_{};                   /**< Slopes of mesh spines */
        Array2D s_{};                   /**< Lengths of each spine segment */
        std::vector<double> mData_{};   /**< Data holder for m_ */
        std::vector<double> sData_{};   /**< Data holder for s_ */

        // Mutables
        mutable size_t iSave_;        /**< Cached x index for search acceleration */
        mutable size_t jSave_;        /**< Cached y index for search acceleration */
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // MESH2DGRID_HPP
