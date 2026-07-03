/**
 * @file Mesh2DGrid.cpp
 * @author Mark Krumholz
 * @brief Implementation of the Mesh2DGrid class
 * @date 2024-06-19
 */

#include "Mesh2DGrid.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mdspan>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

// Disable linting for array bounds checking in this
// file, since the overhead associated with enforcing
// such checks severely interferes with performance
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

namespace interp
{

    Mesh2DGrid::Mesh2DGrid(const Array2D& x,
        const Array1D& y,
        bool copyData) :
        xMin_(0.0), xMax_(0.0),
        yMin_(0.0), yMax_(0.0),
        convex_(true),
        iSave_(0), jSave_(0)
    {
        // Run safety checks on inputs
        safetyCheckInputs(x, y);

        // Initialize local arrays; if we are non-owning,
        // do this just by copying the references passed,
        // while if we are owning also copy the underlying data
        if (copyData)
        {
            // Copy x data; note that we copy explicitly using
            // indices so that this works even if x is non-contiguous
            xData_.resize(x.extent(0) * x.extent(1));
            x_ = Array2D(xData_.data(), x.extent(0), x.extent(1));
            for (size_t i = 0; i < x.extent(0); ++i) {
                for (size_t j = 0; j < x.extent(1); ++j) {
                    x_[i,j] = x[i,j];
                }
            }

            // Copy y data
            yData_.resize(y.size());
            y_ = Array1D(yData_.data(), y.extent(0));
            for (size_t j = 0; j < y.size(); ++j) {
                y_[j] = y[j];
            }
        }
        else
        {
            // Just copy metadata
            x_ = x;
            y_ = y;
        }

        // Compute spine slopes and lengths
        computeSlopesAndLengths();

        // Set convexity of the mesh
        setConvexity();
    }


    // Safety checking routine
    void Mesh2DGrid::safetyCheckInputs(const Array2D& x,
        const Array1D& y)
    {
        // Safety check that dimensions match
        if (x.extent(1) != y.size())
        {
            throw std::runtime_error("Mesh2DGrid: y must match size of second dimension of x");
        }
        // Safety check that x and y are non-decreasing
        for (size_t j = 0; j < x.extent(1); ++j) {
            for (size_t i = 1; i < x.extent(0); ++i) {
                if (x[i-1,j] > x[i,j]) {
                    throw std::runtime_error("Mesh2DGrid: each row of x must be non-decreasing");
                }
            }
        }
        for (size_t j = 1; j < y.size(); j++) {
            if (y[j-1] > y[j]) {
                throw std::runtime_error("Mesh2DGrid: y must be non-decreasing");
            }
        }
        // Safety check that x and y cover some non-zero range
        for (size_t j = 0; j < x.extent(1); ++j) {
            if (x[0,j] == x[x.extent(0)-1,j]) {
                throw std::runtime_error("Mesh2DGrid: each row of x must span non-zero size");
            }
        }
        if (y[0] == y[y.size()-1]) {
            throw std::runtime_error("Mesh2DGrid: y must span a non-zero size");
        }
    }


    // Compute spine slopes and lengths
    void Mesh2DGrid::computeSlopesAndLengths()
    {
        // Compute the slopes
        mData_.resize(nx() * (ny() - 1));
        m_ = Array2D(mData_.data(), nx(), ny()-1);
        for (size_t j = 0; j < ny()-1; ++j) {
            for (size_t i = 0; i < nx(); ++i) {
                if (x_[i,j+1] != x_[i,j]) {
	                m_[i,j] = (y_[j+1] - y_[j]) / (x_[i,j+1] - x_[i,j]);
                } else {
                    m_[i,j] = bigNum;
                }
            }
        }

        // Compute spine segment lengths
        sData_.resize(nx() * ny());
        s_ = Array2D(sData_.data(), nx(), ny());
        for (size_t i = 0; i < nx(); ++i) {
            s_[i,0] = 0.0;
            for (size_t j = 1; j < ny(); ++j) {
                s_[i,j] = s_[i,j-1] + std::sqrt(
                    std::pow(x_[i,j] - x_[i,j-1], 2) +
                    std::pow(y_[j] - y_[j-1], 2)  
                );
            }
        }

        // Compute outer extent of mesh
        xMin_ = x_[0,0];
        xMax_ = x_[nx()-1,0];
        for (size_t j = 1; j < ny(); ++j) {
            xMin_ = std::min(xMin_, x_[0,j]);
            xMax_ = std::max(xMax_, x_[nx()-1,j]);
        }
        yMin_ = y_[0];
        yMax_ = y_[ny() - 1];
    }

    // Set convexity
    void Mesh2DGrid::setConvexity()
    {
        // Check if the mesh is convex; the condition for
        // convexity is that the slopes on the left edge
        // never change from positive to negative, and those
        // on the right edge never change from negative to
        // positive
        bool leftSlopePos = false;
        bool rightSlopeNeg = false;
        for (size_t j = 0; j < ny()-1; ++j) {
            if (m_[0,j] != bigNum) {
                if (leftSlopePos && m_[0,j] < 0) { convex_ = false; }
                if (m_[0,j] > 0) { leftSlopePos = true; }
            }
            if (m_[nx()-1,j] != bigNum) {
                if (rightSlopeNeg && m_[nx()-1,j] > 0) { convex_ = false; }
                if (m_[nx()-1,j] < 0) { rightSlopeNeg = true; }
            }
            if (!convex_) { break; }
        }
    }

    // Check if a given input point is exactly on the mesh
    auto Mesh2DGrid::onMesh(const double x, const double y) const ->
        std::pair<bool, xIntersectionDescriptor>
    {
        xIntersectionDescriptor d = {
            .y = NAN, 
            .xs = NAN, 
            .t = IntersectionType::none,
            .idx = 0,
            .meshExit = false }; // Output holder
        if (!contains(x,y)) {
            return std::make_pair(false, d); // Trivial case
        }
        auto [i,j] = xyIdx(x, y); // Get indices for point     

        // Check if we are on a rib
        const double dy = y - y_[j];
        if (dy == 0.0) {
            d.y = y;
            d.xs = x;
            d.t = IntersectionType::rib;
            d.idx = j;
            d.meshExit = false;
            return std::make_pair(true, d);
        }

        // Special case: check if we are on the top rib; this
        // needs to be handled separately because xyIdx will never return
        // that we are exactly on the top rib
        if (y == y_[ny()-1]) {
            d.y = y;
            d.xs = x;
            d.t = IntersectionType::rib;
            d.idx = ny()-1;
            d.meshExit = true;
            return std::make_pair(true, d);
        }

        // Check if we are on a spine
        double dx = x - x_[i,j];
        if (m_[i,j] == dy / dx ||
            (m_[i,j] == bigNum && dx == 0.0)) 
        {
            d.y = y;
            d.xs = s_[i,j] +
                std::sqrt(std::pow(dx,2) + std::pow(dy,2));
            d.t = IntersectionType::spine;
            d.idx = i;
            d.meshExit = false;
            return std::make_pair(true, d);
        }

        // Special case: check if we are on the rightmost spine; this
        // is a special case for the same reason as the top rib
        if (i == nx()-2) {
            dx = x - x_[i+1,j];
            if (m_[i+1,j] == dy / dx ||
	            (m_[i+1,j] == bigNum && dx == 0))
            {
                d.y = y;
                d.xs = s_[i+1,j] + 
                    std::sqrt(std::pow(dx,2) + std::pow(dy,2));
                d.t = IntersectionType::spine;
                d.idx = i+1;
                d.meshExit = true;
                return std::make_pair(true, d);
            }
        }

        // If we get here, we are not on a rib or spine
        return std::make_pair(false, d);
    }

    // Compute limits on y at fixed x
    auto Mesh2DGrid::yLim(const double x) const -> std::vector<std::pair<double,double>>
    {
        if (x < xMin_ || x > xMax_)
        {
            // Case where line misses mesh entirely; return
            // empty vector
            return {};
        }

        // If we are here, the line intersects the
        // mesh at least once; create accumulator for result
        std::vector<double> yL; // We'll break this up into pairs at the end

        // Check if the x we have been given lies within the range covered
        // by the bottom rib
        if (x >= x_[0,0] && x <= x_[nx()-1,0])
        {
            // Yes, this x is within the bottom rib, so first intersection
            // is y coordinate of this rib
            yL.push_back(yMin_);
            jSave_ = 0; // Cached pointer
        }
        else
        {
            // Point we have been given is outside the bottom rib, so the
            // line must hit the bottom of the mesh on one of the side
            // spines.
            if (x < x_[0,0]) { iSave_ = 0; } // Left of bottom rib
            else { iSave_ = nx() - 1; } // Right of bottom rib
            jSave_ = 0; // Initial position
            
            // Now march upward until we encounter the mesh edge
            while ((x_[iSave_,jSave_] - x) * (x_[iSave_,jSave_+1] - x) > 0)
            { ++jSave_; }
            yL.push_back(y_[jSave_] + 
                (m_[iSave_,jSave_] * (x - x_[iSave_,jSave_])));
            ++jSave_;
        }

        // Now march upward through the mesh, recording where we exit
        while (yLimTraverse(x, yL)) {}

        // Make sure we don't have a bad iSave_
        if (iSave_ == nx()-1) { --iSave_; }
    
        // If we've hit the top of the mesh and we're still inside it, add
        // the mesh top as the final point
        if (yL.size() % 2 == 1) { yL.push_back(yMax_); }

        // Now put final result into pairs
        std::vector<std::pair<double,double>> result(yL.size()/2);
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = { yL[2*i], yL[(2*i) + 1] };
        }

        // Return final result
        return result;

    }

    // Traversal function used by yLim
    auto Mesh2DGrid::yLimTraverse(
        const double x, 
        std::vector<double>& yL
    ) const -> bool
    {
        // Check left edge; be careful with corner cases, where we need
        // to consider both the slope and whether we are currently
        // inside or outside the mesh to decide if we have a hit
        double dx0 = x_[0,jSave_] - x;
        double dx1 = x_[0,jSave_+1] - x;
        if (dx0 * dx1 < 0)
        {
            // This is the regular case
            yL.push_back(y_[jSave_] + 
                (m_[0,jSave_] * (x - x_[0,jSave_])));
        }
        else if (dx1 == 0 && jSave_ < ny()-2) 
        {
            if ((m_[0,jSave_+1] > 0 &&
                yL.size() % 2 == 1 &&
                m_[0,jSave_+1] != bigNum) ||
                (m_[0,jSave_+1] < 0 &&
                yL.size() % 2 == 0))
            {
                // This is the corner case
                yL.push_back(y_[jSave_] + 
                    (m_[0,jSave_] * (x - x_[0,jSave_])));
            }
        }

        // Check right edge; again, be careful of corner cases
        dx0 = x_[nx()-1,jSave_] - x;
        dx1 = x_[nx()-1,jSave_+1] - x;
        if (dx0 * dx1 < 0)
        {
            // This is the regular case
            yL.push_back(y_[jSave_] + 
                (m_[nx()-1,jSave_] * (x - x_[nx()-1,jSave_])));
        } 
        else if (dx1 == 0 && jSave_ < ny()-2)
        {
            if ((m_[nx()-1,jSave_+1] < 0 &&
                yL.size() % 2 == 1) ||
                (m_[nx()-1,jSave_+1] > 0 &&
                yL.size() % 2 == 0 &&
                m_[nx()-1,jSave_+1] != bigNum))
            {
                // This is the corner case
                yL.push_back(y_[jSave_] + 
                    (m_[nx()-1,jSave_] * (x - x_[nx()-1,jSave_])));
            }
        }

        // Special check: if we started outside the mesh, it is possible
        // that we crossed both edges; if we did then it is also
        // possible that we got the order wrong, in that we added the
        // left crossing before the right, when it should have been the
        // other way around. Here we check for that possibility and fix
        // the problem if it has occurred.
        if (yL.size() >= 2)
        {
            if (yL[yL.size()-2] > yL[yL.size()-1])
            {
                std::swap(yL[yL.size()-2], yL[yL.size()-1]);
            }
        }
    
        // If we're convex, there are only 2 limits to find; stop if
        // we've found both
        if (convex_ && yL.size() == 2) { return false; }
    
        // Either increment j, or exit if we've reached the top
        if (jSave_ == ny()-2) { return false; }
        jSave_++;

        // If we reach here, continue traversal
        return true;

    }

    // Compute intersections in the x direction
    auto Mesh2DGrid::xIntersect(const double x,
        const double yLo,
        const double yHi) const ->
        std::vector<xIntersectionDescriptor>
    {
        std::vector<xIntersectionDescriptor> intList; // Output holder

        // First find where line crosses out edge of mesh
        auto yL = yLim(x);
        if (yL.empty()) {
            return intList;  // Trivial case where line misses mesh
        }

        // Loop over line segments that pass through the mesh
        for (const auto [yMin, yMax] : yL)
        {
            // Apply yLo and yHi values
            const double yLoSeg = std::max(yMin, yLo);
            const double yHiSeg = std::min(yMax, yHi);

            // Skip segment if applying limits makes it zero length
            if (yLoSeg >= yHiSeg) { continue; }

            // Mark whether the starting and ending points for this segment
            // are on the edge of the mesh
            const bool startInterior = (yLo > yMin);
            const bool endInterior = (yHi < yMax);

            // Find all the intersection points for this segment
            auto intListSeg = xIntersectSeg(x, yMin, yMax,
                startInterior, endInterior);

            // Append to list
            intList.append_range(intListSeg);
        }

        // Return final intersection list
        return intList;
    }

    // Helper function to find intersections with segments at constant x
    auto Mesh2DGrid::xIntersectSeg(const double x, 
        const double yMin,
        const double yMax,
        const bool startInterior,
        const bool endInterior) const
        -> std::vector<xIntersectionDescriptor>
    {
        // Allocate empty output holder
        std::vector<xIntersectionDescriptor> intList;

        // Get starting indices, and, if we're starting on the mesh edge,
        // record the first intersection point and set the intersection
        // flags
        double y = yMin;
        bool lastIntersectLeft = false;
        bool lastIntersectRight = false;
        if (startInterior) {
            xIntersectSegStartInterior(x, y, intList,
                lastIntersectLeft, lastIntersectRight);
        }
        else
        {
            const bool tangent = 
                xIntersectSegStartEdge(x, y, yMin, intList,
                    lastIntersectLeft, lastIntersectRight);
            if (tangent) { return intList; }
        }
  
        // We have now set iSave_ and jSave_ to give the indices of the
        // cell that contains the starting point, where "contains" includes
        // the lower left edges of the cell and excludes the upper right
        // edges. Now we loop to find other intersection points, continuing
        // until we reach the termination condition.
        while (true)
        {
            // Find next intersection point
            const auto [continueSearch, nextInt] =
            findNextIntersect(
                x, y, yMax, endInterior, 
                lastIntersectLeft, lastIntersectRight);

            // Record intersection point if valid one found
            if (nextInt.t != IntersectionType::none)
            {
                intList.push_back(nextInt);
            }

            // Stop we have reached the end of the segment
            if (!continueSearch) { break;}
        }

        // Return final intersection list
        return intList;
    }


    // Helper to start segment traversal on mesh interior
    void Mesh2DGrid::xIntersectSegStartInterior(
        const double x,
        double& y,
        std::vector<xIntersectionDescriptor>& intList,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const
    {
        // Check if our starting point is exactly on a rib or spine; note that
        // this call also sets the i and j cache pointers
        const auto [startOnMesh, d] = onMesh(x, y);
        if (startOnMesh) { 
            
            // Yes, starting point is exactly on a spine or rib
            intList.push_back(d);
    
            // Are we on a spine or a rib?
            if (d.t == IntersectionType::rib)
            {

                // Set flags
                lastIntersectLeft = false;
                lastIntersectRight = false;

                // Adjust index to handle degenerate tracks
                while (y == y_[jSave_+1])
                {
                    ++jSave_;
                    if (jSave_ == ny()-2) { break; }
                }

                // Handle the case where the intersection point is a cell corner
                if (x == x_[iSave_,jSave_])
                {
                    if (m_[iSave_,0] > 0)
                    {
                        lastIntersectLeft = true;
                        iSave_--;
                    }
                    else 
                    {
                        lastIntersectRight = true;
                    }
                }

            } 
            else
            {

                // Spine

                // Set intersection flags
                if (m_[iSave_,jSave_] == bigNum ||
                    m_[iSave_,jSave_] <= 0)
                {
                    lastIntersectLeft = false;
                    lastIntersectRight = true;
                } 
                else
                {
                    lastIntersectLeft = true;
                    lastIntersectRight = false;
                    iSave_--;
                }

            }
    
        } 
        else
        {

            // We're not starting on a rib or spine; set intersection flags
            lastIntersectLeft = false;
            lastIntersectRight = false;

        }

    }

    // Helper to start segment traversal on mesh edge
    auto Mesh2DGrid::xIntersectSegStartEdge( //NOLINT(readability-function-cognitive-complexity)
        const double x,
        double& y,
        const double yMin,
        std::vector<xIntersectionDescriptor>& intList,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const -> bool
    {
        // Determine is starting point is on the bottom of the mesh
        if (x >= x_[0,0] && x <= x_[nx()-1,0] && 
            yMin <= y_[0])
        {

            // Starting point is on the bottom rib
            iSave_ = xIdx(x, 0);
            jSave_ = 0;
            lastIntersectLeft = false;
            lastIntersectRight = false;

            // Add intersection point to output holder
            intList.push_back({ 
                .y = y,
                .xs = x,
                .t = IntersectionType::rib,
                .idx = 0,
                .meshExit = false
            });
    
            // Handle degenerate tracks
            while (y_[jSave_] == y_[jSave_+1]) { ++jSave_; }

            // Handle the case where the starting point is a cell corner
            if (x == x_[iSave_,jSave_])
            {
                if (m_[iSave_,jSave_] > 0)
                {
                    lastIntersectLeft = true;
                    if (iSave_ == 0) { return true; }
                    iSave_--;
                }
                else
                {
                    lastIntersectRight = true;
                }
            }
    
        } 
        else
        {

            // Starting point is not on the bottom rib, so it on one of the
            // side spines
            jSave_ = ySearch(y, 0, ny()-1);
            if (x <= x_[0,jSave_]) { iSave_ = 0; }
            else { iSave_ = nx()-1; }

            // Add intersection point to output holder
            const double s = s_[iSave_,jSave_] +
                std::sqrt(std::pow(x - x_[iSave_,jSave_],2) +
                    std::pow(y - y_[jSave_], 2));
            intList.push_back({ 
                .y = y, 
                .xs = s,
                .t = IntersectionType::spine,
                .idx = iSave_,
                .meshExit = false
            });

            // Set intersection flags, and adjust i index if necessary; be
            // careful to handle degenerate edge tracks correctly
            if (iSave_ == 0)
            {
                while (x_[iSave_,jSave_] == x_[iSave_+1,jSave_] &&
                    x_[iSave_,jSave_+1] == x_[iSave_+1,jSave_+1] &&
                    m_[iSave_,jSave_] < 0) { iSave_++; }
                lastIntersectLeft = false;
                lastIntersectRight = true;
            } 
            else
            {
                iSave_--;
                while (x_[iSave_,jSave_] == x_[iSave_+1,jSave_] &&
                    x_[iSave_,jSave_+1] == x_[iSave_+1,jSave_+1] &&
                    m_[iSave_,jSave_] > 0) { iSave_--; }
                lastIntersectLeft = true;
                lastIntersectRight = false;
            }      

        }

        return false; // If we're here, we are not in the tangent case
    }


    // Helper function to find distance to next intersection
    auto Mesh2DGrid::findIntersectDistance(const double x,
            const double y,
            const bool searchUp,
            const bool lastIntersectLeft,
            const bool lastIntersectRight) const ->
            std::tuple<double, double, double>
    {
        // Output quantities
        double dyY = bigNum;
        double dyL = bigNum;
        double dyR = bigNum;

        // Handle sign flips for up versus down search
        const double dir = searchUp ? 1.0 : -1.0;

        // Distance to rib
        dyY = dir * (y_[jSave_+1] - y);

        // Distance to left spine
        if (m_[iSave_,jSave_] != bigNum && !lastIntersectRight)
        {
            dyL = dir * (y_[jSave_] +
	            (m_[iSave_,jSave_] * (x - x_[iSave_,jSave_]))
                - y
            );
            if (dyL < 0.0) { dyL = bigNum; }
        } 

        // Distance to right spine
        if (m_[iSave_+1,jSave_] != bigNum && !lastIntersectLeft)
        {
            dyR = dir * (y_[jSave_] +
	            (m_[iSave_+1,jSave_] * (x - x_[iSave_+1,jSave_])) 
                - y
            );
            if (dyR < 0.0) { dyR = bigNum; }
        }

        // Return results
        return std::make_tuple(dyY, dyL, dyR);

    }

    // Helper function to find next intersection for a traversal
    // of the mesh at constant x
    auto Mesh2DGrid::findNextIntersect(
            const double x,
            double& y,
            const double yStop,
            const bool endInterior,
            bool& lastIntersectLeft,
            bool& lastIntersectRight
        ) const -> std::pair<bool, xIntersectionDescriptor>
    {

        // Flag if we're searching upward (y increasing) or downward
        const bool searchUp = yStop > y;

        // Get vertical distances to the horizontal edge and the
        // left and right vertical edges of this cell
        const auto [dyY, dyL, dyR] =
            findIntersectDistance(x, y, searchUp, 
                lastIntersectLeft, 
                lastIntersectRight);

        // See which cell edge we hit
        if (dyY <= dyL && dyY <= dyR)
        {
            // We hit a rib
            if (y < yStop)
            {
                // Top rib
                return handleNextIntersectTopRib(
                    y, x, yStop,
                    endInterior, 
                    lastIntersectLeft, 
                    lastIntersectRight);
            }

            // Bottom rib
            return handleNextIntersectBottomRib(
                y, x, yStop,
                endInterior, 
                lastIntersectLeft, 
                lastIntersectRight);
        } 

        // If we're heree, we hit a spine
        if (dyL < dyR)
        {
            // Left spine
            return handleNextIntersectLeftSpine(
                y, x, yStop, dyL,
                endInterior,
                lastIntersectLeft,
                lastIntersectRight);
        }

        // Right spine
        return handleNextIntersectRightSpine(
            y, x, yStop, dyR,
            endInterior,
            lastIntersectLeft,
            lastIntersectRight);

    }

    // Handler for if the intersection point is in the +y direction
    auto Mesh2DGrid::handleNextIntersectTopRib(double& y,
        const double x,
        const double yStop,
        const bool endInterior,
        bool& lastIntersectLeft,
        bool& lastIntersectRight)
    const -> std::pair<bool, xIntersectionDescriptor>
    {
        // Create empty intersection point object
        xIntersectionDescriptor d = {
            .y = NAN, 
            .xs = NAN, 
            .t = IntersectionType::none,
            .idx = 0,
            .meshExit = false };

        // Update position
        y = y_[jSave_+1];

        // Check termination condition
        if (y > yStop && endInterior) {
            return std::make_pair(false, d);
        }

        // Update the index
        while (true) {
            jSave_++;
            if (jSave_ == ny()-1 || 
                y_[jSave_+1] != y_[jSave_]) { break; }
        }

        // Record the hit
        d.y = y;
        d.xs = x;
        d.t = IntersectionType::rib;
        d.idx = jSave_;

        // Check if we have hit the top of the mesh
        if (jSave_ == ny()-1)
        {
            jSave_--;
            d.meshExit = true;
            return std::make_pair(false, d);
        } 
        d.meshExit = false;

        // Set flags
        lastIntersectLeft = false;
        lastIntersectRight = false;

        // Check for corner cases
        const bool meshExit = handleTopRibCornerCases(x,
            lastIntersectLeft,
            lastIntersectRight);
        if (meshExit)
        {
            // Flag if we exited mesh
            d.meshExit = true;
            return std::make_pair(false, d);
        }

        // If we have gotten to here, we are still inside the mesh and not
        // at the end point
        return std::make_pair(true, d);
    
    }

    // Handler for if the intersection point is in the -y direction
    auto Mesh2DGrid::handleNextIntersectBottomRib(double& y,
        const double x,
        const double yStop,
        const bool endInterior,
        bool& lastIntersectLeft,
        bool& lastIntersectRight)
    const -> std::pair<bool, xIntersectionDescriptor>
    {
        // Create empty intersection point object
        xIntersectionDescriptor d = {
            .y = NAN, 
            .xs = NAN, 
            .t = IntersectionType::none,
            .idx = 0,
            .meshExit = false };

        // Update position
        y = y_[jSave_];

        // Check termination condition
        if (y < yStop && endInterior)
        {
            return std::make_pair(false, d);
        }

        // Record the hit
        d.y = y;
        d.xs = x;
        d.t = IntersectionType::rib;
        d.idx = jSave_;

        // Stop if we have hit the bottom of the mesh
        if (y == y_[0])
        {
            d.meshExit = true;
            return std::make_pair(false, d);
        }

        // Update the index
        while (true)
        {
            jSave_--;
            if (jSave_ == 0 || 
                y_[jSave_-1] != y_[jSave_]) { break; }
        }

        // Set flags
        lastIntersectLeft = false;
        lastIntersectRight = false;

        // Check for corner cases
        const bool meshExit = handleBottomRibCornerCases(x,
            lastIntersectLeft,
            lastIntersectRight);
        if (meshExit)
        {
            // Flag if we exited mesh
            d.meshExit = true;
            return std::make_pair(false, d);
        }

        // If we have gotten to here, we are still inside the mesh and not
        // at the end point
        return std::make_pair(true, d);
    
    }


    // Handle corner cases for the top rib
    auto Mesh2DGrid::handleTopRibCornerCases( //NOLINT(readability-function-cognitive-complexity)
        const double x,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const -> bool
    {
        // Handle corner cases:
        //
        // Case 1: we hit the upper left corner of the cell and the
        //   slope above us is positive -- in this case we're crossing
        //   both a horizontal and a vertical track, and we need to update
        //   the i index as well as the j index. In the diagram below,
        //   the x marks the starting position, and the * marks the
        //   corner we hit.
        //
        //         /    /
        //        *----/
        //       /    /
        //      /-x--/
        //
        // Case 2: we hit the upper left corner of the cell and the
        //   slope above us is negative; this only happens if the slope
        //   changes across the intersection point, and in this case we
        //   do not cross the vertical track boundary, we are just
        //   tangent to it. The diagram is:
        //
        //       \     \
        //        *----/
        //       /    /
        //      /-x--/
        //
        // Cases 3 and 4 are the same, except that they involve hitting
        // the upper right corner instead of the upper left one.
        //
        // Depending on which case we are in, we may or may not be
        // exiting the grid, we may or may not need to update the i
        // index, and we may or may not need to flag that we have just
        // crossed a particular i track.
        //
        if (x == x_[iSave_,jSave_])
        {
            // Case 1 or 2: hitting the upper left corner
            if (m_[iSave_,jSave_] > 0 &&
                !(m_[iSave_,jSave_] == bigNum))
            {
                // Case 1: hitting upper left corner, crossing track; set
                // intersection flag, and check for exiting grid
                lastIntersectLeft = true;
                if (iSave_ == 0)
                {
                    return true;
                }
                while (x == x_[iSave_,jSave_])
                {
                    iSave_--;
                    if (iSave_ == 0) { break; }
                }
                if (iSave_ == 0 && x == x_[iSave_,jSave_])
                {
                    return true;
                } 
            }
            else
            {
                // Case 2: hitting upper left corner but tangent, so not
                // crossing track; just set intersection flag
                lastIntersectRight = true;
            }
        } 
        else if (x == x_[iSave_+1,jSave_])
        {
            // Case 3 or 4: hitting upper right corner
            if (m_[iSave_+1,jSave_] < 0)
            {
                // Case 3: hitting upper right corner, crossing track
                lastIntersectRight = true;
                if (iSave_ == nx()-2)
                {
                    return true;
                }
                while (x == x_[iSave_+1,jSave_])
                {
                    if (iSave_ == nx()-2)
                    {
                        return true;
                    }
                    iSave_++;
                }
            } 
            else
            {
                // Case 4: hitting upper right corner, but tangent, not
                // crossing track
                lastIntersectLeft = true;
            }
        }

        // If we are here, we did not exit the mesh
        return false;

    }

    // Handle corner cases for the bottom rib
    auto Mesh2DGrid::handleBottomRibCornerCases( //NOLINT(readability-function-cognitive-complexity)
        const double x,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const -> bool
    {
        // Note that the cases here are exactly the same as in
        // handleTopRibCornerCases, just mirror-reversed
        if (x == x_[iSave_,jSave_+1])
        {
            // Case 1 or 2: hitting the lower left corner
            if (m_[iSave_,jSave_] < 0)
            {
                // Case 1: hitting lower left corner, crossing track; set
                // intersection flag, and check for exiting grid
                lastIntersectLeft = true;
                if (iSave_ == 0) {
                    return true;
                }
                while (x == x_[iSave_,jSave_+1])
                {
                    iSave_--;
                    if (iSave_ == 0) { break; }
                }
                if (iSave_ == 0 && x == x_[iSave_,jSave_+1])
                {
                    return true;
                }
            } 
            else 
            {
                // Case 2: hitting lower left corner but tangent, so not
                // crossing track; just set intersection flag
                lastIntersectRight = true;
            }
        } 
        else if (x == x_[iSave_+1,jSave_+1])
        {
            // Case 3 or 4: hitting lower right corner
            if (m_[iSave_+1,jSave_] > 0 &&
                !(m_[iSave_+1,jSave_] == bigNum))
            {
                // Case 3: hitting lower right corner, crossing track
                lastIntersectRight = true;
                if (iSave_ == nx()-1) {
                    return true;
                }
                while (x == x_[iSave_+1,jSave_+1])
                {
                    if (iSave_ == nx()-2) {
                        return true;
                    }
                    iSave_++;
                }
            } 
            else 
            {
                // Case 4: hitting lower right corner, but tangent, not
                // crossing track
                lastIntersectLeft = true;
            }

        }

        // If we are here, we did not exit the mesh
        return false;

    }


    // Handle left spine intersection case
    auto Mesh2DGrid::handleNextIntersectLeftSpine(
        double& y,
        const double x,
        const double yStop,
        const double dy,
        const bool endInterior,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const -> std::pair<bool, xIntersectionDescriptor>
    {
        // Create empty intersection point object
        xIntersectionDescriptor d = {
            .y = NAN, 
            .xs = NAN, 
            .t = IntersectionType::none,
            .idx = 0,
            .meshExit = false };

        // Get search direction
        const bool searchUp = yStop > y;

        // Update position
        y = y_[jSave_] +
            (m_[iSave_,jSave_] * (x - x_[iSave_,jSave_]));

        // Check termination condition
        if (endInterior) {
            if ((y > yStop && searchUp) ||
                (y < yStop && !searchUp))
            {
                return std::make_pair(false, d);
            }
        }

        // Record the hit
        const double s = s_[iSave_,jSave_] +
            std::sqrt( 
                std::pow(x - x_[iSave_,jSave_], 2) +
                std::pow(dy, 2));
        d.y = y;
        d.xs = s;
        d.t = IntersectionType::spine;
        d.idx = iSave_;

        // Stop if we are at mesh edge
        if (iSave_ == 0)
        {
            d.meshExit = true;
            return std::make_pair(false, d);
        }
        d.meshExit = false;

        // Update index
        while (true)
        {
            iSave_--;
            if (iSave_ == 0 ||
                (x_[iSave_,jSave_] != x_[iSave_+1,jSave_]) ||
                (x_[iSave_,jSave_+1] != x_[iSave_+1,jSave_+1]))
            { 
                break; 
            }
        }

        // Stop if we are at mesh edge; this second check is needed to
        // handle the case where the mesh left edge is degenerate, so we
        // may have hit the edge even though our i index wasn't 0 to
        // start
        if (iSave_ == 0 &&
            x_[0,jSave_] == x_[1,jSave_] &&
            x_[0,jSave_+1] == x_[1,jSave_+1])
        {
            d.meshExit = true;
            return std::make_pair(false, d);
        }

        // Set flags
        lastIntersectLeft = true;
        lastIntersectRight = false;


        // If we have gotten to here, we are still inside the mesh and not
        // at the end point
        return std::make_pair(true, d);

    }


    // Handle right spine intersection case
    auto Mesh2DGrid::handleNextIntersectRightSpine(
        double& y,
        const double x,
        const double yStop,
        const double dy,
        const bool endInterior,
        bool& lastIntersectLeft,
        bool& lastIntersectRight
    ) const -> std::pair<bool, xIntersectionDescriptor>
    {
        // Create empty intersection point object
        xIntersectionDescriptor d = {
            .y = NAN, 
            .xs = NAN, 
            .t = IntersectionType::none,
            .idx = 0,
            .meshExit = false };

        // Get search direction
        const bool searchUp = yStop > y;

        // Update position
        y = y_[jSave_] +
            (m_[iSave_+1,jSave_] * (x - x_[iSave_+1,jSave_]));

        // Check termination condition
        if (endInterior)
        {
            if ((y > yStop && searchUp) ||
                (y < yStop && !searchUp))
            {
                return std::make_pair(false, d);
            }
        }
    
        // Record the hit
        const double s = s_[iSave_+1,jSave_] +
            std::sqrt( 
                std::pow(x - x_[iSave_+1,jSave_], 2) +
                std::pow(dy, 2));
        d.y = y;
        d.xs = s;
        d.t = IntersectionType::spine;
        d.idx = iSave_ + 1;

        // Check if we have exited the mesh
        if (iSave_ == nx()-2)
        {
            d.meshExit = true;
            return std::make_pair(false, d);
        }
        d.meshExit = false;

        // Update the index
        while (true) {
            iSave_++;
            if (iSave_ == nx()-1 ||
                (x_[iSave_,jSave_] != x_[iSave_+1,jSave_]) ||
                (x_[iSave_,jSave_+1] != x_[iSave_+1,jSave_+1]))
            {
                break;
            }
        }

        // Second check to see if we have exited the mesh; needed in
        // case the right edge is degenerate
        if (iSave_ == nx()-1)
        {
            iSave_--;
            d.meshExit = true;
            return std::make_pair(false, d);
        }

        // Set flags
        lastIntersectLeft = false;
        lastIntersectRight = true;

        // If we have gotten to here, we are still inside the mesh and not
        // at the end point
        return std::make_pair(true, d);

    }


    // Find mesh intersections at constant y
    auto Mesh2DGrid::yIntersect(
        const double y,
        const double xLo,
        const double xHi
    ) const -> std::vector<yIntersectionDescriptor>
    {
        std::vector<yIntersectionDescriptor> intList; // Output holder

        // Handle trivial case where inputs miss the mesh entirely
        if (y < yMin_ || y > yMax_) { return intList; }
        const double xMinY = xMin(y);
        const double xMaxY = xMax(y);
        if (xHi < xMinY || xLo > xMaxY) { return intList; }

        // Get y index and offset of input y value
        jSave_ = yIdx(y);
        const double dy = y - y_[jSave_];

        // Get starting point for traversal
        double x = std::max(xLo, xMinY);

        // Check if starting position is exactly on a vertical spine,
        // and store the point if it is
        if (x == xMinY) 
        {
            // Case where we start at mesh left edge
            iSave_ = 0;
            const double dx = x - x_[iSave_,jSave_];
            intList.push_back({
                .x = x, 
                .s = s_[iSave_, jSave_] +
                    std::sqrt(std::pow(dx, 2) + std::pow(dy, 2)),
                .idx = 0
            });
        }
        else
        {
            // Case where we start on mesh interior; note that
            // the call to onMesh here automatically caches the
            // i and j indices of the starting cell in iSave_
            // and jSave_
            auto [onSpine, d] = onMesh(x, y);
            if (onSpine)
            {
                // If starting point is exactly on spine, store it
                intList.push_back({
                    .x = x,
                    .s = d.xs,
                    .idx = d.idx
                });
            }
        }

        // Now traverse mesh
        while (iSave_ < nx() - 1)
        {

            // Move to right edge of cell
            const double dx = dy / m_[iSave_+1, jSave_];
            x = x_[iSave_+1, jSave_] + dx;

            // Enforce maximum x
            if (x > xHi) { return intList; }

            // Save intersection point
            intList.push_back({
                .x = x,
                .s = s_[iSave_+1, jSave_] +
                    std::sqrt(std::pow(dx,2) + std::pow(dy,2)),
                .idx = iSave_ + 1
            });

            // Move index, being careful to handle degenerate tracks
            while (true)
            {
                iSave_++;
                if (iSave_ == nx() - 1) { break; }
                if (x_[iSave_,jSave_] != x_[iSave_+1,jSave_] ||
                    x_[iSave_+1,jSave_] != x_[iSave_+1,jSave_]) { break; }
            }

        }

        // Return final list
        return intList;

    }

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
