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
#include <limits>
#include <mdspan>
#include <stdexcept>
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
            if (x[0,j] == x[x.extent(1)-1,j]) {
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
                    m_[i,j] = std::numeric_limits<double>::max();
                }
            }
        }

        // Compute spine segment lengths
        sData_.resize(nx() * ny());
        s_ = Array2D(sData_.data(), nx(), ny());
        for (size_t j = 0; j < ny(); ++j) {
            for (size_t i = 0; i < nx(); ++i) {
                s_[i,j] = std::sqrt(
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
            if (m_[0,j] != std::numeric_limits<double>::max()) {
                if (leftSlopePos && m_[0,j] < 0) { convex_ = false; }
                if (m_[0,j] > 0) { leftSlopePos = true; }
            }
            if (m_[nx()-1,j] != std::numeric_limits<double>::max()) {
                if (rightSlopeNeg && m_[nx()-1,j] > 0) { convex_ = false; }
                if (m_[nx()-1,j] < 0) { rightSlopeNeg = true; }
            }
            if (!convex_) { break; }
        }
    }

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
