/**
 * @file Mesh3DInterpolator.hpp
 * @author Mark Krumholz
 * @brief Machinery to do interpolation on a 3D mesh
 * @date 2024-06-19
 * @details
 * A class to interpolate on a 3D semi-tensor mesh
 */

#ifndef MESH3DINTERPOLATOR_HPP
#define MESH3DINTERPOLATOR_HPP

#include "Interpolator1D.hpp"
#include "Mesh2DInterpolator.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_interp.h>
#include <limits>
#include <mdspan>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif

// Disable linting for array bounds checking in this
// file, since the overhead associated with enforcing
// such checks severely interferes with performance
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

namespace interp
{

    /**
     * @class Mesh3DInterpolator
     * @tparam NF Number of quantities to interpolate
     * @brief A class to perform interpolation on a 3D mesh
     * @details
     * This class performs interpolation on a 3D mesh where the y and z
     * coordinates are arranged in a tensor grid (one y value per y-plane,
     * one z value per z-plane), while the x coordinates may vary at each
     * (y, z) position. The function values and x coordinates are copied
     * into internal storage on construction. The number of quantities to
     * be interpolated is determined by the template parameter NF.
     */
    template <size_t NF = 1>
    class Mesh3DInterpolator
    {
    public:

        // Shorten array types
        using Array4D = std::mdspan<double, std::dextents<size_t, 4>>;
        using Array3D = std::mdspan<double, std::dextents<size_t, 3>>;
        using Array1D = std::mdspan<double, std::dextents<size_t, 1>>;

        // Constructors and destructor
        /**
         * @brief Construct a Mesh3DInterpolator
         * @param x A 3D array (Nx × Ny × Nz) giving the x coordinates of the mesh points
         * @param y A 1D array (Ny) giving the y coordinates of the mesh points
         * @param z A 1D array (Nz) giving the z coordinates of the mesh points
         * @param f A 4D array (Nx × Ny × Nz × NF) giving the values of the
         *          quantities to be interpolated
         * @param interpType The type of interpolation to use
         * @param monotonic Enforce that interpolation is monotonicity-preserving
         * @details
         * If monotonic is set to true, the interpolation type will always be
         * gsl_interp_linear regardless of the value of interpType.
         */
        Mesh3DInterpolator(
            const Array3D& x,
            const Array1D& y,
            const Array1D& z,
            const Array4D& f,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) :
        monotonic_(monotonic),
        interpType_(monotonic ? gsl_interp_linear : interpType),
        nx_(x.extent(0)),
        ny_(x.extent(1)),
        nz_(x.extent(2))
        {
            if (x.extent(1) != y.extent(0) || x.extent(2) != z.extent(0))
            {
                throw std::runtime_error(
                    "Mesh3DInterpolator: y and z sizes must match the "
                    "second and third dimensions of x");
            }
            if (f.extent(0) != x.extent(0) ||
                f.extent(1) != x.extent(1) ||
                f.extent(2) != x.extent(2) ||
                f.extent(3) != NF)
            {
                throw std::runtime_error(
                    "Mesh3DInterpolator: x and f must have the same leading "
                    "three dimensions, and the fourth dimension of f must "
                    "match the number of interpolated quantities");
            }
            if (x.extent(0) < 2 ||
                (x.extent(1) < 2 && x.extent(2) < 2))
            {
                throw std::runtime_error(
                    "Mesh3DInterpolator: mesh size nx x ny x nz must obey "
                    "nx > 1 and either ny > 1 or nz > 1"
                );
            }
            copyMeshData(x, y, z, f);
            computeLengths();
            computeInterpolators();
        }

        /**
         * @brief Construct a Mesh3DInterpolator for a single quantity (NF == 1)
         * @param x A 3D array (Nx × Ny × Nz) giving the x coordinates of the mesh points
         * @param y A 1D array (Ny) giving the y coordinates of the mesh points
         * @param z A 1D array (Nz) giving the z coordinates of the mesh points
         * @param f A 3D array (Nx × Ny × Nz) giving the values of the quantity
         *          to be interpolated
         * @param interpType The type of interpolation to use
         * @param monotonic Enforce that interpolation is monotonicity-preserving
         * @details
         * This constructor is valid only if NF == 1. It wraps the 3D array f
         * in a 4D view with a trailing extent of 1 before delegating to the
         * primary constructor.
         */
        Mesh3DInterpolator(
            const Array3D& x,
            const Array1D& y,
            const Array1D& z,
            const Array3D& f,
            const gsl_interp_type* interpType = gsl_interp_steffen,
            const bool monotonic = false
        ) requires (NF == 1) :
        Mesh3DInterpolator(x, y, z,
            std::mdspan(f.data_handle(), f.extent(0), f.extent(1), f.extent(2), 1),
            interpType,
            monotonic)
        { }

        virtual ~Mesh3DInterpolator() = default;

        // Disallow copy constructor due to the unique_ptrs we use
        // internally, but allow move constructor
        Mesh3DInterpolator(const Mesh3DInterpolator&) = delete;
        Mesh3DInterpolator(Mesh3DInterpolator&&) = default;
        auto operator=(const Mesh3DInterpolator&) -> Mesh3DInterpolator& = delete;
        auto operator=(Mesh3DInterpolator&&) noexcept -> Mesh3DInterpolator& = default;

        // Observers
        /**
         * @brief Return total number of mesh points
         * @returns nx() * ny() * nz()
         */
        [[nodiscard]] auto size() const { return nx_ * ny_ * nz_; }

        /**
         * @brief Return number of mesh points in the x direction
         * @returns Number of mesh points in the x direction
         */
        [[nodiscard]] auto nx() const { return nx_; }

        /**
         * @brief Return number of mesh points in the y direction
         * @returns Number of mesh points in the y direction
         */
        [[nodiscard]] auto ny() const { return ny_; }

        /**
         * @brief Return number of mesh points in the z direction
         * @returns Number of mesh points in the z direction
         */
        [[nodiscard]] auto nz() const { return nz_; }

        /**
         * @brief Return an mdspan view of the x coordinate data
         * @returns x as an mdspan of shape (Nx, Ny, Nz)
         */
        [[nodiscard]] auto x() const
        {
            return std::mdspan<const double, std::dextents<size_t, 3>>(
                xData_.data(), nx_, ny_, nz_);
        }

        /**
         * @brief Return an mdspan view of the y coordinate data
         * @returns y as an mdspan of shape (Ny)
         */
        [[nodiscard]] auto y() const
        {
            return std::mdspan<const double, std::dextents<size_t, 1>>(
                yData_.data(), ny_);
        }

        /**
         * @brief Return an mdspan view of the z coordinate data
         * @returns z as an mdspan of shape (Nz)
         */
        [[nodiscard]] auto z() const
        {
            return std::mdspan<const double, std::dextents<size_t, 1>>(
                zData_.data(), nz_);
        }

        /**
         * @brief Return an mdspan view of the function value data
         * @returns f as an mdspan of shape (Nx, Ny, Nz, NF)
         */
        [[nodiscard]] auto f() const
        {
            return std::mdspan<const double, std::dextents<size_t, 4>>(
                fData_.data(), nx_, ny_, nz_, NF);
        }

        /**
         * @brief Minimum x value across the entire mesh
         * @returns Minimum x value in mesh
         */
        [[nodiscard]] auto xMin() const { return *std::ranges::min_element(xData_); }

        /**
         * @brief Maximum x value across the entire mesh
         * @returns Maximum x value in mesh
         */
        [[nodiscard]] auto xMax() const { return *std::ranges::max_element(xData_); }

        /**
         * @brief Minimum y value in mesh
         * @returns Minimum y value in mesh
         */
        [[nodiscard]] auto yMin() const { return yData_.front(); }

        /**
         * @brief Maximum y value in mesh
         * @returns Maximum y value in mesh
         */
        [[nodiscard]] auto yMax() const { return yData_.back(); }

        /**
         * @brief Minimum z value in mesh
         * @returns Minimum z value in mesh
         */
        [[nodiscard]] auto zMin() const { return zData_.front(); }

        /**
         * @brief Maximum z value in mesh
         * @returns Maximum z value in mesh
         */
        [[nodiscard]] auto zMax() const { return zData_.back(); }

        /**
         * @brief Return an mdspan view of the y arc-lengths
         * @returns sy as an mdspan of shape (Nx, Nz, Ny); sy[i,k,j] is the
         *          cumulative arc-length along the y direction from j=0 to j,
         *          at mesh position (i, k) in (x, z)
         */
        [[nodiscard]] auto sy() const
        {
            return std::mdspan<const double, std::dextents<size_t, 3>>(
                syData_.data(), nx_, nz_, ny_);
        }

        /**
         * @brief Return an mdspan view of the z arc-lengths
         * @returns sz as an mdspan of shape (Nx, Ny, Nz); sz[i,j,k] is the
         *          cumulative arc-length along the z direction from k=0 to k,
         *          at mesh position (i, j) in (x, y)
         */
        [[nodiscard]] auto sz() const
        {
            return std::mdspan<const double, std::dextents<size_t, 3>>(
                szData_.data(), nx_, ny_, nz_);
        }

        /**
         * @brief Return an mdspan view of the y-direction interpolators
         * @returns yInterp as an mdspan of shape (Nx, Nz); yInterp[i,k] is
         *          the Interpolator1D<NF> that interpolates f in the y
         *          direction at fixed (i,k), using sy as the interpolation
         *          variable
         */
        [[nodiscard]] auto yInterp() const
        {
            return std::mdspan<const std::unique_ptr<Interpolator1D<NF>>, std::dextents<size_t, 2>>(
                yInterpData_.data(), nx_, nz_);
        }

        /**
         * @brief Return an mdspan view of the z-direction interpolators
         * @returns zInterp as an mdspan of shape (Nx, Ny); zInterp[i,j] is
         *          the Interpolator1D<NF> that interpolates f in the z
         *          direction at fixed (i,j), using sz as the interpolation
         *          variable
         */
        [[nodiscard]] auto zInterp() const
        {
            return std::mdspan<const std::unique_ptr<Interpolator1D<NF>>, std::dextents<size_t, 2>>(
                zInterpData_.data(), nx_, ny_);
        }

        /**
         * @brief Construct a 2D slice of the mesh at fixed y
         * @param y0 The y coordinate at which to slice
         * @returns A const reference to a Mesh2DInterpolator<NF>
         *          representing the (x, z) slice of the mesh at y = y0
         * @details
         * If y0 exactly matches one of the mesh y values, the slice is
         * built directly from the corresponding x and f mesh points.
         * Otherwise, the x coordinates of the slice are found by linear
         * interpolation between the two bracketing y planes, and the
         * function values are found using the y-direction interpolators
         * returned by yInterp(), evaluated at the sy arc-length coordinate
         * that corresponds to y0 in each (i,k) column.
         *
         * The constructed Mesh2DInterpolator is cached, along with the
         * value of y0 used to build it; if this function is called again
         * with the same y0, the cached object is returned directly rather
         * than being rebuilt. The reference returned remains valid only
         * until the next call to sliceConstY with a different y0, or
         * until this Mesh3DInterpolator is destroyed. Use
         * sliceConstYCopy() instead if you need an independent,
         * caller-owned object.
         *
         * This method is NOT thread-safe and must never be called from
         * inside an openMP parallel region: it writes to the shared
         * ySliceCache_/ySliceCacheVal_ cache, and two threads racing to
         * populate that cache could corrupt it or return a slice built
         * for the wrong y0. In a build with openMP enabled, calling
         * this from inside a parallel region trips an assertion in
         * debug builds; in release builds it is simply undefined
         * behavior. The standard usage pattern -- building the cache
         * once at startup before any parallel region for single-[Fe/H]
         * runs, and using sliceConstYCopy() instead whenever multiple
         * [Fe/H] values are in play -- never needs this method inside a
         * parallel region in the first place. The cache was not made
         * thread-private (unlike Mesh2DGrid's iSave_/jSave_ or
         * Interpolator1D's acc_) because each cached slice can be tens
         * of megabytes, and giving every thread its own copy would be
         * expensive on a many-core node.
         */
        [[nodiscard]] auto sliceConstY(double y0) const -> const Mesh2DInterpolator<NF>&
        {
#ifdef _OPENMP
            assert(!omp_in_parallel());
#endif
            // Return the cached slice if it was built for this y0
            if (ySliceCache_ && ySliceCacheVal_ == y0) { return *ySliceCache_; }

            ySliceCache_ = std::make_unique<Mesh2DInterpolator<NF>>(buildSliceY(y0));
            ySliceCacheVal_ = y0;
            return *ySliceCache_;
        }

        /**
         * @brief Construct a new, independent 2D slice of the mesh at fixed y
         * @param y0 The y coordinate at which to slice
         * @returns A newly-constructed Mesh2DInterpolator<NF> representing
         *          the (x, z) slice of the mesh at y = y0
         * @details
         * Identical to sliceConstY() in every respect except that the
         * result is not cached: a new Mesh2DInterpolator is built on
         * every call and returned by value, for use in situations where
         * the caller needs to own the slice independently of this
         * Mesh3DInterpolator (e.g. to keep it alive after later calls
         * to sliceConstY() with a different y0).
         */
        [[nodiscard]] auto sliceConstYCopy(double y0) const -> Mesh2DInterpolator<NF>
        {
            return buildSliceY(y0);
        }

        /**
         * @brief Construct a 2D slice of the mesh at fixed z
         * @param z0 The z coordinate at which to slice
         * @returns A const reference to a Mesh2DInterpolator<NF>
         *          representing the (x, y) slice of the mesh at z = z0
         * @details
         * If z0 exactly matches one of the mesh z values, the slice is
         * built directly from the corresponding x and f mesh points.
         * Otherwise, the x coordinates of the slice are found by linear
         * interpolation between the two bracketing z planes, and the
         * function values are found using the z-direction interpolators
         * returned by zInterp(), evaluated at the sz arc-length coordinate
         * that corresponds to z0 in each (i,j) column.
         *
         * The constructed Mesh2DInterpolator is cached, along with the
         * value of z0 used to build it; if this function is called again
         * with the same z0, the cached object is returned directly rather
         * than being rebuilt. The reference returned remains valid only
         * until the next call to sliceConstZ with a different z0, or
         * until this Mesh3DInterpolator is destroyed. Use
         * sliceConstZCopy() instead if you need an independent,
         * caller-owned object.
         *
         * This method is NOT thread-safe and must never be called from
         * inside an openMP parallel region: it writes to the shared
         * zSliceCache_/zSliceCacheVal_ cache, and two threads racing to
         * populate that cache could corrupt it or return a slice built
         * for the wrong z0. In a build with openMP enabled, calling
         * this from inside a parallel region trips an assertion in
         * debug builds; in release builds it is simply undefined
         * behavior. The standard usage pattern -- building the cache
         * once at startup before any parallel region for single-[Fe/H]
         * runs, and using sliceConstZCopy() instead whenever multiple
         * [Fe/H] values are in play -- never needs this method inside a
         * parallel region in the first place. The cache was not made
         * thread-private (unlike Mesh2DGrid's iSave_/jSave_ or
         * Interpolator1D's acc_) because each cached slice can be tens
         * of megabytes, and giving every thread its own copy would be
         * expensive on a many-core node.
         */
        [[nodiscard]] auto sliceConstZ(double z0) const -> const Mesh2DInterpolator<NF>&
        {
#ifdef _OPENMP
            assert(!omp_in_parallel());
#endif
            // Return the cached slice if it was built for this z0
            if (zSliceCache_ && zSliceCacheVal_ == z0) { return *zSliceCache_; }

            zSliceCache_ = std::make_unique<Mesh2DInterpolator<NF>>(buildSliceZ(z0));
            zSliceCacheVal_ = z0;
            return *zSliceCache_;
        }

        /**
         * @brief Construct a new, independent 2D slice of the mesh at fixed z
         * @param z0 The z coordinate at which to slice
         * @returns A newly-constructed Mesh2DInterpolator<NF> representing
         *          the (x, y) slice of the mesh at z = z0
         * @details
         * Identical to sliceConstZ() in every respect except that the
         * result is not cached: a new Mesh2DInterpolator is built on
         * every call and returned by value, for use in situations where
         * the caller needs to own the slice independently of this
         * Mesh3DInterpolator (e.g. to keep it alive after later calls
         * to sliceConstZ() with a different z0).
         */
        [[nodiscard]] auto sliceConstZCopy(double z0) const -> Mesh2DInterpolator<NF>
        {
            return buildSliceZ(z0);
        }

    private:

        bool monotonic_;                    /**< True if interpolation is monotonicity-preserving */
        const gsl_interp_type* interpType_; /**< Type of interpolation used */
        size_t nx_;                         /**< Number of mesh points in x direction */
        size_t ny_;                         /**< Number of mesh points in y direction */
        size_t nz_;                         /**< Number of mesh points in z direction */
        std::vector<double> xData_;         /**< x coordinates of mesh points (Nx × Ny × Nz, row-major) */
        std::vector<double> yData_;         /**< y coordinates of mesh points (length Ny) */
        std::vector<double> zData_;         /**< z coordinates of mesh points (length Nz) */
        std::vector<double> fData_;         /**< Function values at mesh points (Nx × Ny × Nz × NF, row-major) */
        std::vector<double> syData_;        /**< y arc-lengths (Nx × Nz × Ny, row-major) */
        std::vector<double> szData_;        /**< z arc-lengths (Nx × Ny × Nz, row-major) */

        // Interpolators are stored as vectors of unique_ptr rather than
        // vectors of Interpolator1D because we need to be sure to invoke
        // the destructor of Interpolator1D objects once only to avoid
        // memory issues with gsl opaque objects
        std::vector<
            std::unique_ptr<Interpolator1D<NF>>
        > yInterpData_;                     /**< y-direction interpolators (Nx × Nz, row-major) */
        std::vector<
            std::unique_ptr<Interpolator1D<NF>>
        > zInterpData_;                     /**< z-direction interpolators (Nx × Ny, row-major) */

        // Caches for the most recently constructed y and z slices, so
        // that repeated calls to sliceConstY/sliceConstZ with the same
        // y0/z0 do not need to rebuild the slice. Mutable because they
        // are lazily populated from const observer methods.
        mutable std::unique_ptr<Mesh2DInterpolator<NF>> ySliceCache_;  /**< Cached result of the last sliceConstY call */
        mutable std::unique_ptr<Mesh2DInterpolator<NF>> zSliceCache_;  /**< Cached result of the last sliceConstZ call */
        mutable double ySliceCacheVal_ =
            std::numeric_limits<double>::quiet_NaN();                 /**< y0 value for which ySliceCache_ was built */
        mutable double zSliceCacheVal_ =
            std::numeric_limits<double>::quiet_NaN();                 /**< z0 value for which zSliceCache_ was built */

        /**
         * @brief Copy the input coordinate and function-value data
         * @details
         * Copies element-by-element via explicit indices, rather than
         * copying a contiguous range of the underlying pointer, so that
         * this works correctly even if the input mdspans are non-contiguous.
         */
        auto copyMeshData(const Array3D& x, const Array1D& y,
            const Array1D& z, const Array4D& f) -> void
        {
            xData_.resize(nx_ * ny_ * nz_);
            auto xView = Array3D(xData_.data(), nx_, ny_, nz_);
            for (size_t i = 0; i < nx_; ++i)
            {
                for (size_t j = 0; j < ny_; ++j)
                {
                    for (size_t k = 0; k < nz_; ++k) { xView[i, j, k] = x[i, j, k]; }
                }
            }

            yData_.resize(ny_);
            auto yView = Array1D(yData_.data(), ny_);
            for (size_t j = 0; j < ny_; ++j) { yView[j] = y[j]; }

            zData_.resize(nz_);
            auto zView = Array1D(zData_.data(), nz_);
            for (size_t k = 0; k < nz_; ++k) { zView[k] = z[k]; }

            fData_.resize(nx_ * ny_ * nz_ * NF);
            auto fView = Array4D(fData_.data(), nx_, ny_, nz_, NF);
            for (size_t i = 0; i < nx_; ++i)
            {
                for (size_t j = 0; j < ny_; ++j)
                {
                    for (size_t k = 0; k < nz_; ++k)
                    {
                        for (size_t n = 0; n < NF; ++n) { fView[i, j, k, n] = f[i, j, k, n]; }
                    }
                }
            }
        }

        /**
         * @brief Compute cumulative arc-lengths along the y and z directions
         * @details
         * Fills syData_ with arc-lengths along y for each (i,k) pair,
         * stored as an (Nx × Nz × Ny) array, and szData_ with arc-lengths
         * along z for each (i,j) pair, stored as an (Nx × Ny × Nz) array.
         * The trailing dimension is always the direction of integration so
         * that sy[i,k,0] == sz[i,j,0] == 0 by definition.
         */
        auto computeLengths() -> void
        {
            auto xView = Array3D(xData_.data(), nx_, ny_, nz_);
            auto yView = Array1D(yData_.data(), ny_);
            auto zView = Array1D(zData_.data(), nz_);

            // sy: shape (Nx, Nz, Ny)
            if (ny_ > 1)
            {
                syData_.resize(nx_ * nz_ * ny_);
                auto syView = std::mdspan<double, std::dextents<size_t, 3>>(
                    syData_.data(), nx_, nz_, ny_);
                for (size_t i = 0; i < nx_; ++i)
                {
                    for (size_t k = 0; k < nz_; ++k)
                    {
                        syView[i, k, 0] = 0.0;
                        for (size_t j = 1; j < ny_; ++j)
                        {
                            syView[i, k, j] = syView[i, k, j - 1] + std::sqrt(
                                std::pow(xView[i, j, k] - xView[i, j - 1, k], 2) +
                                std::pow(yView[j] - yView[j - 1], 2));
                        }
                    }
                }
            }

            // sz: shape (Nx, Ny, Nz)
            if (nz_ > 1)
            {
                szData_.resize(nx_ * ny_ * nz_);
                auto szView = std::mdspan<double, std::dextents<size_t, 3>>(
                    szData_.data(), nx_, ny_, nz_);
                for (size_t i = 0; i < nx_; ++i)
                {
                    for (size_t j = 0; j < ny_; ++j)
                    {
                        szView[i, j, 0] = 0.0;
                        for (size_t k = 1; k < nz_; ++k)
                        {
                            szView[i, j, k] = szView[i, j, k - 1] + std::sqrt(
                                std::pow(xView[i, j, k] - xView[i, j, k - 1], 2) +
                                std::pow(zView[k] - zView[k - 1], 2));
                        }
                    }
                }
            }
        }

        /**
         * @brief Build one direction's per-column Interpolator1D objects
         * @param n1 Size of the first (outer) index of the output array
         * @param n2 Size of the second (outer) index of the output array
         * @param nPrimary Number of points along the interpolated direction
         * @param sView Arc-length mdspan of shape (n1, n2, nPrimary), giving
         *              the interpolation variable for each (a, b, p) triple
         * @param fAt Callable (a, b, p, n) -> double returning the n-th
         *            function value at the mesh point corresponding to (a, b, p)
         * @param out Output vector, resized to n1 * n2 and filled with one
         *            Interpolator1D per (a, b) pair, indexed as out[a, b]
         */
        template <typename SView, typename FAt>
        auto buildColumnInterpolators(size_t n1, size_t n2, size_t nPrimary,
            SView sView, FAt fAt,
            std::vector<std::unique_ptr<Interpolator1D<NF>>>& out) -> void
        {
            out.resize(n1 * n2);
            auto outView = std::mdspan<std::unique_ptr<Interpolator1D<NF>>, std::dextents<size_t, 2>>(
                out.data(), n1, n2);
            for (size_t a = 0; a < n1; ++a)
            {
                for (size_t b = 0; b < n2; ++b)
                {
                    std::vector<double> col(nPrimary);
                    std::array<std::vector<double>, NF> fCol;
                    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
                    for (size_t n = 0; n < NF; ++n) { fCol[n].resize(nPrimary); }
                    for (size_t p = 0; p < nPrimary; ++p)
                    {
                        col[p] = sView[a, b, p];
                        for (size_t n = 0; n < NF; ++n) { fCol[n][p] = fAt(a, b, p, n); }
                    }
                    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
                    outView[a, b] = std::make_unique<Interpolator1D<NF>>(col, fCol, interpType_);
                }
            }
        }

        /**
         * @brief Construct per-column Interpolator1D objects along y and z
         * @details
         * Fills yInterpData_ with one Interpolator1D<NF> for each (i,k)
         * pair, interpolating f in the y direction at that fixed x- and
         * z-index using sy as the interpolation variable, stored as an
         * (Nx × Nz) array. Fills zInterpData_ with one Interpolator1D<NF>
         * for each (i,j) pair, interpolating f in the z direction at that
         * fixed x- and y-index using sz as the interpolation variable,
         * stored as an (Nx × Ny) array. Must be called after
         * computeLengths(), since it relies on syData_ and szData_.
         */
        auto computeInterpolators() -> void
        {
            auto fView = Array4D(fData_.data(), nx_, ny_, nz_, NF);

            // y-direction interpolators, one per (i, k) pair
            if (ny_ > 1)
            {
                buildColumnInterpolators(nx_, nz_, ny_, sy(),
                    [&fView](size_t i, size_t k, size_t j, size_t n) -> double { return fView[i, j, k, n]; },
                    yInterpData_);
            }

            // z-direction interpolators, one per (i, j) pair
            if (nz_ > 1)
            {
                buildColumnInterpolators(nx_, ny_, nz_, sz(),
                    [&fView](size_t i, size_t j, size_t k, size_t n) -> double { return fView[i, j, k, n]; },
                    zInterpData_);
            }
        }

        /**
         * @brief Build a 2D slice of the mesh at fixed y
         * @param y0 The y coordinate at which to slice
         * @returns A newly-constructed Mesh2DInterpolator<NF> representing
         *          the (x, z) slice of the mesh at y = y0
         * @details
         * This is the actual slice-construction logic shared by
         * sliceConstY() and sliceConstYCopy(); see sliceConstY() for a
         * full description of how the slice is built.
         * @throws std::runtime_error if the mesh has only one point in
         *         the z direction, since there is then no way to build a
         *         valid (x, z) slice (a Mesh2DInterpolator needs at least
         *         two points along each of its axes)
         */
        [[nodiscard]] auto buildSliceY(double y0) const -> Mesh2DInterpolator<NF>
        {
            assert(y0 >= yMin() && y0 <= yMax());

            // If there is only one y-plane, the mesh is already entirely
            // at y = y0 (the assert above guarantees y0 matches it), so
            // the slice is just this mesh's (x, z, f) data with the
            // degenerate y dimension dropped -- no interpolation needed.
            // Since ny_ == 1 here, the constructor guard guarantees
            // nz_ >= 2, so the resulting Mesh2DInterpolator is valid.
            // Mesh2DInterpolator requires a mutable view even though it
            // only reads from it to build its own internal copies, so
            // casting away constness here is safe.
            if (ny_ == 1)
            {
                auto* xPtr = const_cast<double*>(xData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto* zPtr = const_cast<double*>(zData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto* fPtr = const_cast<double*>(fData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto xView = std::mdspan<double, std::dextents<size_t, 2>>(xPtr, nx_, nz_);
                auto zView = std::mdspan<double, std::dextents<size_t, 1>>(zPtr, nz_);
                auto fView = std::mdspan<double, std::dextents<size_t, 3>>(fPtr, nx_, nz_, NF);
                return Mesh2DInterpolator<NF>(xView, zView, fView, interpType_, monotonic_);
            }

            // If there is only one z-plane, the (x, z) slice this
            // function is meant to build would have a degenerate z axis
            // of size 1, which Mesh2DInterpolator cannot represent
            if (nz_ == 1)
            {
                throw std::runtime_error(
                    "Mesh3DInterpolator::sliceConstY: cannot slice at "
                    "fixed y because the mesh has only one point in the "
                    "z direction");
            }

            auto xv = x();
            auto fv = f();

            // The Mesh2DInterpolator constructor takes mutable mdspan
            // views, so the z coordinate array must be copied
            std::vector<double> zCopy(zData_.begin(), zData_.end());
            auto zSlice = std::mdspan<double, std::dextents<size_t, 1>>(zCopy.data(), nz_);

            std::vector<double> xSliceData(nx_ * nz_);
            std::vector<double> fSliceData(nx_ * nz_ * NF);
            auto xSlice = std::mdspan<double, std::dextents<size_t, 2>>(xSliceData.data(), nx_, nz_);
            auto fSlice = std::mdspan<double, std::dextents<size_t, 3>>(fSliceData.data(), nx_, nz_, NF);

            // Check for an exact match with one of the mesh y values
            for (size_t j = 0; j < ny_; ++j)
            {
                if (yData_[j] == y0)
                {
                    for (size_t i = 0; i < nx_; ++i)
                    {
                        for (size_t k = 0; k < nz_; ++k)
                        {
                            xSlice[i, k] = xv[i, j, k];
                            for (size_t n = 0; n < NF; ++n) { fSlice[i, k, n] = fv[i, j, k, n]; }
                        }
                    }
                    return Mesh2DInterpolator<NF>(xSlice, zSlice, fSlice, interpType_, monotonic_);
                }
            }

            // No exact match; find the bracketing y planes via binary
            // search and interpolate
            const size_t j0 = static_cast<size_t>(
                std::ranges::upper_bound(yData_, y0) - yData_.begin()) - 1;
            const double t = (y0 - yData_[j0]) / (yData_[j0 + 1] - yData_[j0]);

            auto syView = sy();
            auto yInterpView = yInterp();
            for (size_t i = 0; i < nx_; ++i)
            {
                for (size_t k = 0; k < nz_; ++k)
                {
                    xSlice[i, k] = xv[i, j0, k] +
                        (t * (xv[i, j0 + 1, k] - xv[i, j0, k]));

                    const double syTarget = syView[i, k, j0] +
                        (t * (syView[i, k, j0 + 1] - syView[i, k, j0]));
                    const auto fInterp = (*(yInterpView[i, k]))(syTarget);
                    storeSliceResult(fSlice, i, k, fInterp);
                }
            }

            return Mesh2DInterpolator<NF>(xSlice, zSlice, fSlice, interpType_, monotonic_);
        }

        /**
         * @brief Build a 2D slice of the mesh at fixed z
         * @param z0 The z coordinate at which to slice
         * @returns A newly-constructed Mesh2DInterpolator<NF> representing
         *          the (x, y) slice of the mesh at z = z0
         * @details
         * This is the actual slice-construction logic shared by
         * sliceConstZ() and sliceConstZCopy(); see sliceConstZ() for a
         * full description of how the slice is built.
         * @throws std::runtime_error if the mesh has only one point in
         *         the y direction, since there is then no way to build a
         *         valid (x, y) slice (a Mesh2DInterpolator needs at least
         *         two points along each of its axes)
         */
        [[nodiscard]] auto buildSliceZ(double z0) const -> Mesh2DInterpolator<NF>
        {
            assert(z0 >= zMin() && z0 <= zMax());

            // If there is only one z-plane, the mesh is already entirely
            // at z = z0 (the assert above guarantees z0 matches it), so
            // the slice is just this mesh's (x, y, f) data with the
            // degenerate z dimension dropped -- no interpolation needed.
            // Since nz_ == 1 here, the constructor guard guarantees
            // ny_ >= 2, so the resulting Mesh2DInterpolator is valid.
            // Mesh2DInterpolator requires a mutable view even though it
            // only reads from it to build its own internal copies, so
            // casting away constness here is safe.
            if (nz_ == 1)
            {
                auto* xPtr = const_cast<double*>(xData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto* yPtr = const_cast<double*>(yData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto* fPtr = const_cast<double*>(fData_.data()); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- see comment above: casting away constness here is safe
                auto xView = std::mdspan<double, std::dextents<size_t, 2>>(xPtr, nx_, ny_);
                auto yView = std::mdspan<double, std::dextents<size_t, 1>>(yPtr, ny_);
                auto fView = std::mdspan<double, std::dextents<size_t, 3>>(fPtr, nx_, ny_, NF);
                return Mesh2DInterpolator<NF>(xView, yView, fView, interpType_, monotonic_);
            }

            // If there is only one y-plane, the (x, y) slice this
            // function is meant to build would have a degenerate y axis
            // of size 1, which Mesh2DInterpolator cannot represent
            if (ny_ == 1)
            {
                throw std::runtime_error(
                    "Mesh3DInterpolator::sliceConstZ: cannot slice at "
                    "fixed z because the mesh has only one point in the "
                    "y direction");
            }

            auto xv = x();
            auto fv = f();

            // The Mesh2DInterpolator constructor takes mutable mdspan
            // views, so the y coordinate array must be copied
            std::vector<double> yCopy(yData_.begin(), yData_.end());
            auto ySlice = std::mdspan<double, std::dextents<size_t, 1>>(yCopy.data(), ny_);

            std::vector<double> xSliceData(nx_ * ny_);
            std::vector<double> fSliceData(nx_ * ny_ * NF);
            auto xSlice = std::mdspan<double, std::dextents<size_t, 2>>(xSliceData.data(), nx_, ny_);
            auto fSlice = std::mdspan<double, std::dextents<size_t, 3>>(fSliceData.data(), nx_, ny_, NF);

            // Check for an exact match with one of the mesh z values
            for (size_t k = 0; k < nz_; ++k)
            {
                if (zData_[k] == z0)
                {
                    for (size_t i = 0; i < nx_; ++i)
                    {
                        for (size_t j = 0; j < ny_; ++j)
                        {
                            xSlice[i, j] = xv[i, j, k];
                            for (size_t n = 0; n < NF; ++n) { fSlice[i, j, n] = fv[i, j, k, n]; }
                        }
                    }
                    return Mesh2DInterpolator<NF>(xSlice, ySlice, fSlice, interpType_, monotonic_);
                }
            }

            // No exact match; find the bracketing z planes via binary
            // search and interpolate
            const size_t k0 = static_cast<size_t>(
                std::ranges::upper_bound(zData_, z0) - zData_.begin()) - 1;
            const double t = (z0 - zData_[k0]) / (zData_[k0 + 1] - zData_[k0]);

            auto szView = sz();
            auto zInterpView = zInterp();
            for (size_t i = 0; i < nx_; ++i)
            {
                for (size_t j = 0; j < ny_; ++j)
                {
                    xSlice[i, j] = xv[i, j, k0] +
                        (t * (xv[i, j, k0 + 1] - xv[i, j, k0]));

                    const double szTarget = szView[i, j, k0] +
                        (t * (szView[i, j, k0 + 1] - szView[i, j, k0]));
                    const auto fInterp = (*(zInterpView[i, j]))(szTarget);
                    storeSliceResult(fSlice, i, j, fInterp);
                }
            }

            return Mesh2DInterpolator<NF>(xSlice, ySlice, fSlice, interpType_, monotonic_);
        }

        /**
         * @brief Store an evaluated interpolation result into a slice
         * @param fSlice Function-value slice of shape (D0, D1, NF)
         * @param d0 First index at which to store the result
         * @param d1 Second index at which to store the result
         * @param fInterp The evaluated result: a double if NF == 1, or an
         *                array-like of NF doubles otherwise
         */
        static auto storeSliceResult(
            std::mdspan<double, std::dextents<size_t, 3>> fSlice,
            size_t d0, size_t d1, const auto& fInterp) -> void
        {
            if constexpr (NF == 1) { fSlice[d0, d1, 0] = fInterp; }
            else
            {
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
                for (size_t n = 0; n < NF; ++n) { fSlice[d0, d1, n] = fInterp[n]; }
                // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
            }
        }
    };

} // namespace interp

// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

#endif // MESH3DINTERPOLATOR_HPP
