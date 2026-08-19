/**
 * @file SpecsynLib2D.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLib2D.hpp
 * @date 2026-08-07
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "SpecsynLib2D.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace specsyn
{
    template <OOBPolicy Policy>
    void SpecsynLib2D<Policy>::resample(const std::vector<double>& wlNew)
    {
        // Mirrors SpecsynLib::resample(const std::vector<double>&)
        // exactly, but iterating over (dim2_, dim3_) alone, with the
        // degenerate first tensor index hardcoded to 0 -- see this
        // method's own header comment for why the base implementation
        // can't be used unmodified here.
        for (std::size_t i2 = 0; i2 < this->dim2_.size(); ++i2) // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- i2, i3 are both < the corresponding grid's size by construction
        {
            for (std::size_t i3 = 0; i3 < this->dim3_.size(); ++i3)
            {
                auto& spectrum = this->grid_[0, i2, i3];
                if (spectrum.empty()) { continue; } // unpopulated grid point: leave empty

                spectrum = SpecsynLib<Policy>::resample(this->wl_, wlNew, spectrum);
            }
        }

        this->wl_ = wlNew;
    }

    template <OOBPolicy Policy>
    auto SpecsynLib2D<Policy>::spec(const double d2, const double d3) const -> std::vector<double> // NOLINT(readability-function-cognitive-complexity) -- mirrors SpecsynLib::spec(double, double, double)'s own identical structure, just with the leading dimension removed; see that function's own NOLINT for why splitting this further would only add indirection
    {
        // Locate the bracketing cell on each axis -- exactly as
        // SpecsynLib::spec(double, double, double) does for its own
        // three axes.
        const auto b2 = detail::findBracket(this->dim2_, d2, this->dim2Cache_());
        const auto b3 = detail::findBracket(this->dim3_, d3, this->dim3Cache_());

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- b2/b3 indices are all < the corresponding grid's size by construction, and the interpolation loop below is a hot path where the cost of bounds checking matters
        // Every one of the 4 neighboring grid points must actually
        // have a spectrum -- interpolating across an unpopulated
        // point would be meaningless -- or this query point counts as
        // out of bounds. Under OOBPolicy::coerce, a query point is
        // only out of bounds if none of its 4 neighbors has a
        // spectrum; if at least one does, spec() instead interpolates
        // using only the valid neighbors (see below), rather than
        // requiring every corner to be populated. Mirrors
        // SpecsynLib::spec(double, double, double)'s own identical
        // logic, just over 4 corners instead of 8.
        bool hasValidNeighbor = false; // NOLINT(misc-const-correctness) -- see the identical NOLINT on SpecsynLib::spec's own hasValidNeighbor
        for (const std::size_t i2 : { b2.lo_, b2.hi_ })
        {
            for (const std::size_t i3 : { b3.lo_, b3.hi_ })
            {
                if constexpr (Policy == OOBPolicy::coerce)
                {
                    if (!this->grid_[0, i2, i3].empty())
                    {
                        hasValidNeighbor = true;
                        break;
                    }
                }
                else
                {
                    if (this->grid_[0, i2, i3].empty())
                    {
                        return SpecsynLib<Policy>::outOfBoundsResult(
                            "SpecsynLib2D: point (" + std::to_string(d2) + ", " +
                            std::to_string(d3) + ") falls in a gap in this library's grid");
                    }
                }
            }
        }
        if constexpr (Policy == OOBPolicy::coerce)
        {
            if (!hasValidNeighbor)
            {
                return SpecsynLib<Policy>::outOfBoundsResult(
                    "SpecsynLib2D: point (" + std::to_string(d2) + ", " +
                    std::to_string(d3) + ") has no valid neighboring grid points to coerce to");
            }
        }

        // Bilinear interpolation of the stored quantity over the 4
        // neighboring grid points; no scaling (e.g. by surface area)
        // is applied here -- that is left entirely to the caller.
        // Under OOBPolicy::coerce, an unpopulated corner simply
        // contributes nothing, and wSum (the total weight of the
        // valid corners actually used, guaranteed to be 1 if every
        // corner is populated) renormalizes the result at the end so
        // it still represents a properly weighted average rather than
        // an artificially dimmed spectrum. Mirrors SpecsynLib::spec(
        // double, double, double)'s own identical logic exactly.
        std::vector<double> result(this->wl_.size(), 0.0);
        double wSum = 0.0; // NOLINT(misc-const-correctness) -- see the identical NOLINT on SpecsynLib::spec's own wSum
        for (int b2i = 0; b2i < 2; ++b2i)
        {
            const std::size_t i2 = (b2i == 0) ? b2.lo_ : b2.hi_;
            const double wgt2 = (b2i == 0) ? (1.0 - b2.t_) : b2.t_;
            for (int b3i = 0; b3i < 2; ++b3i)
            {
                const std::size_t i3 = (b3i == 0) ? b3.lo_ : b3.hi_;
                const double wgt3 = (b3i == 0) ? (1.0 - b3.t_) : b3.t_;

                const double weight = wgt2 * wgt3;
                if (weight == 0.0) { continue; } // degenerate axis or exact grid hit: skip a zero-weight corner

                const auto& corner = this->grid_[0, i2, i3];
                if constexpr (Policy == OOBPolicy::coerce)
                {
                    if (corner.empty()) { continue; }
                    wSum += weight;
                }
                for (std::size_t w = 0; w < result.size(); ++w)
                {
                    result[w] += weight * corner[w];
                }
            }
        }
        if constexpr (Policy == OOBPolicy::coerce)
        {
            // wSum can still be exactly 0 here despite hasValidNeighbor
            // being true above -- see SpecsynLib::spec(double, double,
            // double)'s own identical comment for why -- so treat that
            // the same as having no valid neighbor at all, rather than
            // dividing by 0 and silently producing a NaN/Inf "result".
            if (wSum == 0.0)
            {
                return SpecsynLib<Policy>::outOfBoundsResult(
                    "SpecsynLib2D: point (" + std::to_string(d2) + ", " +
                    std::to_string(d3) + ") has no valid neighboring grid points "
                    "with nonzero weight to coerce to");
            }
            for (auto& v : result) { v /= wSum; }
        }
        return result;
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }

    // Explicit instantiation for every OOBPolicy value actually used;
    // this keeps the class's implementation in this .cpp file, as with
    // every other class in src/specsyn, rather than forcing it into
    // the header just because it is a template. OOBPolicy::coerce and
    // OOBPolicy::silent behave identically for this class, since there
    // are no unpopulated interior grid points to coerce around in the
    // first place -- only a derived class's own out-of-grid-range
    // checks (before ever calling into this spec()) actually
    // distinguish the three policies -- but all three are instantiated
    // regardless, for uniformity with every other SpecsynLib subclass.
    template class SpecsynLib2D<OOBPolicy::raise>;
    template class SpecsynLib2D<OOBPolicy::silent>;
    template class SpecsynLib2D<OOBPolicy::coerce>;

} // namespace specsyn
