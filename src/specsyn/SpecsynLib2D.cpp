/**
 * @file SpecsynLib2D.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLib2D.hpp
 * @date 2026-08-07
 */

#include "SpecsynLib2D.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <cstddef>
#include <vector>

namespace specsyn
{
    template <OOBPolicy Policy>
    auto SpecsynLib2D<Policy>::spec(const double d2, const double d3) const -> std::vector<double>
    {
        // Locate the bracketing cell on each axis, exactly as
        // SpecsynLib::spec(double, double, double) does for its own
        // three axes -- see this method's own comment for why no
        // unpopulated-neighbor checking is needed here.
        const auto b2 = detail::findBracket(this->dim2_, d2, this->dim2Cache_());
        const auto b3 = detail::findBracket(this->dim3_, d3, this->dim3Cache_());

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- b2/b3 indices are all < the corresponding grid's size by construction, and the interpolation loop below is a hot path where the cost of bounds checking matters
        std::vector<double> result(this->wl_.size(), 0.0);
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
                for (std::size_t w = 0; w < result.size(); ++w)
                {
                    result[w] += weight * corner[w];
                }
            }
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
