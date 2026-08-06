/**
 * @file SpecsynLib2D.hpp
 * @author Mark Krumholz
 * @brief A SpecsynLib specialization for spectral libraries with no first axis
 * @date 2026-08-07
 */

#ifndef SPECSYNLIB2D_HPP
#define SPECSYNLIB2D_HPP

#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

namespace specsyn
{

    /**
     * @class SpecsynLib2D
     * @brief A SpecsynLib specialization for a library with no first (dim1_) axis
     * @tparam Policy See SpecsynLib.
     * @details
     * Some spectral libraries -- e.g. the Tremblay et al. white dwarf
     * atmosphere grids SpecsynLibWD reads -- have only two physical
     * axes (log(g) and Teff), not three, and are distributed as a
     * single, completely filled tensor over those two axes, with no
     * unpopulated gaps at all. This class adapts SpecsynLib's own
     * (dim1_, dim2_, dim3_) machinery to that case by simply treating
     * dim1_ as degenerate (left empty; grid_'s own first extent is
     * hardcoded to 1 rather than dim1_.size(), so a derived class's
     * constructor never needs to populate dim1_ with a placeholder
     * value at all -- see SpecsynLibWD's own constructor), and
     * providing a simplified, two-argument spec() that interpolates
     * only across dim2_ and dim3_, fixing the first tensor index to 0.
     *
     * Because the grid this class was designed for is always
     * completely filled, this simplified spec() does not do any of
     * the unpopulated-neighbor checking SpecsynLib::spec(double,
     * double, double) does -- there is nothing to coerce around, or
     * to report as a gap -- so OOBPolicy only ever matters for
     * whatever grid-boundary checking a derived class's own spec(
     * const StarData&, double) override does before calling down into
     * this class's spec(double, double) at all (see SpecsynLibWD's
     * own spec()).
     */
    template <OOBPolicy Policy>
    class SpecsynLib2D : public SpecsynLib<Policy>
    {
    public:

        /**
         * @brief Construct an empty SpecsynLib2D
         * @param controls Simulation controls; forwarded unchanged to
         *   SpecsynLib's own constructor -- see its comment
         * @details
         * Leaves dim1_, dim2_, dim3_, and spectra_ empty, and grid_
         * default-constructed, exactly as SpecsynLib's own default
         * constructor does. Populating dim2_, dim3_, and spectra_ (and
         * constructing grid_ to view it, with a first extent of 1) is
         * left entirely to a derived class's own constructor, since
         * this class has no way to know the grid's shape until the
         * derived class has read its own library file.
         */
        explicit SpecsynLib2D(const io::SimControls& controls) :
            SpecsynLib<Policy>(controls) { }

    protected:

        /**
         * @brief Compute a spectrum by bilinear interpolation at a point in the 2D grid
         * @param d2 Query coordinate along dim2_
         * @param d3 Query coordinate along dim3_
         * @return The interpolated spectrum, evaluated on the
         *   wavelength grid returned by wl(), with no scaling applied
         *   beyond the interpolation itself -- any further scaling
         *   (e.g. by a star's surface area) is left entirely to the
         *   caller, exactly as SpecsynLib::spec(double, double,
         *   double) leaves it to its own callers
         * @details
         * Callers (i.e. a derived class's own spec(const StarData&,
         * double) override) are responsible for having already
         * checked that d2 and d3 each lie within [dim2_.front(),
         * dim2_.back()] and [dim3_.front(), dim3_.back()]
         * respectively. This method only locates the bracketing grid
         * cell along each axis (via detail::findBracket -- the same
         * cache-accelerated binary search SpecsynLib::spec(double,
         * double, double) itself uses, reusing dim2Cache_/dim3Cache_
         * rather than keeping a redundant cache of its own) and
         * bilinearly interpolates the spectrum from the 4 neighboring
         * grid points, with the degenerate first tensor index
         * hardcoded to 0. Unlike SpecsynLib::spec(double, double,
         * double), this does not check that those 4 neighbors are
         * actually populated -- see this class's own @details for
         * why that is unnecessary here.
         */
        [[nodiscard]] auto spec(double d2, double d3) const -> std::vector<double>;
    };

} // namespace specsyn

#endif // SPECSYNLIB2D_HPP
