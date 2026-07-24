/**
 * @file SpecsynLib.hpp
 * @author Mark Krumholz
 * @brief Abstract base for a spectral synthesizer backed by a 3D tensor-grid library
 * @date 2026-07-20
 */

#ifndef SPECSYNLIB_HPP
#define SPECSYNLIB_HPP

#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <string>
#include <vector>

namespace specsyn
{

    /**
     * @class SpecsynLib
     * @brief Abstract base for a Specsyn backed by a 3D tensor-grid spectral library
     * @tparam Policy How this SpecsynLib should handle a query point
     *   that falls outside its tensor grid, or in an unpopulated gap
     *   within it -- see OOBPolicy. This is a template parameter,
     *   rather than a constructor argument or runtime flag, so that
     *   spec() can compile the chosen behavior directly into its hot
     *   path instead of branching on it at runtime.
     * @details
     * This class holds the machinery shared by every spectral library
     * that stores its spectra on a 3D tensor grid and interpolates
     * trilinearly within it: the grid's three axes (dim1_, dim2_,
     * dim3_), the (possibly only partially populated) spectra on that
     * grid (spectra_, viewed through the grid_ mdspan), and the
     * trilinear-interpolation spec() overload that operates on them.
     * It knows nothing about what the three axes actually mean -- that
     * mapping (e.g. from a StarData and [Fe/H] to a (FeH, logg, Teff)
     * point, or from a Wolf-Rayet star's properties to a point in a
     * fundamentally different coordinate system) is entirely the job
     * of a derived class, such as SpecsynLibNoWind (stars without
     * optically thick winds, covering libraries like BOSZ and TLUSTY)
     * or SpecsynLibWR (Wolf-Rayet stars, whose atmospheres are
     * parameterized by different variables entirely).
     *
     * A default-constructed SpecsynLib has every one of dim1_, dim2_,
     * dim3_, and spectra_ empty, and grid_ default-constructed (an
     * empty view over nothing). Populating all of these -- including
     * sizing spectra_ and constructing grid_ to view it -- is
     * entirely the responsibility of the derived class's own
     * constructor, since this base class has no way to know the
     * grid's shape until the derived class has read its own library
     * file.
     */
    template <OOBPolicy Policy>
    class SpecsynLib : public Specsyn
    {
    public:

        /**
         * @brief Construct an empty SpecsynLib
         * @details
         * Leaves dim1_, dim2_, dim3_, and spectra_ empty, and grid_
         * default-constructed. See the class-level details for why
         * populating them is left entirely to derived classes.
         */
        SpecsynLib() = default;

        /**
         * @brief Compute a star's spectrum
         * @param props Stellar properties, as produced by evaluating
         *   the Interpolator1D returned by Tracks2D::getIsochrone at
         *   this star's mass
         * @param feh [Fe/H] value of the star; needed because it is
         *   not carried by props itself
         * @return The star's spectrum, evaluated on the wavelength
         *   grid returned by wl(), in units of erg/s/Angstrom; a
         *   size-0 vector if the star falls outside this library's
         *   grid and Policy is OOBPolicy::silent
         * @throws std::runtime_error if the star falls outside this
         *   library's grid and Policy is OOBPolicy::raise
         * @details
         * Still pure virtual here (re-declared, rather than simply
         * inherited unimplemented from Specsyn, purely so this
         * class's own division of labor can be documented at the
         * point where it matters): a derived class implements this to
         * map props and feh onto this library's own
         * (dim1_, dim2_, dim3_) coordinate system -- bounds-checking
         * as appropriate for its own variables -- then calls the
         * three-argument spec() below to do the actual trilinear
         * interpolation, and finally applies whatever scaling (e.g.
         * by stellar surface area) is appropriate to convert the
         * library's stored quantity into a specific luminosity, which
         * this base class knows nothing about.
         */
        [[nodiscard]] auto spec(const StarData& props, double feh) const
        -> std::vector<double> override = 0;

        /**
         * @brief Resample every spectrum in this library onto a new wavelength grid
         * @param wlNew The new wavelength grid, in Angstrom
         * @details
         * For every populated (non-empty) point in the tensor grid,
         * builds an Interpolator1D of that point's flux versus this
         * library's existing wavelength grid (wl_), then evaluates it
         * at every wavelength in wlNew to produce that point's
         * resampled flux; wavelengths in wlNew that fall outside the
         * range spanned by wl_ are assigned a flux of zero rather than
         * extrapolated. Once every populated point has been resampled
         * this way, wl_ itself is replaced with wlNew, so wl() and
         * every subsequent spec() call reflect the new grid.
         * Unpopulated grid points are left as empty vectors, exactly
         * as before.
         */
        void resample(const std::vector<double>& wlNew);

    protected:

        /** @brief The shape of grid_, the mdspan view onto spectra_ */
        using SpectraGrid = std::mdspan<std::vector<double>, std::dextents<std::size_t, 3>>; // NOLINT(misc-include-cleaner) -- <mdspan> is already included above; clang-tidy-18's libc++-18 header-mapping data doesn't yet recognize it as std::mdspan/std::dextents' canonical header

        /**
         * @brief Compute a spectrum by trilinear interpolation at a point in the tensor grid
         * @param d1 Query coordinate along dim1_
         * @param d2 Query coordinate along dim2_
         * @param d3 Query coordinate along dim3_
         * @return The interpolated spectrum, evaluated on the
         *   wavelength grid returned by wl(), with no scaling applied
         *   beyond the interpolation itself -- any further scaling
         *   (e.g. by a star's surface area) is left entirely to the
         *   caller, since this class knows nothing about what
         *   quantity is actually stored on the grid; a size-0 vector
         *   if any of the 8 tensor-grid corners bracketing
         *   (d1, d2, d3) is unpopulated and Policy is OOBPolicy::silent
         * @throws std::runtime_error if any of those 8 corners is
         *   unpopulated and Policy is OOBPolicy::raise
         * @details
         * Callers (i.e. derived classes) are responsible for having
         * already checked that d1, d2, d3 each lie within
         * [dim1_.front(), dim1_.back()], etc. -- this method only
         * locates the bracketing grid cell along each axis (via an
         * O(log n) binary search, since none of the three axes can be
         * assumed evenly spaced), confirms every one of the 8
         * neighboring corners actually has a spectrum (interpolating
         * across an unpopulated point would be meaningless), and
         * trilinearly interpolates across them.
         */
        [[nodiscard]] auto spec(double d1, double d2, double d3) const -> std::vector<double>;

        /**
         * @brief Handle a query point that falls outside this library's grid
         * @param message Description of why the point is out of bounds
         * @return A size-0 vector, if Policy is OOBPolicy::silent
         * @throws std::runtime_error with message, if Policy is OOBPolicy::raise
         * @details
         * protected, rather than private, so that a derived class's
         * own spec() override can use it too for the bounds checks
         * (e.g. on Teff, logg) that are specific to its own variables
         * and therefore not handled by this class's own spec().
         */
        [[nodiscard]] static auto outOfBoundsResult(const std::string& message) -> std::vector<double>;

        std::vector<double> dim1_; /**< Grid points along the tensor grid's first axis */
        std::vector<double> dim2_; /**< Grid points along the tensor grid's second axis */
        std::vector<double> dim3_; /**< Grid points along the tensor grid's third axis */

        /**
         * @brief Spectra on the (dim1_, dim2_, dim3_) tensor grid
         * @details
         * A flattened array of shape (dim1_.size(), dim2_.size(),
         * dim3_.size()) -- viewed through grid_ -- holding one
         * spectrum (a vector of the same length as wl_) per populated
         * grid point. Not every point in this outer product need
         * actually have a spectrum in the library; unpopulated points
         * are left as empty (size-0) vectors.
         */
        std::vector<std::vector<double>> spectra_;

        /**
         * @brief An mdspan view onto spectra_, shaped (dim1_.size(), dim2_.size(), dim3_.size())
         * @details
         * Constructed by the derived class once it has finished
         * sizing spectra_ to its final shape; default-constructed to
         * an empty view over nothing until then. Safe to keep across
         * spectra_'s lifetime as long as the derived class never
         * resizes the outer vector again after constructing grid_ --
         * only mutates individual elements in place -- since resizing
         * could reallocate spectra_'s backing storage and leave
         * grid_ dangling.
         */
        SpectraGrid grid_;
    };

} // namespace specsyn

#endif // SPECSYNLIB_HPP
