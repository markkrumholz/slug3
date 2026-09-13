/**
 * @file YieldChannel.hpp
 * @author Mark Krumholz
 * @brief A class to represent the nucleosynthetic yield data for one channel
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDCHANNEL_HPP
#define YIELDCHANNEL_HPP

#include "../elem/IsotopeData.hpp"
#include "../utils/GridBracket.hpp"
#include "../utils/ThreadVec.hpp"
#include "YieldCommons.hpp"
#include <cassert>
#include <cstddef>
#include <functional>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <optional>
#include <string>
#include <vector>

namespace yields
{
    /**
     * @class YieldChannel
     * @brief Nucleosynthetic yield data for one channel (e.g. ccsn),
     *   one model (e.g. sukhbold16), over a range of [Fe/H]
     */
    class YieldChannel
    {
    public:

        /** @brief mdspan view of yieldData_ -- see yld()'s own comment */
        using Array3D = std::mdspan<const double, std::dextents<std::size_t, 3>>; // NOLINT(misc-include-cleaner)

        // Constructors and destructors

        /**
         * @brief Construct a YieldChannel from a yield model on disk
         * @param descriptor Which channel/model to load, and the mass
         *   range it should cover -- see YieldChannelDescriptor's own
         *   comment for each member; descriptor.mMin_/mMax_ mean exactly
         *   what rebuildMassGrid()'s own mMin/mMax parameters do
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param registryName Name of the yield registry file
         * @throws std::runtime_error if descriptor.channel_/modelName_ is
         *   not found in the registry, if fehMin or fehMax lies outside
         *   the [Fe/H] range actually available for that channel/model,
         *   or if the underlying HDF5 file cannot be read or is malformed
         * @details
         * Calls rebuildMassGrid(descriptor.mMin_, descriptor.mMax_) once
         * the model's own native mass grid (massesOrig_) has been read
         * -- see rebuildMassGrid()'s own comment for how those turn
         * into masses_/yieldData_. fehMin/fehMax/registryName are
         * separate parameters, rather than YieldChannelDescriptor
         * members, because a whole run's worth of channels (see the
         * Yields class) share one [Fe/H] range and registry, but each
         * has its own channel/model/mass range.
         */
        YieldChannel(
            const YieldChannelDescriptor& descriptor,
            double fehMin,
            double fehMax,
            const std::string& registryName = defaultRegistry);

        ~YieldChannel() = default;

        // Neither copyable nor movable: massCache_/fehCache_ are
        // utils::ThreadVec, itself neither copyable nor movable (see
        // its own comment), for the same reason SpecsynLib's own,
        // identical dim1Cache_/dim2Cache_/dim3Cache_ leave SpecsynLib
        // neither copyable nor movable either -- a per-thread cache
        // has no well-defined "the copy/move target's own cache"
        // to become. Constructed once (an HDF5 read) and queried
        // through a reference/pointer thereafter, the same as
        // SpecsynLib itself.
        YieldChannel(const YieldChannel&) = delete;
        YieldChannel(YieldChannel&&) = delete;
        auto operator=(const YieldChannel&) -> YieldChannel& = delete;
        auto operator=(YieldChannel&&) -> YieldChannel& = delete;

        /**
         * @brief Rebuild masses_/yieldData_ over [mMin, mMax] from massesOrig_/yieldDataOrig_
         * @param mMin Minimum stellar mass this channel should cover;
         *   defaults (nullopt) to massesOrig().front()
         * @param mMax Maximum stellar mass this channel should cover;
         *   defaults (nullopt) to massesOrig().back()
         * @throws std::invalid_argument if the resolved mMin or mMax is
         *   not finite and strictly positive (a stellar mass), or if
         *   the resolved mMin is not strictly less than the resolved mMax
         * @details
         * Called once by the constructor, with whatever mMin/mMax it
         * was given; also public, so a caller (e.g. from Python, after
         * changing which mass range a channel should cover) can rerun
         * this on an already-constructed YieldChannel without building
         * a new one -- the same role loadCurve()/rebuildCache() play
         * for Extinct, and loadTable() for Nebular.
         *
         * masses_ is set to mMin, followed by every value in
         * massesOrig_ strictly between mMin and mMax, followed by mMax
         * -- so masses_ is always exactly [mMin, mMax] endpoint-
         * inclusive, whether that's wider than massesOrig_'s own range
         * (extrapolation), narrower (a truncated sub-range), or
         * identical to it (mMin/mMax left at their defaults).
         *
         * yieldData_ is filled in one mass column at a time. For a
         * masses_ entry that exactly matches some value in massesOrig_
         * (including, as a special case, every value copied verbatim
         * from massesOrig_ above), utils::findBracket returns a
         * bracket weight of exactly 0 or 1, which the same interpolation
         * formula used for a genuine interior point below reduces to an
         * exact copy of that one massesOrig_ column -- so no separate
         * exact-match branch is needed for that case. For an entry
         * strictly between two massesOrig_ values, that same
         * utils::findBracket bracket/weight linearly interpolates
         * between the two neighboring massesOrig_ columns. For an entry
         * outside [massesOrig_.front(), massesOrig_.back()] altogether
         * -- the actual extrapolation case this method exists for --
         * utils::findBracket can't help (it clamps out-of-range queries
         * to the nearest edge, weight 0 or 1, rather than reporting
         * them as out of range), so that case is detected explicitly
         * and handled by scaling every isotope's yield in the nearest
         * massesOrig_ column (front() if mMin is below it, back() if
         * mMax is above it) by the ratio of the requested mass to that
         * column's own mass -- e.g. extrapolating a massesOrig_.front()
         * of 13 Msun down to mMin = 8 Msun scales that column's every
         * yield by 8/13.
         *
         * massCache_ is reset to 0 for every thread once masses_ has a
         * new size: an index cached against the old masses_ could
         * otherwise be at or past the end of a smaller new one, which
         * yield()'s own unchecked utils::findBracket call requires
         * never happens. fehCache_/feH_ are untouched -- this method
         * only ever changes the mass axis.
         */
        void rebuildMassGrid(
            std::optional<double> mMin = std::nullopt,
            std::optional<double> mMax = std::nullopt);

        // Observers

        /**
         * @brief Return which nucleosynthetic channel this object holds
         * @return The channel passed to the constructor
         */
        [[nodiscard]] auto channel() const { return channel_; }

        /**
         * @brief Return the stellar masses this channel's yields are tabulated at
         * @return A const reference to the masses (Msun), ascending --
         *   this is the grid yield()/hasYield() actually use, which can
         *   extend below/above massesOrig()'s own range (extrapolated)
         *   or be a narrower sub-range of it (see rebuildMassGrid()'s
         *   own comment)
         */
        [[nodiscard]] auto masses() const -> const std::vector<double>& { return masses_; }

        /**
         * @brief Return the stellar masses natively tabulated by the underlying model
         * @return A const reference to the masses (Msun), ascending,
         *   exactly as read from the model's own HDF5 file -- unlike
         *   masses(), never affected by the mMin/mMax passed to the
         *   constructor or rebuildMassGrid()
         */
        [[nodiscard]] auto massesOrig() const -> const std::vector<double>& { return massesOrig_; }

        /**
         * @brief Return the isotopes this channel's yields are tabulated for
         * @return A const reference to the isotopes, in the same order
         *   as yld()'s own third axis -- each one a reference into the
         *   single, global elem::isotopeTable(), resolved once (via the
         *   (Z, A) pairs read from the model's own HDF5 file) in the
         *   constructor
         */
        [[nodiscard]] auto isotopes() const
            -> const std::vector<std::reference_wrapper<const elem::IsotopeData>>&
        {
            return isotopes_;
        }

        /**
         * @brief Return the [Fe/H] values this channel's yields are tabulated at
         * @return A const reference to the [Fe/H] values, ascending --
         *   see the constructor's own comment for how this can be wider
         *   than [fehMin, fehMax], the range actually requested
         */
        [[nodiscard]] auto feH() const -> const std::vector<double>& { return feH_; }

        /**
         * @brief Return the yield data as a 3D view
         * @return An mdspan of shape (feH().size(), masses().size(),
         *   isotopes().size()) into yieldData_, i.e. yld()[f, m, i] is
         *   the yield (Msun) of isotope i, from a star of mass
         *   masses()[m], at metallicity feH()[f]
         * @details
         * Built fresh from yieldData_ on every call, rather than cached
         * as its own sibling member, mirroring Mesh3DInterpolator's own
         * x()/y()/z()/f() accessors -- avoids storing a pointer-into-
         * self that would need special handling if this class ever did
         * become copyable or movable in the future (it does not
         * currently support either -- see the constructor's own
         * comment on why), and costs nothing extra: an mdspan is just a
         * pointer and three extents, cheap to construct on demand.
         */
        [[nodiscard]] auto yld() const -> Array3D
        {
            return Array3D(yieldData_.data(), feH_.size(), masses_.size(), isotopes_.size());
        }

        /**
         * @brief Check whether a stellar mass falls within this channel's mass grid
         * @param mass Stellar mass to check (Msun)
         * @return True if mass lies within [masses().front(), masses().back()], false otherwise
         */
        [[nodiscard]] auto hasYield(const double mass) const -> bool
        {
            return mass >= masses_.front() && mass <= masses_.back();
        }

        /**
         * @brief Return the yield of every isotope for a star of given mass and [Fe/H]
         * @param mass Stellar mass (Msun); must satisfy hasYield(mass)
         * @param feH [Fe/H]; must lie within [feH().front(), feH().back()]
         * @return A vector of isotopes().size() yields (Msun), in the
         *   same order as isotopes(), bilinearly interpolated from
         *   yld() in (feH, mass)
         * @details
         * Callers must check hasYield(mass) (and that feH lies within
         * feH()'s own range) themselves before calling this -- enforced
         * here only via assert(), so a violation is caught in a debug
         * build but costs nothing in an optimized one, and via
         * unchecked mdspan/vector element access throughout, for the
         * same reason. utils::findBracket handles a singular (size-1)
         * feH() axis -- e.g. a Solar-only yield model such as
         * sukhbold16 -- automatically, returning weight 0 on one side
         * and 1 on the other with no special-casing needed here; see
         * its own comment.
         *
         * massCache_/fehCache_ accelerate repeated nearby queries the
         * same way SpecsynLib's own dim*Cache_ members do for spec()
         * queries (see that class's own comment) -- this is also why
         * this method, though logically const, needs those two members
         * mutable.
         */
        [[nodiscard]] auto yield(const double mass, const double feH) const -> std::vector<double>
        {
            assert(hasYield(mass));
            assert(feH >= feH_.front() && feH <= feH_.back());

            const auto bm = utils::findBracket(masses_, mass, massCache_());
            const auto bf = utils::findBracket(feH_, feH, fehCache_());
            const auto view = yld();
            const std::size_t niso = isotopes_.size();

            std::vector<double> result(niso, 0.0);
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bm/bf indices are all < the corresponding grid's size by construction, and this loop is a hot path where the cost of bounds checking matters -- see this method's own comment
            for (int fi = 0; fi < 2; ++fi)
            {
                const std::size_t f = (fi == 0) ? bf.lo_ : bf.hi_;
                const double wf = (fi == 0) ? (1.0 - bf.t_) : bf.t_;
                if (wf == 0.0) { continue; } // degenerate/singular feH axis or exact grid hit
                for (int mi = 0; mi < 2; ++mi)
                {
                    const std::size_t m = (mi == 0) ? bm.lo_ : bm.hi_;
                    const double wm = (mi == 0) ? (1.0 - bm.t_) : bm.t_;
                    const double weight = wf * wm;
                    if (weight == 0.0) { continue; } // exact grid hit on mass
                    for (std::size_t iso = 0; iso < niso; ++iso)
                    {
                        result[iso] += weight * view[f, m, iso];
                    }
                }
            }
            // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            return result;
        }

    private:

        Channel channel_;                  /**< Which nucleosynthetic channel this is */
        std::vector<double> masses_;       /**< Stellar masses (Msun) this channel's yields are tabulated at -- see masses()'s own comment */
        std::vector<double> massesOrig_;   /**< Stellar masses (Msun) as read from the model's own HDF5 file -- see massesOrig()'s own comment */
        std::vector<std::reference_wrapper<const elem::IsotopeData>> isotopes_; /**< Isotopes this channel's yields are tabulated for -- see isotopes()'s own comment */
        std::vector<double> feH_;          /**< [Fe/H] values for all yields */ // NOLINT(readability-identifier-naming)
        std::vector<double> yieldData_;    /**< Backing storage for yld(), over masses_ -- see its own comment */
        std::vector<double> yieldDataOrig_; /**< Backing storage over massesOrig_, from which yieldData_ is (re)derived -- see rebuildMassGrid()'s own comment */
        mutable utils::ThreadVec<std::size_t> massCache_; /**< Cached bracket index for masses_, see yield()'s own comment */
        mutable utils::ThreadVec<std::size_t> fehCache_;  /**< Cached bracket index for feH_, see yield()'s own comment */

    };

} // namespace yields

#endif // YIELDCHANNEL_HPP
