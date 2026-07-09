/**
 * @file Tracks2D.hpp
 * @author Mark Krumholz
 * @brief A class to represent a 2D set of stellar tracks
 * @date 2024-07-09
 */

#ifndef TRACK2D_HPP
#define TRACK2D_HPP

#include "../interpolation/Mesh2DInterpolator.hpp"
#include "H5Ipublic.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

/**
 * @brief A namespace to hold functions dealing with stellar tracks
*/
namespace tracks
{

    /**
     * @brief enum of properties stored in the tracks
     */
    enum class FieldIdx : std::uint8_t
    {
        mass,   /**< Present-day mass in Msun */
        mdot,   /**< Mass loss rate in Msun/yr */
        logL,   /**< Log luminosity in Lsun */
        logTe,  /**< Log effective temperature in K */
        hSurf,  /**< Surface H mass fraction */
        heSurf, /**< Surface He mass fraction */
        cSurf,  /**< Surface C mass fraction */
        nSurf,  /**< Surface N mass fraction */
        oSurf,  /**< Surface O mass fraction */
        nTrackQty /**< Number of quantities in the tracks */
    };

    constexpr std::array<std::string_view,
        static_cast<size_t>(FieldIdx::nTrackQty)> FieldStr{
        "mass",
        "mdot",
        "log_L",
        "log_Teff",
        "h_surf",
        "he_surf",
        "c_surf",
        "n_surf",
        "o_surf"
    };  /**< Field names for each field quantity in the files */

    /**
     * @class Track2D
     * @brief A class representing a 2D set of stellar tracks
     */
    class Tracks2D
    {
    public:

        // Constructors and destructors
        /**
         * @brief Construct a Tracks2D object
         * @param grp An HDF5 file handle to the group
         * @param ntMin If specified, minimum number of times in the tracks
         * @details
         * Tracks are stored for each mass, and the number of time
         * points may not be the same for every mass. Tracks with fewer
         * time points will be padded at the end so that the final
         * set of times and stellar properties are square arrays. If
         * ntMin is specified, this gives the minimum number of times in
         * the tracks, so tracks will be padded to at least this length.
        */
        Tracks2D(hid_t grp, size_t ntMin = 0);
        virtual ~Tracks2D() = default;
        Tracks2D(const Tracks2D&) = delete;
        Tracks2D(Tracks2D&&) = default;
        auto operator=(const Tracks2D&) -> Tracks2D& = delete;
        auto operator=(Tracks2D&&) -> Tracks2D& = delete;

        // Observers

        /**
         * @brief Return the minimum mass in the tracks
         * @return Minimum mass in the tracks
         */
        [[nodiscard]] auto mMin() const { return interp_->xMin(); }
        /**
         * @brief Return the maximum mass in the tracks
         * @return Maximum mass in the tracks
         */
        [[nodiscard]] auto mMax() const { return interp_->xMax(); }

        /**
         * @brief Return the minimum time in the tracks
         * @return Minimum time in the tracks
         */
        [[nodiscard]] auto tMin() const { return interp_->yMin(); }
        /**
         * @brief Return the minimum time in the tracks
         * @return Minimum time in the tracks
         */
        [[nodiscard]] auto tMax() const { return interp_->yMax(); }

        // Calculation methods

        /** 
         * @brief Return the lifetime of a star of specified mass 
         * @param m Mass of the star whose lifetime should be returned
         * @return The lifetime of a star of mass m
         */
       [[nodiscard]] auto starLifetime(const double m) { return interp_->xMax(m); }

       /**
        * @brief Return the range of stellar masses that are alive at a given time
        * @param t Time at which to evaluate
        * @return The range of stellar masses alive at the given time
        */
       [[nodiscard]] auto liveMassRange(const double t) { return interp_->yLim(t); }

        /**
         * @brief Return the track for a star of a given mass
         * @param m Mass of the star whose track should be computed
         * @return An unique_ptr to an Interpolator1D describing the track for a given mass
         */
        [[nodiscard]] auto getTrack(const double m) { return interp_->interpConstY(m); }

        /**
         * @brief Return the isochrone at a given time
         * @param t The time of the isochrone
         * @return A vector of unique_ptr to Interpolator1D's describing the isochrone
         * @details
         * Note that this method returns a vector of Interpolator1D objects
         * rather than a single one because for non-monotonic tracks there
         * may be multiple disjoint segments to the isochrone.
         */
        [[nodiscard]] auto getIsochrone(const double t) { return interp_->interpConstX(t); }

    private:

        // Track data
        std::unique_ptr<interp::Mesh2DInterpolator<
            static_cast<size_t>(FieldIdx::nTrackQty)>> interp_; /**< Interpolator for the tracks */

    };

} // namespace tracks

#endif // TRACK2D