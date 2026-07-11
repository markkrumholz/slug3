/**
 * @file Tracks2D.hpp
 * @author Mark Krumholz
 * @brief A class to represent a 2D set of stellar tracks
 * @date 2024-07-09
 */

#ifndef TRACKS2D_HPP
#define TRACKS2D_HPP

#include "TrackCommons.hpp"
#include "../interpolation/Mesh2DInterpolator.hpp"
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace tracks
{

    /**
     * @class Tracks2D
     * @brief A class representing a 2D set of stellar tracks
     */
    class Tracks2D
    {
    public:

        // Shorten type name
        using M2DPtr = std::unique_ptr<
            interp::Mesh2DInterpolator<
            static_cast<size_t>(FieldIdx::nTrackQty)>>;

        // Constructors and destructors
        /**
         * @brief Construct a Tracks2D object from tracks on disk
         * @param trackName Name of track set
         * @param feh [Fe/H] value
         * @param vvcrit Rotation rate v/vcrit
         * @param afe Value of [alpha/Fe]
         * @param registryName Name of the track registry file
         * @param ntMin If specified, minimum number of times in the tracks
         * @details
         * Uses findTrack to locate the unique track in track set
         * trackName matching feh, vvcrit, and afe, and throws a
         * runtime error if no such track can be found. Tracks are
         * stored for each mass, and the number of time points may not
         * be the same for every mass. Tracks with fewer time points
         * will be padded at the end so that the final set of times and
         * stellar properties are square arrays. If ntMin is specified,
         * this gives the minimum number of times in the tracks, so
         * tracks will be padded to at least this length.
        */
        Tracks2D(
            const std::string& trackName,
            double feh = 0.0,
            double vvcrit = 0.0,
            double afe = 0.0,
            const std::string& registryName = defaultRegistry,
            size_t ntMin = 0);
        /**
         * @brief Construct a Tracks2D object from a supplied Mesh2DInterpolator
         * @param m2d A unique_ptr to the interpolator from which to construct the track
         * @param feH The [Fe/H] value of this set of tracks
         * @param aFe The [alpha/Fe] value of this set of tracks, if any;
         *            defaults to quiet_NaN if not specified
         * @param vVcrit The v/vcrit value of this set of tracks, if any;
         *               defaults to quiet_NaN if not specified
         */
        Tracks2D(M2DPtr&& m2d, double feH,
            double aFe = std::numeric_limits<double>::quiet_NaN(),
            double vVcrit = std::numeric_limits<double>::quiet_NaN())
        : interp_(std::move(m2d)), FeH_(feH), AFe_(aFe), vVcrit_(vVcrit) {};
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
        [[nodiscard]] auto mMin() const { return interp_->yMin(); }
        /**
         * @brief Return the maximum mass in the tracks
         * @return Maximum mass in the tracks
         */
        [[nodiscard]] auto mMax() const { return interp_->yMax(); }

        /**
         * @brief Return the minimum time in the tracks
         * @return Minimum time in the tracks
         */
        [[nodiscard]] auto tMin() const { return interp_->xMin(); }
        /**
         * @brief Return the maximum time in the tracks
         * @return Maximum time in the tracks
         */
        [[nodiscard]] auto tMax() const { return interp_->xMax(); }

        /**
         * @brief Return the [Fe/H] value of this set of tracks
         * @return [Fe/H] value of this set of tracks
         */
        [[nodiscard]] auto feH() const { return FeH_; }

        /**
         * @brief Return the [alpha/Fe] value of this set of tracks
         * @return [alpha/Fe] value of this set of tracks, or quiet_NaN if not available
         */
        [[nodiscard]] auto aFe() const { return AFe_; }

        /**
         * @brief Return the v/vcrit value of this set of tracks
         * @return v/vcrit value of this set of tracks, or quiet_NaN if not available
         */
        [[nodiscard]] auto vVcrit() const { return vVcrit_; }

        // Calculation methods

        /** 
         * @brief Return the lifetime of a star of specified mass 
         * @param m Mass of the star whose lifetime should be returned
         * @return The lifetime of a star of mass m
         */
       [[nodiscard]] auto starLifetime(const double m) const { return interp_->xMax(m); }

       /**
        * @brief Return the range of stellar masses that are alive at a given time
        * @param t Time at which to evaluate
        * @return The range of stellar masses alive at the given time
        */
       [[nodiscard]] auto liveMassRange(const double t) const { return interp_->yLim(t); }

        /**
         * @brief Return the track for a star of a given mass
         * @param m Mass of the star whose track should be computed
         * @return An unique_ptr to an Interpolator1D describing the track for a given mass
         */
        [[nodiscard]] auto getTrack(const double m) const { return interp_->interpConstY(m); }

        /**
         * @brief Return the isochrone at a given time
         * @param t The time of the isochrone
         * @return A vector of unique_ptr to Interpolator1D's describing the isochrone
         * @details
         * Note that this method returns a vector of Interpolator1D objects
         * rather than a single one because for non-monotonic tracks there
         * may be multiple disjoint segments to the isochrone.
         */
        [[nodiscard]] auto getIsochrone(const double t) const { return interp_->interpConstX(t); }

    private:

        // Track data
        M2DPtr interp_; /**< Interpolator for the tracks */
        double FeH_;    /**< [Fe/H] value of this set of tracks */
        double AFe_;    /**< [alpha/Fe] value of this set of tracks, or quiet_NaN if not available */
        double vVcrit_; /**< v/vcrit value of this set of tracks, or quiet_NaN if not available */

    };

} // namespace tracks

#endif // TRACKS2D_HPP