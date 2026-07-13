/**
 * @file Tracks3D.hpp
 * @author Mark Krumholz
 * @brief A class to represent a 3D set of stellar tracks
 * @date 2026-07-10
 */

#ifndef TRACKS3D_HPP
#define TRACKS3D_HPP

#include "../interpolation/Mesh2DInterpolator.hpp"
#include "../interpolation/Mesh3DInterpolator.hpp"
#include "TrackCommons.hpp"
#include "Tracks2D.hpp"
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tracks
{

    /**
     * @class Tracks3D
     * @brief A class representing a 3D set of stellar tracks
     */
    class Tracks3D
    {
    public:

        // Shorten type name
        using M3DPtr = std::unique_ptr<
            interp::Mesh3DInterpolator<
            static_cast<size_t>(FieldIdx::nTrackQty)>>;

        // Constructors and destructors

        /**
         * @brief Numpy-style docstring for the Python constructor binding
         */
        static constexpr std::string_view constructorDocstring = R"doc(Construct a Tracks3D object from tracks on disk.

Parameters
----------
trackName : str
    Name of the track set within the registry.
fehMin : float
    Minimum [Fe/H] value to include.
fehMax : float
    Maximum [Fe/H] value to include.
vvcrit : float, optional
    Rotation rate v/vcrit of the desired tracks. Default is 0.0.
afe : float, optional
    [alpha/Fe] value of the desired tracks. Default is 0.0.
registryName : str, optional
    Path to the track registry file. Default is the package's default
    registry (data/tracks/tracks.toml).

Throws
------
RuntimeError
    If no tracks in trackName match vvcrit and afe within the
    requested [Fe/H] range.)doc";

        /**
         * @brief Construct a Tracks3D object from tracks on disk
         * @param trackName Name of track set
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param vvcrit Rotation rate v/vcrit
         * @param afe Value of [alpha/Fe]
         * @param registryName Name of the track registry file
        */
        Tracks3D(
            const std::string& trackName,
            double fehMin,
            double fehMax,
            double vvcrit = 0.0,
            double afe = 0.0,
            const std::string& registryName = defaultRegistry);
        /**
         * @brief Construct an empty, invalid Tracks3D object
         */
        Tracks3D() : 
            AFe_(std::numeric_limits<double>::quiet_NaN()),
            vVcrit_(std::numeric_limits<double>::quiet_NaN())
        { }
        virtual ~Tracks3D() = default;
        Tracks3D(const Tracks3D&) = delete;
        Tracks3D(Tracks3D&&) = default;
        auto operator=(const Tracks3D&) -> Tracks3D& = delete;
        auto operator=(Tracks3D&&) -> Tracks3D& = delete;

        // Observers

        /**
         * @brief Numpy-style docstring for the Python mMin binding
         */
        static constexpr std::string_view mMinDocstring = R"doc(Return the minimum mass spanned by this set of tracks.

Returns
-------
mmin : float
    Minimum mass, in Msun.)doc";

        /**
         * @brief Return the minimum mass in the tracks
         * @return Minimum mass in the tracks
         */
        [[nodiscard]] auto mMin() const { return interp_->yMin(); }

        /**
         * @brief Numpy-style docstring for the Python mMax binding
         */
        static constexpr std::string_view mMaxDocstring = R"doc(Return the maximum mass spanned by this set of tracks.

Returns
-------
mmax : float
    Maximum mass, in Msun.)doc";

        /**
         * @brief Return the maximum mass in the tracks
         * @return Maximum mass in the tracks
         */
        [[nodiscard]] auto mMax() const { return interp_->yMax(); }

        /**
         * @brief Numpy-style docstring for the Python logTMin binding
         */
        static constexpr std::string_view logTMinDocstring = R"doc(Return the minimum log10(time) spanned by this set of tracks.

Returns
-------
logtmin : float
    Minimum log10(time / yr).)doc";

        /**
         * @brief Return the minimum log of time in the tracks
         * @return Minimum log10(time) in the tracks
         */
        [[nodiscard]] auto logTMin() const { return interp_->xMin(); }

        /**
         * @brief Numpy-style docstring for the Python logTMax binding
         */
        static constexpr std::string_view logTMaxDocstring = R"doc(Return the maximum log10(time) spanned by this set of tracks.

Returns
-------
logtmax : float
    Maximum log10(time / yr).)doc";

        /**
         * @brief Return the maximum log of time in the tracks
         * @return Maximum log10(time) in the tracks
         */
        [[nodiscard]] auto logTMax() const { return interp_->xMax(); }

        /**
         * @brief Numpy-style docstring for the Python feH binding
         */
        static constexpr std::string_view feHDocstring = R"doc(Return the [Fe/H] values spanned by this set of tracks.

Returns
-------
feh : list of float
    [Fe/H] values spanned by this set of tracks.)doc";

        /**
         * @brief Return the [Fe/H] values spanned by this set of tracks
         * @return A const reference to the [Fe/H] values spanned by this set of tracks
         */
        [[nodiscard]] auto feH() const -> const std::vector<double>& { return FeH_; }

        /**
         * @brief Numpy-style docstring for the Python aFe binding
         */
        static constexpr std::string_view aFeDocstring = R"doc(Return the [alpha/Fe] value of this set of tracks.

Returns
-------
afe : float
    [alpha/Fe] value of this set of tracks, or NaN if not available.)doc";

        /**
         * @brief Return the [alpha/Fe] value of this set of tracks
         * @return [alpha/Fe] value of this set of tracks, or quiet_NaN if not available
         */
        [[nodiscard]] auto aFe() const { return AFe_; }

        /**
         * @brief Numpy-style docstring for the Python vVcrit binding
         */
        static constexpr std::string_view vVcritDocstring = R"doc(Return the v/vcrit value of this set of tracks.

Returns
-------
vvcrit : float
    v/vcrit value of this set of tracks, or NaN if not available.)doc";

        /**
         * @brief Return the v/vcrit value of this set of tracks
         * @return v/vcrit value of this set of tracks, or quiet_NaN if not available
         */
        [[nodiscard]] auto vVcrit() const { return vVcrit_; }

        // Calculation methods

        /**
         * @brief Return the lifetime of a star of specified mass and [Fe/H]
         * @param m Mass of the star whose lifetime should be returned
         * @param feh [Fe/H] value of the tracks to use
         * @return The lifetime of a star of mass m at the given [Fe/H]
         */
       [[nodiscard]] auto starLifetime(const double m, const double feh) const
       { return interp_->sliceConstZ(feh).xMax(m); }

       /**
        * @brief Return the range of stellar masses that are alive at a given time and [Fe/H]
        * @param t Time at which to evaluate
        * @param feh [Fe/H] value of the tracks to use
        * @return The range of stellar masses alive at the given time and [Fe/H]
        */
       [[nodiscard]] auto liveMassRange(const double t, const double feh) const
       { return interp_->sliceConstZ(feh).yLim(t); }

        /**
         * @brief Numpy-style docstring for the Python getTrack binding
         */
        static constexpr std::string_view getTrackDocstring = R"doc(Return the track for a star of a given mass and [Fe/H].

Parameters
----------
m : float
    Stellar mass, in Msun.
feh : float
    [Fe/H] value of the tracks to use.

Returns
-------
interp : Interpolator1D
    An Interpolator1D object containing the interpolating track,
    parameterized by log10(time / yr).

Throws
------
RuntimeError
    If m is less than mMin() or greater than mMax().)doc";

        /**
         * @brief Return the track for a star of a given mass and [Fe/H]
         * @param m Mass of the star whose track should be computed
         * @param feh [Fe/H] value of the tracks to use
         * @return An unique_ptr to an Interpolator1D describing the track for a given mass
         */
        [[nodiscard]] auto getTrack(const double m, const double feh) const
        { return interp_->sliceConstZ(feh).interpConstY(m); }

        /**
         * @brief Numpy-style docstring for the Python getIsochrone binding
         */
        static constexpr std::string_view getIsochroneDocstring = R"doc(Return the isochrone at a given log time and [Fe/H].

Parameters
----------
logT : float
    log10(time / yr) of the isochrone.
feh : float
    [Fe/H] value of the tracks to use.

Returns
-------
isochrone : list of Interpolator1D
    A list of Interpolator1D objects, each parameterized by mass,
    giving the isochrone. More than one entry indicates disjoint
    segments, which can occur for non-monotonic tracks; an empty list
    indicates that logT lies outside the tracks' time range, or that
    it touches the tracks at only a single mass (too few points to
    interpolate).)doc";

        /**
         * @brief Return the isochrone at a given log time and [Fe/H]
         * @param logT The log10(time) of the isochrone
         * @param feh [Fe/H] value of the tracks to use
         * @return A vector of unique_ptr to Interpolator1D's describing the isochrone
         * @details
         * Note that this method returns a vector of Interpolator1D objects
         * rather than a single one because for non-monotonic tracks there
         * may be multiple disjoint segments to the isochrone.
         */
        [[nodiscard]] auto getIsochrone(const double logT, const double feh) const
        { return interp_->sliceConstZ(feh).interpConstX(logT); }

        /**
         * @brief Construct a Tracks2D object for a slice at fixed [Fe/H]
         * @param feh [Fe/H] value at which to slice
         * @return A Tracks2D object representing the (mass, time) slice at the given [Fe/H]
         * @details
         * Unlike getTrack() and getIsochrone(), which evaluate the cached
         * slice owned by this Tracks3D object, this method builds a new,
         * independent Mesh2DInterpolator (via Mesh3DInterpolator::sliceConstZCopy)
         * and uses it to construct a Tracks2D object that owns its own
         * memory, and so remains valid even after this Tracks3D object is
         * destroyed or after later calls to getTrack()/getIsochrone() with
         * a different [Fe/H] value.
         */
        [[nodiscard]] auto sliceConstFeH(const double feh) const -> Tracks2D
        {
            Tracks2D::M2DPtr m2d = std::make_unique<
                interp::Mesh2DInterpolator<static_cast<size_t>(FieldIdx::nTrackQty)>>(
                interp_->sliceConstZCopy(feh));
            return { std::move(m2d), feh, AFe_, vVcrit_ };
        }

    private:

        // Track data
        M3DPtr interp_;           /**< Interpolator for the tracks */
        std::vector<double> FeH_; /**< [Fe/H] values for all tracks */  // NOLINT(readability-identifier-naming)
        double AFe_;              /**< [alpha/Fe] value of this set of tracks, or quiet_NaN if not available */ // NOLINT(readability-identifier-naming)
        double vVcrit_;           /**< v/vcrit value of this set of tracks, or quiet_NaN if not available */   

    };

} // namespace tracks

#endif // TRACKS3D_HPP
