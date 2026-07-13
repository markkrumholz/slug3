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
#include <string_view>
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
         * @brief Numpy-style docstring for the Python constructor binding
         */
        static constexpr std::string_view constructorDocstring = R"doc(Construct a Tracks2D object from tracks on disk.

Parameters
----------
trackName : str
    Name of the track set within the registry.
feh : float, optional
    [Fe/H] value of the desired track. Default is 0.0.
vvcrit : float, optional
    Rotation rate v/vcrit of the desired track. Default is 0.0.
afe : float, optional
    [alpha/Fe] value of the desired track. Default is 0.0.
registryName : str, optional
    Path to the track registry file. Default is the package's default
    registry (data/tracks/tracks.toml).

Throws
------
RuntimeError
    If no track in trackName matches feh, vvcrit, and afe exactly.)doc";

        /**
         * @brief Construct a Tracks2D object from tracks on disk
         * @param trackName Name of track set
         * @param feh [Fe/H] value
         * @param vvcrit Rotation rate v/vcrit
         * @param afe Value of [alpha/Fe]
         * @param registryName Name of the track registry file
         * @details
         * Uses findTrack to locate the unique track in track set
         * trackName matching feh, vvcrit, and afe, and throws a
         * runtime error if no such track can be found. Tracks are
         * stored for each mass, and the number of time points may not
         * be the same for every mass. Tracks with fewer time points
         * will be padded at the end so that the final set of times and
         * stellar properties are square arrays.
        */
        Tracks2D(
            const std::string& trackName,
            double feh = 0.0,
            double vvcrit = defaultVVcrit,
            double afe = defaultAFe,
            const std::string& registryName = defaultRegistry);
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
        static constexpr std::string_view feHDocstring = R"doc(Return the [Fe/H] value of this set of tracks.

Returns
-------
feh : float
    [Fe/H] value of this set of tracks.)doc";

        /**
         * @brief Return the [Fe/H] value of this set of tracks
         * @return [Fe/H] value of this set of tracks
         */
        [[nodiscard]] auto feH() const { return FeH_; }

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
         * @brief Numpy-style docstring for the Python starLifetime binding
         */
        static constexpr std::string_view starLifetimeDocstring = R"doc(Return the lifetime of a star of a given mass.

Parameters
----------
m : float
    Stellar mass, in Msun.

Returns
-------
logt : float
    log10(time / yr) at which a star of mass m ends this track.)doc";

        /**
         * @brief Return the lifetime of a star of specified mass
         * @param m Mass of the star whose lifetime should be returned
         * @return The lifetime of a star of mass m
         */
       [[nodiscard]] auto starLifetime(const double m) const { return interp_->xMax(m); }

       /**
        * @brief Numpy-style docstring for the Python liveMassRange binding
        */
       static constexpr std::string_view liveMassRangeDocstring = R"doc(Return the range(s) of stellar mass alive at a given time.

Parameters
----------
t : float
    log10(time / yr) at which to evaluate.

Returns
-------
ranges : list of tuple of float
    A list of (mMin, mMax) pairs giving the mass ranges alive at time
    t. Usually contains a single pair, but may contain more than one
    for non-monotonic tracks.)doc";

       /**
        * @brief Return the range of stellar masses that are alive at a given time
        * @param t Time at which to evaluate
        * @return The range of stellar masses alive at the given time
        */
       [[nodiscard]] auto liveMassRange(const double t) const { return interp_->yLim(t); }

        /**
         * @brief Numpy-style docstring for the Python getTrack binding
         */
        static constexpr std::string_view getTrackDocstring = R"doc(Return the track for a star of a given mass.

Parameters
----------
m : float
    Stellar mass, in Msun.

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
         * @brief Return the track for a star of a given mass
         * @param m Mass of the star whose track should be computed
         * @return An unique_ptr to an Interpolator1D describing the track for a given mass
         */
        [[nodiscard]] auto getTrack(const double m) const { return interp_->interpConstY(m); }

        /**
         * @brief Numpy-style docstring for the Python getIsochrone binding
         */
        static constexpr std::string_view getIsochroneDocstring = R"doc(Return the isochrone at a given log time.

Parameters
----------
logT : float
    log10(time / yr) of the isochrone.

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
         * @brief Return the isochrone at a given log time
         * @param logT The log10(time) of the isochrone
         * @return A vector of unique_ptr to Interpolator1D's describing the isochrone
         * @details
         * Note that this method returns a vector of Interpolator1D objects
         * rather than a single one because for non-monotonic tracks there
         * may be multiple disjoint segments to the isochrone.
         */
        [[nodiscard]] auto getIsochrone(const double logT) const { return interp_->interpConstX(logT); }

    private:

        // Track data
        M2DPtr interp_; /**< Interpolator for the tracks */
        double FeH_;    /**< [Fe/H] value of this set of tracks */
        double AFe_;    /**< [alpha/Fe] value of this set of tracks, or quiet_NaN if not available */
        double vVcrit_; /**< v/vcrit value of this set of tracks, or quiet_NaN if not available */

    };

} // namespace tracks

#endif // TRACKS2D_HPP