/**
 * @file Tracks3D.hpp
 * @author Mark Krumholz
 * @brief A class to represent a 3D set of stellar tracks
 * @date 2024-07-10
 */

#ifndef TRACKS3D_HPP
#define TRACKS3D_HPP

#include "TrackCommons.hpp"
#include "../interpolation/Mesh3DInterpolator.hpp"
#include "H5Ipublic.h"
#include <cstddef>
#include <memory>
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
         * @brief Construct a Tracks3D object from tracks on disk
         * @param registryName Name of the track registry file
         * @param trackName Name of track set
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param vvcrit Rotation rate v/vcrit
         * @param afe Value of [alpha/Fe] 
        */
        Tracks3D(
            const std::string& registryName,
            const std::string& trackName,
            double fehMin, 
            double fehMax,
            double vvcrit = 0.0,
            double afe = 0.0);
        virtual ~Tracks3D() = default;
        Tracks3D(const Tracks3D&) = delete;
        Tracks3D(Tracks3D&&) = default;
        auto operator=(const Tracks3D&) -> Tracks3D& = delete;
        auto operator=(Tracks3D&&) -> Tracks3D& = delete;

    private:

        // Track data
        M3DPtr interp_;           /**< Interpolator for the tracks */
        std::vector<double> FeH_; /**< [Fe/H] values for all tracks */
        double AFe_;              /**< [alpha/Fe] value of this set of tracks, or quiet_NaN if not available */
        double vVcrit_;           /**< v/vcrit value of this set of tracks, or quiet_NaN if not available */

    };

} // namespace tracks

#endif // TRACKS3D_HPP
