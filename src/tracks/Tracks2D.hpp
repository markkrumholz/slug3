/**
 * @file Tracks2D.hpp
 * @author Mark Krumholz
 * @brief A class to represent a 2D set of stellar tracks
 * @date 2024-07-09
 */

#ifndef TRACK2D_HPP
#define TRACK2D_HPP

#include "../interpolation/Mesh2DInterpolator.hpp"
#include "hdf5.h"
#include <memory>

/**
 * @brief A namespace to hold functions dealing with stellar tracks
*/
namespace tracks
{

    constexpr size_t nTrackQty = 9;  /**< Number of quantities at a given track point */

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
        Tracks2D(const hid_t grp, const size_t ntMin = 0);
        virtual ~Tracks2D() = default;
        Tracks2D(const Tracks2D&) = delete;
        Tracks2D(Tracks2D&&) = default;
        auto operator=(const Tracks2D&) -> Tracks2D& = delete;
        auto operator=(Tracks2D&&) -> Tracks2D& = delete;

    private:

        // Track data
        std::unique_ptr<interp::Mesh2DInterpolator<nTrackQty>> interp_;

    };

} // namespace tracks

#endif // TRACK2D