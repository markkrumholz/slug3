/**
 * @file YieldChannel.hpp
 * @author Mark Krumholz
 * @brief A class to represent the nucleosynthetic yield data for one channel
 * @date 2026-09-12
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDCHANNEL_HPP
#define YIELDCHANNEL_HPP

#include "YieldCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
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
         * @param channel Which nucleosynthetic channel to load (e.g. Channel::ccsn_)
         * @param modelName Name of the yield model (e.g. "sukhbold16")
         * @param fehMin Minimum [Fe/H] value
         * @param fehMax Maximum [Fe/H] value
         * @param registryName Name of the yield registry file
         * @throws std::runtime_error if channel/modelName is not found
         *   in the registry, if fehMin or fehMax lies outside the
         *   [Fe/H] range actually available for that channel/model, or
         *   if the underlying HDF5 file cannot be read or is malformed
         */
        YieldChannel(
            Channel channel,
            const std::string& modelName,
            double fehMin,
            double fehMax,
            const std::string& registryName = defaultRegistry);

        ~YieldChannel() = default;
        YieldChannel(const YieldChannel&) = default;
        YieldChannel(YieldChannel&&) = default;
        auto operator=(const YieldChannel&) -> YieldChannel& = default;
        auto operator=(YieldChannel&&) -> YieldChannel& = default;

        // Observers

        /**
         * @brief Return which nucleosynthetic channel this object holds
         * @return The channel passed to the constructor
         */
        [[nodiscard]] auto channel() const { return channel_; }

        /**
         * @brief Return the stellar masses this channel's yields are tabulated at
         * @return A const reference to the masses (Msun), ascending
         */
        [[nodiscard]] auto masses() const -> const std::vector<double>& { return masses_; }

        /**
         * @brief Return the atomic number of each isotope this channel's yields are tabulated for
         * @return A const reference to the atomic numbers, in the same
         *   order as yldA() and yld()'s own third axis
         */
        [[nodiscard]] auto yldZ() const -> const std::vector<unsigned int>& { return yldZ_; }

        /**
         * @brief Return the mass number of each isotope this channel's yields are tabulated for
         * @return A const reference to the mass numbers, in the same
         *   order as yldZ() and yld()'s own third axis
         */
        [[nodiscard]] auto yldA() const -> const std::vector<unsigned int>& { return yldA_; }

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
         *   yldZ().size()) into yieldData_, i.e. yld()[f, m, i] is the
         *   yield (Msun) of isotope i, from a star of mass masses()[m],
         *   at metallicity feH()[f]
         * @details
         * Built fresh from yieldData_ on every call, rather than cached
         * as its own member alongside it, so that copying or moving a
         * YieldChannel (both merely copy/move yieldData_ itself) can
         * never leave a stale view pointing at another object's own,
         * separately-relocated backing storage -- mirroring
         * Mesh3DInterpolator's own x()/y()/z()/f() accessors, built the
         * same way for the same reason.
         */
        [[nodiscard]] auto yld() const -> Array3D
        {
            return Array3D(yieldData_.data(), feH_.size(), masses_.size(), yldZ_.size());
        }

    private:

        Channel channel_;                  /**< Which nucleosynthetic channel this is */
        std::vector<double> masses_;       /**< Stellar masses (Msun) this channel's yields are tabulated at */
        std::vector<unsigned int> yldZ_;   /**< Atomic number of each isotope */
        std::vector<unsigned int> yldA_;   /**< Mass number of each isotope */
        std::vector<double> feH_;          /**< [Fe/H] values for all yields */ // NOLINT(readability-identifier-naming)
        std::vector<double> yieldData_;    /**< Backing storage for yld() -- see its own comment */

    };

} // namespace yields

#endif // YIELDCHANNEL_HPP
