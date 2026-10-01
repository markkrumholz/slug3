/**
 * @file FeedbackCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions used by stellar feedback classes
 * @date 2026-09-30
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef FEEDBACKCOMMONS_HPP
#define FEEDBACKCOMMONS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace feedback
{
    /**
     * @brief enum of the available Wolf-Rayet wind velocity models
     * @details
     * Selects how Winds::vWindWR() computes a Wolf-Rayet star's
     * terminal wind velocity -- see that method's own comment for
     * what each one does.
     */
    enum class WRwindModel : std::uint8_t
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore on each enumerator matches yields::Channel's own identical convention (see its own comment)
        none_,         /**< No WR wind: velocity is zero */
        lOverc_,       /**< Single-scattering momentum limit, v = L / (mdot c) */
        nugisLamers00_ /**< Nugis & Lamers (2000) empirical prescription */
        // NOLINTEND(readability-identifier-naming)
    };

    /**
     * @brief Names of each WRwindModel, as used by the input deck's
     *   feedback.wr_winds key and the Python bindings, in the same
     *   order as the enumerators themselves
     */
    constexpr std::array<std::string_view, 3> wrWindModelStr{
        "none",
        "l_over_c",
        "nugis_lamers_00"
    };

    /**
     * @brief Translate a WR wind model name into a WRwindModel
     * @param name One of the entries of wrWindModelStr
     * @return The corresponding WRwindModel
     * @throws std::invalid_argument if name is not one of the entries
     *   of wrWindModelStr
     */
    inline auto wrWindModelFromString(const std::string_view name) -> WRwindModel
    {
        const auto* const it = std::ranges::find(wrWindModelStr, name);
        if (it == wrWindModelStr.end())
        {
            throw std::invalid_argument("'" + std::string(name) +
                "' is not a recognized WR wind model (expected none, "
                "l_over_c, or nugis_lamers_00)");
        }
        return static_cast<WRwindModel>(std::distance(wrWindModelStr.begin(), it));
    }

    /**
     * @brief Translate a WRwindModel into its name
     * @param model The model to translate
     * @return The entry of wrWindModelStr corresponding to model
     */
    inline auto wrWindModelToString(const WRwindModel model) -> std::string_view
    {
        return wrWindModelStr.at(static_cast<std::size_t>(model));
    }

    /**
     * @brief enum of the available O and B star wind velocity models
     * @details
     * Selects how Winds::vWindOB() computes an O or B star's terminal
     * wind velocity -- see that method's own comment for what each
     * one does.
     */
    enum class OBwindModel : std::uint8_t
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore on each enumerator matches yields::Channel's own identical convention (see its own comment)
        none_,        /**< No OB wind: velocity is zero */
        vink01_,      /**< Vink, de Koter, & Lamers (2001) prescription */
        vinkSander21_ /**< Vink & Sander (2021) prescription */
        // NOLINTEND(readability-identifier-naming)
    };

    /**
     * @brief Names of each OBwindModel, as used by the input deck's
     *   feedback.ob_winds key and the Python bindings, in the same
     *   order as the enumerators themselves
     */
    constexpr std::array<std::string_view, 3> obWindModelStr{
        "none",
        "vink_01",
        "vink_sander_21"
    };

    /**
     * @brief Translate an OB wind model name into an OBwindModel
     * @param name One of the entries of obWindModelStr
     * @return The corresponding OBwindModel
     * @throws std::invalid_argument if name is not one of the entries
     *   of obWindModelStr
     */
    inline auto obWindModelFromString(const std::string_view name) -> OBwindModel
    {
        const auto* const it = std::ranges::find(obWindModelStr, name);
        if (it == obWindModelStr.end())
        {
            throw std::invalid_argument("'" + std::string(name) +
                "' is not a recognized OB wind model (expected none, "
                "vink_01, or vink_sander_21)");
        }
        return static_cast<OBwindModel>(std::distance(obWindModelStr.begin(), it));
    }

    /**
     * @brief Translate an OBwindModel into its name
     * @param model The model to translate
     * @return The entry of obWindModelStr corresponding to model
     */
    inline auto obWindModelToString(const OBwindModel model) -> std::string_view
    {
        return obWindModelStr.at(static_cast<std::size_t>(model));
    }

    /**
     * @brief enum of the available AGB star wind velocity models
     * @details
     * Selects how Winds::vWindAGB() computes an AGB star's terminal
     * wind velocity -- see that method's own comment for what each
     * one does.
     */
    enum class AGBwindModel : std::uint8_t
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore on each enumerator matches yields::Channel's own identical convention (see its own comment)
        none_, /**< No AGB wind: velocity is zero */
        slug2_ /**< The prescription used by slug2 (Elitzur & Ivezic 2001 scaling, Goldman et al. 2017 normalization) */
        // NOLINTEND(readability-identifier-naming)
    };

    /**
     * @brief Names of each AGBwindModel, as used by the input deck's
     *   feedback.agb_winds key and the Python bindings, in the same
     *   order as the enumerators themselves
     */
    constexpr std::array<std::string_view, 2> agbWindModelStr{
        "none",
        "slug2"
    };

    /**
     * @brief Translate an AGB wind model name into an AGBwindModel
     * @param name One of the entries of agbWindModelStr
     * @return The corresponding AGBwindModel
     * @throws std::invalid_argument if name is not one of the entries
     *   of agbWindModelStr
     */
    inline auto agbWindModelFromString(const std::string_view name) -> AGBwindModel
    {
        const auto* const it = std::ranges::find(agbWindModelStr, name);
        if (it == agbWindModelStr.end())
        {
            throw std::invalid_argument("'" + std::string(name) +
                "' is not a recognized AGB wind model (expected none or slug2)");
        }
        return static_cast<AGBwindModel>(std::distance(agbWindModelStr.begin(), it));
    }

    /**
     * @brief Translate an AGBwindModel into its name
     * @param model The model to translate
     * @return The entry of agbWindModelStr corresponding to model
     */
    inline auto agbWindModelToString(const AGBwindModel model) -> std::string_view
    {
        return agbWindModelStr.at(static_cast<std::size_t>(model));
    }

    /**
     * @brief enum of the available wind velocity models for all stars
     *   not covered by WRwindModel, OBwindModel, or AGBwindModel
     * @details
     * Selects how Winds::vWindOther() computes such a star's terminal
     * wind velocity -- see that method's own comment for what each
     * one does.
     */
    enum class OtherwindModel : std::uint8_t
    {
        // NOLINTBEGIN(readability-identifier-naming) -- trailing underscore on each enumerator matches yields::Channel's own identical convention (see its own comment)
        none_, /**< No wind: velocity is zero */
        vesc_  /**< Wind velocity equal to the surface escape speed */
        // NOLINTEND(readability-identifier-naming)
    };

    /**
     * @brief Names of each OtherwindModel, as used by the input deck's
     *   feedback.other_winds key and the Python bindings, in the same
     *   order as the enumerators themselves
     */
    constexpr std::array<std::string_view, 2> otherWindModelStr{
        "none",
        "vesc"
    };

    /**
     * @brief Translate an other-star wind model name into an OtherwindModel
     * @param name One of the entries of otherWindModelStr
     * @return The corresponding OtherwindModel
     * @throws std::invalid_argument if name is not one of the entries
     *   of otherWindModelStr
     */
    inline auto otherWindModelFromString(const std::string_view name) -> OtherwindModel
    {
        const auto* const it = std::ranges::find(otherWindModelStr, name);
        if (it == otherWindModelStr.end())
        {
            throw std::invalid_argument("'" + std::string(name) +
                "' is not a recognized other-star wind model (expected none or vesc)");
        }
        return static_cast<OtherwindModel>(std::distance(otherWindModelStr.begin(), it));
    }

    /**
     * @brief Translate an OtherwindModel into its name
     * @param model The model to translate
     * @return The entry of otherWindModelStr corresponding to model
     */
    inline auto otherWindModelToString(const OtherwindModel model) -> std::string_view
    {
        return otherWindModelStr.at(static_cast<std::size_t>(model));
    }

} // namespace feedback

#endif // FEEDBACKCOMMONS_HPP
