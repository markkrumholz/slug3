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

} // namespace feedback

#endif // FEEDBACKCOMMONS_HPP
