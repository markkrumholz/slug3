/**
 * @file Extinct.cpp
 * @author Mark Krumholz
 * @brief Implementation of Extinct
 * @date 2026-08-04
 * @details
 * Out-of-line so this file, not Extinct.hpp, is the one that needs
 * io::SimControls's complete type -- SimControls.hpp itself includes
 * Extinct.hpp (for its own extinct_ member), so Extinct.hpp can only
 * forward-declare io::SimControls without creating a header cycle.
 * Mirrors Specsyn.cpp's identical situation/comment exactly.
 */

#include "Extinct.hpp"
#include "../io/SimControls.hpp"
#include <algorithm>
#include <vector>

auto extinct::Extinct::wlObs() const -> std::vector<double>
{
    const double z = controls_.z();
    std::vector<double> wlObs(wl_.size());
    std::ranges::transform(wl_, wlObs.begin(),
        [z](const double wl) -> double { return wl * (1.0 + z); });
    return wlObs;
}
