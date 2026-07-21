/**
 * @file SpecsynLibChained.cpp
 * @author Mark Krumholz
 * @brief Implementation of SpecsynLibChained.hpp
 * @date 2026-07-21
 */

#include "SpecsynLibChained.hpp"
#include "Specsyn.hpp"
#include "SpecsynCommons.hpp"
#include "SpecsynLib.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specsyn
{
    SpecsynLibChained::SpecsynLibChained(
        const std::vector<std::string>& spectraName,
        const double fehMin,
        const double fehMax,
        const double afe,
        const double cfe,
        const std::vector<double>& microTurb,
        const double r,
        const std::string& registryName,
        const double z) :
        Specsyn(z)
    {
        if (spectraName.empty())
        {
            throw std::runtime_error(
                "SpecsynLibChained: spectraName must contain at least one library name");
        }
        if (!microTurb.empty() && microTurb.size() != spectraName.size())
        {
            throw std::runtime_error(
                "SpecsynLibChained: microTurb must be empty or have the same "
                "number of entries as spectraName");
        }

        const size_t n = spectraName.size();
        libs_.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            const double mt = microTurb.empty() ? defaultMicroTurb : microTurb[i];
            if (i + 1 < n)
            {
                libs_.push_back(std::make_unique<SpecsynLib<OOBPolicy::silent>>(
                    spectraName[i], fehMin, fehMax, afe, cfe, mt, r, registryName, z));
            }
            else
            {
                libs_.push_back(std::make_unique<SpecsynLib<OOBPolicy::Throw>>(
                    spectraName[i], fehMin, fehMax, afe, cfe, mt, r, registryName, z));
            }
        }

        wl_ = libs_.front()->wl();
    }

    auto SpecsynLibChained::spec(const StarData& props, const double feh) const -> std::vector<double>
    {
        for (size_t i = 0; i + 1 < libs_.size(); ++i)
        {
            auto result = libs_[i]->spec(props, feh);
            if (!result.empty()) { return result; }
        }
        return libs_.back()->spec(props, feh);
    }

} // namespace specsyn
