/**
 * @file SimPhysics.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimPhysics
 * @date 2026-07-12
 */

#include "SimPhysics.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include "../pdfs/PDFSegmentPowerlaw.hpp"
#include "../specsyn/SpecsynBlackbody.hpp"
#include "../specsyn/SpecsynCommons.hpp"
#include "../specsyn/SpecsynLibChained.hpp"
#include "../specsyn/SpecsynLibNoWind.hpp"
#include "../specsyn/SpecsynLibWR.hpp"
#include "../specsyn/SpecsynUtils.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/ParseUtils.hpp"
#include "SimControls.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

// SimPhysics constructor
io::SimPhysics::SimPhysics(const toml::table& inputDeck, SimControls::SimType simType)
{
    // Read IMF, CMF, and FeH
    imf_ = utils::initPDFFromKey(inputDeck, "stars.IMF",
        (std::filesystem::path("data") / std::filesystem::path("imfs")).string());
    cmf_ = utils::initPDFFromKey(inputDeck, "clusters.CMF");
    fehDist_ = utils::initPDFFromKey(inputDeck, "stars.FeH");

    // Read the spectral synthesis model to use, if any -- spectra.model
    // is optional, since not every simulation needs spectra computed.
    // Needs fehDist_ (just set above) to pick the [Fe/H] range a
    // library-based model is loaded over.
    readSpectra(inputDeck);

    // In a galaxy simulation, read CLF and SFR
    if (simType == SimControls::SimType::galaxy)
    {
        // CLF
        clf_ = utils::initPDFFromKey(inputDeck, "clusters.CLF");

        // SFR -- this requires special handling because here
        // a numerical value is not interpreted as a delta function
        // but as the normalization for a non-normalized PDF that is
        // constant in time
        const std::optional<double> sfr = 
            inputDeck.at_path("galaxy.sfr").value<double>();
        if (sfr.has_value())
        {
            // We have been given a numerical value, so construct a
            // constant PDF from t = 0 to T for a big number T, with
            // the weight set to T / sfr so that the mass of stars
            // formed in any time interval dt comes out to sfr * dt.
            const double tMax = std::numeric_limits<double>::max() *
                std::min(sfr.value(), 1.0);
            const double wgt = std::numeric_limits<double>::max() /
                std::max(sfr.value(), 1.0);
            auto pl = std::make_unique<pdfs::PDFSegmentPowerlaw>(0.0, tMax, 0.0);
            sfr_ = pdfs::PDF(std::move(pl), wgt);
        }
        else
        {
            // We have been given a file
            const std::optional<std::string> sfrFile =
                inputDeck.at_path("galaxy.sfr").value<std::string>();
            if (sfrFile.has_value())
            {            
                sfr_ = pdfs::parsePDFDescriptor(sfrFile.value());
            }
            else
            {
                throw std::runtime_error(
                    "SimPhysics: invalid entry for galaxy.sfr");
            }
        }
    }

    // Read the tracks
    readTracks(inputDeck);

    // Warn if the tracks don't extend down to the IMF's minimum mass:
    // stars in that gap have no track data, so they end up being
    // treated as having zero luminosity when spectra are computed
    // (see Cluster::computeSpec)
    if (tracks_.mMin() > imf_.getMin())
    {
        std::cout << "slug: warning: minimum mass in selected tracks is "
            << tracks_.mMin() << " but IMF minimum mass is " << imf_.getMin()
            << "; stars with masses from " << imf_.getMin() << " to "
            << tracks_.mMin() << " will be treated as having zero luminosity\n";
    }

    // If this simulation has a fixed [Fe/H], precompute the slice at
    // that value once here, up front, so that Cluster objects can
    // share it for the lifetime of the simulation instead of each
    // computing their own copy or racing to populate Tracks3D's
    // internal cache
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
    }

    // Read minimum stochastic mass
    auto minSM = utils::getTOMLKeyWithError<double>(inputDeck, "stars.min_stoch_mass");
    if (minSM.has_value()) 
    { 
        minStochMass_ = minSM.value();
        fracStochMass_ = imf_.integral(minStochMass_, imf_.getMax());
    }
}

// Track reader
void io::SimPhysics::readTracks(const toml::table& inputDeck)
{
    // Get required tracks key
    auto trackName = utils::getTOMLKeyWithError<std::string>(inputDeck, "stars.tracks", true);

    // Check for optional parameters
    auto registryName = utils::getTOMLKeyWithError<std::string>(inputDeck, "stars.track_registry");
    auto vvcrit = utils::getTOMLKeyWithError<double>(inputDeck, "stars.v_vcrit");
    auto afe = utils::getTOMLKeyWithError<double>(inputDeck, "stars.alphaFe");

    // Construct tracks from input data
    tracks_ = tracks::Tracks3D(
        trackName.value(), // NOLINT(bugprone-unchecked-optional-access) -- we verified this was valid a few lines ago
        fehDist_.getMin(),
        fehDist_.getMax(),
        vvcrit.value_or(tracks::defaultVVcrit),
        afe.value_or(tracks::defaultAFe),
        registryName.value_or(tracks::defaultRegistry));
}

// Spectral synthesizer reader
void io::SimPhysics::readSpectra(const toml::table& inputDeck)
{
    // Check for an optional alternative registry
    auto registryNameInput = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "spectra.registry");
    const std::string registryName = registryNameInput.value_or(specsyn::defaultRegistry);

    // spectra.model is optional -- if it is absent, this simulation
    // computes no spectra, and specsyn_ stays null
    const auto modelNode = inputDeck.at_path("spectra.model");
    if (!modelNode) { return; }

    // A single string names one model directly; anything else must be
    // an array of strings, chained together via SpecsynLibChained
    if (const auto model = modelNode.value<std::string>(); model.has_value())
    {
        // "blackbody" is a special value, used for testing, that does
        // not require a spectral library at all
        if (model.value() == "blackbody")
        {
            specsyn_ = std::make_unique<specsyn::SpecsynBlackbody>();
            return;
        }

        // Look the model up in the registry, and use its WR_grid
        // entry (if any) to decide which SpecsynLib specialization
        // applies: Wolf-Rayet libraries -- parameterized by
        // transformed radius and stellar temperature rather than
        // logg and Teff -- need SpecsynLibWR, every other library
        // needs SpecsynLibNoWind
        auto [registry, registryPath] = specsyn::parseRegistry(registryName);
        const auto modelEntry = registry.at_path(model.value());
        if (!modelEntry)
        {
            throw std::runtime_error(
                "SimPhysics: spectra.model '" + model.value() +
                "' not found in spectra registry " + registryPath.string());
        }
        const bool wrGrid = modelEntry.at_path("WR_grid").value<bool>().value_or(false);

        if (wrGrid)
        {
            specsyn_ = std::make_unique<specsyn::SpecsynLibWR<specsyn::OOBPolicy::Throw>>(
                model.value(), fehDist_.getMin(), fehDist_.getMax(), registryName);
        }
        else
        {
            specsyn_ = std::make_unique<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::Throw>>(
                model.value(), fehDist_.getMin(), fehDist_.getMax(),
                tracks::defaultAFe, specsyn::defaultCFe,
                std::numeric_limits<double>::quiet_NaN(), specsyn::defaultR,
                registryName);
        }
        return;
    }

    const toml::array* modelArr = modelNode.as_array();
    if (modelArr == nullptr)
    {
        throw std::runtime_error(
            "SimPhysics: spectra.model must be a string or an array of strings");
    }
    std::vector<std::string> models;
    modelArr->for_each([&models](auto&& el) -> void {
        if constexpr (toml::is_string<decltype(el)>) { models.push_back(std::string(el)); }
    });
    if (models.empty())
    {
        throw std::runtime_error(
            "SimPhysics: spectra.model array must contain at least one string entry");
    }

    // Note: SpecsynLibChained does not yet support WR-type libraries
    // in its priority chain (see its own TODO); that support is
    // planned as a follow-up.
    specsyn_ = std::make_unique<specsyn::SpecsynLibChained>(
        models, fehDist_.getMin(), fehDist_.getMax(),
        tracks::defaultAFe, specsyn::defaultCFe, std::vector<double>{},
        specsyn::defaultR, registryName);
}
