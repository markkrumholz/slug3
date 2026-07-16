/**
 * @file SimPhysics.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimPhysics
 * @date 2026-07-12
 */

#include "SimPhysics.hpp"
#include "../extern/tomlplusplus/toml.hpp"
#include "../pdfs/PDF.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include "../pdfs/PDFSegmentPowerlaw.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/ParseUtils.hpp"
#include "SimControls.hpp"
#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

// SimPhysics constructor
io::SimPhysics::SimPhysics(const toml::table& inputDeck, SimControls::SimType simType) :
    minStochMass_(0.0),
    fracStochMass_(1.0)
{
    // Read IMF, CMF, and FeH
    imf_ = utils::initPDFFromKey(inputDeck, "stars.IMF",
        (std::filesystem::path("data") / std::filesystem::path("imfs")).string());
    cmf_ = utils::initPDFFromKey(inputDeck, "clusters.CMF");
    fehDist_ = utils::initPDFFromKey(inputDeck, "stars.FeH");

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

