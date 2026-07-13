/**
 * @file SimPhysics.cpp
 * @author Mark Krumholz
 * @brief Implementation of SimPhysics
 * @date 2026-07-12
 */

#include "SimPhysics.hpp"
#include "../extern/tomlplusplus/toml.hpp"
#include "../pdfs/PDFFileParser.hpp"
#include "../pdfs/PDFSegmentDelta.hpp"
#include "../pdfs/PDFSegmentPowerlaw.hpp"
#include "../utils/MiscUtils.hpp"
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Initialize a PDF from a toml key
     * @param inputDeck Input deck for simulation
     * @param key Name of key
     * @param prefix Prefix to look for PDF file relative to SLUG_DIR
     * @details
     * If the key resolves to a numerical value, this is interpreted
     * as specifying a delta function at that value. If it resolves
     * to a string, this is interpreted as supplying the name of the
     * PDF file descriptor, with the path to the file resolved by
     * utils::getFilePath.
    */
    auto initPDFFromKey(const toml::table& inputDeck, 
        const std::string& key,
        const std::string& prefix = "")
    {
        // First check for numerical value
        const std::optional<double> num =
            inputDeck.at_path(key).value<double>();
        if (num.has_value())
        {
            auto delta = std::make_unique<pdfs::PDFSegmentDelta>(num.value());
            auto pdf = pdfs::PDF(std::move(delta));
            return std::move(pdf);
        }

        // Next check for string value
        const std::optional<std::string> pdfFile =
            inputDeck.at_path(key).value<std::string>();
        if (pdfFile.has_value())
        {
            auto pdfFilePath = utils::getFilePath(pdfFile.value(), prefix).string();
            if (pdfFilePath.empty())
            {
                throw std::runtime_error(
                    "SimPhysics: pdf file " + pdfFile.value() + " not found");
            }
            auto pdf = pdfs::parsePDFDescriptor(pdfFilePath);
            return std::move(pdf);
        }

        // If we get here, key does not exist or has invalid type
        throw std::runtime_error(
            "SimPhysics: invalid entry for " + key);
    }

    /**
     * @brief Check for TOML key of type and return error if type is wrong
     * @tparam T The type to check for
     * @param inputDeck Input deck for simulation
     * @param key Name of key
     * @param required True if key is required
     * @details
     * This routine checks if the input key exists and can be retrieved
     * as of type T. Behavior is as follows: (1) if the key exists and can
     * be interpreted as type T, return a full optional<T>; (2) if the key
     * exists and cannot be interpreted as T, throw a runtime error; (3) if
     * the key does not exist and required is false, return ane empty
     * optional; (4) if the key does not exist and required is true, throw a
     * runtime error.
     */
    template<class T> auto getTOMLKeyWithError(
        const toml::table& inputDeck,
        const std::string& key,
        bool required = false) -> std::optional<T>
    {
        const auto node = inputDeck.at_path(key);
        if (node)
        {
            auto ret = node.value<T>();
            if (ret.has_value()) {
                return ret;
            }
            else
            {
                throw std::runtime_error(
                    "SimPhysics: cannot understand value for key " +
                    key);
            }
        }
        if (required) 
        { 
            throw std::runtime_error(
                "SimPhysics: required key " + key + 
                " not found"); 
        }
        return std::nullopt;
    }

} // Namespace

// SimPhysics constructor
core::SimPhysics::SimPhysics(const toml::table& inputDeck) :
    simType_(SimType::none)
{
    // First determine simulation type
    const auto simType = getTOMLKeyWithError<std::string>(
        inputDeck, "sim_type", true);
    if (simType.value() == "galaxy") { simType_ = SimType::galaxy; }
    else if (simType.value() == "cluster") { simType_ = SimType::cluster; }
    else { throw std::runtime_error("SimPhysics: sim_type must be 'galaxy' or 'cluster'"); }

    // Read IMF, CMF, and FeH
    imf_ = initPDFFromKey(inputDeck, "stars.IMF",
        (std::filesystem::path("data") / std::filesystem::path("imfs")).string());
    cmf_ = initPDFFromKey(inputDeck, "clusters.CMF");
    fehDist_ = initPDFFromKey(inputDeck, "stars.FeH");

    // In a galaxy simulation, read CLF and SFR
    if (simType_ == SimType::galaxy)
    {
        // CLF
        clf_ = initPDFFromKey(inputDeck, "clusters.CLF");

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
            double tMax = std::numeric_limits<double>::max() *
                std::min(sfr.value(), 1.0);
            double wgt = std::numeric_limits<double>::max() /
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
}

// Track reader
void core::SimPhysics::readTracks(const toml::table& inputDeck)
{
    // Get required tracks key
    auto trackName = getTOMLKeyWithError<std::string>(inputDeck, "stars.tracks", true);

    // Check for optional parameters
    auto registryName = getTOMLKeyWithError<std::string>(inputDeck, "stars.track_registry");
    auto vvcrit = getTOMLKeyWithError<double>(inputDeck, "stars.v_vcrit");
    auto afe = getTOMLKeyWithError<double>(inputDeck, "stars.alphaFe");

    // Construct tracks from input data
    tracks_ = tracks::Tracks3D(
        trackName.value(),
        fehDist_.getMin(),
        fehDist_.getMax(),
        vvcrit.value_or(tracks::defaultVVcrit),
        afe.value_or(tracks::defaultAFe),
        registryName.value_or(tracks::defaultRegistry));
}

