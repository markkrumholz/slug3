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
#include "../phot/FilterCollection.hpp"
#include "../phot/FilterCommons.hpp"
#include "../phot/PhotCommons.hpp"
#include "../phot/VegaSpectrum.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../specsyn/SpecsynBlackbody.hpp"
#include "../specsyn/SpecsynCommons.hpp"
#include "../specsyn/SpecsynLibChained.hpp"
#include "../specsyn/SpecsynLibNoWind.hpp"
#include "../specsyn/SpecsynLibWR.hpp"
#include "../specsyn/SpecsynUtils.hpp"
#include "../tracks/TrackCommons.hpp"
#include "../utils/ParseUtils.hpp"
#include "../utils/TOMLUtils.hpp"
#include "SimControls.hpp"
#include <algorithm>
#include <cstddef>
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
io::SimPhysics::SimPhysics(const toml::table& inputDeck, const SimControls& controls)
{
    // Read IMF, CMF, and FeH
    imf_ = utils::initPDFFromKey(inputDeck, "stars.IMF", imfPrefix);
    cmf_ = utils::initPDFFromKey(inputDeck, "clusters.CMF");
    fehDist_ = utils::initPDFFromKey(inputDeck, "stars.FeH");

    // Read the spectral synthesis model to use, if any -- spectra.model
    // is optional, since not every simulation needs spectra computed.
    // Needs fehDist_ (just set above) to pick the [Fe/H] range a
    // library-based model is loaded over.
    readSpectra(inputDeck, controls);

    // Read the photometric filter collection to use, if any --
    // phot.filters is optional. Needs specsyn_ (just set above) to
    // check that a spectral synthesizer is actually available before
    // building a filter collection that will need one.
    readFilters(inputDeck);

    // In a galaxy simulation, read CLF and SFR
    if (controls.simType() == SimControls::SimType::galaxy)
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

namespace
{
    // Shared null-check for the integrator tolerance wrappers below:
    // specsyn_ is null whenever spectra.model was not set in the
    // input deck, so there is no spectral synthesizer to forward to
    void checkSpecsyn(const specsyn::Specsyn* specsyn)
    {
        if (specsyn == nullptr)
        {
            throw std::runtime_error(
                "SimPhysics: no spectral synthesizer was requested "
                "(spectra.model was not set in the input deck)");
        }
    }
} // namespace

// Set the [Fe/H] distribution, recomputing tracks2D() (constFeHTracks_)
// from the current tracks_ if the new fehDist_ is fixed -- mirrors the
// constructor's own post-readTracks() step, which does the same thing
// once, after both tracks_ and fehDist_ are set
void io::SimPhysics::setFeH(const std::string& feH)
{
    fehDist_ = utils::initPDFFromString(feH);
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
    }
}

// Set the stellar tracks, recomputing tracks2D() (constFeHTracks_)
// from the new tracks_ if fehDist_ is already fixed -- see setFeH()'s
// own comment
void io::SimPhysics::setTracks(tracks::Tracks3D tracks)
{
    tracks_ = std::move(tracks);
    if (constFeH())
    {
        constFeHTracks_ = tracks_.sliceConstFeH(fehDist_.getMin());
    }
}

// Set the star formation rate -- mirrors the galaxy.sfr handling in
// the constructor above exactly, including not resolving a file name
// through utils::getFilePath (unlike setIMF()/setCMF()/setFeH()/
// setCLF(), which all go through utils::initPDFFromString())
void io::SimPhysics::setSFR(const std::string& sfr)
{
    try
    {
        const double sfrVal = utils::stod(sfr);

        // We have been given a numerical value, so construct a
        // constant PDF from t = 0 to T for a big number T, with
        // the weight set to T / sfr so that the mass of stars
        // formed in any time interval dt comes out to sfr * dt.
        const double tMax = std::numeric_limits<double>::max() *
            std::min(sfrVal, 1.0);
        const double wgt = std::numeric_limits<double>::max() /
            std::max(sfrVal, 1.0);
        auto pl = std::make_unique<pdfs::PDFSegmentPowerlaw>(0.0, tMax, 0.0);
        sfr_ = pdfs::PDF(std::move(pl), wgt);
    }
    catch (const std::invalid_argument&)
    {
        // We have been given a file
        sfr_ = pdfs::parsePDFDescriptor(sfr);
    }
}

// Integrator tolerance getters and setters: thin wrappers around
// specsyn_'s own Specsyn::intRelTol() etc.
auto io::SimPhysics::intRelTol() const -> double
{
    checkSpecsyn(specsyn_.get());
    return specsyn_->intRelTol();
}

auto io::SimPhysics::intAbsTol() const -> double
{
    checkSpecsyn(specsyn_.get());
    return specsyn_->intAbsTol();
}

auto io::SimPhysics::intMaxIter() const -> std::size_t
{
    checkSpecsyn(specsyn_.get());
    return specsyn_->intMaxIter();
}

void io::SimPhysics::setIntRelTol(const double tol)
{
    checkSpecsyn(specsyn_.get());
    specsyn_->setIntRelTol(tol);
}

void io::SimPhysics::setIntAbsTol(const double tol)
{
    checkSpecsyn(specsyn_.get());
    specsyn_->setIntAbsTol(tol);
}

void io::SimPhysics::setIntMaxIter(const std::size_t n)
{
    checkSpecsyn(specsyn_.get());
    specsyn_->setIntMaxIter(n);
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
void io::SimPhysics::readSpectra(const toml::table& inputDeck, const SimControls& controls)
{
    // Check for an optional alternative registry
    auto registryNameInput = utils::getTOMLKeyWithError<std::string>(
        inputDeck, "spectra.registry");
    const std::string registryName = registryNameInput.value_or(specsyn::defaultRegistry);

    // spectra.model is optional -- if it is absent, this simulation
    // computes no spectra, and specsyn_ stays null
    const auto modelNode = inputDeck.at_path("spectra.model");
    if (!modelNode) { return; }

    // Optional user-requested output wavelength grid: spectra.wl_min
    // and spectra.wl_max (in Angstrom), and spectra.nwl (the number
    // of output wavelengths). All three are individually optional,
    // but wl_min and wl_max only make sense together with an nwl to
    // say how finely to sample between them, so if either endpoint is
    // given, all three must be; nwl alone (a request to resample onto
    // the default wavelength range at a different resolution) is fine
    // on its own. Any not supplied falls back to specsyn::defaultWlMin/
    // defaultWlMax/defaultNWl -- a fixed grid from ~91 Angstrom (10
    // Rydberg, deep enough into the EUV to capture ionizing flux) to
    // 1e5 Angstrom (10 micron) at 2048 points -- rather than each
    // library's own native grid, so that every spectral synthesizer
    // built without explicit wavelength settings covers the same
    // physically-motivated range by default.
    const auto wlMinInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.wl_min");
    const auto wlMaxInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.wl_max");
    const auto nWlInput = utils::getTOMLKeyWithError<unsigned long>(inputDeck, "spectra.nwl");
    if ((wlMinInput.has_value() || wlMaxInput.has_value()) &&
        !(wlMinInput.has_value() && wlMaxInput.has_value() && nWlInput.has_value()))
    {
        throw std::runtime_error(
            "SimPhysics: spectra.wl_min and spectra.wl_max must be "
            "given together with each other and with spectra.nwl");
    }
    wlMin_ = wlMinInput.value_or(specsyn::defaultWlMin);
    wlMax_ = wlMaxInput.value_or(specsyn::defaultWlMax);
    nWl_ = nWlInput.value_or(specsyn::defaultNWl);

    // Optional redshift, applied by every Specsyn's own wlObs(); 0
    // (no redshift) if not supplied
    const auto zInput = utils::getTOMLKeyWithError<double>(inputDeck, "spectra.z");
    const double z = zInput.value_or(0.0);

    // Optional stars.alphaFe and stars.CFe: if not supplied, fall back
    // to the library defaults. stars.alphaFe is the same key that
    // readTracks() also reads (tracks and spectra share the same AFe
    // value), while stars.CFe is spectra-only (tracks have no CFe axis).
    const auto afeInput = utils::getTOMLKeyWithError<double>(inputDeck, "stars.alphaFe");
    const double afe = afeInput.value_or(tracks::defaultAFe);
    const auto cfeInput = utils::getTOMLKeyWithError<double>(inputDeck, "stars.CFe");
    const double cfe = cfeInput.value_or(specsyn::defaultCFe);

    // A single string names one model directly; anything else must be
    // an array of strings, chained together via SpecsynLibChained
    if (const auto model = modelNode.value<std::string>(); model.has_value())
    {
        // "blackbody" is a special value, used for testing, that does
        // not require a spectral library at all
        if (model.value() == "blackbody")
        {
            specsyn_ = std::make_unique<specsyn::SpecsynBlackbody>(wlMin_, wlMax_, nWl_, z, controls);
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
            specsyn_ = std::make_unique<specsyn::SpecsynLibWR<specsyn::OOBPolicy::raise>>(
                model.value(), fehDist_.getMin(), fehDist_.getMax(), registryName,
                wlMin_, wlMax_, nWl_, z, controls);
        }
        else
        {
            specsyn_ = std::make_unique<specsyn::SpecsynLibNoWind<specsyn::OOBPolicy::raise>>(
                model.value(), fehDist_.getMin(), fehDist_.getMax(),
                afe, cfe,
                std::numeric_limits<double>::quiet_NaN(), specsyn::defaultR,
                registryName, wlMin_, wlMax_, nWl_, z, controls);
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

    specsyn_ = std::make_unique<specsyn::SpecsynLibChained>(
        models, fehDist_.getMin(), fehDist_.getMax(),
        afe, cfe, std::vector<double>{},
        specsyn::defaultR, registryName, wlMin_, wlMax_, nWl_, z, true, controls);
}

// Photometric filter collection reader
void io::SimPhysics::readFilters(const toml::table& inputDeck)
{
    // phot.system: optional; if present, must name one of the defined
    // photometric systems. Flambda (a raw flux, needing no Vega
    // spectrum or other extra data) is the default if the key is
    // absent entirely.
    phot::PhotSystem photSystem = phot::PhotSystem::Flambda;
    const auto systemNode = inputDeck.at_path("phot.system");
    if (systemNode)
    {
        const auto systemStr = systemNode.value<std::string>();
        if (!systemStr.has_value())
        {
            throw std::runtime_error("SimPhysics: phot.system must be a string");
        }
        if (systemStr.value() == "Flambda") { photSystem = phot::PhotSystem::Flambda; }
        else if (systemStr.value() == "Fnu") { photSystem = phot::PhotSystem::Fnu; }
        else if (systemStr.value() == "ST") { photSystem = phot::PhotSystem::ST; }
        else if (systemStr.value() == "AB") { photSystem = phot::PhotSystem::AB; }
        else if (systemStr.value() == "Vega") { photSystem = phot::PhotSystem::Vega; }
        else
        {
            throw std::runtime_error(
                "SimPhysics: phot.system '" + systemStr.value() +
                "' is not a recognized photometric system "
                "(expected Flambda, Fnu, ST, AB, or Vega)");
        }
    }

    // phot.registry: optional string override of the default filter registry
    const auto registryInput = utils::getTOMLKeyWithError<std::string>(inputDeck, "phot.registry");
    const std::string registryName = registryInput.value_or(phot::defaultRegistry);

    // phot.vega: optional string override of the default Vega
    // reference spectrum file. The global Vega spectrum (see
    // phot::vegaSpectrum()) is a lazily-constructed, program-wide
    // singleton -- only the very first call to vegaSpectrum() anywhere
    // determines which file actually gets loaded, so if the deck names
    // a non-default file, force that first call to happen here, now,
    // before any filter's own lazy Filter::fluxVega() can beat it to
    // the punch with the default file instead. The returned spectrum
    // itself is not needed here, only the side effect of loading it.
    const auto vegaInput = utils::getTOMLKeyWithError<std::string>(inputDeck, "phot.vega");
    if (vegaInput.has_value()) { phot::vegaSpectrum(vegaInput.value()); }

    // phot.filters: optional; a single string names one filter
    // directly, interpreted as an array of length 1; anything else
    // must be an array of strings
    std::vector<std::string> filterNames;
    const auto filtersNode = inputDeck.at_path("phot.filters");
    if (filtersNode)
    {
        if (const auto single = filtersNode.value<std::string>(); single.has_value())
        {
            filterNames.push_back(single.value());
        }
        else
        {
            const toml::array* filtersArr = filtersNode.as_array();
            if (filtersArr == nullptr)
            {
                throw std::runtime_error(
                    "SimPhysics: phot.filters must be a string or an array of strings");
            }
            filterNames = utils::stringArrayContents(filtersArr);
        }
    }

    // "Lbol" (bolometric luminosity) is photometry-like, but is
    // computed outside the filter machinery entirely (see
    // Cluster::computeLbol()); pull it out of the filter list here
    // and just record that it was requested
    const auto lbolIt = std::ranges::find(filterNames, "Lbol");
    if (lbolIt != filterNames.end())
    {
        filterNames.erase(lbolIt);
        computeLbol_ = true;
    }

    // Nothing further to do if no actual filters were requested
    if (filterNames.empty()) { return; }

    // Photometry requires a spectral synthesizer to generate the
    // spectra filters are convolved against
    if (specsyn_ == nullptr)
    {
        throw std::runtime_error(
            "SimPhysics: phot.filters was given but no spectral "
            "synthesizer was requested (spectra.model was not set "
            "in the input deck)");
    }

    filters_ = std::make_unique<phot::FilterCollection>(
        filterNames, photSystem, registryName);
}
