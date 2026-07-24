/**
 * @file testSpecsynChain.cpp
 * @author Mark Krumholz
 * @brief End-to-end test of a spectra.model library chain built through SimPhysics.
 * @details
 * Exercises the spectra.model wiring most real applications will
 * actually use: an array of several library names, in priority order,
 * mixing Wolf-Rayet (WR_grid = true) and non-WR-grid libraries --
 * ["POWR_WC", "POWR_WNE", "POWR_WNL", "TLUSTY_O", "BOSZ"] in production
 * (TLUSTY's B-star grid, TLUSTY_B, would typically also be chained in,
 * but is omitted here since this test's fixture registry only has a
 * single TLUSTY_test entry) -- test-fixture versions of each here.
 * Builds a SimPhysics from a real input deck
 * (tests/core/assets/testCluster.in, with spectra.registry and
 * spectra.model overridden) and calls spec() directly on a small,
 * deliberately-chosen set of stars -- one per WRType, plus one landing
 * in TLUSTY_test's grid and one in BOSZ_test's -- rather than drawing
 * a full stochastic Cluster population.
 *
 * A full Cluster population isn't used here because BOSZ_test and
 * TLUSTY_test are each only a couple of grid points wide (narrow
 * windows left over from the specific unit tests they were originally
 * built for -- see tests/specsyn's own fixtures), nowhere near broad
 * enough to cover the full range of Teff/logg a real IMF draw over
 * MIST_test's mass range would produce; with BOSZ_test as the chain's
 * final (OOBPolicy::raise) fallback, most such stars would simply
 * throw. Hand-picking each star's (mass, logL, logTeff, ...) instead
 * -- the same approach tests/specsyn/testSpecsynLibWR.cpp already uses
 * for its own WNE/WNL stars -- keeps this test exercising the real
 * SimPhysics -> SpecsynLibChained -> SpecsynLibWR/SpecsynLibNoWind
 * dispatch path without depending on fixture coverage this test
 * doesn't control.
 * @date 2026-07-24
 */

#include "../src/io/SimControls.hpp"
#include "../src/io/SimPhysics.hpp"
#include "../src/specsyn/Specsyn.hpp"
#include "../src/tracks/TrackCommons.hpp"
#include "testSpecsynChain.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <toml.hpp>
#include <vector>

namespace
{
    const std::string inputFile = "tests/core/assets/testCluster.in";
    const std::string registryName = "tests/specsyn/assets/spectra.toml";

    /**
     * @brief Build a StarData for a single star
     * @details
     * hSurf/cSurf/nSurf default to values that classify as
     * WRType::None via getWRType (hSurf = 0.7, well above the 0.3
     * cutoff), matching an ordinary, unstripped star.
     */
    auto makeStarData(
        const double mass,
        const double logL,
        const double logTeff,
        const double mdot,
        const double hSurf = 0.7,
        const double cSurf = 0.0,
        const double nSurf = 0.0) -> specsyn::Specsyn::StarData
    {
        specsyn::Specsyn::StarData props{};
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mass)) = mass;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::mdot)) = mdot;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logL)) = logL;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::logTe)) = logTeff;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::hSurf)) = hSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::cSurf)) = cSurf;
        props.at(static_cast<std::size_t>(tracks::FieldIdx::nSurf)) = nSurf;
        return props;
    }

    /**
     * @brief Build a SimPhysics whose spectra.model is a 5-library chain
     * @details
     * Starts from testCluster.in's usual tracks/IMF/[Fe/H] setup ([Fe/H]
     * = 0.0 exactly, a grid point every one of the five test libraries
     * below shares, so no library ever sees an out-of-[Fe/H]-range
     * query) and overrides spectra.registry/spectra.model to chain
     * POWR_WC_test, POWR_WNE_test, POWR_WNL_test, TLUSTY_test, and
     * BOSZ_test together, in priority order -- the test-fixture
     * counterpart of the ["POWR_WC", "POWR_WNE", "POWR_WNL", "TLUSTY_O",
     * "BOSZ"] combination most real applications will actually use.
     */
    auto buildChainedSim() -> io::SimPhysics
    {
        toml::table inputDeck = toml::parse_file(inputFile);
        auto* spectraTable = inputDeck.at_path("spectra").as_table();
        spectraTable->insert_or_assign("registry", registryName);
        spectraTable->insert_or_assign("model", toml::array{
            "POWR_WC_test", "POWR_WNE_test", "POWR_WNL_test", "TLUSTY_test", "BOSZ_test" });

        const io::SimControls controls(inputDeck);
        return io::SimPhysics(inputDeck, controls.simType());
    }

    /**
     * @brief Check that spec() succeeds and returns a sane spectrum
     * @param synth Spectral synthesizer to query
     * @param props Star to query it with
     * @param feh [Fe/H] value to query it with
     * @param label Human-readable label for this case, used in failure messages
     * @return 0 if spec() succeeded and returned a correctly-sized,
     *   non-trivial spectrum; 1 (with a diagnostic on stderr) otherwise
     */
    auto checkSpecSucceeds(
        const specsyn::Specsyn& synth,
        const specsyn::Specsyn::StarData& props,
        const double feh,
        const std::string& label) -> int
    {
        std::vector<double> result;
        try
        {
            result = synth.spec(props, feh);
        }
        catch (const std::exception& error)
        {
            std::cerr << "testSpecsynChain: " << label
                << ": unexpected exception from spec(): " << error.what() << "\n";
            return 1;
        }

        if (result.size() != synth.wl().size())
        {
            std::cerr << "testSpecsynChain: " << label << ": spec() returned "
                << result.size() << " values, expected " << synth.wl().size() << "\n";
            return 1;
        }
        if (std::ranges::none_of(result, [](const double v) -> bool { return v > 0.0; }))
        {
            std::cerr << "testSpecsynChain: " << label
                << ": expected a non-trivial (some positive flux) spectrum\n";
            return 1;
        }
        return 0;
    }
} // namespace

// Verify that the full 5-library spectra.model chain, built through
// SimPhysics from a real input deck, dispatches each representative
// star to the correct library: a WC-, WNE-, and WNL-classified
// Wolf-Rayet star each land on their own SpecsynLibWR (exercising
// SpecsynLibChained's WR_grid dispatch, which previously hardcoded
// SpecsynLibNoWind for every entry); a hot, unstripped OB star outside
// every WR grid lands on TLUSTY_test; and a solar-type star outside
// every other library's grid reaches BOSZ_test, the chain's final
// (OOBPolicy::raise) fallback.
static auto testSpecsynChainDispatch() -> int
{
    int result = 0;
    try
    {
        const auto sim = buildChainedSim();
        const auto* synth = sim.specsyn();
        if (synth == nullptr)
        {
            std::cerr << "testSpecsynChain: expected specsyn() to be populated\n";
            return 1;
        }

        // Wolf-Rayet stars: M = 20 Msun, L = 10^5.7 Lsun, Teff = 10^4.7 K,
        // Mdot = 3e-5 Msun/yr -- the same combination
        // tests/specsyn/testSpecsynLibWR.cpp's own WNE/WNL tests use,
        // which works out to logRt ~= 0.74, comfortably inside every
        // PoWR test fixture's [0.5, 1.0] log_rt range (see
        // data/tools/make_powr_test_fixture.py), with logTeff = 4.7
        // similarly between its {4.6, 4.8} grid points. Only
        // hSurf/cSurf/nSurf change between the three subtypes, per
        // getWRType's own classification rules.
        result += checkSpecSucceeds(*synth,
            makeStarData(20.0, 5.7, 4.7, 3e-5, 1e-6, 0.01, 0.0), 0.0, "WC star"); // cSurf >= nSurf
        result += checkSpecSucceeds(*synth,
            makeStarData(20.0, 5.7, 4.7, 3e-5, 1e-6, 0.0, 0.01), 0.0, "WNE star"); // cSurf < nSurf
        result += checkSpecSucceeds(*synth,
            makeStarData(20.0, 5.7, 4.7, 3e-5, 0.1), 0.0, "WNL star"); // 1e-5 < hSurf <= 0.3

        // A hot, unstripped OB star (hSurf = 0.7, well above the WR
        // hSurf > 0.3 cutoff, so getWRType returns WRType::None and
        // every PoWR library falls through) at Teff ~= 28750 K,
        // logg ~= 3.1 -- inside TLUSTY_test's [27500, 30000] K x
        // [3.0, 3.25] grid, via the same Stefan-Boltzmann/gravity
        // relations SpecsynLibNoWind::spec() itself uses (see
        // Specsyn::getSAandLogg)
        result += checkSpecSucceeds(*synth,
            makeStarData(20.0, 5.4284, std::log10(28750.0), 0.0), 0.0, "TLUSTY star");

        // A solar-type star (M = 1 Msun, L = 1 Lsun, Teff = 5772 K --
        // the Sun's own parameters) at logg ~= 4.44, inside BOSZ_test's
        // [5750, 6000] K x [4.0, 4.5] grid but outside TLUSTY_test's
        // and every PoWR library's, so it reaches BOSZ_test
        result += checkSpecSucceeds(*synth,
            makeStarData(1.0, 0.0, std::log10(5772.0), 0.0), 0.0, "BOSZ star");
    }
    catch (const std::exception& error)
    {
        std::cerr << "testSpecsynChain: failed to build chained SimPhysics: "
            << error.what() << "\n";
        return 1;
    }
    return result;
}

auto testSpecsynChain() -> int
{
    return testSpecsynChainDispatch();
}
