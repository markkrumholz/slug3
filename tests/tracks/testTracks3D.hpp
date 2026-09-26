/**
 * @file testTracks3D.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the Tracks3D class.
 * @details
 * This file contains unit tests for the Tracks3D class, which reads a
 * grid of stellar evolutionary tracks spanning a range of [Fe/H]
 * values from an HDF5 file. Unlike the Tracks2D tests, these tests act
 * only on the small, reduced test track set in tests/tracks/assets;
 * building a Tracks3D from the full-size track files in data/tracks
 * would be far too slow for a unit test.
 * @date 2024-07-10
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTTRACKS3D_HPP
#define TESTTRACKS3D_HPP

#include "../../src/tracks/Tracks3D.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "trackFieldFixture.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

/**
 * @brief Unit test for the Tracks3D class.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This function constructs a Tracks3D object from the MIST_test track
 * set in tests/tracks/assets/tracks.toml, which contains 5 groups at
 * afe = -0.2, vvcrit = 0.0, and feh = -1.0, -0.5, -0.25, 0.0, and 0.5,
 * and verifies that construction succeeds without error.
 */
inline auto testTracks3D() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";

    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, -0.5, 0.0, 0.0, -0.2, registryName);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3D: failed to construct Tracks3D from "
            << registryName << ", track set " << trackName << ": "
            << e.what() << "\n";
        return 1;
    }

    return 0;
}

/**
 * @brief Unit test that an out-of-range [Fe/H] request is rejected.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * The MIST_test track set in tests/tracks/assets/tracks.toml provides
 * feh = -1.0, -0.5, -0.25, 0.0, 0.5 at afe = -0.2, vvcrit = 0.0. This
 * checks that requesting a fehMin below -1.0 or a fehMax above 0.5
 * throws, rather than silently bracketing against the nearest
 * available value the way findMatchingTracks() does internally, and
 * that a request fully inside that range still succeeds.
 */
inline auto testTracks3DFeHRangeGuard() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    int result = 0;

    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, -2.0, 0.0, 0.0, -0.2, registryName);
        std::cerr << "testTracks3DFeHRangeGuard: construction with "
            "fehMin = -2.0 (below the available minimum of -1.0) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, 0.0, 1.0, 0.0, -0.2, registryName);
        std::cerr << "testTracks3DFeHRangeGuard: construction with "
            "fehMax = 1.0 (above the available maximum of 0.5) "
            "should have thrown, but did not\n";
        result = 1;
    }
    catch (const std::runtime_error&) { /* expected */ }

    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, -1.0, 0.5, 0.0, -0.2, registryName);
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3DFeHRangeGuard: construction spanning "
            "the full available [-1.0, 0.5] range unexpectedly threw: "
            << e.what() << "\n";
        result = 1;
    }

    return result;
}

// Suppress clang-tidy warnings iun this namespace caused by just including
// hdf5.h, instead of the individual HDF5 headers, since this is the paradigm
// that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

/**
 * @brief Regression test that getTrack() returns fields in the
 *   correct order.
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * This is a regression test for a bug in which the columns of field
 * data read from an HDF5 track file were mismatched against their
 * canonical tracks::FieldIdx order, because the age column (present
 * in every track dataset, but not one of the nQty tracked
 * quantities) was not properly accounted for when mapping canonical
 * field index to on-disk column. It constructs a Tracks3D object with
 * fehMin = fehMax = 0.0, an exact point on the MIST_test set's [Fe/H]
 * grid, which exercises the single-slice code path (a mesh one
 * element wide in the feh direction). It then calls getTrack() for
 * mass = 5.0 (an exact point on the mass grid) and evaluates the
 * result at the exact age of an arbitrary interior row of the raw
 * track_m5.000 dataset in the feh_0.00_afe_-0.2_vvcrit_0.00 group of
 * tests/tracks/assets/MIST_test.h5. Since mass = 5.0 is on the mesh's
 * mass grid, feh = 0.0 is the mesh's only feh point, and the query age
 * is on that mass's own age grid, the interpolated result should
 * reproduce the raw row exactly (up to floating-point round-off); the
 * raw row itself is read independently of the Tracks3D field-mapping
 * logic via trackFieldFixture.hpp, so this test does not depend on
 * that logic being correct to establish its expectations.
 */
inline auto testTracks3DFieldOrder() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    const std::string h5Path = "tests/tracks/assets/MIST_test.h5";
    const std::string groupName = "feh_0.00_afe_-0.2_vvcrit_0.00";
    constexpr double feh = 0.0;
    constexpr double mass = 5.0;
    constexpr size_t rowIdx = 500;

    const hid_t file = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks3DFieldOrder: unable to open file "
            << h5Path << "\n";
        return 1;
    }
    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks3DFieldOrder: unable to open group "
            << groupName << " in " << h5Path << "\n";
        H5Fclose(file);
        return 1;
    }

    int result = 0;
    try
    {
        const auto [age, expected] = testutil::readRawFields(grp, mass, rowIdx);

        const tracks::Tracks3D tracks3d(
            trackName, feh, feh, 0.0, -0.2, registryName);
        const auto track = tracks3d.getTrack(mass, feh);
        if (!track)
        {
            std::cerr << "testTracks3DFieldOrder: getTrack(" << mass
                << ", " << feh << ") returned null\n";
            result = 1;
        }
        else
        {
            const auto actual = (*track)(std::log10(age));
            for (size_t k = 0; k < testutil::nQty; ++k)
            {
                if (!testutil::fieldsMatch(actual.at(k), expected.at(k)))
                {
                    std::cerr << "testTracks3DFieldOrder: field "
                        << tracks::fieldStr.at(k) << " (index " << k
                        << ") mismatch: expected " << expected.at(k)
                        << ", got " << actual.at(k) << "\n";
                    result = 1;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3DFieldOrder: unexpected exception: "
            << e.what() << "\n";
        result = 1;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return result;
}

/**
 * @brief Regression/unit test for getStar()'s field order and values
 * @return 0 if the test passes, 1 if it fails.
 * @details
 * Uses the same ground truth and setup as testTracks3DFieldOrder(),
 * but calls getStar(mass, log10(age), feh) directly instead of going
 * through getTrack(). Since mass = 5.0 is on the mesh's mass grid,
 * feh = 0.0 is the mesh's only feh point, and the query age is on
 * that mass's own age grid, (mass, log10(age)) is an exact vertex of
 * the underlying feh = 0.0 slice's mesh, so getStar() should
 * reproduce the raw row exactly (up to floating-point round-off).
 */
inline auto testTracks3DGetStar() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    const std::string h5Path = "tests/tracks/assets/MIST_test.h5";
    const std::string groupName = "feh_0.00_afe_-0.2_vvcrit_0.00";
    constexpr double feh = 0.0;
    constexpr double mass = 5.0;
    constexpr size_t rowIdx = 500;

    const hid_t file = H5Fopen(h5Path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        std::cerr << "testTracks3DGetStar: unable to open file "
            << h5Path << "\n";
        return 1;
    }
    const hid_t grp = H5Gopen2(file, groupName.c_str(), H5P_DEFAULT);
    if (grp < 0)
    {
        std::cerr << "testTracks3DGetStar: unable to open group "
            << groupName << " in " << h5Path << "\n";
        H5Fclose(file);
        return 1;
    }

    int result = 0;
    try
    {
        const auto [age, expected] = testutil::readRawFields(grp, mass, rowIdx);

        const tracks::Tracks3D tracks3d(
            trackName, feh, feh, 0.0, -0.2, registryName);
        const auto actual = tracks3d.getStar(mass, std::log10(age), feh);
        for (size_t k = 0; k < testutil::nQty; ++k)
        {
            if (!testutil::fieldsMatch(actual.at(k), expected.at(k)))
            {
                std::cerr << "testTracks3DGetStar: field "
                    << tracks::fieldStr.at(k) << " (index " << k
                    << ") mismatch: expected " << expected.at(k)
                    << ", got " << actual.at(k) << "\n";
                result = 1;
            }
        }

        // Also check that linear = true doesn't throw and reproduces
        // the same ground truth at this same exact-grid point
        const auto actualLinear =
            tracks3d.getStar(mass, std::log10(age), feh, true);
        for (size_t k = 0; k < testutil::nQty; ++k)
        {
            if (!testutil::fieldsMatch(actualLinear.at(k), expected.at(k)))
            {
                std::cerr << "testTracks3DGetStar: (linear) field "
                    << tracks::fieldStr.at(k) << " (index " << k
                    << ") mismatch: expected " << expected.at(k)
                    << ", got " << actualLinear.at(k) << "\n";
                result = 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3DGetStar: unexpected exception: "
            << e.what() << "\n";
        result = 1;
    }

    H5Gclose(grp);
    H5Fclose(file);
    return result;
}

/**
 * @brief Unit test for Tracks3D::massAndDerivFromLifetime()
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Mirrors Tracks2D::testTracks2DMassAndDerivFromLifetime()'s own
 * setup and reasoning exactly, at feh = 0.0 (one of MIST_test's own
 * tabulated [Fe/H] grid values, so this Tracks3D slice reduces to
 * exactly the same feh_0.00_afe_-0.2_vvcrit_0.00 group Tracks2D itself
 * reads, with no [Fe/H] interpolation error to account for).
 */
inline auto testTracks3DMassAndDerivFromLifetime() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    constexpr double feh = 0.0;

    int result = 0;
    try
    {
        const tracks::Tracks3D tracks3d(
            trackName, feh, feh, 0.0, -0.2, registryName);

        constexpr double m1 = 1.0;
        constexpr double m2 = 5.0;
        const double logT1 = std::log10(tracks3d.starLifetime(m1, feh));
        const double logT2 = std::log10(tracks3d.starLifetime(m2, feh));

        const double logTQuery = 0.5 * (logT1 + logT2);
        const double expectedSlope = (m2 - m1) / (logT2 - logT1);
        const double expectedMass = m1 + (expectedSlope * (logTQuery - logT1));

        const auto edges = tracks3d.massAndDerivFromLifetime(logTQuery, feh);
        if (edges.size() != 1)
        {
            std::cerr << "testTracks3DMassAndDerivFromLifetime: expected 1 "
                "(mass, slope) pair at logT = " << logTQuery << ", got "
                << edges.size() << "\n";
            result = 1;
        }
        else if (!testutil::fieldsMatch(edges.at(0).first, expectedMass) ||
            !testutil::fieldsMatch(edges.at(0).second, expectedSlope))
        {
            std::cerr << "testTracks3DMassAndDerivFromLifetime: expected (mass, "
                "dm/dlogT) = (" << expectedMass << ", " << expectedSlope <<
                "), got (" << edges.at(0).first << ", " << edges.at(0).second << ")\n";
            result = 1;
        }

        constexpr double logTShort = 5.0; // shorter than every tabulated lifetime
        const auto noEdges = tracks3d.massAndDerivFromLifetime(logTShort, feh);
        if (!noEdges.empty())
        {
            std::cerr << "testTracks3DMassAndDerivFromLifetime: expected no "
                "(mass, slope) pairs at logT = " << logTShort << ", got "
                << noEdges.size() << "\n";
            result = 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "testTracks3DMassAndDerivFromLifetime: unexpected exception: "
            << e.what() << "\n";
        result = 1;
    }

    return result;
}

/**
 * @brief Unit test that Tracks3D's lazily-evaluated queries agree with an eagerly-built slice
 * @return 0 if the test passes, 1 if it fails
 * @details
 * Every Tracks3D query at a given [Fe/H] evaluates a lazily-evaluated
 * slice of the tracks (see Tracks3D's own class comment). At [Fe/H]
 * values between the MIST_test set's own grid points and exactly on
 * one, this compares starLifetime(), liveMassRange(), isAlive(),
 * getTrack(), getIsochrone() and getStar() -- and the same queries on
 * a Tracks2D returned by lazySliceConstFeH() -- against an eagerly-
 * built slice from sliceConstFeH(), requiring agreement to 1e-10
 * relative. Also checks that a lazySliceConstFeH() slice of a
 * shared_ptr-owned Tracks3D keeps it alive, and so still gives the
 * same answers, after every other reference to it is released.
 */
inline auto testTracks3DLazyMatchesEager() -> int
{
    const std::string registryName = "tests/tracks/assets/tracks.toml";
    const std::string trackName = "MIST_test";
    constexpr double tol = 1e-10;
    const auto close = [tol](const double a, const double b) -> bool
    { return std::abs(a - b) <= tol * std::max({ 1.0, std::abs(a), std::abs(b) }); };
    int nBad = 0;
    const auto fail = [&nBad](const std::string& msg) -> void
    {
        std::cerr << "testTracks3DLazyMatchesEager: " << msg << "\n";
        ++nBad;
    };

    // Compare two tracks::Tracks2D::getTrack()/getIsochrone()-style
    // Interpolator1D results at evenly spaced points
    const auto compareInterp = [&](const auto* lazy, const auto* eager, const std::string& label)
    {
        if ((lazy == nullptr) != (eager == nullptr)) { fail(label + ": null mismatch"); return; }
        if (lazy == nullptr) { return; }
        if (!close(lazy->xMin(), eager->xMin()) || !close(lazy->xMax(), eager->xMax()))
        {
            fail(label + ": range mismatch");
            return;
        }
        for (int p = 0; p <= 10; ++p)
        {
            const double q = eager->xMin() + ((eager->xMax() - eager->xMin()) * p / 10.0);
            const auto a = (*lazy)(std::clamp(q, lazy->xMin(), lazy->xMax()));
            const auto b = (*eager)(std::clamp(q, eager->xMin(), eager->xMax()));
            for (size_t k = 0; k < a.size(); ++k)
            {
                if (!close(a.at(k), b.at(k))) { fail(label + ": value mismatch"); return; }
            }
        }
    };

    try
    {
        auto tracks3d = std::make_shared<const tracks::Tracks3D>(
            trackName, -1.0, 0.5, 0.0, -0.2, registryName);
        for (const double feh : { -0.8, -0.3, 0.0, 0.2 })
        {
            const auto eager = tracks3d->sliceConstFeH(feh);
            const auto lazy = tracks3d->lazySliceConstFeH(feh);
            const std::string fehLabel = "feh = " + std::to_string(feh);
            for (const double m : { 0.5, 1.0, 2.7, 9.0, 40.0 })
            {
                const std::string label = fehLabel + ", m = " + std::to_string(m);
                const double tl = eager.starLifetime(m);
                if (!close(tracks3d->starLifetime(m, feh), tl) || !close(lazy.starLifetime(m), tl))
                {
                    fail(label + ": starLifetime mismatch");
                }
                const auto et = eager.getTrack(m);
                compareInterp(tracks3d->getTrack(m, feh).get(), et.get(), label + ", getTrack");
                compareInterp(lazy.getTrack(m).get(), et.get(), label + ", lazy getTrack");
            }
            for (const double logT : { 6.2, 7.0, 8.3, 9.5 })
            {
                const std::string label = fehLabel + ", logT = " + std::to_string(logT);
                const auto el = eager.liveMassRange(logT);
                const auto ll = tracks3d->liveMassRange(logT, feh);
                bool limOk = el.size() == ll.size();
                for (size_t r = 0; limOk && r < el.size(); ++r)
                {
                    limOk = close(el[r].first, ll[r].first) && close(el[r].second, ll[r].second);
                }
                if (!limOk) { fail(label + ": liveMassRange mismatch"); }
                const auto ei = eager.getIsochrone(logT);
                const auto li = tracks3d->getIsochrone(logT, feh);
                const auto lli = lazy.getIsochrone(logT);
                if (li.size() != ei.size() || lli.size() != ei.size())
                {
                    fail(label + ": isochrone segment count mismatch");
                    continue;
                }
                for (size_t sg = 0; sg < ei.size(); ++sg)
                {
                    compareInterp(li[sg].get(), ei[sg].get(), label + ", getIsochrone");
                    compareInterp(lli[sg].get(), ei[sg].get(), label + ", lazy getIsochrone");
                }
                for (const double m : { 0.3, 1.1, 3.0 })
                {
                    const bool alive = eager.isAlive(m, logT);
                    if (tracks3d->isAlive(m, logT, feh) != alive) { fail(label + ": isAlive mismatch"); }
                    if (!alive) { continue; }
                    const auto es = eager.getStar(m, logT);
                    const auto ls = tracks3d->getStar(m, logT, feh);
                    for (size_t k = 0; k < es.size(); ++k)
                    {
                        if (!close(ls.at(k), es.at(k))) { fail(label + ": getStar mismatch"); break; }
                    }
                }
            }
        }

        // A lazy slice keeps its shared_ptr-owned Tracks3D alive
        const auto lazyKept = tracks3d->lazySliceConstFeH(-0.3);
        const double expected = tracks3d->sliceConstFeH(-0.3).starLifetime(2.7);
        tracks3d.reset();
        if (!close(lazyKept.starLifetime(2.7), expected))
        {
            fail("lazySliceConstFeH() slice gave a different answer after its Tracks3D was released");
        }
    }
    catch (const std::exception& e)
    {
        fail(std::string("unexpected exception: ") + e.what());
    }
    return nBad > 0 ? 1 : 0;
}

// NOLINTEND(misc-include-cleaner)

#endif // TESTTRACKS3D_HPP
