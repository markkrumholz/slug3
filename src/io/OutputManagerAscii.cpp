/**
 * @file OutputManagerAscii.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerAscii
 * @date 2026-07-17
 */

#include "OutputManagerAscii.hpp"
#include "../core/Cluster.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "SimPhysics.hpp"
#include "io/SlugVersion.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>

// Column headings, and the field widths used to lay them out, for
// the ascii cluster output file. uidWidth accommodates a 9-digit
// integer; numWidth accommodates a number in exponential notation
// with six decimal places (e.g. "-1.234567e+01"). Both include a
// couple of extra characters of padding so that columns are visibly
// separated.
static constexpr int uidWidth = 12;
static constexpr int numWidth = 16;

// Number of digits used to zero-pad the trial/uid columns, and the
// number of digits after the decimal point used for the
// exponential-notation columns
static constexpr int uidDigits = 9;
static constexpr int sciPrecision = 6;

// Format value as a zero-padded, fixed-width unsigned integer
static auto formatUid(const unsigned long value) -> std::string
{
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(uidDigits) << value;
    return stream.str();
}

// Format value in exponential notation with sciPrecision digits
// after the decimal point
static auto formatSci(const double value) -> std::string
{
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(sciPrecision) << value;
    return stream.str();
}

// Write the cluster-output ascii header (column names followed by a
// dashed rule) to file
static void writeClustersHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "target_mass"
         << std::setw(numWidth) << "birth_mass"
         << std::setw(numWidth) << "form_time"
         << std::setw(numWidth) << "feh" << "\n";
    constexpr auto numColumns = 4;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Ascii constructor: open the summary file, write the header
// (slug-hash, date, time), then dump the toml input deck, and close
// the file. If the simulation outputs individual clusters, also
// open the cluster output file, write its column-header rows, and
// leave it open for later writing.
io::OutputManagerAscii::OutputManagerAscii(
    const SimControls& simControls, const SimPhysics& simPhysics,
    const toml::table& inputDeck) :
    OutputManager(simControls, simPhysics, inputDeck)
{
    const auto path = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_summary.txt");
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + path.string() + " already exists");
    }

    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + path.string());
    }

    const auto [date, time] = currentDateAndTime();
    file << "slug-hash  " << slugGitHash << "\n"
         << "date  " << date << "\n"
         << "time  " << time << "\n"
         << "rng_state  " << currentRngStateString() << "\n";

    file << "input_deck\n" << inputDeck_ << "\n";

    file.close();

    if (simControls_.simType() == SimControls::SimType::cluster ||
        simControls_.outputClusters())
    {
        const auto clustersPath = std::filesystem::path(simControls_.outDir()) /
            (simControls_.modelName() + "_clusters.txt");
        if (std::filesystem::exists(clustersPath))
        {
            throw std::runtime_error(
                "OutputManagerAscii: output file " + clustersPath.string() + " already exists");
        }

        clustersFile_.open(clustersPath);
        if (!clustersFile_)
        {
            throw std::runtime_error(
                "OutputManagerAscii: unable to open output file " + clustersPath.string());
        }
        writeClustersHeader(clustersFile_);
    }
}

io::OutputManagerAscii::~OutputManagerAscii()
{
    if (clustersFile_.is_open()) { clustersFile_.close(); }
}

// Write one fixed-width row of cluster data to the cluster output
// file. A no-op if cluster output was not enabled for this
// simulation.
void io::OutputManagerAscii::writeCluster(
    const unsigned long trial, const core::Cluster& cluster)
{
    if (!clustersFile_.is_open()) { return; }

    // Guard the actual write against concurrent callers from other
    // threads; unlike the constructor, this method is expected to be
    // called from inside an openMP parallel region
#ifdef _OPENMP
#pragma omp critical(clusterOutputWrite)
#endif
    {
        clustersFile_ << std::right
                      << std::setw(uidWidth) << formatUid(trial)
                      << std::setw(uidWidth) << formatUid(cluster.uid())
                      << std::setw(numWidth) << formatSci(cluster.targetMass())
                      << std::setw(numWidth) << formatSci(cluster.birthMass())
                      << std::setw(numWidth) << formatSci(cluster.formTime())
                      << std::setw(numWidth) << formatSci(cluster.feH()) << "\n";
    }
}
