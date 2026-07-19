/**
 * @file OutputManagerAscii.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerAscii
 * @date 2026-07-17
 */

#include "OutputManagerAscii.hpp"
#include "../core/Cluster.hpp"
#include "../specsyn/Specsyn.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "SimPhysics.hpp"
#include "io/SlugVersion.hpp"
#include <cstddef>
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

// Write the cluster-spectra ascii header (column names followed by a
// dashed rule) to file. Unlike the cluster output file, this file is
// laid out one (wavelength, specific luminosity) pair per line
// rather than one cluster per line, since a spectrum can have
// thousands of wavelength points -- far too many to lay out as
// columns and still be human-readable.
static void writeClusterSpectraHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "spec" << "\n";
    constexpr auto numColumns = 3;
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

    if (simPhysics_.specsyn() != nullptr)
    {
        wlObs_ = simPhysics_.specsyn()->wlObs();

        const auto clusterSpectraPath = std::filesystem::path(simControls_.outDir()) /
            (simControls_.modelName() + "_cluster_spectra.txt");
        if (std::filesystem::exists(clusterSpectraPath))
        {
            throw std::runtime_error(
                "OutputManagerAscii: output file " + clusterSpectraPath.string() + " already exists");
        }

        clusterSpectraFile_.open(clusterSpectraPath);
        if (!clusterSpectraFile_)
        {
            throw std::runtime_error(
                "OutputManagerAscii: unable to open output file " + clusterSpectraPath.string());
        }
        writeClusterSpectraHeader(clusterSpectraFile_);
    }
}

io::OutputManagerAscii::~OutputManagerAscii()
{
    if (clustersFile_.is_open()) { clustersFile_.close(); }
    if (clusterSpectraFile_.is_open()) { clusterSpectraFile_.close(); }
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

// Write one line per wavelength (trial, time, uid, wavelength,
// specific luminosity) to the cluster-spectra output file. A no-op
// if spectral synthesis was not enabled for this simulation, or if
// the cluster has disrupted -- a disrupted cluster is no longer an
// observable object, though its light still belongs in the total
// galaxy spectrum, which is handled elsewhere.
void io::OutputManagerAscii::writeClusterSpec(
    const unsigned long trial, const double time, const core::Cluster& cluster)
{
    if (!clusterSpectraFile_.is_open()) { return; }
    if (cluster.isDisrupted()) { return; }

    const unsigned long uid = cluster.uid();
    const auto& spec = cluster.spec();

    // Guard the actual writes against concurrent callers from other
    // threads; unlike the constructor, this method is expected to be
    // called from inside an openMP parallel region. Uses its own
    // critical section, distinct from writeCluster's, since the two
    // methods write to independent files and so don't need to be
    // serialized against each other.
#ifdef _OPENMP
#pragma omp critical(clusterSpecOutputWrite)
#endif
    {
        for (std::size_t i = 0; i < wlObs_.size(); ++i)
        {
            clusterSpectraFile_ << std::right
                                 << std::setw(uidWidth) << formatUid(trial)
                                 << std::setw(numWidth) << formatSci(time)
                                 << std::setw(uidWidth) << formatUid(uid)
                                 << std::setw(numWidth) << formatSci(wlObs_.at(i))
                                 << std::setw(numWidth) << formatSci(spec.at(i)) << "\n";
        }
    }
}
