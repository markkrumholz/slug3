/**
 * @file OutputManagerAscii.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerAscii
 * @date 2026-07-17
 */

#include "OutputManagerAscii.hpp"
#include "../core/Cluster.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../utils/RngThread.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "io/SlugVersion.hpp"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

// Column headings, and the field widths used to lay them out, for
// the ascii cluster output file. uidWidth accommodates a 9-digit
// integer; numWidth accommodates a number in exponential notation
// with six decimal places (e.g. "-1.234567e+01"). Both include a
// couple of extra characters of padding so that columns are visibly
// separated. rngWidth accommodates a serialized rng state (see
// utils::RngState), which -- unlike every other column -- can be up
// to rngStateWidth - 1 = 127 characters wide.
static constexpr int uidWidth = 12;
static constexpr int numWidth = 16;
static constexpr int rngWidth = static_cast<int>(utils::rngStateWidth) + 4;

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

// Write the cluster-output ascii header (column names, a row of
// units -- "none" for a dimensionless column -- and a dashed rule) to
// file
static void writeClustersHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "target_mass"
         << std::setw(numWidth) << "birth_mass"
         << std::setw(numWidth) << "form_time"
         << std::setw(numWidth) << "feh"
         << std::setw(rngWidth) << "rng" << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "Msun"
         << std::setw(numWidth) << "Msun"
         << std::setw(numWidth) << "yr"
         << std::setw(numWidth) << "none"
         << std::setw(rngWidth) << "none" << "\n";
    constexpr auto numColumns = 4;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-')
         << std::string(static_cast<std::string::size_type>(rngWidth), '-') << "\n";
}

// Write the cluster-spectra ascii header (column names, a row of
// units, and a dashed rule) to file. Unlike the cluster output file,
// this file is laid out one (wavelength, specific luminosity) pair
// per line rather than one cluster per line, since a spectrum can
// have thousands of wavelength points -- far too many to lay out as
// columns and still be human-readable.
static void writeClusterSpectraHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "spec" << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "Angstrom"
         << std::setw(numWidth) << "erg/s/Angstrom" << "\n";
    constexpr auto numColumns = 3;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Number of blank characters of padding a filter column is guaranteed
// beyond its own name/unit string, or numWidth if that's already
// wider -- see computePhotColWidths() below
static constexpr int photColPad = 2;

// Compute the column width to use for each filter in the
// cluster-photometry file: numWidth (wide enough for any formatSci()
// value) unless the filter's own name or unit string is longer than
// that, in which case the column is widened to fit it (plus
// photColPad blank characters), so a long filter name/unit never runs
// into its neighboring columns
static auto computePhotColWidths(const std::vector<std::string>& filterNames,
    const std::vector<std::string>& filterUnits) -> std::vector<int>
{
    std::vector<int> widths(filterNames.size());
    for (std::size_t i = 0; i < filterNames.size(); ++i)
    {
        const auto nameWidth = static_cast<int>(filterNames.at(i).size()) + photColPad;
        const auto unitWidth = static_cast<int>(filterUnits.at(i).size()) + photColPad;
        widths.at(i) = std::max({ numWidth, nameWidth, unitWidth });
    }
    return widths;
}

// Write the cluster-photometry ascii header (column names, a row of
// units, and a dashed rule) to file. Like the cluster output file,
// this is laid out one cluster (at one output time) per line, with
// one column per filter -- named after Filter::name(), via
// FilterCollection::filterNames(), with units from
// FilterCollection::filterUnits() -- rather than the spectrum's
// thousands of per-wavelength rows that writeClusterSpectraHeader
// lays out. colWidths (see computePhotColWidths()) gives each filter
// column's own width, which may be wider than numWidth if the
// filter's name or unit string doesn't otherwise fit.
static void writeClusterPhotHeader(std::ofstream& file,
    const std::vector<std::string>& filterNames,
    const std::vector<std::string>& filterUnits,
    const std::vector<int>& colWidths)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid";
    for (std::size_t i = 0; i < filterNames.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterNames.at(i);
    }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(uidWidth) << "none";
    for (std::size_t i = 0; i < filterUnits.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterUnits.at(i);
    }
    file << "\n";
    auto totalWidth = (static_cast<std::string::size_type>(2) * uidWidth) +
        static_cast<std::string::size_type>(numWidth);
    for (const int w : colWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    file << std::string(totalWidth, '-') << "\n";
}

// Ascii constructor: open the summary file, write the header
// (slug-hash, date, time), then dump the toml input deck, and close
// the file. If the simulation outputs individual clusters, also
// open the cluster output file, write its column-header rows, and
// leave it open for later writing.
io::OutputManagerAscii::OutputManagerAscii(
    const SimControls& simControls,
    const toml::table& inputDeck) :
    OutputManager(simControls, inputDeck)
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

    if (simControls_.specsyn() != nullptr)
    {
        wlObs_ = simControls_.specsyn()->wlObs();

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

    if (simControls_.filters() != nullptr || simControls_.computeLbol())
    {
        const auto clusterPhotPath = std::filesystem::path(simControls_.outDir()) /
            (simControls_.modelName() + "_cluster_phot.txt");
        if (std::filesystem::exists(clusterPhotPath))
        {
            throw std::runtime_error(
                "OutputManagerAscii: output file " + clusterPhotPath.string() + " already exists");
        }

        clusterPhotFile_.open(clusterPhotPath);
        if (!clusterPhotFile_)
        {
            throw std::runtime_error(
                "OutputManagerAscii: unable to open output file " + clusterPhotPath.string());
        }
        std::vector<std::string> filterNames;
        std::vector<std::string> filterUnits;
        if (simControls_.filters() != nullptr)
        {
            filterNames = simControls_.filters()->filterNames();
            filterUnits = simControls_.filters()->filterUnits();
        }
        if (simControls_.computeLbol())
        {
            filterNames.emplace_back("Lbol");
            filterUnits.emplace_back("Lsun");
        }
        photColWidths_ = computePhotColWidths(filterNames, filterUnits);
        writeClusterPhotHeader(clusterPhotFile_, filterNames, filterUnits, photColWidths_);
    }
}

io::OutputManagerAscii::~OutputManagerAscii()
{
    if (clustersFile_.is_open()) { clustersFile_.close(); }
    if (clusterSpectraFile_.is_open()) { clusterSpectraFile_.close(); }
    if (clusterPhotFile_.is_open()) { clusterPhotFile_.close(); }
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
                      << std::setw(numWidth) << formatSci(cluster.feH())
                      << std::setw(rngWidth) << std::string(cluster.rngState().data()) << "\n";
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

// Write one line (trial, time, uid, then one column per filter) to
// the cluster-photometry output file. A no-op if no filter collection
// or bolometric luminosity was requested for this simulation, or if
// the cluster has disrupted -- a disrupted cluster is no longer an
// observable object.
void io::OutputManagerAscii::writeClusterPhot(
    const unsigned long trial, const double time, const core::Cluster& cluster)
{
    if (!clusterPhotFile_.is_open()) { return; }
    if (cluster.isDisrupted()) { return; }

    const unsigned long uid = cluster.uid();
    auto phot = cluster.phot();
    if (simControls_.computeLbol()) { phot.push_back(cluster.lbol()); }

    // Guard the actual writes against concurrent callers from other
    // threads; unlike the constructor, this method is expected to be
    // called from inside an openMP parallel region. Uses its own
    // critical section, distinct from writeCluster's/writeClusterSpec's,
    // since all three methods write to independent files and so don't
    // need to be serialized against each other.
#ifdef _OPENMP
#pragma omp critical(clusterPhotOutputWrite)
#endif
    {
        clusterPhotFile_ << std::right
                          << std::setw(uidWidth) << formatUid(trial)
                          << std::setw(numWidth) << formatSci(time)
                          << std::setw(uidWidth) << formatUid(uid);
        for (std::size_t i = 0; i < phot.size(); ++i)
        {
            clusterPhotFile_ << std::setw(photColWidths_.at(i)) << formatSci(phot.at(i));
        }
        clusterPhotFile_ << "\n";
    }
}
