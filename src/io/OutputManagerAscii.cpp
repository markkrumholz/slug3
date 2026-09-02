/**
 * @file OutputManagerAscii.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerAscii
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "OutputManagerAscii.hpp"
#include "../core/Cluster.hpp"
#include "../core/Galaxy.hpp"
#include "../extinct/Extinct.hpp"
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
#include <utility>
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
// file. hasExtinct adds an "a_v" (mag) column, right before "rng",
// when SimControls::extinct() is set.
static void writeClustersHeader(std::ofstream& file, const bool hasExtinct)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "target_mass"
         << std::setw(numWidth) << "birth_mass"
         << std::setw(numWidth) << "form_time"
         << std::setw(numWidth) << "feh";
    if (hasExtinct) { file << std::setw(numWidth) << "a_v"; }
    file << std::setw(rngWidth) << "rng" << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "Msun"
         << std::setw(numWidth) << "Msun"
         << std::setw(numWidth) << "yr"
         << std::setw(numWidth) << "none";
    if (hasExtinct) { file << std::setw(numWidth) << "mag"; }
    file << std::setw(rngWidth) << "none" << "\n";
    const int numColumns = hasExtinct ? 5 : 4;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-')
         << std::string(static_cast<std::string::size_type>(rngWidth), '-') << "\n";
}

// Write the "spec_neb" and (if extinction was also requested)
// "spec_neb_extinct" columns, if a nebular emission grid was
// requested, for wavelength index i of one spectrum row -- a no-op
// otherwise. Factored out of writeClusterSpec()/writeGalaxySpec() to
// keep each within its cognitive-complexity budget. wlOffset/
// specNebExtinct share those methods' own wlOffset()-based
// fallback-to-0 handling for specNebExtinct's restricted coverage --
// see their callers' own comments.
static void writeSpecNebColumns(std::ofstream& file, const io::SimControls& simControls,
    const std::size_t i, const std::size_t wlOffset,
    const std::vector<double>& specNeb, const std::vector<double>& specNebExtinct)
{
    if (simControls.nebular() == nullptr) { return; }
    file << std::setw(numWidth) << formatSci(specNeb.at(i));

    if (simControls.extinct() == nullptr) { return; }
    const double specNebEx = (i >= wlOffset && (i - wlOffset) < specNebExtinct.size()) ?
        specNebExtinct.at(i - wlOffset) : 0.0;
    file << std::setw(numWidth) << formatSci(specNebEx);
}

// Write the cluster-spectra ascii header (column names, a row of
// units, and a dashed rule) to file. Unlike the cluster output file,
// this file is laid out one (wavelength, specific luminosity) pair
// per line rather than one cluster per line, since a spectrum can
// have thousands of wavelength points -- far too many to lay out as
// columns and still be human-readable. hasExtinct adds a "spec_ex"
// column (same units as "spec") when SimControls::extinct() is set --
// see writeClusterSpec's own comment for how its value is filled in
// at wavelengths outside the extincted spectrum's own coverage.
// hasNebular likewise adds a "spec_neb" column when
// SimControls::nebular() is set, and (if both hasExtinct and
// hasNebular) a further "spec_neb_ex" column.
static void writeClusterSpectraHeader(std::ofstream& file, const bool hasExtinct, const bool hasNebular)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "spec";
    if (hasExtinct) { file << std::setw(numWidth) << "spec_ex"; }
    if (hasNebular) { file << std::setw(numWidth) << "spec_neb"; }
    if (hasExtinct && hasNebular) { file << std::setw(numWidth) << "spec_neb_ex"; }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "Angstrom"
         << std::setw(numWidth) << "erg/s/Angstrom";
    if (hasExtinct) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    if (hasNebular) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    if (hasExtinct && hasNebular) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    file << "\n";
    const int numColumns = 3 + (hasExtinct ? 1 : 0) + (hasNebular ? 1 : 0) +
        ((hasExtinct && hasNebular) ? 1 : 0);
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

// Compute the column width to use for the "line_label" column in the
// cluster- and galaxy-nebular-line-luminosity files: numWidth unless
// some line's own label is longer than that, in which case the column
// is widened to fit it (plus photColPad blank characters) -- mirrors
// computePhotColWidths()'s own single-column logic
static auto computeLineLabelWidth(const std::vector<std::string>& lineLabels) -> int
{
    int width = numWidth;
    for (const auto& label : lineLabels)
    {
        width = std::max(width, static_cast<int>(label.size()) + photColPad);
    }
    return width;
}

// Build the "<filter>_ex" column names/units for the
// cluster-photometry file's extincted-filter columns: one per real
// filter, named after Filter::name() with "_ex" appended and using
// the same units as the ordinary filter columns -- excluding Lbol,
// which has no extincted counterpart (see
// OutputManagerH5::openClusterPhotGroup's identical phot_extinct
// sizing rationale). Both vectors come back empty if extinction was
// not requested, or no filter collection exists.
static auto buildExtinctFilterColumns(const io::SimControls& simControls)
    -> std::pair<std::vector<std::string>, std::vector<std::string>>
{
    if (simControls.extinct() == nullptr || simControls.filters() == nullptr)
    {
        return {};
    }
    std::vector<std::string> extinctFilterNames;
    for (const auto& name : simControls.filters()->filterNames())
    {
        extinctFilterNames.push_back(name + "_ex");
    }
    return { extinctFilterNames, simControls.filters()->filterUnits() };
}

// Build the "<filter>_neb" column names/units for the
// cluster-photometry file's nebular-inclusive filter columns -- one
// per real filter, mirroring buildExtinctFilterColumns's own
// "_ex"-suffix convention. Both vectors come back empty if nebular
// emission was not requested, or no filter collection exists.
static auto buildNebularFilterColumns(const io::SimControls& simControls)
    -> std::pair<std::vector<std::string>, std::vector<std::string>>
{
    if (simControls.nebular() == nullptr || simControls.filters() == nullptr)
    {
        return {};
    }
    std::vector<std::string> nebFilterNames;
    for (const auto& name : simControls.filters()->filterNames())
    {
        nebFilterNames.push_back(name + "_neb");
    }
    return { nebFilterNames, simControls.filters()->filterUnits() };
}

// Build the "<filter>_neb_ex" column names/units for the
// cluster-photometry file's extincted-and-nebular-inclusive filter
// columns -- one per real filter. Both vectors come back empty unless
// both nebular emission and extinction were requested, and a filter
// collection exists.
static auto buildNebularExtinctFilterColumns(const io::SimControls& simControls)
    -> std::pair<std::vector<std::string>, std::vector<std::string>>
{
    if (simControls.nebular() == nullptr || simControls.extinct() == nullptr ||
        simControls.filters() == nullptr)
    {
        return {};
    }
    std::vector<std::string> nebExtinctFilterNames;
    for (const auto& name : simControls.filters()->filterNames())
    {
        nebExtinctFilterNames.push_back(name + "_neb_ex");
    }
    return { nebExtinctFilterNames, simControls.filters()->filterUnits() };
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
// filter's name or unit string doesn't otherwise fit. extinctFilterNames/
// extinctFilterUnits/extinctColWidths lay out one further "<filter>_ex"
// column per real filter (excluding "Lbol", which has no extincted
// counterpart) after the ordinary filter columns, when
// SimControls::extinct() is set; all three are empty otherwise.
// nebFilterNames/nebFilterUnits/nebColWidths and
// nebExtinctFilterNames/nebExtinctFilterUnits/nebExtinctColWidths
// likewise lay out "<filter>_neb" and "<filter>_neb_ex" columns (see
// buildNebularFilterColumns()/buildNebularExtinctFilterColumns())
// after the extincted-filter columns, when SimControls::nebular() (and,
// for the "_neb_ex" columns, also SimControls::extinct()) is set.
static void writeClusterPhotHeader(std::ofstream& file,
    const std::vector<std::string>& filterNames,
    const std::vector<std::string>& filterUnits,
    const std::vector<int>& colWidths,
    const std::vector<std::string>& extinctFilterNames,
    const std::vector<std::string>& extinctFilterUnits,
    const std::vector<int>& extinctColWidths,
    const std::vector<std::string>& nebFilterNames,
    const std::vector<std::string>& nebFilterUnits,
    const std::vector<int>& nebColWidths,
    const std::vector<std::string>& nebExtinctFilterNames,
    const std::vector<std::string>& nebExtinctFilterUnits,
    const std::vector<int>& nebExtinctColWidths)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid";
    for (std::size_t i = 0; i < filterNames.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterNames.at(i);
    }
    for (std::size_t i = 0; i < extinctFilterNames.size(); ++i)
    {
        file << std::setw(extinctColWidths.at(i)) << extinctFilterNames.at(i);
    }
    for (std::size_t i = 0; i < nebFilterNames.size(); ++i)
    {
        file << std::setw(nebColWidths.at(i)) << nebFilterNames.at(i);
    }
    for (std::size_t i = 0; i < nebExtinctFilterNames.size(); ++i)
    {
        file << std::setw(nebExtinctColWidths.at(i)) << nebExtinctFilterNames.at(i);
    }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(uidWidth) << "none";
    for (std::size_t i = 0; i < filterUnits.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterUnits.at(i);
    }
    for (std::size_t i = 0; i < extinctFilterUnits.size(); ++i)
    {
        file << std::setw(extinctColWidths.at(i)) << extinctFilterUnits.at(i);
    }
    for (std::size_t i = 0; i < nebFilterUnits.size(); ++i)
    {
        file << std::setw(nebColWidths.at(i)) << nebFilterUnits.at(i);
    }
    for (std::size_t i = 0; i < nebExtinctFilterUnits.size(); ++i)
    {
        file << std::setw(nebExtinctColWidths.at(i)) << nebExtinctFilterUnits.at(i);
    }
    file << "\n";
    auto totalWidth = (static_cast<std::string::size_type>(2) * uidWidth) +
        static_cast<std::string::size_type>(numWidth);
    for (const int w : colWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : extinctColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : nebColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : nebExtinctColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    file << std::string(totalWidth, '-') << "\n";
}

// Write the galaxy-output ascii header (column names, a row of
// units, and a dashed rule) to file: one row per (trial, time) pair,
// holding the target and actual total stellar mass formed by that
// time -- unlike the cluster output file, a galaxy has no per-object
// identity, so there is no "uid" column here.
static void writeGalaxyHeader(std::ofstream& file)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(numWidth) << "target_mass"
         << std::setw(numWidth) << "actual_mass" << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(numWidth) << "Msun"
         << std::setw(numWidth) << "Msun" << "\n";
    constexpr int numColumns = 3;
    file << std::string(static_cast<std::string::size_type>(uidWidth), '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Write the galaxy-spectra ascii header (column names, a row of
// units, and a dashed rule) to file. Mirrors writeClusterSpectraHeader's
// own one-(wavelength, specific luminosity)-pair-per-line layout and
// "spec_ex"/"spec_neb"/"spec_neb_ex" naming, but with no "uid" column,
// since a galaxy (unlike a cluster) has no individual identity.
static void writeGalaxySpectraHeader(std::ofstream& file, const bool hasExtinct, const bool hasNebular)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "spec";
    if (hasExtinct) { file << std::setw(numWidth) << "spec_ex"; }
    if (hasNebular) { file << std::setw(numWidth) << "spec_neb"; }
    if (hasExtinct && hasNebular) { file << std::setw(numWidth) << "spec_neb_ex"; }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(numWidth) << "Angstrom"
         << std::setw(numWidth) << "erg/s/Angstrom";
    if (hasExtinct) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    if (hasNebular) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    if (hasExtinct && hasNebular) { file << std::setw(numWidth) << "erg/s/Angstrom"; }
    file << "\n";
    const int numColumns = 3 + (hasExtinct ? 1 : 0) + (hasNebular ? 1 : 0) +
        ((hasExtinct && hasNebular) ? 1 : 0);
    file << std::string(static_cast<std::string::size_type>(uidWidth), '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Write the galaxy-photometry ascii header (column names, a row of
// units, and a dashed rule) to file. Mirrors writeClusterPhotHeader's
// own one-row-per-(trial, time)-with-one-column-per-filter layout, but
// with no "uid" column, since a galaxy (unlike a cluster) has no
// individual identity. colWidths/extinctFilterNames/extinctFilterUnits/
// extinctColWidths/nebFilterNames/nebFilterUnits/nebColWidths/
// nebExtinctFilterNames/nebExtinctFilterUnits/nebExtinctColWidths carry
// exactly the same meaning as in writeClusterPhotHeader -- and, in
// practice, the same values, since both are built from the same
// SimControls::filters().
static void writeGalaxyPhotHeader(std::ofstream& file,
    const std::vector<std::string>& filterNames,
    const std::vector<std::string>& filterUnits,
    const std::vector<int>& colWidths,
    const std::vector<std::string>& extinctFilterNames,
    const std::vector<std::string>& extinctFilterUnits,
    const std::vector<int>& extinctColWidths,
    const std::vector<std::string>& nebFilterNames,
    const std::vector<std::string>& nebFilterUnits,
    const std::vector<int>& nebColWidths,
    const std::vector<std::string>& nebExtinctFilterNames,
    const std::vector<std::string>& nebExtinctFilterUnits,
    const std::vector<int>& nebExtinctColWidths)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time";
    for (std::size_t i = 0; i < filterNames.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterNames.at(i);
    }
    for (std::size_t i = 0; i < extinctFilterNames.size(); ++i)
    {
        file << std::setw(extinctColWidths.at(i)) << extinctFilterNames.at(i);
    }
    for (std::size_t i = 0; i < nebFilterNames.size(); ++i)
    {
        file << std::setw(nebColWidths.at(i)) << nebFilterNames.at(i);
    }
    for (std::size_t i = 0; i < nebExtinctFilterNames.size(); ++i)
    {
        file << std::setw(nebExtinctColWidths.at(i)) << nebExtinctFilterNames.at(i);
    }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr";
    for (std::size_t i = 0; i < filterUnits.size(); ++i)
    {
        file << std::setw(colWidths.at(i)) << filterUnits.at(i);
    }
    for (std::size_t i = 0; i < extinctFilterUnits.size(); ++i)
    {
        file << std::setw(extinctColWidths.at(i)) << extinctFilterUnits.at(i);
    }
    for (std::size_t i = 0; i < nebFilterUnits.size(); ++i)
    {
        file << std::setw(nebColWidths.at(i)) << nebFilterUnits.at(i);
    }
    for (std::size_t i = 0; i < nebExtinctFilterUnits.size(); ++i)
    {
        file << std::setw(nebExtinctColWidths.at(i)) << nebExtinctFilterUnits.at(i);
    }
    file << "\n";
    auto totalWidth = static_cast<std::string::size_type>(uidWidth) +
        static_cast<std::string::size_type>(numWidth);
    for (const int w : colWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : extinctColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : nebColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    for (const int w : nebExtinctColWidths) { totalWidth += static_cast<std::string::size_type>(w); }
    file << std::string(totalWidth, '-') << "\n";
}

// Write the cluster-nebular-line-luminosity ascii header (column
// names, a row of units, and a dashed rule) to file. Like the
// cluster-spectra file, this is laid out one row per (line, at one
// output time for one cluster) rather than one cluster per line, since
// there can be many lines -- but unlike it, each row also carries its
// own "line_label" column (width lineLabelWidth -- see
// computeLineLabelWidth()) identifying which line that row belongs to,
// since (unlike wavelength) a line's own position in the table is not
// self-describing. hasExtinct adds a "line_lum_ex" column (same units
// as "line_lum") when SimControls::extinct() is set.
static void writeClusterNebLinesHeader(std::ofstream& file, const int lineLabelWidth, const bool hasExtinct)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(uidWidth) << "uid"
         << std::setw(lineLabelWidth) << "line_label"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "line_lum";
    if (hasExtinct) { file << std::setw(numWidth) << "line_lum_ex"; }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(uidWidth) << "none"
         << std::setw(lineLabelWidth) << "none"
         << std::setw(numWidth) << "Angstrom"
         << std::setw(numWidth) << "erg/s";
    if (hasExtinct) { file << std::setw(numWidth) << "erg/s"; }
    file << "\n";
    const int numColumns = hasExtinct ? 4 : 3;
    file << std::string(static_cast<std::string::size_type>(2) * uidWidth, '-')
         << std::string(static_cast<std::string::size_type>(lineLabelWidth), '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
}

// Write the galaxy-nebular-line-luminosity ascii header (column names,
// a row of units, and a dashed rule) to file. Mirrors
// writeClusterNebLinesHeader()'s own layout, but with no "uid" column,
// since a galaxy (unlike a cluster) has no individual identity.
static void writeGalaxyNebLinesHeader(std::ofstream& file, const int lineLabelWidth, const bool hasExtinct)
{
    file << std::right << std::setw(uidWidth) << "trial"
         << std::setw(numWidth) << "time"
         << std::setw(lineLabelWidth) << "line_label"
         << std::setw(numWidth) << "wl"
         << std::setw(numWidth) << "line_lum";
    if (hasExtinct) { file << std::setw(numWidth) << "line_lum_ex"; }
    file << "\n";
    file << std::right << std::setw(uidWidth) << "none"
         << std::setw(numWidth) << "yr"
         << std::setw(lineLabelWidth) << "none"
         << std::setw(numWidth) << "Angstrom"
         << std::setw(numWidth) << "erg/s";
    if (hasExtinct) { file << std::setw(numWidth) << "erg/s"; }
    file << "\n";
    const int numColumns = hasExtinct ? 4 : 3;
    file << std::string(static_cast<std::string::size_type>(uidWidth), '-')
         << std::string(static_cast<std::string::size_type>(lineLabelWidth), '-')
         << std::string(static_cast<std::string::size_type>(numColumns) * numWidth, '-') << "\n";
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

    openClustersFile();
    openClusterSpectraFile();
    openClusterNebLinesFile();
    openClusterPhotFile();
    openGalaxyFile();
    openGalaxySpectraFile();
    openGalaxyNebLinesFile();
    openGalaxyPhotFile();
}

// Open the cluster output file and write its header, if cluster
// output is enabled for this simulation (see OutputManagerH5::
// openClustersGroup()'s identical gating condition)
void io::OutputManagerAscii::openClustersFile()
{
    if (!writeCluster_) { return; }

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
    writeClustersHeader(clustersFile_, simControls_.extinct() != nullptr);
}

// Open the cluster-spectra output file and write its header, if a
// spectral synthesizer was requested and output.write_cluster_spec
// (optional, defaults to true) was not set to false -- spectra can be
// wanted only as an intermediate for computing photometry, in which
// case writing them out as well just wastes disk space
void io::OutputManagerAscii::openClusterSpectraFile()
{
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeClusterSpec_) { return; }

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
    writeClusterSpectraHeader(clusterSpectraFile_, simControls_.extinct() != nullptr,
        simControls_.nebular() != nullptr);
}

// Open the cluster-nebular-line-luminosity output file and write its
// header, if a nebular emission grid was requested -- see this
// method's own header comment for the specsyn()/write_cluster_spec
// gating it mirrors from openClusterSpectraFile()
void io::OutputManagerAscii::openClusterNebLinesFile()
{
    if (simControls_.nebular() == nullptr) { return; }
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeClusterSpec_) { return; }

    const auto clusterNebLinesPath = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_cluster_neb_lines.txt");
    if (std::filesystem::exists(clusterNebLinesPath))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + clusterNebLinesPath.string() + " already exists");
    }

    clusterNebLinesFile_.open(clusterNebLinesPath);
    if (!clusterNebLinesFile_)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + clusterNebLinesPath.string());
    }
    lineLabelWidth_ = computeLineLabelWidth(simControls_.nebular()->lineLabel());
    writeClusterNebLinesHeader(clusterNebLinesFile_, lineLabelWidth_, simControls_.extinct() != nullptr);
}

// Open the cluster-photometry output file and write its header, if a
// filter collection or the bolometric luminosity was requested
void io::OutputManagerAscii::openClusterPhotFile()
{
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeClusterPhot_) { return; }

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

    const auto [extinctFilterNames, extinctFilterUnits] = buildExtinctFilterColumns(simControls_);
    photExtinctColWidths_ = computePhotColWidths(extinctFilterNames, extinctFilterUnits);
    const auto [nebFilterNames, nebFilterUnits] = buildNebularFilterColumns(simControls_);
    photNebColWidths_ = computePhotColWidths(nebFilterNames, nebFilterUnits);
    const auto [nebExtinctFilterNames, nebExtinctFilterUnits] = buildNebularExtinctFilterColumns(simControls_);
    photNebExtinctColWidths_ = computePhotColWidths(nebExtinctFilterNames, nebExtinctFilterUnits);
    writeClusterPhotHeader(clusterPhotFile_, filterNames, filterUnits, photColWidths_,
        extinctFilterNames, extinctFilterUnits, photExtinctColWidths_,
        nebFilterNames, nebFilterUnits, photNebColWidths_,
        nebExtinctFilterNames, nebExtinctFilterUnits, photNebExtinctColWidths_);
}

// Open the galaxy output file and write its header, for a galaxy-type
// simulation with output.write_galaxy (optional, defaults to true)
// not set to false. A no-op for a cluster-type simulation, since a
// cluster-type simulation has no Galaxy object at all.
void io::OutputManagerAscii::openGalaxyFile()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (!writeGalaxy_) { return; }

    const auto galaxyPath = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_galaxy.txt");
    if (std::filesystem::exists(galaxyPath))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + galaxyPath.string() + " already exists");
    }

    galaxyFile_.open(galaxyPath);
    if (!galaxyFile_)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + galaxyPath.string());
    }
    writeGalaxyHeader(galaxyFile_);
}

// Open the galaxy-spectra output file and write its header, for a
// galaxy-type simulation with a spectral synthesizer requested
void io::OutputManagerAscii::openGalaxySpectraFile()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeGalaxySpec_) { return; }

    wlObs_ = simControls_.specsyn()->wlObs();

    const auto galaxySpectraPath = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_galaxy_spectra.txt");
    if (std::filesystem::exists(galaxySpectraPath))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + galaxySpectraPath.string() + " already exists");
    }

    galaxySpectraFile_.open(galaxySpectraPath);
    if (!galaxySpectraFile_)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + galaxySpectraPath.string());
    }
    writeGalaxySpectraHeader(galaxySpectraFile_, simControls_.extinct() != nullptr,
        simControls_.nebular() != nullptr);
}

// Open the galaxy-nebular-line-luminosity output file and write its
// header, for a galaxy-type simulation with a nebular emission grid
// requested -- see openClusterNebLinesFile()'s own comment for the
// specsyn()/write_galaxy_spec gating it mirrors from
// openGalaxySpectraFile()
void io::OutputManagerAscii::openGalaxyNebLinesFile()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.nebular() == nullptr) { return; }
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeGalaxySpec_) { return; }

    const auto galaxyNebLinesPath = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_galaxy_neb_lines.txt");
    if (std::filesystem::exists(galaxyNebLinesPath))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + galaxyNebLinesPath.string() + " already exists");
    }

    galaxyNebLinesFile_.open(galaxyNebLinesPath);
    if (!galaxyNebLinesFile_)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + galaxyNebLinesPath.string());
    }
    lineLabelWidth_ = computeLineLabelWidth(simControls_.nebular()->lineLabel());
    writeGalaxyNebLinesHeader(galaxyNebLinesFile_, lineLabelWidth_, simControls_.extinct() != nullptr);
}

// Open the galaxy-photometry output file and write its header, for a
// galaxy-type simulation with a filter collection or the bolometric
// luminosity requested -- mirrors openClusterPhotFile()'s own
// filter-list construction
void io::OutputManagerAscii::openGalaxyPhotFile()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeGalaxyPhot_) { return; }

    const auto galaxyPhotPath = std::filesystem::path(simControls_.outDir()) /
        (simControls_.modelName() + "_galaxy_phot.txt");
    if (std::filesystem::exists(galaxyPhotPath))
    {
        throw std::runtime_error(
            "OutputManagerAscii: output file " + galaxyPhotPath.string() + " already exists");
    }

    galaxyPhotFile_.open(galaxyPhotPath);
    if (!galaxyPhotFile_)
    {
        throw std::runtime_error(
            "OutputManagerAscii: unable to open output file " + galaxyPhotPath.string());
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

    const auto [extinctFilterNames, extinctFilterUnits] = buildExtinctFilterColumns(simControls_);
    photExtinctColWidths_ = computePhotColWidths(extinctFilterNames, extinctFilterUnits);
    const auto [nebFilterNames, nebFilterUnits] = buildNebularFilterColumns(simControls_);
    photNebColWidths_ = computePhotColWidths(nebFilterNames, nebFilterUnits);
    const auto [nebExtinctFilterNames, nebExtinctFilterUnits] = buildNebularExtinctFilterColumns(simControls_);
    photNebExtinctColWidths_ = computePhotColWidths(nebExtinctFilterNames, nebExtinctFilterUnits);
    writeGalaxyPhotHeader(galaxyPhotFile_, filterNames, filterUnits, photColWidths_,
        extinctFilterNames, extinctFilterUnits, photExtinctColWidths_,
        nebFilterNames, nebFilterUnits, photNebColWidths_,
        nebExtinctFilterNames, nebExtinctFilterUnits, photNebExtinctColWidths_);
}

io::OutputManagerAscii::~OutputManagerAscii()
{
    if (clustersFile_.is_open()) { clustersFile_.close(); }
    if (clusterSpectraFile_.is_open()) { clusterSpectraFile_.close(); }
    if (clusterNebLinesFile_.is_open()) { clusterNebLinesFile_.close(); }
    if (clusterPhotFile_.is_open()) { clusterPhotFile_.close(); }
    if (galaxyFile_.is_open()) { galaxyFile_.close(); }
    if (galaxySpectraFile_.is_open()) { galaxySpectraFile_.close(); }
    if (galaxyNebLinesFile_.is_open()) { galaxyNebLinesFile_.close(); }
    if (galaxyPhotFile_.is_open()) { galaxyPhotFile_.close(); }
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
                      << std::setw(numWidth) << formatSci(cluster.feH());
        if (simControls_.extinct() != nullptr)
        {
            clustersFile_ << std::setw(numWidth) << formatSci(cluster.aV());
        }
        clustersFile_ << std::setw(rngWidth) << std::string(cluster.rngState().data()) << "\n";
    }
}

// Write one line per wavelength (trial, time, uid, wavelength,
// specific luminosity) to the cluster-spectra output file. A no-op
// if spectral synthesis was not enabled for this simulation, or if
// the cluster has disrupted -- a disrupted cluster is no longer an
// observable object, though its light still belongs in the total
// galaxy spectrum, which is handled elsewhere.
void io::OutputManagerAscii::writeClusterSpec(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (!clusterSpectraFile_.is_open()) { return; }
    if (cluster.isDisrupted()) { return; }

    const unsigned long uid = cluster.uid();
    const auto& spec = cluster.spec();
    const auto* ext = simControls_.extinct();
    const auto& specExtinct = cluster.specExtinct();
    // specExtinct is tabulated on a subset of wlObs_ -- the first
    // wlOffset() entries, and any past specExtinct's own end, fall
    // outside the extinction curve's native coverage and have no
    // extincted value, so spec_ex reads 0 there rather than the
    // (nonsensical) unextincted spec value
    const std::size_t wlOffset = (ext != nullptr) ? ext->wlOffset() : 0;
    // specNeb is tabulated on the full wlObs_ grid, same as spec, so
    // needs no offset handling of its own; specNebExtinct, like
    // specExtinct, is restricted to the extinction curve's own
    // coverage and so shares wlOffset with it
    const auto& specNeb = cluster.specNeb();
    const auto& specNebExtinct = cluster.specNebExtinct();

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
                                 << std::setw(numWidth) << formatSci(spec.at(i));
            if (ext != nullptr)
            {
                const double specEx = (i >= wlOffset && (i - wlOffset) < specExtinct.size()) ?
                    specExtinct.at(i - wlOffset) : 0.0;
                clusterSpectraFile_ << std::setw(numWidth) << formatSci(specEx);
            }
            writeSpecNebColumns(clusterSpectraFile_, simControls_, i, wlOffset, specNeb, specNebExtinct);
            clusterSpectraFile_ << "\n";
        }
    }

    writeClusterNebLines(trial, time, cluster);
}

// Write one row per nebular emission line (trial, time, uid, line
// label, line wavelength, and line luminosity, plus -- if extinction
// was also requested -- the extincted line luminosity) to the
// cluster-nebular-line-luminosity output file. A no-op if that file
// was not opened (see openClusterNebLinesFile()) -- in particular, if
// no nebular emission grid was requested. Called from writeClusterSpec()
// after it finishes writing the cluster's own spectrum; cluster.lineLum()/
// lineLumExtinct() are already current by then (writeClusterSpec() has
// already forced a current spectrum via cluster.spec()), so no separate
// "fetch outside the critical section" step is needed here, unlike
// writeClusterSpec()'s own spec()/specExtinct() fetch.
void io::OutputManagerAscii::writeClusterNebLines(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (!clusterNebLinesFile_.is_open()) { return; }

    const auto* neb = simControls_.nebular();
    const auto* ext = simControls_.extinct();
    const auto& lineWl = neb->lineWl();
    const auto& lineLabel = neb->lineLabel();
    const auto& lineLum = cluster.lineLum();
    const auto& lineLumExtinct = cluster.lineLumExtinct();
    const unsigned long uid = cluster.uid();

#ifdef _OPENMP
#pragma omp critical(clusterNebLinesOutputWrite)
#endif
    {
        for (std::size_t i = 0; i < lineWl.size(); ++i)
        {
            clusterNebLinesFile_ << std::right
                                  << std::setw(uidWidth) << formatUid(trial)
                                  << std::setw(numWidth) << formatSci(time)
                                  << std::setw(uidWidth) << formatUid(uid)
                                  << std::setw(lineLabelWidth_) << lineLabel.at(i)
                                  << std::setw(numWidth) << formatSci(lineWl.at(i))
                                  << std::setw(numWidth) << formatSci(lineLum.at(i));
            if (ext != nullptr)
            {
                clusterNebLinesFile_ << std::setw(numWidth) << formatSci(lineLumExtinct.at(i));
            }
            clusterNebLinesFile_ << "\n";
        }
    }
}

// Write one line (trial, time, uid, then one column per filter) to
// the cluster-photometry output file. A no-op if no filter collection
// or bolometric luminosity was requested for this simulation, or if
// the cluster has disrupted -- a disrupted cluster is no longer an
// observable object.
void io::OutputManagerAscii::writeClusterPhot(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (!clusterPhotFile_.is_open()) { return; }
    if (cluster.isDisrupted()) { return; }

    // phot()/photExtinct()/lbol() are computed here, outside the
    // critical section below, since they may need to lazily
    // (re)compute -- potentially expensive work that must not run
    // while holding an OpenMP critical section, which would
    // needlessly serialize it across threads
    const unsigned long uid = cluster.uid();
    auto phot = cluster.phot();
    if (simControls_.computeLbol()) { phot.push_back(cluster.lbol()); }
    const auto& photExtinct = cluster.photExtinct();
    const auto& photNeb = cluster.photNeb();
    const auto& photNebExtinct = cluster.photNebExtinct();

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
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            for (std::size_t i = 0; i < photExtinct.size(); ++i)
            {
                clusterPhotFile_ << std::setw(photExtinctColWidths_.at(i)) << formatSci(photExtinct.at(i));
            }
        }
        if (simControls_.nebular() != nullptr && simControls_.filters() != nullptr)
        {
            for (std::size_t i = 0; i < photNeb.size(); ++i)
            {
                clusterPhotFile_ << std::setw(photNebColWidths_.at(i)) << formatSci(photNeb.at(i));
            }
            if (simControls_.extinct() != nullptr)
            {
                for (std::size_t i = 0; i < photNebExtinct.size(); ++i)
                {
                    clusterPhotFile_ << std::setw(photNebExtinctColWidths_.at(i)) << formatSci(photNebExtinct.at(i));
                }
            }
        }
        clusterPhotFile_ << "\n";
    }
}

// Write one fixed-width row of galaxy data (trial, time, target_mass,
// actual_mass -- no uid, since a galaxy has no individual identity)
// to the galaxy output file, then call writeCluster() on every
// currently-alive (non-disrupted) cluster in galaxy. A no-op if
// galaxy output was not enabled for this simulation.
void io::OutputManagerAscii::writeGalaxy(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyFile_.is_open())
    {
        // Guard the actual write against concurrent callers from other
        // threads; unlike the constructor, this method is expected to
        // be called from inside an openMP parallel region
#ifdef _OPENMP
#pragma omp critical(galaxyOutputWrite)
#endif
        {
            galaxyFile_ << std::right
                        << std::setw(uidWidth) << formatUid(trial)
                        << std::setw(numWidth) << formatSci(time)
                        << std::setw(numWidth) << formatSci(galaxy.targetMass())
                        << std::setw(numWidth) << formatSci(galaxy.actualMass()) << "\n";
        }
    }

    for (const auto& cluster : galaxy.clusters()) { writeCluster(trial, cluster); }
}

// Write one line per wavelength (trial, time, wavelength, specific
// luminosity -- no uid) to the galaxy-spectra output file, then call
// writeClusterSpec() on every currently-alive (non-disrupted) cluster
// in galaxy. A no-op if spectral synthesis was not enabled for this
// simulation.
void io::OutputManagerAscii::writeGalaxySpec(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxySpectraFile_.is_open())
    {
        const auto& spec = galaxy.spec();
        const auto* ext = simControls_.extinct();
        const auto& specExtinct = galaxy.specExtinct();
        // See writeClusterSpec's own comment on wlOffset()/spec_ex's
        // fallback to 0 outside the extinction curve's own coverage
        const std::size_t wlOffset = (ext != nullptr) ? ext->wlOffset() : 0;
        const auto& specNeb = galaxy.specNeb();
        const auto& specNebExtinct = galaxy.specNebExtinct();

        // Guard the actual writes against concurrent callers from
        // other threads; uses its own critical section, distinct from
        // writeGalaxy's, since the two write to independent files
#ifdef _OPENMP
#pragma omp critical(galaxySpecOutputWrite)
#endif
        {
            for (std::size_t i = 0; i < wlObs_.size(); ++i)
            {
                galaxySpectraFile_ << std::right
                                    << std::setw(uidWidth) << formatUid(trial)
                                    << std::setw(numWidth) << formatSci(time)
                                    << std::setw(numWidth) << formatSci(wlObs_.at(i))
                                    << std::setw(numWidth) << formatSci(spec.at(i));
                if (ext != nullptr)
                {
                    const double specEx = (i >= wlOffset && (i - wlOffset) < specExtinct.size()) ?
                        specExtinct.at(i - wlOffset) : 0.0;
                    galaxySpectraFile_ << std::setw(numWidth) << formatSci(specEx);
                }
                writeSpecNebColumns(galaxySpectraFile_, simControls_, i, wlOffset, specNeb, specNebExtinct);
                galaxySpectraFile_ << "\n";
            }
        }
    }

    writeGalaxyNebLines(trial, time, galaxy);

    for (auto& cluster : galaxy.clusters()) { writeClusterSpec(trial, time, cluster); }
}

// Write one row per nebular emission line (trial, time, line label,
// line wavelength, and line luminosity, plus -- if extinction was also
// requested -- the extincted line luminosity) to the
// galaxy-nebular-line-luminosity output file. A no-op if that file was
// not opened (see openGalaxyNebLinesFile()) -- in particular, if no
// nebular emission grid was requested. Called from writeGalaxySpec(),
// after it finishes writing the galaxy's own spectrum but before it
// recurses into writeClusterSpec() for each cluster -- see
// writeClusterNebLines()'s own comment for why galaxy.lineLum()/
// lineLumExtinct() need no separate "fetch outside the critical
// section" step here either.
void io::OutputManagerAscii::writeGalaxyNebLines(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (!galaxyNebLinesFile_.is_open()) { return; }

    const auto* neb = simControls_.nebular();
    const auto* ext = simControls_.extinct();
    const auto& lineWl = neb->lineWl();
    const auto& lineLabel = neb->lineLabel();
    const auto& lineLum = galaxy.lineLum();
    const auto& lineLumExtinct = galaxy.lineLumExtinct();

#ifdef _OPENMP
#pragma omp critical(galaxyNebLinesOutputWrite)
#endif
    {
        for (std::size_t i = 0; i < lineWl.size(); ++i)
        {
            galaxyNebLinesFile_ << std::right
                                 << std::setw(uidWidth) << formatUid(trial)
                                 << std::setw(numWidth) << formatSci(time)
                                 << std::setw(lineLabelWidth_) << lineLabel.at(i)
                                 << std::setw(numWidth) << formatSci(lineWl.at(i))
                                 << std::setw(numWidth) << formatSci(lineLum.at(i));
            if (ext != nullptr)
            {
                galaxyNebLinesFile_ << std::setw(numWidth) << formatSci(lineLumExtinct.at(i));
            }
            galaxyNebLinesFile_ << "\n";
        }
    }
}

// Write one line (trial, time, then one column per filter) to the
// galaxy-photometry output file, then call writeClusterPhot() on
// every currently-alive (non-disrupted) cluster in galaxy. A no-op if
// no filter collection or bolometric luminosity was requested for
// this simulation.
void io::OutputManagerAscii::writeGalaxyPhot(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyPhotFile_.is_open())
    {
        // phot()/photExtinct()/lbol() are computed here, outside the
        // critical section below -- see writeClusterPhot's own comment
        auto phot = galaxy.phot();
        if (simControls_.computeLbol()) { phot.push_back(galaxy.lbol()); }
        const auto& photExtinct = galaxy.photExtinct();
        const auto& photNeb = galaxy.photNeb();
        const auto& photNebExtinct = galaxy.photNebExtinct();

        // Guard the actual writes against concurrent callers from
        // other threads; uses its own critical section, distinct from
        // writeGalaxy's/writeGalaxySpec's, since all three write to
        // independent files
#ifdef _OPENMP
#pragma omp critical(galaxyPhotOutputWrite)
#endif
        {
            galaxyPhotFile_ << std::right
                             << std::setw(uidWidth) << formatUid(trial)
                             << std::setw(numWidth) << formatSci(time);
            for (std::size_t i = 0; i < phot.size(); ++i)
            {
                galaxyPhotFile_ << std::setw(photColWidths_.at(i)) << formatSci(phot.at(i));
            }
            if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
            {
                for (std::size_t i = 0; i < photExtinct.size(); ++i)
                {
                    galaxyPhotFile_ << std::setw(photExtinctColWidths_.at(i)) << formatSci(photExtinct.at(i));
                }
            }
            if (simControls_.nebular() != nullptr && simControls_.filters() != nullptr)
            {
                for (std::size_t i = 0; i < photNeb.size(); ++i)
                {
                    galaxyPhotFile_ << std::setw(photNebColWidths_.at(i)) << formatSci(photNeb.at(i));
                }
                if (simControls_.extinct() != nullptr)
                {
                    for (std::size_t i = 0; i < photNebExtinct.size(); ++i)
                    {
                        galaxyPhotFile_ << std::setw(photNebExtinctColWidths_.at(i)) << formatSci(photNebExtinct.at(i));
                    }
                }
            }
            galaxyPhotFile_ << "\n";
        }
    }

    for (auto& cluster : galaxy.clusters()) { writeClusterPhot(trial, time, cluster); }
}

// See this method's own header comment: checkpointing is only ever
// supported with HDF5 output, so this always throws.
void io::OutputManagerAscii::checkpoint(unsigned long /*trialsCompleted*/)
{
    throw std::runtime_error(
        "OutputManagerAscii::checkpoint: checkpointing is not supported "
        "with ascii output");
}

// See this method's own header comment: restarting is only ever
// supported with HDF5 output, so this always throws.
auto io::OutputManagerAscii::restartTrialsDone() const -> unsigned long
{
    throw std::runtime_error(
        "OutputManagerAscii::restartTrialsDone: restarting is not "
        "supported with ascii output");
}

// See this method's own header comment: restarting is only ever
// supported with HDF5 output, so this always throws.
auto io::OutputManagerAscii::restartMaxTrial() const -> unsigned long
{
    throw std::runtime_error(
        "OutputManagerAscii::restartMaxTrial: restarting is not "
        "supported with ascii output");
}

// See this method's own header comment: nothing to do for ascii output
void io::OutputManagerAscii::notifyEarlyTermination(unsigned long /*trialsCompleted*/)
{
}
