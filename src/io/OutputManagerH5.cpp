/**
 * @file OutputManagerH5.cpp
 * @author Mark Krumholz
 * @brief Implementation of OutputManagerH5
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#include "OutputManagerH5.hpp"
#include "../core/Cluster.hpp"
#include "../core/Galaxy.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../utils/HDF5Utils.hpp"
#include "../utils/RngThread.hpp"
#include "../utils/ThreadVec.hpp"
#include "../utils/UniqueIDManager.hpp"
#include "OutputManager.hpp"
#include "SimControls.hpp"
#include "hdf5.h" // NOLINT(misc-include-cleaner)
#include "io/SlugVersion.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iomanip> // NOLINT(misc-include-cleaner) -- used for std::setw/std::setfill in the constructor's own thread_NNNN.h5 filename construction
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <toml.hpp>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif

// Suppress clang-tidy warnings in this namespace caused by just
// including hdf5.h, instead of the individual HDF5 headers, since
// this is the paradigm that HDF5 wants
// NOLINTBEGIN(misc-include-cleaner)

// Scan outDir for entries -- either modelName_chkNNNNN.h5 files or
// modelName_chkNNNNN directories (see checkpointModelName()) -- whose
// name starts with prefix, and return the largest NNNNN found, or no
// value if outDir does not exist or contains no such entry. Factored
// out of restartSetup() purely to keep that function's own cognitive
// complexity down; has no other caller.
static auto findMaxCheckpointNumber(const std::filesystem::path& outDir,
    const std::string& prefix) -> std::optional<unsigned long>
{
    std::optional<unsigned long> maxCheckpoint;
    if (!std::filesystem::exists(outDir)) { return maxCheckpoint; }

    for (const auto& entry : std::filesystem::directory_iterator(outDir))
    {
        std::string suffix;
        if (entry.is_regular_file() && entry.path().extension() == ".h5")
        {
            const auto stem = entry.path().stem().string();
            if (!stem.starts_with(prefix)) { continue; }
            suffix = stem.substr(prefix.size());
        }
        else if (entry.is_directory())
        {
            const auto dirName = entry.path().filename().string();
            if (!dirName.starts_with(prefix)) { continue; }
            suffix = dirName.substr(prefix.size());
        }
        else { continue; }

        // suffix must be exactly the 5-digit, zero-padded checkpoint
        // number checkpointModelName() itself always produces --
        // anything else (wrong width, non-digits) is not actually one
        // of this run's own checkpoints, so is silently skipped rather
        // than treated as an error
        if (suffix.size() != 5) { continue; }
        unsigned long checkpointNum = 0;
        const auto* const suffixEnd = suffix.data() + suffix.size(); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- suffix's own end pointer is always in-bounds
        const auto parseResult = std::from_chars(suffix.data(), suffixEnd, checkpointNum);
        if (parseResult.ec != std::errc{} || parseResult.ptr != suffixEnd) { continue; }

        if (!maxCheckpoint.has_value() || checkpointNum > *maxCheckpoint)
        { maxCheckpoint = checkpointNum; }
    }
    return maxCheckpoint;
}

// The trials_completed/restart_uid/max_trial attributes closeOutputFile()
// writes on every checkpoint file, as read back by restartSetup()
struct CheckpointAttrs
{
    unsigned long trialsCompleted_ = 0;
    unsigned long restartUid_ = 0;
    unsigned long maxTrial_ = 0;
};

// Open the HDF5 file at path read-only, read its trials_completed/
// restart_uid/max_trial attributes (see CheckpointAttrs), close it,
// and return them -- or throw std::runtime_error, naming deleteHint
// as what the caller should delete before restarting again, if the
// file cannot be opened or is missing any of the three. Factored out
// of restartSetup() purely to keep that function's own cognitive
// complexity down (it is called once directly, and once per
// thread_NNNN.h5 file when restarting from an unconsolidated,
// per-thread checkpoint); has no other caller.
static auto readCheckpointAttrs(const std::filesystem::path& path,
    const std::string& deleteHint) -> CheckpointAttrs
{
    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5::restartSetup: unable to open " + path.string());
    }
    CheckpointAttrs attrs;
    try
    {
        attrs.trialsCompleted_ = utils::readULongAttr(file, "trials_completed");
        attrs.restartUid_ = utils::readULongAttr(file, "restart_uid");
        attrs.maxTrial_ = utils::readULongAttr(file, "max_trial");
    }
    catch (const std::exception&)
    {
        H5Fclose(file);
        throw std::runtime_error(
            "OutputManagerH5::restartSetup: " + path.string() +
            " has no trials_completed/restart_uid/max_trial attribute "
            "-- it may still have been open when the run being "
            "restarted stopped; if so, delete " + deleteHint +
            " and restart again");
    }
    H5Fclose(file);
    // NOLINTEND(misc-include-cleaner)
    return attrs;
}

// Record value as common's own first-seen value, or, if common
// already holds one, throw std::runtime_error naming attrName if
// value disagrees with it -- used by restartSetup() to check that
// every thread_NNNN.h5 file in a not-yet-consolidated checkpoint
// directory agrees on each of its own trials_completed/restart_uid/
// max_trial attributes (closeOutputFile() always wrote them the same
// value across every thread in the first place, so a mismatch here
// can only mean the on-disk checkpoint itself is corrupt or was
// tampered with). Factored out of restartSetup() purely to keep that
// function's own cognitive complexity down; has no other caller.
static void crossCheckAttr(std::optional<unsigned long>& common, const unsigned long value,
    const std::string& attrName, const std::filesystem::path& dirPath)
{
    if (!common.has_value()) { common = value; return; }
    if (value != *common)
    {
        throw std::runtime_error(
            "OutputManagerH5::restartSetup: thread output files in " +
            dirPath.string() + " disagree on " + attrName + " (" +
            std::to_string(*common) + " vs " + std::to_string(value) + ")");
    }
}

// Create the "spec_neb" 2D dataset in the given spectra group (sized
// nWl, the same wavelength grid as the group's own "spec" dataset), if
// a nebular emission grid was requested -- a no-op otherwise. Factored
// out of openClusterSpectraGroup()/openGalaxySpectraGroup() to keep
// each within its cognitive-complexity budget.
static void createSpecNebDataset(const io::SimControls& simControls, const hid_t group, const hsize_t nWl)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t dset = utils::createExtensible2dDataset(group, "spec_neb", H5T_NATIVE_DOUBLE, nWl);
    utils::writeStringAttr(dset, "units", "erg/(s Angstrom)");
    H5Dclose(dset);
}

// Create the "spec_neb_extinct" 2D dataset in the given spectra group
// (sized nWlExtinct, the same restricted grid as the group's own
// "spec_extinct" dataset), if a nebular emission grid was requested --
// a no-op otherwise. Must only be called when SimControls::extinct()
// is already known to be non-null (specNebExtinct() has no meaning
// without an extinction curve). Factored out of
// openClusterSpectraGroup()/openGalaxySpectraGroup() to keep each
// within its cognitive-complexity budget.
static void createSpecNebExtinctDataset(const io::SimControls& simControls, const hid_t group,
    const hsize_t nWlExtinct)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t dset = utils::createExtensible2dDataset(group, "spec_neb_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
    utils::writeStringAttr(dset, "units", "erg/(s Angstrom)");
    H5Dclose(dset);
}

// Create the "phot_neb" and (if extinction was also requested)
// "phot_neb_extinct" 2D datasets (each sized nRealFilters, excluding
// any appended "Lbol" entry -- see this function's callers' own
// comments) in the given photometry group, if a nebular emission grid
// was requested -- a no-op otherwise. Must only be called when
// SimControls::filters() is already known to be non-null. Factored out
// of openClusterPhotGroup()/openGalaxyPhotGroup() to keep each within
// its cognitive-complexity budget.
static void createPhotNebDatasets(const io::SimControls& simControls, const hid_t group,
    const hsize_t nRealFilters, const std::vector<std::string>& realFilterUnits)
{
    if (simControls.nebular() == nullptr) { return; }
    const hid_t photNebDset = utils::createExtensible2dDataset(group, "phot_neb", H5T_NATIVE_DOUBLE, nRealFilters);
    utils::writeStringArrayAttr(photNebDset, "units", realFilterUnits);
    H5Dclose(photNebDset);

    if (simControls.extinct() == nullptr) { return; }
    const hid_t photNebExtinctDset = utils::createExtensible2dDataset(
        group, "phot_neb_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
    utils::writeStringArrayAttr(photNebExtinctDset, "units", realFilterUnits);
    H5Dclose(photNebExtinctDset);
}

// Append one row to the "phot_neb" and (if extinction was also
// requested) "phot_neb_extinct" datasets in the given photometry
// group, if a nebular emission grid was requested -- a no-op
// otherwise. Must only be called when SimControls::filters() is
// already known to be non-null, and from inside the same critical
// section as the caller's other utils::appendRowToDataset2d() calls.
// Factored out of writeClusterPhot()/writeGalaxyPhot() to keep each
// within its cognitive-complexity budget.
static void appendPhotNebRows(const io::SimControls& simControls, const hid_t group,
    const std::vector<double>& photNeb, const std::vector<double>& photNebExtinct)
{
    if (simControls.nebular() == nullptr || simControls.filters() == nullptr) { return; }
    utils::appendRowToDataset2d(group, "phot_neb", H5T_NATIVE_DOUBLE, photNeb.data());

    if (simControls.extinct() == nullptr) { return; }
    utils::appendRowToDataset2d(group, "phot_neb_extinct", H5T_NATIVE_DOUBLE, photNebExtinct.data());
}

// Create the "line_wl"/"line_label" fixed datasets and the
// "neb_lines" (and, if extinction was also requested,
// "neb_lines_extinct") extensible 2D datasets in the given spectra
// group, if a nebular emission grid was requested -- a no-op
// otherwise. line_label is written as a fixed-length string dataset
// (see utils::readStringDataset1D's own comment), sized to the
// longest of SimControls::nebular()'s own lineLabel() strings plus a
// null terminator -- computed fresh here each time, since Nebular
// itself does not retain whatever fixed length its own source table
// used. Factored out of openClusterSpectraGroup()/
// openGalaxySpectraGroup() to keep each within its cognitive-
// complexity budget.
static void createNebLinesDatasets(const io::SimControls& simControls, const hid_t group)
{
    const auto* neb = simControls.nebular();
    if (neb == nullptr) { return; }

    const auto& lineWl = neb->lineWl();
    const auto& lineLabel = neb->lineLabel();
    const auto nLines = static_cast<hsize_t>(lineWl.size());

    utils::writeFixed1dDataset(group, "line_wl", H5T_NATIVE_DOUBLE, lineWl.data(), nLines, "Angstrom");

    std::size_t maxLabelLen = 0;
    for (const auto& label : lineLabel) { maxLabelLen = std::max(maxLabelLen, label.size()); }
    const std::size_t labelStrSize = maxLabelLen + 1;
    std::vector<char> labelBuf(lineLabel.size() * labelStrSize, '\0');
    for (std::size_t ell = 0; ell < lineLabel.size(); ++ell)
    {
        std::ranges::copy(lineLabel.at(ell), labelBuf.begin() + static_cast<std::ptrdiff_t>(ell * labelStrSize));
    }
    const hid_t labelType = utils::fixedStrType(labelStrSize);
    utils::writeFixed1dDataset(group, "line_label", labelType, labelBuf.data(), nLines, "");
    H5Tclose(labelType);

    const hid_t linesDset = utils::createExtensible2dDataset(group, "neb_lines", H5T_NATIVE_DOUBLE, nLines);
    utils::writeStringAttr(linesDset, "units", "erg/s");
    H5Dclose(linesDset);

    if (simControls.extinct() == nullptr) { return; }
    const hid_t linesExtinctDset = utils::createExtensible2dDataset(group, "neb_lines_extinct", H5T_NATIVE_DOUBLE, nLines);
    utils::writeStringAttr(linesExtinctDset, "units", "erg/s");
    H5Dclose(linesExtinctDset);
}

// Append one row to the "neb_lines" and (if extinction was also
// requested) "neb_lines_extinct" datasets in the given spectra group,
// if a nebular emission grid was requested -- a no-op otherwise. Must
// be called from inside the same critical section as the caller's
// other utils::appendRowToDataset2d() calls. Factored out of
// writeClusterSpec()/writeGalaxySpec() to keep each within its
// cognitive-complexity budget.
static void appendNebLinesRow(const io::SimControls& simControls, const hid_t group,
    const std::vector<double>& lineLum, const std::vector<double>& lineLumExtinct)
{
    if (simControls.nebular() == nullptr) { return; }
    utils::appendRowToDataset2d(group, "neb_lines", H5T_NATIVE_DOUBLE, lineLum.data());

    if (simControls.extinct() == nullptr) { return; }
    utils::appendRowToDataset2d(group, "neb_lines_extinct", H5T_NATIVE_DOUBLE, lineLumExtinct.data());
}

// NOLINTEND(misc-include-cleaner)

// H5 constructor: see this class's own header comment for what "the
// output file(s)" means here. Without OpenMP, this opens a single
// outDir/modelName.h5, exactly as before. With OpenMP, each thread
// instead gets its own private outDir/modelName/thread_NNNN.h5 --
// eliminating any concurrent access to a single HDF5 file, which HDF5
// does not safely support unless built with its own (opt-in)
// thread-safety support (see this class's own header comment) -- and
// the destructor consolidates them back into a single
// outDir/modelName.h5 once the run completes, unless
// SimControls::outputMode() is OutputMode::h5divided. See
// openNewOutputFiles()'s own comment for the full detail of what
// happens below.
io::OutputManagerH5::OutputManagerH5(
    const SimControls& simControls,
    const toml::table& inputDeck, const bool restart) :
    OutputManager(simControls, inputDeck)
{
    // See this constructor's own header comment for why this is
    // unreachable from the CLI in practice, but still worth guarding
    // against directly
    if (restart && simControls_.outputMode() == SimControls::OutputMode::ascii)
    {
        throw std::runtime_error(
            "OutputManagerH5: restart is true, but simControls.outputMode() "
            "is OutputMode::ascii -- restarting is only supported with "
            "HDF5 output");
    }
    // See this constructor's own header comment for why this
    // combination is rejected rather than left to silently write
    // output slug_reader's own checkpoint discovery would then ignore
    if (restart && simControls_.checkpointInterval() == 0)
    {
        throw std::runtime_error(
            "OutputManagerH5: restart is true, but "
            "simControls.checkpointInterval() is 0 -- restarting "
            "requires checkpointing to remain enabled for this session "
            "too (set output.checkpoint_interval, or call "
            "setCheckpointInterval(), to a non-zero value)");
    }
    if (restart)
    {
        restartSetup();
        // See maxTrial_'s own comment for why this needs to start from
        // restartMaxTrial_ (rather than its own default of 0), even
        // though every write* method also keeps it up to date from
        // here on: this session might end up writing nothing at all
        // before its own checkpoint closes (e.g. an immediate SIGTERM),
        // and that checkpoint's own "max_trial" attribute still needs
        // to correctly reflect every trial number ever written by any
        // earlier session, not just this one.
        maxTrial_ = restartMaxTrial_;
    }
    openNewOutputFiles();
}

// See this method's own header comment for the full design
void io::OutputManagerH5::restartSetup()
{
    const std::string prefix = simControls_.modelName() + "_chk";
    const auto maxCheckpoint = findMaxCheckpointNumber(simControls_.outDir(), prefix);

    if (!maxCheckpoint.has_value())
    {
        if (simControls_.verbosity() > 0)
        {
            std::cout << "slug: restart requested, but no checkpoint found in "
                << simControls_.outDir() << " -- starting from trial 0\n";
        }
        return;
    }

    checkpointNumber_ = *maxCheckpoint + 1;
    const std::string name = checkpointModelName(*maxCheckpoint);
    const auto h5Path = std::filesystem::path(simControls_.outDir()) / (name + ".h5");
    const auto dirPath = std::filesystem::path(simControls_.outDir()) / name;

    unsigned long restartUid = 0;
    std::string source;
    if (std::filesystem::is_regular_file(h5Path))
    {
        const auto attrs = readCheckpointAttrs(h5Path,
            "it (and its own thread_NNNN.h5 files, if it is a directory "
            "rather than a single file)");
        restartTrialsDone_ = attrs.trialsCompleted_;
        restartUid = attrs.restartUid_;
        restartMaxTrial_ = attrs.maxTrial_;
        source = h5Path.string();
    }
    else if (std::filesystem::is_directory(dirPath))
    {
        std::vector<std::filesystem::path> threadFiles;
        for (const auto& entry : std::filesystem::directory_iterator(dirPath))
        {
            if (!entry.is_regular_file()) { continue; }
            const auto& fname = entry.path().filename().string();
            if (fname.starts_with("thread_") && entry.path().extension() == ".h5")
            { threadFiles.push_back(entry.path()); }
        }
        if (threadFiles.empty())
        {
            throw std::runtime_error(
                "OutputManagerH5::restartSetup: no thread_*.h5 files found in " +
                dirPath.string());
        }

        std::optional<unsigned long> commonTrialsCompleted;
        std::optional<unsigned long> commonRestartUid;
        std::optional<unsigned long> commonMaxTrial;
        for (const auto& threadFile : threadFiles)
        {
            const auto attrs = readCheckpointAttrs(threadFile,
                dirPath.string() + " (its own directory)");
            crossCheckAttr(commonTrialsCompleted, attrs.trialsCompleted_,
                "trials_completed", dirPath);
            crossCheckAttr(commonRestartUid, attrs.restartUid_, "restart_uid", dirPath);
            crossCheckAttr(commonMaxTrial, attrs.maxTrial_, "max_trial", dirPath);
        }
        // threadFiles is non-empty (checked above), and crossCheckAttr()
        // always sets each of these on the loop's first iteration, so
        // this has_value() check can never actually fail -- present
        // anyway so a future change to this function that broke the
        // guarantee would fail loudly here instead of reading garbage.
        if (!commonTrialsCompleted.has_value() || !commonRestartUid.has_value() ||
            !commonMaxTrial.has_value())
        {
            throw std::runtime_error(
                "OutputManagerH5::restartSetup: internal error: no thread "
                "output files were actually read in " + dirPath.string());
        }
        // clang-tidy's bugprone-unchecked-optional-access does not
        // credit the has_value() guard immediately above (tried as a
        // single combined condition, then as three separate
        // single-optional ifs -- neither satisfied it, and the latter
        // pushed this function's own cognitive complexity back over
        // its threshold for no real benefit), so it is suppressed
        // explicitly below rather than fought further; the guard above
        // is the actual safety net.
        restartTrialsDone_ = commonTrialsCompleted.value(); // NOLINT(bugprone-unchecked-optional-access)
        restartUid = commonRestartUid.value(); // NOLINT(bugprone-unchecked-optional-access)
        restartMaxTrial_ = commonMaxTrial.value(); // NOLINT(bugprone-unchecked-optional-access)
        source = dirPath.string();
    }
    else
    {
        throw std::runtime_error(
            "OutputManagerH5::restartSetup: found checkpoint number " +
            std::to_string(*maxCheckpoint) + ", but neither " + h5Path.string() +
            " nor " + dirPath.string() + " exists");
    }

    // Resume ID generation from exactly where the run being restarted
    // left off -- see closeOutputFile()'s own comment for why this is
    // the correct value to resume from, rather than either 0 (this
    // process's own uniqueID() starting fresh) or leaving it alone
    utils::uniqueID().set(restartUid);

    if (simControls_.verbosity() > 0)
    {
        std::cout << "slug: restarting from checkpoint " << source << ", "
            << restartTrialsDone_ << " trials completed, highest trial "
            "number written " << restartMaxTrial_ << ", next uid " <<
            restartUid << "\n";
    }
}

// See this method's own header comment for the full design, in
// particular the guarantees it relies on its caller to have already
// established before calling this
void io::OutputManagerH5::checkpoint(const unsigned long trialsCompleted)
{
#ifdef _OPENMP
#pragma omp parallel
    {
        closeOutputFile(trialsCompleted);
    }
#else
    closeOutputFile(trialsCompleted);
#endif
    ++checkpointNumber_;
    openNewOutputFiles();
}

// See this method's own header comment
auto io::OutputManagerH5::checkpointModelName(const unsigned long checkpointNum) const -> std::string
{
    std::ostringstream name;
    name << simControls_.modelName() << "_chk" <<
        std::setfill('0') << std::setw(5) << checkpointNum;
    return name.str();
}

// See this method's own header comment for the full design
void io::OutputManagerH5::openNewOutputFiles()
{
    const std::string modelName = (simControls_.checkpointInterval() != 0) ?
        checkpointModelName(checkpointNumber_) : simControls_.modelName();

#ifdef _OPENMP
    const auto threadDir = std::filesystem::path(simControls_.outDir()) / modelName;
    const auto finalPath = std::filesystem::path(simControls_.outDir()) / (modelName + ".h5");
    if (std::filesystem::exists(threadDir) || std::filesystem::exists(finalPath))
    {
        throw std::runtime_error(
            "OutputManagerH5: output " + threadDir.string() + " or " +
            finalPath.string() + " already exists");
    }
    std::filesystem::create_directory(threadDir);

#pragma omp parallel
    {
        std::ostringstream threadFile;
        threadFile << "thread_" << std::setfill('0') << std::setw(4) <<
            omp_get_thread_num() << ".h5";
        openOutputFile(threadDir / threadFile.str());
    }
#else
    const auto finalPath = std::filesystem::path(simControls_.outDir()) / (modelName + ".h5");
    if (std::filesystem::exists(finalPath))
    {
        throw std::runtime_error(
            "OutputManagerH5: output file " + finalPath.string() + " already exists");
    }
    openOutputFile(finalPath);
#endif
}

// Create one HDF5 file at path, write its header (slug-hash, date,
// time, rng_state) as top-level attributes, then dump the toml input
// deck into an input_deck group, and finally create whichever of the
// clusters/cluster_spectra/cluster_phot/galaxy/galaxy_spectra/
// galaxy_phot groups are enabled -- see this method's own header
// comment
void io::OutputManagerH5::openOutputFile(const std::filesystem::path& path)
{
    // Reset this thread's own handles to -1 (the "not open" sentinel
    // every hid_t handle in this class starts from) before opening
    // its file: openClustersGroup()/etc. below each leave their own
    // handle untouched, rather than explicitly setting it to -1, when
    // the corresponding output.write_* flag is false, relying on it
    // already being -1 here -- see openNewOutputFiles()'s own comment
    // for why this happens per-thread, right here, rather than as a
    // separate, whole-ThreadVec reset up front the way earlier,
    // pre-checkpointing code did.
    file_() = -1;
    clustersGroup_() = -1;
    clusterSpectraGroup_() = -1;
    clusterPhotGroup_() = -1;
    galaxyGroup_() = -1;
    galaxySpectraGroup_() = -1;
    galaxyPhotGroup_() = -1;

    // The HDF5 build linked here is not configured with its own
    // (opt-in) thread-safety support (see this class's own header
    // comment), which turns out to mean it is not safe to call from
    // two threads concurrently AT ALL -- even when each thread only
    // ever touches its own, entirely separate file. A per-file
    // critical section (as every write* method below once used) does
    // not protect against this: two threads each creating their own
    // file can still race on the library's own internal global state
    // and crash. So every call into HDF5 anywhere in this class,
    // across every thread, shares this single, global critical
    // section, regardless of which file/object it targets.
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        file_() = H5Fcreate(path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        if (file_() < 0)
        {
            throw std::runtime_error(
                "OutputManagerH5: unable to create output file " + path.string());
        }

        const auto [date, time] = currentDateAndTime();
        utils::writeStringAttr(file_(), "slug-hash", slugGitHash);
        utils::writeStringAttr(file_(), "date", date);
        utils::writeStringAttr(file_(), "time", time);
        utils::writeStringAttr(file_(), "rng_state", currentRngStateString());

        const hid_t inputDeckGrp = H5Gcreate2(file_(), "input_deck",
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (inputDeckGrp < 0)
        {
            H5Fclose(file_());
            throw std::runtime_error(
                "OutputManagerH5: unable to create input_deck group");
        }
        std::ostringstream tomlStream;
        tomlStream << inputDeck_;
        utils::writeStringDataset(inputDeckGrp, "toml", tomlStream.str());
        H5Gclose(inputDeckGrp);
        // NOLINTEND(misc-include-cleaner)

        openClustersGroup();
        openClusterSpectraGroup();
        openClusterPhotGroup();
        openGalaxyGroup();
        openGalaxySpectraGroup();
        openGalaxyPhotGroup();
    }
}

// Create the clusters group and its datasets, if output.write_cluster
// (optional, defaults to true) was not set to false
void io::OutputManagerH5::openClustersGroup()
{
    if (!writeCluster_) { return; }

    // NOLINTBEGIN(misc-include-cleaner)
    clustersGroup_() = H5Gcreate2(file_(), "clusters",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clustersGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create clusters group");
    }

    const hid_t trialDset = utils::createExtensible1dDataset(
        clustersGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t uidDset = utils::createExtensible1dDataset(
        clustersGroup_(), "uid", H5T_NATIVE_ULONG);
    utils::writeStringAttr(uidDset, "units", "");
    H5Dclose(uidDset);
    const hid_t targetMassDset = utils::createExtensible1dDataset(
        clustersGroup_(), "target_mass", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t birthMassDset = utils::createExtensible1dDataset(
        clustersGroup_(), "birth_mass", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(birthMassDset, "units", "Msun");
    H5Dclose(birthMassDset);
    const hid_t formTimeDset = utils::createExtensible1dDataset(
        clustersGroup_(), "form_time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(formTimeDset, "units", "yr");
    H5Dclose(formTimeDset);
    const hid_t fehDset = utils::createExtensible1dDataset(
        clustersGroup_(), "feh", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(fehDset, "units", "");
    H5Dclose(fehDset);
    const hid_t rngType = utils::fixedStrType(utils::rngStateWidth);
    const hid_t rngDset = utils::createExtensible1dDataset(
        clustersGroup_(), "rng", rngType);
    utils::writeStringAttr(rngDset, "units", "");
    H5Dclose(rngDset);
    H5Tclose(rngType);

    if (simControls_.extinct() != nullptr)
    {
        const hid_t aVDset = utils::createExtensible1dDataset(
            clustersGroup_(), "A_V", H5T_NATIVE_DOUBLE);
        utils::writeStringAttr(aVDset, "units", "magnitudes");
        H5Dclose(aVDset);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_spectra group and its datasets, if a spectral
// synthesizer was requested for this simulation and output.write_cluster_spec
// (optional, defaults to true) was not set to false -- spectra can be
// wanted only as an intermediate for computing photometry, in which
// case writing them out as well just wastes disk space
void io::OutputManagerH5::openClusterSpectraGroup()
{
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeClusterSpec_) { return; }

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    clusterSpectraGroup_() = H5Gcreate2(file_(), "cluster_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterSpectraGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_spectra group");
    }

    utils::writeFixed1dDataset(clusterSpectraGroup_(), "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = utils::createExtensible1dDataset(
        clusterSpectraGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = utils::createExtensible1dDataset(
        clusterSpectraGroup_(), "time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t uidSpecDset = utils::createExtensible1dDataset(
        clusterSpectraGroup_(), "uid", H5T_NATIVE_ULONG);
    utils::writeStringAttr(uidSpecDset, "units", "");
    H5Dclose(uidSpecDset);
    const hid_t specDset = utils::createExtensible2dDataset(
        clusterSpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, nWl);
    utils::writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    // specNeb (if requested) lives on the same wl grid as spec -- no
    // separate wavelength dataset is needed
    createSpecNebDataset(simControls_, clusterSpectraGroup_(), nWl);
    createNebLinesDatasets(simControls_, clusterSpectraGroup_());

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        utils::writeFixed1dDataset(clusterSpectraGroup_(), "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = utils::createExtensible2dDataset(
            clusterSpectraGroup_(), "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        utils::writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);

        // specNebExtinct (if requested) lives on the same
        // (extinction-curve-restricted) grid as specExtinct -- reuses
        // wl_extinct
        createSpecNebExtinctDataset(simControls_, clusterSpectraGroup_(), nWlExtinct);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the cluster_phot group and its datasets, if a filter
// collection or the bolometric luminosity was requested for this
// simulation
void io::OutputManagerH5::openClusterPhotGroup()
{
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeClusterPhot_) { return; }

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
    const auto nFilters = static_cast<hsize_t>(filterNames.size());

    // NOLINTBEGIN(misc-include-cleaner)
    clusterPhotGroup_() = H5Gcreate2(file_(), "cluster_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (clusterPhotGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create cluster_phot group");
    }

    utils::writeStringArrayAttr(clusterPhotGroup_(), "filters", filterNames);

    const hid_t trialPhotDset = utils::createExtensible1dDataset(
        clusterPhotGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = utils::createExtensible1dDataset(
        clusterPhotGroup_(), "time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t uidPhotDset = utils::createExtensible1dDataset(
        clusterPhotGroup_(), "uid", H5T_NATIVE_ULONG);
    utils::writeStringAttr(uidPhotDset, "units", "");
    H5Dclose(uidPhotDset);
    const hid_t photDset = utils::createExtensible2dDataset(
        clusterPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit (e.g. a photon-count filter's
    // "photon/s" alongside another filter's magnitude system), so
    // this is a per-column string array -- unlike every other dataset
    // here, whose units are uniform across the whole dataset -- in
    // the same order as the "filters" attribute above
    utils::writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct/phot_neb/phot_neb_extinct only ever cover real
    // filters, never Lbol -- Lbol is a separate bolometric quantity
    // computed directly from the stellar tracks (see
    // Cluster::computeLbol()), not from the (extincted/nebular-
    // inclusive or otherwise) spectrum, so it has no extincted or
    // nebular-inclusive counterpart; sized independently of
    // nFilters/phot above rather than reusing them, since those may
    // include the "Lbol" entry appended earlier in this function
    if (simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());

        if (simControls_.extinct() != nullptr)
        {
            const hid_t photExtinctDset = utils::createExtensible2dDataset(
                clusterPhotGroup_(), "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
            utils::writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
            H5Dclose(photExtinctDset);
        }

        createPhotNebDatasets(simControls_, clusterPhotGroup_(), nRealFilters, realFilterUnits);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy group and its datasets, for a galaxy-type
// simulation. A no-op for a cluster-type simulation, which has no
// Galaxy object at all.
void io::OutputManagerH5::openGalaxyGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (!writeGalaxy_) { return; }

    // NOLINTBEGIN(misc-include-cleaner)
    galaxyGroup_() = H5Gcreate2(file_(), "galaxy",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy group");
    }

    const hid_t trialDset = utils::createExtensible1dDataset(
        galaxyGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialDset, "units", "");
    H5Dclose(trialDset);
    const hid_t timeDset = utils::createExtensible1dDataset(
        galaxyGroup_(), "time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t targetMassDset = utils::createExtensible1dDataset(
        galaxyGroup_(), "target_mass", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(targetMassDset, "units", "Msun");
    H5Dclose(targetMassDset);
    const hid_t actualMassDset = utils::createExtensible1dDataset(
        galaxyGroup_(), "actual_mass", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(actualMassDset, "units", "Msun");
    H5Dclose(actualMassDset);
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy_spectra group and its datasets, for a galaxy-type
// simulation with a spectral synthesizer requested. A no-op for a
// cluster-type simulation, or if no spectral synthesizer was
// requested.
void io::OutputManagerH5::openGalaxySpectraGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.specsyn() == nullptr) { return; }
    if (!writeGalaxySpec_) { return; }

    const auto& synth = *simControls_.specsyn();
    const std::vector<double> wlObs = synth.wlObs();
    const auto nWl = static_cast<hsize_t>(wlObs.size());

    // NOLINTBEGIN(misc-include-cleaner)
    galaxySpectraGroup_() = H5Gcreate2(file_(), "galaxy_spectra",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxySpectraGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_spectra group");
    }

    utils::writeFixed1dDataset(galaxySpectraGroup_(), "wl", H5T_NATIVE_DOUBLE,
        wlObs.data(), nWl, "Angstrom");

    const hid_t trialSpecDset = utils::createExtensible1dDataset(
        galaxySpectraGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialSpecDset, "units", "");
    H5Dclose(trialSpecDset);
    const hid_t timeDset = utils::createExtensible1dDataset(
        galaxySpectraGroup_(), "time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(timeDset, "units", "yr");
    H5Dclose(timeDset);
    const hid_t specDset = utils::createExtensible2dDataset(
        galaxySpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, nWl);
    utils::writeStringAttr(specDset, "units", "erg/(s Angstrom)");
    H5Dclose(specDset);

    // specNeb (if requested) lives on the same wl grid as spec -- see
    // openClusterSpectraGroup()'s own identical comment
    createSpecNebDataset(simControls_, galaxySpectraGroup_(), nWl);
    createNebLinesDatasets(simControls_, galaxySpectraGroup_());

    if (simControls_.extinct() != nullptr)
    {
        const std::vector<double> wlExtinctObs = simControls_.extinct()->wlObs();
        const auto nWlExtinct = static_cast<hsize_t>(wlExtinctObs.size());
        utils::writeFixed1dDataset(galaxySpectraGroup_(), "wl_extinct", H5T_NATIVE_DOUBLE,
            wlExtinctObs.data(), nWlExtinct, "Angstrom");
        const hid_t specExtinctDset = utils::createExtensible2dDataset(
            galaxySpectraGroup_(), "spec_extinct", H5T_NATIVE_DOUBLE, nWlExtinct);
        utils::writeStringAttr(specExtinctDset, "units", "erg/(s Angstrom)");
        H5Dclose(specExtinctDset);

        createSpecNebExtinctDataset(simControls_, galaxySpectraGroup_(), nWlExtinct);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Create the galaxy_phot group and its datasets, for a galaxy-type
// simulation with a filter collection or the bolometric luminosity
// requested -- mirrors openClusterPhotGroup()'s own filter-list
// construction. A no-op for a cluster-type simulation, or if neither
// was requested.
void io::OutputManagerH5::openGalaxyPhotGroup()
{
    if (simControls_.simType() != SimControls::SimType::galaxy) { return; }
    if (simControls_.filters() == nullptr && !simControls_.computeLbol()) { return; }
    if (!writeGalaxyPhot_) { return; }

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
    const auto nFilters = static_cast<hsize_t>(filterNames.size());

    // NOLINTBEGIN(misc-include-cleaner)
    galaxyPhotGroup_() = H5Gcreate2(file_(), "galaxy_phot",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (galaxyPhotGroup_() < 0)
    {
        H5Fclose(file_());
        throw std::runtime_error(
            "OutputManagerH5: unable to create galaxy_phot group");
    }

    utils::writeStringArrayAttr(galaxyPhotGroup_(), "filters", filterNames);

    const hid_t trialPhotDset = utils::createExtensible1dDataset(
        galaxyPhotGroup_(), "trial", H5T_NATIVE_ULONG);
    utils::writeStringAttr(trialPhotDset, "units", "");
    H5Dclose(trialPhotDset);
    const hid_t timePhotDset = utils::createExtensible1dDataset(
        galaxyPhotGroup_(), "time", H5T_NATIVE_DOUBLE);
    utils::writeStringAttr(timePhotDset, "units", "yr");
    H5Dclose(timePhotDset);
    const hid_t photDset = utils::createExtensible2dDataset(
        galaxyPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, nFilters);
    // Each filter can have its own unit, as in openClusterPhotGroup()
    utils::writeStringArrayAttr(photDset, "units", filterUnits);
    H5Dclose(photDset);

    // phot_extinct/phot_neb/phot_neb_extinct only ever cover real
    // filters, never Lbol -- see openClusterPhotGroup()'s own
    // identical comment
    if (simControls_.filters() != nullptr)
    {
        const auto& realFilterNames = simControls_.filters()->filterNames();
        const auto& realFilterUnits = simControls_.filters()->filterUnits();
        const auto nRealFilters = static_cast<hsize_t>(realFilterNames.size());

        if (simControls_.extinct() != nullptr)
        {
            const hid_t photExtinctDset = utils::createExtensible2dDataset(
                galaxyPhotGroup_(), "phot_extinct", H5T_NATIVE_DOUBLE, nRealFilters);
            utils::writeStringArrayAttr(photExtinctDset, "units", realFilterUnits);
            H5Dclose(photExtinctDset);
        }

        createPhotNebDatasets(simControls_, galaxyPhotGroup_(), nRealFilters, realFilterUnits);
    }
    // NOLINTEND(misc-include-cleaner)
}

// Close every thread's own file(s) -- see this class's own header
// comment -- then, with OpenMP, consolidate them back into a single
// outDir/modelName.h5 (or, per checkpoint, outDir/modelName_chkNNNNN.h5
// -- see checkpointModelName()) unless SimControls::outputMode() is
// OutputMode::h5divided, in which case they are left as-is.
// Consolidation happens once per checkpoint (or once, unchecked, if
// checkpointing is disabled), single-threaded, after every thread has
// closed its own file, since it needs every thread_NNNN.h5 file
// closed (and so fully flushed to disk) before reading them back in.
// Every checkpoint from 0 to checkpointNumber_ is consolidated here,
// not just the last: checkpoint() itself deliberately never
// consolidates (see its own comment), so every earlier checkpoint's
// own thread_NNNN.h5 files are still sitting there, unmerged, until
// this runs.
io::OutputManagerH5::~OutputManagerH5()
{
    // Ordinarily, being destroyed at all means every trial in the run
    // must already be complete, so simControls_.nTrial() is the right
    // trials_completed to close out with -- there is no partial-run
    // count to pass on the way checkpoint() has one. The one exception
    // is a run that ended early (a per-trial exception, or a caught
    // SIGTERM -- see SimCluster::run()'s/SimGalaxy::run()'s own
    // comments) and called notifyEarlyTermination() before returning:
    // earlyTerminationTrialsCompleted_ then holds the true count
    // instead, which value_or() prefers here.
    // Neither closeOutputFile() nor consolidateFiles() may be allowed
    // to throw out of a destructor (undefined behavior at best, a
    // std::terminate() at worst if this fires during stack unwinding
    // from some other exception already in flight) -- both are caught
    // here and reported to stderr instead, on a strictly best-effort
    // basis: there is no caller left to hand a thrown exception to by
    // the time a destructor is running.
    const auto trialsCompleted =
        earlyTerminationTrialsCompleted_.value_or(simControls_.nTrial());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        try
        {
            closeOutputFile(trialsCompleted);
        }
        catch (const std::exception& error)
        {
#ifdef _OPENMP
#pragma omp critical
#endif
            std::cerr << "slug: OutputManagerH5: error closing output file: "
                << error.what() << "\n";
        }
    }

#ifdef _OPENMP
    if (simControls_.outputMode() == SimControls::OutputMode::h5)
    {
        if (simControls_.checkpointInterval() != 0)
        {
            for (unsigned long chk = 0; chk <= checkpointNumber_; ++chk)
            {
                const auto chkPath = std::filesystem::path(simControls_.outDir()) /
                    checkpointModelName(chk);

                // A restarted run (see restartSetup()'s own comment)
                // can inherit checkpoint numbers below its own
                // checkpointNumber_ that a *previous* run's own
                // destructor already consolidated (and, per
                // consolidateFiles()'s own comment, already deleted
                // the thread_NNNN.h5 directory for) before this run
                // ever began -- skip those rather than trying to
                // consolidate a directory that no longer exists. Every
                // checkpoint number below checkpointNumber_ that was
                // instead left behind by a run that never reached its
                // own destructor (e.g. an actual crash -- the
                // ordinary case restarting exists for) is still
                // exactly the not-yet-consolidated directory
                // consolidateFiles() expects, so this only ever skips
                // checkpoints that a previous run's own destructor has
                // already fully consolidated.
                if (!std::filesystem::is_directory(chkPath)) { continue; }
                try
                {
                    consolidateFiles(chkPath);
                }
                catch (const std::exception& error)
                {
                    std::cerr << "slug: OutputManagerH5: error consolidating " <<
                        chkPath.string() << ": " << error.what() << "\n";
                }
            }
        }
        else
        {
            try
            {
                consolidateFiles(std::filesystem::path(simControls_.outDir()) /
                    simControls_.modelName());
            }
            catch (const std::exception& error)
            {
                std::cerr << "slug: OutputManagerH5: error consolidating output: "
                    << error.what() << "\n";
            }
        }
    }
#endif
}

// Close whichever of this thread's own groups are open, then its
// file -- see openOutputFile()'s own comment for why this shares its
// critical section
void io::OutputManagerH5::closeOutputFile(const unsigned long trialsCompleted)
{
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        // NOLINTBEGIN(misc-include-cleaner)
        if (clustersGroup_() >= 0) { H5Gclose(clustersGroup_()); }
        if (clusterSpectraGroup_() >= 0) { H5Gclose(clusterSpectraGroup_()); }
        if (clusterPhotGroup_() >= 0) { H5Gclose(clusterPhotGroup_()); }
        if (galaxyGroup_() >= 0) { H5Gclose(galaxyGroup_()); }
        if (galaxySpectraGroup_() >= 0) { H5Gclose(galaxySpectraGroup_()); }
        if (galaxyPhotGroup_() >= 0) { H5Gclose(galaxyPhotGroup_()); }
        utils::writeULongAttr(file_(), "trials_completed", trialsCompleted);
        utils::writeULongAttr(file_(), "restart_uid", utils::uniqueID().read());
        utils::writeULongAttr(file_(), "max_trial", maxTrial_);
        H5Fclose(file_());
        // NOLINTEND(misc-include-cleaner)
    }
}

// See this class's own header comment for the full merge algorithm
void io::OutputManagerH5::consolidateFiles(const std::filesystem::path& path)
{
    if (!std::filesystem::is_directory(path))
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: " + path.string() + " is not a directory");
    }

    std::vector<std::filesystem::path> threadFiles;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (!entry.is_regular_file()) { continue; }
        const auto& name = entry.path().filename().string();
        if (name.starts_with("thread_") && entry.path().extension() == ".h5")
        { threadFiles.push_back(entry.path()); }
    }
    if (threadFiles.empty())
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: no thread_*.h5 files found in " + path.string());
    }
    // Zero-padded thread numbers (see the constructor's own
    // thread_NNNN.h5 naming) sort lexicographically in the same order
    // as numerically, so a plain path sort puts thread_0000.h5 first
    std::ranges::sort(threadFiles);

    const auto destPath = path.string() + ".h5";
    std::filesystem::copy_file(threadFiles.front(), destPath);

    // NOLINTBEGIN(misc-include-cleaner)
    const hid_t destFile = H5Fopen(destPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (destFile < 0)
    {
        throw std::runtime_error(
            "OutputManagerH5::consolidateFiles: unable to open " + destPath + " for updating");
    }

    for (std::size_t i = 1; i < threadFiles.size(); ++i)
    {
        const hid_t srcFile = H5Fopen(threadFiles.at(i).string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (srcFile < 0)
        {
            H5Fclose(destFile);
            throw std::runtime_error(
                "OutputManagerH5::consolidateFiles: unable to open " + threadFiles.at(i).string());
        }
        utils::appendFileContents(srcFile, destFile);
        H5Fclose(srcFile);
    }
    H5Fclose(destFile);
    // NOLINTEND(misc-include-cleaner)

    for (const auto& threadFile : threadFiles) { std::filesystem::remove(threadFile); }
    std::filesystem::remove(path);
}

// Append one element to each of the clusters datasets. A no-op if
// cluster output was not enabled for this simulation. Guards the
// actual writes with the same critical section openOutputFile() uses
// -- see its own comment for why every call into HDF5 needs this,
// not just concurrent calls that happen to target the same file.
void io::OutputManagerH5::writeCluster(
    const unsigned long trial, const core::Cluster& cluster)
{
    if (clustersGroup_() < 0) { return; }

    const unsigned long uid = cluster.uid();
    const double targetMass = cluster.targetMass();
    const double birthMass = cluster.birthMass();
    const double formTime = cluster.formTime();
    const double feH = cluster.feH();
    const auto& rngState = cluster.rngState();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(clustersGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(clustersGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        utils::appendToDataset(clustersGroup_(), "target_mass", H5T_NATIVE_DOUBLE, &targetMass);
        utils::appendToDataset(clustersGroup_(), "birth_mass", H5T_NATIVE_DOUBLE, &birthMass);
        utils::appendToDataset(clustersGroup_(), "form_time", H5T_NATIVE_DOUBLE, &formTime);
        utils::appendToDataset(clustersGroup_(), "feh", H5T_NATIVE_DOUBLE, &feH);
        const hid_t rngType = utils::fixedStrType(utils::rngStateWidth);
        utils::appendToDataset(clustersGroup_(), "rng", rngType, rngState.data());
        H5Tclose(rngType);
        if (simControls_.extinct() != nullptr)
        {
            const double aV = cluster.aV();
            utils::appendToDataset(clustersGroup_(), "A_V", H5T_NATIVE_DOUBLE, &aV);
        }

        // Flush this thread's own file now, rather than waiting for
        // its eventual H5Fclose() (in this class's own destructor):
        // the cluster's own rng state, just written above, is what a
        // later crash (e.g. an uncaught exception from spectral
        // synthesis below, which aborts the whole process before any
        // destructor can run) needs to be recoverable from disk, to
        // deterministically reproduce that exact cluster afterward
        // (see Cluster's own second constructor overload, which
        // replays a cluster's stochastic draws from a saved rng
        // state). Costs extra I/O on every cluster write, deemed
        // worth it for making every future crash like this one
        // investigable from the output file alone.
        H5Fflush(file_(), H5F_SCOPE_LOCAL);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/uid/spec
// cluster-spectra datasets. A no-op if spectral synthesis was not
// enabled for this simulation (the cluster_spectra group does not
// exist), or if the cluster has disrupted -- a disrupted cluster is
// no longer an observable object, though its light still belongs in
// the total galaxy spectrum, which is handled elsewhere.
void io::OutputManagerH5::writeClusterSpec(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (clusterSpectraGroup_() < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    // spec()/specExtinct() are computed here, outside the critical
    // section below, since they may need to lazily (re)compute the
    // cluster's spectrum -- potentially expensive work that must not
    // run while holding the critical section, which would needlessly
    // serialize it across threads (see this class's own header
    // comment on write* taking non-const Cluster&/Galaxy&)
    const unsigned long uid = cluster.uid();
    const auto& spec = cluster.spec();
    const auto& specExtinct = cluster.specExtinct();
    const auto& specNeb = cluster.specNeb();
    const auto& specNebExtinct = cluster.specNebExtinct();
    const auto& lineLum = cluster.lineLum();
    const auto& lineLumExtinct = cluster.lineLumExtinct();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(clusterSpectraGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(clusterSpectraGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        utils::appendToDataset(clusterSpectraGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        utils::appendRowToDataset2d(clusterSpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, spec.data());
        if (simControls_.nebular() != nullptr)
        {
            utils::appendRowToDataset2d(clusterSpectraGroup_(), "spec_neb",
                H5T_NATIVE_DOUBLE, specNeb.data());
        }
        if (simControls_.extinct() != nullptr)
        {
            utils::appendRowToDataset2d(clusterSpectraGroup_(), "spec_extinct",
                H5T_NATIVE_DOUBLE, specExtinct.data());
            if (simControls_.nebular() != nullptr)
            {
                utils::appendRowToDataset2d(clusterSpectraGroup_(), "spec_neb_extinct",
                    H5T_NATIVE_DOUBLE, specNebExtinct.data());
            }
        }
        appendNebLinesRow(simControls_, clusterSpectraGroup_(), lineLum, lineLumExtinct);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/uid/phot cluster_phot
// datasets. A no-op if no filter collection or bolometric luminosity
// was requested for this simulation (the cluster_phot group does not
// exist), or if the cluster has disrupted -- a disrupted cluster is
// no longer an observable object.
void io::OutputManagerH5::writeClusterPhot(
    const unsigned long trial, const double time, core::Cluster& cluster)
{
    if (clusterPhotGroup_() < 0) { return; }
    if (cluster.isDisrupted()) { return; }

    // phot()/photExtinct()/lbol() are computed here, outside the
    // critical section below -- see writeClusterSpec's own comment
    const unsigned long uid = cluster.uid();
    auto phot = cluster.phot();
    if (simControls_.computeLbol()) { phot.push_back(cluster.lbol()); }
    const auto& photExtinct = cluster.photExtinct();
    const auto& photNeb = cluster.photNeb();
    const auto& photNebExtinct = cluster.photNebExtinct();

#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(clusterPhotGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(clusterPhotGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        utils::appendToDataset(clusterPhotGroup_(), "uid", H5T_NATIVE_ULONG, &uid);
        utils::appendRowToDataset2d(clusterPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, phot.data());
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            utils::appendRowToDataset2d(clusterPhotGroup_(), "phot_extinct",
                H5T_NATIVE_DOUBLE, photExtinct.data());
        }
        appendPhotNebRows(simControls_, clusterPhotGroup_(), photNeb, photNebExtinct);
        // NOLINTEND(misc-include-cleaner)
    }
}

// Append one element to each of the trial/time/target_mass/actual_mass
// galaxy datasets, then call writeCluster() on every currently-alive
// (non-disrupted) cluster in galaxy, so each is also recorded in the
// clusters datasets. A no-op if galaxy output was not enabled for this
// simulation (the galaxy group does not exist).
void io::OutputManagerH5::writeGalaxy(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyGroup_() < 0) { return; }

    const double targetMass = galaxy.targetMass();
    const double actualMass = galaxy.actualMass();

    // Released before the writeCluster() calls below, each of which
    // takes this same critical section itself; OpenMP critical
    // regions are not reentrant, so those calls must happen outside
    // this block.
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(galaxyGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(galaxyGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        utils::appendToDataset(galaxyGroup_(), "target_mass", H5T_NATIVE_DOUBLE, &targetMass);
        utils::appendToDataset(galaxyGroup_(), "actual_mass", H5T_NATIVE_DOUBLE, &actualMass);
        // NOLINTEND(misc-include-cleaner)
    }

    for (const auto& cluster : galaxy.clusters()) { writeCluster(trial, cluster); }
}

// Append one element to each of the trial/time/spec galaxy_spectra
// datasets, then call writeClusterSpec() on every currently-alive
// (non-disrupted) cluster in galaxy. A no-op if spectral synthesis
// was not enabled for this simulation (the galaxy_spectra group does
// not exist).
void io::OutputManagerH5::writeGalaxySpec(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxySpectraGroup_() < 0) { return; }

    // spec()/specExtinct() are computed here, outside the critical
    // section below -- see writeClusterSpec's own comment
    const auto& spec = galaxy.spec();
    const auto& specExtinct = galaxy.specExtinct();
    const auto& specNeb = galaxy.specNeb();
    const auto& specNebExtinct = galaxy.specNebExtinct();
    const auto& lineLum = galaxy.lineLum();
    const auto& lineLumExtinct = galaxy.lineLumExtinct();

    // See writeGalaxy's own comment on this critical section
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(galaxySpectraGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(galaxySpectraGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        utils::appendRowToDataset2d(galaxySpectraGroup_(), "spec", H5T_NATIVE_DOUBLE, spec.data());
        if (simControls_.nebular() != nullptr)
        {
            utils::appendRowToDataset2d(galaxySpectraGroup_(), "spec_neb",
                H5T_NATIVE_DOUBLE, specNeb.data());
        }
        if (simControls_.extinct() != nullptr)
        {
            utils::appendRowToDataset2d(galaxySpectraGroup_(), "spec_extinct",
                H5T_NATIVE_DOUBLE, specExtinct.data());
            if (simControls_.nebular() != nullptr)
            {
                utils::appendRowToDataset2d(galaxySpectraGroup_(), "spec_neb_extinct",
                    H5T_NATIVE_DOUBLE, specNebExtinct.data());
            }
        }
        appendNebLinesRow(simControls_, galaxySpectraGroup_(), lineLum, lineLumExtinct);
        // NOLINTEND(misc-include-cleaner)
    }

    for (auto& cluster : galaxy.clusters()) { writeClusterSpec(trial, time, cluster); }
}

// Append one element to each of the trial/time/phot galaxy_phot
// datasets, then call writeClusterPhot() on every currently-alive
// (non-disrupted) cluster in galaxy. A no-op if no filter collection
// or bolometric luminosity was requested for this simulation (the
// galaxy_phot group does not exist).
void io::OutputManagerH5::writeGalaxyPhot(
    const unsigned long trial, const double time, core::Galaxy& galaxy)
{
    if (galaxyPhotGroup_() < 0) { return; }

    // phot()/photExtinct()/lbol() are computed here, outside the
    // critical section below -- see writeClusterSpec's own comment
    auto phot = galaxy.phot();
    if (simControls_.computeLbol()) { phot.push_back(galaxy.lbol()); }
    const auto& photExtinct = galaxy.photExtinct();
    const auto& photNeb = galaxy.photNeb();
    const auto& photNebExtinct = galaxy.photNebExtinct();

    // See writeGalaxy's own comment on this critical section
#ifdef _OPENMP
#pragma omp critical(h5ThreadSafety)
#endif
    {
        if (trial > maxTrial_) { maxTrial_ = trial; }

        // NOLINTBEGIN(misc-include-cleaner)
        utils::appendToDataset(galaxyPhotGroup_(), "trial", H5T_NATIVE_ULONG, &trial);
        utils::appendToDataset(galaxyPhotGroup_(), "time", H5T_NATIVE_DOUBLE, &time);
        utils::appendRowToDataset2d(galaxyPhotGroup_(), "phot", H5T_NATIVE_DOUBLE, phot.data());
        if (simControls_.extinct() != nullptr && simControls_.filters() != nullptr)
        {
            utils::appendRowToDataset2d(galaxyPhotGroup_(), "phot_extinct",
                H5T_NATIVE_DOUBLE, photExtinct.data());
        }
        appendPhotNebRows(simControls_, galaxyPhotGroup_(), photNeb, photNebExtinct);
        // NOLINTEND(misc-include-cleaner)
    }

    for (auto& cluster : galaxy.clusters()) { writeClusterPhot(trial, time, cluster); }
}
