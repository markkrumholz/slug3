/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow and physics settings
 * @date 16-07-2026
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../extinct/Extinct.hpp"
#include "../nebular/Nebular.hpp"
#include "../nebular/NebularCommons.hpp"
#include "../pdfs/PDF.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/ParseUtils.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <toml.hpp>
#include <utility>
#include <vector>

namespace io
{
    /**
     * @brief Prefix under which to resolve an IMF PDF file
     * @details
     * Shared by SimControls's constructor (reading stars.IMF via
     * utils::initPDFFromKey()) and SimControls::setIMF() (via
     * utils::initPDFFromString()), so both resolve an IMF file name
     * identically.
     */
    inline const std::string imfPrefix = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("imfs")).string();

    /**
     * @class SimControls
     * @brief A class to hold simulation control flow information and physics settings
     * @details
     * Originally two separate classes -- SimControls (control flow:
     * IO, trial count, output timing) and SimPhysics (physics
     * choices: IMF, tracks, spectral synthesis) -- merged into one,
     * since in practice almost every consumer (Cluster, SimCluster,
     * OutputManager, every Specsyn-derived constructor) needed both
     * together anyway, and keeping them separate produced duplicated
     * state (e.g. the integrator tolerances below, which used to be
     * copied into Specsyn and Cluster independently at construction
     * time). Every object built by this class that itself needs the
     * integrator tolerances -- the spectral synthesizer built by
     * readSpectra() -- holds a live reference back to this object
     * (see Specsyn's own controls_ member) rather than a snapshot, so
     * this object is neither copyable nor movable: its address must
     * stay fixed for the lifetime of anything holding such a
     * reference.
     */
    class SimControls
    {
    public:

        /**
         * @brief An enum to hold output modes
         */
        enum class OutputMode : std::uint8_t {
            h5,         /**< HDF5 output, consolidated into a single file (see OutputManagerH5's own header comment on what this means when built with OpenMP) */
            h5divided,  /**< Same as h5, but skips the final consolidation step when built with OpenMP, leaving one HDF5 file per thread -- preferable for very large runs where consolidating into a single file would itself be costly */
            ascii       /**< ASCII output */
        };

        /**
         * @brief An enum to hold simulation types
         */
        enum class SimType : std::uint8_t {
            cluster, /**< Cluster simulation */
            galaxy,  /**< Galaxy simulation */
            none     /**< Dummy value */
        };

        /**
         * @brief Default-construct a SimControls with all defaults
         * @details
         * Produces a SimControls with every control-flow parameter at
         * its default value (simType = none, verbosity = 0, nTrial =
         * 1, etc.) and every physics setting left invalid/empty (no
         * IMF, no tracks, no spectral synthesizer, ...). Intended for
         * use where only the control-flow defaults (in particular the
         * integrator tolerances) matter, not the physics settings --
         * see io::defaultControls() in the Python bindings for the
         * shared instance used for exactly this purpose.
         */
        SimControls() = default;

        /**
         * @brief Initialize the simulation controls and physics settings from the input deck
         * @param inputDeck A toml table holding the input deck
         */
        explicit SimControls(const toml::table& inputDeck);

        // Non-copyable and non-movable: tracks_ and constFeHTracks_
        // are themselves non-copyable (see Tracks3D/Tracks2D), and
        // specsyn_/filters_ are held by unique_ptr, so this class was
        // already implicitly non-copyable. It is additionally made
        // non-movable here, since the spectral synthesizer readSpectra()
        // builds, and the nebular emission grid readNebular() builds,
        // each store a live reference back to *this (see Specsyn's own
        // controls_ member, and Nebular's own simControls_ member) --
        // this object's address must never change after construction
        // for those references to stay valid.
        SimControls(const SimControls&) = delete;
        auto operator=(const SimControls&) -> SimControls& = delete;
        SimControls(SimControls&&) = delete;
        auto operator=(SimControls&&) -> SimControls& = delete;
        ~SimControls() = default;

        // Getters for control flow

        /**
         * @brief Return simulation type
         * @return Simulation type
         */
        [[nodiscard]] auto simType() const { return simType_; }

        /**
         * @brief Return output mode
         * @return Output mode
         */
        [[nodiscard]] auto outputMode() const { return outputMode_; }

        /**
         * @brief Return model name
         * @return Model name
         */
        [[nodiscard]] auto modelName() const { return modelName_; }

        /**
         * @brief Return output directory
         * @return Output directory
         * @details
         * This is the directory into which output will be written.
         * An empty string (the default) means output will be written
         * into the current working directory.
         */
        [[nodiscard]] auto outDir() const { return outDir_; }

        /**
         * @brief Return verbosity level
         * @return Verbosity level
         */
        [[nodiscard]] auto verbosity() const { return verbosity_; }

         /**
         * @brief Return number of trials in the simulation
         * @return Number of trials
         */
        [[nodiscard]] auto nTrial() const { return nTrial_; }

        /**
         * @brief Return the times at which output should occur
         * @return A vector of output times
         * @details
         * If explicit output times were specified in the input deck
         * (either directly, or as a uniformly- or log-spaced grid),
         * returns a copy of that array. If instead a distribution of
         * output times was specified, returns a single-element vector
         * containing one time drawn from that distribution.
         */
        [[nodiscard]] auto outTimes() const -> std::vector<double>
        {
            if (!outTimes_.empty()) { return outTimes_; }
            return { outTimeDist_.draw() };
        }

        /**
         * @brief Return the relative tolerance for PDF integration
         * @return Relative tolerance passed to PDFIntegrator (default 1e-2)
         */
        [[nodiscard]] auto intRelTol() const { return intRelTol_; }

        /**
         * @brief Return the absolute tolerance for PDF integration
         * @return Absolute tolerance passed to PDFIntegrator (default 1e-3)
         */
        [[nodiscard]] auto intAbsTol() const { return intAbsTol_; }

        /**
         * @brief Return the maximum number of bisection iterations for PDF integration
         * @return Max bisection iterations passed to PDFIntegrator (0 =
         *   unlimited; default 2^19 -- see intMaxIter_'s own comment
         *   for why); not a count of raw integrand evaluations, which
         *   each iteration costs several of -- see GKIntegrator::
         *   integrate()'s own comment
         */
        [[nodiscard]] auto intMaxIter() const { return intMaxIter_; }

        /**
         * @brief Return the redshift
         * @return Redshift applied by every Specsyn's and Extinct's
         *   own wlObs() (default 0, i.e. no redshift)
         */
        [[nodiscard]] auto z() const { return z_; }

        /**
         * @brief Return the checkpoint interval, in trials
         * @return The number of trials between checkpoints; 0 (the
         *   default) means checkpointing is disabled. See
         *   OutputManagerH5::checkpoint()'s own comment for what a
         *   non-zero value actually does.
         */
        [[nodiscard]] auto checkpointInterval() const { return checkpointInterval_; }

        /**
         * @brief Return the input deck's own text
         * @return The toml table this SimControls was constructed
         *   from, re-serialized back to text -- not necessarily
         *   byte-identical to whatever text/file originally produced
         *   that table (toml++'s own operator<<, not the original
         *   source, does the formatting), but parses back to an
         *   equivalent table. Empty if this SimControls was built by
         *   the default constructor, with no input deck at all.
         * @details
         * Read by OutputManagerH5/OutputManagerAscii to record the
         * deck verbatim into the output file for provenance, so that a
         * SimControls built without a backing input deck at all (e.g.
         * from Python) still gives them something equivalent to record
         * -- unlike the six output.write_* flags (see writeCluster()
         * and its own siblings, just below), computed and cached once,
         * at construction, this has no corresponding setter: unlike
         * those flags, there is no legitimate reason for a caller to
         * override what deck actually built this SimControls.
         */
        [[nodiscard]] auto inputDeckStr() const -> const std::string& { return inputDeckStr_; }

        /**
         * @brief Whether the clusters group/file should be written
         * @return True (the default) unless output.write_cluster was
         *   set to false in the input deck; see OutputManager's own
         *   constructor for how this is enforced
         */
        [[nodiscard]] auto writeCluster() const { return writeCluster_; }

        /**
         * @brief Whether the cluster_spectra group/file should be written
         * @return True (the default) unless output.write_cluster_spec
         *   was set to false in the input deck; see OutputManager's
         *   own constructor for how this is enforced
         */
        [[nodiscard]] auto writeClusterSpec() const { return writeClusterSpec_; }

        /**
         * @brief Whether the cluster_phot group/file should be written
         * @return True (the default) unless output.write_cluster_phot
         *   was set to false in the input deck; see OutputManager's
         *   own constructor for how this is enforced
         */
        [[nodiscard]] auto writeClusterPhot() const { return writeClusterPhot_; }

        /**
         * @brief Whether the galaxy group/file should be written
         * @return True (the default) unless output.write_galaxy was
         *   set to false in the input deck; only meaningful for a
         *   galaxy-type simulation -- see OutputManager's own
         *   constructor for how this is enforced
         */
        [[nodiscard]] auto writeGalaxy() const { return writeGalaxy_; }

        /**
         * @brief Whether the galaxy_spectra group/file should be written
         * @return True (the default) unless output.write_galaxy_spec
         *   was set to false in the input deck; only meaningful for a
         *   galaxy-type simulation -- see OutputManager's own
         *   constructor for how this is enforced
         */
        [[nodiscard]] auto writeGalaxySpec() const { return writeGalaxySpec_; }

        /**
         * @brief Whether the galaxy_phot group/file should be written
         * @return True (the default) unless output.write_galaxy_phot
         *   was set to false in the input deck; only meaningful for a
         *   galaxy-type simulation -- see OutputManager's own
         *   constructor for how this is enforced
         */
        [[nodiscard]] auto writeGalaxyPhot() const { return writeGalaxyPhot_; }

        /**
         * @brief Set the relative tolerance for PDF integration
         * @param tol New relative tolerance
         * @details
         * Any spectral synthesizer built by this SimControls (see
         * specsyn()) reads this value live, not a snapshot, so the
         * new tolerance takes effect the next time it integrates --
         * no need to rebuild the synthesizer or any Cluster built
         * from this SimControls.
         */
        void setIntRelTol(double tol) { intRelTol_ = tol; }

        /**
         * @brief Set the absolute tolerance for PDF integration
         * @param tol New absolute tolerance
         * @details
         * See setIntRelTol()'s own comment on live (not snapshotted) effect.
         */
        void setIntAbsTol(double tol) { intAbsTol_ = tol; }

        /**
         * @brief Set the maximum number of bisection iterations for PDF integration
         * @param n New maximum (0 = unlimited); not a count of raw
         *   integrand evaluations -- see intMaxIter_'s own comment
         * @details
         * See setIntRelTol()'s own comment on live (not snapshotted) effect.
         */
        void setIntMaxIter(std::size_t n) { intMaxIter_ = n; }

        /**
         * @brief Set the redshift
         * @param z New redshift
         * @details
         * Every Specsyn/Extinct built by this SimControls reads this
         * value live, not a snapshot (see z()), so the new redshift
         * takes effect the next time their own wlObs() is called --
         * no need to rebuild them.
         */
        void setZ(double z) { z_ = z; }

        /**
         * @brief Set the checkpoint interval, in trials
         * @param interval New checkpoint interval; 0 disables
         *   checkpointing
         * @details
         * Unlike setting output.checkpoint_interval in an input deck,
         * this plain setter does not check outputMode() -- a caller
         * (e.g. from Python) could use it to set up the same illegal
         * combination (a non-zero interval with ascii output) the
         * constructor rejects when parsing a real input deck. That
         * combination is instead caught defensively where it would
         * actually matter, by OutputManagerAscii::checkpoint()
         * throwing (see its own comment), rather than here.
         */
        void setCheckpointInterval(unsigned long interval) { checkpointInterval_ = interval; }

        /**
         * @brief Set whether the clusters group/file should be written
         * @param value New value for writeCluster()
         * @details
         * Lets a caller (e.g. from Python, where there is no
         * output.write_cluster input-deck entry to set this through)
         * enable or suppress cluster output on an already-constructed
         * SimControls. Unlike setIntRelTol()/setZ()/etc., this is not
         * read live by an already-built OutputManagerH5/
         * OutputManagerAscii: each one decides once, at its own
         * construction, which groups/files to create at all (see
         * OutputManager's own constructor), so this setter only
         * affects an OutputManagerH5/OutputManagerAscii built from
         * this SimControls afterward, not one already built from it.
         */
        void setWriteCluster(bool value) { writeCluster_ = value; }

        /**
         * @brief Set whether the cluster_spectra group/file should be written
         * @param value New value for writeClusterSpec()
         * @details
         * See setWriteCluster()'s own comment on when this does (and
         * does not) take effect.
         */
        void setWriteClusterSpec(bool value) { writeClusterSpec_ = value; }

        /**
         * @brief Set whether the cluster_phot group/file should be written
         * @param value New value for writeClusterPhot()
         * @details
         * See setWriteCluster()'s own comment on when this does (and
         * does not) take effect.
         */
        void setWriteClusterPhot(bool value) { writeClusterPhot_ = value; }

        /**
         * @brief Set whether the galaxy group/file should be written
         * @param value New value for writeGalaxy()
         * @details
         * Only meaningful for a galaxy-type simulation. See
         * setWriteCluster()'s own comment on when this does (and does
         * not) take effect.
         */
        void setWriteGalaxy(bool value) { writeGalaxy_ = value; }

        /**
         * @brief Set whether the galaxy_spectra group/file should be written
         * @param value New value for writeGalaxySpec()
         * @details
         * Only meaningful for a galaxy-type simulation. See
         * setWriteCluster()'s own comment on when this does (and does
         * not) take effect.
         */
        void setWriteGalaxySpec(bool value) { writeGalaxySpec_ = value; }

        /**
         * @brief Set whether the galaxy_phot group/file should be written
         * @param value New value for writeGalaxyPhot()
         * @details
         * Only meaningful for a galaxy-type simulation. See
         * setWriteCluster()'s own comment on when this does (and does
         * not) take effect.
         */
        void setWriteGalaxyPhot(bool value) { writeGalaxyPhot_ = value; }

        // Getters for the physics settings
        /**
         * @brief Get simulation initial mass function
         * @return Pointer to the simulation initial mass function
         */
        [[nodiscard]] auto imf() const -> const auto& { return imf_; }

        /**
         * @brief Get simulation cluster mass function
         * @return Pointer to the simulation cluster mass function
         */
        [[nodiscard]] auto cmf() const -> const auto& { return cmf_; }

        /**
         * @brief Get simulation [Fe/H] distribution
         * @return Pointer to the simulation [Fe/H] distribution
         */
        [[nodiscard]] auto fehDist() const -> const auto& { return fehDist_; }

        /**
         * @brief Check whether the simulation has a spread in [Fe/H]
         * @return True if the simulation has a fixed value of [Fe/H]
         */
        [[nodiscard]] auto constFeH() const { return fehDist_.getMin() == fehDist_.getMax(); }

        /**
         * @brief Get simulation star formation rate
         * @return A const reference to the star formation rate, in
         *   Msun/yr as a function of time -- see buildConstantSFR()'s
         *   own comment for the exact meaning; invalid/empty if
         *   galaxy.sfr_dist was given instead of galaxy.sfr (see
         *   sfrDist()'s own comment)
         */
        [[nodiscard]] auto sfr() const -> const auto& { return sfr_; }

        /**
         * @brief Get the distribution from which a single, constant star formation rate is drawn
         * @return A const reference to the distribution requested via
         *   galaxy.sfr_dist; invalid/empty if galaxy.sfr was given
         *   instead (see sfr()'s own comment)
         * @details
         * Unlike sfr() itself (a rate as a function of time, in
         * general), this is a distribution over a single, scalar rate
         * value -- exactly one of galaxy.sfr/galaxy.sfr_dist is
         * required for a galaxy-type simulation (see the constructor's
         * own comment, in the .cpp file, for the exact rule). When
         * sfr_dist is the one given, no
         * SimControls-level rate exists at all: each Galaxy instead
         * draws its own, independent constant rate from this
         * distribution at construction, via buildConstantSFR() -- see
         * Galaxy::Galaxy()'s own comment.
         */
        [[nodiscard]] auto sfrDist() const -> const auto& { return sfrDist_; }

        /**
         * @brief Get simulation cluster lifetime function
         * @return Pointer to the simulation cluster lifetime function
         */
        [[nodiscard]] auto clf() const -> const auto& { return clf_; }

        /**
         * @brief Get the fraction of stellar mass formed in stochastically-treated clusters
         * @return The fraction of a galaxy simulation's stellar mass
         *   that forms in clusters treated stochastically (as
         *   individual Cluster objects); the remaining (1 - fCluster())
         *   is treated as a stellar population continuous in both mass
         *   and time. Defaults to 1.0 (every star forms in a
         *   stochastic cluster); only meaningful for, and only read
         *   from, a galaxy-type simulation's own clusters.f_cluster.
         */
        [[nodiscard]] auto fCluster() const { return fCluster_; }

        /**
         * @brief Get simulation stellar tracks
         * @return Pointer to the simulation stellar tracks
         */
        [[nodiscard]] auto tracks() const -> const auto& { return tracks_; }

        /**
         * @brief Get the tracks sliced to this simulation's fixed [Fe/H]
         * @return A const reference to a Tracks2D object sliced at
         *         fehDist_'s (single) value
         * @details
         * Only valid to call if constFeH() is true. This slice is
         * computed once, in the constructor, and cached for the
         * lifetime of this SimControls object, so that Cluster objects
         * sharing a single [Fe/H] value across a simulation can all
         * query it directly instead of each computing (or racing to
         * compute) their own copy.
         */
        [[nodiscard]] auto tracks2D() const -> const auto& { return constFeHTracks_; }

        /**
         * @brief Get minimum mass for fully stochastic treatment
         * @return Minimum mass for fully stochastic treatment
         */
        [[nodiscard]] auto minStochMass() const { return minStochMass_; }

        /**
         * @brief Get fraction of stellar mass being treated stochastically
         * @return Fraction of stellar mass being treated stochastically
         */
        [[nodiscard]] auto fracStochMass() const { return fracStochMass_; }

        /**
         * @brief Get the spectral synthesizer, if any
         * @return A pointer to the spectral synthesizer requested via
         *   spectra.model, or nullptr if spectra.model was not given
         */
        [[nodiscard]] auto specsyn() const -> const specsyn::Specsyn* { return specsyn_.get(); }

        /**
         * @brief Get the filter collection used to compute photometry, if any
         * @return A const reference to the filter collection requested
         *   via phot.filters, or nullptr if phot.filters was not given
         */
        [[nodiscard]] auto filters() const -> const auto& { return filters_; }

        /**
         * @brief Get the distribution of V-band extinction for clustered stars, if any
         * @return A const reference to the A_V distribution requested
         *   via extinct.AV; if neither extinct.AV nor extinct.AV_field
         *   was given, an invalid/empty PDF. If extinct.AV_field was
         *   given but extinct.AV was not, a valid delta function PDF
         *   at 0 (see readExtinct()'s own comment for why) -- so
         *   avDist()/avDistField() are always either both valid or
         *   both invalid, never just one.
         */
        [[nodiscard]] auto avDist() const -> const auto& { return avDist_; }

        /**
         * @brief Get the distribution of V-band extinction for field stars, if any
         * @return A const reference to the A_V distribution requested
         *   via extinct.AV_field -- see avDist()'s own comment for the
         *   identical validity rules this mirrors (with extinct.AV_field/
         *   extinct.AV's own roles swapped)
         */
        [[nodiscard]] auto avDistField() const -> const auto& { return avDistField_; }

        /**
         * @brief Get the extinction curve, if any
         * @return A pointer to the extinction curve requested via
         *   extinct.model, or nullptr if neither extinct.AV nor
         *   extinct.AV_field was given
         */
        [[nodiscard]] auto extinct() const -> const extinct::Extinct* { return extinct_.get(); }

        /**
         * @brief Get the nebular emission control parameters
         * @return A const reference to the control parameters
         *   populated from the input deck's own [nebular] stanza (see
         *   readNebular()), or their own defaults for any key that
         *   was not given
         */
        [[nodiscard]] auto nebControls() const -> const nebular::NebularControls& { return nebControls_; }

        /**
         * @brief Get the nebular emission grid, if any
         * @return A pointer to the nebular emission grid built from
         *   nebControls() and the [nebular] stanza's own table/track
         *   settings (see readNebular()), or nullptr if
         *   nebControls().computeNeb_ is false
         */
        [[nodiscard]] auto nebular() const -> const nebular::Nebular* { return nebular_.get(); }

        /**
         * @brief Check whether the bolometric luminosity was requested as an output
         * @return True if "Lbol" was included in phot.filters
         */
        [[nodiscard]] auto computeLbol() const { return computeLbol_; }

        /**
         * @brief Set whether the bolometric luminosity should be computed as an output
         * @param value New value for computeLbol()
         * @details
         * Lets a caller (e.g. from Python, where there is no
         * phot.filters input-deck entry to set "Lbol" through) request
         * or suppress Lbol output on an already-constructed SimControls.
         */
        void setComputeLbol(bool value) { computeLbol_ = value; }

        // Physics-setting setters: thin wrappers around
        // utils::initPDFFromString(), letting a caller (e.g. from
        // Python, building a SimControls without an input deck at all)
        // set each physics PDF directly from a string playing the same
        // role as the corresponding toml key's value would in the
        // constructor -- a numerical value becomes a delta function,
        // anything else is interpreted as a PDF file name. See
        // utils::initPDFFromString()'s own comment for the exact rules.

        /**
         * @brief Set the initial mass function
         * @param imf A numerical value (interpreted as a delta-function
         *   IMF at that mass) or the name of an IMF PDF file, resolved
         *   relative to SLUG_DIR/REPO_DIR under data/imfs -- the same
         *   way stars.IMF is resolved by the constructor
         * @throws std::runtime_error if imf is not numeric and does
         *   not name a file that can be found
         */
        void setIMF(const std::string& imf) { imf_ = utils::initPDFFromString(imf, imfPrefix); }

        /**
         * @brief Set the cluster mass function
         * @param cmf A numerical value (interpreted as a delta-function
         *   CMF at that mass) or the name of a CMF PDF file
         * @throws std::runtime_error if cmf is not numeric and does
         *   not name a file that can be found
         */
        void setCMF(const std::string& cmf) { cmf_ = utils::initPDFFromString(cmf); }

        /**
         * @brief Set the [Fe/H] distribution
         * @param feH A numerical value (interpreted as a fixed
         *   [Fe/H]) or the name of a [Fe/H] PDF file
         * @throws std::runtime_error if feH is not numeric and does
         *   not name a file that can be found
         * @details
         * If the new fehDist_ is fixed (constFeH() becomes true),
         * also recomputes tracks2D() (the [Fe/H]-sliced cache) from
         * the current tracks_, mirroring the constructor's own
         * post-readTracks() step.
         */
        void setFeH(const std::string& feH);

        /**
         * @brief Set the cluster lifetime function
         * @param clf A numerical value (interpreted as a delta-function
         *   CLF at that lifetime) or the name of a CLF PDF file
         * @throws std::runtime_error if clf is not numeric and does
         *   not name a file that can be found
         */
        void setCLF(const std::string& clf) { clf_ = utils::initPDFFromString(clf); }

        /**
         * @brief Set the star formation rate
         * @param sfr A numerical value (interpreted as a constant star
         *   formation rate, in Msun/yr) or the name of an SFR PDF file
         * @throws std::runtime_error if sfr is neither numeric nor
         *   names a file that can be parsed as a PDF descriptor
         * @details
         * Mirrors the special handling the constructor itself applies
         * to galaxy.sfr, unlike every setter above: a numerical value
         * is not interpreted as a delta function, but as the
         * normalization of a non-normalized PDF that is constant in
         * time (see the .cpp file for the exact construction). A file
         * name, unlike every setter above, is not resolved via
         * utils::getFilePath -- it is passed to
         * pdfs::parsePDFDescriptor() as-is, exactly mirroring the
         * constructor's own galaxy.sfr handling. Also clears sfrDist_
         * back to invalid -- see setSFRDist()'s own comment for why.
         */
        void setSFR(const std::string& sfr);

        /**
         * @brief Set the distribution from which a single, constant star formation rate is drawn
         * @param sfrDist A numerical value (interpreted as a delta
         *   function at that rate -- equivalent to just using setSFR()
         *   directly, but supported for consistency with every other
         *   PDF-valued setter) or the name of an SFR-distribution PDF
         *   file
         * @throws std::runtime_error if sfrDist is not numeric and
         *   does not name a file that can be found
         * @details
         * Unlike setSFR(), this is a plain PDF-valued setter (a
         * numerical value becomes a delta function, exactly as
         * setCLF()/setCMF()/etc. already work) -- see sfrDist()'s own
         * comment for why sfrDist_ is a genuine distribution, not
         * itself a rate as a function of time the way sfr_ is. Also
         * clears sfr_ back to invalid: only one of sfr_/sfrDist_ is
         * ever meant to be valid at a time (mirroring the exclusive-or
         * rule the constructor enforces between galaxy.sfr/
         * galaxy.sfr_dist when parsing a real input deck), so setting
         * one via its own setter invalidates the other, regardless of
         * which was set first.
         */
        void setSFRDist(const std::string& sfrDist)
        {
            sfrDist_ = utils::initPDFFromString(sfrDist);
            sfr_ = pdfs::PDF();
        }

        /**
         * @brief Build a PDF representing a fixed star formation rate, constant in time
         * @param sfr The (constant) star formation rate, in Msun/yr
         * @return A PDF over time (in yr) whose integral over any
         *   [a, b] gives sfr * (b - a) -- see the .cpp file's own
         *   comment for the exact construction
         * @details
         * Exposed as a public static method (rather than a private
         * implementation detail of the constructor's own galaxy.sfr
         * handling and setSFR(), which both use it internally) so that
         * Galaxy::Galaxy() can build the
         * same kind of PDF itself, from a rate it draws from
         * sfrDist() rather than one read directly from galaxy.sfr --
         * see its own comment.
         */
        [[nodiscard]] static auto buildConstantSFR(double sfr) -> pdfs::PDF;

        /**
         * @brief Set the fraction of stellar mass formed in stochastically-treated clusters
         * @param fCluster New fraction; see fCluster()'s own comment
         * @details
         * Unlike setCLF()/setSFR(), fCluster_ is a plain double (not a
         * pdfs::PDF), so this is a direct assignment, mirroring
         * setZ()'s/setIntRelTol()'s own simple setters rather than
         * setCLF()'s/setSFR()'s string-parsing ones.
         */
        void setFCluster(double fCluster) { fCluster_ = fCluster; }

        /**
         * @brief Set the nebular emission control parameters
         * @param nebControls New control parameters; see nebControls()'s own comment
         * @details
         * Like setFCluster(), a plain direct assignment for
         * logU_/covFac_/lineWidth_ -- all three are read live, so
         * updating them here takes effect the next time
         * Nebular::getCluster()/getGalaxy() runs, with no need to
         * rebuild anything. computeNeb_ is deliberately excluded from
         * that assignment and left at whatever this SimControls was
         * actually constructed with: unlike the other three fields,
         * computeNeb_ only ever takes effect once, at construction (it
         * decides whether readNebular() builds nebular_ at all --
         * see nebular()'s own comment), so nebular()'s own null-ness
         * cannot be changed after the fact by assigning here. Setting
         * it here without also rebuilding/discarding nebular_ would
         * silently desynchronize nebControls().computeNeb_ from
         * nebular() itself (e.g. reporting computeNeb_ == false while
         * nebular() stays non-null and Cluster/Galaxy keep computing
         * nebular products from it, since they gate on nebular() being
         * non-null, not on this flag).
         */
        void setNebControls(const nebular::NebularControls& nebControls)
        {
            auto updated = nebControls;
            updated.computeNeb_ = nebControls_.computeNeb_;
            nebControls_ = updated;
        }

        // Object-replacement setters: unlike the string-driven setters
        // above, these accept an already-built object -- e.g. from
        // Python, where Specsyn, FilterCollection, and Tracks3D are
        // all directly constructible -- and install it in place of
        // whatever this SimControls already holds.

        /**
         * @brief Set the spectral synthesizer
         * @param specsyn The spectral synthesizer to use; ownership is
         *   transferred to this SimControls
         * @details
         * Lets a caller replace this SimControls's spectral synthesizer
         * with its own, without needing an input deck.
         */
        void setSpecsyn(std::unique_ptr<specsyn::Specsyn> specsyn) { specsyn_ = std::move(specsyn); }

        /**
         * @brief Set the photometric filter collection
         * @param filters The filter collection to use; ownership is
         *   transferred to this SimControls
         * @details
         * Lets a caller replace this SimControls's filter collection
         * with its own, without needing an input deck.
         */
        void setFilters(std::unique_ptr<phot::FilterCollection> filters) { filters_ = std::move(filters); }

        /**
         * @brief Set the stellar tracks
         * @param tracks The stellar tracks to use
         * @details
         * Lets a caller replace this SimControls's stellar tracks with
         * its own, without needing an input deck. If constFeH() is
         * true, also recomputes tracks2D() (the [Fe/H]-sliced cache)
         * from the new tracks, mirroring setFeH()'s own amendment (see
         * its comment) so the cache never goes stale relative to
         * whichever of tracks_/fehDist_ changed most recently.
         */
        void setTracks(tracks::Tracks3D tracks);

        /**
         * @brief Set the minimum mass for fully stochastic treatment
         * @param minStochMass New minimum mass for fully stochastic
         *   treatment
         * @details
         * Also recomputes fracStochMass() as
         * imf().integral(minStochMass, imf().getMax()), exactly as
         * the constructor does when stars.min_stoch_mass is given.
         */
        void setMinStochMass(double minStochMass)
        {
            minStochMass_ = minStochMass;
            fracStochMass_ = imf_.integral(minStochMass_, imf_.getMax());
        }

    private:

        /**
         * @brief Parse every control-flow setting from the input deck
         * @param inputDeck A toml table holding the input deck
         * @details
         * sim_type, verbosity, output mode/model name/directory,
         * n_trial, output timing (via setOutputTimes), the output
         * content flags (via readOutput()), the optional rng_seed, and
         * the integrator tolerances -- everything the original,
         * pre-merge SimControls class itself used to parse. Split out
         * of the constructor purely to keep its own cognitive
         * complexity down; see initPhysics() for the other half.
         */
        void initControlFlow(const toml::table& inputDeck);

        /**
         * @brief Parse every physics setting from the input deck
         * @param inputDeck A toml table holding the input deck
         * @details
         * IMF, CMF, [Fe/H], the spectral synthesizer and photometric
         * filters (via readSpectra/readFilters), the galaxy-only CLF/
         * SFR, the stellar tracks (via readTracks) and their [Fe/H]-
         * sliced cache, and the minimum stochastic mass -- everything
         * the original, pre-merge SimPhysics class itself used to
         * parse. Must run after initControlFlow(), since the
         * galaxy-only CLF/SFR block depends on simType_. Split out of
         * the constructor purely to keep its own cognitive complexity
         * down; see initControlFlow() for the other half.
         */
        void initPhysics(const toml::table& inputDeck);

        /**
         * @brief Compute output times
         * @param inputDeck A toml table holding the input deck
         */
        void setOutputTimes(const toml::table& inputDeck);

        /**
         * @brief Parse the output content flags from the input deck
         * @param inputDeck A toml table holding the input deck
         * @details
         * Reads the six optional output.write_cluster/
         * write_cluster_spec/write_cluster_phot/write_galaxy/
         * write_galaxy_spec/write_galaxy_phot keys (each defaulting to
         * true) into writeCluster_/writeClusterSpec_/writeClusterPhot_/
         * writeGalaxy_/writeGalaxySpec_/writeGalaxyPhot_. Formerly done
         * by OutputManager's own constructor directly from the input
         * deck; moved here so that a SimControls built without an
         * input deck at all (e.g. from Python) still carries these
         * settings, and an OutputManagerH5/OutputManagerAscii built
         * from it sees the same write_* values a real deck would have
         * produced, not just the all-true defaults -- see
         * OutputManager's own constructor, which now reads these via
         * writeCluster()/etc. instead of parsing them itself.
         */
        void readOutput(const toml::table& inputDeck);

        /**
         * @brief Load a set of tracks specified by input deck
         * @param inputDeck Name of input deck
         * @returns A Tracks3D object with the correct tracks loaded
         */
        void readTracks(const toml::table& inputDeck);

        /**
         * @brief Load the spectral synthesizer specified by input deck
         * @param inputDeck Name of input deck
         * @details
         * Reads spectra.model (and, if present, spectra.registry) and
         * sets specsyn_ accordingly; see the .cpp file for the exact
         * rules. spectra.model is optional, so specsyn_ is left null
         * if it is absent. The spectral synthesizer this builds stores
         * a live reference back to *this (for its own integrator
         * tolerances), so this must only ever be called from this
         * SimControls's own constructor, on an object that will not be
         * moved or destroyed before the synthesizer is.
         */
        void readSpectra(const toml::table& inputDeck);

        /**
         * @brief Load the photometric filter collection specified by input deck
         * @param inputDeck Name of input deck
         * @details
         * Reads phot.system, phot.filters, phot.registry, and
         * phot.vega, and sets filters_/computeLbol_ accordingly; see
         * the .cpp file for the exact rules. phot.filters is optional,
         * so filters_ is left null if it (or the non-"Lbol" remainder
         * of it) is empty.
         */
        void readFilters(const toml::table& inputDeck);

        /**
         * @brief Load the extinction curve specified by input deck
         * @param inputDeck A toml table holding the input deck
         * @details
         * Reads extinct.AV, extinct.AV_field, extinct.model, and
         * extinct.registry, and sets avDist_/avDistField_/extinct_
         * accordingly; see the .cpp file for the exact rules. Both
         * extinct.AV and extinct.AV_field are optional, and
         * avDist_/avDistField_/extinct_ are all left at their default/
         * null state if neither is given. If either is given,
         * extinct.model becomes mandatory, and whichever of the two
         * was not given is set to a valid delta function PDF at 0
         * (rather than left invalid) -- so avDist_/avDistField_ are
         * always either both valid or both invalid, and callers never
         * need to check one without the other. Requires a spectral
         * synthesizer to already be set (see readSpectra()), since the
         * extinction curve is interpolated onto its own wavelength
         * grid, and the nebular emission grid to already be set (see
         * readNebular(), called before this method for exactly that
         * reason), since the extinction curve is also interpolated
         * onto its own line wavelengths, if any (see Extinct's own
         * extinctLines_ member).
         */
        void readExtinct(const toml::table& inputDeck);

        /**
         * @brief Load the nebular emission controls and grid specified by input deck
         * @param inputDeck A toml table holding the input deck
         * @details
         * Reads nebular.log_U, nebular.cov_fac, and nebular.line_width
         * into nebControls_, each left at its own nebular::defaultXxx value
         * (already NebularControls's own default member initializers)
         * if its key was not given -- so, unlike readSpectra()'s or
         * readExtinct()'s own all-or-nothing stanzas, every key here
         * is independently optional. Also reads nebular.table (falling
         * back to nebular::defaultTable), and stars.tracks/
         * stars.v_vcrit -- the same keys, read the same way, as
         * readTracks() -- and uses them to construct nebular_.
         */
        void readNebular(const toml::table& inputDeck);

        // Simulation control parameters
        SimType simType_ = SimType::none;              /**< Simulation type */
        unsigned int verbosity_ = 0;                   /**< Level of verbosity */
        unsigned long nTrial_ = 1;                     /**< Number of trials */
        OutputMode outputMode_ = OutputMode::h5;       /**< Output mode */
        std::string modelName_ = "slug_sim";           /**< Name of this model */
        std::string outDir_;                           /**< Directory into which output will be written */
        std::vector<double> outTimes_;                 /**< Times to write output */
        pdfs::PDF outTimeDist_;                        /**< Distribution of output times */
        double intRelTol_ = 1e-2;                      /**< Relative tolerance for PDF integrator */
        /**
         * @brief Absolute tolerance for PDF integrator
         * @details
         * Nonzero by default (unlike a bare relative tolerance) because
         * a purely relative criterion is pathological wherever the
         * integrand's own true value passes near zero -- e.g.
         * Specsyn::continuousSpecIntegrand()'s per-wavelength flux,
         * which can be genuinely tiny at wavelengths a given stellar
         * population barely emits at -- forcing the integrator to keep
         * refining indefinitely chasing relative precision on a
         * near-zero quantity. 1e-3 is a sane floor at the same order as
         * intRelTol_'s own default, not a value tuned to any specific
         * integral.
         */
        double intAbsTol_ = 1e-3;
        /**
         * @brief Max bisection iterations for PDF integrator (0 = unlimited)
         * @details
         * Not a cap on raw evaluations of the integrand itself -- each
         * bisection iteration costs a small, fixed number of further
         * evaluations, not a growing one (see GKIntegrator::
         * integrate()'s own comment) -- but a cap on how many times
         * PDFIntegrator (see its own class comment; built on
         * GKIntegrator, an adaptive Gauss-Kronrod quadrature, since
         * replacing an earlier, cubature-package-based implementation)
         * is allowed to keep bisecting whichever subinterval currently
         * contributes the most error, chasing intRelTol_/intAbsTol_.
         * Nonzero by default: with 0 (unlimited), an integrand that is
         * slow to converge to the requested tolerance -- e.g.
         * Specsyn::continuousSpecIntegrand()'s own dead-star
         * discontinuity in (age, mass) space, whose exact effect on
         * convergence rate isn't knowable in advance for an arbitrary
         * track set/SFH -- has no cap on how long it can run. Unlike
         * the cubature-based implementation this replaced -- whose own
         * point cache could grow by a whole resolution *doubling* per
         * iteration, an inherently discontinuous jump in memory demand
         * that once caused a real memory blowup in this same codebase
         * -- each of GKIntegrator's own bisections adds at most one net
         * new subinterval, so its memory use grows linearly with
         * intMaxIter_, not exponentially; 2^19 (524288), inherited
         * unchanged from that earlier implementation's own tuning, has
         * not needed revisiting since.
         */
        std::size_t intMaxIter_ = 1UL << 19;
        double z_ = 0.0;                               /**< Redshift, read live by every Specsyn/Extinct built from this SimControls */
        unsigned long checkpointInterval_ = 0;         /**< Number of trials between checkpoints (0 = disabled); see checkpointInterval() */
        std::string inputDeckStr_;                     /**< The input deck's own text, re-serialized; see inputDeckStr() */

        // Output content flags, parsed by readOutput(); see
        // writeCluster()'s own comment and OutputManager's own
        // constructor for where these are enforced/consumed.
        bool writeCluster_ = true;      /**< Whether to write the clusters group/file; see writeCluster() */
        bool writeClusterSpec_ = true;  /**< Whether to write the cluster_spectra group/file; see writeClusterSpec() */
        bool writeClusterPhot_ = true;  /**< Whether to write the cluster_phot group/file; see writeClusterPhot() */
        bool writeGalaxy_ = true;       /**< Whether to write the galaxy group/file (galaxy-type simulations only); see writeGalaxy() */
        bool writeGalaxySpec_ = true;   /**< Whether to write the galaxy_spectra group/file (galaxy-type simulations only); see writeGalaxySpec() */
        bool writeGalaxyPhot_ = true;   /**< Whether to write the galaxy_phot group/file (galaxy-type simulations only); see writeGalaxyPhot() */

        // Physics settings
        pdfs::PDF imf_;            /**< The IMF to use for the simulation */
        pdfs::PDF cmf_;            /**< Cluster mass function */
        pdfs::PDF fehDist_;        /**< [Fe/H] distribution */
        pdfs::PDF sfr_;            /**< Star formation rate, as a function of time -- see sfr()'s own comment for when this is invalid */
        pdfs::PDF sfrDist_;        /**< Distribution from which a single, constant star formation rate is drawn -- see sfrDist()'s own comment */
        pdfs::PDF clf_;            /**< Cluster lifetime function */
        double fCluster_ = 1.0;    /**< Fraction of stellar mass formed in stochastically-treated clusters (galaxy sims only) */
        tracks::Tracks3D tracks_;  /**< Stellar tracks */
        tracks::Tracks2D constFeHTracks_; /**< Tracks sliced at fehDist_'s value, if constFeH() */
        double minStochMass_ = 0.0;   /**< Minimum mass for fully stochastic treatment */
        double fracStochMass_ = 1.0;  /**< Fraction of mass being treated stochastically */
        std::unique_ptr<specsyn::Specsyn> specsyn_; /**< Spectral synthesizer, or nullptr if spectra.model was not given */
        std::unique_ptr<phot::FilterCollection> filters_; /**< Photometric filters requested via phot.filters, or nullptr if none were given */
        bool computeLbol_ = false; /**< True if "Lbol" was included in phot.filters; see Cluster::computeLbol() for where it is actually computed */
        pdfs::PDF avDist_; /**< Distribution of V-band extinction (A_V) for clustered stars -- see avDist()'s own comment for exactly when this is valid/a delta at 0/invalid */
        pdfs::PDF avDistField_; /**< Distribution of V-band extinction (A_V) for field stars -- see avDistField()'s own comment */
        std::unique_ptr<extinct::Extinct> extinct_; /**< Extinction curve requested via extinct.model, or nullptr if neither extinct.AV nor extinct.AV_field was given */
        nebular::NebularControls nebControls_; /**< Nebular emission control parameters, see nebControls() */
        std::unique_ptr<nebular::Nebular> nebular_; /**< Nebular emission grid requested via the [nebular] stanza */

        // Output wavelength grid (spectra.wl_min, spectra.wl_max,
        // spectra.nwl), read by readSpectra and passed through to
        // whichever spectral synthesizer constructor it builds. Any
        // of the three not supplied in the deck falls back to
        // specsyn::defaultWlMin/defaultWlMax/defaultNWl (see
        // readSpectra's own comment for why); these in-class
        // initializers are only ever visible if readSpectra returns
        // early because spectra.model itself was absent, in which
        // case no spectral synthesizer exists and these values are
        // never read.
        double wlMin_ = 0.0;      /**< Output minimum wavelength, in Angstrom */
        double wlMax_ = 0.0;      /**< Output maximum wavelength, in Angstrom */
        unsigned long nWl_ = 0;   /**< Number of output wavelengths */
    };

} // namespace io

#endif // SIMCONTROLS_HPP
