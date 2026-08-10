/**
 * @file SimControls.hpp
 * @author Mark Krumholz
 * @brief Implements a class to control simulation flow and physics settings
 * @date 16-07-2026
 */

#ifndef SIMCONTROLS_HPP
#define SIMCONTROLS_HPP

#include "../extinct/Extinct.hpp"
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
            h5,      /**< HDF5 output */
            ascii   /**< ASCII output */
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
        // builds stores a live reference back to *this (see Specsyn's
        // own controls_ member) -- this object's address must never
        // change after construction for that reference to stay valid.
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
         * @return Absolute tolerance passed to PDFIntegrator (default 0)
         */
        [[nodiscard]] auto intAbsTol() const { return intAbsTol_; }

        /**
         * @brief Return the maximum number of evaluations for PDF integration
         * @return Max evaluations passed to PDFIntegrator (0 = unlimited, the default)
         */
        [[nodiscard]] auto intMaxIter() const { return intMaxIter_; }

        /**
         * @brief Return the redshift
         * @return Redshift applied by every Specsyn's and Extinct's
         *   own wlObs() (default 0, i.e. no redshift)
         */
        [[nodiscard]] auto z() const { return z_; }

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
         * @brief Set the maximum number of evaluations for PDF integration
         * @param n New maximum (0 = unlimited)
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
         * @return Pointer to the simulation star formation rate
         */
        [[nodiscard]] auto sfr() const -> const auto& { return sfr_; }

        /**
         * @brief Get simulation cluster lifetime function
         * @return Pointer to the simulation cluster lifetime function
         */
        [[nodiscard]] auto clf() const -> const auto& { return clf_; }

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
         * @brief Get the distribution of V-band extinction, if any
         * @return A const reference to the A_V distribution requested
         *   via extinct.AV, or an invalid/empty PDF if extinct.AV was
         *   not given
         */
        [[nodiscard]] auto avDist() const -> const auto& { return avDist_; }

        /**
         * @brief Get the extinction curve, if any
         * @return A pointer to the extinction curve requested via
         *   extinct.model, or nullptr if extinct.AV was not given
         */
        [[nodiscard]] auto extinct() const -> const extinct::Extinct* { return extinct_.get(); }

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
         * constructor's own galaxy.sfr handling.
         */
        void setSFR(const std::string& sfr);

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
         * n_trial, output timing (via setOutputTimes), the optional
         * rng_seed, and the integrator tolerances -- everything the
         * original, pre-merge SimControls class itself used to parse.
         * Split out of the constructor purely to keep its own
         * cognitive complexity down; see initPhysics() for the other
         * half.
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
         * Reads extinct.AV, extinct.model, and extinct.registry, and
         * sets avDist_/extinct_ accordingly; see the .cpp file for the
         * exact rules. extinct.AV is optional, so avDist_/extinct_ are
         * left at their default/null state if it is absent. Requires a
         * spectral synthesizer to already be set (see readSpectra()),
         * since the extinction curve is interpolated onto its own
         * wavelength grid.
         */
        void readExtinct(const toml::table& inputDeck);

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
        double intAbsTol_ = 0.0;                       /**< Absolute tolerance for PDF integrator */
        std::size_t intMaxIter_ = 0;                   /**< Max evaluations for PDF integrator (0 = unlimited) */
        double z_ = 0.0;                               /**< Redshift, read live by every Specsyn/Extinct built from this SimControls */

        // Physics settings
        pdfs::PDF imf_;            /**< The IMF to use for the simulation */
        pdfs::PDF cmf_;            /**< Cluster mass function */
        pdfs::PDF fehDist_;        /**< [Fe/H] distribution */
        pdfs::PDF sfr_;            /**< Star formation rate */
        pdfs::PDF clf_;            /**< Cluster lifetime function */
        tracks::Tracks3D tracks_;  /**< Stellar tracks */
        tracks::Tracks2D constFeHTracks_; /**< Tracks sliced at fehDist_'s value, if constFeH() */
        double minStochMass_ = 0.0;   /**< Minimum mass for fully stochastic treatment */
        double fracStochMass_ = 1.0;  /**< Fraction of mass being treated stochastically */
        std::unique_ptr<specsyn::Specsyn> specsyn_; /**< Spectral synthesizer, or nullptr if spectra.model was not given */
        std::unique_ptr<phot::FilterCollection> filters_; /**< Photometric filters requested via phot.filters, or nullptr if none were given */
        bool computeLbol_ = false; /**< True if "Lbol" was included in phot.filters; see Cluster::computeLbol() for where it is actually computed */
        pdfs::PDF avDist_; /**< Distribution of V-band extinction (A_V), requested via extinct.AV; invalid/empty if extinct.AV was not given */
        std::unique_ptr<extinct::Extinct> extinct_; /**< Extinction curve requested via extinct.model, or nullptr if extinct.AV was not given */

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
