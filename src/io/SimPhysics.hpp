/**
 * @file SimPhysics.hpp
 * @author Mark Krumholz
 * @brief A class to hold physics settings for the simulation
 * @date 2026-07-12
 */

#ifndef SIMPHYSICS_HPP
#define SIMPHYSICS_HPP

#include "../pdfs/PDF.hpp"
#include "../phot/FilterCollection.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../tracks/Tracks3D.hpp"
#include "../utils/ParseUtils.hpp"
#include "SimControls.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <toml.hpp>

namespace io
{
    /**
     * @brief Prefix under which to resolve an IMF PDF file
     * @details
     * Shared by SimPhysics's constructor (reading stars.IMF via
     * utils::initPDFFromKey()) and SimPhysics::setIMF() (via
     * utils::initPDFFromString()), so both resolve an IMF file name
     * identically.
     */
    inline const std::string imfPrefix = // NOLINT(bugprone-throwing-static-initialization,cert-err58-cpp) -- built from fixed string literals, so the (theoretically throwing) path conversion can never actually throw here
        (std::filesystem::path("data") / std::filesystem::path("imfs")).string();

    /**
     * @class SimPhysics
     * @brief A class to hold physics settings for a simulation
     */
    class SimPhysics
    {
    public:

        /**
         * @brief Initialize the simulation physics settings
         * @param inputDeck A toml table holding the input deck
         * @param controls Simulation controls; used to determine the
         *   simulation type and to thread integrator tolerance settings
         *   through to the spectral synthesizer
         */
        SimPhysics(const toml::table& inputDeck, const SimControls& controls);

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
         * lifetime of this SimPhysics object, so that Cluster objects
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
         * or suppress Lbol output on an already-constructed SimPhysics.
         */
        void setComputeLbol(bool value) { computeLbol_ = value; }

        // Physics-setting setters: thin wrappers around
        // utils::initPDFFromString(), letting a caller (e.g. from
        // Python, building a SimPhysics without an input deck at all)
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
        // whatever this SimPhysics already holds.

        /**
         * @brief Set the spectral synthesizer
         * @param specsyn The spectral synthesizer to use; ownership is
         *   transferred to this SimPhysics
         * @details
         * Lets a caller replace this SimPhysics's spectral synthesizer
         * with its own, without needing an input deck.
         */
        void setSpecsyn(std::unique_ptr<specsyn::Specsyn> specsyn) { specsyn_ = std::move(specsyn); }

        /**
         * @brief Set the photometric filter collection
         * @param filters The filter collection to use; ownership is
         *   transferred to this SimPhysics
         * @details
         * Lets a caller replace this SimPhysics's filter collection
         * with its own, without needing an input deck.
         */
        void setFilters(std::unique_ptr<phot::FilterCollection> filters) { filters_ = std::move(filters); }

        /**
         * @brief Set the stellar tracks
         * @param tracks The stellar tracks to use
         * @details
         * Lets a caller replace this SimPhysics's stellar tracks with
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

        // Integrator tolerance getters and setters: thin wrappers
        // around the spectral synthesizer's own Specsyn::intRelTol()
        // etc., so a caller holding a SimPhysics (e.g. from Python,
        // where a Cluster only ever reaches the spectral synthesizer
        // indirectly through the SimPhysics it was built from) can
        // retune integration tolerances on an already-constructed
        // synthesizer, without building an entirely new SimPhysics.
        // All six throw if no spectral synthesizer was requested
        // (spectra.model was not set in the input deck).

        /**
         * @brief Get the relative tolerance for the spectral synthesizer's PDF integration
         * @return Relative tolerance passed to PDFIntegrator
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        [[nodiscard]] auto intRelTol() const -> double;

        /**
         * @brief Get the absolute tolerance for the spectral synthesizer's PDF integration
         * @return Absolute tolerance passed to PDFIntegrator
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        [[nodiscard]] auto intAbsTol() const -> double;

        /**
         * @brief Get the max evaluations for the spectral synthesizer's PDF integration
         * @return Max evaluations passed to PDFIntegrator (0 = unlimited)
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        [[nodiscard]] auto intMaxIter() const -> std::size_t;

        /**
         * @brief Set the relative tolerance for the spectral synthesizer's PDF integration
         * @param tol New relative tolerance
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        void setIntRelTol(double tol);

        /**
         * @brief Set the absolute tolerance for the spectral synthesizer's PDF integration
         * @param tol New absolute tolerance
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        void setIntAbsTol(double tol);

        /**
         * @brief Set the max evaluations for the spectral synthesizer's PDF integration
         * @param n New maximum (0 = unlimited)
         * @throws std::runtime_error if no spectral synthesizer was requested
         */
        void setIntMaxIter(std::size_t n);

    private:

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
         * if it is absent.
         */
        void readSpectra(const toml::table& inputDeck, const SimControls& controls);

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

#endif // SIMPHYSICS_HPP