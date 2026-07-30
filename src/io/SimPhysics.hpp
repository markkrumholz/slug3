/**
 * @file SimPhysics.hpp
 * @author Mark Krumholz
 * @brief A class to hold physics settings for the simulation
 * @date 2026-07-12
 */

#ifndef SIMPHYSICS_HPP
#define SIMPHYSICS_HPP

#include "../pdfs/PDF.hpp"
#include "../specsyn/Specsyn.hpp"
#include "../tracks/Tracks2D.hpp"
#include "../tracks/Tracks3D.hpp"
#include "SimControls.hpp"
#include <cstddef>
#include <memory>
#include <toml.hpp>

namespace io
{

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