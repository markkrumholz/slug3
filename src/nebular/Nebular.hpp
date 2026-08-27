/**
 * @file Nebular.hpp
 * @author Mark Krumholz
 * @brief A class to provide nebular emission per ionizing photon from a pre-computed cloudy grid
 * @date 2026-08-26
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef NEBULAR_HPP
#define NEBULAR_HPP

#include "../phot/FilterIdeal.hpp"
#include "../tracks/TrackCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <string>
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

/**
 * @brief A namespace to hold machinery for nebular emission from a pre-computed cloudy grid
 */
namespace nebular
{

    /**
     * @class Nebular
     * @brief Nebular emission line and continuum luminosity per ionizing photon, from a pre-computed cloudy grid
     * @details
     * This class looks up the "quick and dirty" nebular emission grid
     * built by data/tools/cloudy's own three-stage pipeline (see that
     * directory's own scripts) for one track set, and provides the
     * line and continuum luminosity per ionizing photon it holds, as
     * a function of [Fe/H] and (for a cluster) age -- so that slug can
     * add nebular emission to a stellar spectrum without running
     * cloudy itself.
     */
    class Nebular
    {
    public:

        // Shorten mdspan types
        using Grid2D = std::mdspan<double, std::dextents<std::size_t, 2>>; // NOLINT(misc-include-cleaner)
        using Grid3D = std::mdspan<double, std::dextents<std::size_t, 3>>;

        /**
         * @brief Construct a Nebular object
         * @param tableName Path to the nebular emission table (see
         *   nebular::defaultTable) to load this track set's own grid
         *   from; used only during construction, not retained afterward
         * @param trackName Name of the track set this nebular grid is
         *   for; used only during construction (to locate this track
         *   set's own cloudy grid data), not retained afterward
         * @param simControls Simulation controls (physics and
         *   control-flow settings); also the source of the simulation
         *   wavelength grid the continuum luminosity grids should
         *   ultimately be resampled onto (simControls.specsyn()->wl()),
         *   so that grid need not be passed in separately
         * @param vvcrit Rotation rate v/vcrit of the track set; like
         *   tableName and trackName, used only during construction,
         *   not retained afterward
         * @details
         * simControls is stored by reference, so the object passed in
         * must outlive this Nebular. This constructor takes wl_
         * directly from simControls.specsyn()->wl() and, from it,
         * builds ctmLumPerQCluster_/ctmLumPerQGalaxy_ (the table's own
         * continuum data, resampled onto wl_ via Interpolator1D) and
         * lineLumPerQCluster_/lineLumPerQGalaxy_ (the table's own line
         * luminosities, read directly, for the table's full line list
         * in lineWl_/lineLabel_).
         * @throws std::runtime_error if tableName cannot be found (via
         *   utils::getFilePath), trackName has no group of its own in
         *   it, or -- for any of that group's own [Fe/H] values -- no
         *   v/vcrit subgroup matches vvcrit exactly, or
         *   simControls.nebControls().logU_ falls outside the range of
         *   logU values actually tabulated for a given [Fe/H]/v/vcrit
         *   combination
         */
        Nebular(
            const std::string& tableName,
            const std::string& trackName,
            const io::SimControls& simControls,
            double vvcrit = tracks::defaultVVcrit);

        ~Nebular() = default;

        // The mdspan members below are non-owning views into this
        // object's own backing vectors (lineLumPerQClusterData_ etc);
        // a naive copy or move would leave the copy's/moved-from's own
        // views pointing at the wrong object's data, so both are
        // disallowed for now -- mirrors utils::ThreadVec's own
        // identical reasoning for its own vector-backed member.
        Nebular(const Nebular&) = delete;
        Nebular(Nebular&&) = delete;
        auto operator=(const Nebular&) -> Nebular& = delete;
        auto operator=(Nebular&&) -> Nebular& = delete;

        // Observers

        /**
         * @brief Get the label of each of this Nebular's own nebular emission lines
         * @return lineLabel_, in the same order as lineWl() -- see
         *   lineWl_'s own comment for that order's own source
         */
        [[nodiscard]] auto lineLabel() const -> const std::vector<std::string>& { return lineLabel_; }

        /**
         * @brief Get the wavelength of each of this Nebular's own nebular emission lines
         * @return lineWl_, in Angstrom, in the table's own line
         *   ordering (the same order as lineLabel())
         */
        [[nodiscard]] auto lineWl() const -> const std::vector<double>& { return lineWl_; }

        // Nebular emission

        /**
         * @brief Add whole-galaxy nebular emission to an input stellar spectrum
         * @param spec Input stellar spectrum, on this object's own
         *   wl_ grid (i.e. simControls.specsyn()->wl(), see the
         *   constructor's own comment), in erg/s/Angstrom
         * @param feH [Fe/H] of the stellar population spec was computed for
         * @return A pair whose first element is the stellar + nebular
         *   continuum + nebular line spectrum, on wl_'s own grid, in
         *   erg/s/Angstrom, and whose second element is the
         *   luminosity of each of lineWl()'s own lines, in erg/s
         * @throws std::runtime_error if feH falls outside feH_'s own tabulated range
         * @details
         * Q(HI) (the H-ionizing photon rate) is read off spec via
         * qhiFilter_; every quantity added below scales with it.
         * Every per-[Fe/H] quantity (lineLumPerQGalaxy_,
         * ctmLumPerQGalaxy_) is linearly interpolated between the two
         * tabulated [Fe/H] values bracketing feH. The continuum is
         * added everywhere; the input spectrum itself is kept only
         * above the H-ionization edge (wavelengths <= qhiFilter_.wlMax()
         * are zeroed, since a real HII region reprocesses essentially
         * all ionizing stellar flux); lines are deposited via
         * depositLines().
         */
        [[nodiscard]] auto getGalaxy(const std::vector<double>& spec, double feH) const
            -> std::pair<std::vector<double>, std::vector<double>>;

        /**
         * @brief Add one cluster's own nebular emission to an input stellar spectrum
         * @param spec Input stellar spectrum, on this object's own
         *   wl_ grid, in erg/s/Angstrom
         * @param feH [Fe/H] of the cluster spec was computed for
         * @param age Cluster age, in yr
         * @return Same as getGalaxy(), but interpolated in cluster
         *   age as well as [Fe/H] -- except that, for age above
         *   clusterAge_'s own tabulated range (this table has no data
         *   for a cluster this old), the returned line luminosities
         *   are all zero and the returned spectrum is just spec with
         *   every bin at or below the H-ionization edge zeroed (the
         *   same edge-zeroing getGalaxy() applies, with no continuum
         *   or line emission added)
         * @throws std::runtime_error if feH falls outside feH_'s own
         *   tabulated range (age has no such failure mode: below
         *   clusterAge_'s own minimum is pinned to that minimum, and
         *   above its own maximum takes the early-return path
         *   described above)
         */
        [[nodiscard]] auto getCluster(const std::vector<double>& spec, double feH, double age) const
            -> std::pair<std::vector<double>, std::vector<double>>;

    private:

        /**
         * @brief spec with every bin at or below the H-ionization edge (qhiFilter_.wlMax()) zeroed
         * @param spec Input stellar spectrum, on wl_'s own grid, in erg/s/Angstrom
         * @return A copy of spec, zeroed for every wl_ bin at or below qhiFilter_.wlMax()
         * @details Shared by getGalaxy()/getCluster() as the base every nebular emission addition builds on.
         */
        [[nodiscard]] auto stellarAboveEdge(const std::vector<double>& spec) const -> std::vector<double>;

        /**
         * @brief Add every line's own deposited power into spec, in place
         * @param lineLum Luminosity of each of lineWl_'s own lines, in erg/s
         * @param spec Spectrum (on wl_'s own grid, in erg/s/Angstrom) to add line power into
         * @details
         * For each line with nonzero lineLum, walks lineDepositFrac_'s
         * own window around lineCenterIdx_.at(ell), converting each
         * bin's own deposited power (lineLum times that bin's own
         * fraction) to a flux density by dividing by that bin's own
         * width (the geometric-mean-boundary width used throughout
         * this class -- see computeLineDepositWindows()'s own
         * boundaries). A bin whose own width can't be computed without
         * running off wl_'s own edge (wl_'s own first and last bins,
         * which have no left/right neighbor respectively) is skipped
         * rather than crossing that edge.
         */
        void depositLines(const std::vector<double>& lineLum, std::vector<double>& spec) const;

        const io::SimControls& simControls_; /**< Simulation controls (physics and control-flow settings) */

        /**
         * @brief Ideal filter used to extract Q(HI), the H-ionizing photon rate, from an input stellar spectrum
         * @details
         * Built once, at construction, as phot::FilterIdeal("Q(HI)") --
         * see that constructor overload's own comment for how the
         * "Q(<spec>)" name is parsed. Used by getGalaxy()/getCluster()
         * to read Q(HI) off an input spectrum (via qhiFilter_.phot())
         * and to locate the H-ionization edge (via qhiFilter_.wlMax()).
         */
        phot::FilterIdeal qhiFilter_;

        /**
         * @brief This object's own nebular spectral grid
         * @details
         * Set once, at construction, directly from
         * simControls_.specsyn()->wl() -- this class does not build a
         * grid of its own. This is the wavelength axis every
         * ctmLumPerQCluster_/ctmLumPerQGalaxy_ entry is resampled
         * onto, and so defines the size of their own final dimension.
         */
        std::vector<double> wl_;

        std::vector<std::string> lineLabel_; /**< Label of each of the table's own nebular emission lines */
        std::vector<double> lineWl_;         /**< Wavelength of each of the table's own nebular emission lines */

        /**
         * @brief Each of lineWl_'s own lines' index into wl_ of the bin its central wavelength falls into
         * @details
         * 0 for a line whose central wavelength falls outside wl_'s
         * own range -- such a line's own lineDepositFrac_ column is
         * all zero, so which index is recorded for it here is
         * otherwise immaterial. Set once, at construction, by
         * Nebular.cpp's own computeLineDepositWindows(); used by
         * depositLines() to know where in wl_ each line's own window
         * (lineDepositFrac_) is centered.
         */
        std::vector<size_t> lineCenterIdx_;

        /**
         * @brief Fraction of each of lineWl_'s own lines' power to deposit in each wl_ bin of a window around lineCenterIdx_, shaped (window, line)
         * @details
         * Every line's own column is centered on its lineCenterIdx_
         * within the shared window (sized to the widest of every
         * line's own individually computed window -- see
         * computeLineDepositWindows()), zero-padded on both sides out
         * to that width. Set once, at construction; consumed by
         * depositLines().
         */
        Grid2D lineDepositFrac_;
        std::vector<double> lineDepositFracData_; /**< Data holder for lineDepositFrac_ */

        std::vector<double> feH_;        /**< [Fe/H] grid this track's own nebular data is tabulated at, ascending */
        std::vector<double> clusterAge_; /**< Cluster age grid (yr) ctmLumPerQCluster_'s own age axis is tabulated at, from the table's own top-level "time" dataset */

        /** @brief Cluster line luminosity per ionizing photon, shaped ([Fe/H], age, line) */
        Grid3D lineLumPerQCluster_;
        std::vector<double> lineLumPerQClusterData_; /**< Data holder for lineLumPerQCluster_ */

        /** @brief Galaxy line luminosity per ionizing photon, shaped ([Fe/H], line) */
        Grid2D lineLumPerQGalaxy_;
        std::vector<double> lineLumPerQGalaxyData_; /**< Data holder for lineLumPerQGalaxy_ */

        /** @brief Cluster continuum luminosity per ionizing photon, shaped (feH_, clusterAge_, wl_) */
        Grid3D ctmLumPerQCluster_;
        std::vector<double> ctmLumPerQClusterData_; /**< Data holder for ctmLumPerQCluster_ */

        /** @brief Galaxy continuum luminosity per ionizing photon, shaped (feH_, wl_) */
        Grid2D ctmLumPerQGalaxy_;
        std::vector<double> ctmLumPerQGalaxyData_; /**< Data holder for ctmLumPerQGalaxy_ */
    };

} // namespace nebular

#endif // NEBULAR_HPP
