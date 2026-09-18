/**
 * @file Yields.hpp
 * @author Mark Krumholz
 * @brief A class to chain together nucleosynthetic yield channels
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDS_HPP
#define YIELDS_HPP

#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace io
{
    class SimControls;
} // namespace io

namespace yields
{
    /**
     * @class Yields
     * @brief Chains together the nucleosynthetic yield channels a
     *   SimControls requests, and is the connector between them and
     *   the rest of the code
     * @details
     * Built from controls.yieldChannels() (see the constructor's own
     * comment): one YieldChannel per descriptor, added via addChannel()
     * and owned in yieldChannels_. The constructor also calls
     * rebuildYieldGrid() once, which collects every loaded channel's
     * own isotopesOrig() into one deduplicated, sorted isotopes_ (the
     * union of every isotope any requested channel covers), then pushes
     * that same isotopes_ back down into every channel's own
     * rebuildYieldGrid() -- so every YieldChannel this Yields owns ends
     * up synchronized onto the same isotope list, in the same order.
     * There will be more to this class -- nothing yet combines the
     * channels' actual yield values together.
     */
    class Yields
    {
    public:

        /** @brief mdspan view returned by yield() -- see its own comment */
        using Array2D = std::mdspan<const double, std::dextents<std::size_t, 2>>; // NOLINT(misc-include-cleaner)

        /**
         * @brief Construct a Yields from a SimControls's own yieldChannels()
         * @param controls Simulation controls this Yields reads its
         *   yield-channel descriptors (controls.yieldChannels()) and
         *   [Fe/H] range (controls.fehDist()) from, live, for the rest
         *   of its lifetime -- see controls_'s own comment. Must
         *   outlive this Yields.
         * @param registryName Name of the yield registry file
         * @throws std::runtime_error if any descriptor in
         *   controls.yieldChannels() names a channel/model not found in
         *   registryName, or whose [Fe/H] range doesn't cover
         *   controls.fehDist() -- see addChannel()'s own comment
         * @details
         * Prints a "slug: warning" (to std::cout, not thrown -- see
         * below) if controls.yieldChannels() requests the same channel
         * more than once: a suspicious pattern that's easy to hit by
         * accident (e.g. a copy-pasted yields.channelN table with the
         * "model" key never updated), but not forbidden outright, since
         * there are legitimate reasons for it too -- comparing two
         * different yield tables for the same channel, or covering one
         * channel's own mass range with two different models over two
         * different sub-ranges.
         *
         * Then calls addChannel() once per entry in
         * controls.yieldChannels(), in order, then rebuildYieldGrid()
         * once -- see its own comment for what that does.
         */
        Yields(const io::SimControls& controls,
            std::string registryName = defaultRegistry);

        // Movable (moving yieldChannels_'s own ownership, and rebinding
        // controls_ to the same referent) but neither copyable (
        // yieldChannels_ holds move-only std::unique_ptr<YieldChannel>
        // elements) nor assignable (controls_ can't be reseated,
        // matching Extinct's/Specsyn's own identical controls_ members
        // -- see either one's own comment).
        Yields(const Yields&) = delete;
        Yields(Yields&&) = default;
        auto operator=(const Yields&) -> Yields& = delete;
        auto operator=(Yields&&) -> Yields& = delete;
        ~Yields() = default;

        /**
         * @brief Add one YieldChannel, built from a descriptor, to yieldChannels_
         * @param descriptor Which channel/model to load, and the mass
         *   range it should cover
         * @throws std::runtime_error if descriptor.channel_/modelName_
         *   is not found in the registry named by registryName(), or if
         *   controls_.fehDist()'s own range lies outside the [Fe/H]
         *   range actually available for that channel/model -- see
         *   YieldChannel::YieldChannel()'s own comment
         * @details
         * The new YieldChannel is loaded over controls_.fehDist()'s own
         * [min, max] range -- mirrors SimControls::readTracks()'s
         * identical use of fehDist_.getMin()/getMax() to pick the
         * [Fe/H] range a model is loaded over. The new YieldChannel
         * itself caches a copy of descriptor (see its own descriptor()
         * observer), so rebuildYieldGrid() can later recover each
         * channel's own mMin_/mMax_ directly from the channel itself,
         * without needing a separate, parallel record of its own here.
         */
        void addChannel(const YieldChannelDescriptor& descriptor);

        /**
         * @brief Add one already-built YieldChannel to yieldChannels_
         * @param channel The channel to add; ownership is transferred
         *   to yieldChannels_. Must not be null.
         * @throws std::invalid_argument if channel is null
         * @details
         * Unlike addChannel(const YieldChannelDescriptor&), which
         * constructs a fresh YieldChannel from a descriptor, this
         * overload just moves an already-built channel into place --
         * e.g. one built and rebuiltYieldGrid()-ed by a caller (from
         * Python) with settings addChannel(descriptor) has no way to
         * express itself. Does not call rebuildYieldGrid(): channel is
         * appended as-is, whatever isotope list it was itself last
         * built over, so a caller that wants it synchronized onto
         * isotopes_ (or the reverse -- isotopes_ recomputed to include
         * its own isotopesOrig()) must call rebuildYieldGrid()
         * afterward, exactly as after the descriptor-taking overload.
         */
        void addChannel(std::unique_ptr<YieldChannel> channel);

        /**
         * @brief Remove one entry from yieldChannels_
         * @param index Index, into yieldChannels_, of the channel to remove
         * @throws std::out_of_range if index >= yieldChannels().size()
         * @details
         * Does not call rebuildYieldGrid(): isotopes_ (and every
         * remaining channel's own isotopes()) is left describing the
         * union that existed before index was removed, until a caller
         * reruns rebuildYieldGrid() -- e.g. to drop an isotope that
         * only the removed channel tabulated.
         *
         * A raw pointer or reference into yieldChannels()[index], held
         * by a caller from before this call, dangles once this returns
         * -- yieldChannels_ owns each YieldChannel via unique_ptr, so
         * removing an entry destroys it. The same is true of
         * setChannels() (both overloads), which discard every existing
         * entry outright.
         */
        void deleteChannel(std::size_t index);

        /**
         * @brief Replace yieldChannels_ wholesale with a list of already-built channels
         * @param channels The channels to install, in order; ownership
         *   of each is transferred to yieldChannels_. None may be null.
         * @throws std::invalid_argument if any entry of channels is
         *   null -- checked before yieldChannels_ is touched, so a
         *   rejected call leaves the existing yieldChannels_ untouched
         * @details
         * yieldChannels_ is discarded (freeing every channel it
         * previously held) and replaced by moving channels into its
         * place -- an O(1) transfer, not a per-element addChannel()
         * loop, since every element here is already a built
         * YieldChannel, not a descriptor to build one from. Does not
         * call rebuildYieldGrid(): as with addChannel(unique_ptr<
         * YieldChannel>), a caller that wants isotopes_ resynchronized
         * onto the new channels must call it explicitly afterward.
         */
        void setChannels(std::vector<std::unique_ptr<YieldChannel>> channels);

        /**
         * @brief Replace yieldChannels_ wholesale with channels built from a list of descriptors
         * @param descriptors The descriptors to build fresh channels
         *   from, in order -- see addChannel(const
         *   YieldChannelDescriptor&)'s own comment for what each one
         *   means and how it's loaded
         * @throws std::runtime_error if any descriptor fails to load
         *   (e.g. an unknown channel/model, or a [Fe/H] range mismatch
         *   -- see the YieldChannel constructor's own comment);
         *   yieldChannels_ is left completely untouched in this case,
         *   still describing whatever channels it held before this
         *   call
         * @details
         * Every replacement channel is built fresh from disk, in
         * order, into a temporary vector -- unlike the
         * std::unique_ptr<YieldChannel> overload, which moves in
         * already-built ones -- and yieldChannels_ is only replaced
         * (via a single move) once every descriptor has succeeded, so
         * a failure partway through never leaves yieldChannels_ a mix
         * of some new channels and none of the old ones. Does not call
         * rebuildYieldGrid(): as with the other overload, a caller
         * must call it explicitly afterward to resynchronize isotopes_
         * onto the new channels.
         */
        void setChannels(const std::vector<YieldChannelDescriptor>& descriptors);

        /**
         * @brief Rebuild isotopes_ from yieldChannels_, then push it back into every channel
         * @param isotopes Restricts isotopes_ to its own intersection
         *   with this list; an empty list (the default) means "keep
         *   every isotope any loaded channel tabulates" -- see @details
         * @throws std::invalid_argument if propagated from some
         *   channel's own rebuildYieldGrid() call (should not happen in
         *   practice: the mMin/mMax passed to each are exactly what
         *   that channel's own descriptor already validated, indirectly,
         *   the first time it was built)
         * @throws std::runtime_error if isotopes is non-empty but
         *   matches nothing any loaded channel tabulates, leaving
         *   isotopes_ empty -- see @details
         * @details
         * First, collects every entry in yieldChannels_'s own
         * isotopesOrig() (not isotopes() -- see below) into one
         * deduplicated, sorted isotopes_: the union of every isotope
         * any loaded channel's own model tabulates, in IsotopeData's
         * own Z-then-A order, comparing the referenced IsotopeData
         * objects themselves (not reference_wrapper identity), so two
         * channels tabulating the same isotope (necessarily the same
         * global elem::isotopeTable() entry) are correctly recognized
         * as one, not kept as duplicates.
         *
         * If isotopes is non-empty, isotopes_ is then narrowed down to
         * just the entries that also appear (by the same IsotopeData
         * equality, i.e. matching (Z, A)) somewhere in isotopes --
         * letting a caller (see SimControls::readYields()'s own
         * yields.isotopes handling) restrict which isotopes actually
         * end up tabulated, even though every loaded channel's own
         * model may cover a much larger set. An isotopes entry that
         * doesn't match anything in the union is simply ignored, rather
         * than treated as an error, since it may simply be an isotope
         * no loaded channel happens to tabulate -- but if isotopes
         * itself is non-empty and none of its entries match anything
         * (isotopes_ would end up entirely empty), that throws instead:
         * every yieldChannels_ entry's own isotopesOrig() is always
         * non-empty in practice, so this can only mean the caller's own
         * isotopes list is entirely wrong, almost certainly a mistake
         * worth surfacing clearly, before it can instead reach
         * OutputManagerH5's own group-creation code as an opaque
         * zero-column HDF5 dataset failure.
         *
         * Then, for each entry i in yieldChannels_, calls
         * yieldChannels_[i]->rebuildYieldGrid(mMin, mMax, isotopes_),
         * where mMin/mMax are yieldChannels_[i]->descriptor()'s own
         * mMin_/mMax_ -- each channel caches its own descriptor (see
         * YieldChannel::descriptor()'s own comment), so this reads it
         * directly from the channel itself, independent of whatever
         * controls_.yieldChannels() looks like by the time this runs.
         * This is what actually synchronizes every channel onto the
         * same, shared isotopes_: passing it to rebuildYieldGrid()
         * remaps that channel's own yieldData_ onto isotopes_'s exact
         * order, backfilling 0 for any isotope this channel's own model
         * doesn't tabulate (see YieldChannel::rebuildYieldGrid()'s own
         * comment for the mechanics) -- reads isotopesOrig() again for
         * this reason, since isotopes() on a channel not yet
         * synchronized would still be empty, or already the same
         * (stale) list from a previous rebuildYieldGrid() call here.
         *
         * Called once by the constructor, right after every requested
         * channel has been added; also public, so a caller (e.g. from
         * Python, after addChannel() adds a further channel to an
         * already-built Yields) can rerun this to re-synchronize every
         * channel -- including ones added earlier -- onto the new,
         * larger union of isotopes, or to change which isotopes subset
         * is kept.
         */
        void rebuildYieldGrid(const IsotopeList& isotopes = {});

        /**
         * @brief Return the SimControls this Yields reads its channel descriptors from
         * @return A const reference to controls_
         */
        [[nodiscard]] auto controls() const -> const io::SimControls& { return controls_; }

        /**
         * @brief Return the name of the yield registry this Yields reads from
         * @return A const reference to registryName_
         */
        [[nodiscard]] auto registryName() const -> const std::string& { return registryName_; }

        /**
         * @brief Return the yield channels loaded so far
         * @return A const reference to yieldChannels_, in the same
         *   order as controls().yieldChannels()
         */
        [[nodiscard]] auto yieldChannels() const -> const std::vector<std::unique_ptr<YieldChannel>>&
        {
            return yieldChannels_;
        }

        /**
         * @brief Return the union of every loaded channel's own isotopesOrig()
         * @return A const reference to isotopes_: every isotope that
         *   appears in at least one of yieldChannels()'s own
         *   isotopesOrig() lists, deduplicated and sorted (by
         *   IsotopeData's own Z-then-A ordering) -- once
         *   rebuildYieldGrid() has run (always true after the
         *   constructor itself returns), this is also exactly what
         *   every yieldChannels() entry's own isotopes() equals, in the
         *   same order -- see rebuildYieldGrid()'s own comment
         */
        [[nodiscard]] auto isotopes() const -> const IsotopeList& { return isotopes_; }

        /**
         * @brief Return every channel's own yield, as one (nchannels, isotopes().size()) array
         * @param mass Stellar mass (Msun); need not satisfy every
         *   yieldChannels() entry's own hasYield(mass) -- see @details
         * @param feH [Fe/H]; need not lie within every yieldChannels()
         *   entry's own feH() range -- see @details
         * @return A pair (view, data): data is the backing storage,
         *   data.data() the origin of view; view is an mdspan of shape
         *   (yieldChannels().size(), isotopes().size()), i.e.
         *   view[i, j] is the yield (Msun) of isotopes()[j] from
         *   yieldChannels()[i], for a star of the given mass/feH
         * @throws std::runtime_error if any yieldChannels() entry's own
         *   yield() throws -- see its own comment (e.g. rebuildYieldGrid()
         *   was never called on that particular channel)
         * @details
         * Row i is yieldChannels()[i]->yield(mass, feH) if
         * yieldChannels()[i]->hasYield(mass) is true and feH lies
         * within yieldChannels()[i]->feH()'s own [front(), back()]
         * range, or left all zero otherwise -- so a mass or [Fe/H]
         * outside one channel's own range simply contributes nothing
         * from that channel, rather than every channel needing to
         * cover the same mass/[Fe/H] (e.g. a caller like
         * Galaxy::yieldsRate(double) averaging over every [Fe/H] grid
         * point the tracks are defined at, some of which may lie
         * outside a narrower yield channel's own tabulated range).
         * Each populated row is
         * isotopes().size() long, in isotopes()'s own order, once
         * rebuildYieldGrid() has synchronized every channel onto
         * isotopes() (which the constructor always does before
         * returning; see its own comment for the one way this could
         * stop being true -- addChannel() called again afterward
         * without a following rebuildYieldGrid()).
         *
         * Every call recomputes mass/feH's own yield from every channel
         * from scratch and returns freshly-allocated storage -- unlike
         * YieldChannel::yld(), this has no backing member of its own to
         * return a view into. view's own pointer is captured from
         * data.data() before data is moved into the returned pair;
         * std::vector's move constructor never reallocates (always an
         * O(1) pointer transfer), so that pointer stays valid, pointing
         * into the returned pair's own data, wherever the pair itself
         * ends up.
         */
        [[nodiscard]] auto yield(double mass, double feH) const -> std::pair<Array2D, std::vector<double>>;

        /**
         * @brief Return the sum, over every channel, of yield(mass, feH)
         * @param mass Stellar mass (Msun); see yield()'s own comment
         * @param feH [Fe/H]; see yield()'s own comment
         * @return A vector of isotopes().size() yields (Msun), in
         *   isotopes()'s own order: result[j] = sum over i of
         *   yield(mass, feH)'s own view[i, j]
         * @throws std::runtime_error under the same conditions as yield()
         * @details
         * Built by summing yield()'s own (nchannels, isotopes().size())
         * array along its first axis, rather than calling every
         * channel's own yield() a second time -- numerically identical
         * to summing "the outputs of the calls to YieldChannel::yield"
         * directly, since that is exactly what yield()'s own rows
         * already are.
         */
        [[nodiscard]] auto yieldSum(double mass, double feH) const -> std::vector<double>;

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Extinct's/Specsyn's own identical controls_ members exactly -- see either one's own comment for why. Only ever used through the same non-copyable ownership pattern (unique_ptr in SimControls's own yields_) as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
        std::string registryName_;        /**< Name of the yield registry file */
        std::vector<std::unique_ptr<YieldChannel>> yieldChannels_; /**< Yield channels built via addChannel(), one per entry in controls_.yieldChannels() -- see yieldChannels()'s own comment */
        IsotopeList isotopes_; /**< Union of every yieldChannels_ entry's own isotopesOrig(), deduplicated and sorted -- see isotopes()'s own comment */

    };

} // namespace yields

#endif // YIELDS_HPP
