/**
 * @file Yields.hpp
 * @author Mark Krumholz
 * @brief A class to chain together nucleosynthetic yield channels
 * @date 2026-09-13
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef YIELDS_HPP
#define YIELDS_HPP

#include "../elem/DecayChain.hpp"
#include "../elem/ElemCommons.hpp"
#include "YieldChannel.hpp"
#include "YieldCommons.hpp"
#include <cstddef>
#include <mdspan> // NOLINT(misc-include-cleaner)
#include <memory>
#include <optional>
#include <span>
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
        // controls_ to the same referent) but not assignable (controls_
        // can't be reseated, matching Extinct's/Specsyn's own identical
        // controls_ members -- see either one's own comment).
        // Deliberately kept non-copyable too, even though
        // yieldChannels_'s own shared_ptr<YieldChannel> elements are
        // themselves freely copyable: a shallow copy would leave two
        // Yields instances sharing the same live YieldChannel objects
        // while each computes its own, independent isotopes_ from them
        // -- calling rebuildYieldGrid() on one would silently remap the
        // shared channels' own yield grids out from under the other.
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
         * yieldChannels_ owns each YieldChannel via shared_ptr, not
         * unique_ptr, so a copy of yieldChannels()[index] held by a
         * caller from before this call stays fully valid afterward --
         * it's simply no longer one of the entries yieldChannels()
         * itself returns. The same is true of setChannels() (both
         * overloads), which discard every existing entry outright.
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
         * previously held whose shared_ptr use_count drops to zero --
         * see yieldChannels()'s own comment on why an entry can outlive
         * this call) and replaced with channels, converting each
         * incoming unique_ptr<YieldChannel> to the shared_ptr this
         * class actually stores them as -- not a per-element
         * addChannel() loop, since every element here is already a
         * built YieldChannel, not a descriptor to build one from. Does
         * not call rebuildYieldGrid(): as with addChannel(unique_ptr<
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
         * @param isotopes The isotopes the caller wants; isotopes_ is
         *   narrowed to these plus whatever decay-chain context they
         *   need (see @details). An empty list (the default) means
         *   "keep every isotope any loaded channel tabulates"
         * @throws std::invalid_argument if propagated from some
         *   channel's own rebuildYieldGrid() call (should not happen in
         *   practice: the mMin/mMax passed to each are exactly what
         *   that channel's own descriptor already validated, indirectly,
         *   the first time it was built)
         * @throws std::runtime_error if isotopes is non-empty but
         *   matches nothing any loaded channel tabulates (or that is
         *   a decay product of something one tabulates), leaving
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
         * isotopes_ is then force-expanded: for every unstable entry,
         * every isotope named by its own daughters() is resolved (via
         * elem::isotopeTable()) and added if not already present, and
         * this repeats until a full pass adds nothing further (a newly
         * added daughter can itself be unstable, with daughters() of its
         * own -- e.g. Ni56 -> Co56 -> Fe56, where only Ni56 and Fe56 are
         * directly tabulated by some loaded channel, but Co56 must still
         * be tracked, or its share of Ni56's own decayed mass would
         * simply vanish when applyDecay() runs). Re-sorted/deduplicated
         * the same way as the initial union, once this closure is
         * complete.
         *
         * If isotopes is non-empty, isotopes_ is then narrowed to the
         * requested isotopes plus their decay-chain context: it is
         * *expanded*, not simply intersected with isotopes, so an
         * explicit request can never leave the decay network broken.
         * Applied to the decay-closed list above, this keeps
         * (1) every entry that also appears (by IsotopeData equality,
         * i.e. matching (Z, A)) in isotopes; (2) every entry that
         * decays, directly or through intermediates, into one of
         * those; and (3) every decay product, direct or indirect, of
         * anything kept by (1) or (2). Everything else is dropped.
         * In (2), an emitted proton or alpha (H1 or He4, which the
         * isotope data lists as daughters of every proton or alpha
         * emitter alongside the heavy daughter nuclide) does not
         * count as a decay link, so requesting H1 or He4 does not pull
         * in every emitter of them; in (3) they do count, so a kept
         * alpha emitter brings He4 along, as elem::DecayChain
         * requires.
         * So, for example, requesting Fe56 from models that tabulate
         * Ni56 also keeps Ni56 and Co56 (rule 2: they decay into Fe56,
         * and the requested Fe56 yield includes their contribution),
         * and requesting Ni56 alone also keeps Co56 and Fe56 (rule 3:
         * its decay products, which elem::DecayChain requires be
         * present). Rule 3 is not fed back into rule 2: an isotope
         * kept only as a decay product does not pull in its own other
         * parents, so its yield can omit decay contributions from
         * parents that were neither requested nor on a chain leading
         * to a requested isotope. Where it matters, request such an
         * isotope explicitly.
         *
         * An isotopes entry that doesn't match anything in the
         * (force-expanded) union is simply ignored, rather than
         * treated as an error, since it may simply be an isotope no
         * loaded channel happens to tabulate -- but if isotopes itself
         * is non-empty and none of its entries match anything
         * (isotopes_ would end up entirely empty), that throws
         * instead: every yieldChannels_ entry's own isotopesOrig() is
         * always non-empty in practice, so this can only mean the
         * caller's own isotopes list is entirely wrong, almost
         * certainly a mistake worth surfacing clearly, before it can
         * instead reach OutputManagerH5's own group-creation code as
         * an opaque zero-column HDF5 dataset failure.
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
         * Force-expansion may have added isotopes no loaded channel
         * tabulates at all (e.g. Co56 above); those simply get an
         * all-zero row from every channel, exactly like any other
         * isotope a particular channel's own model doesn't cover.
         *
         * Finally, decayChain_ is rebuilt from scratch to match the
         * now-final isotopes_: a single elem::DecayChain, constructed
         * from the whole of isotopes_ at once -- see applyDecay()'s own
         * comment for how yield()/yieldSum() actually use it.
         *
         * Called once by the constructor, right after every requested
         * channel has been added; also public, so a caller (e.g. from
         * Python, after addChannel() adds a further channel to an
         * already-built Yields) can rerun this to re-synchronize every
         * channel -- including ones added earlier -- onto the new,
         * larger union of isotopes, or to change which isotopes are
         * kept.
         */
        void rebuildYieldGrid(const elem::IsotopeList& isotopes = {});

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
         *   order as controls().yieldChannels(). Each element is a
         *   shared_ptr, not a unique_ptr: a caller (e.g. from Python)
         *   that copies one out and retains it keeps that particular
         *   YieldChannel fully valid even if a later deleteChannel()/
         *   setChannels() call removes it from this Yields -- it's
         *   simply no longer one of the entries this method itself
         *   returns.
         */
        [[nodiscard]] auto yieldChannels() const -> const std::vector<std::shared_ptr<YieldChannel>>&
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
        [[nodiscard]] auto isotopes() const -> const elem::IsotopeList& { return isotopes_; }

        /**
         * @brief Return every channel's own yield, as one (nchannels, isotopes().size()) array
         * @param mass Stellar mass (Msun); need not satisfy every
         *   yieldChannels() entry's own hasYield(mass) -- see @details
         * @param feH [Fe/H]; need not lie within every yieldChannels()
         *   entry's own feH() range -- see @details
         * @param dtDecay Elapsed time, in yr, over which to apply
         *   radioactive decay to the raw per-channel yields, before
         *   returning them; 0 (the default) applies no decay at all.
         *   Ignored entirely if controls().noDecay() is true -- see
         *   @details
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
         * If controls().noDecay() is false (the default), applyDecay()
         * is then called once per row, in place, with dtDecay -- see
         * its own comment for exactly what this does. If
         * controls().noDecay() is true, dtDecay is ignored and every
         * row is left exactly as the raw yield tables report it: the
         * cumulative amount of each isotope ever produced, regardless
         * of any radioactive decay since.
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
        [[nodiscard]] auto yield(double mass, double feH, double dtDecay = 0.0) const
            -> std::pair<Array2D, std::vector<double>>;

        /**
         * @brief Return the sum, over every channel, of yield(mass, feH)
         * @param mass Stellar mass (Msun); see yield()'s own comment
         * @param feH [Fe/H]; see yield()'s own comment
         * @param dtDecay Elapsed time over which to apply radioactive
         *   decay -- see yield()'s own comment for exactly what this
         *   means and when it's ignored; applied once, to the summed
         *   total, rather than once per channel -- see @details
         * @return A vector of isotopes().size() yields (Msun), in
         *   isotopes()'s own order: result[j] = sum over i of
         *   yield(mass, feH, 0)'s own view[i, j], decayed by dtDecay
         *   (unless controls().noDecay() is true)
         * @throws std::runtime_error under the same conditions as yield()
         * @details
         * Built by summing yield(mass, feH, 0)'s own raw (undecayed --
         * see its own comment for why passing 0 here suffices)
         * (nchannels, isotopes().size()) array along its first axis,
         * rather than calling every channel's own yield() a second
         * time, then applying decay once (if controls().noDecay() is
         * false) to that summed total via applyDecay(), rather than
         * once per channel: decay is linear in each isotope's own mass
         * (DecayChain::applyDecay()'s matrix exponential acts
         * identically on any input vector, regardless of which channel
         * a given mass came from), so summing first and then decaying
         * once is numerically identical to decaying every channel's
         * own row and then summing -- just cheaper, since the matrix
         * exponential itself is only ever computed once per
         * yieldSum() call this way, not once per channel.
         */
        [[nodiscard]] auto yieldSum(double mass, double feH, double dtDecay = 0.0) const -> std::vector<double>;

        /**
         * @brief Apply radioactive decay, over dtDecay, to one array of per-isotope masses, in place
         * @param dtDecay Elapsed time, in yr, to advance values by
         * @param values One mass (Msun) per entry of isotopes_, in the
         *   same order; overwritten in place with the post-decay masses
         * @details
         * A thin wrapper around decayChain_'s own applyDecay(): decayChain_
         * is a single elem::DecayChain built from the whole of isotopes_
         * (see rebuildYieldGrid()'s own comment), so values is already
         * laid out in exactly the order decayChain_ expects, with no
         * further per-isotope resolution needed at this level at all --
         * unlike the old per-isotope-DecayChain design this replaced,
         * there is no possibility of a decay product isotopes_ doesn't
         * track: rebuildYieldGrid() -- see its own comment -- keeps
         * isotopes_ closed under decay, both when force-expanding the
         * isotopes the channels tabulate and when expanding a
         * caller-supplied isotope list, so every decay product of a
         * retained isotope is always present.
         *
         * Public (unlike yield()/yieldSum()'s own internal use of it)
         * so that a caller already holding a raw per-isotope array it
         * knows is undecayed -- e.g. Galaxy's own fieldYields_, or a
         * field star's instantaneous mass-return rate -- can apply the
         * same decay physics to it directly, without needing to
         * reconstruct a (mass, feH) query this Yields itself already
         * evaluated.
         */
        void applyDecay(double dtDecay, std::span<double> values) const;

        /**
         * @brief Apply radioactive decay in place to a possibly channel-decomposed per-isotope array
         * @param dtDecay Elapsed time, in yr, to apply -- see the
         *   other applyDecay() overload's own comment
         * @param values One entry per isotope (isotopes().size()), or
         *   one channel-major row of isotopes().size() entries per
         *   channel (yieldChannels().size() * isotopes().size() total,
         *   in yield()'s own (nchannels, niso) layout) if decomposed;
         *   overwritten in place with the post-decay masses, exactly
         *   like the other overload
         * @param decomposed Whether values is channel-decomposed
         * @details
         * The other applyDecay() overload only ever accepts a single,
         * isotopes().size()-length array -- yield()/yieldSum() apply
         * it once per channel row internally for exactly this reason
         * (decay is evaluated per isotope, not per channel). This
         * overload exists for callers -- Cluster::yields_/
         * Galaxy::fieldYields_, and yieldsRate()'s own result, each of
         * which can be either shape depending on
         * controls().yieldsChannelDecomposed() -- that hold an array
         * whose own shape isn't known to be one or the other until
         * runtime: splits a decomposed array into its own channel-major
         * rows and applies the other overload to each in turn, or
         * simply forwards to it unchanged if not decomposed.
         */
        void applyDecay(double dtDecay, std::span<double> values, bool decomposed) const;

    private:

        const io::SimControls& controls_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) -- deliberately a live reference, not a copy, matching Extinct's/Specsyn's own identical controls_ members exactly -- see either one's own comment for why. Only ever used through the same shared_ptr ownership pattern (shared_ptr<Yields> in SimControls's own yields_) as those, so the usual objection (disabling implicit copy/move assignment) doesn't apply in practice.
        std::string registryName_;        /**< Name of the yield registry file */
        std::vector<std::shared_ptr<YieldChannel>> yieldChannels_; /**< Yield channels built via addChannel(), one per entry in controls_.yieldChannels() -- see yieldChannels()'s own comment */
        elem::IsotopeList isotopes_; /**< Union of every yieldChannels_ entry's own isotopesOrig(), deduplicated and sorted -- see isotopes()'s own comment */
        std::optional<elem::DecayChain> decayChain_; /**< Built from the whole of isotopes_ at once -- (re)populated by rebuildYieldGrid(), see its own comment; used by applyDecay(). optional (rather than a plain value) because DecayChain has no default constructor, but decayChain_ must be re-buildable in place whenever rebuildYieldGrid() reruns */

    };

} // namespace yields

#endif // YIELDS_HPP
