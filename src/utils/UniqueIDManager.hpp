/**
 * @file UniqueIDManager.hpp
 * @author Mark Krumholz
 * @brief Implements thread-safe unique ID generation
 * @date 2026-07-17
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef UNIQUEIDMANAGER_HPP
#define UNIQUEIDMANAGER_HPP

namespace utils {

    /**
     * @class UniqueIDManager
     * @brief A thread-safe generator of sequential unique ID numbers
     */
    class UniqueIDManager {

    public:

        /**
         * @brief Construct a UniqueIDManager, with the first ID it
         *   hands out being 0
         */
        UniqueIDManager() = default;

        ~UniqueIDManager() = default;

        // Disable copy and move constructors and assignment operators,
        // since this class is meant to be used only via a single,
        // shared instance
        UniqueIDManager(const UniqueIDManager&) = delete;
        UniqueIDManager(UniqueIDManager&&) = delete;
        auto operator=(const UniqueIDManager&) -> UniqueIDManager& = delete;
        auto operator=(UniqueIDManager&&) -> UniqueIDManager& = delete;

        /**
         * @brief Return the next unique ID
         * @return An ID that has never been, and will never again be,
         *   returned by any other call to get() on this object -- so
         *   long as set() is never used to rewind uniqueID_ back past
         *   a value already handed out this way (see set()'s own
         *   comment for the one caller that does use it, and why that
         *   caller's own use is still safe)
         * @details
         * This routine properly handles inter-thread synchronization
         * when openMP is enabled: uniqueID_ is read and incremented as
         * a single atomic operation, so no two calls -- whether from
         * the same thread or different ones -- can ever be handed the
         * same value.
         */
        [[nodiscard]] auto get() -> unsigned long
        {
            unsigned long id{};
#ifdef _OPENMP
#pragma omp atomic capture
#endif
            { id = uniqueID_; uniqueID_++; }
            return id;
        }

        /**
         * @brief Set the next ID get() will hand out
         * @param value The next ID get() should return
         * @details
         * Atomic, like get() itself, but this alone does not make
         * calling this concurrently with get() meaningful -- which of
         * the two "wins" (i.e. whether a get() racing this set() sees
         * the value before or after it) is unspecified. The one
         * caller that actually uses this,
         * OutputManagerH5::restartSetup(), only ever calls it once, on
         * a single thread, before any thread has had a chance to call
         * get() again (mirroring the same "only ever called from
         * outside any active parallel region" rule checkpoint() itself
         * relies on -- see its own comment) -- restoring uniqueID_ to
         * the "restart_uid" value read back from the checkpoint being
         * resumed from, i.e. the value get() was about to hand out
         * next when that checkpoint was closed, so this process's own
         * IDs continue exactly where the run being resumed left off.
         * Any IDs a since-abandoned, never-fully-written checkpoint's
         * own in-flight trials had already been handed simply become
         * gaps in the sequence, not reused -- get()'s own "never
         * returns the same value twice" guarantee still holds.
         */
        void set(const unsigned long value)
        {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            uniqueID_ = value;
        }

        /**
         * @brief Return the next ID get() would hand out, without handing it out
         * @return The value get() would return if called right now --
         *   i.e. get() without its own increment
         * @details
         * Atomic, like get() itself. Used by
         * OutputManagerH5::closeOutputFile() to record the next ID
         * about to be handed out as each checkpoint's own "restart_uid"
         * attribute, without itself consuming that ID (calling get()
         * instead would waste one ID per checkpoint for no reason, and
         * -- worse -- record the ID *after* the one actually still
         * next, since get()'s own increment would already have run by
         * the time its return value was written out).
         */
        [[nodiscard]] auto read() const -> unsigned long
        {
            unsigned long id{};
#ifdef _OPENMP
#pragma omp atomic read
#endif
            id = uniqueID_;
            return id;
        }

    private:

        unsigned long uniqueID_ = 0; /**< Next ID to be handed out */

    };

    /**
     * @brief Return the global, thread-safe ID generator
     * @details
     * The UniqueIDManager instance is a function-local static, so it is
     * constructed on first use rather than before main() begins. Being
     * a local static in an inline function also guarantees a single,
     * program-wide instance, rather than one per translation unit.
     * Returns the instance itself, rather than calling get() on
     * callers' behalf, so callers needing set()/read() (currently only
     * OutputManagerH5) can reach those too, through the same single,
     * shared instance -- ordinary ID generation is still just
     * uniqueID().get().
     */
    inline auto uniqueID() -> UniqueIDManager&
    {
        static UniqueIDManager instance;
        return instance;
    }

} // namespace utils

#endif // UNIQUEIDMANAGER_HPP
