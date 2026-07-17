/**
 * @file UniqueIDManager.hpp
 * @author Mark Krumholz
 * @brief Implements thread-safe unique ID generation
 * @date 2026-07-17
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
         *   returned by any other call to get() on this object
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

    private:

        unsigned long uniqueID_ = 0; /**< Next ID to be handed out */

    };

    /**
     * @brief Return the next ID from the global, thread-safe ID generator
     * @details
     * The UniqueIDManager instance is a function-local static, so it is
     * constructed on first use rather than before main() begins. Being
     * a local static in an inline function also guarantees a single,
     * program-wide instance, rather than one per translation unit.
     */
    inline auto getID() -> unsigned long
    {
        static UniqueIDManager instance;
        return instance.get();
    }

} // namespace utils

#endif // UNIQUEIDMANAGER_HPP
