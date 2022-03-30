#pragma once

#include <cstring>

namespace AboveInfinity {

/**
 * Policy class that defines methods used for memory allocation, deallocation and copying performed on the CPU
 */
struct CPUMemoryPolicy {
    /**
     * Allocates a block of memory
     *
     * @param size: size of the memory that needs to be allocated
     * @return: allocated block of memory residing on the heap
     */
    static inline void* Allocate(std::size_t size) { return malloc(size); }

    /**
     * Copies the given size of bytes from src pointer to dest pointer
     *
     * @param dest: pointer to the destination array to which the content is being copied
     * @param src: pointer to the source array that's being copied
     */
    static inline void Copy(void* dest, const void* src, std::size_t size) { memcpy(dest, src, size); }

    /**
     * Releases a block of memory allocated on the heap
     *
     * @param buffer: pointer to the memory that's being released
     */
    static inline void Release(void* buffer) noexcept { free(buffer); }
};

} // namespace AboveInfinity