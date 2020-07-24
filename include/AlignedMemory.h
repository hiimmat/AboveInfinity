#include <memory>
#include <stdexcept>

namespace AboveInfinity {

/*
 * Definitions of _aligned_malloc and _aligned_free
 * The first attempt was based on the C++17 operator new with additional alignment
 * However, that didn't work with msvc version 1924, so, to keep the code portable,
 * I decided to use this approach */
#ifdef __linux__
    #include <cstdlib>
    #define aligned_alloc(size, alignment) std::aligned_alloc(alignment, size)
    #define aligned_free                   free
#elif _WIN32
    #define aligned_alloc(size, alignment) _aligned_malloc(size, alignment)
    #define aligned_free                   _aligned_free
#endif

#include "Requirements.h"

namespace internal {

/* https://stackoverflow.com/a/45353838 */
template<auto T>
using val = std::integral_constant<std::decay_t<decltype(T)>, T>;

} // namespace internal

/*
 * This class represents a move-only aligned block of memory
 * The HeapTensor uses this class for memory allocations
 */
template<typename T>
class AlignedMemory {
    requires(std::is_object_v<T>);

private:
    std::unique_ptr<T[], internal::val<aligned_free>> _memory;
    std::size_t _size{0U};
    std::size_t _alignment{0U};

public:
    constexpr inline AlignedMemory() = default;

    inline explicit AlignedMemory(std::size_t sizeToAllocate, std::size_t memoryAlignment) :
        _size{sizeToAllocate}, _alignment{memoryAlignment} {
        if(!sizeToAllocate) throw std::invalid_argument("Allocation size can't be 0!\n");

        if(memoryAlignment && (memoryAlignment & (memoryAlignment - 1U)))
            throw std::invalid_argument("Invalid alignment given to allocate memory!\n");

        _memory = std::unique_ptr<T[], internal::val<aligned_free>>{
            static_cast<T*>(aligned_alloc(sizeToAllocate * sizeof(T), memoryAlignment))};

        if(!_memory || reinterpret_cast<uintptr_t>(_memory.get()) % memoryAlignment) throw std::bad_alloc();
    }

    inline explicit AlignedMemory(std::size_t sizeToAllocate, std::size_t memoryAlignment, T value) :
        AlignedMemory(sizeToAllocate, memoryAlignment) {
        std::fill(_memory.get(), _memory.get() + sizeToAllocate, value);
    }

    AlignedMemory(const AlignedMemory&) = delete;
    AlignedMemory& operator=(const AlignedMemory&) = delete;

    inline AlignedMemory(AlignedMemory&&) noexcept = default;
    inline AlignedMemory& operator=(AlignedMemory&&) noexcept = default;

    /* Returns a pointer to the beginning of the data */
    inline T* data() noexcept { return _memory.get(); }

    /* Returns a const pointer to the beginning of the data */
    inline const T* data() const noexcept { return _memory.get(); }

    /* Returns the size used for the allocation */
    inline std::size_t size() const noexcept { return _size; }

    /* Returns the alignment of the allocated memory */
    inline std::size_t alignment() const noexcept { return _alignment; }
};

} // namespace AboveInfinity