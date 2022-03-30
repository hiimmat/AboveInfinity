#pragma once

#include "CPUMemoryPolicy.h"

#include <atomic>
#include <cstring>
#include <type_traits>

namespace AboveInfinity {

/**
 * Class representing a buffer with reference counting and memory allocation/deletion defined through a policy
 *
 * @tparam T: Type of the underlying buffer
 * @tparam MemoryPolicy: Policy with methods for memory allocation and deletion
 */
template<typename T, typename MemoryPolicy = CPUMemoryPolicy>
class PolicyBuffer {
    static_assert(std::is_arithmetic_v<T>);

protected:
    T* _buffer{nullptr};
    std::atomic_uint* _refCount{nullptr};
    std::size_t _size{};

    /**
     * Helper function performing a shallow copy of another PolicyBuffer
     *
     * @param other: PolicyBuffer object that's being copied
     */
    inline void ShallowCopy(const PolicyBuffer& other) noexcept {
        this->_buffer = other._buffer;
        this->_refCount = other._refCount;
        this->_size = other._size;
        if(_refCount) ++(*this->_refCount);
    }

    /**
     * Helper function that cleans up the custom allocated members of the object
     */
    inline void Release() noexcept {
        if(_refCount) {
            if(*_refCount == 1) {
                MemoryPolicy::Release(_buffer);
                delete _refCount;
            } else {
                --(*_refCount);
            }
        }
    }

public:
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using difference_type = std::ptrdiff_t;

    /**
     * Default constructor
     */
    PolicyBuffer() = default;

    /**
     * Constructs a new PolicyBuffer with the requested size
     *
     * @param size: the size to allocate
     */
    explicit PolicyBuffer(std::size_t size) : _size(size) {
        _buffer = static_cast<value_type*>(MemoryPolicy::Allocate(_size));
        _refCount = new std::atomic_uint(1u);
    }

    /**
     * Copy constructor (performs shallow copy)
     *
     * @param other: the object that's being copied
     */
    inline PolicyBuffer(const PolicyBuffer& other) noexcept { ShallowCopy(other); }

    /**
     * Move constructor
     *
     * @param other: the object that's being moved
     */
    inline PolicyBuffer(PolicyBuffer&& other) noexcept { swap(*this, other); }

    /**
     * Copy operator (performs shallow copy)
     *
     * @param other: Object that's being copied
     * @return: new PolicyBuffer instance
     */
    inline PolicyBuffer& operator=(const PolicyBuffer& other) noexcept {
        if(this != &other) {
            Release();
            ShallowCopy(other);
        }

        return *this;
    }

    /**
     * Move operator
     *
     * @param other: Object that's being moved
     * @return: new PolicyBuffer instance
     */
    inline PolicyBuffer& operator=(PolicyBuffer&& other) noexcept {
        if(this != &other) swap(*this, other);

        return *this;
    }

    /**
     * Copies the allocated memory from another buffer into this buffer
     * 
     * @param other: PolicyBuffer whose underlying allocated memory is copied
     */
    void Copy(const PolicyBuffer& other) {
        if(this->_size != other._size) throw std::runtime_error("Buffer sizes aren't equal");

        MemoryPolicy::Copy(_buffer, other._buffer, _size);
    }

    /**
     * Performs a deep copy of the passed input buffer
     *
     * @param other: PolicyBuffer that's being cloned
     */
    void Clone(const PolicyBuffer& other) {
        Release();

        _size = other._size;
        _buffer = static_cast<value_type*>(MemoryPolicy::Allocate(_size));
        _refCount = new std::atomic_uint(1u);
        MemoryPolicy::Copy(_buffer, other._buffer(), _size);
    }

    /**
     * Returns a const pointer to the specified element
     * 
     * @param index: index of the specified element
     * @return: pointer to the specified element
     */
    inline const_pointer At(std::size_t index) const { return _buffer + index; }

    /**
     * Returns a const pointer to the specified element
     * 
     * @tparam index: index of the specified element 
     * @return: pointer to the specified element
     */
    template<std::size_t index>
    inline const_pointer At() const {
        return _buffer + index;
    }

    /**
     * Changes the value of the specified element
     * 
     * @param index: index of the specified element
     * @param value: new value of the specified element
     */
    inline void Set(std::size_t index, value_type value) { _buffer[index] = value; }

    /**
     * Changes the value of the specified element
     * 
     * @tparam index: index of the specified element
     * @param value: new value of the specified element
     */
    template<std::size_t index>
    void Set(value_type value) {
        _buffer[index] = value;
    }

    /**
     * Checks if the buffer is allocated
     */
    inline bool Empty() const noexcept { return _buffer == nullptr; }

    /**
     * Retrieves the buffers' allocated size
     *
     * @return: buffers' allocated size
     */
    inline std::size_t Size() const noexcept { return _size; }

    /**
     * Retrieves the number of PolicyBuffer objects referring to the same allocated memory
     *
     * @return: number of PolicyBuffer objects referring to the same allocated memory
     */
    inline int RefCount() const noexcept { return *_refCount; }

    /**
     * Swaps two PolicyBuffers
     *
     * @param first: lhs object that gets swapped
     * @param second: rhs object that gets swapped
     */
    inline friend void swap(PolicyBuffer& first, PolicyBuffer& second) noexcept {
        std::swap(first._buffer, second._buffer);
        std::swap(first._refCount, second._refCount);
        std::swap(first._size, second._size);
    }

    /**
     * Deallocates the allocated memory if no PolicyBuffers link to it
     */
    virtual ~PolicyBuffer() noexcept { Release(); }
};

} // namespace AboveInfinity