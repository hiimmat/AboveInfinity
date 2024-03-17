#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "aligned_allocator.hpp"
#include "concepts.hpp"

namespace ntensor {

/*
 * Dense buffer with compile-time policy for allocating/deallocating memory
 * Parameters:
 * @tparam Allocator: allocator class with methods used for allocating/deallocating memory
 * @tparam T: underlying memory type of the buffer
 * Constraints:
 * The Allocator type has to satisfy the allocator concept
 */
template <arithmetic T, typename Allocator = nt_allocator<T>>
  requires allocator<Allocator, T>
class DenseBuffer {
 private:
  std::shared_ptr<Allocator> _allocator{};
  std::shared_ptr<T[]> _memory{nullptr};
  std::size_t _size{};

 public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using value_type = T;

  /*
   * Default constructor
   */
  DenseBuffer() noexcept = default;

  /*
   * Allocates memory for the specified number of elements
   * Parameters:
   * @param size: number of elements for which the memory needs to be allocated
   */
  explicit DenseBuffer(std::size_t size) : _size{size} {
#ifdef ENABLE_NT_EXPECTS
    Expects(size > 0u && size < std::numeric_limits<std::size_t>::max() / sizeof(T));
#endif

    _allocator = std::make_shared<Allocator>();
    _memory = std::allocate_shared_for_overwrite<T[]>(*_allocator, size);

#ifdef ENABLE_NT_EXPECTS
    Expects(_memory && !(reinterpret_cast<uintptr_t>(_memory.get()) % NT_ALIGNMENT));
#endif
  }

  /*
   * Copy constructor
   */
  DenseBuffer(const DenseBuffer& other) : _allocator{other._allocator}, _memory{other._memory}, _size{other._size} {}

  /*
   * Copy assignment operator
   */
  DenseBuffer& operator=(const DenseBuffer& other) {
    if (this != &other) [[likely]] {
      _allocator = other._allocator;
      _memory = other._memory;
      _size = other._size;
    }
    return *this;
  }

  /*
   * Move constructor
   */
  DenseBuffer(DenseBuffer&& other) noexcept
      : _allocator{std::move(other._allocator)}, _memory{std::move(other._memory)}, _size{other._size} {
    other._size = 0u;
  }

  /*
   * Move assignment operator
   */
  DenseBuffer& operator=(DenseBuffer&& other) {
    if (this != &other) [[likely]] {
      _allocator = std::move(other._allocator);
      _memory = std::move(other._memory);
      _size = other._size;
      other._size = 0u;
    }
    return *this;
  }

  /*
   * Compares two dense buffers for equality
   * Parameters:
   * param lhs: first (left-hand side) dense buffer
   * param rhs: second (right-hand side) dense buffer
   * True if the buffers are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const DenseBuffer& lhs, const DenseBuffer& rhs) noexcept {
    return lhs._memory == rhs._memory && lhs._size == rhs._size;
  }

  /*
   * Returns a pointer to the allocated memory
   * Parameters:
   * @return: pointer to the allocated memory
   */
  [[nodiscard]] inline pointer data() { return std::assume_aligned<NT_ALIGNMENT>(_memory.get()); }

  /*
   * Returns a const pointer to the allocated memory
   * Parameters:
   * @return: const pointer to the allocated memory
   */
  [[nodiscard]] inline const_pointer data() const { return std::assume_aligned<NT_ALIGNMENT>(_memory.get()); }

  /*
   * Returns a reference to the element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: reference to the element at the specified location
   */
  [[nodiscard]] inline reference operator[](std::size_t index) { return _memory[index]; }

  /*
   * Returns a const reference to the element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: const reference to the element at the specified location
   */
  [[nodiscard]] inline const_reference operator[](std::size_t index) const { return _memory[index]; }

  /*
   * Returns a reference to the element at the specified location, with optional bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: reference to the element at the specified location
   */
  [[nodiscard]] inline reference at(std::size_t index) {
#ifdef ENABLE_NT_EXPECTS
    Expects(index < _size);
#endif
    return _memory[index];
  }

  /*
   * Returns a const reference to the element at the specified location, with optional bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: const reference to the element at the specified location
   */
  [[nodiscard]] inline const_reference at(std::size_t index) const {
#ifdef ENABLE_NT_EXPECTS
    Expects(index < _size);
#endif
    return _memory[index];
  }

  /*
   * Returns the number of elements found in the buffer
   * Parameters:
   * @return: number of elements found in the buffer
   */
  [[nodiscard]] inline std::size_t size() const noexcept { return _size; }
};

}  // namespace ntensor
