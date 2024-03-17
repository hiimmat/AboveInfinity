#pragma once

#include <iostream>

#include "dimensions.hpp"
#include "strides.hpp"

namespace ntensor {

/*
 * Planes represent a building block for Tensors, as they determine their underlying structure
 * Thanks for this structure, Tensors can be interleaved, semi-interleaved, or planar
 * They can have different dimensions and strides for each plane (for example, a plane can be have double or half the
 * size of the previous plane) And they can either reuse the same buffer across the planes, or make use of several
 * buffers instead At the same time, there are no restrictions for the buffer, so the buffer can be dense or sparse,
 * allocated on the stack, heap, etc. This allows Tensors to support many more distinct memory formats compared to a
 * basic multidimensional array Parameters:
 * @tparam Buffer: Type of the buffer that will represent the plane's data
 * @tparam _dimensions: plane's dimensions
 * @tparam _strides: plane's strides
 * @tparam _channels: plane's channels (if there's more than 1 channel, the plane is interleaved)
 * Constraints:
 * The Buffer type has to satisfy the buffer concept
 * Dimensions and strides have to have the same rank (size)
 * Each plane has to have at least 1 channel
 */
template <buffer Buffer, Dimensions _dimensions, Strides _strides, std::size_t _channels = 1u>
  requires(_dimensions.rank() == _strides.rank() && _channels > 0u)
class Plane {
 private:
  Buffer _buffer;
  long long _offset{};

 public:
  using type = Plane<Buffer, _dimensions, _strides, _channels>;
  using buffer_type = Buffer;

  using size_type = typename Buffer::size_type;
  using difference_type = typename Buffer::difference_type;
  using pointer = typename Buffer::pointer;
  using const_pointer = typename Buffer::const_pointer;
  using reference = typename Buffer::reference;
  using const_reference = typename Buffer::const_reference;
  using value_type = typename Buffer::value_type;

  /*
   * Default constructor
   */
  Plane() noexcept = default;

  /*
   * Creates a plane with a given buffer and an offset
   * Parameters:
   * @param buffer: buffer representing the plane's memory
   * @param offset: offset needed to access the first plane element in the buffer
   * Constraints:
   * The smallest and the largest buffer index have to be inside the bounds of 0u and buffer size
   */
  explicit Plane(const Buffer& buffer, long long offset = 0) : _buffer{buffer}, _offset{offset} {
#ifdef ENABLE_NT_ENSURES
    // Check the min and max index that we use to access the buffer, to ensure that we can't access the buffer out of
    // bounds

    // Use channels in the min_buffer_index computation only if it can further reduce the minimum index
    const long long min_computed_index = min_index(_dimensions, _strides);
    const long long min_buffer_index =
        min_computed_index >= 0 ? min_computed_index + offset : min_computed_index * _channels + offset;
    Ensures(min_buffer_index >= 0 && static_cast<std::size_t>(min_buffer_index) <= _buffer.size());

    // Since max_buffer_index has to be positive, multiplying with channels can either have no effect, or a positive
    // effect on the max buffer index computation
    const long long max_buffer_index = max_index(_dimensions, _strides) * _channels + offset;
    Ensures(max_buffer_index >= 0 && static_cast<std::size_t>(max_buffer_index) <= _buffer.size());
#endif
  }

  /*
   * Creates a plane with a given buffer and an offset
   * Parameters:
   * @param buffer: buffer representing the plane's memory
   * @param offset: offset needed to access the first plane element in the buffer
   * Constraitns:
   * The smallest and the largest buffer index have to be inside the bounds of 0u and buffer size
   */
  explicit Plane(Buffer&& buffer, long long offset = 0) : _buffer{std::move(buffer)}, _offset{offset} {
    // Check the min and max index that we use to access the buffer, to ensure that we can't access the buffer out of
    // bounds Use channels in the min_buffer_index computation only if it can further reduce the minimum index
    const long long min_computed_index = min_index(_dimensions, _strides);
    const long long min_buffer_index =
        min_computed_index >= 0 ? min_computed_index + offset : min_computed_index * _channels + offset;
#ifdef ENABLE_NT_ENSURES
    Ensures(min_buffer_index >= 0u && min_buffer_index <= _buffer.size());
#endif

    // Since max_buffer_index has to be positive, multiplying with channels can either have no effect, or a positive
    // effect on the max buffer index computation
    const long long max_buffer_index = max_index(_dimensions, _strides) * _channels + offset;
#ifdef ENABLE_NT_ENSURES
    Ensures(max_buffer_index >= 0u && max_buffer_index <= _buffer.size());
#endif
  }

  /*
   * Copy constructor
   */
  Plane(const Plane& other) : _buffer{other._buffer}, _offset{other._offset} {}

  /*
   * Copy assignment operator
   */
  Plane& operator=(const Plane& other) {
    if (this != &other) [[likely]] {
      _buffer = other._buffer;
      _offset = other._offset;
    }

    return *this;
  }

  /*
   * Move constructor
   */
  Plane(Plane&& other) noexcept : _buffer{std::move(other._buffer)}, _offset{other._offset} { other._offset = 0; }

  /*
   * Move assignment operator
   */
  Plane& operator=(Plane&& other) noexcept {
    if (this != &other) [[likely]] {
      _buffer = std::move(other._buffer);
      _offset = other._offset;
      other._offset = 0;
    }

    return *this;
  }

  /*
   * Compares two planes for equality
   * Parameters:
   * param lhs: first (left-hand side) plane
   * param rhs: second (right-hand side) plane
   * True if the planes are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const Plane& lhs, const Plane& rhs) noexcept {
    return lhs._buffer == rhs._buffer && lhs._offset == rhs._offset;
  }

  /*
   * Creates a copy of this plane with the same underlying structure (same buffer), but with a possibly different
   * representation (dimensions, strides, channels, offset)
   * @tparam dimensions: dimensions used to create the plane
   * @tparam strides: strides used to create the plane
   * @tparam channels: channels used to create the plane
   * @param offset: offset added to/subtracted from the current offset. Affects only the newly created plane
   * @return: copy of this plane with a possibly different representation
   */
  template <Dimensions dimensions = _dimensions, Strides strides = _strides, std::size_t channels = _channels>
  auto like(long long offset = 0u) const {
    // If preconditions are enabled, all of the parameters passed to this function are going to be validated in the
    // Plane constructor, in order to assure that we can't attempt an out of bounds memory access.
    // Validating these members here separately would be more restrictive, and would not yield any additional benefits.

    return Plane<Buffer, dimensions, strides, channels>{_buffer, _offset + offset};
  }

  /*
   * Returns the plane's dimensions
   * Parameters:
   * @return: plane's dimensions
   */
  [[nodiscard]] static consteval auto dimensions() noexcept { return _dimensions; }

  /*
   * Returns the plane's strides
   * Parameters:
   * @return: plane's strides
   */
  [[nodiscard]] static consteval auto strides() noexcept { return _strides; }

  /*
   * Returns the plane's channels
   * Parameters:
   * @return: plane's channels
   */
  [[nodiscard]] static consteval auto channels() noexcept { return _channels; }

  /*
   * Returns the plane's buffer offset
   * Parameters:
   * @return: plane's buffer offset
   */
  [[nodiscard]] inline long long offset() const noexcept { return _offset; }

  /*
   * Returns the plane's rank
   */
  [[nodiscard]] static consteval std::size_t rank() noexcept { return _dimensions.rank(); }

  /*
   * Returns a reference to the plane's element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: reference to the plane's element at the specified location
   * Restrictions:
   * The plane's buffer has to have the subscript operator implemented, and the subscript operator has to return a
   * reference
   */
  reference operator[](std::size_t index)
    requires requires {
      { _buffer[0u] } -> std::same_as<typename Buffer::reference>;
    }
  {
    return _buffer[index + _offset];
  }

  /*
   * Returns a const reference to the plane's element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: const reference to the plane's element at the specified location
   * Restrictions:
   * The plane's buffer has to have the subscript operator implemented, and the subscript operator has to return a const
   * reference
   */
  const_reference operator[](std::size_t index) const
    requires requires {
      { _buffer[0u] } -> std::same_as<typename Buffer::const_reference>;
    }
  {
    return _buffer[index + _offset];
  }

  /*
   * Returns a reference to the plane's element at the specified location, with bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: reference to the plane's element at the specified location
   * Restrictions:
   * The plane's buffer has to have the method at(std::size_t) implemented, and it has to return a reference
   */
  reference at(std::size_t index)
    requires requires {
      { _buffer.at(0u) } -> std::same_as<typename Buffer::reference>;
    }
  {
    return _buffer.at(index + _offset);
  }

  /*
   * Returns a const reference to the plane's element at the specified location, with bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: const reference to the plane's element at the specified location
   * Restrictions:
   * The plane's buffer has to have the method at(std::size_t) implemented, and it has to return a const reference
   */
  const_reference at(std::size_t index) const
    requires requires {
      { _buffer.at(0u) } -> std::same_as<typename Buffer::const_reference>;
    }
  {
    return _buffer.at(index + _offset);
  }

  /*
   * Returns the effective memory size (buffer size that can be used by the plane)
   * This is mainly useful for planes that share the same memory
   * In that case, planes have a defined offset that the buffer needs to skip to access that plane's memory
   * Parameters:
   * @return: plane's effective size
   */
  [[nodiscard]] inline std::size_t effective_size() const noexcept { return _buffer.size() - std::abs(_offset); }

  /*
   * Returns the total allocated buffer size
   * Parameters:
   * @return: total allocated size
   */
  [[nodiscard]] inline std::size_t real_size() const noexcept { return _buffer.size(); }
};

/*
 * Helper method used for creating a plane
 * Parameters:
 * @tparam BufferType: Type of the underlying buffer
 * @tparam dimensions: Dimensions of the newly created plane
 * @tparam channels: Number of channels fo the newly created plane
 * @tparam aligned_strides: Variable determining whether the plane should have aligned or unaligned strides
 */
template <typename BufferType, Dimensions dimensions, std::size_t channels = 1u, bool aligned_strides = true>
[[nodiscard]] auto create_plane(long long offset = 0) {
  if constexpr (aligned_strides) {
    static constexpr auto strides = compute_aligned_strides<typename BufferType::value_type>(dimensions);
    constexpr auto max_size = max_product(dimensions, strides);
    static_assert(max_size > 0);
    BufferType buffer{max_size * channels};
    return Plane<BufferType, dimensions, strides, channels>{buffer, offset};
  } else {
    static constexpr auto strides = compute_unaligned_strides(dimensions);
    constexpr auto max_size = max_product(dimensions, strides);
    static_assert(max_size > 0);
    BufferType buffer{max_size * channels};
    return Plane<BufferType, dimensions, strides, channels>{buffer, offset};
  }
}

}  // namespace ntensor
