#pragma once

#include <functional>
#include <limits>
#include <unordered_map>

#include "aligned_allocator.hpp"
#include "concepts.hpp"

namespace ntensor {

/*
 * Constraints:
 * T has to satisfy the arithmetic concept
 * WARNING:
 * Most of the assignment operators can change the value of an existing SparseBuffer element to zero
 * However once an element is added to the SparseBuffer, even if it's set to 0, it won't be automatically removed from
 * it
 */
template <arithmetic T,
          typename Container = std::unordered_map<std::size_t, T, std::hash<std::size_t>, std::equal_to<std::size_t>,
                                                  nt_allocator<std::pair<const std::size_t, T>>>>
class SparseBuffer {
 private:
  std::shared_ptr<Container> _memory;
  std::size_t _size{};

 public:
  /*
   * Class representing an element of the SparseBuffer
   */
  template <typename ValueType>
  class SparseValue {
   private:
    inline static ValueType zero{};
    ValueType* _ptr{&zero};
    std::function<void(ValueType*&, const ValueType&)> _callback;

   public:
    /*
     * Default constructor
     */
    SparseValue() noexcept = default;

    /*
     * Constructs a new SparseValue that points to an element stored inside the SparseBuffer
     * Parameters:
     * @param value: element residing inside the SparseBuffer
     */
    explicit SparseValue(ValueType& value) : _ptr{&value} {}

    /*
     * Constructs a new SparseValue that points to an empty element
     * Parameters:
     * @param callback: invocable used for setting a value for the requested element and storing it into the
     * SparseBuffer
     */
    explicit SparseValue(std::invocable<ValueType*&, ValueType> auto callback) : _callback{callback} {}

    /*
     * Implicit conversion operator converting SparseValue to const T&
     */
    [[nodiscard]] inline operator const ValueType&() const { return *_ptr; }

    /*
     * Simple assignment operator
     */
    inline void operator=(const ValueType& value) {
      if (*_ptr) [[unlikely]]
        *_ptr = value;
      else [[likely]]
        _callback(_ptr, value);
    }

    /*
     * Unary plus operator
     */
    [[nodiscard]] inline ValueType operator+() const { return +*_ptr; }

    /*
     * Unary minus operator
     */
    [[nodiscard]] inline ValueType operator-() const { return -*_ptr; }

    /*
     * Addition operator
     */
    [[nodiscard]] inline ValueType operator+(const ValueType& value) const { return *_ptr + value; }

    /*
     * Addition assignment operator
     */
    void operator+=(const ValueType& value) {
      if (*_ptr) [[unlikely]]
        *_ptr += value;
      else [[likely]]
        _callback(_ptr, value);
    }

    /*
     * Subtraction operator
     */
    [[nodiscard]] inline T operator-(const ValueType& value) const { return *_ptr - value; }

    /*
     * Subtraction assignment operator
     */
    void operator-=(const ValueType& value) {
      if (*_ptr) [[unlikely]]
        *_ptr -= value;
      else [[likely]]
        _callback(_ptr, ValueType{} - value);
    }

    /*
     * Multiplication operator
     */
    [[nodiscard]] inline T operator*(const ValueType& value) const { return *_ptr * value; }

    /*
     * Multiplication assignment operator
     */
    void operator*=(const ValueType& value) {
      if (*_ptr) [[unlikely]]
        *_ptr *= value;
    }

    /*
     * Division operator
     */
    [[nodiscard]] inline ValueType operator/(const ValueType& value) const { return *_ptr / value; }

    /*
     * Division assignment operator
     */
    void operator/=(const ValueType& value) {
      if (*_ptr) [[unlikely]]
        *_ptr /= value;
      else [[likely]]
        _callback(_ptr, std::numeric_limits<ValueType>::infinity());
    }

    /*
     * Equality operator
     * Parameters:
     * param other: element to compare against
     * True if the elements are same, false otherwise
     */
    [[nodiscard]] inline bool operator==(const ValueType& other) const noexcept { return *_ptr == other; }

    /*
     * Three-way comparison operator
     * Parameters:
     * param other: element to compare against
     * @return: result of the comparison
     */
    [[nodiscard]] inline auto operator<=>(const ValueType& other) const noexcept { return *_ptr <=> other; }
  };

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = SparseValue<T*>;
  using const_pointer = SparseValue<const T*>;
  using reference = SparseValue<T>;
  using const_reference = SparseValue<const T>;
  using value_type = SparseValue<T>;

  /*
   * Default constructor
   */
  SparseBuffer() noexcept = default;

  /*
   * Reserves memory for the specified number of elements
   * Parameters:
   * @param size: maximum number of elements that can reside inside the buffer
   */
  explicit SparseBuffer(std::size_t size) : _size{size} {
#ifdef ENABLE_NT_EXPECTS
    Expects(size > 0u && size < std::numeric_limits<std::size_t>::max() / sizeof(T));
#endif
    _memory = std::allocate_shared<Container>(nt_allocator<Container>());
    _memory->reserve(_size);
  }

  /*
   * Copy constructor
   */
  SparseBuffer(const SparseBuffer& other) : _memory{other._memory}, _size{other._size} {}

  /*
   * Copy assignment operator
   */
  SparseBuffer& operator=(const SparseBuffer& other) {
    if (this != &other) [[likely]] {
      _memory = other._memory;
      _size = other._size;
    }
    return *this;
  }

  /*
   * Move constructor
   */
  SparseBuffer(SparseBuffer&& other) noexcept : _memory{std::move(other._memory)}, _size{other._size} {
    other._size = 0u;
  }

  /*
   * Move assignment operator
   */
  SparseBuffer& operator=(SparseBuffer&& other) noexcept {
    if (this != &other) [[likely]] {
      _memory = std::move(other._memory);
      _size = other._size;
      other._size = 0u;
    }
    return *this;
  }

  /*
   * Compares two sparse buffers for equality
   * Parameters:
   * param lhs: first (left-hand side) sparse buffer
   * param rhs: second (right-hand side) sparse buffer
   * True if the buffers are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const SparseBuffer& lhs, const SparseBuffer& rhs) noexcept {
    return lhs._memory == rhs._memory && lhs._size == rhs._size;
  }

  /*
   * Returns a reference to the element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: reference to the element at the specified location
   */
  [[nodiscard]] reference operator[](std::size_t index) {
    if (_memory->contains(index)) [[unlikely]] {
      return SparseValue((*_memory)[index]);
    } else [[likely]] {
      auto callback = [this, index](T*& ptr, const T& value) {
        if (value == T{}) [[unlikely]] {
          return;
        }

        (*_memory)[index] = value;
        ptr = &(*_memory)[index];
      };

      return SparseValue<T>(callback);
    }
  }

  /*
   * Returns a const reference to the element at the specified location, without bounds checking
   * Parameters:
   * @param index: location of the element to retrieve
   * @return: const reference to the element at the specified location
   */
  [[nodiscard]] const_reference operator[](std::size_t index) const {
    // hash map doesn't have a const operator[] version, so find is used here as a workaround
    if (auto it = _memory->find(index); it != _memory->end()) [[unlikely]] {
      const auto& value = it->second;
      return SparseValue(value);
    } else [[likely]] {
      return SparseValue<const T>{};
    }
  }

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
    return this->operator[](index);
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
    return this->operator[](index);
  }

  /*
   * Returns the maximum number of elements that can reside inside the buffer
   * Parameters:
   * @return: maximum number of elements that can reside inside the buffer
   */
  [[nodiscard]] inline std::size_t size() const noexcept { return _size; }
};

}  // namespace ntensor
