#pragma once

#include <concepts>

namespace ntensor {

/*
 * Concept for defining a sequence of natural numbers
 * Meaning, the numbers have to be positive integers between 1 and infinity
 */
template <std::unsigned_integral auto... vs>
concept natural_numbers = (vs * ...) > 0u;

/*
 * Concept defining the minimal requirements for a const type to be arithmetic
 */
template <typename T>
concept const_arithmetic = requires(T v) {
  +v;
  -v;
  v + T{};
  v - T{};
  v* T{};
  v / T{};
  v == T{};
  v != T{};
  v <=> T{};
};

/*
 * Concept defining the minimal requirements for either a const or a non-const type to be arithmetic
 */
template <typename T>
concept arithmetic = const_arithmetic<T> && (std::is_const_v<T> || requires(T v) {
                       v = T{};
                       v += T{};
                       v -= T{};
                       v *= T{};
                       v /= T{};
                     });

/*
 * Concept for defining a multiplicative value sequence
 * Meaning, all of the values inside the value sequence have to be able to perform multiplication with each other
 */
template <auto... vs>
concept multiplicative = requires() { (vs * ...); };

/*
 * Concept specifying the minimal requirements to define an allocator
 * Parameters:
 * @tparam Allocator: class template representing an allocator
 * tparam T: value type used for allocating memory
 * Constraints:
 * T has to be an arithmetic type
 * The allocator has to have a method for allocating memory, named "allocate"
 * The method "allocate" has to receive the allocation size, and it should return the newly allocated memory of type T*
 * The allocator has to have a method for deallocating memory, named "deallocate"
 * The method "deallocate" has to receive two elements: pointer to the memory that needs to be deallocated, and the size
 * of the allocated buffer
 */
template <typename Allocator, typename T>
concept allocator = requires(Allocator a) {
  std::is_arithmetic_v<T>;
  { a.allocate(0u) } -> std::same_as<T*>;
  a.deallocate(nullptr, 0u);
};

/*
 * Concept specifying the minimal requirements to define a buffer
 * Constraints:
 * The buffer has to have defined aliases: reference and const_reference
 * The buffer has to have an implementation of the "size" method
 * The buffer has to implement either the method "at" or a subscript operator for both const and non-const context
 * These methods have to return either a reference or a const_reference, depending on the context
 * Shortcomings:
 * Unfortunately, this concept doesn't support basic C-style arrays
 */
template <typename T>
concept buffer = requires(std::remove_reference_t<T> const v) {
  typename T::reference;
  typename T::const_reference;
  { v.size() } -> std::same_as<std::size_t>;
} && (((requires(std::remove_reference_t<T> const v) {
                   { v.at(0u) } -> std::same_as<typename T::const_reference>;
                 }) && (requires(std::remove_reference_t<T> v) {
                   { v.at(0u) } -> std::same_as<typename T::reference>;
                 })) || ((requires(std::remove_reference_t<T> const v) {
                           { v[0u] } -> std::same_as<typename T::const_reference>;
                         }) && (requires(std::remove_reference_t<T> v) {
                           { v[0u] } -> std::same_as<typename T::reference>;
                         })));

}  // namespace ntensor
