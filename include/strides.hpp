#pragma once

#include "concepts.hpp"
#include "dimensions.hpp"

namespace ntensor {

/*
 * Strides of a multidimensional array (offset that has to be skipped to reach a certain dimension)
 * Parameters:
 * @tparam strides: sequence of strides
 * Constraints:
 * The strides sequence has to contain at least one stride
 */
template <long long... strides>
  requires(sizeof...(strides) > 0u)
class Strides {
  static constexpr size_t _rank = sizeof...(strides);

 public:
  /*
   * Returns the element at the specified position
   * Parameters:
   * @tparam N: index of the element that needs to be retrieved
   * @return: the element at the specified position found in the strides sequence
   */
  template <std::size_t N>
  [[nodiscard]] consteval int at() const noexcept {
    return nth_element<N, strides...>();
  }

  /*
   * Returns the total number of strides
   * @return: total number of strides in the strides sequence
   */
  [[nodiscard]] static consteval std::size_t rank() noexcept { return _rank; }
};

namespace internal {

/*
 * Returns true if strides contain a padding, false otherwise
 * Parameters:
 * @tparam dimensions: dimensions of a multidimensional array
 * @tparam strides: strides of a multidimensional array
 * Constraints:
 * The dimensions have to be natural numbers
 * There has to be an equal number of dimensions and strides
 */
template <std::size_t... dimensions, long long... strides>
  requires(natural_numbers<dimensions...> && sizeof...(dimensions) == sizeof...(strides))
[[nodiscard]] consteval bool strides_contain_padding(Dimensions<dimensions...>, Strides<strides...>) noexcept {
  constexpr std::size_t max_length = []<std::size_t... is>(std::index_sequence<is...>) {
    constexpr std::array lengths{(dimensions * strides)...};
    std::size_t max_length = lengths[0U];
    ((lengths[is] > max_length ? max_length = lengths[is], 0 : 0), ...);
    return max_length;
  }(std::make_index_sequence<sizeof...(strides)>());

  constexpr auto dimensions_product = (dimensions * ...);

  // It should never happen that the number of elements in a multidimensional array is larger than the array size itself
  static_assert(dimensions_product <= max_length);

  return (dimensions_product < max_length) ? true : false;
}

/*
 * Checks whether any of the given strides is negative
 * @tparam strides: Sequence of strides
 */
template <long long... strides>
[[nodiscard]] consteval bool contains_negative_strides(Strides<strides...>) noexcept {
  return ((strides < 0) || ...);
}

}  // namespace internal

/*
 * Implementation of the "compute_aligned_strides" method
 */
template <typename T, std::unsigned_integral auto... dimensions>
  requires(natural_numbers<dimensions...>)
[[nodiscard]] consteval auto compute_aligned_strides() noexcept {
  constexpr auto first_dimension = fis_v<dimensions...>;
  constexpr auto first_aligned_stride_pos =
      first_dimension != 1u ? 1u : find_with_predicate<dimensions...>([](int d) consteval { return d != 1u; });
  constexpr std::size_t N = sizeof...(dimensions);

  /*
   * when creating a multidimensional array, the first stride is the stride of the innermost dimension
   * the first stride cannot be the first aligned stride as well, since the innermost dimension has always the stride
   * equal to 1
   */
  static_assert(first_aligned_stride_pos > 0u);

  return []<std::size_t... is>(std::index_sequence<is...>) {
    constexpr int alignMask = NT_ALIGNMENT / static_cast<int>(sizeof(T)) - 1;
    static_assert(alignMask > 0);

    constexpr std::array<fis_t<dimensions...>, N> dimensions_array{dimensions...};
    constexpr std::array<long long, N> strides_array{
        is < first_aligned_stride_pos ? 1
        : is == first_aligned_stride_pos
            ? (static_cast<long long>(dimensions_array[is - 1u]) + alignMask) & ~alignMask
            : (static_cast<long long>(dimensions_array[is - 1u]) * strides_array[is - 1u])...};

    return Strides<strides_array[is]...>{};
  }(std::make_index_sequence<N>());
}

/*
 * Computes the aligned strides of a multidimensional array
 * Parameters:
 * @tparam T: value type for which the strides are being computed
 * @tparam dimensions: array's dimensions
 * @return: computed strides
 * Constraints:
 * The dimensions have to be natural numbers
 */
template <typename T, std::unsigned_integral auto... dimensions>
  requires(natural_numbers<dimensions...>)
[[nodiscard]] consteval auto compute_aligned_strides(Dimensions<dimensions...>) noexcept {
  return compute_aligned_strides<T, dimensions...>();
}

/*
 * Implementation of the "compute_unaligned_strides" method
 */
template <std::unsigned_integral auto... dimensions>
  requires(natural_numbers<dimensions...>)
[[nodiscard]] consteval auto compute_unaligned_strides() noexcept {
  constexpr std::size_t N = sizeof...(dimensions);
  return []<std::size_t... is>(std::index_sequence<is...>) {
    constexpr std::array<fis_t<dimensions...>, N> dimensions_array{dimensions...};
    constexpr std::array<long long, N> strides_array{
        is == 0 ? 1 : (static_cast<long long>(dimensions_array[is - 1u]) * strides_array[is - 1u])...};

    return Strides<strides_array[is]...>{};
  }(std::make_index_sequence<N>());
}

/*
 * Computes the unaligned strides of a multidimensional array
 * Parameters:
 * @tparam dimensions: array's dimensions
 * @return: computed strides
 * Constraints:
 * The dimensions have to be natural numbers
 */
template <std::unsigned_integral auto... dimensions>
  requires(natural_numbers<dimensions...>)
[[nodiscard]] consteval auto compute_unaligned_strides(Dimensions<dimensions...>) noexcept {
  return compute_unaligned_strides<dimensions...>();
}

/*
 * Given the dimensions of a multidimensional array and the first N strides, where M is the size of the multidimensional
 * array, N > 0 and N < M, computes the rest of the strides of the array Parameters:
 * @tparam dimensions: dimensions for which the strides need to be computed
 * @tparam partial_strides: strides preceding the newly computed strides
 * @return: computed strides
 * Constraints:
 * The dimensions have to be natural numbers
 * There has to be at least one stride in the passed strides sequence
 */
template <std::unsigned_integral auto... dimensions, std::signed_integral auto... partial_strides>
  requires(natural_numbers<dimensions...> && sizeof...(partial_strides) > 0u)
[[nodiscard]] consteval auto partially_compute_strides(Dimensions<dimensions...>,
                                                       Strides<partial_strides...>) noexcept {
  constexpr std::size_t dimensions_size = sizeof...(dimensions);
  constexpr std::size_t partial_strides_size = sizeof...(partial_strides);

  return []<std::size_t... is>(std::index_sequence<is...>) {
    constexpr std::array<fis_t<dimensions...>, dimensions_size> dimensions_array{dimensions...};
    constexpr std::array<long long, dimensions_size + partial_strides_size> partial_strides_array{
        partial_strides...,
        (static_cast<long long>(dimensions_array[is]) * partial_strides_array[partial_strides_size + is - 1u])...};
    return Strides<partial_strides..., partial_strides_array[partial_strides_size + is]...>{};
  }(std::make_index_sequence<dimensions_size>());
}

}  // namespace ntensor
