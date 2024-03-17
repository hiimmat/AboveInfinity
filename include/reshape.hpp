#pragma once

#include "execute.hpp"

namespace ntensor {

inline namespace internal {

/*
 * Implementation of the can_reshape_in_place method
 */
template <std::size_t initial_dimension, std::size_t reshaped_dimension, std::size_t... reshaped_dimensions>
[[nodiscard]] consteval bool can_reshape_in_place() noexcept {
  if constexpr (sizeof...(reshaped_dimensions) == 0u)
    return initial_dimension / reshaped_dimension == 1u && initial_dimension % reshaped_dimension == 0u;
  else if constexpr (initial_dimension / reshaped_dimension == 1u)
    return initial_dimension % reshaped_dimension == 0u;
  else if constexpr (initial_dimension / reshaped_dimension > 1u && initial_dimension % reshaped_dimension == 0u)
    return can_reshape_in_place<initial_dimension / reshaped_dimension, reshaped_dimensions...>();
  else
    return false;
}

/*
 * Determines whether a plane can be reshaped in place based on its first dimension, and the new shape of the plane
 * Parameters:
 * @tparam initial_dimension: first dimension of the plane
 * @tparam reshaped_dimensions: new shape of the plane
 */
template <std::size_t initial_dimension, auto... reshaped_dimensions>
[[nodiscard]] consteval bool can_reshape_in_place(Dimensions<reshaped_dimensions...>) noexcept {
  return can_reshape_in_place<initial_dimension, reshaped_dimensions...>();
}

/*
 * Implementation of the greatest_common_divisor method
 */
template <std::size_t initial_dimension, std::size_t reshaped_dimension, std::size_t... reshaped_dimensions>
[[nodiscard]] consteval std::size_t greatest_common_divisor() noexcept {
  if constexpr (sizeof...(reshaped_dimensions) == 0u)
    return 1U;
  else if constexpr (initial_dimension / reshaped_dimension == 1u)
    return 1U;
  else
    return 1U + greatest_common_divisor<initial_dimension / reshaped_dimension, reshaped_dimensions...>();
}

/*
 * Computes the number of reshaped dimensions that can be squeezed into the current first dimension of the plane
 * Parameters:
 * @param initial_dimension: first dimension of the plane
 * @param reshaped_dimensions: new shape of the plane
 * Constraints:
 * The reshape operation has to be possible to perform without allocating any extra memory
 */
template <std::size_t initial_dimension, std::size_t... reshaped_dimensions>
  requires(can_reshape_in_place<initial_dimension, reshaped_dimensions...>())
[[nodiscard]] consteval std::size_t greatest_common_divisor(Dimensions<reshaped_dimensions...>) noexcept {
  return greatest_common_divisor<initial_dimension, reshaped_dimensions...>();
}

}  // namespace internal

/*
 * Changes the shape of a plane without changing its underlying data
 * Parameters:
 * @tparam reshaped_dimensions: New dimensions of the plane. The product of these has to be equal to the product of the
 * old dimensions
 * @tparam plane_idx: Index of the plane whose shape we're modifying
 * @param tensor: Tensor on whom the operation will be performed
 */
template <Dimensions reshaped_dimensions, std::size_t plane_idx = 0u, typename _Tensor>
[[nodiscard]] auto reshape(_Tensor&& tensor) {
  auto plane = tensor.planes().template plane<plane_idx>();
  static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
  static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

  // If the reshaped lengths match the current lengths, return the same object
  if constexpr (std::is_same_v<std::decay_t<decltype(dimensions)>, std::decay_t<decltype(reshaped_dimensions)>>)
    return tensor;
  else {
    /*
     * Dimensions of length equal to 1 have no effect, however removing them might avoid some edge cases
     * For example, we might wrongly assume that we have a padding in between the first and the second dimension, if we
     * have a plane of two dimensions, and the second dimension has a length of 1 Furthermore, it might happen that two
     * Dimensions structures are the same, however, one contains dimensions of length equal to 1. Due to this, the
     * comparison would fail And since strides can be 0, the dimensions can be of any length, and still have no effect
     * However, these dimensions might affect the result of the product method
     * Furthermore, strides that are equal to 0 would affect the sorting
     */
    static constexpr auto filtered_dimensions_indexes =
        find_all_with_predicate(dimensions, [](std::size_t d) consteval { return d != 1u; });
    static constexpr auto filtered_strides_indexes =
        find_all_with_predicate(strides, [](long long s) consteval { return s != 0; });
    static constexpr auto intersected_indexes = set_intersection(filtered_dimensions_indexes, filtered_strides_indexes);

    // Assure that not all dimensions have a length of one or a stride equal to 0
    static_assert(intersected_indexes.size() > 0u);

    static constexpr auto filtered_dimensions = []<std::size_t... is>(std::index_sequence<is...>) {
      return Dimensions<dimensions.template at<is>()...>{};
    }(intersected_indexes);
    static constexpr auto filtered_strides = []<std::size_t... is>(std::index_sequence<is...>) {
      return Strides<strides.template at<is>()...>{};
    }(intersected_indexes);
    static constexpr auto filtered_rank = std::decay_t<decltype(filtered_strides)>::rank();
    static constexpr auto filtered_idx_sequence = std::make_index_sequence<filtered_rank>();

    /*
     * Assure that the new dimensions are compatible with the old dimensions
     * Even if the dimensions were not compatible, if we have a padding in between the first and the second dimension
     * (due to strides), we might have enough space to perform a reshape in place However, that would leave us with new,
     * uninitialized elements, which sounds error-prone
     */
    static_assert(product(filtered_dimensions) == product(reshaped_dimensions));

    static constexpr bool are_strides_negative = contains_negative_strides(filtered_strides);

    /*
     * If there are no negative strides and if there's no padding between the first and second dimension, we can avoid
     * allocating a new tensor Instead, we can just change the layout of the existing one
     */
    if constexpr (!are_strides_negative && !strides_contain_padding(filtered_dimensions, filtered_strides)) {
      static constexpr auto reshaped_strides = compute_unaligned_strides(reshaped_dimensions);
      const auto reshaped_plane = plane.template like<reshaped_dimensions, reshaped_strides>();
      const auto updated_planes = tensor.planes().template replace<plane_idx>(reshaped_plane);
      return tensor.like(updated_planes);
    } else {
      /*
       * Even if the tensor contains a padding, try to avoid allocating a new tensor
       * To do this:
       * Try to unpermute dimensions and strides (if they are permuted), so that we can get the correct stride ordering
       * See if we can reshape dimensions in place despite having a padding
       * WARNING:
       * Unpermuting dimensions and strides won't work if some of the strides are negative
       */

      // Try to restore the correct stride ordering
      static constexpr auto sorted_stride_indexes_array = []<std::size_t... is>(std::index_sequence<is...>) consteval {
        constexpr std::array filtered_strides_array{filtered_strides.template at<is>()...};
        std::array strides_indexes{is...};
        std::sort(begin(strides_indexes), end(strides_indexes),
                  [filtered_strides_array](std::size_t lhs, std::size_t rhs) {
                    return filtered_strides_array[lhs] < filtered_strides_array[rhs];
                  });
        return strides_indexes;
      }(filtered_idx_sequence);

      static constexpr auto sorted_stride_indexes = []<std::size_t... is>(std::index_sequence<is...>) {
        return std::index_sequence<sorted_stride_indexes_array[is]...>{};
      }(filtered_idx_sequence);

      static constexpr auto sorted_dimensions = []<std::size_t... is>(std::index_sequence<is...>) {
        return Dimensions<filtered_dimensions.template at<is>()...>{};
      }(sorted_stride_indexes);
      static constexpr auto sorted_strides = []<std::size_t... is>(std::index_sequence<is...>) {
        return Strides<filtered_strides.template at<is>()...>{};
      }(sorted_stride_indexes);

      static constexpr bool can_reshape_using_dimensions =
          can_reshape_in_place<sorted_dimensions.template at<0u>()>(reshaped_dimensions);

      if constexpr (!are_strides_negative && can_reshape_using_dimensions) {
        static constexpr std::size_t innermost_dimension = sorted_dimensions.template at<0u>();
        static constexpr long long first_aligned_stride = sorted_strides.template at<1u>();
        static constexpr std::size_t N = greatest_common_divisor<innermost_dimension>(reshaped_dimensions);
        static constexpr auto partial_strides = []<std::size_t... is>(std::index_sequence<is...>) {
          return Strides<1, static_cast<long long>(reshaped_dimensions.template at<is>())..., first_aligned_stride>{};
        }(std::make_index_sequence<N - 1u>());
        static constexpr auto partial_dimensions = []<std::size_t... is>(std::index_sequence<is...>) {
          return Dimensions<reshaped_dimensions.template at<is>()...>{};
        }(create_index_sequence<N, std::decay_t<decltype(reshaped_dimensions)>::rank() - 1u>());
        static constexpr auto reshaped_strides = partially_compute_strides(partial_dimensions, partial_strides);
        auto reshaped_plane = plane.template like<reshaped_dimensions, reshaped_strides>();
        const auto updated_planes = tensor.planes().template replace<plane_idx>(reshaped_plane);
        return tensor.like(updated_planes);
      }
      // Worst case scenario: we can't avoid allocating a new tensor
      else {
        auto reshaped_plane = create_plane<typename std::decay_t<decltype(plane)>::buffer_type, reshaped_dimensions,
                                           std::decay_t<decltype(plane)>::channels()>();
        const auto updated_planes = tensor.planes().template replace<plane_idx>(reshaped_plane);
        auto reshaped_tensor = tensor.like(updated_planes);
        execute([](auto& lhs, auto& rhs) { lhs = rhs; }, reshaped_tensor, tensor);
        return reshaped_tensor;
      }
    }
  }
}

}  // namespace ntensor
