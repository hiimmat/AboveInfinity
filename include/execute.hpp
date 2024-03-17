#pragma once

#include "strides.hpp"

namespace ntensor {

inline namespace internal {
/*
 * The following is just an inverse operation of the index computation in a 1D array
 * If we know how to compute an index in a 1D array, then we can reverse that operation to compute the position in a
 * multidimensional array
 *
 * Each dimension represents a layer of a tensor
 * So all that we have to do is divide the index of the requested element with the stride (offset) to the last
 * dimension, and the result will be the position of the requested element in the last dimension Next, we take the
 * result of the modulo calculation and use it to compute the position of the requested element in one dimension lower
 * We repeat this process until we iterate all of the dimensions, and that way we retrieve the resulting position
 * For example:
 * Let's say that we have a 3D tensor, that has all equal dimensions of lengths 3 (3x3x3). And let's say that we want
 * to access the 8th element First we compute the stride to the 3rd dimension which is equal to the multiplication of
 * the length of the previous dimension times its stride: 3x3=9 So we have to skip a 2D array of 9 elements to reach
 * the 3rd and final dimension Now we divide the index with the 3rd dimension stride: 8/9=0 Next, we compute the
 * modulo: 8%9=8 This tells us that the requested index is in the first layer of the 3rd dimension Now we repeat this
 * process with the length of the second dimension. So we divide 8 with the 2nd dimension stride: 8/3=2 This tells us
 * that the requested element lies within the 3rd row Once again we compute the modulo: 8%3=2 And use it within the
 * innermost dimension, which represents the columns So the final position of the element in the array is: [2][2][0],
 * where [0] is the outtermost dimension
 *
 * TODO:
 * Try to optimize this algorithm. It's used inside a loop.
 *
 * Parameters:
 * @tparam us: unaligned strides (offsets) of the multidimensional array
 * @tparam as: aligned strides of the multidimensional array
 * @param channels: number of channels in each dimension of the multidimensional array
 * @param index: position of the element in a 1D array
 */
template <std::size_t channels, long long... us, long long... as>
[[nodiscard]] long long compute_array_position_from_index(Strides<us...>, Strides<as...>, std::size_t channel,
                                                          std::size_t index) {
  static constexpr std::array unaligned_strides_array{us...};
  static constexpr std::array aligned_strides_array{as...};
  std::size_t N = aligned_strides_array.size() - 1u;
  long long position = 0u;

  while (N > 1u) {
    position += ((index / unaligned_strides_array[N]) * aligned_strides_array[N]);
    index %= unaligned_strides_array[N];
    --N;
  }
  position += ((index / unaligned_strides_array[N]) * aligned_strides_array[N]);
  position += ((index % unaligned_strides_array[N]) * aligned_strides_array[N - 1u]);
  position *= channels;
  position += channel;

  return position;
}

/*
 * Specialization of the execute method that supports iterating multiple planes simultaneously, whose dimensions don't
 * match, but have the same product. For example, let's say that we have the first plane containing dimensions [2, 2,
 * 6], and the second plane containing dimensions [4, 6]. Since their product of dimensions is equal, these can be
 * iterated simultaenously. Parameters:
 * @param invocable: Invocable called on each element of a plane/s
 * @param planes: Variadic number of planes
 */
template <typename Invocable, typename... Planes>
void iterative_execute(Invocable&& invocable, Planes&&... planes) {
  using first_plane_type = std::decay_t<fts_t<Planes...>>;
  static constexpr std::size_t N = product(first_plane_type::dimensions());
  static constexpr std::size_t channels = first_plane_type::channels();
  static constexpr std::tuple strides{std::decay_t<Planes>::strides()...};
  static constexpr std::tuple unaligned_strides{compute_unaligned_strides(std::decay_t<Planes>::dimensions())...};

  for (std::size_t idx = 0u; idx < N; ++idx) {
    for (std::size_t c = 0u; c < channels; ++c) {
      [c, idx, &invocable, &planes...]<std::size_t... is>(std::index_sequence<is...>) {
        const std::array positions{compute_array_position_from_index<channels>(std::get<is>(unaligned_strides),
                                                                               std::get<is>(strides), c, idx)...};
        invocable(planes.at(positions[is])...);
      }(std::make_index_sequence<sizeof...(Planes)>());
    }
  }
}

/*
 * Specialization of the execute method that supports iterating several planes simultaenously
 * However, it has the restriction that all of the planes have to have the same dimensions
 * Parameters:
 * @param invocable: Invocable called on each element of a plane/s
 * @param planes: Variadic number of planes
 */
template <typename Invocable, typename... _Planes>
void recursive_execute(Invocable&& invocable, _Planes&&... planes) {
  using planes_type = std::decay_t<fts_t<_Planes...>>;
  static constexpr std::size_t rank = planes_type::rank();

  if constexpr (rank > 1u) {
    for (std::size_t i = 0u; i < planes_type::dimensions().template at<rank - 1U>(); ++i) {
      recursive_execute(invocable,
                        planes.template like<remove_nth_element<rank - 1u>(planes_type::dimensions()),
                                             remove_nth_element<rank - 1u>(planes_type::strides())>(
                            i * planes_type::strides().template at<rank - 1u>() * planes_type::channels())...);
    }
  } else if constexpr (rank == 1U) {
    static constexpr std::size_t dimension = planes_type::dimensions().template at<0u>();
    static constexpr std::size_t batch_size =
        (NT_ALIGNMENT / sizeof(typename planes_type::value_type)) / planes_type::channels();
    // The batch_size has to be at least 1, otherwise we won't iterate any elements
    static_assert(batch_size > 0u);
    if constexpr (dimension <= batch_size)
      for (std::size_t d = 0u; d < dimension; ++d)
        for (std::size_t c = 0u; c < planes_type::channels(); ++c)
          invocable(planes.at(d * planes_type::strides().template at<0u>() * planes_type::channels() + c)...);
    else {
      std::size_t iter_size = dimension;
      std::size_t curr_iter = 0u;

      while (iter_size > batch_size) {
        for (std::size_t d = curr_iter; d < curr_iter + batch_size; ++d)
          for (std::size_t c = 0u; c < planes_type::channels(); ++c)
            invocable(planes.at(d * planes_type::strides().template at<0u>() * planes_type::channels() + c)...);
        iter_size -= batch_size;
        curr_iter += batch_size;
      }
      for (std::size_t d = curr_iter; d < curr_iter + iter_size; ++d)
        for (std::size_t c = 0u; c < planes_type::channels(); ++c)
          invocable(planes.at(d * planes_type::strides().template at<0u>() * planes_type::channels() + c)...);
    }
  }
}

}  // namespace internal

/*
 * Calls an invocable on each element of a tensor
 * Parameters:
 * @param invocable: Invocable called on each element of a tensor
 * @param tensor: Tensor on whose elements the invocable is called upon
 */
template <typename Invocable, typename Tensor>
void execute(Invocable&& invocable, Tensor&& tensor) {
  for_each_plane(
      [&invocable](auto&& plane) {
        static_assert(std::is_invocable_v<Invocable, decltype(plane.at(0u))>);
        recursive_execute(invocable, std::forward<decltype(plane)>(plane));
      },
      tensor);
}

/*
 * Calls an invocable on each element of any number of tensors
 * There are two possible pathways within this method:
 * 1) The invocable receives N elements. In this case, we try to iterate all of the tensors simultaneously, and execute
 * the invocable across an element of each tensor 2) The invocable receives a single element. In this case, we iterate
 * each tensor separately, executing the invocable on each element as we go Parameters:
 * @param invocable: Invocable called on each element of a tensor/s
 * @param tensors: Variadic number of tensors
 * Constraints:
 * The method has to receive more than one tensors
 */
template <typename Invocable, typename... Tensors>
  requires(sizeof...(Tensors) > 1u)
void execute(Invocable&& invocable, Tensors&&... tensors) {
  // Check if the invocable receives an equal number of arguments as the number of tensors that are passed to this
  // function
  static constexpr auto is_simultaneous_invocation =
      std::is_invocable_v<Invocable, decltype(tensors.planes().template plane<0u>().at(0u))...>;

  // Execute the function across all tensors simultaneously
  if constexpr (is_simultaneous_invocation) {
    using first_tensor = fts_t<Tensors...>;
    static constexpr std::size_t N = std::decay_t<decltype(std::declval<first_tensor>().planes())>::size();

    // Tensors have to have the same number of planes
    static_assert(((N == std::decay_t<decltype(tensors.planes())>::size()) && ...));

    // All planes that are executed simultaneously must have an equal number of channels, and the same product of
    // dimensions (equal number of elements)
    for_all_planes(
        [](auto&&... planes) {
          using first_plane_type = std::decay_t<fts_t<decltype(planes)...>>;
          static constexpr std::size_t expected_dimensions_product = product(first_plane_type::dimensions());
          static constexpr std::size_t expected_num_channels = first_plane_type::channels();
          static_assert(
              ((expected_dimensions_product == product(std::decay_t<decltype(planes)>::dimensions())) && ...));
          static_assert(((expected_num_channels == std::decay_t<decltype(planes)>::channels()) && ...));
        },
        tensors...);

    // Assure that the invocable can be invoked with all planes
    for_all_planes([](auto&&... planes) { static_assert(std::is_invocable_v<Invocable, decltype(planes.at(0u))...>); },
                   tensors...);

    /*
     * Check whether the planes have equal dimensions or just the same number of elements
     * This is used to determine which specialization of the execute method will be called
     */
    for_all_planes(
        [&invocable](auto&&... planes) {
          using first_plane_type = std::decay_t<fts_t<decltype(planes)...>>;
          using first_plane_dimensions = std::decay_t<decltype(first_plane_type::dimensions())>;
          static constexpr bool dimensions_match =
              ((std::is_same_v<first_plane_dimensions,
                               std::decay_t<decltype(std::decay_t<decltype(planes)>::dimensions())>>)&&...);

          if (dimensions_match) {
            recursive_execute(std::forward<Invocable>(invocable), std::forward<decltype(planes)>(planes)...);
          } else {
            // TODO:
            // If I can perform reshape in place, I don't need iterative_execute. I can call recursive_execute instead
            iterative_execute(std::forward<Invocable>(invocable), std::forward<decltype(planes)>(planes)...);
          }
        },
        tensors...);
  }
  // Execute the invocable for each tensor separately
  else {
    (execute(std::forward<Invocable>(invocable), tensors), ...);
  };
}

}  // namespace ntensor
