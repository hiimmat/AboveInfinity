#pragma once

#include "range.hpp"
#include "utilities.hpp"

namespace ntensor {

/*
 * Tensor policy for modifying the shape of each plane
 * @tparam _Tensor: Tensor type
 */
template <typename _Tensor>
struct ShapeTransmutation {
  /*
   * Returns a reference to the planes held by the Tensor object
   * @return: reference to the planes of the Tensor object
   */
  [[nodiscard]] inline auto& planes() noexcept { return static_cast<_Tensor*>(this)->_planes; }

  /*
   * Returns a const reference to the planes held by the Tensor object
   * @return: const reference to the planes of the Tensor object
   */
  [[nodiscard]] inline const auto& planes() const noexcept { return static_cast<const _Tensor*>(this)->_planes; }

  /*
   * Creates a new Tensor object in which the specified plane has reordered dimensions and strides
   * Parameters:
   * @tparam plane_idx: index of the used plane
   * @tparam order: new positions of the dimensions
   * @return: new Tensor object whose specified plane has reordered dimensions and strides
   */
  template <std::size_t plane_idx, std::size_t... order>
  [[nodiscard]] auto permute() const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
    // internal::permute validates the order
    static constexpr auto permuted_dimensions = internal::permute<order...>(dimensions);
    static constexpr auto permuted_strides = internal::permute<order...>(strides);
    const auto permuted_plane = plane.template like<permuted_dimensions, permuted_strides>();
    const auto updated_planes = planes.template replace<plane_idx>(permuted_plane);
    return static_cast<const _Tensor*>(this)->like(updated_planes);
  }

  /*
   * Retrieves a value of a plane by performing several hyperplanes
   * Note:
   * Offsets are specified per dimension, and they start from the outtermost one (opposite of the subspace)
   * If a dimension isn't specified, it will be ignored
   * Parameters:
   * @tparam plane_idx: index of the used plane
   * @param channel: used channel
   * @param offsets: offsets used while accessing dimensions
   * @return: reference to the requested value
   */
  template <std::size_t plane_idx = 0u>
  [[nodiscard]] decltype(auto) slicing_value(std::size_t channel, std::unsigned_integral auto... offsets) {
    return [this, channel, offsets...]<std::size_t... is>(std::index_sequence<is...>) -> decltype(auto) {
      // planes.plane() validates the plane_idx
      auto& plane = static_cast<_Tensor*>(this)->_planes.template plane<plane_idx>();
#ifdef ENABLE_NT_EXPECTS
      Expects(plane.channels() > channel);
#endif
      static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
      static constexpr std::size_t nOffsets = sizeof...(offsets);
      // Assert that the function doesn't receive more offsets than there are dimensions in the specified plane
      static_assert(nOffsets <= dimensions.rank());
      static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
      static constexpr std::size_t rank = dimensions.rank();
      const auto offsetsArr = std::array{offsets...};
#ifdef ENABLE_NT_EXPECTS
      Expects((offsetsArr[nOffsets - is - 1u] < dimensions.template at<rank - is - 1u>()) && ...);
#endif
      const std::size_t offset =
          ((((offsetsArr[nOffsets - is - 1U] * strides.template at<rank - is - 1U>())) + ...)) * plane.channels() +
          channel;
      // Element access is validated in the plane.at() method
      return plane.at(offset);
    }(std::make_index_sequence<sizeof...(offsets)>());
  }

  /*
   * Performs a hyperplane on the specified plane
   * Parameters:
   * @plane_idx: index of the used plane
   * @dimension_to_skip: dimension that has to be removed from the plane
   * @dimensional_offset: offset used instead of the removed dimension
   * @return: new Tensor object whose specified plane has 1 rank lower than the original
   */
  template <std::size_t dimension_to_skip, std::size_t plane_idx = 0u>
  [[nodiscard]] auto slice(std::size_t dimensional_offset) const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
    // Assert that dimensions have more than 1 element, otherwise we'll end up with a Tensor with 0 dimensions
    static_assert(dimensions.rank() > 1u);
    // dimensions.at() validates the dimension_to_skip
#ifdef ENABLE_NT_EXPECTS
    Expects(dimensional_offset < dimensions.template at<dimension_to_skip>());
#endif
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
    const long long offset = dimensional_offset * strides.template at<dimension_to_skip>() * plane.channels();
    static constexpr auto sliced_dimensions = remove_nth_element<dimension_to_skip>(dimensions);
    static constexpr auto sliced_strides = remove_nth_element<dimension_to_skip>(strides);
    const auto sliced_plane = plane.template like<sliced_dimensions, sliced_strides>(offset);
    const auto updated_planes = planes.template replace<plane_idx>(sliced_plane);
    return static_cast<const _Tensor*>(this)->like(updated_planes);
  }

  /*
   * Performs a hyperslab on the specified plane
   * Parameters:
   * @plane_idx: index of the used plane
   * @dimension_to_offset: dimension whose size will be reduced
   * @start: new (offseted) start of the dimension
   * @end: new (reduced) end of the dimension
   * @return: new Tensor object whose specified plane has a subrange of its original dimensions
   */
  template <std::size_t dimension_to_offset, std::size_t start, std::size_t end, std::size_t plane_idx = 0u>
  [[nodiscard]] auto slab() const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
    // dimensions.at() validates the dimension_to_offset
    static_assert(start < dimensions.template at<dimension_to_offset>());
    static_assert(end < dimensions.template at<dimension_to_offset>());
    static_assert(end > start);

    const long long offset = start * strides.template at<dimension_to_offset>() * plane.channels();
    static constexpr auto offseted_dimensions = []<std::size_t... is>(std::index_sequence<is...>) {
      return Dimensions<(is == dimension_to_offset ? end - start : dimensions.template at<is>())...>{};
    }(std::make_index_sequence<dimensions.rank()>());

    const auto offseted_plane = plane.template like<offseted_dimensions>(offset);
    const auto updated_planes = planes.template replace<plane_idx>(offseted_plane);
    return static_cast<const _Tensor*>(this)->like(updated_planes);
  }

  /*
   * Performs several hyperplanes and/or hyperslabs on the requested plane's dimensions, depending on the input
   * parameters Note: If the result of subtracting the start from the end of a range results in a 0, the dimension will
   * be removed If it results in a 1, it's the same as performing a hyperplane, but the dimension will be kept In all
   * other cases, a hyperslab is performed Ranges are specified per dimension, and they start from the innermost one
   * (opposite of the slicing_value) If a dimension isn't specified, it will be ignored, and it will be the same as
   * performing a hyperplane on the first element of the dimension Parameters:
   * @ranges: sequence of range types (start, end) defining the start and the end of a dimension
   * @return: new Tensor object whose specified plane has a subrange of its original dimensions/a lower rank than the
   * original plane
   */
  template <std::size_t plane_idx, range... ranges>
    requires(sizeof...(ranges) > 0U)
  [[nodiscard]] auto subspace() const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();

    // Assert ranges
    // the range class assures that end is greater than or equal to the start
    []<std::size_t... is>(std::index_sequence<is...>) {
      static_assert(((ranges.start() <= dimensions.template at<is>()) && ...));
      static_assert(((ranges.end() <= dimensions.template at<is>()) && ...));
    }(std::make_index_sequence<sizeof...(ranges)>());

    // Compute the plane offset
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
    static constexpr int offset =
        []<std::size_t... is>(std::index_sequence<is...>) {
          // strides.at() will fail static assertion if there are more ranges than there are strides available
          return (((ranges.start() == dimensions.template at<is>() ? ranges.start() - 1u : ranges.start()) *
                   strides.template at<is>()) +
                  ...);
        }(std::make_index_sequence<sizeof...(ranges)>()) *
        plane.channels();

    // Determine indexes of unsliced dimensions and strides
    static constexpr auto leftover_indexes_after_slicing =
        filter_index_sequence(std::make_index_sequence<sizeof...(ranges)>(), [](std::size_t idx) consteval {
          constexpr std::array<std::size_t, sizeof...(ranges)> ranges_diff{(ranges.end() - ranges.start())...};
          return ranges_diff[idx] == 0u;
        });

    // Assert that there's at least 1 element left, otherwise we'll end up with a Tensor with 0 dimensions
    static_assert(leftover_indexes_after_slicing.size() > 0u);

    // Create a new tensor
    return [this, &plane, &planes]<std::size_t... is>(std::index_sequence<is...>) {
      static constexpr std::array<std::size_t, sizeof...(ranges)> ranges_diff{(ranges.end() - ranges.start())...};
      static constexpr Dimensions<ranges_diff[is]...> sliced_dimensions;
      static constexpr Strides<strides.template at<is>()...> sliced_strides;
      const auto sliced_plane = plane.template like<sliced_dimensions, sliced_strides>(offset);
      const auto updated_planes = planes.template replace<plane_idx>(sliced_plane);
      return static_cast<const _Tensor*>(this)->like(updated_planes);
    }(leftover_indexes_after_slicing);
  }

  /*
   * Adds a dimension of length 1 at the specified position
   * Parameters:
   * @plane_idx: index of the used plane
   * @N: index at which the new dimension should be added
   * @return: new Tensor object whose specified plane has 1 rank higher than the original plane
   */
  template <std::size_t N, std::size_t plane_idx = 0u>
  [[nodiscard]] auto new_axis() const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto to_array = []<template <auto...> typename T, auto... vs>(T<vs...>) {
      return std::array{vs...};
    };
    static constexpr auto dimensions_array = to_array(std::decay_t<decltype(plane)>::dimensions());
    static constexpr auto strides_array = to_array(std::decay_t<decltype(plane)>::strides());

    static_assert(N <= dimensions_array.size());

    return [this, &planes, &plane]<std::size_t... is>(std::index_sequence<is...>) {
      static constexpr Dimensions<(is < N    ? dimensions_array[is]
                                   : is == N ? 1u
                                             : dimensions_array[is - 1u])...>
          expanded_dimensions;
      static constexpr Strides<(is < N    ? strides_array[is]
                                : is == N ? 0u
                                          : strides_array[is - 1u])...>
          expanded_strides;
      const auto expanded_plane = plane.template like<expanded_dimensions, expanded_strides>();
      const auto updated_planes = planes.template replace<plane_idx>(expanded_plane);
      return static_cast<const _Tensor*>(this)->like(updated_planes);
    }(std::make_index_sequence<dimensions_array.size() + 1u>());
  }

  /*
   * Removes dimensions whose length is equal to 1 from the specified plane
   * Parameters:
   * @plane_idx: index of the used plane
   * @return: new Tensor object whose specified plane does not contain dimensions of length 1
   */
  template <std::size_t plane_idx = 0u>
  [[nodiscard]] decltype(auto) squeeze() const {
    const auto& planes = static_cast<const _Tensor*>(this)->_planes;
    // planes.plane() validates the plane_idx
    const auto& plane = planes.template plane<plane_idx>();
    static constexpr auto dimensions = std::decay_t<decltype(plane)>::dimensions();
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

    static constexpr auto keep_dimension = [](std::size_t d) consteval { return d != 1u; };
    static constexpr auto positions = find_all_with_predicate(dimensions, keep_dimension);

    return [this, &planes, &plane]<std::size_t... is>(std::index_sequence<is...>) {
      static constexpr Dimensions<dimensions.template at<is>()...> squeezed_dimensions;
      static constexpr Strides<strides.template at<is>()...> squeezed_strides;
      const auto squeezed_plane = plane.template like<squeezed_dimensions, squeezed_strides>();
      const auto updated_planes = planes.template replace<plane_idx>(squeezed_plane);
      return static_cast<const _Tensor*>(this)->like(updated_planes);
    }(positions);
  }
};

}  // namespace ntensor
