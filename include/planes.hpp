#pragma once

#include "bounds.hpp"

namespace ntensor {

/*
 * Class representing an array of planes
 * Parameters:
 * @tparam PlaneTypes: types of planes that the class will hold
 * Restrictions:
 * There has to be at least 1 plane type in the PlaneTypes parameter pack
 * To avoid dangling references/pointers, the types held in the PlaneTypes parameter pack can't be references and/or raw
 * pointers
 */
template <typename... PlaneTypes>
  requires(sizeof...(PlaneTypes) > 0u && ((std::is_same_v<PlaneTypes, std::remove_cvref_t<PlaneTypes>>) && ...))
class Planes {
 private:
  std::tuple<PlaneTypes...> planes;
  static constexpr std::size_t N = sizeof...(PlaneTypes);

  /*
   * Implementation of the split method
   */
  template <std::size_t first, std::size_t... rest>
  [[nodiscard]] auto split_impl() const {
    if constexpr (sizeof...(rest) == 0u) {
      static constexpr std::size_t lower_bound = first;
      static constexpr std::size_t upper_bound = N;

      return [this]<std::size_t... is>(std::index_sequence<is...>) {
        return std::tuple{Planes<std::decay_t<decltype(std::get<is>(planes))>...>{std::get<is>(planes)...}};
      }(create_index_sequence<lower_bound, upper_bound>());
    } else {
      static constexpr std::size_t lower_bound = bounds<first, rest...>::lower();
      static constexpr std::size_t upper_bound = bounds<first, rest...>::upper();

      auto res = [this]<std::size_t... is>(std::index_sequence<is...>) {
        return std::tuple{Planes<std::decay_t<decltype(std::get<is>(planes))>...>{std::get<is>(planes)...}};
      }(create_index_sequence<lower_bound, upper_bound>());

      return std::tuple_cat(res, split_impl<rest...>());
    }
  }

 public:
  /*
   * Default constructor
   */
  Planes() noexcept = default;

  /*
   * Constructs a Planes object that holds the specified array of planes
   */
  explicit Planes(PlaneTypes... planes) : planes{planes...} {}

  /*
   * Compares two planes objects for equality
   * Parameters:
   * param lhs: first (left-hand side) planes object
   * param rhs: second (right-hand side) planes object
   * True if the planes objects are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const Planes& lhs, const Planes& rhs) noexcept {
    if (std::decay_t<decltype(lhs)>::N == std::decay_t<decltype(rhs)>::N) {
      return [&lhs, &rhs]<std::size_t... is>(std::index_sequence<is...>) {
        return ((std::get<is>(lhs.planes) == std::get<is>(rhs.planes)) && ...);
      }(std::make_index_sequence<std::decay_t<decltype(lhs)>::N>());
    }
    return false;
  }

  /*
   * Creates a new Planes object with the given planes appended to the beginning of the planes array
   * Parameters:
   * @tparam OtherPlaneTypes: types of planes that will be appended to the array
   * @param other_planes: plane objects that will be appended to the array
   * @return: new Planes object
   */
  template <typename... OtherPlaneTypes>
  [[nodiscard]] auto push_front(OtherPlaneTypes&&... other_planes) const {
    return [&other_planes..., this]<std::size_t... is>(std::index_sequence<is...>) {
      return Planes<std::remove_cvref_t<OtherPlaneTypes>..., PlaneTypes...>{other_planes..., std::get<is>(planes)...};
    }(std::make_index_sequence<N>());
  }

  /*
   * Creates a new Planes object with the given planes appended to the end of the planes array
   * Parameters:
   * @tparam OtherPlaneTypes: types of planes that will be appended to the array
   * @param other_planes: plane objects that will be appended to the array
   * @return: new Planes object
   */
  template <typename... OtherPlaneTypes>
  [[nodiscard]] auto push_back(OtherPlaneTypes&&... other_planes) const {
    return [&other_planes..., this]<std::size_t... is>(std::index_sequence<is...>) {
      return Planes<PlaneTypes..., std::remove_cvref_t<OtherPlaneTypes>...>{std::get<is>(planes)..., other_planes...};
    }(std::make_index_sequence<N>());
  }

  /*
   * Creates a new Planes object with the given plane inserted onto the specified position in the planes array
   * Parameters:
   * @tparam idx: position in the array on which the plane will be inserted
   * @tparam OtherPlaneType: type of the plane that will be inserted into the array
   * @param other_plane: plane object that will be inserted into the array
   * @return: new Planes object
   * Constraints:
   * The given position has to be less than or equal to the planes array size
   */
  template <std::size_t idx, typename OtherPlaneType>
    requires(idx <= N)
  [[nodiscard]] auto insert(OtherPlaneType&& other_plane) const {
    constexpr auto next_plane = []<std::size_t current_idx>(auto plane, auto planes) {
      if constexpr (current_idx < idx)
        return std::get<current_idx>(planes);
      else if constexpr (current_idx == idx)
        return plane;
      else
        return std::get<current_idx - 1u>(planes);
    };

    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return Planes<std::decay_t<decltype(next_plane.template operator()<is>(other_plane, planes))>...>{
          next_plane.template operator()<is>(other_plane, planes)...};
    }(std::make_index_sequence<N + 1u>());
  }

  /*
   * Creates a new Planes object without the specified planes
   * Parameters:
   * @tparam is: indexes of the planes that should be removed
   * @return: filtered Planes object
   * Constraints:
   * Number of planes that are being removed cannot be larger than the size of the planes array
   * Indexes of the planes that should be removed have to be in bounds of 0 and size of planes array
   * All indexes have to be unique
   */
  template <std::size_t... is>
    requires(sizeof...(is) <= N && ((is < N) && ...) && all_unique<is...>())
  [[nodiscard]] auto remove() const {
    static constexpr auto predicate = [](std::size_t idx) consteval { return ((idx == is) || ...); };
    static constexpr auto idx_sequence = filter_index_sequence(std::make_index_sequence<N>(), predicate);
    return [this]<std::size_t... idxs>(std::index_sequence<idxs...>) {
      return Planes<std::decay_t<decltype(std::get<idxs>(planes))>...>{std::get<idxs>(planes)...};
    }(idx_sequence);
  }

  /*
   * Creates a new Planes object in which the plane on the specified position is replaced with the given plane
   * Parameters:
   * @tparam idx: position of the plane that has to be replaced
   * @tparam OtherPlaneType: type of the new plane
   * @param other_plane: new plane that will replace the current one
   * @return: new Planes object
   * Constraints:
   * The given position has to be less than the planes array size
   */
  template <std::size_t idx, typename OtherPlaneType>
    requires(idx < N)
  [[nodiscard]] auto replace(OtherPlaneType&& other_plane) const {
    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return Planes<std::decay_t<decltype(constexpr_ternary<idx == is>(other_plane, std::get<is>(planes)))>...>{
          constexpr_ternary<idx == is>(other_plane, std::get<is>(planes))...};
    }(std::make_index_sequence<N>());
  }

  /*
   * Creates a new Planes object which holds only the planes with the specified positions
   * Parameters:
   * @tparam is: indexes of the planes that will be kept
   * @return: filtered Planes object
   * Constraints:
   * The number of indexes has to be less than or equal to the number of planes in the array
   * Indexes of the planes that should be removed have to be in bounds of 0 and size of planes array
   * All indexes have to be unique
   */
  template <std::size_t... is>
    requires(sizeof...(is) <= N && ((is < N) && ...) && all_unique<is...>())
  [[nodiscard]] auto keep() const {
    return Planes<std::decay_t<decltype(std::get<is>(planes))>...>{std::get<is>(planes)...};
  }

  /*
   * Creates a new Plane object by merging two arrays of planes
   * Parameters:
   * @tparam OtherPlaneTypes: types of planes that will be appended to this objects' array
   * @param other: array of planes that will be appended to the current array
   * @return: new planes array
   */
  template <typename... OtherPlaneTypes>
  [[nodiscard]] auto merge(Planes<OtherPlaneTypes...>& other) const {
    return
        [this, &other]<std::size_t... is, std::size_t... us>(std::index_sequence<is...>, std::index_sequence<us...>) {
          return Planes<PlaneTypes..., OtherPlaneTypes...>{std::get<is>(planes)..., other.template plane<us>()...};
        }(std::make_index_sequence<sizeof...(PlaneTypes)>(),
          std::make_index_sequence<std::decay_t<decltype(other)>::size()>());
  }

  /*
   * Splits an array of planes into several smaller arrays
   * In example, if we have an array of 6 elements, and we pass indexes 1, 3, 5, we'll get 4 new arrays (0), (1, 2), (3,
   * 4), (5) Parameters:
   * @tparam is: indexes at which the split should be performed
   * Constraints:
   * The number of indexes has to be less than or equal to the number of planes in the array
   * Indexes have to be in bounds of 0 and size of planes array
   * The indexes have to be monotonically increasing (the next index cannot be les than the current index)
   */
  template <std::size_t... is>
    requires(sizeof...(is) <= N && ((is < N) && ...) && is_sequence_monotonically_increasing<is...>())
  [[nodiscard]] auto split() const {
    if constexpr (nth_element<0u, is...>() != 0u) {
      return split_impl<0u, is...>();
    } else {
      return split_impl<is...>();
    }
  }

  /*
   * Returns a reference to the plane at the specified location
   * Parameters:
   * @tparam idx: location of the plane to retrieve
   * @return: reference to the plane at the specified location
   * Constraints:
   * The requested plane index has to be less than the size of the planes' array
   */
  template <std::size_t idx>
    requires(idx < N)
  [[nodiscard]] constexpr auto& plane() noexcept {
    return std::get<idx>(planes);
  }

  /*
   * Returns a const reference to the plane at the specified location
   * Parameters:
   * @tparam idx: location of the plane to retrieve
   * @return: const reference to the plane at the specified location
   * Constraints:
   * The requested plane index has to be less than the size of the planes' array
   */
  template <std::size_t idx>
    requires(idx < N)
  [[nodiscard]] constexpr const auto& plane() const noexcept {
    return std::get<idx>(planes);
  }

  /*
   * Returns the total number of planes held by this object
   * Parameters:
   * @return: total number of planes
   */
  [[nodiscard]] static consteval std::size_t size() noexcept { return N; }
};

/*
 * Helper method used for creating an array of planes
 * Parameters:
 * @tparam PlanesTypes: Types of planes that will be found inside the array
 * @return: new Planes object
 */
template <typename... PlaneTypes>
[[nodiscard]] auto create_planes(PlaneTypes&&... planes) {
  return Planes<std::decay_t<PlaneTypes>...>{planes...};
}

}  // namespace ntensor
