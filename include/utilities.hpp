#pragma once

#include <algorithm>
#include <array>

#include "concepts.hpp"

namespace ntensor {
inline namespace internal {

template <std::size_t, typename...>
struct nth_type;

/*
 * Base case of the nth_type structure
 * Parameters:
 * @tparam T: First type in the type list
 * @tparam Ts: Rest of the type list
 */
template <typename T, typename... Ts>
struct nth_type<0u, T, Ts...> {
  using type = T;
};

/*
 * Finds the nth type in a type list using recursion
 * Parameters:
 * @tparam N: position of the type inside the type list that needs to be extracted
 * @tparam T: First type in the type list
 * @tparam Ts: Rest of the type list
 */
template <std::size_t N, typename T, typename... Ts>
struct nth_type<N, T, Ts...> {
  using type = typename nth_type<N - 1u, Ts...>::type;
};

/*
 * Base case for the nth_type specialization that uses template template parameters
 * Parameters:
 * @tparam T: template containing a type list
 * @tparam U: First type in the type list
 * @tparam Us: Rest of the type list
 */
template <template <typename...> typename T, typename U, typename... Us>
struct nth_type<0u, T<U, Us...>> {
  using type = U;
};

/*
 * Finds the nth type in a type list that is found inside a template parameter, using recursion
 * Parameters:
 * @tparam N: position of the type inside the type list that needs to be extracted
 * @tparam T: template containing a type list
 * @tparam U: First type in the type list
 * @tparam Us: Rest of the type list
 */
template <std::size_t N, template <typename...> typename T, typename U, typename... Us>
struct nth_type<N, T<U, Us...>> {
  using type = typename nth_type<N, U, Us...>::type;
};

/*
 * Returns the type found on the specified position inside a type list
 * Parameters:
 * @tparam N: position of the type inside the type list that needs to be extracted
 * @tparam Ts: type list
 */
template <std::size_t N, typename... Ts>
using nth_type_t = typename nth_type<N, Ts...>::type;

template <auto...>
struct fis;

/*
 * fis - ancronym for first in sequence
 * Extracts the first type and the first value found in a value sequence
 * Parameters:
 * @tparam v: first value in the sequence
 * @tparam vs: rest of the sequence
 */
template <auto v, auto... vs>
struct fis<v, vs...> {
  using type = decltype(v);
  static constexpr auto value = v;
};

/*
 * fis_t - acronym for first in sequence type
 * Returns the first element type found in a value sequence
 * Parameters:
 * @tparam s: value sequence
 */
template <auto... vs>
using fis_t = typename fis<vs...>::type;

/*
 * fis_v - acronym for first in sequence value
 * Returns the first element found in a value sequence
 * Parameters:
 * @tparam s: value sequence
 */
template <auto... vs>
inline constexpr auto fis_v = fis<vs...>::value;

template <typename...>
struct fts;

/*
 * fts - acronym for first type in type sequence
 * Extracts the first type found in a type sequence
 * Parameters:
 * @tparam T: first type in the sequence
 * @tparam Ts: rest of the sequence
 */
template <typename T, typename... Ts>
struct fts<T, Ts...> {
  using type = T;
};

/*
 * Alias for fts::type
 * Parameters:
 * @tparam Ts: type sequence
 */
template <typename... Ts>
using fts_t = typename fts<Ts...>::type;

/*
 * Retrieves the nth element in a value sequence
 * Parameters:
 * @tparam N: index of the element that needs to be retrieved
 * @tparam vs: value sequence
 * @return: the element at the specified position
 * Constraints:
 * The requested element has to be inside the boundaries of the sequence
 * All elements have to have a common type
 * The type has to be default constructible
 */
template <std::size_t N, auto... vs>
  requires(N < sizeof...(vs) && ((std::same_as<fis_t<vs...>, decltype(vs)>) && ...) &&
           (std::default_initializable<fis_t<vs...>>))
[[nodiscard]] consteval auto nth_element() noexcept {
  std::size_t i = 0u;
  fis_t<vs...> result;
  ((i++ == N ? result = vs : 0) || ...);
  return result;
}

/*
 * Finds the index of the first element in a value sequence that matches a predicate
 * Warning:
 * If find_with_predicate fails to find the element, it returns an index outside of the value sequence, equal to the
 * value sequence size
 * Parameters:
 * @tparam vs: value sequence in which the element is searched for
 * @tparam predicate: callable object that checks if an element satisfies a required condition
 * @return: index of the first element satisfying the requried condition
 * Constraints:
 * All elements have to have a common type
 * The predicate has to be a callable that takes one element of the same type as the elements in the sequence and
 * returns a boolean value
 */
template <auto... vs, std::predicate<fis_t<vs...>> Predicate>
[[nodiscard]] consteval auto find_with_predicate(Predicate predicate) noexcept {
  // the method was written for use in compute_aligned_strides
  // if, for an example, all of the dimensions used in that method are ones, this assertion would fail
  // static_assert((predicate(vs) || ...));
  std::size_t pos{};
  ((predicate(vs) ? false : (++pos, true)) && ...);
  return pos;
};

/*
 * Returns indexes of all elements in a value sequence that matches a predicate
 * Parameters:
 * @tparam vs: value sequence in which the elements are searched for
 * @tparam predicate: callable object that checks if an element satisfies a required condition
 * @return: std::array containing the indexes of all elements satisfying the requried condition
 * Constraints:
 * All elements have to have a common type
 * The predicate has to be a callable that takes one element of the same type as the elements in the sequence and
 * returns a boolean value
 */
template <auto... vs, std::predicate<fis_t<vs...>> Predicate>
  requires((std::same_as<fis_t<vs...>, decltype(vs)>) && ...)
[[nodiscard]] consteval auto find_all_with_predicate(Predicate predicate) noexcept {
  constexpr std::size_t num_matches{0u + ((predicate(vs) ? 1u : 0u) + ...)};
  std::array<std::size_t, num_matches> positions;
  std::size_t idx{0u};
  std::size_t pos{0u};
  ((predicate(vs) ? (positions[idx++] = pos, ++pos) : ++pos), ...);
  return positions;
};

/*
 * Specialization of the find_all_with_predicate method that uses a template template parameter instead of a value
 * sequence directly Parameters:
 * @tparam T: type holding the value sequence
 * @tparam vs: value sequence in which the elements are searched for
 * @tparam predicate: callable object that checks if an element satisfies a required condition
 * @return: index sequence containing the indexes of all elements satisfying the requried condition
 */
template <template <auto...> typename T, auto... vs, std::predicate<std::size_t> Predicate>
[[nodiscard]] consteval auto find_all_with_predicate([[maybe_unused]] T<vs...> values, Predicate predicate) noexcept {
  constexpr auto positions = find_all_with_predicate<vs...>(predicate);
  return [positions]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return std::index_sequence<positions[idxs]...>();
  }(std::make_index_sequence<positions.size()>());
}

/*
 * Computes the product of all elements found inside a value sequence
 * Parameters:
 * @tparam T: type holding the value sequence
 * @tparam vs: value sequence whose elements are used for the multiplication
 * Constraints:
 * The elements have to be multipliable with each other
 */
template <template <auto...> typename T, auto... vs>
  requires(multiplicative<vs...>)
consteval auto product(T<vs...>) noexcept {
  return (vs * ...);
}

/*
 * Computes the largest product of two value sequences
 * This methods' purpose is to determe the allocation size of a tensor, based on its dimensions and strides
 * Parameters:
 * @tparam Dimensions: tensors' dimensions
 * @tparam Strides: tensors' strides
 * @tparam ds: dimensions sequence
 * @tparam ss: strides sequence
 * @return: tensors' size based on its dimensions and strides
 * Constraints:
 * the sequences ds and ss have to have the same size
 */
template <template <auto...> typename Dimensions, template <auto...> typename Strides, std::size_t... ds,
          long long... ss>
  requires(sizeof...(ds) == sizeof...(ss))
[[nodiscard]] consteval long long max_product(Dimensions<ds...> dimensions, Strides<ss...> strides) noexcept {
  long long size = dimensions.template at<0u>() * strides.template at<0u>();
  (((static_cast<long long>(ds) * ss) > size ? size = (static_cast<long long>(ds) * ss) : 0), ...);
  return size;
}

/*
 * Computes the minimum index of a tensor that can be accessed using its dimensions and strides
 * Parameters:
 * @tparam Dimensions: tensors' dimensions
 * @tparam Strides: tensors' strides
 * @tparam ds: dimensions sequence
 * @tparam ss: strides sequence
 * @return: the lowest index accessible in a tensor based on its dimensions and strides
 * Constraints:
 * the sequences ds and ss have to have the same size
 */
template <template <std::integral auto...> typename Dimensions, template <std::integral auto...> typename Strides,
          std::integral auto... ds, std::integral auto... ss>
  requires(sizeof...(ds) == sizeof...(ss))
[[nodiscard]] consteval auto min_index(Dimensions<ds...>, Strides<ss...>) noexcept {
  // Ignore dimensions that have a positive strides since they can only increase the minimum index.
  // The minimum index is either negative (if we have negative strides) or 0. It can't be higher than 0, since each
  // dimension that we access starts from 0.
  return ((ss < 0 ? static_cast<long long>(ds - 1u) * ss : 0) + ...);
}

/*
 * Computes the maximum index of a tensor that can be accessed using its dimensions and strides
 * Parameters:
 * @tparam Dimensions: tensors' dimensions
 * @tparam Strides: tensors' strides
 * @tparam ds: dimensions sequence
 * @tparam ss: strides sequence
 * @return: the highest index accessible in a tensor based on its dimensions and strides
 * Constraints:
 * the sequences ds and ss have to have the same size
 */
template <template <std::integral auto...> typename Dimensions, template <std::integral auto...> typename Strides,
          std::integral auto... ds, std::integral auto... ss>
  requires(sizeof...(ds) == sizeof...(ss))
[[nodiscard]] consteval std::size_t max_index(Dimensions<ds...>, Strides<ss...>) noexcept {
  // Ignore dimensions that have a negative stride since they can only reduce the maximum index.
  // The maximum index has to be >= 0. It can't be negative (even if negative strides are used), since each dimension
  // that we access starts from 0.
  return ((ss > 0 ? (ds - 1u) * ss : 0) + ...);
}

/*
 * Compares each element to all of the other elements in the value sequence, to confirm that the sequence does not
 * contain any duplicates
 * @tparam head: first element in the value sequence
 * @tparam tail: rest of the value sequence
 * @return: true if the value sequence contains no duplicate elements, false otherwise
 */
template <auto head, auto... tail>
[[nodiscard]] consteval bool all_unique() noexcept {
  if constexpr (sizeof...(tail) > 0U)
    return (((head != tail) && ...)) && all_unique<tail...>();
  else
    return true;
}

/*
 * Checks if a value sequence is monotonically increasing
 * We say that a sequence is monotonically increasing if the current element is equal to or less than the next element
 * @tparam head: first element in the value sequence
 * @tparam tail: rest of the value sequence
 * @return: true if the sequence is monotonically increasing, false otherwise
 */
template <auto head, auto... tail>
[[nodiscard]] consteval bool is_sequence_monotonically_increasing() noexcept {
  if constexpr (sizeof...(tail) > 0U)
    return (head <= fis_v<tail...>)&&is_sequence_monotonically_increasing<tail...>();
  else
    return true;
}

/*
 * Removes elements from a value sequence that match a predicate
 * Parameters:
 * @tparam vs: value sequence that needs to be filtered
 * @param predicate: callable object that checks if the current element in the value sequence should be kept or removed
 * @return: std::array containing the filtered value sequence
 * Constraints:
 * The predicate has to be a callable that takes a std::size_t value and returns a boolean value
 * The value sequences cannot be empty
 * All elements have to have a common type
 * The type has to be default constructible
 */
template <auto... vs, std::predicate<std::size_t> Predicate>
  requires(((std::same_as<fis_t<vs...>, decltype(vs)>) && ...) && (std::default_initializable<fis_t<vs...>>))
[[nodiscard]] consteval auto filter_value_sequence(Predicate predicate) noexcept {
  constexpr std::size_t num_matches = (0u + ... + (predicate(vs) ? 1u : 0u));
  constexpr std::size_t filtered_size = sizeof...(vs) - num_matches;
  std::array<fis_t<vs...>, filtered_size> filtered_array;
  std::size_t idx{0u};
  ((predicate(vs) ? void() : (filtered_array[idx++] = vs, void())), ...);
  return filtered_array;
}

/*
 * Specialization of the filter_value_sequence method that uses a template template parameter instead of a value
 * sequence directly Parameters:
 * @tparam T: type holding the value sequence
 * @tparam vs: value sequence that needs to be filtered
 * @tparam predicate: callable object that checks if the current element in the value sequence should be kept or removed
 * @return: object of type T holding the filtered value sequence
 */
template <template <auto...> typename T, auto... vs, std::predicate<std::size_t> Predicate>
[[nodiscard]] consteval auto filter_value_sequence(T<vs...> values, Predicate predicate) noexcept {
  constexpr auto filtered_array = filter_value_sequence<vs...>(predicate);
  return [filtered_array]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return T<filtered_array[idxs]...>();
  }(std::make_index_sequence<filtered_array.size()>());
}

/*
 * Removes indexes from an index sequence that match a predicate
 * Parameters:
 * @tparam is: index sequence that needs to be filtered
 * @param predicate: callable object that checks if the current index in the index sequence should be kept or removed
 * @return: std::array containing the filtered index sequence
 * Constraints:
 * The predicate has to be a callable that takes a std::size_t value and returns a boolean value
 */
template <std::size_t... is, std::predicate<std::size_t> Predicate>
[[nodiscard]] consteval auto filter_index_sequence(std::index_sequence<is...>, Predicate predicate) noexcept {
  constexpr auto filtered_array = filter_value_sequence<is...>(predicate);
  return [filtered_array]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return std::index_sequence<filtered_array[idxs]...>();
  }(std::make_index_sequence<filtered_array.size()>());
}

/*
 * Creates a value sequence containing elements that are present in both input sequences
 * Parameters:
 * @tparam lhs: first (left-hand side) value sequence
 * @tparam rhs: second (right-hand side) value sequence
 * @return: value sequence containing elements that are present in both lhs and rhs sequences
 */
template <std::size_t... lhs, std::size_t... rhs>
[[nodiscard]] consteval auto set_intersection(std::index_sequence<lhs...> lsequence,
                                              [[maybe_unused]] std::index_sequence<rhs...> rsequence) noexcept {
  constexpr auto intersect = [](std::size_t v) consteval { return ((v != rhs) && ...); };

  return filter_index_sequence(lsequence, intersect);
}

/*
 * Removes the element at the specified location from a value sequence
 * Parameters:
 * @tparam N: position of the element that has to be removed
 * @tparam vs: value sequence from which the element will be removed
 * @return: std::array containing the reduced value sequence
 * Constraints:
 * The requested element has to be inside the boundaries of the sequence
 * All elements have to have a common type
 * The type has to be default constructible
 */
template <std::size_t N, auto... vs>
  requires(N < sizeof...(vs) && ((std::same_as<fis_t<vs...>, decltype(vs)>) && ...) &&
           (std::default_initializable<fis_t<vs...>>))
[[nodiscard]] consteval auto remove_nth_element() noexcept {
  std::array<fis_t<vs...>, sizeof...(vs) - 1u> filtered_array;
  std::size_t idx{0u};
  ((idx < N ? filtered_array[idx++] = vs : idx == N ? idx++ : filtered_array[idx++ - 1] = vs), ...);
  return filtered_array;
}

/*
 * Specialization of the remove_nth_element method that uses a template template parameter instead of a value sequence
 * directly Parameters:
 * @tparam N: position of the element that has to be removed
 * @tparam T: type holding the value sequence
 * @tparam vs: value sequence that needs to be reduced
 * @return: object of type T holding the reduced value sequence
 */
template <std::size_t N, template <auto...> typename T, auto... vs>
[[nodiscard]] consteval auto remove_nth_element([[maybe_unused]] T<vs...> values) noexcept {
  constexpr auto filtered_array = remove_nth_element<N, vs...>();
  return [filtered_array]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return T<filtered_array[idxs]...>();
  }(std::make_index_sequence<filtered_array.size()>());
}

/*
 * Implementation of the sort method
 * Parameters:
 * @tparam vs: value sequence to sort
 * @tparam Predicate: method for comparing two objects which determines how the value sequence will be sorted
 * @return: std::array containing the sorted value sequence
 * Constraints:
 * There are no constraints if the given sequence contains 0 or 1 elements
 * If there's more than 1 element in the sequence: all elements have to have a common type, and we must be able to
 * invoke the predicate method using them
 */
template <auto... vs, typename Predicate>
  requires(sizeof...(vs) == 1u ||
           (((std::same_as<fis_t<vs...>, decltype(vs)>) && ...) &&
            std::is_invocable_r_v<bool, Predicate, nth_type_t<0u, decltype(vs)...>, nth_type_t<1u, decltype(vs)...>>))
[[nodiscard]] consteval auto sort(Predicate predicate) noexcept {
  std::array values_array{vs...};
  std::sort(begin(values_array), end(values_array), predicate);
  return values_array;
}

/*
 * Sorts the elements of a value sequence based on the given predicate
 * Parameters:
 * @tparam T: type of the object holding the value sequence
 * @tparam vs: value sequence to sort
 * @tparam Predicate: method for comparing two objects which determines how the value sequence will be sorted
 * @return: new object holding the sorted value sequence
 */
template <template <auto...> typename T, auto... vs, typename Predicate>
[[nodiscard]] consteval auto sort(T<vs...> values, Predicate predicate) noexcept {
  constexpr auto sorted_array = sort<vs...>(predicate);
  return [sorted_array]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return T<sorted_array[idxs]...>();
  }(std::make_index_sequence<sorted_array.size()>());
}

/*
 * Sorts the elements of an index sequence based on the given predicate
 * Parameters:
 * @tparam is: index sequence to sort
 * @tparam Predicate: method for comparing two objects which determines how the index sequence will be sorted
 * @return: sorted index sequence
 */
template <std::size_t... is, typename Predicate>
[[nodiscard]] consteval auto sort(std::index_sequence<is...>, Predicate predicate) noexcept {
  constexpr auto sorted_array = sort<is...>(predicate);
  return [sorted_array]<std::size_t... idxs>(std::index_sequence<idxs...>) {
    return std::index_sequence<sorted_array[idxs]...>();
  }(std::make_index_sequence<sorted_array.size()>());
}

/*
 * Provides ternary conditional operator behavior in a constexpr context
 * The ternary conditional operator doesn't work with different return types. However, this method overcomes that
 * shortcoming Parameters:
 * @tparam evaluated_condition: condition that determines which value will be returned (its value can be either true or
 * false)
 * @param lhs: left-hand side object that's returned if the evaluted_condition is true
 * @param rhs: right-hand side object that's returned if the evaluated_condition is false
 * @return: lhs if evaluated_condition is true, rhs otherwise
 */
template <bool evaluated_condition>
[[nodiscard]] constexpr auto constexpr_ternary(auto lhs, auto rhs) noexcept {
  if constexpr (evaluated_condition)
    return lhs;
  else
    return rhs;
}

/*
 * Creates an index sequence using an lower bound and an upper bound
 * Parameters:
 * @tparam lower_bound: lower bound, also the first element of the index sequence
 * @tparam upper_bound: upper bound, excluded from the sequence. Last element is upper_bound - 1. Unless if both the
 * upper and lower bound are the same
 * @return: newly created index sequence
 * Constraints:
 * upper bound has to be equal to or higher to the lower bound
 */
template <std::size_t lower_bound, std::size_t upper_bound>
  requires(upper_bound >= lower_bound)
[[nodiscard]] consteval auto create_index_sequence() noexcept {
  constexpr std::size_t sequence_size = upper_bound == lower_bound ? 1u : upper_bound - lower_bound;
  return []<std::size_t... is>(std::index_sequence<is...>) -> std::index_sequence<(lower_bound + is)...> {
    return {};
  }(std::make_index_sequence<sequence_size>());
}

/*
 * Reorders a value sequence in the specified order
 * Parameters:
 * @tparam order: new positions of the values in the sequence
 * @tparam T: type holding the value sequence
 * @tparam elems: value sequence that needs to be reordered
 * Constraints:
 * Number of indexes has to be the same as the size of the value sequence
 * All indexes have to be unique
 * Indexes have to be in between 0 and the size of the value sequence
 */
template <std::size_t... order, template <auto...> typename T, auto... elems>
  requires(sizeof...(elems) == sizeof...(order) && all_unique<order...>() && ((order < sizeof...(elems)) && ...))
[[nodiscard]] consteval auto permute(T<elems...>) noexcept {
  return T<nth_element<order, elems...>()...>{};
}

}  // namespace internal
}  // namespace ntensor
