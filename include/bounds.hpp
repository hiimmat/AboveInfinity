#pragma once

#include <cstddef>

namespace ntensor {

/*
 * Class that finds the minimum and maximum values for a given input
 */
template <std::size_t...>
struct bounds;

/*
 * Specialization of the bounds class with a single element (representing a tight bound)
 * Parameters:
 * @tparam bound: element that simultaneously represents the lower and upper bound of a function
 */
template <std::size_t bound>
struct bounds<bound> {
  [[nodiscard]] static consteval std::size_t lower() noexcept { return bound; }
  [[nodiscard]] static consteval std::size_t upper() noexcept { return bound; }
};

/*
 * Specialization of the bounds class with at least two elements
 * This specialization takes the first two elements of an index sequence and creates a lower and an upper bound from it
 * Parameters:
 * @tparam _lower: element that represents the lower bound of a function
 * @tparam _upper: element that represents the upper bound of a function
 * @tparam rest: rest of the index sequence
 * Constraints:
 * The upper bound has to be equal to or larger than the lower bound
 */
template <std::size_t _lower, std::size_t _upper, std::size_t... rest>
  requires(_upper >= _lower)
struct bounds<_lower, _upper, rest...> {
  [[nodiscard]] static consteval std::size_t lower() noexcept { return _lower; }
  [[nodiscard]] static consteval std::size_t upper() noexcept { return _upper; }
};

}  // namespace ntensor
