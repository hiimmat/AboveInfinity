#pragma once

#include <string_view>

#include "concepts.hpp"
#include "utilities.hpp"

namespace ntensor {

/*
 * Dimensions of a multidimensional array
 * Parameters:
 * @tparam dimensions: sequence of dimensions
 * Constraints:
 * The dimensions sequence has to contain at least one dimension
 * Dimensions have to be natural numbers
 */
template <std::size_t... dimensions>
  requires(natural_numbers<dimensions...>)
class Dimensions {
 private:
  static constexpr size_t _rank = sizeof...(dimensions);

 public:
  /*
   * Returns the element at the specified position
   * Parameters:
   * @tparam N: index of the element that needs to be retrieved
   * @return: the element at the specified position found in the dimensions sequence
   */
  template <std::size_t N>
  [[nodiscard]] consteval int at() const noexcept {
    return nth_element<N, dimensions...>();
  }

  /*
   * Returns the total number of dimensions
   * @return: total number of dimensions in the dimensions sequence
   */
  [[nodiscard]] static consteval std::size_t rank() noexcept { return _rank; }
};

}  // namespace ntensor
