#pragma once

namespace ntensor {

/*
 * Structure representing a range
 * Parameters:
 * @tparam _start: first element in a range
 * @tparam _end: last element in a range
 * Constraints:
 * The last element in a range has to be equal to or bigger than the first element in a range
 */
template <std::size_t _start, std::size_t _end>
  requires(_end >= _start)
struct range {
  [[nodiscard]] consteval std::size_t start() const noexcept { return _start; }
  [[nodiscard]] consteval std::size_t end() const noexcept { return _end; }
};

}  // namespace ntensor
