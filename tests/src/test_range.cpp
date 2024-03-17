#include <catch2/catch_test_macros.hpp>
#include <range.hpp>

namespace nt = ntensor;

TEST_CASE("range class tests") {
  SECTION("start and end methods") {
    {
      static constexpr nt::range<0u, 0u> range;
      CHECK(range.start() == 0u);
      CHECK(range.end() == 0u);
    }

    {
      static constexpr nt::range<4u, 4u> range;
      CHECK(range.start() == 4u);
      CHECK(range.end() == 4u);
    }

    {
      static constexpr nt::range<0u, 1u> range;
      CHECK(range.start() == 0u);
      CHECK(range.end() == 1u);
    }

    {
      static constexpr nt::range<2u, 6u> range;
      CHECK(range.start() == 2u);
      CHECK(range.end() == 6u);
    }

    {
      static constexpr nt::range<0u, 7u> range;
      CHECK(range.start() == 0u);
      CHECK(range.end() == 7u);
    }
  }
}
