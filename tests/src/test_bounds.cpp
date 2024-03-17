#include <bounds.hpp>
#include <catch2/catch_test_macros.hpp>

namespace nt = ntensor;

TEST_CASE("bounds class tests") {
  SECTION("bound with single template parameter") {
    {
      static constexpr nt::bounds<0u> bounds;

      REQUIRE(bounds.lower() == 0u);
      REQUIRE(bounds.upper() == 0u);
    }

    {
      static constexpr nt::bounds<2u> bounds;

      REQUIRE(bounds.lower() == 2u);
      REQUIRE(bounds.upper() == 2u);
    }

    {
      static constexpr nt::bounds<4u> bounds;

      REQUIRE(bounds.lower() == 4u);
      REQUIRE(bounds.upper() == 4u);
    }
  }

  SECTION("bound with two template parameters") {
    {
      static constexpr nt::bounds<1u, 3u> bounds;

      REQUIRE(bounds.lower() == 1u);
      REQUIRE(bounds.upper() == 3u);
    }

    {
      static constexpr nt::bounds<2u, 4u> bounds;

      REQUIRE(bounds.lower() == 2u);
      REQUIRE(bounds.upper() == 4u);
    }

    {
      static constexpr nt::bounds<7u, 12u> bounds;

      REQUIRE(bounds.lower() == 7u);
      REQUIRE(bounds.upper() == 12u);
    }
  }

  SECTION("bound with multiple template parameters") {
    {
      static constexpr nt::bounds<5u, 7u, 9u> bounds;

      REQUIRE(bounds.lower() == 5u);
      REQUIRE(bounds.upper() == 7u);
    }

    {
      static constexpr nt::bounds<1u, 6u, 4u, 12u> bounds;

      REQUIRE(bounds.lower() == 1u);
      REQUIRE(bounds.upper() == 6u);
    }

    {
      static constexpr nt::bounds<3u, 7u, 12u, 14u, 0u, 21u, 5u, 8u> bounds;

      REQUIRE(bounds.lower() == 3u);
      REQUIRE(bounds.upper() == 7u);
    }
  }
}
