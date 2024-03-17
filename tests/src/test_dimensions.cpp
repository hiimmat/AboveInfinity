#include <catch2/catch_test_macros.hpp>
#include <dimensions.hpp>

namespace nt = ntensor;

TEST_CASE("Dimensions class tests") {
  SECTION("method at") {
    static constexpr nt::Dimensions<2u, 4u, 6u, 9u> dimensions;

    REQUIRE(dimensions.rank() == 4u);
    CHECK(dimensions.template at<0u>() == 2u);
    CHECK(dimensions.template at<1u>() == 4u);
    CHECK(dimensions.template at<2u>() == 6u);
    CHECK(dimensions.template at<3u>() == 9u);
  }

  SECTION("method rank") {
    static constexpr nt::Dimensions<1u> rank_one;
    CHECK(rank_one.rank() == 1u);

    static constexpr nt::Dimensions<1u, 1u> rank_two;
    CHECK(rank_two.rank() == 2u);

    static constexpr nt::Dimensions<2u, 4u, 7u> rank_three;
    CHECK(rank_three.rank() == 3u);
  }
}
