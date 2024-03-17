#include <catch2/catch_test_macros.hpp>
#include <strides.hpp>

namespace nt = ntensor;

TEST_CASE("Strides class tests") {
  SECTION("method at") {
    static constexpr nt::Strides<1, 4, 12, 24> strides;

    REQUIRE(strides.rank() == 4u);
    CHECK(strides.template at<0u>() == 1);
    CHECK(strides.template at<1u>() == 4);
    CHECK(strides.template at<2u>() == 12);
    CHECK(strides.template at<3u>() == 24);
  }

  SECTION("method rank") {
    static constexpr nt::Strides<1u> rank_one;
    CHECK(rank_one.rank() == 1u);

    static constexpr nt::Strides<1u, 1u> rank_two;
    CHECK(rank_two.rank() == 2u);

    static constexpr nt::Strides<2u, 4u, 7u> rank_three;
    CHECK(rank_three.rank() == 3u);
  }

  SECTION("strides_contain_padding") {
    {
      static constexpr nt::Dimensions<1u> dimensions;
      static constexpr nt::Strides<1> strides;

      REQUIRE(dimensions.rank() == 1u);
      REQUIRE(strides.rank() == 1u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<10u> dimensions;
      static constexpr nt::Strides<1> strides;

      REQUIRE(dimensions.rank() == 1u);
      REQUIRE(strides.rank() == 1u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u> dimensions;
      static constexpr nt::Strides<1, 1> strides;

      REQUIRE(dimensions.rank() == 2u);
      REQUIRE(strides.rank() == 2u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u> dimensions;
      static constexpr nt::Strides<1, 32> strides;

      REQUIRE(dimensions.rank() == 2u);
      REQUIRE(strides.rank() == 2u);
      CHECK(nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 2u> dimensions;
      static constexpr nt::Strides<1, 1, 1> strides;

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 1u> dimensions;
      static constexpr nt::Strides<1, 1, 2> strides;

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 2u> dimensions;
      static constexpr nt::Strides<1, 1, 32> strides;

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<7u, 2u, 4u> dimensions;
      static constexpr nt::Strides<1, 7, 14> strides;

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<7u, 2u, 4u> dimensions;
      static constexpr nt::Strides<1, 32, 64> strides;

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(nt::strides_contain_padding(dimensions, strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 1u, 1u> dimensions;
      static constexpr nt::Strides<1, 1, 2, 2> strides;

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
    }
  }

  SECTION("contains_negative_strides") {
    {
      static constexpr nt::Strides<1> strides;

      CHECK(strides.rank() == 1u);
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1> strides;

      CHECK(strides.rank() == 1u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, 1> strides;

      CHECK(strides.rank() == 2u);
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, -1> strides;

      CHECK(strides.rank() == 2u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1, 1> strides;

      CHECK(strides.rank() == 2u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1, -1> strides;

      CHECK(strides.rank() == 2u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, 2, 3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1, 2, 3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, -2, 3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, 2, -3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1, -2, 3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<1, -2, -3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Strides<-1, 2, -3> strides;

      CHECK(strides.rank() == 3u);
      CHECK(nt::contains_negative_strides(strides));
    }
  }

  SECTION("compute_aligned_strides") {
    {
      static constexpr nt::Dimensions<1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u>();

      REQUIRE(dimensions.rank() == 1u);
      REQUIRE(strides.rank() == 1u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u>();

      REQUIRE(dimensions.rank() == 2u);
      REQUIRE(strides.rank() == 2u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u, 1u, 2u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u, 1u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 1);
      CHECK(strides.template at<4u>() == 32);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 2u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 32);
      CHECK(strides.template at<2u>() == 64);
      CHECK(strides.template at<3u>() == 64);
      CHECK(strides.template at<4u>() == 64);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u, 2u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 32);
      CHECK(strides.template at<3u>() == 64);
      CHECK(strides.template at<4u>() == 64);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 2u, 1u, 2u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 1u, 2u, 1u, 2u, 1u>();

      REQUIRE(dimensions.rank() == 6u);
      REQUIRE(strides.rank() == 6u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 32);
      CHECK(strides.template at<3u>() == 64);
      CHECK(strides.template at<4u>() == 64);
      CHECK(strides.template at<5u>() == 128);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 3u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 1u, 2u, 3u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 32);
      CHECK(strides.template at<2u>() == 64);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 2u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 6u, 4u, 2u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 32);
      CHECK(strides.template at<2u>() == 128);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr auto strides = nt::compute_aligned_strides<char, 6u, 4u, 2u>();

      static constexpr auto dimensions = nt::Dimensions<6u, 4u, 2u>();
      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 512);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 2u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<float, 6u, 4u, 2u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 32);
      CHECK(strides.template at<2u>() == 128);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 2u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<double, 6u, 4u, 2u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 16);
      CHECK(strides.template at<2u>() == 64);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 1u, 2u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 6u, 4u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 32);
      CHECK(strides.template at<2u>() == 128);
      CHECK(strides.template at<3u>() == 128);
      CHECK(nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<128u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 128u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 128);
      CHECK(strides.template at<3u>() == 128);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<128u, 2u, 4u> dimensions;
      static constexpr auto strides = nt::compute_aligned_strides<int, 128u, 2u, 4u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 256);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }
  }

  SECTION("compute_unaligned_strides") {
    {
      static constexpr nt::Dimensions<1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u>();

      REQUIRE(dimensions.rank() == 1u);
      REQUIRE(strides.rank() == 1u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u>();

      REQUIRE(dimensions.rank() == 2u);
      REQUIRE(strides.rank() == 2u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 1u, 1u, 2u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u, 1u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 1);
      CHECK(strides.template at<4u>() == 1);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 2u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 2);
      CHECK(strides.template at<3u>() == 2);
      CHECK(strides.template at<4u>() == 2);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 2u, 1u, 2u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u, 2u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 5u);
      REQUIRE(strides.rank() == 5u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 2);
      CHECK(strides.template at<4u>() == 2);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 1u, 2u, 1u, 2u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 1u, 2u, 1u, 2u, 1u>();

      REQUIRE(dimensions.rank() == 6u);
      REQUIRE(strides.rank() == 6u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 1);
      CHECK(strides.template at<3u>() == 2);
      CHECK(strides.template at<4u>() == 2);
      CHECK(strides.template at<5u>() == 4);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<1u, 2u, 3u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<1u, 2u, 3u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
      CHECK(strides.template at<2u>() == 2);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 2u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<6u, 4u, 2u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 6);
      CHECK(strides.template at<2u>() == 24);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<6u, 4u, 1u, 2u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<6u, 4u, 1u, 2u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 6);
      CHECK(strides.template at<2u>() == 24);
      CHECK(strides.template at<3u>() == 24);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<128u, 1u, 1u, 1u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<128u, 1u, 1u, 1u>();

      REQUIRE(dimensions.rank() == 4u);
      REQUIRE(strides.rank() == 4u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 128);
      CHECK(strides.template at<3u>() == 128);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }

    {
      static constexpr nt::Dimensions<128u, 2u, 4u> dimensions;
      static constexpr auto strides = nt::compute_unaligned_strides<128u, 2u, 4u>();

      REQUIRE(dimensions.rank() == 3u);
      REQUIRE(strides.rank() == 3u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 256);
      CHECK(!nt::strides_contain_padding(dimensions, strides));
      CHECK(!nt::contains_negative_strides(strides));
    }
  }

  SECTION("partially_compute_strides") {
    {
      static constexpr nt::Dimensions<1u> partial_dimensions;
      static constexpr nt::Strides<1> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 2u);
      REQUIRE(partial_dimensions.rank() == 1u);
      REQUIRE(partial_strides.rank() == 1u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 1);
    }

    {
      static constexpr nt::Dimensions<10u> partial_dimensions;
      static constexpr nt::Strides<1> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 2u);
      REQUIRE(partial_dimensions.rank() == 1u);
      REQUIRE(partial_strides.rank() == 1u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 10);
    }

    {
      constexpr nt::Dimensions<6u> partial_dimensions;
      constexpr nt::Strides<1, 128> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 3u);
      REQUIRE(partial_dimensions.rank() == 1u);
      REQUIRE(partial_strides.rank() == 2u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 768);
    }

    {
      constexpr nt::Dimensions<4u, 6u> partial_dimensions;
      constexpr nt::Strides<1> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 3u);
      REQUIRE(partial_dimensions.rank() == 2u);
      REQUIRE(partial_strides.rank() == 1u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 4);
      CHECK(strides.template at<2u>() == 24);
    }

    {
      constexpr nt::Dimensions<4u, 6u> partial_dimensions;
      constexpr nt::Strides<1, 128> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 4u);
      REQUIRE(partial_dimensions.rank() == 2u);
      REQUIRE(partial_strides.rank() == 2u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 512);
      CHECK(strides.template at<3u>() == 3072);
    }

    {
      constexpr nt::Dimensions<4u, 1u, 1u, 6u, 1u> partial_dimensions;
      constexpr nt::Strides<1, 128> partial_strides;
      static constexpr auto strides = nt::partially_compute_strides(partial_dimensions, partial_strides);

      REQUIRE(strides.rank() == 7u);
      REQUIRE(partial_dimensions.rank() == 5u);
      REQUIRE(partial_strides.rank() == 2u);
      CHECK(strides.template at<0u>() == 1);
      CHECK(strides.template at<1u>() == 128);
      CHECK(strides.template at<2u>() == 512);
      CHECK(strides.template at<3u>() == 512);
      CHECK(strides.template at<4u>() == 512);
      CHECK(strides.template at<5u>() == 3072);
      CHECK(strides.template at<6u>() == 3072);
    }
  }
}
