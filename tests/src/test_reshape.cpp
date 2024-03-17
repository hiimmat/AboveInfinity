#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <reshape.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

TEST_CASE("reshape method tests") {
  SECTION("identical dimensions") {
    static constexpr nt::Dimensions<6, 2, 4> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    auto reshaped_tensor = nt::reshape<dimensions>(tensor);
    static_assert(std::is_same_v<decltype(reshaped_tensor), decltype(tensor)>);
  }

  SECTION("reshape with padding between the first and the second dimension") {
    static constexpr nt::Dimensions<21, 2, 4> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

    static constexpr nt::Dimensions<3, 7, 2, 4> reshaped_dimensions;
    auto reshaped_tensor = nt::reshape<reshaped_dimensions>(tensor);
    auto& reshaped_plane = reshaped_tensor.planes().template plane<0u>();

    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.dimensions())>,
                                 std::decay_t<decltype(reshaped_dimensions)>>);
    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.strides())>, nt::Strides<1, 3, 32, 64>>);
  }

  SECTION("reshape with no padding") {
    static constexpr nt::Dimensions<32, 2, 4> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    static constexpr nt::Dimensions<1, 2, 8, 2, 2, 4> reshaped_dimensions;
    auto reshaped_tensor = nt::reshape<reshaped_dimensions>(tensor);
    auto& reshaped_plane = reshaped_tensor.planes().template plane<0u>();
    CHECK(!strides_contain_padding(reshaped_plane.dimensions(), reshaped_plane.strides()));
    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.dimensions())>,
                                 std::decay_t<decltype(reshaped_dimensions)>>);
    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.strides())>, nt::Strides<1, 1, 2, 16, 32, 64>>);
  }

  // worst case scenario reshape
  SECTION("reshape with allocation") {
    static constexpr nt::Dimensions<5, 2, 4> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    static constexpr nt::Dimensions<10, 2, 2> reshaped_dimensions;
    auto fn = [](auto& v) {
      static int i = 0;
      v = i++;
    };

    nt::execute(fn, tensor);

    auto reshaped_tensor = nt::reshape<reshaped_dimensions>(tensor);
    auto& reshaped_plane = reshaped_tensor.planes().template plane<0u>();

    CHECK(strides_contain_padding(reshaped_plane.dimensions(), reshaped_plane.strides()));
    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.dimensions())>,
                                 std::decay_t<decltype(reshaped_dimensions)>>);
    static_assert(std::is_same_v<std::decay_t<decltype(reshaped_plane.strides())>, nt::Strides<1, 32, 64>>);

    int value = 0;

    for (std::size_t k = 0u; k < 2u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 10u; ++i) {
          CHECK(reshaped_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }
  }
}
