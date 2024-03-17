#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <dimensions.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

template <nt::Dimensions dimensions>
void test_squeeze() {
  auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
  auto squeezed_tensor = tensor.squeeze();
  using squeezed_plane = std::decay_t<decltype(squeezed_tensor.planes().template plane<0u>())>;
  using squeezed_dimensions = decltype(squeezed_plane::dimensions());
  using squeezed_strides = decltype(squeezed_plane::strides());

  static_assert(std::is_same_v<squeezed_dimensions, nt::Dimensions<2u, 3u, 4u>>);
  static_assert(std::is_same_v<squeezed_strides, nt::Strides<1, 2, 6>>);
}

TEST_CASE("ShapeTransmutation interface tests") {
  SECTION("planes") {
    static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
    static constexpr nt::Strides<1, 2, 6> strides;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    {
      auto& planes = tensor.planes();
      static_assert(
          std::is_same_v<decltype(planes), nt::Planes<nt::Plane<nt::DenseBuffer<int>, dimensions, strides>>&>);
    }
    {
      const auto& planes = tensor.planes();
      static_assert(
          std::is_same_v<decltype(planes), const nt::Planes<nt::Plane<nt::DenseBuffer<int>, dimensions, strides>>&>);
    }
  }

  SECTION("permute") {
    static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

    {
      auto permuted = tensor.template permute<0u, 0u, 1u, 2u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<2u, 3u, 4u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<1, 2, 6>>);
    }

    {
      auto permuted = tensor.template permute<0u, 0u, 2u, 1u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<2u, 4u, 3u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<1, 6, 2>>);
    }

    {
      auto permuted = tensor.template permute<0u, 1u, 2u, 0u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<3u, 4u, 2u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<2, 6, 1>>);
    }

    {
      auto permuted = tensor.template permute<0u, 1u, 0u, 2u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<3u, 2u, 4u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<2, 1, 6>>);
    }

    {
      auto permuted = tensor.template permute<0u, 2u, 0u, 1u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<4u, 2u, 3u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<6, 1, 2>>);
    }
    {
      auto permuted = tensor.template permute<0u, 2u, 1u, 0u>();
      using permuted_plane = std::decay_t<decltype(permuted.planes().template plane<0u>())>;
      using permuted_dimensions = decltype(permuted_plane::dimensions());
      using permuted_strides = decltype(permuted_plane::strides());

      static_assert(std::is_same_v<permuted_dimensions, nt::Dimensions<4u, 3u, 2u>>);
      static_assert(std::is_same_v<permuted_strides, nt::Strides<6, 2, 1>>);
    }
  }

  SECTION("slicing_value") {
    {
      static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
      static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            const auto idx =
                i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
            plane[idx] = idx;
            CHECK(tensor.slicing_value(0, k, j, i) == static_cast<int>(idx));
          }
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          const auto val = i * strides.template at<2u>() + j * strides.template at<1u>();
          CHECK(tensor.slicing_value(0, j, i) == static_cast<int>(val));
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        const auto val = i * strides.template at<2u>();
        CHECK(tensor.slicing_value(0, i) == static_cast<int>(val));
      }
    }

    {
      static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 3u, false>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
      static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
      static constexpr auto channels = std::decay_t<decltype(plane)>::channels();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            for (std::size_t c = 0u; c < channels; ++c) {
              const auto idx =
                  (i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>()) *
                      channels +
                  c;
              plane[idx] = idx;
              CHECK(tensor.slicing_value(c, k, j, i) == static_cast<int>(idx));
            }
          }
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t c = 0u; c < channels; ++c) {
            const auto val = (i * strides.template at<2u>() + j * strides.template at<1u>()) * channels + c;
            CHECK(tensor.slicing_value(c, j, i) == static_cast<int>(val));
          }
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t c = 0u; c < channels; ++c) {
          const auto val = (i * strides.template at<2u>()) * channels + c;
          CHECK(tensor.slicing_value(c, i) == static_cast<int>(val));
        }
      }
    }
  }

  SECTION("slice") {
    {
      static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
      static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            const auto idx =
                i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
            plane[idx] = idx;
          }
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        auto outer_slice = tensor.template slice<2u>(i);
        auto outer_slice_plane = outer_slice.planes().template plane<0u>();
        using outer_slice_dimensions = decltype(std::decay_t<decltype(outer_slice_plane)>::dimensions());
        using outer_slice_strides = decltype(std::decay_t<decltype(outer_slice_plane)>::strides());
        const auto outer_slice_offset = outer_slice_plane.offset();
        static_assert(std::is_same_v<outer_slice_dimensions, nt::Dimensions<2u, 3u>>);
        static_assert(std::is_same_v<outer_slice_strides, nt::Strides<1, 2>>);
        CHECK(outer_slice_offset == static_cast<long long>(i * strides.template at<2u>()));
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          auto mid_slice = outer_slice.template slice<1u>(j);
          auto mid_slice_plane = mid_slice.planes().template plane<0u>();
          using mid_slice_dimensions = decltype(std::decay_t<decltype(mid_slice_plane)>::dimensions());
          using mid_slice_strides = decltype(std::decay_t<decltype(mid_slice_plane)>::strides());
          const auto mid_slice_offset = mid_slice_plane.offset();
          static_assert(std::is_same_v<mid_slice_dimensions, nt::Dimensions<2u>>);
          static_assert(std::is_same_v<mid_slice_strides, nt::Strides<1>>);
          CHECK(mid_slice_offset == static_cast<long long>(outer_slice_offset + j * strides.template at<1u>()));
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        auto slice = tensor.template slice<2u>(i);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<2u, 3u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<1, 2>>);
        CHECK(slice_offset == static_cast<long long>(i * strides.template at<2u>()));
      }

      for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
        auto slice = tensor.template slice<1u>(j);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<2u, 4u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<1, 6>>);
        CHECK(slice_offset == static_cast<long long>(j * strides.template at<1u>()));
      }

      for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
        auto slice = tensor.template slice<0u>(k);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<3u, 4u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<2, 6>>);
        CHECK(slice_offset == static_cast<long long>(k * strides.template at<0u>()));
      }
    }

    {
      static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
      static constexpr auto strides = std::decay_t<decltype(plane)>::strides();
      static constexpr auto channels = std::decay_t<decltype(plane)>::channels();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            for (std::size_t c = 0u; c < channels; ++c) {
              const auto idx =
                  (i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>()) *
                      channels +
                  c;
              plane[idx] = idx;
            }
          }
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        auto outer_slice = tensor.template slice<2u>(i);
        auto outer_slice_plane = outer_slice.planes().template plane<0u>();
        using outer_slice_dimensions = decltype(std::decay_t<decltype(outer_slice_plane)>::dimensions());
        using outer_slice_strides = decltype(std::decay_t<decltype(outer_slice_plane)>::strides());
        const auto outer_slice_offset = outer_slice_plane.offset();
        static_assert(std::is_same_v<outer_slice_dimensions, nt::Dimensions<2u, 3u>>);
        static_assert(std::is_same_v<outer_slice_strides, nt::Strides<1, 2>>);
        CHECK(outer_slice_offset == static_cast<long long>(i * strides.template at<2u>() * channels));
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          auto mid_slice = outer_slice.template slice<1u>(j);
          auto mid_slice_plane = mid_slice.planes().template plane<0u>();
          using mid_slice_dimensions = decltype(std::decay_t<decltype(mid_slice_plane)>::dimensions());
          using mid_slice_strides = decltype(std::decay_t<decltype(mid_slice_plane)>::strides());
          const auto mid_slice_offset = mid_slice_plane.offset();
          static_assert(std::is_same_v<mid_slice_dimensions, nt::Dimensions<2u>>);
          static_assert(std::is_same_v<mid_slice_strides, nt::Strides<1>>);
          CHECK(mid_slice_offset ==
                static_cast<long long>((outer_slice_offset + j * strides.template at<1u>()) * channels));
        }
      }

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        auto slice = tensor.template slice<2u>(i);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<2u, 3u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<1, 2>>);
        CHECK(slice_offset == static_cast<long long>(i * strides.template at<2u>() * channels));
      }

      for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
        auto slice = tensor.template slice<1u>(j);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<2u, 4u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<1, 6>>);
        CHECK(slice_offset == static_cast<long long>(j * strides.template at<1u>() * channels));
      }

      for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
        auto slice = tensor.template slice<0u>(k);
        auto slice_plane = slice.planes().template plane<0u>();
        using slice_dimensions = decltype(std::decay_t<decltype(slice_plane)>::dimensions());
        using slice_strides = decltype(std::decay_t<decltype(slice_plane)>::strides());
        const auto slice_offset = slice_plane.offset();
        static_assert(std::is_same_v<slice_dimensions, nt::Dimensions<3u, 4u>>);
        static_assert(std::is_same_v<slice_strides, nt::Strides<2, 6>>);
        CHECK(slice_offset == static_cast<long long>(k * strides.template at<0u>() * channels));
      }
    }
  }

  SECTION("slab") {
    static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
    static constexpr std::size_t channels = 3u;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels, false>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

    {
      auto slab = tensor.template slab<2u, 0u, 3u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 3u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }

    {
      auto slab = tensor.template slab<2u, 0u, 2u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 2u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }

    {
      auto slab = tensor.template slab<2u, 0u, 1u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 1u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }

    {
      auto slab = tensor.template slab<2u, 1u, 3u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 2u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto slab = tensor.template slab<2u, 2u, 3u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 1u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 2 * strides.template at<2u>() * channels);
    }

    {
      auto slab = tensor.template slab<2u, 1u, 2u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 3u, 1u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto slab = tensor.template slab<1u, 0u, 2u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 2u, 4u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }

    {
      auto slab = tensor.template slab<1u, 0u, 1u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 1u, 4u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }

    {
      auto slab = tensor.template slab<1u, 1u, 2u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<2u, 1u, 4u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == strides.template at<1u>() * channels);
    }

    {
      auto slab = tensor.template slab<0u, 0u, 1u>();
      auto slab_plane = slab.planes().template plane<0u>();
      using slab_dimensions = decltype(std::decay_t<decltype(slab_plane)>::dimensions());
      using slab_strides = decltype(std::decay_t<decltype(slab_plane)>::strides());
      static_assert(std::is_same_v<slab_dimensions, nt::Dimensions<1u, 3u, 4u>>);
      static_assert(std::is_same_v<slab_strides, std::decay_t<decltype(strides)>>);
      CHECK(slab_plane.offset() == 0);
    }
  }

  SECTION("subspace") {
    static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
    static constexpr std::size_t channels = 3u;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels, false>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    static constexpr auto strides = std::decay_t<decltype(plane)>::strides();

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<1u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == strides.template at<0u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 0u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<1u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<1u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<1u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<2u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<2u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<3u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<1u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<0u, 0u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<0u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<0u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<0u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 4u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<1u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<1u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 1>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<1u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<1u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<2u, 2u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<2u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 1>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<2u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 2u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 2 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<3u, 3u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 3 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<3u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u, 1>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() == 3 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<0u, 3u>{}, nt::range<4u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<2u, 3u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2>>);
      CHECK(subspace_plane.offset() == 3 * strides.template at<2u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<1u, 2u>{}, nt::range<2u, 3u>{}, nt::range<3u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u, 1u, 1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1, 2, 6>>);
      CHECK(subspace_plane.offset() ==
            (strides.template at<0u>() + 2 * strides.template at<1u>() + 3 * strides.template at<2u>()) * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<1u, 2u>{}, nt::range<3u, 3u>{}, nt::range<4u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() ==
            (strides.template at<0u>() + 2 * strides.template at<1u>() + 3 * strides.template at<2u>()) * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<2u, 2u>{}, nt::range<2u, 3u>{}, nt::range<4u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<2>>);
      CHECK(subspace_plane.offset() ==
            (strides.template at<0u>() + 2 * strides.template at<1u>() + 3 * strides.template at<2u>()) * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<2u, 2u>{}, nt::range<3u, 3u>{}, nt::range<3u, 4u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<6>>);
      CHECK(subspace_plane.offset() ==
            (strides.template at<0u>() + 2 * strides.template at<1u>() + 3 * strides.template at<2u>()) * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 1u>{}, nt::range<0u, 0u>{}, nt::range<0u, 0u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 0u>{}, nt::range<0u, 1u>{}, nt::range<0u, 0u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 0u>{}, nt::range<0u, 0u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<6>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 1u>{}, nt::range<0u, 0u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<1>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<0u, 0u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<2>>);
      CHECK(subspace_plane.offset() == 0);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<1u, 1u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<2>>);
      CHECK(subspace_plane.offset() == strides.template at<0u>() * channels);
    }

    {
      auto subspace = tensor.template subspace<0u, nt::range<2u, 2u>{}, nt::range<0u, 1u>{}>();
      auto subspace_plane = subspace.planes().template plane<0u>();
      using subspace_dimensions = decltype(std::decay_t<decltype(subspace_plane)>::dimensions());
      using subspace_strides = decltype(std::decay_t<decltype(subspace_plane)>::strides());
      static_assert(std::is_same_v<subspace_dimensions, nt::Dimensions<1u>>);
      static_assert(std::is_same_v<subspace_strides, nt::Strides<2>>);
      CHECK(subspace_plane.offset() == strides.template at<0u>() * channels);
    }
  }

  SECTION("new_axis") {
    static constexpr nt::Dimensions<2u, 3u, 4u> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

    {
      auto newaxis_tensor = tensor.template new_axis<0u>();
      using newaxis_plane = std::decay_t<decltype(newaxis_tensor.planes().template plane<0u>())>;
      using newaxis_dimensions = decltype(newaxis_plane::dimensions());
      using newaxis_strides = decltype(newaxis_plane::strides());

      static_assert(std::is_same_v<newaxis_dimensions, nt::Dimensions<1u, 2u, 3u, 4u>>);
      static_assert(std::is_same_v<newaxis_strides, nt::Strides<0, 1, 2, 6>>);
    }

    {
      auto newaxis_tensor = tensor.template new_axis<1u>();
      using newaxis_plane = std::decay_t<decltype(newaxis_tensor.planes().template plane<0u>())>;
      using newaxis_dimensions = decltype(newaxis_plane::dimensions());
      using newaxis_strides = decltype(newaxis_plane::strides());

      static_assert(std::is_same_v<newaxis_dimensions, nt::Dimensions<2u, 1u, 3u, 4u>>);
      static_assert(std::is_same_v<newaxis_strides, nt::Strides<1, 0, 2, 6>>);
    }

    {
      auto newaxis_tensor = tensor.template new_axis<2u>();
      using newaxis_plane = std::decay_t<decltype(newaxis_tensor.planes().template plane<0u>())>;
      using newaxis_dimensions = decltype(newaxis_plane::dimensions());
      using newaxis_strides = decltype(newaxis_plane::strides());

      static_assert(std::is_same_v<newaxis_dimensions, nt::Dimensions<2u, 3u, 1u, 4u>>);
      static_assert(std::is_same_v<newaxis_strides, nt::Strides<1, 2, 0, 6>>);
    }

    {
      auto newaxis_tensor = tensor.template new_axis<3u>();
      using newaxis_plane = std::decay_t<decltype(newaxis_tensor.planes().template plane<0u>())>;
      using newaxis_dimensions = decltype(newaxis_plane::dimensions());
      using newaxis_strides = decltype(newaxis_plane::strides());

      static_assert(std::is_same_v<newaxis_dimensions, nt::Dimensions<2u, 3u, 4u, 1u>>);
      static_assert(std::is_same_v<newaxis_strides, nt::Strides<1, 2, 6, 0>>);
    }

    {
      auto first_newaxis_tensor = tensor.template new_axis<0u>();
      auto second_newaxis_tensor = first_newaxis_tensor.template new_axis<1u>();
      auto third_newaxis_tensor = second_newaxis_tensor.template new_axis<3u>();
      auto fourth_newaxis_tensor = third_newaxis_tensor.template new_axis<4u>();
      auto fifth_newaxis_tensor = fourth_newaxis_tensor.template new_axis<6u>();
      auto sixth_newaxis_tensor = fifth_newaxis_tensor.template new_axis<7u>();
      auto seventh_newaxis_tensor = sixth_newaxis_tensor.template new_axis<9u>();
      auto eighth_newaxis_tensor = seventh_newaxis_tensor.template new_axis<10u>();
      using newaxis_plane = std::decay_t<decltype(eighth_newaxis_tensor.planes().template plane<0u>())>;
      using newaxis_dimensions = decltype(newaxis_plane::dimensions());
      using newaxis_strides = decltype(newaxis_plane::strides());

      static_assert(std::is_same_v<newaxis_dimensions, nt::Dimensions<1u, 1u, 2u, 1u, 1u, 3u, 1u, 1u, 4u, 1u, 1u>>);
      static_assert(std::is_same_v<newaxis_strides, nt::Strides<0, 0, 1, 0, 0, 2, 0, 0, 6, 0, 0>>);
    }
  }

  SECTION("squeeze") {
    {
      {
        static constexpr nt::Dimensions<1u, 2u, 3u, 4u> dimensions;
        test_squeeze<dimensions>();
      }

      {
        static constexpr nt::Dimensions<2u, 1u, 3u, 4u> dimensions;
        test_squeeze<dimensions>();
      }

      {
        static constexpr nt::Dimensions<2u, 3u, 1u, 4u> dimensions;
        test_squeeze<dimensions>();
      }

      {
        static constexpr nt::Dimensions<2u, 3u, 4u, 1u> dimensions;
        test_squeeze<dimensions>();
      }

      {
        static constexpr nt::Dimensions<1u, 1u, 2u, 1u, 1u, 3u, 1u, 1u, 4u, 1u, 1u> dimensions;
        test_squeeze<dimensions>();
      }
    }
  }
}
