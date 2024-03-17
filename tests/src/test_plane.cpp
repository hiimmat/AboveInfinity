#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <execute.hpp>
#include <plane.hpp>
#include <sparse_buffer.hpp>

namespace nt = ntensor;

TEST_CASE("Plane class tests") {
  SECTION("create_plane method") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_aligned_strides<int, 2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 768u);
      CHECK(plane.real_size() == 768u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, true>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_aligned_strides<int, 2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 768u);
      CHECK(plane.real_size() == 768u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_aligned_strides<int, 3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 896u);
      CHECK(plane.real_size() == 896u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, true>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_aligned_strides<int, 3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 896u);
      CHECK(plane.real_size() == 896u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 48u);
      CHECK(plane.real_size() == 48u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 84u);
      CHECK(plane.real_size() == 84u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(std::is_same_v<
                    decltype(plane_type::strides()),
                    decltype(nt::compute_aligned_strides<typename nt::SparseBuffer<int>::value_type, 2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 96u);
      CHECK(plane.real_size() == 96u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 1, true>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(std::is_same_v<
                    decltype(plane_type::strides()),
                    decltype(nt::compute_aligned_strides<typename nt::SparseBuffer<int>::value_type, 2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 96u);
      CHECK(plane.real_size() == 96u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(std::is_same_v<
                    decltype(plane_type::strides()),
                    decltype(nt::compute_aligned_strides<typename nt::SparseBuffer<int>::value_type, 3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 140u);
      CHECK(plane.real_size() == 140u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 1u, false>();
      CHECK(plane.offset() == 0);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 1u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 84u);
      CHECK(plane.real_size() == 84u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 1u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 48u);
      CHECK(plane.real_size() == 48u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 2u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 2u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 96u);
      CHECK(plane.real_size() == 96u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 2u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 2u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 168u);
      CHECK(plane.real_size() == 168u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 3u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 3u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 144u);
      CHECK(plane.real_size() == 144u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 3u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 3u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 252u);
      CHECK(plane.real_size() == 252u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 2u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 2u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 96u);
      CHECK(plane.real_size() == 96u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 2u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 2u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 168u);
      CHECK(plane.real_size() == 168u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 3u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 3u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 144u);
      CHECK(plane.real_size() == 144u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::SparseBuffer<int>, dimensions, 3u, false>();
      CHECK(plane.offset() == 0);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::SparseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::SparseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::SparseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::SparseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::SparseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::SparseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::SparseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::SparseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 3u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 252u);
      CHECK(plane.real_size() == 252u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>(20);
      CHECK(plane.offset() == 20);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 20u);
      CHECK(plane.effective_size() == 28u);
      CHECK(plane.real_size() == 48u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 3u, false>(50);
      CHECK(plane.offset() == 50);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<2u, 4u, 6u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<2u, 4u, 6u>())>);
      static_assert(plane_type::channels() == 3u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 50u);
      CHECK(plane.effective_size() == 94u);
      CHECK(plane.real_size() == 144u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>(30);
      CHECK(plane.offset() == 30);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 1u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 30u);
      CHECK(plane.effective_size() == 54u);
      CHECK(plane.real_size() == 84u);
    }

    {
      static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 2u, false>(70);
      CHECK(plane.offset() == 70);
      using plane_type = std::decay_t<decltype(plane)>;

      static_assert(std::is_same_v<typename decltype(plane)::type, plane_type>);
      static_assert(std::is_same_v<plane_type::buffer_type, nt::DenseBuffer<int>>);
      static_assert(std::is_same_v<plane_type::size_type, nt::DenseBuffer<int>::size_type>);
      static_assert(std::is_same_v<plane_type::difference_type, nt::DenseBuffer<int>::difference_type>);
      static_assert(std::is_same_v<plane_type::pointer, nt::DenseBuffer<int>::pointer>);
      static_assert(std::is_same_v<plane_type::const_pointer, nt::DenseBuffer<int>::const_pointer>);
      static_assert(std::is_same_v<plane_type::reference, nt::DenseBuffer<int>::reference>);
      static_assert(std::is_same_v<plane_type::const_reference, nt::DenseBuffer<int>::const_reference>);
      static_assert(std::is_same_v<plane_type::value_type, nt::DenseBuffer<int>::value_type>);
      static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(
          std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(plane_type::channels() == 2u);
      static_assert(plane_type::rank() == 3u);

      CHECK(plane.offset() == 70u);
      CHECK(plane.effective_size() == 98u);
      CHECK(plane.real_size() == 168u);
    }
  }

  SECTION("copy semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto fst = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      static constexpr auto strides = std::decay_t<decltype(fst)>::strides();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            const int idx =
                i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
            fst[idx] = idx;
          }
        }
      }

      auto snd{fst};

      CHECK(fst == snd);

      {
        for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
          for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
            for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
              const int idx =
                  i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
              CHECK(fst[idx] == snd[idx]);
              CHECK(fst.at(idx) == snd.at(idx));
            }
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto fst = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      static constexpr auto strides = std::decay_t<decltype(fst)>::strides();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            const int idx =
                i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
            fst[idx] = idx;
          }
        }
      }

      nt::Plane<nt::DenseBuffer<int>, dimensions, nt::compute_unaligned_strides<2u, 4u, 6u>()> snd;

      snd = fst;

      CHECK(fst == snd);

      {
        for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
          for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
            for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
              const int idx =
                  i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
              CHECK(fst[idx] == snd[idx]);
              CHECK(fst.at(idx) == snd.at(idx));
            }
          }
        }
      }
    }
  }

  SECTION("move semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane{nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>()};
      CHECK(plane.offset() == 0);
      CHECK(plane.real_size() == 48u);
      CHECK(plane.effective_size() == 48u);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto fst = nt::create_plane<nt::DenseBuffer<int>, dimensions, 1u, false>();
      static constexpr auto strides = std::decay_t<decltype(fst)>::strides();

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            const int idx =
                i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
            fst[idx] = idx;
          }
        }
      }

      nt::Plane<nt::DenseBuffer<int>, dimensions, nt::compute_unaligned_strides<2u, 4u, 6u>()> snd;

      snd = std::move(fst);

      {
        for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
          for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
            for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
              const int idx =
                  i * strides.template at<2u>() + j * strides.template at<1u>() + k * strides.template at<0u>();
              CHECK(snd[idx] == idx);
              CHECK(snd.at(idx) == idx);
            }
          }
        }
      }
    }
  }

  SECTION("like method") {
    static constexpr nt::Dimensions<3u, 7u, 4u> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, 2u, false>();
    using plane_type = std::decay_t<decltype(plane)>;

    static_assert(std::is_same_v<decltype(plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
    static_assert(
        std::is_same_v<decltype(plane_type::strides()), decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
    static_assert(plane_type::channels() == 2u);
    static_assert(plane_type::rank() == 3u);

    CHECK(plane.offset() == 0u);
    CHECK(plane.effective_size() == 168u);
    CHECK(plane.real_size() == 168u);

    {
      auto first_copy = plane.like(10);
      using first_copy_plane_type = std::decay_t<decltype(first_copy)>;
      static_assert(std::is_same_v<decltype(first_copy_plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(std::is_same_v<decltype(first_copy_plane_type::strides()),
                                   decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(first_copy_plane_type::channels() == 2u);
      static_assert(first_copy_plane_type::rank() == 3u);

      CHECK(first_copy.offset() == 10u);
      CHECK(first_copy.effective_size() == 158u);
      CHECK(first_copy.real_size() == 168u);

      auto second_copy = first_copy.like(50);
      using second_copy_plane_type = std::decay_t<decltype(second_copy)>;
      static_assert(std::is_same_v<decltype(second_copy_plane_type::dimensions()), nt::Dimensions<3u, 7u, 4u>>);
      static_assert(std::is_same_v<decltype(second_copy_plane_type::strides()),
                                   decltype(nt::compute_unaligned_strides<3u, 7u, 4u>())>);
      static_assert(second_copy_plane_type::channels() == 2u);
      static_assert(second_copy_plane_type::rank() == 3u);

      CHECK(second_copy.offset() == 60u);
      CHECK(second_copy.effective_size() == 108u);
      CHECK(second_copy.real_size() == 168u);
    }

    {
      auto first_copy = plane.template like<nt::Dimensions<21u, 4u>{}, nt::compute_unaligned_strides<21u, 4u>()>();
      using first_copy_plane_type = std::decay_t<decltype(first_copy)>;
      static_assert(std::is_same_v<decltype(first_copy_plane_type::dimensions()), nt::Dimensions<21u, 4u>>);
      static_assert(std::is_same_v<decltype(first_copy_plane_type::strides()),
                                   decltype(nt::compute_unaligned_strides<21u, 4u>())>);
      static_assert(first_copy_plane_type::channels() == 2u);
      static_assert(first_copy_plane_type::rank() == 2u);

      CHECK(plane.offset() == 0u);
      CHECK(plane.effective_size() == 168u);
      CHECK(plane.real_size() == 168u);

      auto second_copy =
          first_copy.template like<nt::Dimensions<7u, 3u, 4u>{}, nt::compute_unaligned_strides<7u, 3u, 4u>()>(30);
      using second_copy_plane_type = std::decay_t<decltype(second_copy)>;
      static_assert(std::is_same_v<decltype(second_copy_plane_type::dimensions()), nt::Dimensions<7u, 3u, 4u>>);
      static_assert(std::is_same_v<decltype(second_copy_plane_type::strides()),
                                   decltype(nt::compute_unaligned_strides<7u, 3u, 4u>())>);
      static_assert(second_copy_plane_type::channels() == 2u);
      static_assert(second_copy_plane_type::rank() == 3u);

      CHECK(second_copy.offset() == 30u);
      CHECK(second_copy.effective_size() == 138u);
      CHECK(second_copy.real_size() == 168u);
    }
  }
}
