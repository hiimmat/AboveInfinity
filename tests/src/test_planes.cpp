#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <planes.hpp>

namespace nt = ntensor;

TEST_CASE("Planes class tests") {
  SECTION("push_front") {
    {
      auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
      auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
      auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

      auto planes_one = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes_one)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes_one), nt::Planes<typename decltype(plane_one)::type>>);

      auto planes_two = planes_one.push_front(plane_two);

      static_assert(std::decay_t<decltype(planes_two)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes_two),
                                   nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_one)::type>>);

      auto planes = planes_two.push_front(plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_three)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_one)::type>>);
    }

    {
      auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
      auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
      auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

      auto planes_one = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes_one)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes_one), nt::Planes<typename decltype(plane_one)::type>>);

      auto planes = planes_one.push_front(plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type,
                                              typename decltype(plane_one)::type>>);
    }
  }

  SECTION("push_back") {
    {
      auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
      auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
      auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

      auto planes_one = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes_one)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes_one), nt::Planes<typename decltype(plane_one)::type>>);

      auto planes_two = planes_one.push_back(plane_two);

      static_assert(std::decay_t<decltype(planes_two)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes_two),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);

      auto planes = planes_two.push_back(plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }

    {
      auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
      auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
      auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

      auto planes_one = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes_one)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes_one), nt::Planes<typename decltype(plane_one)::type>>);

      auto planes = planes_one.push_back(plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }
  }

  SECTION("insert") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    {
      auto planes = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type>>);

      auto updated_planes = planes.template insert<0u>(plane_two);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_one)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type>>);

      auto updated_planes = planes.template insert<1u>(plane_two);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);

      auto updated_planes = planes.template insert<0u>(plane_three);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_three)::type, typename decltype(plane_one)::type,
                                              typename decltype(plane_two)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);

      auto updated_planes = planes.template insert<1u>(plane_three);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_three)::type,
                                              typename decltype(plane_two)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);

      auto updated_planes = planes.template insert<2u>(plane_three);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }
  }

  SECTION("remove") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    auto planes = nt::create_planes(plane_one, plane_two, plane_three);

    static_assert(std::decay_t<decltype(planes)>::size() == 3u);
    static_assert(std::is_same_v<decltype(planes),
                                 nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                            typename decltype(plane_three)::type>>);

    {
      auto updated_planes = planes.template remove<0u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template remove<1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template remove<2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template remove<0u, 1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template remove<0u, 2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template remove<1u, 2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_one)::type>>);
    }

    {
      auto updated_planes = planes.template remove<2u, 1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_one)::type>>);
    }

    {
      auto updated_planes = planes.template remove<2u, 0u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template remove<1u, 0u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_three)::type>>);
    }
  }

  SECTION("replace") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    {
      auto planes = nt::create_planes(plane_one);

      static_assert(std::decay_t<decltype(planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type>>);

      auto updated_planes = planes.template replace<0u>(plane_two);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_two)::type>>);

      auto second_updated_planes = updated_planes.template replace<0u>(plane_three);

      static_assert(std::decay_t<decltype(second_updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(second_updated_planes), nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type, decltype(plane_two)::type>>);

      auto updated_planes = planes.template replace<0u>(plane_three);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_three)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type, decltype(plane_two)::type>>);

      auto updated_planes = planes.template replace<1u>(plane_three);

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_three)::type>>);
    }
  }

  SECTION("keep") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    auto planes = nt::create_planes(plane_one, plane_two, plane_three);

    static_assert(std::decay_t<decltype(planes)>::size() == 3u);
    static_assert(std::is_same_v<decltype(planes), nt::Planes<typename decltype(plane_one)::type,
                                                              decltype(plane_two)::type, decltype(plane_three)::type>>);

    {
      auto updated_planes = planes.template keep<0u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_one)::type>>);
    }

    {
      auto updated_planes = planes.template keep<1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template keep<2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 1u);
      static_assert(std::is_same_v<decltype(updated_planes), nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template keep<0u, 1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template keep<0u, 2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template keep<1u, 2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto updated_planes = planes.template keep<2u, 1u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(
          std::is_same_v<decltype(updated_planes),
                         nt::Planes<typename decltype(plane_three)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto updated_planes = planes.template keep<1u, 0u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_one)::type>>);
    }

    {
      auto updated_planes = planes.template keep<0u, 1u, 2u>();

      static_assert(std::decay_t<decltype(updated_planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(updated_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }
  }

  SECTION("merge") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    {
      auto lhs_planes = nt::create_planes(plane_one);
      auto rhs_planes = nt::create_planes(plane_two);

      static_assert(std::decay_t<decltype(lhs_planes)>::size() == 1u);
      static_assert(std::decay_t<decltype(rhs_planes)>::size() == 1u);

      static_assert(std::is_same_v<decltype(lhs_planes), nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(std::is_same_v<decltype(rhs_planes), nt::Planes<typename decltype(plane_two)::type>>);

      auto planes = lhs_planes.merge(rhs_planes);

      static_assert(std::decay_t<decltype(planes)>::size() == 2u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
    }

    {
      auto lhs_planes = nt::create_planes(plane_one, plane_two);
      auto rhs_planes = nt::create_planes(plane_three);

      static_assert(std::decay_t<decltype(lhs_planes)>::size() == 2u);
      static_assert(std::decay_t<decltype(rhs_planes)>::size() == 1u);

      static_assert(std::is_same_v<decltype(lhs_planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
      static_assert(std::is_same_v<decltype(rhs_planes), nt::Planes<typename decltype(plane_three)::type>>);

      auto planes = lhs_planes.merge(rhs_planes);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }

    {
      auto lhs_planes = nt::create_planes(plane_one);
      auto rhs_planes = nt::create_planes(plane_two, plane_three);

      static_assert(std::decay_t<decltype(lhs_planes)>::size() == 1u);
      static_assert(std::decay_t<decltype(rhs_planes)>::size() == 2u);

      static_assert(std::is_same_v<decltype(lhs_planes), nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(
          std::is_same_v<decltype(rhs_planes),
                         nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type>>);

      auto planes = lhs_planes.merge(rhs_planes);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }
  }

  SECTION("split") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<0u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 1u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<1u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 2u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(
          std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                         nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<2u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 2u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<0u, 1u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 2u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(
          std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                         nt::Planes<typename decltype(plane_two)::type, typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<0u, 2u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 2u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<1u, 2u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 3u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_two)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<2u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_three)::type>>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::decay_t<decltype(planes)>::size() == 3u);
      static_assert(std::is_same_v<decltype(planes),
                                   nt::Planes<typename decltype(plane_one)::type, typename decltype(plane_two)::type,
                                              typename decltype(plane_three)::type>>);

      auto split_planes = planes.template split<0u, 1u, 2u>();

      static_assert(std::tuple_size_v<decltype(split_planes)> == 3u);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<0u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_one)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<1u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_two)::type>>);
      static_assert(std::is_same_v<std::decay_t<decltype(std::get<2u>(split_planes))>,
                                   nt::Planes<typename decltype(plane_three)::type>>);
    }
  }

  SECTION("plane") {
    auto plane_one = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<2u>{}>();
    auto plane_two = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<4u>{}>();
    auto plane_three = nt::create_plane<nt::DenseBuffer<int>, nt::Dimensions<6u>{}>();

    {
      auto planes = nt::create_planes(plane_one);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_one)>);
    }

    {
      auto planes = nt::create_planes(plane_two);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_two)>);
    }

    {
      auto planes = nt::create_planes(plane_three);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_three)>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_one)>);
      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<1u>())>, decltype(plane_two)>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_three);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_one)>);
      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<1u>())>, decltype(plane_three)>);
    }

    {
      auto planes = nt::create_planes(plane_two, plane_three);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_two)>);
      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<1u>())>, decltype(plane_three)>);
    }

    {
      auto planes = nt::create_planes(plane_one, plane_two, plane_three);

      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<0u>())>, decltype(plane_one)>);
      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<1u>())>, decltype(plane_two)>);
      static_assert(std::is_same_v<std::decay_t<decltype(planes.template plane<2u>())>, decltype(plane_three)>);
    }
  }
}
