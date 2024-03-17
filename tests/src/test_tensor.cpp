#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

template <typename _Tensor>
struct FirstPolicy {
  decltype(auto) first() { return static_cast<_Tensor*>(this)->_planes; }
};

template <typename _Tensor>
struct SecondPolicy {
  decltype(auto) second() { return static_cast<_Tensor*>(this)->_planes; }
};

template <typename _Tensor>
struct ThirdPolicy {
  decltype(auto) third() { return static_cast<_Tensor*>(this)->_planes; }
};

template <typename _Tensor>
struct FourthPolicy {
  decltype(auto) fourth() { return static_cast<_Tensor*>(this)->_planes; }
};

template <typename _Tensor>
struct FifthPolicy {
  decltype(auto) fifth() { return static_cast<_Tensor*>(this)->_planes; }
};

TEST_CASE("Tensor class with policies tests") {
  SECTION("Tensor policy tests") {
    static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    using planes_type = nt::Planes<std::decay_t<decltype(plane)>>;

    {
      auto tensor = nt::create_tensor<FirstPolicy, SecondPolicy, ThirdPolicy>(plane);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.third())>, planes_type>);
    }

    {
      auto tensor = nt::create_tensor<FirstPolicy, ThirdPolicy, SecondPolicy>(plane);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.third())>, planes_type>);
    }

    {
      auto tensor = nt::create_tensor<SecondPolicy, FirstPolicy, ThirdPolicy>(plane);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.third())>, planes_type>);
    }

    {
      auto tensor = nt::create_tensor<FirstPolicy, SecondPolicy, ThirdPolicy>(plane);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.third())>, planes_type>);

      auto updated_tensor = tensor.like<FourthPolicy, FifthPolicy>();
      static_assert(std::is_same_v<std::decay_t<decltype(updated_tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(updated_tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(updated_tensor.third())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(updated_tensor.fourth())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(updated_tensor.fifth())>, planes_type>);
    }

    {
      auto tensor = nt::create_tensor<FirstPolicy, SecondPolicy, ThirdPolicy, FourthPolicy, FifthPolicy>(plane);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.first())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.second())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.third())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.fourth())>, planes_type>);
      static_assert(std::is_same_v<std::decay_t<decltype(tensor.fifth())>, planes_type>);
    }
  }

  SECTION("copy semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor<nt::ShapeTransmutation>(plane);
      auto snd(fst);

      CHECK(fst == snd);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor<nt::ShapeTransmutation>(plane);
      auto snd = fst;

      CHECK(fst == snd);
    }
  }

  SECTION("move semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto tensor(nt::create_tensor<nt::ShapeTransmutation>(plane));

      CHECK(std::decay_t<decltype(tensor.planes())>::size() == 1u);
      CHECK(tensor.planes().template plane<0u>() == plane);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor<nt::ShapeTransmutation>(plane);
      auto snd = std::move(fst);

      CHECK(std::decay_t<decltype(snd.planes())>::size() == 1u);
      CHECK(snd.planes().template plane<0u>() == plane);
    }
  }

  SECTION("like method") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

      static constexpr nt::Dimensions<7u, 6u, 4u> new_dimensions;
      auto new_plane = nt::create_plane<nt::DenseBuffer<int>, new_dimensions>();
      auto new_planes = nt::create_planes(new_plane);
      auto new_tensor = tensor.like(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> snd_plane_dimensions;
      auto fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_plane_dimensions>();
      auto snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_plane_dimensions>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(fst_plane, snd_plane);

      static constexpr nt::Dimensions<7u, 6u, 4u> new_dimensions;
      auto new_plane = nt::create_plane<nt::DenseBuffer<int>, new_dimensions>();
      auto new_planes = nt::create_planes(new_plane);
      auto new_tensor = tensor.like(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<7u, 6u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

      static constexpr nt::Dimensions<2u, 4u, 6u> new_fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> new_snd_plane_dimensions;
      auto new_fst_plane = nt::create_plane<nt::DenseBuffer<int>, new_fst_plane_dimensions>();
      auto new_snd_plane = nt::create_plane<nt::DenseBuffer<int>, new_snd_plane_dimensions>();
      auto new_planes = nt::create_planes(new_fst_plane, new_snd_plane);
      auto new_tensor = tensor.like(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<7u, 6u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

      static constexpr nt::Dimensions<2u, 4u, 6u> new_fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> new_snd_plane_dimensions;
      auto new_fst_plane = nt::create_plane<nt::DenseBuffer<int>, new_fst_plane_dimensions>();
      auto new_snd_plane = nt::create_plane<nt::DenseBuffer<int>, new_snd_plane_dimensions>();
      auto new_planes = nt::create_planes(new_fst_plane, new_snd_plane);
      auto new_tensor = tensor.like<FirstPolicy>(new_planes);
      CHECK(new_tensor.planes() == new_planes);
      CHECK(new_tensor.first() == new_planes);
    }
  }
}

TEST_CASE("Tensor class without policies tests") {
  SECTION("copy semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor(plane);
      auto snd(fst);

      CHECK(fst == snd);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor(plane);
      auto snd = fst;

      CHECK(fst == snd);
    }
  }

  SECTION("move semantics") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto tensor(nt::create_tensor(plane));
      auto policy_tensor = tensor.like<nt::ShapeTransmutation>();

      CHECK(std::decay_t<decltype(policy_tensor.planes())>::size() == 1u);
      CHECK(policy_tensor.planes().template plane<0u>() == plane);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();

      auto fst = nt::create_tensor(plane);
      auto snd = std::move(fst);

      auto policy_tensor = snd.like<nt::ShapeTransmutation>();

      CHECK(std::decay_t<decltype(policy_tensor.planes())>::size() == 1u);
      CHECK(policy_tensor.planes().template plane<0u>() == plane);
    }
  }

  SECTION("like method") {
    {
      static constexpr nt::Dimensions<2u, 4u, 6u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor(plane);

      static constexpr nt::Dimensions<7u, 6u, 4u> new_dimensions;
      auto new_plane = nt::create_plane<nt::DenseBuffer<int>, new_dimensions>();
      auto new_planes = nt::create_planes(new_plane);
      auto new_tensor = tensor.like<nt::ShapeTransmutation>(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<2u, 4u, 6u> fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> snd_plane_dimensions;
      auto fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_plane_dimensions>();
      auto snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_plane_dimensions>();
      auto tensor = nt::create_tensor(fst_plane, snd_plane);

      static constexpr nt::Dimensions<7u, 6u, 4u> new_dimensions;
      auto new_plane = nt::create_plane<nt::DenseBuffer<int>, new_dimensions>();
      auto new_planes = nt::create_planes(new_plane);
      auto new_tensor = tensor.like<nt::ShapeTransmutation>(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<7u, 6u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor(plane);

      static constexpr nt::Dimensions<2u, 4u, 6u> new_fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> new_snd_plane_dimensions;
      auto new_fst_plane = nt::create_plane<nt::DenseBuffer<int>, new_fst_plane_dimensions>();
      auto new_snd_plane = nt::create_plane<nt::DenseBuffer<int>, new_snd_plane_dimensions>();
      auto new_planes = nt::create_planes(new_fst_plane, new_snd_plane);
      auto new_tensor = tensor.like<nt::ShapeTransmutation>(new_planes);
      CHECK(new_tensor.planes() == new_planes);
    }

    {
      static constexpr nt::Dimensions<7u, 6u, 4u> dimensions;
      auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto tensor = nt::create_tensor(plane);

      static constexpr nt::Dimensions<2u, 4u, 6u> new_fst_plane_dimensions;
      static constexpr nt::Dimensions<3u, 4u, 7u> new_snd_plane_dimensions;
      auto new_fst_plane = nt::create_plane<nt::DenseBuffer<int>, new_fst_plane_dimensions>();
      auto new_snd_plane = nt::create_plane<nt::DenseBuffer<int>, new_snd_plane_dimensions>();
      auto new_planes = nt::create_planes(new_fst_plane, new_snd_plane);
      auto new_tensor = tensor.like(new_planes);
    }
  }
}
