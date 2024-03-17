#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>

namespace nt = ntensor;

TEST_CASE("DenseBuffer class tests") {
  SECTION("aliases") {
    {
      nt::DenseBuffer<int> buffer;
      using buffer_type = std::decay_t<decltype(buffer)>;
      static_assert(std::is_same_v<buffer_type::size_type, std::size_t>);
      static_assert(std::is_same_v<buffer_type::difference_type, std::ptrdiff_t>);

      static_assert(std::is_same_v<buffer_type::pointer, int*>);
      static_assert(std::is_same_v<buffer_type::const_pointer, const int*>);
      static_assert(std::is_same_v<buffer_type::reference, int&>);
      static_assert(std::is_same_v<buffer_type::const_reference, const int&>);
      static_assert(std::is_same_v<buffer_type::value_type, int>);
    }

    {
      nt::DenseBuffer<float> buffer;
      using buffer_type = std::decay_t<decltype(buffer)>;
      static_assert(std::is_same_v<buffer_type::size_type, std::size_t>);
      static_assert(std::is_same_v<buffer_type::difference_type, std::ptrdiff_t>);

      static_assert(std::is_same_v<buffer_type::pointer, float*>);
      static_assert(std::is_same_v<buffer_type::const_pointer, const float*>);
      static_assert(std::is_same_v<buffer_type::reference, float&>);
      static_assert(std::is_same_v<buffer_type::const_reference, const float&>);
      static_assert(std::is_same_v<buffer_type::value_type, float>);
    }
  }

  SECTION("copy semantics") {
    {
      nt::DenseBuffer<int> fst{10};
      nt::DenseBuffer<int> snd{fst};

      CHECK(fst.size() == 10u);
      CHECK(fst.data() != nullptr);

      CHECK(snd.size() == 10u);
      CHECK(snd.data() == fst.data());

      CHECK(fst == snd);

      for (int i = 0; i < 10; ++i) {
        fst[i] = i;
        CHECK(fst[i] == i);
        CHECK(snd[i] == i);
        CHECK(fst.at(i) == i);
        CHECK(snd.at(i) == i);
      }
    }

    {
      nt::DenseBuffer<int> fst;
      nt::DenseBuffer<int> snd{10};

      CHECK(fst.size() == 0u);
      CHECK(fst.data() == nullptr);

      CHECK(snd.size() == 10u);
      CHECK(snd.data() != nullptr);

      CHECK(fst != snd);

      fst = snd;

      CHECK(fst.size() == 10u);
      CHECK(fst.data() == snd.data());

      CHECK(snd.size() == 10u);
      CHECK(snd.data() != nullptr);

      CHECK(fst == snd);

      for (int i = 0; i < 10; ++i) {
        snd[i] = i;
        CHECK(fst[i] == i);
        CHECK(snd[i] == i);
        CHECK(fst.at(i) == i);
        CHECK(snd.at(i) == i);
      }
    }
  }

  SECTION("move semantics") {
    {
      nt::DenseBuffer<int> buffer{nt::DenseBuffer<int>{10}};

      CHECK(buffer.size() == 10u);
      CHECK(buffer.data() != nullptr);
    }

    {
      nt::DenseBuffer<int> fst;
      nt::DenseBuffer<int> snd{10};

      CHECK(fst.size() == 0u);
      CHECK(fst.data() == nullptr);

      CHECK(snd.size() == 10u);
      CHECK(snd.data() != nullptr);

      for (int i = 0; i < 10; ++i) {
        snd[i] = i;
        CHECK(snd[i] == i);
        CHECK(snd.at(i) == i);
      }

      fst = std::move(snd);

      CHECK(fst.size() == 10u);
      CHECK(fst.data() != nullptr);

      for (int i = 0; i < 10; ++i) {
        CHECK(fst[i] == i);
        CHECK(fst.at(i) == i);
      }
    }
  }
}
