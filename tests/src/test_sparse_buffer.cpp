#include <catch2/catch_test_macros.hpp>
#include <sparse_buffer.hpp>

namespace nt = ntensor;

TEST_CASE("SparseBuffer class tests") {
  SECTION("aliases") {
    {
      nt::SparseBuffer<int> buffer;
      using buffer_type = std::decay_t<decltype(buffer)>;
      static_assert(std::is_same_v<buffer_type::size_type, std::size_t>);
      static_assert(std::is_same_v<buffer_type::difference_type, std::ptrdiff_t>);

      static_assert(std::is_same_v<buffer_type::pointer, nt::SparseBuffer<int>::SparseValue<int*>>);
      static_assert(std::is_same_v<buffer_type::const_pointer, nt::SparseBuffer<int>::SparseValue<const int*>>);
      static_assert(std::is_same_v<buffer_type::reference, nt::SparseBuffer<int>::SparseValue<int>>);
      static_assert(std::is_same_v<buffer_type::const_reference, nt::SparseBuffer<int>::SparseValue<const int>>);
      static_assert(std::is_same_v<buffer_type::value_type, nt::SparseBuffer<int>::SparseValue<int>>);
    }

    {
      nt::SparseBuffer<float> buffer;
      using buffer_type = std::decay_t<decltype(buffer)>;
      static_assert(std::is_same_v<buffer_type::size_type, std::size_t>);
      static_assert(std::is_same_v<buffer_type::difference_type, std::ptrdiff_t>);

      static_assert(std::is_same_v<buffer_type::pointer, nt::SparseBuffer<float>::SparseValue<float*>>);
      static_assert(std::is_same_v<buffer_type::const_pointer, nt::SparseBuffer<float>::SparseValue<const float*>>);
      static_assert(std::is_same_v<buffer_type::reference, nt::SparseBuffer<float>::SparseValue<float>>);
      static_assert(std::is_same_v<buffer_type::const_reference, nt::SparseBuffer<float>::SparseValue<const float>>);
      static_assert(std::is_same_v<buffer_type::value_type, nt::SparseBuffer<float>::SparseValue<float>>);
    }
  }

  SECTION("copy semantics") {
    {
      nt::SparseBuffer<int> fst{10};
      nt::SparseBuffer<int> snd{fst};

      CHECK(fst.size() == 10u);
      CHECK(snd.size() == 10u);

      CHECK(fst == snd);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(fst[i] == 0);
        CHECK(snd[i] == 0);

        fst[i] = value;

        CHECK(fst[i] == value);
        CHECK(fst.at(i) == value);
        CHECK(snd[i] == value);
        CHECK(snd.at(i) == value);
      }
    }

    {
      nt::SparseBuffer<int> fst;
      nt::SparseBuffer<int> snd{10};

      CHECK(fst.size() == 0u);
      CHECK(snd.size() == 10u);

      CHECK(fst != snd);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(snd[i] == 0);
        CHECK(snd.at(i) == 0);

        snd[i] = value;

        CHECK(snd[i] == value);
        CHECK(snd.at(i) == value);
      }

      fst = snd;

      CHECK(fst.size() == 10u);
      CHECK(snd.size() == 10u);

      CHECK(fst == snd);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(fst[i] == value);
        CHECK(snd[i] == value);

        CHECK(fst.at(i) == value);
        CHECK(snd.at(i) == value);
      }
    }
  }

  SECTION("move semantics") {
    {
      nt::SparseBuffer<int> buffer{nt::SparseBuffer<int>{10}};

      CHECK(buffer.size() == 10u);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(buffer[i] == 0);
        CHECK(buffer.at(i) == 0);

        buffer[i] = value;

        CHECK(buffer[i] == value);
        CHECK(buffer.at(i) == value);
      }
    }

    {
      nt::SparseBuffer<int> fst;
      nt::SparseBuffer<int> snd{10};

      CHECK(fst.size() == 0u);
      CHECK(snd.size() == 10u);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(snd[i] == 0);
        CHECK(snd.at(i) == 0);

        snd[i] = value;

        CHECK(snd[i] == value);
        CHECK(snd.at(i) == value);
      }

      fst = std::move(snd);

      CHECK(fst.size() == 10u);

      for (int i = 0; i < 10; ++i) {
        const int value = i + 1;

        CHECK(fst[i] == value);
        CHECK(fst.at(i) == value);
      }
    }
  }
}

TEST_CASE("SparseValue class tests") {
  SECTION("operators") {
    {
      nt::SparseBuffer<int>::SparseValue<int> sparse_value;
      int value = sparse_value;

      CHECK(value == 0);
      CHECK(sparse_value == 0);
      CHECK(!(sparse_value > 0));
      CHECK(!(sparse_value < 0));
      CHECK(sparse_value + 5 == 5);
      CHECK(sparse_value - 5 == -5);
      CHECK(sparse_value * 5 == 0);
      CHECK(sparse_value / 5 == 0);
    }

    {
      int assigned_value = 5;
      nt::SparseBuffer<int>::SparseValue<int> sparse_value{assigned_value};
      int value = sparse_value;

      CHECK(value == assigned_value);
      CHECK(sparse_value == assigned_value);
      CHECK(!(sparse_value > assigned_value));
      CHECK(!(sparse_value < assigned_value));
      CHECK(+sparse_value == +assigned_value);
      CHECK(-sparse_value == -assigned_value);
      CHECK(sparse_value + 5 == assigned_value + 5);
      CHECK(sparse_value - 5 == assigned_value - 5);
      CHECK(sparse_value * 5 == assigned_value * 5);
      CHECK(sparse_value / 5 == assigned_value / 5);
    }

    {
      int assigned_value = 5;
      int value = 0;
      auto callback = [&value](int*& ptr, const int& v) {
        ptr = &value;
        value = v;
      };
      nt::SparseBuffer<int>::SparseValue<int> sparse_value{callback};
      sparse_value = assigned_value;

      CHECK(sparse_value == assigned_value);
      CHECK(!(sparse_value > assigned_value));
      CHECK(!(sparse_value < assigned_value));
      CHECK(+sparse_value == +assigned_value);
      CHECK(-sparse_value == -assigned_value);
      CHECK(sparse_value + 5 == assigned_value + 5);
      CHECK(sparse_value - 5 == assigned_value - 5);
      CHECK(sparse_value * 5 == assigned_value * 5);
      CHECK(sparse_value / 5 == assigned_value / 5);
      sparse_value += 5;
      CHECK(sparse_value == 10);
      sparse_value -= 6;
      CHECK(sparse_value == 4);
      sparse_value *= 5;
      CHECK(sparse_value == 20);
      sparse_value /= 2;
      CHECK(sparse_value == 10);
    }
  }
}
