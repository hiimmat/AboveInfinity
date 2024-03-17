#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <execute.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

TEST_CASE("execute method tests") {
  SECTION("single tensor, one plane") {
    static constexpr nt::Dimensions<6, 2, 4> dimensions;
    auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(tensor.slicing_value(0u, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("single tensor, two planes") {
    static constexpr nt::Dimensions<6, 2, 4> first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> second_plane_dimensions;
    auto first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(first_plane, second_plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(tensor.slicing_value(0u, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 5u; ++k) {
      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 7u; ++i) {
          CHECK(tensor.template slicing_value<1u>(0u, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("single tensor, two planes, one plane with three channels") {
    static constexpr nt::Dimensions<6, 2, 4> first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> second_plane_dimensions;
    auto first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions, 3u>();
    auto second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions>();
    auto tensor = nt::create_tensor<nt::ShapeTransmutation>(first_plane, second_plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          for (std::size_t c = 0u; c < 3u; ++c) {
            CHECK(tensor.slicing_value(c, i, j, k) == value);
            ++value;
          }
        }
      }
    }

    for (std::size_t k = 0u; k < 5u; ++k) {
      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 7u; ++i) {
          CHECK(tensor.template slicing_value<1u>(0u, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with one plane, recursive executed called") {
    static constexpr nt::Dimensions<6, 2, 4> dimensions;
    auto first_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto second_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_plane);
    auto second_tensor = nt::create_tensor<nt::ShapeTransmutation>(second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(first_tensor.slicing_value(0, i, j, k) == value);
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, recursive executed called") {
    static constexpr nt::Dimensions<6, 2, 4> first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto second_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(first_tensor.slicing_value(0, i, j, k) == value);
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 5u; ++k) {
      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 7u; ++i) {
          CHECK(first_tensor.template slicing_value<1u>(0, i, j, k) == value);
          CHECK(second_tensor.template slicing_value<1u>(0, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, one plane with three channels, recursive executed called") {
    static constexpr nt::Dimensions<6, 2, 4> first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions, 3u>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto second_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions, 3u>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(first_tensor.slicing_value(0, i, j, k) == value);
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 5u; ++k) {
      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 7u; ++i) {
          for (std::size_t c = 0u; c < 3u; ++c) {
            CHECK(first_tensor.template slicing_value<1u>(c, i, j, k) == value);
            CHECK(second_tensor.template slicing_value<1u>(c, i, j, k) == value);
            ++value;
          }
        }
      }
    }
  }

  SECTION("two tensors, each with one plane, iterative execute called") {
    static constexpr nt::Dimensions<6, 2, 4> first_plane_dimensions;
    static constexpr nt::Dimensions<12, 4> second_plane_dimensions;
    auto first_plane = nt::create_plane<nt::DenseBuffer<int>, first_plane_dimensions>();
    auto second_plane = nt::create_plane<nt::DenseBuffer<int>, second_plane_dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_plane);
    auto second_tensor = nt::create_tensor<nt::ShapeTransmutation>(second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    {
      int value = 0;

      for (std::size_t k = 0u; k < 4u; ++k) {
        for (std::size_t j = 0u; j < 2u; ++j) {
          for (std::size_t i = 0u; i < 6u; ++i) {
            CHECK(first_tensor.slicing_value(0, i, j, k) == value);
            ++value;
          }
        }
      }
    }

    {
      int value = 0;

      for (std::size_t j = 0u; j < 4u; ++j) {
        for (std::size_t i = 0u; i < 12u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, iterative execute called") {
    static constexpr nt::Dimensions<6, 2, 4> first_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<12, 4> second_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> first_tensor_second_plane_dimensions;
    static constexpr nt::Dimensions<35, 3> second_tensor_second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_first_plane_dimensions>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_second_plane_dimensions>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_first_plane_dimensions>();
    auto second_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_second_plane_dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    {
      int value = 0;

      for (std::size_t k = 0u; k < 4u; ++k) {
        for (std::size_t j = 0u; j < 2u; ++j) {
          for (std::size_t i = 0u; i < 6u; ++i) {
            CHECK(first_tensor.slicing_value(0, i, j, k) == value);
            ++value;
          }
        }
      }

      for (std::size_t k = 0u; k < 5u; ++k) {
        for (std::size_t j = 0u; j < 3u; ++j) {
          for (std::size_t i = 0u; i < 7u; ++i) {
            CHECK(first_tensor.template slicing_value<1u>(0, i, j, k) == value);
            ++value;
          }
        }
      }
    }

    {
      int value = 0;

      for (std::size_t j = 0u; j < 4u; ++j) {
        for (std::size_t i = 0u; i < 12u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j) == value);
          ++value;
        }
      }

      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 35u; ++i) {
          CHECK(second_tensor.template slicing_value<1u>(0, i, j) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, one plane with three channels, iterative executed called") {
    static constexpr nt::Dimensions<6, 2, 4> first_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<12, 4> second_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<7, 3, 5> first_tensor_second_plane_dimensions;
    static constexpr nt::Dimensions<35, 3> second_tensor_second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_first_plane_dimensions>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_second_plane_dimensions, 2u>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_first_plane_dimensions>();
    auto second_tensor_second_plane =
        nt::create_plane<nt::DenseBuffer<int>, second_tensor_second_plane_dimensions, 2u>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& fst, int& snd) {
      static int i = 0;
      fst = i;
      snd = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    {
      int value = 0;

      for (std::size_t k = 0u; k < 4u; ++k) {
        for (std::size_t j = 0u; j < 2u; ++j) {
          for (std::size_t i = 0u; i < 6u; ++i) {
            CHECK(first_tensor.slicing_value(0, i, j, k) == value);
            ++value;
          }
        }
      }

      for (std::size_t k = 0u; k < 5u; ++k) {
        for (std::size_t j = 0u; j < 3u; ++j) {
          for (std::size_t i = 0u; i < 7u; ++i) {
            for (std::size_t c = 0u; c < 2u; ++c) {
              CHECK(first_tensor.template slicing_value<1u>(c, i, j, k) == value);
              ++value;
            }
          }
        }
      }
    }

    {
      int value = 0;

      for (std::size_t j = 0u; j < 4u; ++j) {
        for (std::size_t i = 0u; i < 12u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j) == value);
          ++value;
        }
      }

      for (std::size_t j = 0u; j < 3u; ++j) {
        for (std::size_t i = 0u; i < 35u; ++i) {
          for (std::size_t c = 0u; c < 2u; ++c) {
            CHECK(second_tensor.template slicing_value<1u>(c, i, j) == value);
            ++value;
          }
        }
      }
    }
  }

  SECTION("two tensors, each with one plane, executed separately") {
    static constexpr nt::Dimensions<6, 2, 4> dimensions;
    auto first_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto second_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_plane);
    auto second_tensor = nt::create_tensor<nt::ShapeTransmutation>(second_plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(first_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, executed separately") {
    static constexpr nt::Dimensions<6, 2, 4> first_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<3, 7, 2> first_tensor_second_plane_dimensions;
    static constexpr nt::Dimensions<5, 4, 8> second_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<6, 8, 3> second_tensor_second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_first_plane_dimensions>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_second_plane_dimensions>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_first_plane_dimensions>();
    auto second_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_second_plane_dimensions>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(first_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 2u; ++k) {
      for (std::size_t j = 0u; j < 7u; ++j) {
        for (std::size_t i = 0u; i < 3u; ++i) {
          CHECK(first_tensor.template slicing_value<1u>(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 8u; ++k) {
      for (std::size_t j = 0u; j < 4u; ++j) {
        for (std::size_t i = 0u; i < 5u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 3u; ++k) {
      for (std::size_t j = 0u; j < 8u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          CHECK(second_tensor.template slicing_value<1u>(0, i, j, k) == value);
          ++value;
        }
      }
    }
  }

  SECTION("two tensors, each with two planes, one plane with three channels, executed separately") {
    static constexpr nt::Dimensions<6, 2, 4> first_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<3, 7, 2> first_tensor_second_plane_dimensions;
    static constexpr nt::Dimensions<5, 4, 8> second_tensor_first_plane_dimensions;
    static constexpr nt::Dimensions<6, 8, 3> second_tensor_second_plane_dimensions;
    auto first_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_first_plane_dimensions, 2u>();
    auto first_tensor_second_plane = nt::create_plane<nt::DenseBuffer<int>, first_tensor_second_plane_dimensions, 3u>();
    auto second_tensor_first_plane = nt::create_plane<nt::DenseBuffer<int>, second_tensor_first_plane_dimensions>();
    auto second_tensor_second_plane =
        nt::create_plane<nt::DenseBuffer<int>, second_tensor_second_plane_dimensions, 4u>();
    auto first_tensor = nt::create_tensor<nt::ShapeTransmutation>(first_tensor_first_plane, first_tensor_second_plane);
    auto second_tensor =
        nt::create_tensor<nt::ShapeTransmutation>(second_tensor_first_plane, second_tensor_second_plane);
    auto fn = [](int& v) {
      static int i = 0;
      v = i++;
    };
    nt::execute(fn, first_tensor, second_tensor);

    int value = 0;

    for (std::size_t k = 0u; k < 4u; ++k) {
      for (std::size_t j = 0u; j < 2u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          for (std::size_t c = 0u; c < 2u; ++c) {
            CHECK(first_tensor.slicing_value(c, i, j, k) == value);
            ++value;
          }
        }
      }
    }

    for (std::size_t k = 0u; k < 2u; ++k) {
      for (std::size_t j = 0u; j < 7u; ++j) {
        for (std::size_t i = 0u; i < 3u; ++i) {
          for (std::size_t c = 0u; c < 3u; ++c) {
            CHECK(first_tensor.template slicing_value<1u>(c, i, j, k) == value);
            ++value;
          }
        }
      }
    }

    for (std::size_t k = 0u; k < 8u; ++k) {
      for (std::size_t j = 0u; j < 4u; ++j) {
        for (std::size_t i = 0u; i < 5u; ++i) {
          CHECK(second_tensor.slicing_value(0, i, j, k) == value);
          ++value;
        }
      }
    }

    for (std::size_t k = 0u; k < 3u; ++k) {
      for (std::size_t j = 0u; j < 8u; ++j) {
        for (std::size_t i = 0u; i < 6u; ++i) {
          for (std::size_t c = 0u; c < 4u; ++c) {
            CHECK(second_tensor.template slicing_value<1u>(c, i, j, k) == value);
            ++value;
          }
        }
      }
    }
  }
}
