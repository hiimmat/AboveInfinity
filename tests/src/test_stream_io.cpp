#include <catch2/catch_test_macros.hpp>
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <sparse_buffer.hpp>
#include <sstream>
#include <stream_io.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

TEST_CASE("write_to_sink and load_from_source") {
  int expected_value = 0;
  auto fill_fn = [&expected_value](int& v) { v = expected_value++; };

  SECTION("write to stream and read from it") {
    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<2u, 6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < dimensions.template at<1u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<0u>(); ++j) {
          CHECK(in_tensor.slicing_value(0u, j, i) == value++);
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            CHECK(in_tensor.slicing_value(0u, k, j, i) == value++);
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> dimensions;
      static constexpr std::size_t channels{3u};
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            for (std::size_t c = 0u; c < channels; ++c) {
              CHECK(in_tensor.slicing_value(c, k, j, i) == value++);
            }
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> dimensions;
      static constexpr std::size_t channels{3u};
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels, false>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions, channels>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < dimensions.template at<0u>(); ++k) {
            for (std::size_t c = 0u; c < channels; ++c) {
              CHECK(in_tensor.slicing_value(c, k, j, i) == value++);
            }
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> fst_dimensions;
      static constexpr nt::Dimensions<3u, 7u> snd_dimensions;
      auto out_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions, 1u, false>();
      auto out_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions, 1u, false>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_fst_plane, out_snd_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions>();
      auto in_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_fst_plane, in_snd_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < fst_dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < fst_dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < fst_dimensions.template at<0u>(); ++k) {
            CHECK(in_tensor.slicing_value(0u, k, j, i) == value++);
          }
        }
      }

      for (std::size_t i = 0u; i < snd_dimensions.template at<1u>(); ++i) {
        for (std::size_t j = 0u; j < snd_dimensions.template at<0u>(); ++j) {
          CHECK(in_tensor.template slicing_value<1u>(0u, j, i) == value++);
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> fst_dimensions;
      static constexpr nt::Dimensions<3u, 7u> snd_dimensions;
      static constexpr std::size_t channels{2u};
      auto out_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions>();
      auto out_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions, channels, false>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_fst_plane, out_snd_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions>();
      auto in_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions, channels>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_fst_plane, in_snd_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < fst_dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < fst_dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < fst_dimensions.template at<0u>(); ++k) {
            CHECK(in_tensor.slicing_value(0u, k, j, i) == value++);
          }
        }
      }

      for (std::size_t i = 0u; i < snd_dimensions.template at<1u>(); ++i) {
        for (std::size_t j = 0u; j < snd_dimensions.template at<0u>(); ++j) {
          for (std::size_t c = 0u; c < channels; ++c) {
            CHECK(in_tensor.template slicing_value<1u>(c, j, i) == value++);
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<5u, 2u, 4u> fst_dimensions;
      static constexpr nt::Dimensions<3u, 7u> snd_dimensions;
      static constexpr std::size_t channels{2u};
      auto out_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions>();
      auto out_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions, channels>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_fst_plane, out_snd_plane);
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss);

      auto in_fst_plane = nt::create_plane<nt::DenseBuffer<int>, fst_dimensions, 1u, false>();
      auto in_snd_plane = nt::create_plane<nt::DenseBuffer<int>, snd_dimensions, channels, false>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_fst_plane, in_snd_plane);
      nt::load_from_source(in_tensor, ss.view());

      int value = 0;

      for (std::size_t i = 0u; i < fst_dimensions.template at<2u>(); ++i) {
        for (std::size_t j = 0u; j < fst_dimensions.template at<1u>(); ++j) {
          for (std::size_t k = 0u; k < fst_dimensions.template at<0u>(); ++k) {
            CHECK(in_tensor.slicing_value(0u, k, j, i) == value++);
          }
        }
      }

      for (std::size_t i = 0u; i < snd_dimensions.template at<1u>(); ++i) {
        for (std::size_t j = 0u; j < snd_dimensions.template at<0u>(); ++j) {
          for (std::size_t c = 0u; c < channels; ++c) {
            CHECK(in_tensor.template slicing_value<1u>(c, j, i) == value++);
          }
        }
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string header = "This is a header";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{header});

      const std::string expected_stream{"#This is a header\n{\n[0,1,2,3,4,5]\n}\n\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string footer = "This is a footer";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{.footer = footer});

      const std::string expected_stream{"{\n[0,1,2,3,4,5]\n}\n\n#This is a footer\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string header = "This is a header";
      std::string footer = "\nThis is a footer";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{header, footer});

      const std::string expected_stream{"#This is a header\n{\n[0,1,2,3,4,5]\n}\n\n#\n#This is a footer\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string header = "\n\n\nThis\n is\n\n a\n \n\n\n\nheader\n\n";
      std::string footer = "\nThis\n\n is\n\n a\n\n footer\n\n\n\n";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{header, footer});

      const std::string expected_stream{
          "#\n#\n#\n#This\n# is\n#\n# a\n# \n#\n#\n#\n#header\n#\n#\n{\n[0,1,2,3,4,5]\n}\n\n#\n#This\n#\n# is\n#\n# "
          "a\n#\n# footer\n#\n#\n#\n#\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string header = "This is a header";
      std::string footer = "T\nhis\n is a fo\noter";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{header, footer});

      const std::string expected_stream{"#This is a header\n{\n[0,1,2,3,4,5]\n}\n\n#T\n#his\n# is a fo\n#oter\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }

    {
      static constexpr nt::Dimensions<6u> dimensions;
      auto out_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto out_tensor = nt::create_tensor<nt::ShapeTransmutation>(out_plane);
      std::string header = "T\nhis\n is a he\nader";
      std::string footer = "This is a footer";
      std::stringstream ss;

      nt::execute(fill_fn, out_tensor);
      expected_value = 0;
      nt::write_to_sink(out_tensor, ss, nt::additional_output_content{header, footer});

      const std::string expected_stream{"#T\n#his\n# is a he\n#ader\n{\n[0,1,2,3,4,5]\n}\n\n#This is a footer\n"};
      CHECK(ss.view() == expected_stream);

      auto in_plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
      auto in_tensor = nt::create_tensor<nt::ShapeTransmutation>(in_plane);
      nt::load_from_source(in_tensor, ss.view());

      for (std::size_t i = 0u; i < dimensions.template at<0u>(); ++i) {
        CHECK(in_tensor.slicing_value(0u, i) == static_cast<int>(i));
      }
    }
  }
}
