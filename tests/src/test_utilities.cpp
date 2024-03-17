#include <catch2/catch_test_macros.hpp>
#include <strides.hpp>
#include <utilities.hpp>

namespace nt = ntensor;

template <auto... values>
struct ValueList {};

TEST_CASE("utility tests") {
  SECTION("nth type") {
    static_assert(std::is_same_v<nt::nth_type_t<0u, int, float, double, bool, char, double, double, int>, int>);
    static_assert(std::is_same_v<nt::nth_type_t<1u, int, float, double, bool, char, double, double, int>, float>);
    static_assert(std::is_same_v<nt::nth_type_t<2u, int, float, double, bool, char, double, double, int>, double>);
    static_assert(std::is_same_v<nt::nth_type_t<3u, int, float, double, bool, char, double, double, int>, bool>);
    static_assert(std::is_same_v<nt::nth_type_t<4u, int, float, double, bool, char, double, double, int>, char>);
    static_assert(std::is_same_v<nt::nth_type_t<5u, int, float, double, bool, char, double, double, int>, double>);
    static_assert(std::is_same_v<nt::nth_type_t<6u, int, float, double, bool, char, double, double, int>, double>);
    static_assert(std::is_same_v<nt::nth_type_t<7u, int, float, double, bool, char, double, double, float>, float>);
  }

  SECTION("first type in value sequence") {
    static_assert(std::is_same_v<nt::fis_t<4, true, 3.f, 4.0>, int>);
    static_assert(std::is_same_v<nt::fis_t<true, 4, 3.f, 4.0>, bool>);
    static_assert(std::is_same_v<nt::fis_t<3.f, true, 4, 4.0>, float>);
    static_assert(std::is_same_v<nt::fis_t<4.0, 3.f, true, 4>, double>);
    static_assert(std::is_same_v<nt::fis_t<'c', 4.0, 3.f, true, 4>, char>);
  }

  SECTION("first value in value sequence") {
    static_assert(nt::fis_v<4, true, 3.f, 4.0> == 4);
    static_assert(nt::fis_v<true, 4, 3.f, 4.0> == true);
    static_assert(nt::fis_v<3.f, true, 4, 4.0> == 3.f);
    static_assert(nt::fis_v<4.0, 3.f, true, 4> == 4.0);
    static_assert(nt::fis_v<'c', 4.0, 3.f, true, 4> == 'c');
  }

  SECTION("first type in type sequence") {
    static_assert(std::is_same_v<nt::fts_t<int, float, double, bool, char>, int>);
    static_assert(std::is_same_v<nt::fts_t<float, double, bool, char, int>, float>);
    static_assert(std::is_same_v<nt::fts_t<double, bool, char, int, float>, double>);
    static_assert(std::is_same_v<nt::fts_t<bool, char, int, float, double>, bool>);
    static_assert(std::is_same_v<nt::fts_t<char, int, float, double, bool>, char>);
  }

  SECTION("nth element") {
    static_assert(nt::nth_element<0u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 7);
    static_assert(nt::nth_element<1u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 1);
    static_assert(nt::nth_element<2u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 2);
    static_assert(nt::nth_element<3u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 4);
    static_assert(nt::nth_element<4u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 8);
    static_assert(nt::nth_element<5u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 4);
    static_assert(nt::nth_element<6u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 0);
    static_assert(nt::nth_element<7u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 2);
    static_assert(nt::nth_element<8u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 6);
    static_assert(nt::nth_element<9u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 5);
    static_assert(nt::nth_element<10u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 3);
    static_assert(nt::nth_element<11u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 12);
    static_assert(nt::nth_element<12u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 9);
    static_assert(nt::nth_element<13u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 34);
    static_assert(nt::nth_element<14u, 7, 1, 2, 4, 8, 4, 0, 2, 6, 5, 3, 12, 9, 34, 6>() == 6);
  }

  SECTION("find with predicate") {
    static_assert(nt::find_with_predicate<2, 4, 6, 1, 5, 7, 8, 3, 0>([](int e) { return e == 6; }) == 2);
    static_assert(nt::find_with_predicate<2, 4, 6, 1, 5, 7, 8, 3, 0>([](int e) { return e == 1; }) == 3);
    static_assert(nt::find_with_predicate<2, 4, 6, 1, 5, 7, 8, 3, 0>([](int e) { return e == 0; }) == 8);
    static_assert(nt::find_with_predicate<2, 4, 6, 1, 5, 7, 8, 6, 0>([](int e) { return e == 6; }) == 2);
    static_assert(nt::find_with_predicate<2, 2, 2, 2, 2, 2, 2, 2, 2>([](int e) { return e == 2; }) == 0);
    static_assert(nt::find_with_predicate<2, 4, 6, 1, 5, 7, 8, 3, 0>([](int e) { return e == 12; }) == 9);
  }

  SECTION("find all with predicate") {
    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<2, 4, 6, 1, 5, 7, 8, 3, 0>(),
                                                                      [](int e) { return e == 6; })),
                                 std::index_sequence<2>>);

    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<2, 4, 6, 1, 5, 7, 8, 3, 0>(),
                                                                      [](int e) { return e == 2; })),
                                 std::index_sequence<0>>);

    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<2, 4, 6, 1, 5, 7, 8, 3, 0>(),
                                                                      [](int e) { return e == 0; })),
                                 std::index_sequence<8>>);

    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<2, 4, 6, 1, 5, 7, 8, 3, 0>(),
                                                                      [](int e) { return e == 12; })),
                                 std::index_sequence<>>);

    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<2, 4, 6, 1, 5, 7, 8, 7, 0>(),
                                                                      [](int e) { return e == 7; })),
                                 std::index_sequence<5, 7>>);

    static_assert(std::is_same_v<decltype(nt::find_all_with_predicate(ValueList<7, 7, 7, 7, 7, 7, 7, 7, 7>(),
                                                                      [](int e) { return e == 7; })),
                                 std::index_sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>>);
  }

  SECTION("product") {
    static_assert(nt::product(nt::Dimensions<1>()) == 1);
    static_assert(nt::product(nt::Dimensions<1, 2>()) == 2);
    static_assert(nt::product(nt::Dimensions<1, 2, 3>()) == 6);
    static_assert(nt::product(nt::Dimensions<2, 4, 6, 12>()) == 576);
  }

  SECTION("max product") {
    static_assert(nt::max_product(nt::Dimensions<1u>(), nt::Strides<1>()) == 1);
    static_assert(nt::max_product(nt::Dimensions<1u>(), nt::Strides<1>()) == 1);
    static_assert(nt::max_product(nt::Dimensions<1u, 2u>(), nt::Strides<1, 2>()) == 4);
    static_assert(nt::max_product(nt::Dimensions<2u, 1u>(), nt::Strides<2, 1>()) == 4);
    static_assert(nt::max_product(nt::Dimensions<2u, 1u>(), nt::Strides<1, 2>()) == 2);
    static_assert(nt::max_product(nt::Dimensions<1u, 2u>(), nt::Strides<2, 1>()) == 2);
    static_assert(nt::max_product(nt::Dimensions<1u, 7u, 6u, 4u, 9u>(), nt::Strides<1, 1, 7, 42, 168>()) == 1512);
  }

  SECTION("min index") {
    static_assert(nt::min_index(nt::Dimensions<1u>(), nt::Strides<1>()) == 0);
    static_assert(nt::min_index(nt::Dimensions<1u>(), nt::Strides<2>()) == 0);
    static_assert(nt::min_index(nt::Dimensions<2u>(), nt::Strides<0>()) == 0);
    static_assert(nt::min_index(nt::Dimensions<1u>(), nt::Strides<-1>()) == 0);
    static_assert(nt::min_index(nt::Dimensions<2u>(), nt::Strides<-1>()) == -1);
    static_assert(nt::min_index(nt::Dimensions<10u>(), nt::Strides<-1>()) == -9);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, 7, 21>()) == 0);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, -7, 21>()) == -14);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, 7, 21>()) == -6);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, 7, -21>()) == -63);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, 7, -21>()) == -69);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, -7, -21>()) == -77);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, -7, 21>()) == -20);
    static_assert(nt::min_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, -7, -21>()) == -83);
  }

  SECTION("max index") {
    static_assert(nt::max_index(nt::Dimensions<1u>(), nt::Strides<1>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<1u>(), nt::Strides<2>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<2u>(), nt::Strides<0>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<1u>(), nt::Strides<-1>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<2u>(), nt::Strides<-1>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<10u>(), nt::Strides<-1>()) == 0);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, 7, 21>()) == 83);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, -7, 21>()) == 69);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, 7, 21>()) == 77);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, 7, -21>()) == 20);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, 7, -21>()) == 14);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<1, -7, -21>()) == 6);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, -7, 21>()) == 63);
    static_assert(nt::max_index(nt::Dimensions<7u, 3u, 4u>(), nt::Strides<-1, -7, -21>()) == 0);
  }

  SECTION("all unique") {
    static_assert(nt::all_unique<1, 2, 3, 4>());
    static_assert(!nt::all_unique<1, 1, 3, 4>());
    static_assert(!nt::all_unique<1, 2, 1, 4>());
    static_assert(!nt::all_unique<1, 2, 3, 1>());
    static_assert(!nt::all_unique<1, 1, 1, 4>());
    static_assert(!nt::all_unique<1, 1, 3, 1>());
    static_assert(!nt::all_unique<1, 1, 1, 1>());
  }

  SECTION("is sequence monotonically increasing") {
    static_assert(nt::is_sequence_monotonically_increasing<1>());
    static_assert(nt::is_sequence_monotonically_increasing<1, 2>());
    static_assert(nt::is_sequence_monotonically_increasing<1, 2, 3>());
    static_assert(nt::is_sequence_monotonically_increasing<1, 2, 3, 4>());
    static_assert(nt::is_sequence_monotonically_increasing<1, 2, 3, 4, 5>());
    static_assert(!nt::is_sequence_monotonically_increasing<2, 1, 3, 4, 5>());
    static_assert(!nt::is_sequence_monotonically_increasing<1, 3, 2, 4, 5>());
    static_assert(!nt::is_sequence_monotonically_increasing<1, 2, 4, 3, 5>());
    static_assert(!nt::is_sequence_monotonically_increasing<1, 2, 3, 5, 4>());
    static_assert(!nt::is_sequence_monotonically_increasing<5, 2, 4, 3, 1>());
    static_assert(!nt::is_sequence_monotonically_increasing<5, 4, 3, 2, 1>());
    static_assert(!nt::is_sequence_monotonically_increasing<1, 2, 4, 3>());
    static_assert(!nt::is_sequence_monotonically_increasing<1, 3, 2>());
    static_assert(!nt::is_sequence_monotonically_increasing<2, 1>());
  }

  SECTION("filter value sequence") {
    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 2, 3, 4, 5>(), [](int e) { return e == 1; })),
                       ValueList<2, 3, 4, 5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 1, 3, 4, 5>(), [](int e) { return e == 1; })),
                       ValueList<3, 4, 5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 1, 1, 4, 5>(), [](int e) { return e == 1; })),
                       ValueList<4, 5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 1, 1, 1, 5>(), [](int e) { return e == 1; })),
                       ValueList<5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 2, 1, 4, 5>(), [](int e) { return e == 1; })),
                       ValueList<2, 4, 5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 2, 3, 1, 5>(), [](int e) { return e == 1; })),
                       ValueList<2, 3, 5>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 2, 3, 4, 1>(), [](int e) { return e == 1; })),
                       ValueList<2, 3, 4>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<1, 2, 1, 4, 1>(), [](int e) { return e == 1; })),
                       ValueList<2, 4>>);

    static_assert(
        std::is_same_v<decltype(nt::filter_value_sequence(ValueList<2, 1, 3, 1, 5>(), [](int e) { return e == 1; })),
                       ValueList<2, 3, 5>>);
  }

  SECTION("filter index sequence") {
    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 2, 3, 4, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 3, 4, 5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 1, 3, 4, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<3, 4, 5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 1, 1, 4, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<4, 5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 1, 1, 1, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 2, 1, 4, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 4, 5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 2, 3, 1, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 3, 5>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 2, 3, 4, 1>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 3, 4>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<1, 2, 1, 4, 1>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 4>>);

    static_assert(std::is_same_v<decltype(nt::filter_index_sequence(std::index_sequence<2, 1, 3, 1, 5>(),
                                                                    [](int e) { return e == 1; })),
                                 std::index_sequence<2, 3, 5>>);
  }

  SECTION("set intersection") {
    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(), std::index_sequence<>())),
                       std::index_sequence<>>);

    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(), std::index_sequence<2u>())),
                       std::index_sequence<2u>>);

    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(), std::index_sequence<2u, 4u>())),
                       std::index_sequence<2u, 4u>>);

    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(), std::index_sequence<2u, 6u>())),
                       std::index_sequence<2u, 6u>>);

    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(), std::index_sequence<4u, 6u>())),
                       std::index_sequence<4u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 4u, 6u>(),
                                                               std::index_sequence<2u, 4u, 6u>())),
                                 std::index_sequence<2u, 4u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 2u, 4u, 6u>(),
                                                               std::index_sequence<2u, 4u, 6u>())),
                                 std::index_sequence<2u, 2u, 4u, 6u>>);

    static_assert(
        std::is_same_v<decltype(nt::set_intersection(std::index_sequence<2u, 2u, 2u, 5u, 4u, 4u, 9u, 9u, 6u, 1u, 6u, 0u,
                                                                         6u, 6u, 7u, 12u, 8u, 3u>(),
                                                     std::index_sequence<2u, 4u, 6u>())),
                       std::index_sequence<2u, 2u, 2u, 4u, 4u, 6u, 6u, 6u, 6u>>);
  }

  SECTION("remove nth element") {
    static_assert(std::is_same_v<decltype(nt::remove_nth_element<0u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::remove_nth_element<1u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<2u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::remove_nth_element<2u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<2u, 4u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::remove_nth_element<3u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<2u, 4u, 6u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::remove_nth_element<4u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<2u, 4u, 6u, 7u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>>);

    static_assert(std::is_same_v<decltype(nt::remove_nth_element<11u>(
                                     ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>())),
                                 ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u>>);

    {
      auto tmp = nt::remove_nth_element<5u>(ValueList<2u, 4u, 6u, 7u, 6u, 3u, 1u, 0u, 8u, 8u, 8u, 6u>());
      static_assert(
          std::is_same_v<decltype(nt::remove_nth_element<7u>(tmp)), ValueList<2u, 4u, 6u, 7u, 6u, 1u, 0u, 8u, 8u, 6u>>);
    }
  }

  SECTION("sort") {
    static_assert(
        std::is_same_v<decltype(nt::sort(ValueList<5>(), [](int lhs, int rhs) { return lhs < rhs; })), ValueList<5>>);

    static_assert(std::is_same_v<decltype(nt::sort(ValueList<9, 1, 7, 3, 2, 4, 0, 5, 5, 1, 4, 7, 8, 12, 6, 10>(),
                                                   [](int lhs, int rhs) { return lhs < rhs; })),
                                 ValueList<0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10, 12>>);

    static_assert(std::is_same_v<decltype(nt::sort(ValueList<9, 1, 7, 3, 2, 4, 0, 5, 5, 1, 4, 7, 8, 12, 6, 10>(),
                                                   [](int lhs, int rhs) { return lhs > rhs; })),
                                 ValueList<12, 10, 9, 8, 7, 7, 6, 5, 5, 4, 4, 3, 2, 1, 1, 0>>);

    static_assert(
        std::is_same_v<decltype(nt::sort(std::index_sequence<5u>(), [](int lhs, int rhs) { return lhs < rhs; })),
                       std::index_sequence<5u>>);

    static_assert(
        std::is_same_v<decltype(nt::sort(
                           std::index_sequence<9u, 1u, 7u, 3u, 2u, 4u, 0u, 5u, 5u, 1u, 4u, 7u, 8u, 12u, 6u, 10u>(),
                           [](int lhs, int rhs) { return lhs < rhs; })),
                       std::index_sequence<0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 12u>>);

    static_assert(
        std::is_same_v<decltype(nt::sort(
                           std::index_sequence<9u, 1u, 7u, 3u, 2u, 4u, 0u, 5u, 5u, 1u, 4u, 7u, 8u, 12u, 6u, 10u>(),
                           [](int lhs, int rhs) { return lhs > rhs; })),
                       std::index_sequence<12u, 10u, 9u, 8u, 7u, 7u, 6u, 5u, 5u, 4u, 4u, 3u, 2u, 1u, 1u, 0u>>);
  }

  SECTION("constexpr ternary") {
    static_assert(nt::constexpr_ternary<true>(4, 5) == 4);
    static_assert(nt::constexpr_ternary<false>(4, 5) == 5);
  }

  SECTION("create_index_sequence") {
    static_assert(std::is_same_v<decltype(nt::create_index_sequence<0u, 0u>()), std::index_sequence<0u>>);

    static_assert(
        std::is_same_v<decltype(nt::create_index_sequence<0u, 5u>()), std::index_sequence<0u, 1u, 2u, 3u, 4u>>);

    static_assert(std::is_same_v<decltype(nt::create_index_sequence<3u, 5u>()), std::index_sequence<3u, 4u>>);

    static_assert(std::is_same_v<decltype(nt::create_index_sequence<5u, 5u>()), std::index_sequence<5u>>);
  }

  SECTION("permute") {
    static_assert(std::is_same_v<decltype(nt::permute<0u, 1u, 2u>(ValueList<3, 4, 6>())), ValueList<3, 4, 6>>);
    static_assert(std::is_same_v<decltype(nt::permute<0u, 2u, 1u>(ValueList<3, 4, 6>())), ValueList<3, 6, 4>>);
    static_assert(std::is_same_v<decltype(nt::permute<2u, 1u, 0u>(ValueList<3, 4, 6>())), ValueList<6, 4, 3>>);
    static_assert(std::is_same_v<decltype(nt::permute<1u, 0u, 2u>(ValueList<3, 4, 6>())), ValueList<4, 3, 6>>);
    static_assert(std::is_same_v<decltype(nt::permute<1u, 2u, 0u>(ValueList<3, 4, 6>())), ValueList<4, 6, 3>>);
    static_assert(std::is_same_v<decltype(nt::permute<1u, 2u, 0u, 9u, 7u, 4u, 5u, 3u, 8u, 6u>(
                                     ValueList<3, 4, 6, 7, 6, 2, 4, 6, 4, 3>())),
                                 ValueList<4, 6, 3, 3, 6, 6, 2, 7, 4, 4>>);
  }
}
