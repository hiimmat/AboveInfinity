#include "TensorLayout.h"

#include <assert.h>

using namespace AboveInfinity;

void testLengths() {
    /* Test construction */
    Lengths<4, 1, 7, 6> lens;
    static_assert(std::is_same_v<std::decay_t<decltype(lens)>, Lengths<4, 1, 7, 6>>);

    /* Test permuting dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(lens.fastPermute<1U, 3U, 0U, 2U>())>, Lengths<1, 6, 4, 7>>);

    /* Test reducing dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(lens.dimensionReduction<2U>())>, Lengths<4, 1, 6>>);

    /* Test reducing a length */
    static_assert(std::is_same_v<std::decay_t<decltype(lens.lengthReduction<2U, 3>())>, Lengths<4, 1, 4, 6>>);

    /* Test conversion to 1D */
    static_assert(std::is_same_v<std::decay_t<decltype(lens.flatten())>, Lengths<168>>);

    /* Test manually modifying a length */
    static_assert(std::is_same_v<std::decay_t<decltype(lens.setLength<0U, 2>())>, Lengths<2, 1, 7, 6>>);

    /* Test getter */
    static_assert(lens.get<1U>() == 1);

    /* Test size */
    static_assert(lens.size() == 4U);

    /* Test tuple conversion */
    static_assert(lens.tuple() == std::tuple{4, 1, 7, 6});

    /* Test array conversion - the comparison doesn't seem to be constexpr */
    std::array<int, 4U> lensArr{4, 1, 7, 6};
    assert(lens.array() == lensArr);
}

void testStrides() {
    /* Test construction */
    Strides<int, 1, 16, 32> strides;

    /* Test permuting dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(strides.fastPermute<2U, 1U, 0U>())>, Strides<int, 32, 16, 1>>);

    /* Test reducing dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(strides.dimensionReduction<1U>())>, Strides<int, 1, 32>>);

    /* Test getter */
    static_assert(strides.get<2U>() == 32);

    /* Test size */
    static_assert(strides.size() == 3U);

    /* Test tuple conversion */
    static_assert(strides.tuple() == std::tuple{1, 16, 32});

    /* Test array conversion - the comparison doesn't seem to be constexpr */
    std::array<int, 3U> stridesArr{1, 16, 32};
    assert(strides.array() == stridesArr);
}

void testHelpers() {
    /* Test the allUnique function */
    internal::allUnique<1, 4, 3, 0, 2, 6, 8, 7>();

    /* Test function computeAlignedStrides and assert that the computed strides are correct */
    static_assert(std::is_same_v<std::decay_t<decltype(internal::computeAlignedStrides<float, Lengths<2, 4, 6>>())>,
                                 Strides<float, 1, 4, 16>>);

    /* Test function computeUnalignedStrides and assert that the computed strides are correct */
    static_assert(std::is_same_v<std::decay_t<decltype(internal::computeUnalignedStrides<float, Lengths<2, 4, 6>>())>,
                                 Strides<float, 1, 2, 8>>);

    /* test function partiallyComputeStrides using unaligned strides and assert that the computed strides are correct */
    static_assert(std::is_same_v<
                  std::decay_t<decltype(internal::partiallyComputeStrides<Lengths<2, 6, 8>, Strides<int, 1>, false>())>,
                  Strides<int, 1, 2, 12>>);

    /* test function partiallyComputeStrides using aligned strides and assert that the computed strides are correct */
    static_assert(
        std::is_same_v<std::decay_t<decltype(internal::partiallyComputeStrides<Lengths<2, 6, 8>, Strides<int, 1>>())>,
                       Strides<int, 1, 4, 24>>);

    /* Test function findMinIndex and assert that it returns the index of the smallest element */
    static_assert(internal::findMinIndex<8, 2, 6, 1, 4, 0, 3, 5, 7>(std::make_index_sequence<9U>()) == 5);

    /* Test function sortedIndexesAsc and assert that the computed index order is correct */
    static_assert(internal::sortedIndexesAsc<64, 1, 128, 4, 16, 32, 16>(std::make_index_sequence<7U>(),
                                                                        std::make_index_sequence<6U>()) ==
                  std::tuple(1, 3, 4, 6, 5, 0, 2));

    /* Test function count */
    static_assert(internal::count<2, 1, 0, 4, 6, 8, 0, 3, 0, 0, 7, 2, 0>([](int e) { return e == 0; }) == 5);

    /* Test function squeezeCount and assert that it counts the correct number of ones */
    static_assert(internal::squeezeCount<1, 6, 1, 3, 4, 2, 1, 1, 1, 0, 9, 3, 2, 1>() == 6);

    /* Test function packToTuple and expect that it returns an empty tuple */
    static_assert(internal::packToTuple<false, 1, 2, 6>() == std::tuple<>());

    /* Test function packToTuple and expect that it returns a tuple containing the passed elements */
    static_assert(internal::packToTuple<true, 1, 2, 6>() == std::tuple(1, 2, 6));

    /* Test function skipIdx by expecting it to keep the passed element */
    static_assert(internal::skipIdx<6, 1, 0, 4, 2, 3, 1>() == std::tuple(6));

    /* Test function skipIdx by expecting it to discard the passed element */
    static_assert(internal::skipIdx<3, 1, 0, 4, 2, 3, 6, 1>() == std::tuple<>());

    /* Test function maxStrideIndex by expecting it to find the index with the highest value */
    static_assert(internal::maxStrideIndex<3, 1, 0, 6, 4, 9, 2, 8, 5, 7>(std::make_index_sequence<10U>()) == 5);
}

void testShape() {
    /* Test construction */
    Shape<Lengths<2, 4, 6>, Strides<float, 1, 16, 96>> shape;

    /* Test permuting dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.fastPermute<2U, 1U, 0U>())>,
                                 Shape<Lengths<6, 4, 2>, Strides<float, 96, 16, 1>>>);

    /* Test undoing permutation */
    Shape<Lengths<2, 1, 7>, Strides<float, 1, 16, 112>> shapeUnpermuted;
    auto shapePermuted = shapeUnpermuted.fastPermute<1U, 2U, 0U>();
    static_assert(std::is_same_v<std::decay_t<decltype(shapePermuted.undoPermutation())>,
                                 Shape<Lengths<2, 1, 7>, Strides<float, 1, 16, 112>>>);

    /* Test reducing dimensions */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.dimensionReduction<0U>())>,
                                 Shape<Lengths<4, 6>, Strides<float, 16, 96>>>);

    /* Test reducing a length */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.lengthReduction<1U, 2>())>,
                                 Shape<Lengths<2, 2, 6>, Strides<float, 1, 16, 96>>>);

    /* Test manually modifying a length */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.setLength<2U, 3>())>,
                                 Shape<Lengths<2, 4, 3>, Strides<float, 1, 16, 96>>>);

    /* Test subspace */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.subspace<pair<0, 2>, pair<0, 0>, pair<5, 5>>())>,
                                 Shape<Lengths<2>, Strides<float, 1>>>);

    /* Test subspace */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.subspace<pair<1, 2>, pair<0, 4>>())>,
                                 Shape<Lengths<1, 4>, Strides<float, 1, 16>>>);

    /* Test adding a newaxis */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.newAxis<3U>())>,
                                 Shape<Lengths<2, 4, 6, 1>, Strides<float, 1, 16, 96, 0>>>);

    /* Test adding a newaxis */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.newAxis<1U>())>,
                                 Shape<Lengths<2, 1, 4, 6>, Strides<float, 1, 0, 16, 96>>>);

    /* Test removing single dimensions */
    Shape<Lengths<1, 1, 2, 1, 1, 1, 4, 1, 6, 1, 1>, Strides<float, 1, 16, 16, 32, 32, 32, 32, 128, 128, 768, 768>>
        shapeWithOnes;
    static_assert(std::is_same_v<std::decay_t<decltype(shapeWithOnes.squeeze())>,
                                 Shape<Lengths<2, 4, 6>, Strides<float, 16, 32, 128>>>);

    /* Test if shape contains a padding between the first and second dimension */
    static_assert(shape.containsPadding() == true);

    /* Test if shape contains a padding between the first and second dimension */
    auto shapeWithPaddingPermuted = shape.fastPermute<2, 1, 0>();
    static_assert(shapeWithPaddingPermuted.containsPadding() == true);

    /* Test if shape contains a padding between the first and second dimension */
    Shape<Lengths<16, 32, 64>, Strides<int, 1, 16, 512>> shapeNoPadding;
    static_assert(shapeNoPadding.containsPadding() == false);

    /* Test if shape contains a padding between the first and second dimension */
    Shape<Lengths<16, 64, 32>, Strides<int, 1, 512, 16>> shapeNoPaddingPermuted;
    static_assert(shapeNoPaddingPermuted.containsPadding() == false);

    /* Test raveling */
    Shape<Lengths<16>, Strides<int, 1>> shapeOneDimension;
    static_assert(
        std::is_same_v<std::decay_t<decltype(shapeOneDimension.ravel())>, Shape<Lengths<16>, Strides<int, 1>>>);

    /* Test raveling */
    static_assert(
        std::is_same_v<std::decay_t<decltype(shapeNoPadding.ravel())>, Shape<Lengths<32768>, Strides<int, 1>>>);

    /* Test raveling */
    static_assert(
        std::is_same_v<std::decay_t<decltype(shapeNoPaddingPermuted.ravel())>, Shape<Lengths<32768>, Strides<int, 1>>>);

    /* Test retrieving lengths */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.lengths())>, Lengths<2, 4, 6>>);

    /* Test retrieving a single length */
    static_assert(shape.template length<0U>() == 2);
    static_assert(shape.template length<1U>() == 4);
    static_assert(shape.template length<2U>() == 6);

    /* Test retrieving strides */
    static_assert(std::is_same_v<std::decay_t<decltype(shape.strides())>, Strides<float, 1, 16, 96>>);

    /* Test retrieving a single stride */
    static_assert(shape.template stride<0U>() == 1);
    static_assert(shape.template stride<1U>() == 16);
    static_assert(shape.template stride<2U>() == 96);

    /* Test rank */
    static_assert(shape.rank() == 3U);
}

void testAliases() {
    /* Test shape with aligned strides */
    static_assert(std::is_same_v<TShape<float, 3, 6, 18>, Shape<Lengths<3, 6, 18>, Strides<float, 1, 4, 24>>>);

    /* Test shape with aligned strides and defined Lengths class */
    static_assert(std::is_same_v<TLShape<int, Lengths, 2, 4, 8>, Shape<Lengths<2, 4, 8>, Strides<int, 1, 4, 16>>>);

    /* Test shape with unaligned strides */
    static_assert(std::is_same_v<TUShape<double, 2, 6, 4>, Shape<Lengths<2, 6, 4>, Strides<double, 1, 2, 12>>>);

    /* Test shape with unaligned strides and defined Lengths class */
    static_assert(
        std::is_same_v<TULShape<char, Lengths, 12, 4, 16>, Shape<Lengths<12, 4, 16>, Strides<char, 1, 12, 48>>>);

    /* Test shape with partially computed strides */
    static_assert(std::is_same_v<TPShape<Lengths<2, 1, 6, 8, 4, 3>, Strides<bool, 1, 4, 4, 24>>,
                                 Shape<Lengths<2, 1, 6, 8, 4, 3>, Strides<bool, 1, 4, 4, 24, 192, 768>>>);

    /* Test shape with partially computed aligned strides */
    static_assert(std::is_same_v<TPShape<Lengths<2, 1, 6, 3>, Strides<int, 1>>,
                                 Shape<Lengths<2, 1, 6, 3>, Strides<int, 1, 4, 4, 24>>>);

    /* Test shape with partially computed unaligned strides */
    static_assert(std::is_same_v<TPShape<Lengths<2, 1, 6, 8, 4, 3>, Strides<bool, 1>, false>,
                                 Shape<Lengths<2, 1, 6, 8, 4, 3>, Strides<bool, 1, 2, 2, 12, 96, 384>>>);
}

int main() {
    testLengths();
    testStrides();
    testHelpers();
    testShape();
    testAliases();

    return 0;
}