#include "TensorOps.h"

#include <assert.h>
#include <iostream>

using namespace AboveInfinity;

void testHelpers() {
    TShape<int, 12, 4, 6> initialShape;
    auto permuted = initialShape.fastPermute<2U, 1U, 0U>();

    /* Test function determineInitialShape */
    auto determinedInitialAfterPermute = internal::determineInitialShape(permuted);
    auto determinedInitialWithoutPermute = internal::determineInitialShape(initialShape);
    static_assert(std::is_same_v<TShape<int, 12, 4, 6>, std::decay_t<decltype(determinedInitialAfterPermute)>>);
    static_assert(std::is_same_v<TShape<int, 12, 4, 6>, std::decay_t<decltype(determinedInitialWithoutPermute)>>);

    /* Test canReshapePaddedInplaceImpl */
    static_assert(internal::canReshapePaddedInplace<24>(Lengths<4, 3, 2, 6>()) == true);
    static_assert(internal::canReshapePaddedInplace<15>(Lengths<5, 1, 2, 7, 3>()) == false);

    /* Test numElementsToCopy */
    static_assert(internal::numElementsToCopy<6>(Lengths<6, 3, 2, 6>()) == 1);
    static_assert(internal::numElementsToCopy<24>(Lengths<4, 3, 2, 7, 3>()) == 3);
    static_assert(internal::numElementsToCopy<15>(Lengths<5, 1, 3>()) == 3);

    /* Test partialReshapedStrides */
    static_assert(
        std::is_same_v<std::decay_t<decltype(internal::partialReshapedStrides<int, 2, Lengths<12, 1, 4, 3, 1, 5>, 2>())>,
                       Strides<int, 1, 12, 1, 2>>);

    /* Test type_name */
    assert(internal::type_name<int>() == "int");
    assert(internal::type_name<float>() == "float");
    assert(internal::type_name<double>() == "double");
    assert(internal::type_name<bool>() == "bool");
    assert(internal::type_name<char>() == "char");

    /* Test front */
    TensorView<TShape<int, 2, 4, 6>, 2> x;
    TensorView<TShape<int, 3, 7, 2>, 1> y;
    auto z = internal::front(x, y);
    static_assert(std::is_same_v<decltype(x), decltype(z)>);
    assert(x.data() == z.data());
    assert(x.data<1U>() == z.data<1U>());
}

void testExecute() {
    Tensor<TShape<int, 2, 2, 2>, 2> fst;
    Tensor<TShape<float, 2, 2, 2>, 2> snd;
    Tensor<TShape<double, 4, 3, 2>, 2> thd;

    TensorView<TShape<int, 2, 2, 2>, 2> fstView = fst.view();
    TensorView<TShape<float, 2, 2, 2>, 2> sndView = snd.view();
    TensorView<TShape<double, 4, 3, 2>, 2> thdView = thd.view();

    /* Test execute with 2 TensorViews simultaneously */
    execute(
        [](int& x, float& y) {
            static int i = 1;
            static float j = 1.0F;
            x = i++;
            y = j++;
        },
        fstView,
        sndView);

    int intVal = 1;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k) {
                    assert(*fstView.slicingPointer(p, k, j, i) == intVal);
                    intVal++;
                }

    float floatVal = 1.0F;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k) {
                    assert(*sndView.slicingPointer(p, k, j, i) == floatVal);
                    floatVal++;
                }

    /* Test execute with a single TensorView */
    execute([](double& x) { x = 6.0; }, thdView);

    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k) assert(*thdView.slicingPointer(p, k, j, i) == 6.0);

    /* Test execute with 2 TensorViews not simultaneously */
    execute([](auto& x) { x *= 2; }, fstView, thdView);

    intVal = 1;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k) {
                    assert(*fstView.slicingPointer(p, k, j, i) == intVal * 2);
                    intVal++;
                }

    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 2; ++k) assert(*thdView.slicingPointer(p, k, j, i) == 12.0);

    /* BC = batched */
    Tensor<TShape<int, 12, 2, 3>, 2> fstBC;
    Tensor<TShape<float, 12, 2, 3>, 2> sndBC;
    Tensor<TShape<double, 14, 3, 2>, 2> thdBC;

    TensorView<TShape<int, 12, 2, 3>, 2> fstBCView = fstBC.view();
    TensorView<TShape<float, 12, 2, 3>, 2> sndBCView = sndBC.view();
    TensorView<TShape<double, 14, 3, 2>, 2> thdBCView = thdBC.view();

    /* Test execute with 2 TensorViews simultaneously in batches */
    execute(
        [](int& x, float& y) {
            static int i = 1;
            static float j = 1.0F;
            x = i++;
            y = j++;
        },
        fstBCView,
        sndBCView);

    intVal = 1;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 12; ++k) {
                    assert(*fstBCView.slicingPointer(p, k, j, i) == intVal);
                    intVal++;
                }

    floatVal = 1.0F;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 12; ++k) {
                    assert(*sndBCView.slicingPointer(p, k, j, i) == floatVal);
                    floatVal++;
                }

    /* Test execute with a single TensorView in batches */
    execute([](double& x) { x = 6.0; }, thdBCView);

    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 14; ++k) assert(*thdBCView.slicingPointer(p, k, j, i) == 6.0);

    /* Test execute with 2 TensorViews not simultaneously in batches */
    execute([](auto& x) { x *= 2; }, fstBCView, thdBCView);

    intVal = 1;
    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 2; ++j)
                for(int k = 0; k < 12; ++k) {
                    assert(*fstBCView.slicingPointer(p, k, j, i) == intVal * 2);
                    intVal++;
                }

    for(std::size_t p = 0U; p < 2U; ++p)
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 14; ++k) assert(*thdBCView.slicingPointer(p, k, j, i) == 12.0);
}

void testFlatten() {
    /* KO - keepOrder */
    /* NP - no permutation */
    /* SO = stride one */
    /* SNO = stride not one */

    Tensor<TShape<int, 12>, 1> vec;
    Tensor<TShape<int, 12, 1, 1, 1>, 1> vecSqueeze;
    Tensor<TShape<int, 3, 4, 2>, 2> paddedTensor;
    Tensor<TShape<int, 4, 6, 2>, 2> unpaddedTensor;
    Tensor<TShape<int, 17, 2, 3>, 2> paddedBatchTensor;
    Tensor<TShape<int, 16, 2, 4>, 2> unpaddedBatchTensor;

    TensorView<TShape<int, 12>, 1> vecView = vec.view();
    TensorView<TShape<int, 3, 4, 2>, 2> paddedTensorView = paddedTensor.view();
    TensorView<TShape<int, 4, 6, 2>, 2> unpaddedTensorView = unpaddedTensor.view();
    TensorView<TShape<int, 17, 2, 3>, 2> paddedBatchTensorView = paddedBatchTensor.view();
    TensorView<TShape<int, 16, 2, 4>, 2> unpaddedBatchTensorView = unpaddedBatchTensor.view();

    auto flattenedVecKO = flatten<true>(vecView);
    static_assert(std::is_same_v<decltype(flattenedVecKO), decltype(vecView)>);
    assert(flattenedVecKO.data() == vec.data());

    auto flattenedVecSqueeze = flatten<true>(vecSqueeze.view());
    static_assert(std::is_same_v<decltype(flattenedVecSqueeze), TensorView<Shape<Lengths<12>, Strides<int, 1>>, 1>>);
    assert(flattenedVecSqueeze.data() == vecSqueeze.data());

    auto flattenedVec = flatten<false>(vecView);
    static_assert(std::is_same_v<decltype(flattenedVec), decltype(vecView)>);
    assert(flattenedVec.data() == vec.data());

    execute(
        [](int& x) {
            static int i = 1;
            x = i++;
        },
        paddedTensorView,
        unpaddedTensorView,
        paddedBatchTensorView,
        unpaddedBatchTensorView);

    /* Test flatten with keeping the memory order on a tensor with padding */
    auto flattenedPaddedTensorKO = flatten<true>(paddedTensorView);
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorKO), Tensor<TShape<int, 24>, 2>>);

    int val = 1;
    for(std::size_t p = 0U; p < flattenedPaddedTensorKO.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorKO.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorKO.slicingPointer(p, i) == val);
            val++;
        }
    /*
     * Test flatten with not keeping the memory order, without a permutated tensor (ignores the newaxis)
     * on a tensor with padding
     */
    auto flattenedPaddedTensorNP = flatten<false>(paddedTensorView.newAxis<1U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorNP), Tensor<TShape<int, 24>, 2>>);

    val = 1;
    for(std::size_t p = 0U; p < flattenedPaddedTensorNP.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorNP.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorKO.slicingPointer(p, i) == val);
            val++;
        }

    /* Test flatten with keeping the memory order on a tensor that without a padding */
    auto flattenedUnpaddedTensorKO = flatten<true>(unpaddedTensorView);
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorKO), Tensor<TShape<int, 48>, 2>>);

    for(std::size_t p = 0U; p < flattenedUnpaddedTensorKO.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorKO.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorKO.slicingPointer(p, i) == val);
            val++;
        }

    /*
     * Test flatten with not keeping the memory order, without a permutated tensor,
     * without a padding
     */
    auto flattenedUnpaddedTensorNP = flatten<false>(unpaddedTensorView);
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorNP), Tensor<TShape<int, 48>, 2>>);

    val = 49;
    for(std::size_t p = 0U; p < flattenedUnpaddedTensorNP.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorNP.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorNP.slicingPointer(p, i) == val);
            val++;
        }

    /* Test flatten with keeping the memory order on a tensor with padding in batches */
    auto flattenedPaddedBatchTensorKO = flatten<true>(paddedBatchTensorView);
    static_assert(std::is_same_v<decltype(flattenedPaddedBatchTensorKO), Tensor<TShape<int, 102>, 2>>);

    for(std::size_t p = 0U; p < flattenedPaddedBatchTensorKO.planes(); ++p)
        for(int i = 0; i < flattenedPaddedBatchTensorKO.template length<0U>(); ++i) {
            assert(*flattenedPaddedBatchTensorKO.slicingPointer(p, i) == val);
            val++;
        }

    /*
     * Test flatten with not keeping the memory order, without a permutated tensor (ignores the newaxis)
     * on a tensor with padding in batches
     */
    auto flattenedPaddedBatchTensorNP = flatten<false>(paddedBatchTensorView.newAxis<1U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedBatchTensorNP), Tensor<TShape<int, 102>, 2>>);

    val = 145;
    for(std::size_t p = 0U; p < flattenedPaddedBatchTensorNP.planes(); ++p)
        for(int i = 0; i < flattenedPaddedBatchTensorNP.template length<0U>(); ++i) {
            assert(*flattenedPaddedBatchTensorNP.slicingPointer(p, i) == val);
            val++;
        }

    /* Test flatten with keeping the memory order on a tensor that without a padding in batches */
    auto flattenedUnpaddedBatchTensorKO = flatten<true>(unpaddedBatchTensorView);
    static_assert(std::is_same_v<decltype(flattenedUnpaddedBatchTensorKO), Tensor<TShape<int, 128>, 2>>);

    val = 349;
    for(std::size_t p = 0U; p < flattenedUnpaddedBatchTensorKO.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedBatchTensorKO.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedBatchTensorKO.slicingPointer(p, i) == val);
            val++;
        }

    /*
     * Test flatten with not keeping the memory order, without a permutated tensor,
     * without a padding in batches
     */
    auto flattenedUnpaddedBatchTensorNP = flatten<false>(unpaddedBatchTensorView);
    static_assert(std::is_same_v<decltype(flattenedUnpaddedBatchTensorNP), Tensor<TShape<int, 128>, 2>>);

    val = 349;
    for(std::size_t p = 0U; p < flattenedUnpaddedBatchTensorNP.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedBatchTensorNP.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedBatchTensorNP.slicingPointer(p, i) == val);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride being 1, with padding */
    auto flattenedPaddedTensorPermutedSO = flatten<false>(paddedTensorView.fastPermute<0U, 2U, 1U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorPermutedSO), Tensor<TShape<int, 24>, 2>>);

    auto flattenedPaddedArray = std::array{1,  2,  3,  13, 14, 15, 4,  5,  6,  16, 17, 18, 7,  8,  9,  19,
                                           20, 21, 10, 11, 12, 22, 23, 24, 25, 26, 27, 37, 38, 39, 28, 29,
                                           30, 40, 41, 42, 31, 32, 33, 43, 44, 45, 34, 35, 36, 46, 47, 48};

    val = 0;
    for(std::size_t p = 0U; p < flattenedPaddedTensorPermutedSO.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorPermutedSO.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorPermutedSO.slicingPointer(p, i) == flattenedPaddedArray[val]);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride not being 1,
       with padding */
    auto flattenedPaddedTensorPermutedSNO = flatten<false>(paddedTensorView.fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorPermutedSNO), Tensor<TShape<int, 24>, 2>>);

    flattenedPaddedArray = std::array{1,  13, 4,  16, 7,  19, 10, 22, 2,  14, 5,  17, 8,  20, 11, 23,
                                      3,  15, 6,  18, 9,  21, 12, 24, 25, 37, 28, 40, 31, 43, 34, 46,
                                      26, 38, 29, 41, 32, 44, 35, 47, 27, 39, 30, 42, 33, 45, 36, 48};

    val = 0;
    for(std::size_t p = 0U; p < flattenedPaddedTensorPermutedSNO.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorPermutedSNO.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorPermutedSNO.slicingPointer(p, i) == flattenedPaddedArray[val]);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride being 1, without padding */
    auto flattenedUnpaddedTensorPermutedSO = flatten<false>(unpaddedTensorView.fastPermute<0U, 2U, 1U>());
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorPermutedSO), Tensor<TShape<int, 48>, 2>>);

    auto flattenedUnpaddedArray =
        std::array{49,  50,  51,  52,  73,  74,  75,  76,  53,  54,  55,  56,  77,  78,  79,  80,  57,  58,  59,  60,
                   81,  82,  83,  84,  61,  62,  63,  64,  85,  86,  87,  88,  65,  66,  67,  68,  89,  90,  91,  92,
                   69,  70,  71,  72,  93,  94,  95,  96,  97,  98,  99,  100, 121, 122, 123, 124, 101, 102, 103, 104,
                   125, 126, 127, 128, 105, 106, 107, 108, 129, 130, 131, 132, 109, 110, 111, 112, 133, 134, 135, 136,
                   113, 114, 115, 116, 137, 138, 139, 140, 117, 118, 119, 120, 141, 142, 143, 144};

    val = 0;
    for(std::size_t p = 0U; p < flattenedUnpaddedTensorPermutedSO.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorPermutedSO.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorPermutedSO.slicingPointer(p, i) == flattenedUnpaddedArray[val]);
            val++;
        }

    /*
     * Test flatten with not keeping the memory order, a permuted tensor with first stride not being 1,
     * without padding
     */
    auto flattenedUnpaddedTensorPermutedSNO = flatten<false>(unpaddedTensorView.fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorPermutedSNO), Tensor<TShape<int, 48>, 2>>);

    flattenedUnpaddedArray = std::array{49,  73,  53,  77,  57,  81,  61,  85,  65,  89,  69,  93,  50,  74,  54,  78,
                                        58,  82,  62,  86,  66,  90,  70,  94,  51,  75,  55,  79,  59,  83,  63,  87,
                                        67,  91,  71,  95,  52,  76,  56,  80,  60,  84,  64,  88,  68,  92,  72,  96,
                                        97,  121, 101, 125, 105, 129, 109, 133, 113, 137, 117, 141, 98,  122, 102, 126,
                                        106, 130, 110, 134, 114, 138, 118, 142, 99,  123, 103, 127, 107, 131, 111, 135,
                                        115, 139, 119, 143, 100, 124, 104, 128, 108, 132, 112, 136, 116, 140, 120, 144};

    val = 0;
    for(std::size_t p = 0U; p < flattenedUnpaddedTensorPermutedSNO.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorPermutedSNO.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorPermutedSNO.slicingPointer(p, i) == flattenedUnpaddedArray[val]);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride being 1, with padding in
     * batches*/
    auto flattenedPaddedTensorPermutedSOBatched = flatten<false>(paddedBatchTensorView.fastPermute<0U, 2U, 1U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorPermutedSOBatched), Tensor<TShape<int, 102>, 2>>);

    auto flattenedPaddedBatchedArray = std::array{
        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 179, 180, 181, 182,
        183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 213, 214, 215, 216, 217, 218, 219, 220,
        221, 222, 223, 224, 225, 226, 227, 228, 229, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
        174, 175, 176, 177, 178, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
        212, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
        250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 281, 282, 283, 284, 285, 286, 287,
        288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
        326, 327, 328, 329, 330, 331, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278,
        279, 280, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 332, 333,
        334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348};

    val = 0;
    for(std::size_t p = 0U; p < flattenedPaddedTensorPermutedSOBatched.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorPermutedSOBatched.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorPermutedSOBatched.slicingPointer(p, i) == flattenedPaddedBatchedArray[val]);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride not being 1, with padding in
     * batches*/
    auto flattenedPaddedTensorPermutedSNOBatched = flatten<false>(paddedBatchTensorView.fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensorPermutedSNOBatched), Tensor<TShape<int, 102>, 2>>);

    flattenedPaddedBatchedArray = std::array{
        145, 179, 213, 162, 196, 230, 146, 180, 214, 163, 197, 231, 147, 181, 215, 164, 198, 232, 148, 182, 216,
        165, 199, 233, 149, 183, 217, 166, 200, 234, 150, 184, 218, 167, 201, 235, 151, 185, 219, 168, 202, 236,
        152, 186, 220, 169, 203, 237, 153, 187, 221, 170, 204, 238, 154, 188, 222, 171, 205, 239, 155, 189, 223,
        172, 206, 240, 156, 190, 224, 173, 207, 241, 157, 191, 225, 174, 208, 242, 158, 192, 226, 175, 209, 243,
        159, 193, 227, 176, 210, 244, 160, 194, 228, 177, 211, 245, 161, 195, 229, 178, 212, 246, 247, 281, 315,
        264, 298, 332, 248, 282, 316, 265, 299, 333, 249, 283, 317, 266, 300, 334, 250, 284, 318, 267, 301, 335,
        251, 285, 319, 268, 302, 336, 252, 286, 320, 269, 303, 337, 253, 287, 321, 270, 304, 338, 254, 288, 322,
        271, 305, 339, 255, 289, 323, 272, 306, 340, 256, 290, 324, 273, 307, 341, 257, 291, 325, 274, 308, 342,
        258, 292, 326, 275, 309, 343, 259, 293, 327, 276, 310, 344, 260, 294, 328, 277, 311, 345, 261, 295, 329,
        278, 312, 346, 262, 296, 330, 279, 313, 347, 263, 297, 331, 280, 314, 348};

    val = 0;
    for(std::size_t p = 0U; p < flattenedPaddedTensorPermutedSNOBatched.planes(); ++p)
        for(int i = 0; i < flattenedPaddedTensorPermutedSNOBatched.template length<0U>(); ++i) {
            assert(*flattenedPaddedTensorPermutedSNOBatched.slicingPointer(p, i) == flattenedPaddedBatchedArray[val]);
            val++;
        }

    /* Test flatten with not keeping the memory order, a permuted tensor with first stride being 1, without padding in
     * batches*/
    auto flattenedUnpaddedTensorPermutedSOBatched = flatten<false>(unpaddedBatchTensorView.fastPermute<0U, 2U, 1U>());
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorPermutedSOBatched), Tensor<TShape<int, 128>, 2>>);

    auto flattenedUnpaddedBatchedArray = std::array{
        349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 381, 382, 383, 384, 385, 386,
        387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424,
        425, 426, 427, 428, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 365, 366,
        367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 397, 398, 399, 400, 401, 402, 403, 404,
        405, 406, 407, 408, 409, 410, 411, 412, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442,
        443, 444, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
        481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518,
        519, 520, 521, 522, 523, 524, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556,
        573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 493, 494, 495, 496, 497, 498,
        499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536,
        537, 538, 539, 540, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 589, 590,
        591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604};

    val = 0;
    for(std::size_t p = 0U; p < flattenedUnpaddedTensorPermutedSOBatched.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorPermutedSOBatched.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorPermutedSOBatched.slicingPointer(p, i) ==
                   flattenedUnpaddedBatchedArray[val]);
            val++;
        }

    /*
     * Test flatten with not keeping the memory order, a permuted tensor with first stride not being 1,
     * without padding in batches
     */
    auto flattenedUnpaddedTensorPermutedSNOBatched = flatten<false>(unpaddedBatchTensorView.fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(flattenedUnpaddedTensorPermutedSNOBatched), Tensor<TShape<int, 128>, 2>>);

    flattenedUnpaddedBatchedArray = std::array{
        349, 381, 413, 445, 365, 397, 429, 461, 350, 382, 414, 446, 366, 398, 430, 462, 351, 383, 415, 447, 367, 399,
        431, 463, 352, 384, 416, 448, 368, 400, 432, 464, 353, 385, 417, 449, 369, 401, 433, 465, 354, 386, 418, 450,
        370, 402, 434, 466, 355, 387, 419, 451, 371, 403, 435, 467, 356, 388, 420, 452, 372, 404, 436, 468, 357, 389,
        421, 453, 373, 405, 437, 469, 358, 390, 422, 454, 374, 406, 438, 470, 359, 391, 423, 455, 375, 407, 439, 471,
        360, 392, 424, 456, 376, 408, 440, 472, 361, 393, 425, 457, 377, 409, 441, 473, 362, 394, 426, 458, 378, 410,
        442, 474, 363, 395, 427, 459, 379, 411, 443, 475, 364, 396, 428, 460, 380, 412, 444, 476, 477, 509, 541, 573,
        493, 525, 557, 589, 478, 510, 542, 574, 494, 526, 558, 590, 479, 511, 543, 575, 495, 527, 559, 591, 480, 512,
        544, 576, 496, 528, 560, 592, 481, 513, 545, 577, 497, 529, 561, 593, 482, 514, 546, 578, 498, 530, 562, 594,
        483, 515, 547, 579, 499, 531, 563, 595, 484, 516, 548, 580, 500, 532, 564, 596, 485, 517, 549, 581, 501, 533,
        565, 597, 486, 518, 550, 582, 502, 534, 566, 598, 487, 519, 551, 583, 503, 535, 567, 599, 488, 520, 552, 584,
        504, 536, 568, 600, 489, 521, 553, 585, 505, 537, 569, 601, 490, 522, 554, 586, 506, 538, 570, 602, 491, 523,
        555, 587, 507, 539, 571, 603, 492, 524, 556, 588, 508, 540, 572, 604};

    val = 0;
    for(std::size_t p = 0U; p < flattenedUnpaddedTensorPermutedSNOBatched.planes(); ++p)
        for(int i = 0; i < flattenedUnpaddedTensorPermutedSNOBatched.template length<0U>(); ++i) {
            assert(*flattenedUnpaddedTensorPermutedSNOBatched.slicingPointer(p, i) ==
                   flattenedUnpaddedBatchedArray[val]);
            val++;
        }
}

void testFileOps() {
    Tensor<TShape<int, 20, 3, 2>, 2> storedTensor;
    Tensor<TShape<int, 20, 3, 2>, 2> loadedTensor;
    Tensor<TShape<int, 2, 3, 20>, 2> loadedPermutedTensor;

    execute(
        [](int& x) {
            static int i = 1;
            x = i++;
        },
        storedTensor.view());

    /*
     * Test that the functions saveAsText and loadFromtext work by storing a tensor to a file and then loading the saved
     * tensor into another tensor. The loaded tensor has to match the tensor that's being stored into the file
     */
    saveAsText(storedTensor.view(), "storedTensor.txt");
    loadFromText(loadedTensor.view(), "storedTensor.txt");
    execute([](int& x, int& y) { assert(x == y); }, storedTensor.view(), loadedTensor.view());

    /* Test that the function loadFromtext works with a permuted output */
    loadFromText(loadedPermutedTensor.view().fastPermute<2U, 1U, 0U>(), "storedTensor.txt");
    execute([](int& x, int& y) { assert(x == y); },
            storedTensor.view(),
            loadedPermutedTensor.view().fastPermute<2U, 1U, 0U>());

    /* Test that the function saveAsText works with a permuted input */
    saveAsText(storedTensor.view().fastPermute<2U, 1U, 0U>(), "storedPermutedTensor.txt");
    loadFromText(loadedPermutedTensor.view(), "storedPermutedTensor.txt");
    execute([](int& x, int& y) { assert(x == y); },
            storedTensor.view().fastPermute<2U, 1U, 0U>(),
            loadedPermutedTensor.view());
}

void testCopy() {
    Tensor<TShape<int, 2, 12, 7>, 2> unbatchedSrc;
    Tensor<TShape<int, 2, 12, 7>, 2> unbatchedDst;

    Tensor<TShape<int, 14, 2, 5>, 2> batchedSrc;
    Tensor<TShape<int, 14, 2, 5>, 2> batchedDst;

    Tensor<TShape<int, 5, 2, 14>, 2> permutedDest;
    auto permutedView = permutedDest.view().fastPermute<2U, 1U, 0U>();

    execute(
        [](int& x) {
            static int i = 1;
            x = i++;
        },
        unbatchedSrc.view());

    execute(
        [](int& x) {
            static int i = 2;
            x = i;
            i += 2;
        },
        batchedSrc.view());

    /* Test copy with same lengths and type - no batch */
    copy(unbatchedSrc.view(), unbatchedDst.view());
    execute([](int& x, int& y) { assert(x == y); }, unbatchedSrc.view(), unbatchedDst.view());

    /* Test copy with same lengths and type - batched */
    copy(batchedSrc.view(), batchedDst.view());
    execute([](int& x, int& y) { assert(x == y); }, batchedSrc.view(), batchedDst.view());

    /* Test copy with different stride - execute is called */
    copy(batchedSrc.view(), permutedView);
    execute([](int& x, int& y) { assert(x == y); }, batchedSrc.view(), permutedView);
}

void testReshape() {
    Tensor<TShape<bool, 16, 6, 2>, 2> initialTensor;
    TensorView<TShape<bool, 16, 6, 2>, 2> initialView = initialTensor.view();

    /* Test reshape with the same lengths as the passed in TensorView */
    auto reshapeSameView = reshape<Lengths<16, 6, 2>>(initialView);
    static_assert(std::is_same_v<decltype(initialView), decltype(reshapeSameView)>);
    assert(reshapeSameView.data() == initialView.data());

    /* Test reshape with no padding */
    auto reshapeNoPadding = reshape<Lengths<1, 2, 4, 2, 3, 1, 2, 2>>(initialView);
    static_assert(std::is_same_v<decltype(reshapeNoPadding), TensorView<TUShape<bool, 1, 2, 4, 2, 3, 1, 2, 2>, 2>>);
    assert(reshapeNoPadding.data() == initialView.data());

    /* Test reshape with padding, but same first dimension */
    Tensor<TShape<bool, 11, 2, 6>, 2> paddedTensor;
    TensorView<TShape<bool, 11, 2, 6>, 2> paddedView = paddedTensor.view();
    auto paddedReshapeInPlace = reshape<Lengths<11, 1, 2, 3, 2>>(paddedView);
    //    static_assert(std::is_same_v<decltype(paddedReshapeInPlace),
    //                                 TensorView<TPShape<Lengths<11, 1, 2, 3, 2>, Strides<bool, 1, 12>>, 2>>);
    TensorView<TPShape<Lengths<11, 1, 2, 3, 2>, Strides<bool, 1, 16>>, 2> x;
    assert(paddedReshapeInPlace.data() == paddedTensor.data());

    /* Test reshape with padding, and different first dimension - no batch */
    /* NB = No batch */
    Tensor<TShape<float, 3, 4, 6>, 2> paddedTensorNB;
    TensorView<TShape<float, 3, 4, 6>, 2> paddedViewNB = paddedTensorNB.view();

    execute(
        [](float& x) {
            static float i = 1.0;
            x = i++;
        },
        paddedViewNB);

    auto reshapePaddedNB = reshape<Lengths<6, 2, 3, 2>>(paddedViewNB);
    static_assert(std::is_same_v<decltype(reshapePaddedNB), Tensor<TShape<float, 6, 2, 3, 2>, 2>>);

    execute(
        [](float& x) {
            static float i = 1.0F;
            assert(x == i);
            i++;
        },
        reshapePaddedNB.view());

    /* Test reshape with padding, and different first dimension - batched */
    Tensor<TShape<float, 25, 3, 2>, 2> paddedTensorBatch;
    TensorView<TShape<float, 25, 3, 2>, 2> paddedViewBatch = paddedTensorBatch.view();

    execute(
        [](float& x) {
            static float i = 1.0;
            x = i++;
        },
        paddedViewBatch);

    auto reshapePaddedBatch = reshape<Lengths<75, 2>>(paddedViewBatch);
    static_assert(std::is_same_v<decltype(reshapePaddedBatch), Tensor<TShape<float, 75, 2>, 2>>);
    execute(
        [](float& x) {
            static float i = 1.0F;
            assert(x == i);
            i++;
        },
        reshapePaddedBatch.view());

    /* Test reshape with padding, and different first dimension - batched, but the result has no padding */
    /* NPO = No padding output */
    Tensor<TShape<int, 26, 2, 3>, 2> paddedTensorBatchNPO;

    execute(
        [](int& x) {
            static int i = 1;
            x = i++;
        },
        paddedTensorBatchNPO.view());

    auto reshapedPaddedTensorBatchNP = reshape<Lengths<52, 3>>(paddedTensorBatchNPO.view());
    static_assert(std::is_same_v<decltype(reshapedPaddedTensorBatchNP), Tensor<TShape<int, 52, 3>, 2>>);
    static_assert(!reshapedPaddedTensorBatchNP.shape().containsPadding());
    execute(
        [](int& x) {
            static int i = 1;
            assert(x == i);
            i++;
        },
        reshapedPaddedTensorBatchNP.view());

    /* Test reshape with padding, and different first dimension - no batch and the result has no padding */
    /* NPO = No padding output */
    Tensor<TShape<int, 2, 2, 2, 3>, 2> paddedTensorNPO;

    execute(
        [](int& x) {
            static int i = 1;
            x = i++;
        },
        paddedTensorNPO.view());

    auto reshapedPaddedTensorNP = reshape<Lengths<8, 3>>(paddedTensorNPO.view());
    static_assert(std::is_same_v<decltype(reshapedPaddedTensorNP), Tensor<TShape<int, 8, 3>, 2>>);
    static_assert(!reshapedPaddedTensorNP.shape().containsPadding());
    execute(
        [](int& x) {
            static int i = 1;
            assert(x == i);
            i++;
        },
        reshapedPaddedTensorNP.view());
}

void testRavel() {
    Tensor<TShape<int, 12>, 2U> vec;
    Tensor<TShape<int, 16, 1, 1, 1, 1>, 2U> vecSqueeze;
    Tensor<TShape<int, 6, 4, 2>, 2U> paddedTensor;
    Tensor<TShape<int, 16, 2, 4>, 1U> unpaddedTensor;

    /* Test ravel on a 1D tensor */
    auto flattenedVecView = ravel<true>(vec.view());
    static_assert(std::is_same_v<decltype(flattenedVecView), TensorView<TShape<int, 12>, 2>>);
    assert(flattenedVecView.data() == vec.data());
    assert(flattenedVecView.data<1U>() == vec.data<1U>());

    /* Test ravel on a tensor with no padding that is 1D when it's squeezed */
    auto squeezedVecView = ravel<true>(vecSqueeze.view());
    static_assert(std::is_same_v<decltype(squeezedVecView), TensorView<TShape<int, 16>, 2>>);
    assert(squeezedVecView.data() == vecSqueeze.data());
    assert(squeezedVecView.data<1U>() == vecSqueeze.data<1U>());

    /* Test ravel on a padded tensor - this should call flatten which we test elsewhere */
    auto flattenedPaddedTensor = ravel<true>(paddedTensor.view());
    static_assert(std::is_same_v<decltype(flattenedPaddedTensor), Tensor<TShape<int, 48>, 2>>);
    assert(flattenedPaddedTensor.data() != paddedTensor.data());
    assert(flattenedPaddedTensor.data<1U>() != paddedTensor.data<1U>());

    /* Test ravel on an unpadded tensor and keep the memory order */
    /* KM = keep memory order */
    auto unpaddedViewKM = ravel<true>(unpaddedTensor.view().fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(unpaddedViewKM), TensorView<TShape<int, 128>, 1U>>);
    assert(unpaddedViewKM.data() == unpaddedTensor.data());

    /* Test ravel on an unpadded tensor. Don't keep the memory order, but there's no permutation */
    auto unpaddedView = ravel<true>(unpaddedTensor.view());
    static_assert(std::is_same_v<decltype(unpaddedView), TensorView<TShape<int, 128>, 1U>>);
    assert(unpaddedView.data() == unpaddedTensor.data());

    /*
     * Test ravel on an unpadded tensor. Don't keep the memory order and perform a permutation.
     * This should call flatten which we test elsewhere
     */
    auto unpaddedTensorPermuted = ravel<false>(unpaddedTensor.view().fastPermute<2U, 1U, 0U>());
    static_assert(std::is_same_v<decltype(unpaddedTensorPermuted), Tensor<TShape<int, 128>, 1U>>);
    assert(unpaddedTensorPermuted.data() != unpaddedTensor.data());
}

void testTensorOps() {
    testExecute();
    testFlatten();
    testFileOps();
    testCopy();
    testReshape();
    testRavel();
}

int main() {
    testHelpers();
    testTensorOps();

    return 0;
}