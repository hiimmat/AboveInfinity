#include "Tensor.h"

#include <assert.h>

using namespace AboveInfinity;

void testTensorSize() {
    static_assert(internal::TensorSize<Shape<Lengths<2, 14, 7>, Strides<int, 1, 2, 24>>, 2>() == 336);
}

void testStackTensor() {
    /* Test default constructor */
    StackTensor<TShape<float, 4, 2, 6>, 2> dcTensor;
    static_assert(std::is_same_v<std::decay_t<decltype(dcTensor)>,
                                 StackTensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test move constructor */
    auto mvTensor = std::move(dcTensor);
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor)>,
                                 StackTensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test the view function */
    auto tview = mvTensor.view();
    static_assert(
        std::is_same_v<std::decay_t<decltype(tview)>, TensorView<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test slicingPointer with compile-time offset computation */
    auto offsetedPtrCT = mvTensor.slicingPointer<0, 1, 5>();
    assert(offsetedPtrCT == mvTensor.data() + 5 * mvTensor.stride<2U>() + mvTensor.stride<1U>());

    /* Test slicingPointer with runtime offset computation */
    auto offsetedPtrRT = mvTensor.slicingPointer(0, 1, 5);
    assert(offsetedPtrRT == mvTensor.data() + 5 * mvTensor.stride<2U>() + mvTensor.stride<1U>());

    /* Test shape function */
    static_assert(
        std::is_same_v<std::decay_t<decltype(mvTensor.shape())>, Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>>);

    /* Test lengths function */
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor.lengths())>, Lengths<4, 2, 6>>);

    /* Test length function */
    static_assert(mvTensor.template length<0U>() == 4);
    static_assert(mvTensor.template length<1U>() == 2);
    static_assert(mvTensor.template length<2U>() == 6);

    /* Test strides function */
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor.strides())>, Strides<float, 1, 4, 8>>);

    /* Test stride function */
    static_assert(mvTensor.template stride<0U>() == 1);
    static_assert(mvTensor.template stride<1U>() == 4);
    static_assert(mvTensor.template stride<2U>() == 8);

    /* Test planes function */
    static_assert(mvTensor.planes() == 2U);

    /* Test rank function */
    static_assert(mvTensor.rank() == 3U);

    /* Test alias */
    static_assert(std::is_same_v<Tensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>,
                                 StackTensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);
}

void testHeapTensor() {
    /* Test default constructor */
    HeapTensor<TShape<float, 4, 2, 6>, 2> dcTensor;
    static_assert(std::is_same_v<std::decay_t<decltype(dcTensor)>,
                                 HeapTensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test move constructor */
    auto mvTensor = std::move(dcTensor);
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor)>,
                                 HeapTensor<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test the view function */
    auto tview = mvTensor.view();
    static_assert(
        std::is_same_v<std::decay_t<decltype(tview)>, TensorView<Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>, 2>>);

    /* Test slicingPointer with compile-time offset computation */
    auto offsetedPtrCT = mvTensor.slicingPointer<0, 1, 5>();
    assert(offsetedPtrCT == mvTensor.data() + 5 * mvTensor.stride<2U>() + mvTensor.stride<1U>());

    /* Test slicingPointer with runtime offset computation */
    auto offsetedPtrRT = mvTensor.slicingPointer(0, 1, 5);
    assert(offsetedPtrRT == mvTensor.data() + 5 * mvTensor.stride<2U>() + mvTensor.stride<1U>());

    /* Test shape function */
    static_assert(
        std::is_same_v<std::decay_t<decltype(mvTensor.shape())>, Shape<Lengths<4, 2, 6>, Strides<float, 1, 4, 8>>>);

    /* Test lengths function */
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor.lengths())>, Lengths<4, 2, 6>>);

    /* Test length function */
    static_assert(mvTensor.template length<0U>() == 4);
    static_assert(mvTensor.template length<1U>() == 2);
    static_assert(mvTensor.template length<2U>() == 6);

    /* Test strides function */
    static_assert(std::is_same_v<std::decay_t<decltype(mvTensor.strides())>, Strides<float, 1, 4, 8>>);

    /* Test stride function */
    static_assert(mvTensor.template stride<0U>() == 1);
    static_assert(mvTensor.template stride<1U>() == 4);
    static_assert(mvTensor.template stride<2U>() == 8);

    /* Test planes function */
    static_assert(mvTensor.planes() == 2U);

    /* Test rank function */
    static_assert(mvTensor.rank() == 3U);

    /* Test alias */
    static_assert(std::is_same_v<Tensor<Shape<Lengths<1024, 2, 6>, Strides<float, 1, 1024, 2048>>, 2>,
                                 HeapTensor<Shape<Lengths<1024, 2, 6>, Strides<float, 1, 1024, 2048>>, 2>>);
}

int main() {
    testTensorSize();
    testStackTensor();
    testHeapTensor();

    return 0;
}