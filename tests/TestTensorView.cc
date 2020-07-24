#include "AlignedMemory.h"
#include "HardwareFeatures.h"
#include "TensorView.h"

#include <assert.h>

using namespace AboveInfinity;

void testTensorView() {
    /* Test default constructor */
    TensorView<TShape<int, 4, 7, 12>, 1U> dview;
    static_assert(std::is_same_v<std::decay_t<decltype(dview.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(dview.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(dview.planes() == 1U);

    /* Allocate memory for testing */
    AlignedMemory<int> fpmem{336U, AIAlignment};
    AlignedMemory<int> spmem{336U, AIAlignment};
    AlignedMemory<int> tpmem{336U, AIAlignment};

    /* Test constructor receiving arrays */
    std::array<int*, 2> _ptrs{fpmem.data(), spmem.data()};
    TensorView<TShape<int, 4, 7, 12>, 2U> aview{_ptrs};
    static_assert(std::is_same_v<std::decay_t<decltype(aview.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(aview.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(aview.planes() == 2U);
    assert(aview.data() == fpmem.data());
    assert(aview.data<1U>() == spmem.data());

    /* Test constructor receiving pointers */
    TensorView<TShape<int, 4, 7, 12>, 2U> pview{fpmem.data(), spmem.data()};
    static_assert(std::is_same_v<std::decay_t<decltype(pview.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(pview.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(pview.planes() == 2U);
    assert(pview.data() == fpmem.data());
    assert(pview.data<1U>() == spmem.data());

    /* Test fastPermute */
    auto permuted = pview.fastPermute<2, 1, 0>();
    static_assert(std::is_same_v<std::decay_t<decltype(permuted)>,
                                 TensorView<Shape<Lengths<12, 7, 4>, Strides<int, 28, 4, 1>>, 2U>>);
    assert(permuted.data() == fpmem.data());
    assert(permuted.data<1U>() == spmem.data());

    /* Test undoPermutation */
    auto unpermuted = permuted.undoPermutation();
    static_assert(std::is_same_v<std::decay_t<decltype(unpermuted)>,
                                 TensorView<Shape<Lengths<4, 7, 12>, Strides<int, 1, 4, 28>>, 2U>>);
    assert(unpermuted.data() == fpmem.data());
    assert(unpermuted.data<1U>() == spmem.data());

    /* Test slicingPointer with compile-time offset computation */
    auto offsetedPtrCT = pview.slicingPointer<0, 0, 6>();
    assert(offsetedPtrCT == fpmem.data() + 6 * pview.stride<2U>());

    /* Test slicingPointer with runtime offset computation */
    auto offsetedPtrRT = pview.slicingPointer(0, 0, 6);
    assert(offsetedPtrRT == fpmem.data() + 6 * pview.stride<2U>());

    /* Test compile-time slice */
    auto ctSlice = pview.slice<1, 2>();
    static_assert(std::is_same_v<std::decay_t<decltype(ctSlice.lengths())>, Lengths<4, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(ctSlice.strides())>, Strides<int, 1, 28>>);
    static_assert(ctSlice.planes() == 2U);
    assert(ctSlice.data() == fpmem.data() + 2 * pview.stride<1U>());
    assert(ctSlice.data<1U>() == spmem.data() + 2 * pview.stride<1U>());

    /* Test run-time slice */
    auto rtSlice = pview.slice<2, 0>();
    static_assert(std::is_same_v<std::decay_t<decltype(rtSlice.lengths())>, Lengths<4, 7>>);
    static_assert(std::is_same_v<std::decay_t<decltype(rtSlice.strides())>, Strides<int, 1, 4>>);
    static_assert(rtSlice.planes() == 2U);
    assert(rtSlice.data() == fpmem.data());
    assert(rtSlice.data<1U>() == spmem.data());

    /* Test slab */
    auto slabView = pview.slab<0, 1, 4>();
    static_assert(std::is_same_v<std::decay_t<decltype(slabView.lengths())>, Lengths<3, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(slabView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(slabView.planes() == 2U);
    assert(slabView.data() == fpmem.data() + slabView.stride<0U>());
    assert(slabView.data<1U>() == spmem.data() + slabView.stride<0U>());

    /* Test subspace */
    auto fstSubView = pview.subspace<pair<0, 0>, pair<5, 7>, pair<12, 12>>();
    static_assert(std::is_same_v<std::decay_t<decltype(fstSubView.lengths())>, Lengths<2>>);
    static_assert(std::is_same_v<std::decay_t<decltype(fstSubView.strides())>, Strides<int, 4>>);
    static_assert(fstSubView.planes() == 2U);
    assert(fstSubView.data() == fpmem.data() + 5 * pview.stride<1U>() + 11 * pview.stride<2U>());
    assert(fstSubView.data<1U>() == spmem.data() + 5 * pview.stride<1U>() + 11 * pview.stride<2U>());

    auto sndSubView = pview.subspace<pair<3, 4>, pair<2, 6>>();
    static_assert(std::is_same_v<std::decay_t<decltype(sndSubView.lengths())>, Lengths<1, 4>>);
    static_assert(std::is_same_v<std::decay_t<decltype(sndSubView.strides())>, Strides<int, 1, 4>>);
    static_assert(sndSubView.planes() == 2U);
    assert(sndSubView.data() == fpmem.data() + 3 * pview.stride<0U>() + 2 * pview.stride<1U>());
    assert(sndSubView.data<1U>() == spmem.data() + 3 * pview.stride<0U>() + 2 * pview.stride<1U>());

    /* Test adding a new axis on the beginning */
    auto axisBeginView = pview.newAxis<0U>();
    static_assert(std::is_same_v<std::decay_t<decltype(axisBeginView.lengths())>, Lengths<1, 4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(axisBeginView.strides())>, Strides<int, 0, 1, 4, 28>>);
    static_assert(axisBeginView.planes() == 2U);
    assert(axisBeginView.data() == fpmem.data());
    assert(axisBeginView.data<1U>() == spmem.data());

    /* Test adding a new axis in the middle */
    auto axisMidView = pview.newAxis<1U>();
    static_assert(std::is_same_v<std::decay_t<decltype(axisMidView.lengths())>, Lengths<4, 1, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(axisMidView.strides())>, Strides<int, 1, 0, 4, 28>>);
    static_assert(axisMidView.planes() == 2U);
    assert(axisMidView.data() == fpmem.data());
    assert(axisMidView.data<1U>() == spmem.data());

    /* Test adding a new axis at the end */
    auto axisEndView = pview.newAxis<3U>();
    static_assert(std::is_same_v<std::decay_t<decltype(axisEndView.lengths())>, Lengths<4, 7, 12, 1>>);
    static_assert(std::is_same_v<std::decay_t<decltype(axisEndView.strides())>, Strides<int, 1, 4, 28, 0>>);
    static_assert(axisEndView.planes() == 2U);
    assert(axisEndView.data() == fpmem.data());
    assert(axisEndView.data<1U>() == spmem.data());

    /* Test squeeze */
    TensorView<TShape<int, 1, 2, 1, 1, 1, 5, 6, 1, 1, 2, 1, 9, 1, 1>, 1> onesView;
    auto squeezed = onesView.squeeze();
    static_assert(std::is_same_v<std::decay_t<decltype(squeezed.lengths())>, Lengths<2, 5, 6, 2, 9>>);
    static_assert(std::is_same_v<std::decay_t<decltype(squeezed.strides())>, Strides<int, 4, 8, 40, 240, 480>>);
    static_assert(squeezed.planes() == 1U);

    /* Allocate views used for testing planar functions */
    TensorView<TShape<int, 4, 7, 12>, 1> singlePlaneView{tpmem.data()};
    TensorView<TShape<int, 4, 7, 12>, 2> dualPlanesView{fpmem.data(), tpmem.data()};
    TensorView<TShape<int, 4, 7, 12>, 3> multiplePlanesView{fpmem.data(), spmem.data(), tpmem.data()};

    /* Test keepPlanes */
    auto kpsView = multiplePlanesView.keepPlanes<0, 2>();
    static_assert(std::is_same_v<std::decay_t<decltype(kpsView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(kpsView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(kpsView.planes() == 2U);
    assert(kpsView.data() == fpmem.data());
    assert(kpsView.data<1>() == tpmem.data());

    /* Test keepPlane */
    auto kpView = multiplePlanesView.keepPlane(1);
    static_assert(std::is_same_v<std::decay_t<decltype(kpView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(kpView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(kpView.planes() == 1U);
    assert(kpView.data() == spmem.data());

    /* Test addPlanes */
    auto apsView = singlePlaneView.addPlanes(spmem.data(), fpmem.data());
    static_assert(std::is_same_v<std::decay_t<decltype(apsView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(apsView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(apsView.planes() == 3U);
    assert(apsView.data() == tpmem.data());
    assert(apsView.data<1>() == spmem.data());
    assert(apsView.data<2>() == fpmem.data());

    /* Test addPlane on the beginning */
    auto apBegView = dualPlanesView.addPlane<0U>(spmem.data());
    static_assert(std::is_same_v<std::decay_t<decltype(apBegView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(apBegView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(apBegView.planes() == 3U);
    assert(apBegView.data() == spmem.data());
    assert(apBegView.data<1>() == fpmem.data());
    assert(apBegView.data<2>() == tpmem.data());

    /* Test addPlane in the middle */
    auto apMidView = dualPlanesView.addPlane<1U>(spmem.data());
    static_assert(std::is_same_v<std::decay_t<decltype(apMidView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(apMidView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(apMidView.planes() == 3U);
    assert(apMidView.data() == fpmem.data());
    assert(apMidView.data<1>() == spmem.data());
    assert(apMidView.data<2>() == tpmem.data());

    /* Test addPlane on the end */
    auto apEndView = dualPlanesView.addPlane<2U>(spmem.data());
    static_assert(std::is_same_v<std::decay_t<decltype(apEndView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(apEndView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(apEndView.planes() == 3U);
    assert(apEndView.data() == fpmem.data());
    assert(apEndView.data<1>() == tpmem.data());
    assert(apEndView.data<2>() == spmem.data());

    /* Test replacePlane */
    auto replaceView = dualPlanesView.replacePlane<1U>(spmem.data());
    static_assert(std::is_same_v<std::decay_t<decltype(replaceView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(replaceView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(replaceView.planes() == 2U);
    assert(replaceView.data() == fpmem.data());
    assert(replaceView.data<1>() == spmem.data());

    /* Test removePlanes by removing first and last planes */
    auto fstRemoveView = multiplePlanesView.removePlanes<0U, 2U>();
    static_assert(std::is_same_v<std::decay_t<decltype(fstRemoveView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(fstRemoveView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(fstRemoveView.planes() == 1U);
    assert(fstRemoveView.data() == spmem.data());

    /* Test removePlanes by removing the middle plane */
    auto sndRemoveView = multiplePlanesView.removePlanes<1U>();
    static_assert(std::is_same_v<std::decay_t<decltype(sndRemoveView.lengths())>, Lengths<4, 7, 12>>);
    static_assert(std::is_same_v<std::decay_t<decltype(sndRemoveView.strides())>, Strides<int, 1, 4, 28>>);
    static_assert(sndRemoveView.planes() == 2U);
    assert(sndRemoveView.data() == fpmem.data());
    assert(sndRemoveView.data<1U>() == tpmem.data());

    /* Test shape function */
    static_assert(
        std::is_same_v<std::decay_t<decltype(pview.shape())>, Shape<Lengths<4, 7, 12>, Strides<int, 1, 4, 28>>>);

    /* Test lengths function */
    static_assert(std::is_same_v<std::decay_t<decltype(pview.lengths())>, Lengths<4, 7, 12>>);

    /* Test length function */
    static_assert(pview.length<0U>() == 4);
    static_assert(pview.length<1U>() == 7);
    static_assert(pview.length<2U>() == 12);

    /* Test strides function */
    static_assert(std::is_same_v<std::decay_t<decltype(pview.strides())>, Strides<int, 1, 4, 28>>);

    /* Test stride function */
    static_assert(pview.stride<0U>() == 1);
    static_assert(pview.stride<1U>() == 4);
    static_assert(pview.stride<2U>() == 28);

    /* Test planes function */
    static_assert(singlePlaneView.planes() == 1U);
    static_assert(dualPlanesView.planes() == 2U);
    static_assert(multiplePlanesView.planes() == 3U);

    /* Test rank function */
    static_assert(pview.rank() == 3U);
    static_assert(onesView.rank() == 14U);
    static_assert(squeezed.rank() == 5U);

    /* Test data functions */
    const int* cpData = pview.data<0U>();
    assert(cpData == fpmem.data());

    int* pData = pview.data<1U>();
    assert(pData == spmem.data());

    const int* rtCPData = pview.data(0U);
    assert(rtCPData == fpmem.data());

    int* rtPData = pview.data(1U);
    assert(rtPData == spmem.data());

    const std::array<int*, 2U> cpViewPtrs = pview.pointers();
    assert(cpViewPtrs[0U] == fpmem.data());
    assert(cpViewPtrs[1U] == spmem.data());

    std::array<int*, 2U> pViewPtrs = pview.pointers();
    assert(pViewPtrs[0U] == fpmem.data());
    assert(pViewPtrs[1U] == spmem.data());
}

int main() {
    testTensorView();

    return 0;
}