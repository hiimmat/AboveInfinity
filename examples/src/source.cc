#include <AboveInfinity/TensorOps.h>

using namespace AboveInfinity;

int main() {
    Tensor<Shape<float, Lengths<20, 3, 2>>, 2> c;
    auto cV = c.view();
    auto cX = c.view();
    auto cP = cV.fastPermute<2, 1, 0>();
    
    Tensor<Shape<char, Lengths<2, 7, 4, 1, 3, 6, 8, 1, 5, 2>>, 2> permExample;
    auto permViewEx = perm.view();
    auto permutedView = permViewEx.fastPermute<9, 2, 8, 5, 7, 1, 4, 3, 6, 0>();
    auto unpermuted = permutedView.undoPermutation();

    auto cs = cP.subspace<pair<0, 0>, pair<1, 2>, pair<1, 3>>();
    constexpr Shape<float, Lengths<1, 2, 3, 1, 4, 6, 1, 1, 1, 3, 1, 7, 2, 1>> s;
    constexpr auto ss = s.squeeze();
    constexpr auto sN = s.newAxis<3>();
    constexpr auto sSq = sN.squeeze();

    auto cSlice = cV.slice<1, 1>();
    auto cSPData = cP.slicingPointer<1, 1, 0, 19>();
    auto cSlab = cV.slab<0, 3, 7>();
    auto cN = cV.newAxis<1U>();
    auto cSq = cN.squeeze();

    auto cPFlattened = flatten<false>(cP);
    auto cR = ravel<true>(cP);

    Tensor<Shape<float, Lengths<6, 2, 8>>, 1> reshapeEx;
    auto reshapeViewEx = reshapeEx.view();
    auto reshapedView = reshape<Lengths<1, 2, 3, 2, 2, 1, 4>>(reshapeViewEx);

    execute(
        [](float& x) {
            static float i = 1;
            x = i++;
        },
        cV);

    execute(
        [](float& x) {
            static float i = 1;
            x = i++;
        },
        cV,
        cP);

    saveAsText(cV, "cView.txt");

    Tensor<Shape<float, Lengths<2000, 3, 2>>, 2> d;
    Tensor<Shape<float, Lengths<2000, 3, 2>>, 2> e;
    auto cD = d.view();
    auto cE = e.view();

    execute(
        [](float& x) {
            static float i = 1;
            x = i++;
        },
        cD);

    copy(cD, cE);

    return 0;
}
