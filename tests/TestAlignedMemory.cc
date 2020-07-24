#include "AlignedMemory.h"

#include <assert.h>


using namespace AboveInfinity;

void testAlignedMemory() {
    /* Test default constructor */
    AlignedMemory<float> dmem;
    assert(dmem.size() == 0U);
    assert(dmem.alignment() == 0U);

    /* Test custom constructor without setting the allocated memory */
    AlignedMemory<int> mem{100U, 16U};
    assert(mem.size() == 100U);
    assert(mem.alignment() == 16U);

    /* Test custom constructor with setting the allocated memory to 1 */
    AlignedMemory<int> inMem{100U, 16U, 1};
    assert(inMem.size() == 100U);
    assert(inMem.alignment() == 16U);
    for(auto i = 0U; i < 100U; ++i) assert(*(inMem.data() + i) == 1);

    /* Test move assignment */
    AlignedMemory<int> moveMem = std::move(inMem);
    assert(moveMem.size() == 100U);
    assert(moveMem.alignment() == 16U);
    for(auto i = 0U; i < 100U; ++i) assert(*(moveMem.data() + i) == 1);
    assert(inMem.data() == nullptr);
}

int main() {
    testAlignedMemory();

    return 0;
}