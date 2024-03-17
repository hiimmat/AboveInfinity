#include <aligned_allocator.hpp>
#include <catch2/catch_test_macros.hpp>

namespace nt = ntensor;

TEST_CASE("AlignedMallocAllocator class tests") {
  SECTION("allocation and deallocation") {
    nt::AlignedMallocAllocator<int> allocator{};
    int* p = allocator.allocate(10);
    REQUIRE(p);
    CHECK(!(reinterpret_cast<uintptr_t>(p) % NT_ALIGNMENT));
    allocator.deallocate(p, 10);
  }

  SECTION("equality operator") {
    CHECK(nt::AlignedMallocAllocator<int>{} == nt::AlignedMallocAllocator<char>{});
    CHECK(nt::AlignedMallocAllocator<int>{} == nt::AlignedMallocAllocator<float>{});
    CHECK(nt::AlignedMallocAllocator<int>{} == nt::AlignedMallocAllocator<double>{});
    CHECK(nt::AlignedMallocAllocator<char>{} == nt::AlignedMallocAllocator<float>{});
    CHECK(nt::AlignedMallocAllocator<char>{} == nt::AlignedMallocAllocator<double>{});
    CHECK(nt::AlignedMallocAllocator<float>{} == nt::AlignedMallocAllocator<double>{});
  }

  SECTION("inequality operator") {
    CHECK(!(nt::AlignedMallocAllocator<int>{} != nt::AlignedMallocAllocator<char>{}));
    CHECK(!(nt::AlignedMallocAllocator<int>{} != nt::AlignedMallocAllocator<float>{}));
    CHECK(!(nt::AlignedMallocAllocator<int>{} != nt::AlignedMallocAllocator<double>{}));
    CHECK(!(nt::AlignedMallocAllocator<char>{} != nt::AlignedMallocAllocator<float>{}));
    CHECK(!(nt::AlignedMallocAllocator<char>{} != nt::AlignedMallocAllocator<double>{}));
    CHECK(!(nt::AlignedMallocAllocator<float>{} != nt::AlignedMallocAllocator<double>{}));
  }
}
