#pragma once

#include <cstdint>
#include <limits>

#if defined ENABLE_NT_EXPECTS || defined ENABLE_NT_ENSURES
#include "assert.hpp"
#endif

namespace ntensor {

/*
 * Custom allocator for allocating aligned memory.
 *
 * The initial idea was to implement a base allocator that just provides methods for allocation/deallocation.
 * And then have more advanced allocators such as the Linear allocator, the Free-List allocator, and a hybrid Slab
 * allocator. All of them would receive their base allocators used for allocation/deallocation as template arguments.
 * And they could be either used as standalone allocators, or in case of the Linear allocator and the Free-List
 * allocator, they could be used internally in the Slab allocator. This way, the Slab allocator would allocate slabs
 * using the Linear allocator or the Free-List allocator. So each time a new allocation is requested, the Slab allocator
 * would first check if any of its slabs has enough unused memory that can be used, instead of performing a new
 * allocation. And only if it fails would it allocate a new slab with the requested size. So it would be a hybrid
 * allocator that has both properties of a Slab allocator and a Linear allocator/Free-List allocator. However,
 * implementing those allocators, assuring that they work, and writing STL adapters for those would take too much time
 * at this moment. So I left them out for a future release instead.
 *
 * Parameters:
 * @tparam T: memory type that needs to be allocated
 */
template <typename T>
class AlignedMallocAllocator {
 public:
  using value_type = T;

  /*
   * Default constructor
   */
  AlignedMallocAllocator() noexcept = default;

  /*
   * Construct the allocator using a allocator with a different type
   * Since the allocator is stateless, the constructor has no visible effect
   */
  template <typename U>
  AlignedMallocAllocator(const AlignedMallocAllocator<U>&) noexcept {}

  /*
   * Allocates n * sizeof(T) bytes of uninitialized storage
   * Parameters:
   * @param n: the number of objects to allocate storage for
   */
  [[nodiscard]] value_type* allocate(std::size_t n) {
    return static_cast<value_type*>(
        ::operator new(n * sizeof(value_type), static_cast<std::align_val_t>(NT_ALIGNMENT)));
  }

  /*
   * Deallocates the storage referenced by the pointer p
   * Parameters:
   * @param p: pointer pointing to the allocated memory
   * @param n: the number of objects that the storage was allocated for
   */
  void deallocate(value_type* p, std::size_t n) noexcept {
    ::operator delete(p, n, static_cast<std::align_val_t>(NT_ALIGNMENT));
  }
};

/*
 * Compares two allocators. Since the allocators are stateless, two allocators are always equal
 */
template <class T, class U>
constexpr bool operator==(const AlignedMallocAllocator<T>&, const AlignedMallocAllocator<U>&) noexcept {
  return true;
}

/*
 * Compares two allocators. Since the allocators are stateless, two allocators are always equal
 */
template <class T, class U>
constexpr bool operator!=(const AlignedMallocAllocator<T>&, const AlignedMallocAllocator<U>&) noexcept {
  return false;
}

template <typename T>
using nt_allocator = AlignedMallocAllocator<T>;

}  // namespace ntensor
