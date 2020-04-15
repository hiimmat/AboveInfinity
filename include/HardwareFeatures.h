#pragma once

#include <cstdio>

#if defined __has_include
#if __has_include("sys_info.h")
#include "SysInfo.h"
#endif
#endif

namespace AboveInfinity {

/* Hardware specific features */
inline namespace hwFeatures {

/* Alignment requirements - specific for the instruction set */
#if defined(__MMX__)
static constexpr const std::size_t AIAlignment = 8U;
#elif defined(__AVX__)
static constexpr const std::size_t AIAlignment = 32U;
#elif defined(__AVX512F__)
static constexpr const std::size_t AIAlignment = 64U;
#else
/* AltiVec, NEON, SSE */
static constexpr const std::size_t AIAlignment = 16U;
#endif

/* Number of elements that can be executed simultaneously - based on experience */
template<typename T>
constexpr const inline std::size_t SimultaneousVecOps() noexcept {
    return AIAlignment / sizeof(T);
}

#ifdef _MAXSTACKALLOCSIZE
static constexpr const std::size_t maxStackAllocSize = _MAXSTACKALLOCSIZE;
#else
static constexpr const std::size_t maxStackAllocSize = 1024U;
#endif

}; // namespace hwFeatures

}; // namespace AboveInfinity
