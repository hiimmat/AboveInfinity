#pragma once

namespace AboveInfinity {
inline namespace Utilities {

/*
 * Since there are a lot of static asserts in this code that some might prefer to skip, I thought that
 * it might be a good idea to enable them depending on a flag being passed during compilatiton.
 * The reason for not having them constantly enabled is that they slow down compilation, and they
 * might not be needed in every build type. I'd rather leave the users decide if they want to use them
 * or not. At the same time, a lot of this code is templated, so these asserts help us to confirm our
 * assumptions.
 */
#define ENABLE_REQUIREMENTS

#ifdef ENABLE_REQUIREMENTS
    #define requires static_assert
#else
    #define requires(...) static_cast<void>(0)
#endif

} // namespace Utilities
} // namespace AboveInfinity
