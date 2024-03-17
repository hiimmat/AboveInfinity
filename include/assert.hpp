#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define NT_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define NT_LIKELY(x) (!!(x))
#endif

#if defined(NT_THROW_ON_CONTRACT_VIOLATION)
#include <stdexcept>

namespace ntensor {
struct fail_fast : public std::logic_error {
  explicit fail_fast(char const* const message) : std::logic_error(message) {}
};

template <typename Exception>
[[noreturn]] void throw_exception(Exception&& exception) {
  throw std::forward<Exception>(exception);
}

}  // namespace ntensor

#define NT_STRINGIFY_DETAIL(x) #x
#define NT_STRINGIFY(x) NT_STRINGIFY_DETAIL(x)

#define NT_CONTRACT_CHECK(type, cond)          \
  (NT_LIKELY(cond) ? static_cast<void>(0)      \
                   : ntensor::throw_exception( \
                         ntensor::fail_fast("NT: " type " failure at " __FILE__ ": " NT_STRINGIFY(__LINE__))))
#else
#define NT_CONTRACT_CHECK(type, cond) (NT_LIKELY(cond) ? static_cast<void>(0) : std::terminate())
#endif

#define Expects(cond) NT_CONTRACT_CHECK("Precondition", cond)
#define Ensures(cond) NT_CONTRACT_CHECK("Postcondition", cond)
