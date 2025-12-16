#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace flashmask {

#define HOST_DEVICE __forceinline__ __host__ __device__

template <typename value_t>
HOST_DEVICE
constexpr
value_t clz(value_t x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return value_t(31 - i);
  }
  return value_t(32);
}

template <typename value_t>
HOST_DEVICE
constexpr
value_t find_log2(value_t x) {
  int a = int(31 - clz(x));
  a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
  return a;
}

// extracted from cutlass, use only some of the main functionalities
struct FastDivmod {
  using value_div_type = int;
  int32_t divisor = 1;
  uint32_t multiplier = 0u;
  uint32_t shift_right = 0u;

  // Find quotient and remainder using device-side intrinsics
  HOST_DEVICE
  void fast_divmod(int& quotient, int& remainder, int dividend) const {

#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if divisor != 1, else simply copy the source.
    quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
    quotient = int((divisor != 1) ? int(((int64_t)dividend * multiplier) >> 32) >> shift_right : dividend);
#endif

    // The remainder.
    remainder = dividend - (quotient * divisor);
  }

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.

  constexpr FastDivmod() = default;

  HOST_DEVICE
  FastDivmod(int divisor_): divisor(divisor_) {
    assert(divisor_ >= 0);
    if (divisor != 1) {
      unsigned int p = 31 + find_log2(divisor);
      unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

      multiplier = m;
      shift_right = p - 32;
    }
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  ///
  /// Simply returns the quotient
  HOST_DEVICE
  int divmod(int &remainder, int dividend) const {
    int quotient;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }
};


}   // namespace flashmask