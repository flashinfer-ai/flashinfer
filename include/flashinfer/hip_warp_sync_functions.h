// ported from <hip/amd_detail/amd_warp_sync_functions.h> in SDK 6.2
#ifndef FLASHINFER_HIP_WARP_SYNC_FUNCTIONS_PORTED_H_
#define FLASHINFER_HIP_WARP_SYNC_FUNCTIONS_PORTED_H_

#include <hip/hip_runtime.h>

// note in SDK we have this statement device_prop.warpSize
#ifndef __warpSize
#define __warpSize 64
#endif

// compiling for 64 bit, ignoring upper 32 bit
#define __hip_adjust_mask_for_wave32(MASK)            \
  do {                                          \
    if (__warpSize == 32) MASK &= 0xFFFFFFFF;     \
  } while (0)

#if defined(NDEBUG)
#define __hip_assert(COND)
#else
#define __hip_assert(COND)                          \
  do {                                              \
    if (!(COND))                                    \
      __builtin_trap();                             \
  } while (0)
#endif

template <typename T>
__device__ inline
T __hip_readfirstlane(T val) {
  // In theory, behaviour is undefined when reading from a union member other
  // than the member that was last assigned to, but it works in practice because
  // we rely on the compiler to do the reasonable thing.
  union {
    unsigned long long l;
    T d;
  } u;
  u.d = val;
  // NOTE: The builtin returns int, so we first cast it to unsigned int and only
  // then extend it to 64 bits.
  unsigned long long lower = (unsigned)__builtin_amdgcn_readfirstlane(u.l);
  unsigned long long upper =
      (unsigned)__builtin_amdgcn_readfirstlane(u.l >> 32);
  u.l = (upper << 32) | lower;
  return u.d;
}

#define __hip_check_mask(MASK)                                                 \
  do {                                                                         \
    __hip_assert(MASK && "mask must be non-zero");                             \
    bool done = false;                                                         \
    while (__any(!done)) {                                                     \
      if (!done) {                                                             \
        auto chosen_mask = __hip_readfirstlane(MASK);                          \
        if (MASK == chosen_mask) {                                             \
          __hip_assert(MASK == __ballot(true) &&                               \
                       "all threads specified in the mask"                     \
                       " must execute the same operation with the same mask"); \
          done = true;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while(0)

template <typename MaskT = long, typename T>
__device__ inline
T __shfl_xor_sync(MaskT mask, T var, int laneMask,
                                    int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_xor(var, laneMask, width);
}

// used by libhipcxx
template <typename MaskT, typename T>
__device__ inline
T __shfl_sync(MaskT mask, T var, int srcLane,
              int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_up_sync(MaskT mask, T var, unsigned int delta,
                                   int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_up(var, delta, width);
}

#endif