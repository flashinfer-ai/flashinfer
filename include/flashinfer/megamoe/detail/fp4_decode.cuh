// Shared E2M1 nibble/sign decoding and register-source fragment assembly.
#pragma once
#include <cstdint>

namespace flashinfer::megamoe::detail {

__device__ __forceinline__ uint32_t byte_perm_unchecked(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
  return out;
}

template <typename Lut>
__device__ __forceinline__ uint2 decode_fp4_word(uint32_t packed, const Lut& lut) {
  const uint32_t selectors = packed & 0x77777777u;
  uint32_t hi = byte_perm_unchecked(lut.x, lut.y, selectors);
  uint32_t lo = byte_perm_unchecked(lut.x, lut.y, selectors >> 16);
  asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(hi) : "r"(packed));
  const uint32_t shifted = packed << 4;
  asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(lo) : "r"(shifted));
  return make_uint2(hi, lo);
}

__device__ __forceinline__ void assemble_rs_word_pair(uint2 lo, uint2 hi, bool keep_hi,
                                                      uint32_t (&frag)[4]) {
  const uint32_t kept_lo = keep_hi ? lo.x : lo.y;
  const uint32_t sent_lo = keep_hi ? lo.y : lo.x;
  const uint32_t recv_lo = __shfl_xor_sync(0xffffffffu, sent_lo, 1);
  frag[0] = keep_hi ? kept_lo : recv_lo;
  frag[1] = keep_hi ? recv_lo : kept_lo;
  const uint32_t kept_hi = keep_hi ? hi.x : hi.y;
  const uint32_t sent_hi = keep_hi ? hi.y : hi.x;
  const uint32_t recv_hi = __shfl_xor_sync(0xffffffffu, sent_hi, 1);
  frag[2] = keep_hi ? kept_hi : recv_hi;
  frag[3] = keep_hi ? recv_hi : kept_hi;
}

}  // namespace flashinfer::megamoe::detail
