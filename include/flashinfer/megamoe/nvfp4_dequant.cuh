// SPDX-License-Identifier: MIT
//
// NVFP4 (E2M1 + UE4M3 scale) -> FP8 (E4M3) dequant helper for SM90
// fused MegaMoE. The packed FP4 layout matches the existing Marlin-style
// byte packing used by the predecessor FP4 path, with NVFP4 UE4M3
// scale bytes applied per 16 K elements.

#pragma once

#include <cuda_fp8.h>

#include <cstdint>
#include <flashinfer/megamoe/detail/fp4_decode.cuh>

namespace flashinfer::megamoe {
namespace nvfp4 {

#define FI_MEGAMOE_NVFP4_INLINE __device__ __forceinline__

// Per-scale lookup table for positive E2M1 magnitudes 0..7 converted to FP8.
// Each row packs magnitudes 0..3 in .x and 4..7 in .y so __byte_perm can map
// four FP4 nibbles to four FP8 bytes with one instruction. Scale code 0x7f is
// treated as a saturating 448.0 fallback; the quantizer emits only 0x00..0x7e.
static __device__ __constant__ __align__(16) const uint2 kE2M1AndUe4m3ToFp8Lut[128] = {
    {0x00000000u, 0x00000000u}, {0x02010000u, 0x06040302u}, {0x03020100u, 0x0c080604u},
    {0x04030200u, 0x110c0906u}, {0x06040200u, 0x14100c08u}, {0x08050200u, 0x17120f0au},
    {0x09060300u, 0x1914110cu}, {0x0a070400u, 0x1a16120eu}, {0x0c080400u, 0x1c181410u},
    {0x0e090400u, 0x1e191611u}, {0x0f0a0500u, 0x1f1a1712u}, {0x100b0600u, 0x201b1813u},
    {0x110c0600u, 0x211c1914u}, {0x120d0600u, 0x221d1a15u}, {0x120e0700u, 0x221e1a16u},
    {0x130f0800u, 0x231f1b17u}, {0x14100800u, 0x24201c18u}, {0x16110900u, 0x26211e19u},
    {0x17120a00u, 0x27221f1au}, {0x18130b00u, 0x2823201bu}, {0x19140c00u, 0x2924211cu},
    {0x1a150d00u, 0x2a25221du}, {0x1a160e00u, 0x2a26221eu}, {0x1b170f00u, 0x2b27231fu},
    {0x1c181000u, 0x2c282420u}, {0x1e191100u, 0x2e292621u}, {0x1f1a1200u, 0x2f2a2722u},
    {0x201b1300u, 0x302b2823u}, {0x211c1400u, 0x312c2924u}, {0x221d1500u, 0x322d2a25u},
    {0x221e1600u, 0x322e2a26u}, {0x231f1700u, 0x332f2b27u}, {0x24201800u, 0x34302c28u},
    {0x26211900u, 0x36312e29u}, {0x27221a00u, 0x37322f2au}, {0x28231b00u, 0x3833302bu},
    {0x29241c00u, 0x3934312cu}, {0x2a251d00u, 0x3a35322du}, {0x2a261e00u, 0x3a36322eu},
    {0x2b271f00u, 0x3b37332fu}, {0x2c282000u, 0x3c383430u}, {0x2e292100u, 0x3e393631u},
    {0x2f2a2200u, 0x3f3a3732u}, {0x302b2300u, 0x403b3833u}, {0x312c2400u, 0x413c3934u},
    {0x322d2500u, 0x423d3a35u}, {0x322e2600u, 0x423e3a36u}, {0x332f2700u, 0x433f3b37u},
    {0x34302800u, 0x44403c38u}, {0x36312900u, 0x46413e39u}, {0x37322a00u, 0x47423f3au},
    {0x38332b00u, 0x4843403bu}, {0x39342c00u, 0x4944413cu}, {0x3a352d00u, 0x4a45423du},
    {0x3a362e00u, 0x4a46423eu}, {0x3b372f00u, 0x4b47433fu}, {0x3c383000u, 0x4c484440u},
    {0x3e393100u, 0x4e494641u}, {0x3f3a3200u, 0x4f4a4742u}, {0x403b3300u, 0x504b4843u},
    {0x413c3400u, 0x514c4944u}, {0x423d3500u, 0x524d4a45u}, {0x423e3600u, 0x524e4a46u},
    {0x433f3700u, 0x534f4b47u}, {0x44403800u, 0x54504c48u}, {0x46413900u, 0x56514e49u},
    {0x47423a00u, 0x57524f4au}, {0x48433b00u, 0x5853504bu}, {0x49443c00u, 0x5954514cu},
    {0x4a453d00u, 0x5a55524du}, {0x4a463e00u, 0x5a56524eu}, {0x4b473f00u, 0x5b57534fu},
    {0x4c484000u, 0x5c585450u}, {0x4e494100u, 0x5e595651u}, {0x4f4a4200u, 0x5f5a5752u},
    {0x504b4300u, 0x605b5853u}, {0x514c4400u, 0x615c5954u}, {0x524d4500u, 0x625d5a55u},
    {0x524e4600u, 0x625e5a56u}, {0x534f4700u, 0x635f5b57u}, {0x54504800u, 0x64605c58u},
    {0x56514900u, 0x66615e59u}, {0x57524a00u, 0x67625f5au}, {0x58534b00u, 0x6863605bu},
    {0x59544c00u, 0x6964615cu}, {0x5a554d00u, 0x6a65625du}, {0x5a564e00u, 0x6a66625eu},
    {0x5b574f00u, 0x6b67635fu}, {0x5c585000u, 0x6c686460u}, {0x5e595100u, 0x6e696661u},
    {0x5f5a5200u, 0x6f6a6762u}, {0x605b5300u, 0x706b6863u}, {0x615c5400u, 0x716c6964u},
    {0x625d5500u, 0x726d6a65u}, {0x625e5600u, 0x726e6a66u}, {0x635f5700u, 0x736f6b67u},
    {0x64605800u, 0x74706c68u}, {0x66615900u, 0x76716e69u}, {0x67625a00u, 0x77726f6au},
    {0x68635b00u, 0x7873706bu}, {0x69645c00u, 0x7974716cu}, {0x6a655d00u, 0x7a75726du},
    {0x6a665e00u, 0x7a76726eu}, {0x6b675f00u, 0x7b77736fu}, {0x6c686000u, 0x7c787470u},
    {0x6e696100u, 0x7e797671u}, {0x6f6a6200u, 0x7f7a7772u}, {0x706b6300u, 0x7f7b7873u},
    {0x716c6400u, 0x7f7c7974u}, {0x726d6500u, 0x7f7d7a75u}, {0x726e6600u, 0x7f7e7a76u},
    {0x736f6700u, 0x7f7f7b77u}, {0x74706800u, 0x7f7f7c78u}, {0x76716900u, 0x7f7f7e79u},
    {0x77726a00u, 0x7f7f7f7au}, {0x78736b00u, 0x7f7f7f7bu}, {0x79746c00u, 0x7f7f7f7cu},
    {0x7a756d00u, 0x7f7f7f7du}, {0x7a766e00u, 0x7f7f7f7eu}, {0x7b776f00u, 0x7f7f7f7fu},
    {0x7c787000u, 0x7f7f7f7fu}, {0x7e797100u, 0x7f7f7f7fu}, {0x7f7a7200u, 0x7f7f7f7fu},
    {0x7f7b7300u, 0x7f7f7f7fu}, {0x7f7c7400u, 0x7f7f7f7fu}, {0x7f7d7500u, 0x7f7f7f7fu},
    {0x7f7e7600u, 0x7f7f7f7fu}, {0x7f7e7600u, 0x7f7f7f7fu}};

FI_MEGAMOE_NVFP4_INLINE uint2 load_e2m1_ue4m3_lut(std::uint32_t scale_ue4m3) {
  return kE2M1AndUe4m3ToFp8Lut[scale_ue4m3 & 0x7fu];
}

template <bool kUseDp4a>
FI_MEGAMOE_NVFP4_INLINE std::uint32_t pack_nvfp4_magnitude_selector(std::uint32_t byte_magnitudes) {
  if constexpr (kUseDp4a) {
    const std::uint32_t lo = __dp4a(byte_magnitudes, 0x00001001u, 0u);
    const std::uint32_t hi = __dp4a(byte_magnitudes, 0x10010000u, 0u);
    return lo + (hi << 8);
  } else {
    const std::uint32_t packed_pairs = byte_magnitudes + (byte_magnitudes >> 4);
    return __byte_perm(packed_pairs, 0u, 0x4420u);
  }
}

FI_MEGAMOE_NVFP4_INLINE std::uint32_t byte_perm_unchecked(std::uint32_t a, std::uint32_t b,
                                                          std::uint32_t selector) {
  return detail::byte_perm_unchecked(a, b, selector);
}

template <bool kUseDp4aHi = false, bool kUseDp4aLo = kUseDp4aHi>
FI_MEGAMOE_NVFP4_INLINE uint2 dequant_nvfp4_to_fp8_pair_with_lut(std::uint32_t uq,
                                                                 const uint2& lut) {
  const std::uint32_t sel_hi = pack_nvfp4_magnitude_selector<kUseDp4aHi>((uq >> 4) & 0x07070707u);
  const std::uint32_t sel_lo = pack_nvfp4_magnitude_selector<kUseDp4aLo>(uq & 0x07070707u);

  std::uint32_t out_hi = byte_perm_unchecked(lut.x, lut.y, sel_hi);
  std::uint32_t out_lo = byte_perm_unchecked(lut.x, lut.y, sel_lo);
  out_hi |= uq & 0x80808080u;
  out_lo |= (uq << 4) & 0x80808080u;
  return make_uint2(out_hi, out_lo);
}

#undef FI_MEGAMOE_NVFP4_INLINE

}  // namespace nvfp4
}  // namespace flashinfer::megamoe
