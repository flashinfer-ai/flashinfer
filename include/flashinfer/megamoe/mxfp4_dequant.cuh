// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1 + E8M0) -> FP8 (E4M3) dequant for SM90 fused MegaMoE.
//
// E8M0 is a pure power of two, so scaling by 2^k is an exponent adjustment:
// adding k to an E4M3 value's 4-bit exponent field is adding (k << 3) to the
// byte, and the whole magnitude table for a scale group fits in two registers.
//
// Folding that shift arithmetically per group is nonetheless the wrong trade on
// SM90 -- the saturation/flush edge cases need byte-wise SIMD intrinsics Hopper
// does not implement. make_scaled_lut() below remains the definition of the
// numerics, but the mainloop reads the tables from a shared-memory copy built
// from it once per CTA; see kScaledLutSize.

#pragma once

#include <cstdint>
#include <flashinfer/megamoe/detail/fp4_decode.cuh>

namespace flashinfer::megamoe {
namespace mxfp4 {

#define FI_MEGAMOE_MXFP4_INLINE __device__ __forceinline__

// __byte_perm without the compiler's range check on the selector; the selectors
// below are masked to 0x7 lanes by construction.
FI_MEGAMOE_MXFP4_INLINE std::uint32_t byte_perm_unchecked(std::uint32_t a, std::uint32_t b,
                                                          std::uint32_t selector) {
  return detail::byte_perm_unchecked(a, b, selector);
}

// E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} encoded as E4M3 at scale 2^0.
// Byte i of (LUT_X, LUT_Y) holds magnitude index i and i+4 respectively, the
// layout __byte_perm consumes.
static constexpr uint32_t kBaseLutX = 0x3c383000u;
static constexpr uint32_t kBaseLutY = 0x4c484440u;

// Largest finite E4M3 (448.0). Saturating adds clamp to 0xff, so results are
// pulled back to this.
static constexpr uint32_t kE4M3MaxBytes = 0x7e7e7e7eu;
// Exponent field 0 encodes zero/subnormal; E4M3 subnormals are not produced by
// this path, so any byte below 0x08 must flush to zero.
static constexpr uint32_t kMinNormalBytes = 0x08080808u;

// The 8-byte magnitude table for one E8M0 scale group, built in registers.
struct ScaledLut {
  std::uint32_t x;
  std::uint32_t y;
};

// Fold scale 2^(code - 127) into the base table -- branchless.
//
// k's sign and magnitude both come from the weight group's own scale byte, so
// they vary per element and are independent across the 32 lanes of a warp
// (each lane dequantizes a different B-tile row). An `if (k >= 0)` or
// `if (in fast range)` here is therefore not a rare/uniform branch: whenever
// two lanes in the same warp fall on different sides, the SM serializes them,
// and that scalar branch cost was found (via SASS inspection -- @!P0 BRA in
// the compiled make_scaled_lut) to dominate the two saturating vadds it was
// meant to save. So both signs are computed unconditionally and blended with
// an arithmetic mask (SEL/PRMT), never a control-flow branch.
//
// k >= 0: saturating byte add, then clamp to 448. Byte 0 of x is magnitude 0
//         and must stay zero, so it is masked off -- adding to it would
//         manufacture a nonzero value.
// k <  0: saturating byte subtract, which already floors at zero, then flush
//         any byte whose exponent field reached 0. That case is reachable with
//         real weights: a group whose max is below 6 * 2^-5 gets k < -5, and
//         its smallest magnitude underflows E4M3's normal range.
//
// |k| * 8 is clamped to 255 before broadcasting; beyond that every byte
// saturates anyway, and an unclamped broadcast would corrupt neighbouring
// bytes.
FI_MEGAMOE_MXFP4_INLINE ScaledLut make_scaled_lut(std::uint32_t scale_ue8m0) {
  const int k = static_cast<int>(scale_ue8m0 & 0xffu) - 127;
  const std::uint32_t magnitude = static_cast<std::uint32_t>(k >= 0 ? k : -k) * 8u;
  const std::uint32_t delta = (magnitude > 255u ? 255u : magnitude) * 0x01010101u;

  const std::uint32_t pos_x = __vminu4(__vaddus4(kBaseLutX, delta), kE4M3MaxBytes) & 0xffffff00u;
  const std::uint32_t pos_y = __vminu4(__vaddus4(kBaseLutY, delta), kE4M3MaxBytes);

  const std::uint32_t neg_x_raw = __vsubus4(kBaseLutX, delta);
  const std::uint32_t neg_y_raw = __vsubus4(kBaseLutY, delta);
  const std::uint32_t neg_x = neg_x_raw & __vcmpgeu4(neg_x_raw, kMinNormalBytes);
  const std::uint32_t neg_y = neg_y_raw & __vcmpgeu4(neg_y_raw, kMinNormalBytes);

  // All-1s if k < 0, all-0s otherwise -- an arithmetic mask, not a predicate
  // on control flow, so ptxas has nothing to branch on.
  const std::uint32_t neg_mask = static_cast<std::uint32_t>(-(k < 0));
  ScaledLut lut;
  lut.x = (pos_x & ~neg_mask) | (neg_x & neg_mask);
  lut.y = (pos_y & ~neg_mask) | (neg_y & neg_mask);
  return lut;
}

// Shared-memory window over the scaled tables.
//
// make_scaled_lut() is exact but expensive on SM90: its byte-wise saturating
// intrinsics (__vaddus4 / __vsubus4 / __vminu4 / __vcmpgeu4) have no native
// instruction on Hopper, so ptxas emulates them -- a single call compiles to
// 88 SASS instructions (39 LOP3 + 8 PRMT + ...) versus 40 for NVFP4's whole
// table-load-plus-decode. Calling it once per 32-element scale group, i.e.
// four times per BK128 weight row per thread, dominated the mainloop.
//
// The table is constant outside a narrow band of E8M0 codes: sweeping all 256
// codes yields only 20 distinct tables, at codes 118..137 (k = -9..+10). Below
// that every magnitude underflows E4M3's normal range and flushes to zero,
// above it every magnitude saturates at 448. That makes a clamped 32-entry
// window bit-exact, and it is what this used to be -- but the clamp is not
// free: it costs two VIMNMX and a subtract per lookup, twelve instructions per
// decode iteration across four slices. Indexing all 256 codes directly costs
// 2 KB of shared memory and none of that arithmetic, so the table is stored
// whole. Still half as many lookups per row as NVFP4, which groups 16 elements
// where MXFP4 groups 32.
static constexpr std::uint32_t kScaledLutSize = 256;

// A decode tile is `kBTileRows` fused rows stored 16-byte-chunk-major: the
// 16-byte chunk `c` of row `r` sits at `c * kBTileRows * 16 + r * 16`, and the
// row's four E8M0 bytes at `kBTileRows * 64 + r * 4`. Laying the chunks out
// this way is what lets the 12 bytes of per-row padding go -- 15 % of the
// weight stream -- while keeping every 16-byte decoder load aligned, which a
// flat 68-byte row would not.
static constexpr std::uint32_t kBTileRows = 128;
static constexpr std::uint32_t kBChunkStride = kBTileRows * 16;
static constexpr std::uint32_t kBTileFP4Bytes = kBTileRows * 64;
static constexpr std::uint32_t kBTileBytes = kBTileFP4Bytes + kBTileRows * 4;

struct DecodeTileRow {
  const std::uint8_t* chunk;  // chunk c at `chunk + c * kBChunkStride`
  const std::uint8_t* scale;
};

FI_MEGAMOE_MXFP4_INLINE DecodeTileRow decode_tile_row(const std::uint8_t* __restrict__ packed_b,
                                                      const std::uint32_t row) {
  const std::uint8_t* tile = packed_b + (row / kBTileRows) * kBTileBytes;
  const std::uint32_t sub = row % kBTileRows;
  return {tile + sub * 16, tile + kBTileFP4Bytes + sub * 4};
}

// Fill the table once during initialization, striding so that a block with
// fewer than kScaledLutSize threads still fills all of it. The entries are
// built by the same make_scaled_lut() the decoders used to call inline, so the
// numerics are unchanged by construction.
FI_MEGAMOE_MXFP4_INLINE void init_scaled_lut(ScaledLut* __restrict__ smem_lut,
                                             const std::uint32_t thread_idx,
                                             const std::uint32_t num_threads) {
  for (std::uint32_t code = thread_idx; code < kScaledLutSize; code += num_threads)
    smem_lut[code] = make_scaled_lut(code);
}

FI_MEGAMOE_MXFP4_INLINE ScaledLut load_scaled_lut(const ScaledLut* __restrict__ smem_lut,
                                                  const std::uint32_t scale_ue8m0) {
  return smem_lut[scale_ue8m0 & 0xffu];
}

// Decode eight packed FP4 nibbles (one uint32) into eight FP8 bytes, using a
// table already scaled by make_scaled_lut.
FI_MEGAMOE_MXFP4_INLINE uint2 dequant_word(std::uint32_t packed, const ScaledLut& lut) {
  return detail::decode_fp4_word(packed, lut);
}

#undef FI_MEGAMOE_MXFP4_INLINE

}  // namespace mxfp4
}  // namespace flashinfer::megamoe
