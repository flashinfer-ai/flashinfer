// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../model/dsv4_nvfp4_layout.cuh"
#include "../../compute/nvfp4_vt_layout.cuh"
#include "../../compute/tile_traits.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

struct Dsv4Nvfp4Sync {
  static constexpr int Q_STAGE = 2;
  static constexpr int MATH = 3;
  static constexpr int GATHER = 4;
  // Gather and Vt preparation use ID 4 in disjoint producer phases.
  static constexpr int VT_PRODUCER = GATHER;
};

struct Dsv4Nvfp4OperandResources {
  static constexpr int KV_SMEM_STRIDE = Dsv4Nvfp4Layout::PACKED_NOPE_BYTES + 16;
  static constexpr int Q_PACKED_STRIDE = Dsv4Nvfp4Layout::PACKED_NOPE_BYTES + 16;
  static constexpr int Q_SCALE_STRIDE = Dsv4Nvfp4Layout::SCALE_BYTES_PER_TOKEN;
};

constexpr int DSV4_NVFP4_NUM_SCALES = Dsv4Nvfp4Layout::NUM_SCALES;
constexpr int DSV4_NVFP4_Q_PACKED_STRIDE = Dsv4Nvfp4OperandResources::Q_PACKED_STRIDE;
constexpr int DSV4_NVFP4_SCALE_STRIDE = Dsv4Nvfp4OperandResources::Q_SCALE_STRIDE;

constexpr int DECODE_N_WARPS = 8;
constexpr int DECODE_IO_WARPS = 2;
constexpr int DECODE_BLOCK_THREADS = (DECODE_N_WARPS + DECODE_IO_WARPS) * 32;
constexpr int DECODE_MATH_THREADS = DECODE_N_WARPS * 32;
constexpr int DECODE_MERGE2_THREADS = 512;
constexpr int DECODE_CAND_WINDOW = NVFP4_VT_CANDIDATES;
constexpr int DECODE_KV_BUF_COUNT = 2;
constexpr int DECODE_ENTRIES_PER_WARP = DECODE_CAND_WINDOW / DECODE_N_WARPS;
constexpr int DECODE_QK_N_TILES = DECODE_ENTRIES_PER_WARP / 8;
constexpr int DECODE_PACKED_NOPE_BYTES = DSV4NVFP4Cache::PACKED_NOPE_BYTES;
constexpr int DECODE_DATA_BYTES_PER_TOKEN = DSV4NVFP4Cache::DATA_BYTES_PER_TOKEN;
constexpr int DECODE_SCALE_BYTES_PER_TOKEN = DSV4NVFP4Cache::SCALE_BYTES_PER_TOKEN;
constexpr int DECODE_BYTES_PER_TOKEN = DSV4NVFP4Cache::BYTES_PER_TOKEN;
constexpr int DECODE_KV_SMEM_STRIDE = Dsv4Nvfp4OperandResources::KV_SMEM_STRIDE;
constexpr int DECODE_W_PACKED_STRIDE = DECODE_CAND_WINDOW / 2 + 16;
constexpr int DECODE_VT_PACKED_K_BYTES = NVFP4_VT_PACKED_K_BYTES;
constexpr int DECODE_VT_SCALE_GROUPS = NVFP4_VT_SCALE_GROUPS;
constexpr int DECODE_VT_DATA_BYTES = NVFP4_VT_DATA_BYTES;
constexpr int DECODE_VT_SCALE_BYTES = NVFP4_VT_SCALE_BYTES;

static_assert(DECODE_DATA_BYTES_PER_TOKEN == 352);
static_assert(DECODE_BYTES_PER_TOKEN == 384);
static_assert(DECODE_KV_SMEM_STRIDE == 240);

template <ModelType MT>
struct DecodeNVFP4Smem {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::DSV4);

  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP4 = HPB * DSV4_NVFP4_Q_PACKED_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * DSV4_NVFP4_SCALE_STRIDE;
  static constexpr size_t SMEM_KV_FP4_BUF = DECODE_CAND_WINDOW * DECODE_KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = DECODE_CAND_WINDOW * DECODE_SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF = DECODE_CAND_WINDOW * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * DECODE_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_SC = HPB * DECODE_VT_SCALE_GROUPS;
  static constexpr size_t SMEM_W_FP4 = HPB * DECODE_W_PACKED_STRIDE;
  static constexpr size_t SMEM_VT_DATA = DECODE_VT_DATA_BYTES;
  static constexpr size_t SMEM_VT_SC = DECODE_VT_SCALE_BYTES;

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP4 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP4 + SMEM_Q_FP4;
  static constexpr size_t OFF_KV_FP4 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_SC = OFF_KV_FP4 + DECODE_KV_BUF_COUNT * SMEM_KV_FP4_BUF;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_SC + DECODE_KV_BUF_COUNT * SMEM_KV_SC_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + DECODE_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_MBAR_VT = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_VT + sizeof(uint64_t);
  static constexpr size_t OFF_W_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP4_UNALIGNED = OFF_W_SC + SMEM_W_SC;
  static constexpr size_t OFF_W_FP4 = (OFF_W_FP4_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_DATA_UNALIGNED = OFF_W_FP4 + SMEM_W_FP4;
  static constexpr size_t OFF_VT_DATA = (OFF_VT_DATA_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_SC = OFF_VT_DATA + SMEM_VT_DATA;
  static constexpr size_t SIZE = OFF_VT_SC + SMEM_VT_SC;

  char* base;

  __device__ static DecodeNVFP4Smem init(char* base) { return DecodeNVFP4Smem{base}; }
  __device__ __forceinline__ bf16* q_rope() const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE);
  }
  __device__ __forceinline__ uint8_t* q_fp4() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP4);
  }
  __device__ __forceinline__ uint8_t* q_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_SC);
  }
  __device__ __forceinline__ uint8_t* kv_fp4(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP4 + parity * SMEM_KV_FP4_BUF);
  }
  __device__ __forceinline__ uint8_t* kv_sc(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC + parity * SMEM_KV_SC_BUF);
  }
  __device__ __forceinline__ bf16* kv_rope(int parity) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE + parity * SMEM_KV_ROPE_BUF);
  }
  __device__ __forceinline__ uint64_t* mbar_full(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_vt() const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT);
  }
  __device__ __forceinline__ float* reduce_scratch() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ float* reduce_scratch_second() const { return reduce_scratch() + DECODE_N_WARPS * HPB; }
  __device__ __forceinline__ uint8_t* w_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_SC);
  }
  __device__ __forceinline__ uint8_t* p_fp4() const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP4);
  }
  __device__ __forceinline__ uint8_t* vt_data() const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_DATA);
  }
  __device__ __forceinline__ uint8_t* vt_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_SC);
  }
};

// This grouped streaming kernel is shared by direct-write prefill and by the
// grouped split-K decode tactic. Phase-specific launchers keep their public
// ABI and workspace policy separate.
constexpr int STREAMING_HEAD_GROUPS = 4;
constexpr int STREAMING_HEADS_PER_CTA = STREAMING_HEAD_GROUPS * HPB;
// A 64-head CTA amortizes one selected-V transpose over four 16-head groups.
// Eight producer warps sustain the online candidate-axis requantization while
// eight math warps retain four output accumulator groups.
constexpr int STREAMING_GATHER_WARPS = 2;
constexpr int STREAMING_IO_WARPS = 8;
constexpr int STREAMING_N_WARPS = 8;
constexpr int STREAMING_IO_MAX_REGS = 40;
constexpr int STREAMING_MATH_MAX_REGS = 216;
constexpr int STREAMING_VT_PIPE_STAGES = 1;
constexpr int STREAMING_CAND_WINDOW = NVFP4_VT_CANDIDATES;
constexpr int STREAMING_KV_BUF_COUNT = 2;
constexpr int STREAMING_ENTRIES_PER_WARP = STREAMING_CAND_WINDOW / STREAMING_N_WARPS;
constexpr int STREAMING_QK_N_TILES = STREAMING_ENTRIES_PER_WARP / 8;
constexpr int STREAMING_PACKED_NOPE_BYTES = DSV4NVFP4Cache::PACKED_NOPE_BYTES;
constexpr int STREAMING_DATA_BYTES_PER_TOKEN = DSV4NVFP4Cache::DATA_BYTES_PER_TOKEN;
constexpr int STREAMING_SCALE_BYTES_PER_TOKEN = DSV4NVFP4Cache::SCALE_BYTES_PER_TOKEN;
constexpr int STREAMING_KV_SMEM_STRIDE = STREAMING_PACKED_NOPE_BYTES;
constexpr int STREAMING_Q_FP4_STRIDE = STREAMING_PACKED_NOPE_BYTES;
constexpr int STREAMING_Q_SCALE_STRIDE = DSV4_NVFP4_NUM_SCALES;
constexpr int STREAMING_W_PACKED_STRIDE = STREAMING_CAND_WINDOW / 2;
constexpr int STREAMING_BLOCK_THREADS = (STREAMING_N_WARPS + STREAMING_IO_WARPS) * 32;
constexpr int STREAMING_MATH_THREADS = STREAMING_N_WARPS * 32;
// Padding breaks the 128-byte head-to-head alias in the probability staging
// matrix.  A multiple of eight BF16 elements preserves ldmatrix alignment
// while cutting the dominant P-quant shared-load bank conflict in half.
constexpr int STREAMING_P_STRIDE = STREAMING_CAND_WINDOW + 8;

// Four 16-head groups share one selected-K gather and one transient V^T tile.
// The resulting 64-head CTA amortizes online V preparation while retaining the
// native NVFP4 QK/PV atoms validated by the decode path.
struct StreamingNVFP4Smem {
  static constexpr size_t SMEM_Q_ROPE =
      STREAMING_HEADS_PER_CTA * DSV4NVFP4Cache::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP4 = STREAMING_HEADS_PER_CTA * STREAMING_Q_FP4_STRIDE;
  static constexpr size_t SMEM_Q_SC = STREAMING_HEADS_PER_CTA * STREAMING_Q_SCALE_STRIDE;
  static constexpr size_t SMEM_KV_FP4_BUF = STREAMING_CAND_WINDOW * STREAMING_KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = STREAMING_CAND_WINDOW * STREAMING_SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF =
      STREAMING_CAND_WINDOW * DSV4NVFP4Cache::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_MBAR_VT_PIPE = STREAMING_VT_PIPE_STAGES * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE =
      STREAMING_HEAD_GROUPS * STREAMING_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_SC = STREAMING_HEADS_PER_CTA * NVFP4_VT_SCALE_GROUPS;
  static constexpr size_t SMEM_W_FP4 = STREAMING_HEADS_PER_CTA * STREAMING_W_PACKED_STRIDE;
  // One CTA-local full V^T tile is consumed immediately after preparation and
  // never leaves shared memory.  KV itself remains double buffered.
  static constexpr size_t SMEM_VT_DATA = STREAMING_VT_PIPE_STAGES * NVFP4_VT_DATA_BYTES;
  static constexpr size_t SMEM_VT_SC = STREAMING_VT_PIPE_STAGES * NVFP4_VT_SCALE_BYTES;
  static constexpr size_t SMEM_P_FULL = STREAMING_HEADS_PER_CTA * STREAMING_P_STRIDE * sizeof(bf16);

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP4 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP4 + SMEM_Q_FP4;
  static constexpr size_t OFF_KV_FP4 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_SC = OFF_KV_FP4 + STREAMING_KV_BUF_COUNT * SMEM_KV_FP4_BUF;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_SC + STREAMING_KV_BUF_COUNT * SMEM_KV_SC_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + STREAMING_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_MBAR_VT_FULL = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_MBAR_VT_EMPTY = OFF_MBAR_VT_FULL + SMEM_MBAR_VT_PIPE;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_VT_EMPTY + SMEM_MBAR_VT_PIPE;
  // Softmax reduction is dead before P quantization starts, and P operands
  // are dead before the next chunk's reduction.  Reuse that storage for W.
  static constexpr size_t OFF_W_SC = OFF_REDUCE;
  static constexpr size_t OFF_W_FP4_UNALIGNED = OFF_W_SC + SMEM_W_SC;
  static constexpr size_t OFF_W_FP4 = (OFF_W_FP4_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_W_END = OFF_W_FP4 + SMEM_W_FP4;
  static constexpr size_t OFF_REDUCE_END = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_VT_DATA_UNALIGNED =
      OFF_W_END > OFF_REDUCE_END ? OFF_W_END : OFF_REDUCE_END;
  static constexpr size_t OFF_VT_DATA = (OFF_VT_DATA_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_SC = OFF_VT_DATA + SMEM_VT_DATA;
  static constexpr size_t OFF_P_FULL_UNALIGNED = OFF_VT_SC + SMEM_VT_SC;
  static constexpr size_t OFF_P_FULL = (OFF_P_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t SIZE = OFF_P_FULL + SMEM_P_FULL;

  char* base;

  __device__ static StreamingNVFP4Smem init(char* base) { return StreamingNVFP4Smem{base}; }
  __device__ __forceinline__ bf16* q_rope(int group) const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE) + group * HPB * DSV4NVFP4Cache::D_ROPE;
  }
  __device__ __forceinline__ uint8_t* q_fp4(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP4) + group * HPB * STREAMING_Q_FP4_STRIDE;
  }
  __device__ __forceinline__ uint8_t* q_sc(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_SC) + group * HPB * STREAMING_Q_SCALE_STRIDE;
  }
  __device__ __forceinline__ uint8_t* kv_fp4(int raw_slot) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP4) + raw_slot * SMEM_KV_FP4_BUF;
  }
  __device__ __forceinline__ uint8_t* kv_sc(int raw_slot) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC) + raw_slot * SMEM_KV_SC_BUF;
  }
  __device__ __forceinline__ bf16* kv_rope(int raw_slot) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE) +
           raw_slot * STREAMING_CAND_WINDOW * DSV4NVFP4Cache::D_ROPE;
  }
  __device__ __forceinline__ uint64_t* mbar_full(int raw_slot) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + raw_slot;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int raw_slot) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + raw_slot;
  }
  __device__ __forceinline__ uint64_t* mbar_vt_full(int stage) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT_FULL) + stage;
  }
  __device__ __forceinline__ uint64_t* mbar_vt_empty(int stage) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT_EMPTY) + stage;
  }
  __device__ __forceinline__ float* reduce_scratch() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ uint8_t* w_sc(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_SC) + group * HPB * NVFP4_VT_SCALE_GROUPS;
  }
  __device__ __forceinline__ uint8_t* p_fp4(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP4) + group * HPB * STREAMING_W_PACKED_STRIDE;
  }
  __device__ __forceinline__ uint8_t* vt_data(int stage) const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_DATA) + stage * NVFP4_VT_DATA_BYTES;
  }
  __device__ __forceinline__ uint8_t* vt_sc(int stage) const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_SC) + stage * NVFP4_VT_SCALE_BYTES;
  }
  __device__ __forceinline__ bf16* p_full(int group) const {
    return reinterpret_cast<bf16*>(base + OFF_P_FULL) + group * HPB * STREAMING_P_STRIDE;
  }
};

static_assert(StreamingNVFP4Smem::SIZE <= 99 * 1024);

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
