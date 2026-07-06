/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace flashinfer {

namespace {

// Tile geometry of the TRTLLM custom-mask spec-dec generation cubins
// used here: Q128 KeepsMmaAbForGeneration with tileSizeQ == tileSizeKv == 128.
// numInstsQ = stepQ / tileSizeQ (2 for the 2Qx1KV SlidingWindowCustom cubins,
// else 1) and numInstsKv = stepKv / tileSizeKv (2 for same-dtype Q/KV cubins,
// 1 for mixed-dtype and for 2Qx1KV); both are passed in by the caller and
// must match the selected kernel's instance split.
constexpr int kTileSizeQ = 128;
constexpr int kTileSizeKv = 128;
constexpr int kThreadsPerBlock = 256;
constexpr int kKeepsMaskLayout = 0;
constexpr int kSwapsMaskLayout = 1;

// Packs a dense spec-dec tree mask (one bool per (draft tokenQ, draft tokenKv)
// pair) into the per-sequence packed custom mask consumed by the kernels.
//
// The bit layout mirrors TRT-LLM's consumer-side packer
// (cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/prepareCustomMask.cu,
// prepareCustomMaskBuffersKernelForKeepsMmaAb): per sequence the packed mask
// is a bit array shaped
//   [numTilesQ, numCustomMaskTilesKv, numInstsQ, numInstsKv,
//    tileSizeQ(=128), tileSizeKv(=128)]
// with rows indexed by tokenQ * numHeadsQPerKv + head (head-broadcast) and
// numCustomMaskTilesKv = ceilDiv(seqLenKv, stepKv) - firstSparseTileKv.
//
// The KV range starts at the tile-aligned firstSparseMaskOffsetKv: KV tokens
// before the draft tail (the prefix) are emitted as visible, draft-tail
// tokens take the tree mask bit, and out-of-range tokens are masked.
//
// window_left >= 0 folds a sliding window into the mask: the published
// custom-mask cubins have no SlidingWindow+Custom mask type, and KV below
// firstSparseMaskOffsetKv is implicitly visible, so the window forces
// firstSparseMaskOffsetKv = 0 and explicit bits over the whole presented KV
// range. The window uses slot indices (kv visible iff
// kv_idx >= prefix + tokenQ - window_left), matching the kernel-side
// window_left rule the non-tree spec-dec paths and FA2 references apply.
// Callers should trim the presented KV (page table + seq_lens) to roughly
// the window so the mask region stays small. Proper SlidingWindow+Custom
// kernel support in TRTLLM would remove the firstSparseMaskOffsetKv = 0
// requirement and the packing cost; that is a planned follow-up.
//
// The mask region size depends on the runtime seqLensKv values, so this runs
// as a device kernel (CUDA-graph capturable) over a static worst-case grid.
__global__ void PackSpecDecMaskKernel(uint32_t* packed_mask, int32_t* first_sparse_mask_offsets_kv,
                                      bool const* tree_mask, int const* seq_lens_kv, int q_len,
                                      int num_heads_q_per_kv, int num_insts_q, int num_insts_kv,
                                      int tile_size_q, int mask_layout, int max_words_per_seq,
                                      int window_left) {
  int const batch_idx = blockIdx.y;
  int const seq_len_kv = seq_lens_kv[batch_idx];
  int const prefix_len = seq_len_kv >= q_len ? seq_len_kv - q_len : 0;
  int const tile_size_kv_per_cta = kTileSizeKv * num_insts_kv;
  bool const has_window = window_left >= 0;
  int const first_sparse_tile_kv = has_window ? 0 : prefix_len / tile_size_kv_per_cta;
  int const adjusted_offset_kv = first_sparse_tile_kv * tile_size_kv_per_cta;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    first_sparse_mask_offsets_kv[batch_idx] = adjusted_offset_kv;
  }
  bool const is_swaps = mask_layout == kSwapsMaskLayout;
  int const padded_tile_size_q = ((tile_size_q + 31) / 32) * 32;
  int const num_tiles_q_per_token =
      (num_heads_q_per_kv + padded_tile_size_q - 1) / padded_tile_size_q;
  int const tile_size_q_per_cta = kTileSizeQ * num_insts_q;
  int const num_tiles_q =
      is_swaps ? q_len * num_tiles_q_per_token
               : (q_len * num_heads_q_per_kv + tile_size_q_per_cta - 1) / tile_size_q_per_cta;
  int const num_mask_tiles_kv =
      (seq_len_kv + tile_size_kv_per_cta - 1) / tile_size_kv_per_cta - first_sparse_tile_kv;
  int const words_per_q = is_swaps ? (tile_size_q + 31) / 32 : 0;
  int const words_per_inst = is_swaps ? kTileSizeKv * words_per_q : kTileSizeQ * kTileSizeKv / 32;
  int const words_per_tile_block = num_insts_q * num_insts_kv * words_per_inst;
  int const num_words = num_tiles_q * num_mask_tiles_kv * words_per_tile_block;

  bool const* local_tree_mask = tree_mask + static_cast<int64_t>(batch_idx) * q_len * q_len;
  uint32_t* local_packed_mask = packed_mask + static_cast<int64_t>(batch_idx) * max_words_per_seq;

  int const linear_idx_start = blockIdx.x * blockDim.x + threadIdx.x;
  int const linear_idx_stride = gridDim.x * blockDim.x;
  if (is_swaps) {
    int const num_positions = q_len * num_mask_tiles_kv * tile_size_kv_per_cta;
    for (int linear_idx = linear_idx_start; linear_idx < num_positions;
         linear_idx += linear_idx_stride) {
      int const token_idx_q = linear_idx / (num_mask_tiles_kv * tile_size_kv_per_cta);
      int const kv_in_region = linear_idx % (num_mask_tiles_kv * tile_size_kv_per_cta);
      int const token_idx_kv = adjusted_offset_kv + kv_in_region;
      if (token_idx_kv >= seq_len_kv) {
        continue;
      }

      int const window_floor = has_window ? prefix_len + token_idx_q - window_left : INT_MIN;
      bool visible = false;
      if (token_idx_kv < prefix_len) {
        visible = token_idx_kv >= window_floor;
      } else {
        visible = local_tree_mask[token_idx_q * q_len + (token_idx_kv - prefix_len)] &&
                  token_idx_kv >= window_floor;
      }
      if (!visible) {
        continue;
      }

      int const tile_idx_kv = kv_in_region / tile_size_kv_per_cta;
      int const inst_idx_kv = (kv_in_region % tile_size_kv_per_cta) / kTileSizeKv;
      int const token_idx_in_tile_kv = (kv_in_region % tile_size_kv_per_cta) % kTileSizeKv;
      for (int head_idx_q = 0; head_idx_q < num_heads_q_per_kv; ++head_idx_q) {
        int const tile_idx_q =
            token_idx_q * num_tiles_q_per_token + head_idx_q / padded_tile_size_q;
        int const token_idx_in_tile_q = head_idx_q % padded_tile_size_q;
        int64_t const tile_offset =
            static_cast<int64_t>(tile_idx_q) * num_mask_tiles_kv + tile_idx_kv;
        int64_t const inst_offset = tile_offset * num_insts_kv + inst_idx_kv;
        int64_t mask_offset = inst_offset * padded_tile_size_q * kTileSizeKv;
        int const thread_idx_q = (token_idx_in_tile_q % 8) / 2;
        int const thread_idx_kv = (token_idx_in_tile_kv % 8) + (token_idx_in_tile_kv / 32) * 8;
        int const token_idx_in_warp_tile_kv = token_idx_in_tile_kv % 32;
        int const elt_idx_in_thread = (token_idx_in_tile_q % 2) +
                                      ((token_idx_in_warp_tile_kv / 8) % 2) * 2 +
                                      (token_idx_in_tile_q / 8) * 4 +
                                      (token_idx_in_warp_tile_kv / 16) * 4 * (tile_size_q / 8);
        mask_offset += (thread_idx_kv * 4 + thread_idx_q) * 32 + elt_idx_in_thread;
        atomicOr(local_packed_mask + (mask_offset >> 5), 1U << (mask_offset & 0x1F));
      }
    }
    return;
  }

  for (int word_idx = linear_idx_start; word_idx < num_words; word_idx += linear_idx_stride) {
    int const tile_block_idx = word_idx / words_per_tile_block;
    int const word_in_block = word_idx % words_per_tile_block;
    int const tile_idx_q = tile_block_idx / num_mask_tiles_kv;
    int const tile_idx_kv = tile_block_idx % num_mask_tiles_kv;
    // Instance order matches the reference layout: [.., numInstsQ, numInstsKv, ..].
    int const inst_idx_q = word_in_block / (num_insts_kv * words_per_inst);
    int const inst_idx_kv = (word_in_block % (num_insts_kv * words_per_inst)) / words_per_inst;
    int const word_in_inst = word_in_block % words_per_inst;
    uint32_t word = 0u;
    int const token_idx_in_tile_q = word_in_inst / (kTileSizeKv / 32);
    int const kv_word = word_in_inst % (kTileSizeKv / 32);
    int const row_q =
        tile_idx_q * tile_size_q_per_cta + inst_idx_q * kTileSizeQ + token_idx_in_tile_q;
    if (row_q < q_len * num_heads_q_per_kv) {
      int const token_idx_q = row_q / num_heads_q_per_kv;
      int const kv_base = adjusted_offset_kv + tile_idx_kv * tile_size_kv_per_cta +
                          inst_idx_kv * kTileSizeKv + kv_word * 32;
      int const window_floor = has_window ? prefix_len + token_idx_q - window_left : INT_MIN;
#pragma unroll
      for (int bit = 0; bit < 32; ++bit) {
        int const token_idx_kv = kv_base + bit;
        bool visible = false;
        if (token_idx_kv < prefix_len) {
          visible = token_idx_kv >= window_floor;
        } else if (token_idx_kv < seq_len_kv) {
          visible = local_tree_mask[token_idx_q * q_len + (token_idx_kv - prefix_len)] &&
                    token_idx_kv >= window_floor;
        }
        word |= (static_cast<uint32_t>(visible) << bit);
      }
    }
    local_packed_mask[word_idx] = word;
  }
}

}  // namespace

void trtllm_fmha_pack_spec_dec_mask(TensorView packed_mask, TensorView first_sparse_mask_offsets_kv,
                                    TensorView tree_mask, TensorView seq_lens,
                                    int64_t num_heads_q_per_kv, int64_t num_insts_q,
                                    int64_t num_insts_kv, int64_t tile_size_q, int64_t mask_layout,
                                    int64_t window_left) {
  TVM_FFI_ICHECK_EQ(tree_mask.dtype(), dl_bool) << "tree_mask must be a bool tensor";
  TVM_FFI_ICHECK_EQ(tree_mask.ndim(), 3) << "tree_mask must have shape [batch, q_len, q_len]";
  TVM_FFI_ICHECK_EQ(tree_mask.size(1), tree_mask.size(2))
      << "tree_mask must have shape [batch, q_len, q_len]";
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32) << "seq_lens must be an int32 tensor";
  TVM_FFI_ICHECK_EQ(packed_mask.dtype(), dl_uint32) << "packed_mask must be a uint32 tensor";
  TVM_FFI_ICHECK_EQ(packed_mask.ndim(), 2)
      << "packed_mask must have shape [batch, max_words_per_seq]";
  TVM_FFI_ICHECK_EQ(first_sparse_mask_offsets_kv.dtype(), dl_int32)
      << "first_sparse_mask_offsets_kv must be an int32 tensor";
  TVM_FFI_ICHECK(num_insts_kv == 1 || num_insts_kv == 2) << "num_insts_kv must be 1 or 2";
  TVM_FFI_ICHECK(num_insts_q == 1 || num_insts_q == 2) << "num_insts_q must be 1 or 2";
  TVM_FFI_ICHECK(num_insts_q * num_insts_kv <= 2)
      << "at most two tile instances are supported (numInstsQ * numInstsKv <= 2)";
  TVM_FFI_ICHECK(num_insts_q == 1 || mask_layout == kKeepsMaskLayout)
      << "num_insts_q == 2 is only supported by the keeps mask layout";
  TVM_FFI_ICHECK(tile_size_q == 8 || tile_size_q == 16 || tile_size_q == 32 || tile_size_q == 128)
      << "tile_size_q must be 8, 16, 32, or 128";
  TVM_FFI_ICHECK(mask_layout == kKeepsMaskLayout || mask_layout == kSwapsMaskLayout)
      << "mask_layout must be keeps or swaps";

  int const batch_size = tree_mask.size(0);
  int const q_len = tree_mask.size(1);
  int const max_words_per_seq = packed_mask.size(1);
  TVM_FFI_ICHECK_EQ(packed_mask.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(first_sparse_mask_offsets_kv.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);

  const auto stream = get_stream(packed_mask.device());
  cudaMemsetAsync(packed_mask.data_ptr(), 0,
                  packed_mask.size(0) * packed_mask.size(1) * sizeof(uint32_t), stream);
  dim3 const grid((max_words_per_seq + kThreadsPerBlock - 1) / kThreadsPerBlock, batch_size);
  PackSpecDecMaskKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      static_cast<uint32_t*>(packed_mask.data_ptr()),
      static_cast<int32_t*>(first_sparse_mask_offsets_kv.data_ptr()),
      static_cast<bool const*>(tree_mask.data_ptr()), static_cast<int const*>(seq_lens.data_ptr()),
      q_len, static_cast<int>(num_heads_q_per_kv), static_cast<int>(num_insts_q),
      static_cast<int>(num_insts_kv), static_cast<int>(tile_size_q), static_cast<int>(mask_layout),
      max_words_per_seq, static_cast<int>(window_left));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fmha_pack_spec_dec_mask, trtllm_fmha_pack_spec_dec_mask);

}  // namespace flashinfer
