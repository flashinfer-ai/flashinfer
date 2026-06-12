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

// Tile geometry of the TRTLLM-GEN custom-mask spec-dec generation cubins
// used here: Q128 KeepsMmaAbForGeneration with stepQ == tileSizeQ == 128
// (numInstsQ = 1) and tileSizeKv == 128. numInstsKv = stepKv / tileSizeKv is
// 2 for same-dtype Q/KV cubins (stepKv 256) and 1 for mixed-dtype (stepKv
// 128); it is passed in by the caller and must match the selected kernel.
constexpr int kTileSizeQ = 128;
constexpr int kTileSizeKv = 128;
constexpr int kThreadsPerBlock = 256;

// Packs a dense spec-dec tree mask (one bool per (draft tokenQ, draft tokenKv)
// pair) into the per-sequence packed custom mask consumed by the kernels.
//
// The bit layout mirrors TRT-LLM's consumer-side packer
// (cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/prepareCustomMask.cu,
// prepareCustomMaskBuffersKernelForKeepsMmaAb) and trtllm-gen's host
// reference (kernels/Fmha/Fmha.cpp): per sequence the packed mask is a bit
// array shaped
//   [numTilesQ, numCustomMaskTilesKv, numInstsQ(=1), numInstsKv,
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
// kernel support in trtllm-gen would remove the firstSparseMaskOffsetKv = 0
// requirement and the packing cost; that is a planned follow-up.
//
// The mask region size depends on the runtime seqLensKv values, so this runs
// as a device kernel (CUDA-graph capturable) over a static worst-case grid.
__global__ void PackSpecDecMaskKernel(uint32_t* packed_mask, int32_t* first_sparse_mask_offsets_kv,
                                      bool const* tree_mask, int const* seq_lens_kv, int q_len,
                                      int num_heads_q_per_kv, int num_insts_kv,
                                      int max_words_per_seq, int window_left) {
  int const batch_idx = blockIdx.y;
  int const seq_len_kv = seq_lens_kv[batch_idx];
  int const prefix_len = seq_len_kv - q_len;
  int const tile_size_kv_per_cta = kTileSizeKv * num_insts_kv;
  bool const has_window = window_left >= 0;
  int const first_sparse_tile_kv = has_window ? 0 : prefix_len / tile_size_kv_per_cta;
  int const adjusted_offset_kv = first_sparse_tile_kv * tile_size_kv_per_cta;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    first_sparse_mask_offsets_kv[batch_idx] = adjusted_offset_kv;
  }
  int const num_tiles_q = (q_len * num_heads_q_per_kv + kTileSizeQ - 1) / kTileSizeQ;
  int const num_mask_tiles_kv =
      (seq_len_kv + tile_size_kv_per_cta - 1) / tile_size_kv_per_cta - first_sparse_tile_kv;
  int const words_per_inst = kTileSizeQ * kTileSizeKv / 32;
  int const words_per_tile_block = num_insts_kv * words_per_inst;
  int const num_words = num_tiles_q * num_mask_tiles_kv * words_per_tile_block;

  bool const* local_tree_mask = tree_mask + static_cast<int64_t>(batch_idx) * q_len * q_len;
  uint32_t* local_packed_mask = packed_mask + static_cast<int64_t>(batch_idx) * max_words_per_seq;

  for (int word_idx = blockIdx.x * blockDim.x + threadIdx.x; word_idx < num_words;
       word_idx += gridDim.x * blockDim.x) {
    int const tile_block_idx = word_idx / words_per_tile_block;
    int const word_in_block = word_idx % words_per_tile_block;
    int const tile_idx_q = tile_block_idx / num_mask_tiles_kv;
    int const tile_idx_kv = tile_block_idx % num_mask_tiles_kv;
    int const inst_idx_kv = word_in_block / words_per_inst;
    int const word_in_inst = word_in_block % words_per_inst;
    int const token_idx_in_tile_q = word_in_inst / (kTileSizeKv / 32);
    int const kv_word = word_in_inst % (kTileSizeKv / 32);

    int const row_q = tile_idx_q * kTileSizeQ + token_idx_in_tile_q;
    uint32_t word = 0u;
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
                                    int64_t num_heads_q_per_kv, int64_t num_insts_kv,
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

  int const batch_size = tree_mask.size(0);
  int const q_len = tree_mask.size(1);
  int const max_words_per_seq = packed_mask.size(1);
  TVM_FFI_ICHECK_EQ(packed_mask.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(first_sparse_mask_offsets_kv.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);

  const auto stream = get_stream(packed_mask.device());
  dim3 const grid((max_words_per_seq + kThreadsPerBlock - 1) / kThreadsPerBlock, batch_size);
  PackSpecDecMaskKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      static_cast<uint32_t*>(packed_mask.data_ptr()),
      static_cast<int32_t*>(first_sparse_mask_offsets_kv.data_ptr()),
      static_cast<bool const*>(tree_mask.data_ptr()), static_cast<int const*>(seq_lens.data_ptr()),
      q_len, static_cast<int>(num_heads_q_per_kv), static_cast<int>(num_insts_kv),
      max_words_per_seq, static_cast<int>(window_left));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fmha_pack_spec_dec_mask, trtllm_fmha_pack_spec_dec_mask);

}  // namespace flashinfer
