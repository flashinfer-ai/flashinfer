/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TVM-FFI entry points of the SM120 DSA indexer: run (the ragged form,
// sm120_dsa_indexer.cuh) and run_decode (the paged form, decode_*.cuh), each
// launching a fixed sequence of kernels on the caller's stream.
#include <algorithm>

#include "decode_scan.cuh"
#include "decode_select_long.cuh"
#include "sm120_dsa_indexer.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sm120_dsa_indexer {
namespace {

template <typename T>
T* ptr(TensorView t) {
  return static_cast<T*>(t.data_ptr());
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(e, cudaSuccess)
      << "sm120 dsa indexer " << what << ": " << cudaGetErrorString(e);
}

CUtensorMap make_tmap(const char* name, const void* base, CUtensorMapDataType type, int rank,
                      const uint64_t* dims, const uint64_t* strides_bytes, const uint32_t* box,
                      CUtensorMapSwizzle swizzle) {
  CUtensorMap m{};
  const uint32_t elem_strides[3] = {1, 1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &m, type, rank, const_cast<void*>(base), dims, strides_bytes, box, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);  // out-of-bounds elements read as zero
  TVM_FFI_ICHECK_EQ(r, CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled(" << name << ") failed: " << int(r) << " base=" << base
      << " dims=" << dims[0] << "," << (rank > 1 ? dims[1] : 0) << "," << (rank > 2 ? dims[2] : 0);
  return m;
}

// The SMs this launch may use: the stream's green context, else the current
// context's SM resource (partitions), else the device. Persistent grids and
// key splits are sized from it.
int stream_sm_count(cudaStream_t stream) {
  CUdevResource res{};
  CUgreenCtx green = nullptr;
  if (cuStreamGetGreenCtx(reinterpret_cast<CUstream>(stream), &green) == CUDA_SUCCESS &&
      green != nullptr &&
      cuGreenCtxGetDevResource(green, &res, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS &&
      res.sm.smCount > 0)
    return static_cast<int>(res.sm.smCount);
  CUcontext ctx = nullptr;
  if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx != nullptr &&
      cuCtxGetDevResource(ctx, &res, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS && res.sm.smCount > 0)
    return static_cast<int>(res.sm.smCount);
  int device = 0, sms = 0;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
                    cudaSuccess);
  return sms;
}

// Key splits for the exact path: the fewest that minimize waves per split (a
// larger count must win by 3%).
int choose_kv_splits(int units, int num_keys, int sms) {
  const int max_tiles = (num_keys + kBlockKV - 1) / kBlockKV;
  int max_splits = (4 * sms) / (units > 0 ? units : 1) + 1;
  if (max_tiles / 8 < max_splits) max_splits = max_tiles / 8;
  if (max_splits < 1) max_splits = 1;
  int best = 1;
  double best_cost = static_cast<double>((units + sms - 1) / sms);
  for (int s = 2; s <= max_splits; ++s) {
    const double cost = static_cast<double>((units * s + sms - 1) / sms) / s;
    if (cost < 0.97 * best_cost) {
      best_cost = cost;
      best = s;
    }
  }
  return best;
}

// The scoring kernels' inputs. FP8: q [T, H, 128] e4m3, k [Tkv, 128] e4m3,
// k_scale [Tkv] fp32. MXFP4: q [T, H, 64] and k [Tkv, 64] packed e2m1, q_sf
// [T, H] and k_scale [Tkv] int32 (4 x ue8m0).
struct ScoreInputs {
  const void* q;
  const void* w;
  const void* k;
  const void* k_scale;
  const void* q_sf;
  int num_rows;
  int num_keys;
  int num_sms;
};

// pdl: programmatic dependent launch. coop: a cooperative launch (grid
// barriers inside).
template <typename... Params, typename... Args>
void launch_ex(void (*kernel)(Params...), int grid, int block, int smem, cudaStream_t stream,
               bool pdl, bool coop, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(static_cast<unsigned>(block));
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attr[1].id = cudaLaunchAttributeCooperative;
  attr[1].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = coop ? 2 : 1;
  TVM_FFI_ICHECK_EQ(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...), cudaSuccess);
}

// The outputs: indices [rows, top_k] int32 and the optional scores [rows,
// top_k] fp32 (null: indices only), both contiguous.
float* check_outputs(TensorView indices, const Optional<TensorView>& scores, int64_t rows,
                     int64_t top_k) {
  CHECK_INPUT_TYPE(indices, dl_int32);
  TVM_FFI_ICHECK(indices.ndim() == 2 && indices.size(0) == rows && indices.size(1) == top_k &&
                 indices.IsContiguous())
      << "indices must be a contiguous [" << rows << ", " << top_k << "] int32 tensor";
  if (!scores.has_value()) return nullptr;
  const TensorView& s = scores.value();
  CHECK_INPUT_TYPE(s, dl_float32);
  TVM_FFI_ICHECK(s.ndim() == 2 && s.size(0) == rows && s.size(1) == top_k && s.IsContiguous())
      << "scores must be a contiguous [" << rows << ", " << top_k << "] float32 tensor";
  return ptr<float>(s);
}

template <int kNumHeads, Mode kMode, bool kFp4, int kSlots>
void launch_score(const ScoreParams& p, const ScoreInputs& in, cudaStream_t stream, bool coop,
                  int smem, bool pdl) {
  using L = ScoreSmem<kNumHeads, kFp4>;
  constexpr int kBlockQ = L::kBlockQ;
  constexpr uint64_t kRow = Fmt<kFp4>::kRow;
  const CUtensorMapSwizzle swizzle = kFp4 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  // q as [T * H, kRow] bytes in boxes of 128 head rows; w (and MXFP4 q_sf)
  // [T, H] words in boxes of kBlockQ rows; k as [Tkv, kRow] bytes in boxes of
  // 128 keys (SAMPLE: 16-key chunks); k_scale [Tkv] words.
  const uint64_t q_dims[2] = {kRow, static_cast<uint64_t>(in.num_rows) * kNumHeads};
  const uint64_t q_strides[1] = {kRow};
  const uint32_t q_box[2] = {static_cast<uint32_t>(kRow), 128};
  const uint64_t w_dims[2] = {kNumHeads, static_cast<uint64_t>(in.num_rows)};
  const uint64_t w_strides[1] = {kNumHeads * 4};
  const uint32_t w_box[2] = {kNumHeads, kBlockQ};
  const uint64_t k_dims[2] = {kRow, static_cast<uint64_t>(in.num_keys)};
  const uint64_t k_strides[1] = {kRow};
  const uint32_t kv_box = kMode == Mode::kSample ? kChunk : kBlockKV;
  const uint32_t k_box[2] = {static_cast<uint32_t>(kRow), kv_box};
  const uint64_t s_dims[2] = {static_cast<uint64_t>(in.num_keys), 1};
  // A size-1 outer dimension: its stride is unused but must be 16-byte aligned.
  const uint64_t s_strides[1] = {(static_cast<uint64_t>(in.num_keys) * 4 + 15) / 16 * 16};
  const uint32_t s_box[2] = {kv_box, 1};
  const CUtensorMap tq =
      make_tmap("q", in.q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q_dims, q_strides, q_box, swizzle);
  const CUtensorMap tw = make_tmap("w", in.w, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, w_dims, w_strides,
                                   w_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tk =
      make_tmap("k", in.k, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, k_dims, k_strides, k_box, swizzle);
  const CUtensorMap ts = make_tmap("k_scale", in.k_scale, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, s_dims,
                                   s_strides, s_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tqs = kFp4 ? make_tmap("q_sf", in.q_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
                                           w_dims, w_strides, w_box, CU_TENSOR_MAP_SWIZZLE_NONE)
                               : tw;
  auto kernel = score_kernel<kNumHeads, kMode, kFp4, kSlots>;
  const int base_bytes =
      kMode == Mode::kResident && L::kResBytes > L::kBytes ? L::kResBytes : L::kBytes;
  const int bytes = smem > base_bytes ? smem : base_bytes;
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
      cudaSuccess);
  // One carveout for every kernel of the call: consecutive launches never wait
  // for an SM reconfiguration.
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (p.num_units == 0) return;
  const int grid = p.num_unit_rows != nullptr || p.tile_prefix != nullptr
                       ? in.num_sms
                       : (p.num_units < in.num_sms ? p.num_units : in.num_sms);
  const char* what = kMode == Mode::kScan       ? "scan"
                     : kMode == Mode::kSample   ? "sample"
                     : kMode == Mode::kResident ? "resident"
                                                : "exact";
  if (coop) {
    int per_sm = 0;
    TVM_FFI_ICHECK_EQ(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kNumWsThreads, bytes),
        cudaSuccess);
    TVM_FFI_ICHECK_GE(per_sm, 1);
  }
  launch_ex(kernel, grid, kNumWsThreads, bytes, stream, pdl, coop, p, tq, tw, tk, ts, tqs);
  check_launch(what);
}

template <Mode kMode, int kSlots = kResSlots>
void dispatch_score(const ScoreParams& p, const ScoreInputs& in, int num_heads, bool fp4,
                    cudaStream_t stream, bool coop, int smem, bool pdl) {
  if (fp4) {
    if (num_heads == 32)
      launch_score<32, kMode, true, kSlots>(p, in, stream, coop, smem, pdl);
    else
      launch_score<64, kMode, true, kSlots>(p, in, stream, coop, smem, pdl);
  } else {
    if (num_heads == 32)
      launch_score<32, kMode, false, kSlots>(p, in, stream, coop, smem, pdl);
    else
      launch_score<64, kMode, false, kSlots>(p, in, stream, coop, smem, pdl);
  }
}

}  // namespace

// The ragged form. row_ks .. sample_slots, res_scratch and row_classes are
// the caller's scratch (_api.py sizes them); res_cap, res_bound and
// res_classes describe the resident classes the call may need, run_scan
// whether some row may be longer than the last class.
void Run(TensorView q, TensorView k, TensorView k_scale, TensorView w, TensorView cu_seqlens_q,
         TensorView cu_seqlens_k, Optional<TensorView> q_causal_offsets, int64_t ratio,
         int64_t top_k, int64_t capacity, int64_t target, TensorView row_ks, TensorView row_ke,
         TensorView tau, TensorView count, TensorView sample, TensorView cand_score,
         TensorView cand_id, TensorView fallback_rows, TensorView fallback_count,
         TensorView indices, Optional<TensorView> scores, TensorView tile_prefix,
         TensorView exact_state, TensorView sample_slots, Optional<TensorView> q_sf,
         Optional<TensorView> res_scratch, Optional<TensorView> row_classes, int64_t res_cap,
         int64_t res_bound, int64_t res_classes, bool run_scan, bool enable_pdl) {
  const bool fp4 = q_sf.has_value();
  const int64_t row_bytes = fp4 ? kHeadDim / 2 : kHeadDim;
  if (fp4) {
    TVM_FFI_ICHECK(q.dtype().bits == 8 && k.dtype().bits == 8) << "MXFP4 q and k are packed bytes";
    CHECK_INPUT_TYPE(k_scale, dl_int32);
    CHECK_INPUT_TYPE(q_sf.value(), dl_int32);
  } else {
    CHECK_INPUT_TYPE(q, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(k, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(k_scale, dl_float32);
  }
  CHECK_INPUT_TYPE(w, dl_float32);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(w);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, row_bytes]";
  TVM_FFI_ICHECK_EQ(q.size(2), row_bytes);
  TVM_FFI_ICHECK_EQ(k.ndim(), 2) << "k must be [Tkv, row_bytes]";
  const int num_rows = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int num_keys = static_cast<int>(k.size(0));
  TVM_FFI_ICHECK(num_heads == 32 || num_heads == 64) << "num_heads must be 32 or 64";
  TVM_FFI_ICHECK(w.ndim() == 2 && w.size(0) == num_rows && w.size(1) == num_heads)
      << "w must be [T, H]";
  const int block_q = 128 / num_heads;
  TVM_FFI_ICHECK_EQ(k.size(1), row_bytes);
  TVM_FFI_ICHECK_EQ(k_scale.numel(), num_keys) << "k_scale must hold one word a key";
  CHECK_INPUT_TYPE(cu_seqlens_q, dl_int32);
  CHECK_INPUT_TYPE(cu_seqlens_k, dl_int32);
  TVM_FFI_ICHECK(cu_seqlens_q.ndim() == 1 && cu_seqlens_q.numel() == cu_seqlens_k.numel() &&
                 cu_seqlens_q.numel() >= 2)
      << "cu_seqlens_q and cu_seqlens_k must be [S + 1]";
  const int num_segments = static_cast<int>(cu_seqlens_q.numel()) - 1;
  if (q_causal_offsets.has_value()) {
    CHECK_INPUT_TYPE(q_causal_offsets.value(), dl_int64);
    TVM_FFI_ICHECK_EQ(q_causal_offsets.value().numel(), num_segments)
        << "q_causal_offsets must be [S]";
  }
  TVM_FFI_ICHECK_GE(ratio, 1) << "ratio must be positive";
  if (fp4) {
    TVM_FFI_ICHECK(q_sf.value().ndim() == 2 && q_sf.value().size(0) == num_rows &&
                   q_sf.value().size(1) == num_heads)
        << "q_sf must be [T, H]";
  }
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= capacity) << "need 1 <= top_k <= capacity";
  TVM_FFI_ICHECK(capacity <= 8192) << "capacity must be <= 8192";
  // The exact path keeps a row's histogram in its candidate buffer.
  TVM_FFI_ICHECK(capacity >= kExactBins) << "capacity must be >= " << kExactBins;
  float* scores_p = check_outputs(indices, scores, num_rows, top_k);
  if (num_rows == 0) return;

  cudaStream_t stream = get_stream(q.device());

  // rows_kernel's per-CTA class masks let a resident launch with none of its
  // class's rows leave at once.
  const int rows_ctas = (num_rows + 255) / 256;
  uint32_t* classes_p = nullptr;
  if (row_classes.has_value() && res_scratch.has_value()) {
    CHECK_INPUT_TYPE(row_classes.value(), dl_int32);
    TVM_FFI_ICHECK_GE(row_classes.value().numel(), rows_ctas) << "row_classes: a word per 256 rows";
    classes_p = ptr<uint32_t>(row_classes.value());
  }
  RowsParams rp{};
  rp.cu_seqlens_q = ptr<const int32_t>(cu_seqlens_q);
  rp.cu_seqlens_k = ptr<const int32_t>(cu_seqlens_k);
  rp.q_causal_offsets =
      q_causal_offsets.has_value() ? ptr<const int64_t>(q_causal_offsets.value()) : nullptr;
  rp.row_ks = ptr<int32_t>(row_ks);
  rp.row_ke = ptr<int32_t>(row_ke);
  rp.row_classes = classes_p;
  rp.class_keys = static_cast<int>(res_cap);
  rp.num_rows = num_rows;
  rp.num_segments = num_segments;
  rp.ratio = static_cast<int>(ratio);
  launch_ex(rows_kernel, rows_ctas, 256, 0, stream, enable_pdl, false, rp);
  check_launch("rows");

  const int num_sms = stream_sm_count(stream);
  const int q_blocks = (num_rows + block_q - 1) / block_q;

  ScoreParams sp{};
  sp.row_ks = ptr<const int32_t>(row_ks);
  sp.row_ke = ptr<const int32_t>(row_ke);
  sp.tau = ptr<const float>(tau);
  sp.sample = ptr<float>(sample);
  sp.sample_slots = ptr<int32_t>(sample_slots);
  sp.k_scale = k_scale.data_ptr();
  sp.sample_target = static_cast<int>(target);
  sp.count = ptr<int32_t>(count);
  sp.cand_score = ptr<float>(cand_score);
  sp.cand_id = ptr<int32_t>(cand_id);
  sp.num_rows = num_rows;
  sp.num_keys = num_keys;
  sp.capacity = static_cast<int>(capacity);
  sp.kv_splits = 1;
  sp.top_k = static_cast<int>(top_k);

  const ScoreInputs in{q.data_ptr(),
                       w.data_ptr(),
                       k.data_ptr(),
                       k_scale.data_ptr(),
                       fp4 ? q_sf.value().data_ptr() : nullptr,
                       num_rows,
                       num_keys,
                       num_sms};
  // The resident route: one launch a class (a bit of res_classes), class i
  // taking the blocks of rows past the class below's cap and at most
  // res_cap * res_class_cap(i) keys (at most the caller's bound).
  if (res_scratch.has_value()) {
    const TensorView& rs = res_scratch.value();
    CHECK_INPUT_TYPE(rs, dl_float32);
    TVM_FFI_ICHECK(res_cap >= 32 && res_cap % 32 == 0 && res_bound % 32 == 0 && res_classes > 0 &&
                   res_classes < (1 << kResClasses))
        << "resident caps are multiples of 32 words";
    sp.num_units = q_blocks;
    sp.fin_p.indices = ptr<int32_t>(indices);
    sp.fin_p.scores = scores_p;
    sp.res_scratch = ptr<float>(rs);
    sp.row_classes = classes_p;
    sp.res_class_ctas = rows_ctas;
    for (int i = 0; i < kResClasses; ++i) {
      if (!(res_classes >> i & 1)) continue;
      const int64_t stride = std::min(res_bound, res_cap * res_class_cap(i));
      const int64_t words = res_cta_words(block_q, stride, res_class_slots(i));
      TVM_FFI_ICHECK_GE(rs.numel(), static_cast<int64_t>(std::min(q_blocks, num_sms)) * words);
      sp.res_stride = stride;
      sp.res_cta_words = words;
      // Blocks are tested only when some row could be longer than a scratch
      // row: beside a longer class or the scan, or past a wrong caller bound
      // (such rows are left out, never written past the scratch row).
      const bool longer = (res_classes >> (i + 1)) != 0 || run_scan || num_keys > stride;
      sp.res_cap = longer ? static_cast<int>(stride) : 0;
      sp.res_lo = i > 0 ? static_cast<int>(res_cap * res_class_cap(i - 1)) : 0;
      sp.res_class = i;
      if (i == 0)
        dispatch_score<Mode::kResident>(sp, in, num_heads, fp4, stream, false, 0, enable_pdl);
      else
        dispatch_score<Mode::kResident, kResSlots / 2>(sp, in, num_heads, fp4, stream, false, 0,
                                                       enable_pdl);
    }
    sp.res_lo = 0;
    sp.row_classes = nullptr;
  }
  if (!run_scan) return;
  // The scan takes the rows in chunks of its buffers' rows (whole query
  // blocks), the same launch sequence a chunk: its memory is bounded by the
  // chunk, not the call.
  const int chunk_rows = static_cast<int>(cand_score.size(0));
  TVM_FFI_ICHECK(chunk_rows >= num_rows || (chunk_rows > 0 && chunk_rows % block_q == 0))
      << "the scan buffers hold all rows or a multiple of block_q rows";
  int top_class = 0;
  while (res_classes >> (top_class + 1)) ++top_class;
  // The scan skips the resident classes' blocks.
  const int scan_cap =
      res_scratch.has_value() ? static_cast<int>(res_cap * res_class_cap(top_class)) : 0;
  int sort_slots = 1;
  while (sort_slots < top_k) sort_slots <<= 1;
  const int fsmem = finalize_smem_bytes(static_cast<int>(capacity), sort_slots);
  TVM_FFI_ICHECK_LE(fsmem, kSmemLimit - 4096) << "top_k too large for the finalize";
  static_assert(sizeof(ExactState) == 32, "exact_state is [T, 8] int32");
  auto scan_chunk = [&](int row0, int num_rows) {
    const int q_blocks = (num_rows + block_q - 1) / block_q;
    // The exact path rescans only the few rows the scan flagged (their count
    // is read on the device): its key splits are sized for one row in 256.
    const int kv_splits = choose_kv_splits(std::max(1, num_rows / 256), num_keys, num_sms);
    ScoreInputs cin = in;
    cin.q = static_cast<const uint8_t*>(in.q) + static_cast<int64_t>(row0) * num_heads * row_bytes;
    cin.w = static_cast<const float*>(in.w) + static_cast<int64_t>(row0) * num_heads;
    if (fp4)
      cin.q_sf = static_cast<const int32_t*>(in.q_sf) + static_cast<int64_t>(row0) * num_heads;
    cin.num_rows = num_rows;
    const int32_t* row_ks_p = ptr<const int32_t>(row_ks) + row0;
    const int32_t* row_ke_p = ptr<const int32_t>(row_ke) + row0;
    int32_t* indices_p = ptr<int32_t>(indices) + static_cast<int64_t>(row0) * top_k;
    float* chunk_scores = scores_p ? scores_p + static_cast<int64_t>(row0) * top_k : nullptr;
    // fallback_rows [2, T]: the rows the scan flagged, and the exact path's
    // (always empty). fallback_count [4]: their counts, zeroed by the sample
    // launch (no fill launch).
    int32_t* fb_rows = ptr<int32_t>(fallback_rows);
    int32_t* fb_count = ptr<int32_t>(fallback_count);
    ExactState* xs = static_cast<ExactState*>(exact_state.data_ptr());
    sp.row_ks = row_ks_p;
    sp.row_ke = row_ke_p;
    sp.num_rows = num_rows;
    sp.res_cap = scan_cap;
    sp.kv_splits = 1;
    sp.unit_rows = nullptr;
    sp.num_unit_rows = nullptr;
    sp.tile_prefix = nullptr;
    sp.xs = nullptr;
    sp.fin_p = FinalizeParams{};
    // One sampled tile a unit when the blocks alone would leave SMs idle.
    sp.sample_unit_tiles = q_blocks * (kSampleTiles / 4) >= 2 * num_sms ? 4 : 1;
    sp.num_units = q_blocks * (kSampleTiles / sp.sample_unit_tiles);

    // Three cooperative launches, a fixed sequence:
    //   1. SAMPLE, then every row's threshold and the scan's tile prefix;
    //   2. SCAN (stream-K), then every row's top-k; rows whose candidate count
    //      missed [min(top_k, visible), capacity] are listed;
    //   3. EXACT over the listed rows, then their top-k (leaves at once with
    //      no listed row).
    // With few rows the per-row steps run inside the launch after a grid
    // barrier; with many, tau_kernel and finalize_kernel give each row a CTA.
    const bool fuse = num_rows <= num_sms;
    sp.fuse_tail = fuse ? 1 : 0;
    if (!fuse)
      TVM_FFI_ICHECK_EQ(
          cudaFuncSetAttribute(finalize_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fsmem),
          cudaSuccess);
    TauParams& tp = sp.tau_p;
    tp = TauParams{};
    tp.zero_words = fb_count + 1;
    tp.num_zero_words = 1;
    tp.tile_prefix = ptr<int32_t>(tile_prefix);
    tp.block_q = block_q;
    tp.num_qblocks = q_blocks;
    tp.sample = ptr<const float>(sample);
    tp.sample_slots = ptr<const int32_t>(sample_slots);
    tp.row_ks = row_ks_p;
    tp.row_ke = row_ke_p;
    tp.tau = ptr<float>(tau);
    tp.count = ptr<int32_t>(count);
    tp.fallback_count = fb_count;
    tp.num_rows = num_rows;
    tp.capacity = static_cast<int>(capacity);
    tp.target = static_cast<int>(target);
    tp.top_k = static_cast<int>(top_k);
    tp.res_cap = scan_cap;
    dispatch_score<Mode::kSample>(sp, cin, num_heads, fp4, stream, fuse, kSampleSlots * 4,
                                  enable_pdl);
    if (!fuse) {
      // num_rows CTAs, and one for the tile prefix.
      launch_ex(tau_kernel, num_rows + 1, 1024, 0, stream, enable_pdl, false, sp.tau_p);
      check_launch("tau");
    }

    FinalizeParams fp{};
    fp.row_ks = row_ks_p;
    fp.row_ke = row_ke_p;
    fp.count = ptr<const int32_t>(count);
    fp.cand_score = ptr<const float>(cand_score);
    fp.cand_id = ptr<const int32_t>(cand_id);
    fp.indices = indices_p;
    fp.scores = chunk_scores;
    fp.num_rows = num_rows;
    fp.capacity = static_cast<int>(capacity);
    fp.top_k = static_cast<int>(top_k);
    fp.sort_slots = sort_slots;
    fp.block_q = block_q;
    fp.res_cap = scan_cap;

    sp.num_units = q_blocks;
    sp.tile_prefix = ptr<const int32_t>(tile_prefix);
    sp.num_qblocks = q_blocks;
    sp.fin_p = fp;
    sp.fin_p.fallback_rows = fb_rows;
    sp.fin_p.fallback_count = fb_count;
    sp.fin_p.xs = xs;
    dispatch_score<Mode::kScan>(sp, cin, num_heads, fp4, stream, fuse, fsmem, enable_pdl);
    if (!fuse) {
      launch_ex(finalize_kernel, num_rows, 1024, fsmem, stream, enable_pdl, false, sp.fin_p);
      check_launch("finalize");
    }

    sp.kv_splits = kv_splits;
    sp.num_units = num_rows * kv_splits;  // a bound: the list length is read on the device
    sp.unit_rows = fb_rows;
    sp.num_unit_rows = fb_count;
    sp.tile_prefix = nullptr;
    sp.xs = xs;
    sp.fin_p = fp;
    sp.fin_p.fallback_rows = fb_rows + num_rows;  // stays empty: the exact path always fits
    sp.fin_p.fallback_count = fb_count + 1;
    sp.fin_p.rows = fb_rows;
    sp.fin_p.rows_count = fb_count;
    sp.fuse_tail = 1;  // the exact path always selects in its own launch
    dispatch_score<Mode::kExact>(sp, cin, num_heads, fp4, stream, true, fsmem, enable_pdl);
  };
  for (int row0 = 0; row0 < num_rows; row0 += chunk_rows)
    scan_chunk(row0, std::min(chunk_rows, num_rows - row0));
}

namespace {

template <int kNumHeads, bool kFp4, int kNextN, typename Kernel>
void launch_scan(Kernel kernel, const ScanParams& p, const void* q, const void* w, const void* kv,
                 const void* q_sf, int num_pages, int64_t page_stride, int grid, bool pdl,
                 cudaStream_t stream, int bytes) {
  constexpr uint64_t kRow = Fmt<kFp4>::kRow;
  const CUtensorMapSwizzle swizzle = kFp4 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  const uint64_t q_dims[2] = {kRow, static_cast<uint64_t>(p.num_rows) * kNumHeads};
  const uint64_t q_strides[1] = {kRow};
  const uint32_t q_box[2] = {static_cast<uint32_t>(kRow), kNextN * kNumHeads};
  const uint64_t w_dims[2] = {kNumHeads, static_cast<uint64_t>(p.num_rows)};
  const uint64_t w_strides[1] = {kNumHeads * 4};
  const uint32_t w_box[2] = {kNumHeads, kNextN};
  const CUtensorMap tq =
      make_tmap("q", q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q_dims, q_strides, q_box, swizzle);
  const CUtensorMap tw = make_tmap("w", w, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, w_dims, w_strides,
                                   w_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tqs = kFp4 ? make_tmap("q_sf", q_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, w_dims,
                                           w_strides, w_box, CU_TENSOR_MAP_SWIZZLE_NONE)
                               : tw;
  // A page: page_rows key rows of kRow bytes, then their page_rows scale words.
  const uint32_t rows = static_cast<uint32_t>(p.page_rows < kBlockKV ? p.page_rows : kBlockKV);
  const uint64_t pk_dims[3] = {kRow, static_cast<uint64_t>(p.page_rows),
                               static_cast<uint64_t>(num_pages)};
  const uint64_t pk_strides[2] = {kRow, static_cast<uint64_t>(page_stride)};
  const uint32_t pk_box[3] = {static_cast<uint32_t>(kRow), rows, 1};
  const CUtensorMap tk =
      make_tmap("kv", kv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, pk_dims, pk_strides, pk_box, swizzle);
  const uint64_t ps_dims[2] = {static_cast<uint64_t>(p.page_rows),
                               static_cast<uint64_t>(num_pages)};
  const uint64_t ps_strides[1] = {static_cast<uint64_t>(page_stride)};
  const uint32_t ps_box[2] = {rows, 1};
  const CUtensorMap ts = make_tmap(
      "kv scales", static_cast<const uint8_t*>(kv) + static_cast<int64_t>(p.page_rows) * kRow,
      CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, ps_dims, ps_strides, ps_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
      cudaSuccess);
  launch_ex(kernel, grid, kNumWsThreads, bytes, stream, pdl, false, p, tq, tw, tk, ts, tqs);
  check_launch("decode scan");
}

}  // namespace

// The paged form. q [num_rows, H, row_bytes] in units of unit_rows (1 or 2)
// rows, each request's units_per_req units consecutive and sharing its
// block_table row; kv the paged cache as [num_pages, page_stride] bytes; w
// [num_rows, H]; row_len [num_rows]. logits [>= num_rows, >= max_context_len]
// fp32 is the scan's output; long_ws [num_rows * kLongWords] (when
// max_context_len > kRowSelKeys) the long rows' selection state, zero at
// first use. Every row_len is at most max_context_len.
void RunDecode(TensorView q, TensorView kv, TensorView w, TensorView row_len,
               TensorView block_table, int64_t page_rows, int64_t unit_rows, int64_t units_per_req,
               int64_t top_k, int64_t max_context_len, Optional<TensorView> q_sf,
               TensorView indices, Optional<TensorView> scores, TensorView logits,
               Optional<TensorView> long_ws, bool enable_pdl) {
  const bool fp4 = q_sf.has_value();
  const int64_t row_bytes = fp4 ? kHeadDim / 2 : kHeadDim;
  if (fp4) {
    TVM_FFI_ICHECK_EQ(q.dtype().bits, 8) << "MXFP4 q is packed bytes";
    CHECK_INPUT_TYPE(q_sf.value(), dl_int32);
    CHECK_CONTIGUOUS(q_sf.value());
  } else {
    CHECK_INPUT_TYPE(q, dl_float8_e4m3fn);
  }
  CHECK_INPUT_TYPE(w, dl_float32);
  CHECK_INPUT_TYPE(row_len, dl_int32);
  CHECK_INPUT_TYPE(block_table, dl_int32);
  CHECK_INPUT_TYPE(kv, dl_uint8);
  CHECK_INPUT_TYPE(logits, dl_float32);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(w);
  CHECK_CONTIGUOUS(row_len);
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(2) == row_bytes) << "q must be [rows, H, row_bytes]";
  const int num_rows = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  TVM_FFI_ICHECK(num_heads == 32 || num_heads == 64) << "num_heads must be 32 or 64";
  TVM_FFI_ICHECK(w.ndim() == 2 && w.size(0) == num_rows && w.size(1) == num_heads)
      << "w must be [rows, H]";
  if (fp4) {
    TVM_FFI_ICHECK(q_sf.value().ndim() == 2 && q_sf.value().size(0) == num_rows &&
                   q_sf.value().size(1) == num_heads)
        << "q_sf must be [rows, H]";
  }
  TVM_FFI_ICHECK_EQ(row_len.numel(), num_rows) << "row_len must be [rows]";
  TVM_FFI_ICHECK(unit_rows == 1 || unit_rows == 2) << "units are of 1 or 2 rows";
  TVM_FFI_ICHECK_EQ(num_rows % unit_rows, 0) << "rows must be whole units";
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= 4096) << "need 1 <= top_k <= 4096";
  TVM_FFI_ICHECK(page_rows == 32 || page_rows == 64 || page_rows == 128 || page_rows == 256)
      << "page_rows must be 32, 64, 128 or 256";
  TVM_FFI_ICHECK(kv.ndim() == 2 && kv.stride(1) == 1)
      << "kv must be [num_pages, page_stride] bytes";
  const int64_t page_stride = kv.stride(0);
  TVM_FFI_ICHECK(page_stride >= page_rows * (row_bytes + 4) && page_stride % 16 == 0)
      << "a page holds page_rows rows and their scales, 16-byte aligned";
  float* scores_p = check_outputs(indices, scores, num_rows, top_k);
  if (num_rows == 0) return;
  const int num_units = num_rows / static_cast<int>(unit_rows);
  TVM_FFI_ICHECK(block_table.ndim() == 2 && block_table.stride(1) == 1)
      << "block_table must be [requests, max_blocks] with unit stride";
  TVM_FFI_ICHECK(units_per_req >= 1 && num_units % units_per_req == 0 &&
                 block_table.size(0) >= num_units / units_per_req)
      << "block_table must have a row per request";
  TVM_FFI_ICHECK(logits.ndim() == 2 && logits.size(0) >= num_rows && logits.stride(1) == 1 &&
                 logits.stride(0) % 4 == 0 && logits.size(1) >= max_context_len)
      << "logits must be [>= rows, >= max_context_len] float32, row stride a multiple of 4";
  const bool may_long = max_context_len > kRowSelKeys;
  if (may_long) {
    TVM_FFI_ICHECK(long_ws.has_value() && long_ws.value().dtype().bits == 32 &&
                   long_ws.value().numel() >= int64_t{num_rows} * kLongWords)
        << "long_ws must hold rows * " << kLongWords << " 32-bit words";
  }
  cudaStream_t stream = get_stream(q.device());
  const int S = stream_sm_count(stream);  // one wave
  const int max_tiles =
      std::max<int>(1, static_cast<int>((max_context_len + kBlockKV - 1) / kBlockKV));
  // The scan's split runs in 32 bits: units * max_tiles virtual tiles times
  // the CTAs, and max_tiles times a row's tiles.
  TVM_FFI_ICHECK(static_cast<uint64_t>(num_units) * max_tiles * S < (1ull << 32) &&
                 static_cast<uint64_t>(max_tiles) * max_tiles < (1ull << 32))
      << "decode rows * max_context_len too large for the split";

  ScanParams p{};
  p.row_len = ptr<const int32_t>(row_len);
  p.block_table = ptr<const int32_t>(block_table);
  p.bt_stride = static_cast<int>(block_table.stride(0));
  p.page_rows = static_cast<int>(page_rows);
  p.num_rows = num_rows;
  p.units_per_req = static_cast<int>(units_per_req);
  p.max_tiles = max_tiles;
  p.logits = ptr<float>(logits);
  p.logits_stride = logits.stride(0);
  p.long_ws = may_long ? ptr<uint32_t>(long_ws.value()) : nullptr;
  // Logits of 4 MB up to a quarter of L2 stay there for the selection: their
  // write-back would take the DRAM-bound scan's bandwidth. Smaller calls keep
  // the plain kernels.
  int l2 = 0;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, q.device().device_id),
                    cudaSuccess);
  const int64_t logits_bytes = int64_t{num_rows} * max_context_len * 4;
  const bool keep =
      may_long && logits_bytes >= (int64_t{4} << 20) && logits_bytes <= int64_t{l2} / 4;

  const void* qsf = fp4 ? q_sf.value().data_ptr() : nullptr;
  const int num_pages = static_cast<int>(kv.size(0));
  auto scan = [&](auto heads_tag, auto fp4_tag, auto unit_tag) {
    constexpr int H = decltype(heads_tag)::value;
    constexpr bool kF = decltype(fp4_tag)::value;
    constexpr int NN = decltype(unit_tag)::value;
    auto go = [&](auto long_tag, auto keep_tag) {
      constexpr bool kL = decltype(long_tag)::value;
      constexpr bool kK = kL && decltype(keep_tag)::value;
      using LS = ScanSmem<H, kF, NN, kL>;
      int st = kScanMaxStages;
      while (st > 2 && LS::bytes(st) > kSmemLimit) --st;
      p.k_stages = st;
      launch_scan<H, kF, NN>(logits_scan_kernel<H, kF, NN, kL, kK>, p, q.data_ptr(), w.data_ptr(),
                             kv.data_ptr(), qsf, num_pages, page_stride, S, enable_pdl, stream,
                             LS::bytes(st));
    };
    if (!may_long)
      go(std::false_type{}, std::false_type{});
    else if (keep)
      go(std::true_type{}, std::true_type{});
    else
      go(std::true_type{}, std::false_type{});
  };
  using I32 = std::integral_constant<int, 32>;
  using I64 = std::integral_constant<int, 64>;
  using U1 = std::integral_constant<int, 1>;
  using U2 = std::integral_constant<int, 2>;
  if (num_heads == 32) {
    if (fp4)
      unit_rows == 1 ? scan(I32{}, std::true_type{}, U1{}) : scan(I32{}, std::true_type{}, U2{});
    else
      unit_rows == 1 ? scan(I32{}, std::false_type{}, U1{}) : scan(I32{}, std::false_type{}, U2{});
  } else {
    if (fp4)
      unit_rows == 1 ? scan(I64{}, std::true_type{}, U1{}) : scan(I64{}, std::true_type{}, U2{});
    else
      unit_rows == 1 ? scan(I64{}, std::false_type{}, U1{}) : scan(I64{}, std::false_type{}, U2{});
  }
  SelParams sp{};
  sp.logits = p.logits;
  sp.stride = p.logits_stride;
  sp.row_len = p.row_len;
  sp.num_rows = num_rows;
  sp.top_k = static_cast<int>(top_k);
  sp.indices = ptr<int32_t>(indices);
  sp.scores = scores_p;
  sp.long_ws = p.long_ws;
  // A CTA a row, then (where a row may be long) one wave for the long rows.
  launch_ex(keep ? select_kernel<true> : select_kernel<false>, num_rows + (may_long ? S : 0),
            kSelThreads, 0, stream, enable_pdl, false, sp);
  check_launch("decode select");
}

int64_t DecodeLongWords() { return kLongWords; }
int64_t DecodeRowSelKeys() { return kRowSelKeys; }

}  // namespace flashinfer::sm120_dsa_indexer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::sm120_dsa_indexer::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_decode, flashinfer::sm120_dsa_indexer::RunDecode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_long_words, flashinfer::sm120_dsa_indexer::DecodeLongWords);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_row_sel_keys, flashinfer::sm120_dsa_indexer::DecodeRowSelKeys);
