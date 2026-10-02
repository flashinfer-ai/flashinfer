/*
 * Vendored sglang DeepSeek-V4 ragged top-k, wired as a FlashInfer JIT module.
 *
 * Device code is copied VERBATIM from sglang (Apache-2.0; provenance in
 * VENDORED.md next to this file, license text in licenses/LICENSE.sglang.txt):
 *   python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh  (ragged kernel +
 *   helpers; the paged/cluster/plan machinery is intentionally not vendored)
 *   python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh and
 *   python/sglang/kernels/jit/include/sgl_kernel/{*.cuh,*.h}  (the selection
 *   classes and support headers, vendored unmodified under ./sgl_kernel/)
 * upstream commit 7f27bf4708 (2026-08).  Only this host launcher (TVM-FFI
 * TensorView + cudaLaunchKernelEx w/ PDL) and topk_varlen_kernel below are
 * FlashInfer-specific.
 *
 * Purpose: apples-to-apples benchmarking of the primitives backends against
 * the exact sglang kernel under the same top_k_varlen contract (fp32 scores,
 * per-row lengths, local column indices out, -1 padded), including the paged
 * output (physical KV slots through a page table) that SGLang's DSA decode
 * path consumes -- see topk_varlen_kernel<kPDL, kPaged> below, which ports
 * upstream's staged transform pass onto the vendored selection classes.
 */

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

// sglang's utils.cuh defines its own CHECK_CUDA; FlashInfer's tvm_ffi_utils.h
// redefines it.  The sglang macro is not used below, so drop it.
#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#include "../tvm_ffi_utils.h"

namespace sglang {

namespace impl = device::topk;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

// upstream's bound: two 1024-thread CTAs per SM, i.e. 32 registers per thread.  Parts
// with 1536 resident threads per SM (SM86 / SM89 / SM120) can never co-schedule the
// second CTA, so there the cap only forces spills (absorbed by enable_smem_spilling);
// kept as upstream has it so the kernel stays a faithful reference.
#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)

struct TopKRaggedParams {
  float* __restrict__ scores;  // NOTE: may write
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ out_offsets;
  int32_t* __restrict__ topk_indices;
  int64_t score_stride;
  uint32_t topk;
};

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

/**
 * \brief Ragged (prefill) top-k: select inside a per-row window, emit indices
 * rebased onto the flattened KV.  (Verbatim from sglang topk_v2.cuh; see the
 * upstream file for the full commentary on the in-place masking of the <= 3
 * columns ahead of unaligned windows -- with row_starts == nullptr every
 * window starts at column 0 and no masking write occurs.)  The kernel reads
 * ``seq_len`` columns as given: a per-row length must not exceed the row width
 * (the top_k_varlen entry below clamps on device, this raw entry does not).
 */
template <bool kPDL>
TOPK_KERNEL void topk_ragged_kernel(const __grid_constant__ TopKRaggedParams params) {
  device::enable_smem_spilling();
  constexpr uint32_t kVecSize = impl::TopKStreaming::kVecSize;
  const auto bx = blockIdx.x;
  // issue all metadata prefetch ahead of time
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto offset = params.out_offsets[bx];
  const auto row_start = params.row_starts == nullptr ? 0u : params.row_starts[bx];
  const auto topk = params.topk;
  const auto out = params.topk_indices + bx * static_cast<int64_t>(topk);

  if (seq_len <= topk) {
    device::PDLWaitPrimary<kPDL>();
    for_each_item(topk, [&](uint32_t tx, uint32_t) {
      out[tx] = tx < seq_len ? static_cast<int32_t>(tx) + offset : -1;  // note: need offset
    });
    return;
  }

  const auto rem = row_start % kVecSize;
  const auto score = params.scores + bx * params.score_stride;
  if (rem != 0) {
    // The mask has to land after the indexer has retired
    // Otherwise it may be accidentally overwritten by DG upstream
    device::PDLWaitPrimary<kPDL>();
    static_assert(kVecSize <= kBlockSize, "not enough threads ");
    if (const auto tx = threadIdx.x; tx < rem) {
      score[row_start - rem + tx] = -std::numeric_limits<float>::max();
    }
  }

  const auto problem = impl::TopKProblem{
      .in = score + (row_start - rem),
      .out = out,
      .page_table = nullptr,  // unused
      .topk = topk,
      .seq_len = seq_len + rem,
      .page_bits = 1,  // unused
      .bias = offset - static_cast<int32_t>(rem),
  };
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= Register2::kMaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= Register4::kMaxSeqLen) {
    Register4::forward<kPDL>(problem, &smem);
  } else {
    Streaming::forward<kPDL>(problem, &smem);
  }
  // PDL trigger secondary at the end the block typically has no use, so ignore it
}

// ---------------------------------------------------------------------------
// FlashInfer-specific (NOT verbatim): the top_k_varlen contract with the
// per-row effective length derived IN-KERNEL from the request-level seq_lens.
// The Python dispatcher used to compute `(seq_lens - next_n + t + 1) // cr`
// and clamp it host-side; those two tiny device ops cost ~3.4 us per call as
// CUDA-graph nodes, doubling the kernel's own ~3.3 us on decode rows.  Every
// window starts at column 0 and emitted indices are local, so the verbatim
// kernel's row_starts / out_offsets machinery is not needed here.
// ---------------------------------------------------------------------------
struct TopKVarlenParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;  // (num_rows / next_n,) request-level
  int32_t* __restrict__ topk_indices;
  int64_t score_stride;
  uint32_t topk;
  int32_t next_n;
  int32_t compress_ratio;
  int32_t n_cols;
  // Paged output (kPaged instantiations only): one int32 page-table row per
  // request (row q = bx / next_n, `page_table_stride` ints apart); every
  // selected column c is emitted as page_table[q][c >> page_bits] << page_bits
  // | (c & mask), -1 padding unchanged -- upstream topk_v2.cuh's fused
  // page-table transform (the DSA decode contract).
  const int32_t* __restrict__ page_table;
  int64_t page_table_stride;
  uint32_t page_bits;  // log2(page_size)
};

/**
 * \brief top_k_varlen entry (FlashInfer-specific).  kPaged reproduces upstream
 * topk_v2.cuh's paged kernel structure verbatim in spirit: the selection emits
 * RAW indices into a shared-memory stage (`problem.out` redirected), then
 * `problem_transform` reads them back (two per thread) and writes the physical
 * slots to the global row in one coalesced pass -- the per-element page-table
 * gather stays out of the atomic-serialized scatter loop.  Rows with
 * seq_len <= topk take the trivial path: transformed positions then -1 pads.
 */
template <bool kPDL, bool kPaged>
TOPK_KERNEL void topk_varlen_kernel(const __grid_constant__ TopKVarlenParams params) {
  device::enable_smem_spilling();
  const auto bx = blockIdx.x;
  // Grouped-row convention: row r belongs to request q = r / next_n and sees
  // the first (seq_lens[q] - next_n + r % next_n + 1) tokens, in units of
  // compress_ratio.  Negative numerators (padded / evicted requests) and
  // lengths past the padded width clamp to [0, n_cols]; after the clamp the
  // truncating division here agrees with Python's flooring one.
  // The plain decode case (next_n == compress_ratio == 1) reads seq_lens[bx]
  // directly: the two runtime integer divisions below cost ~0.4 us of prologue
  // on a 3.3 us row (ncu: 123 vs 37 SASS instructions before the first
  // barrier), and they sit on the dependent path to the seq_len load.
  // PDL: like upstream's topk_ragged_kernel, the metadata (seq_lens, the page
  // table row) is prefetched BEFORE the grid-dependency wait so its latency
  // overlaps the producer's tail; the wait precedes the first scores read.
  // Contract (documented on backend="sglang"): with programmatic dependent
  // launch, the kernel immediately preceding this one must not write them.
  int32_t len;
  if (params.next_n == 1 && params.compress_ratio == 1) {
    len = params.seq_lens[bx];
  } else {
    const int32_t q = static_cast<int32_t>(bx) / params.next_n;
    const int32_t t = static_cast<int32_t>(bx) - q * params.next_n;
    len = (params.seq_lens[q] - params.next_n + t + 1) / params.compress_ratio;
  }
  len = len < 0 ? 0 : (len > params.n_cols ? params.n_cols : len);
  if constexpr (kPaged) {
    // the table may be narrower than the padded width (the API contract: a row
    // past the covered pages is clamped to the covered prefix, so no page the
    // table lacks is ever read)
    const int64_t covered = params.page_table_stride << params.page_bits;
    if (len > covered) len = static_cast<int32_t>(covered);
  }
  const auto seq_len = static_cast<uint32_t>(len);
  const auto topk = params.topk;
  const auto out = params.topk_indices + bx * static_cast<int64_t>(topk);
  // paged: the request's page-table row (the request index q = bx / next_n)
  const int32_t* page_row = nullptr;
  if constexpr (kPaged) {
    const int32_t q =
        params.next_n == 1 ? static_cast<int32_t>(bx) : static_cast<int32_t>(bx) / params.next_n;
    page_row = params.page_table + static_cast<int64_t>(q) * params.page_table_stride;
  }

  if (seq_len <= topk) {
    device::PDLWaitPrimary<kPDL>();
    if constexpr (kPaged) {
      // upstream trivial_transform: positions 0..seq_len-1 mapped, then -1
      const impl::TopKProblem trivial{
          .in = nullptr,
          .out = out,
          .page_table = page_row,
          .topk = topk,
          .seq_len = seq_len,
          .page_bits = params.page_bits,
          .bias = 0,
      };
      for_each_item(topk, [&](uint32_t tx, uint32_t) {
        trivial.transform_output(tx, tx < seq_len ? static_cast<int32_t>(tx) : -1);
      });
    } else {
      for_each_item(topk, [&](uint32_t tx, uint32_t) {
        out[tx] = tx < seq_len ? static_cast<int32_t>(tx) : -1;
      });
    }
    return;
  }

  // paged: raw indices are staged here (upstream `__shared__ topk_indices[kMaxTopK]`)
  __shared__ int32_t topk_stage[kPaged ? kMaxTopK : 1];
  impl::TopKProblem problem{
      .in = params.scores + bx * params.score_stride,
      .out = kPaged ? topk_stage : out,
      .page_table = page_row,
      .topk = topk,
      .seq_len = seq_len,
      .page_bits = kPaged ? params.page_bits : 1u,
      .bias = 0,
  };
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= Register2::kMaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= Register4::kMaxSeqLen) {
    Register4::forward<kPDL>(problem, &smem);
  } else {
    Streaming::forward<kPDL>(problem, &smem);
  }
  if constexpr (kPaged) {
    // upstream problem_transform: the staged raw indices (written by whichever
    // threads emitted them) are read back two per thread after a block barrier,
    // then stored as physical slots to the global row in one coalesced pass
    __syncthreads();
    int32_t source_index[kMaxTopK / kBlockSize];
    for_each_item(topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
    problem.out = out;
    for_each_item(topk,
                  [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
  }
}

}  // namespace sglang

using tvm::ffi::TensorView;

void sglang_dsv4_topk_ragged(TensorView scores, TensorView lengths, TensorView out_offsets,
                             TensorView out_indices, bool enable_pdl) {
  CHECK_INPUT(scores);
  CHECK_INPUT(lengths);
  CHECK_INPUT(out_offsets);
  CHECK_INPUT(out_indices);
  CHECK_DEVICE(lengths, scores);
  CHECK_DEVICE(out_offsets, scores);
  CHECK_DEVICE(out_indices, scores);
  CHECK_DIM(2, scores);       // (num_rows, N)
  CHECK_DIM(1, lengths);      // (num_rows,) int32, per-ROW effective lengths
  CHECK_DIM(1, out_offsets);  // (num_rows,) int32, added to emitted indices
  CHECK_DIM(2, out_indices);  // (num_rows, top_k)
  TVM_FFI_ICHECK_EQ(scores.dtype(), dl_float32) << "sglang dsv4 topk supports fp32 scores only";
  TVM_FFI_ICHECK_EQ(lengths.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(out_offsets.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(out_indices.dtype(), dl_int32);

  const int64_t num_rows = scores.size(0);
  const int64_t n_cols = scores.size(1);
  const int64_t top_k = out_indices.size(1);
  TVM_FFI_ICHECK_EQ(lengths.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(out_offsets.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(out_indices.size(0), num_rows);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= sglang::kMaxTopK)
      << "top_k must be in (0, " << sglang::kMaxTopK << "]";
  TVM_FFI_ICHECK_EQ(n_cols % 4, 0)
      << "score row stride must be a multiple of 4 (16-byte vectorized load)";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0)
      << "scores must be 16-byte aligned (vectorized loads)";
  if (num_rows == 0) return;

  cudaSetDevice(scores.device().device_id);
  const cudaStream_t stream = get_stream(scores.device());

  const sglang::TopKRaggedParams params{
      static_cast<float*>(scores.data_ptr()),
      static_cast<const int32_t*>(lengths.data_ptr()),
      nullptr,  // row_starts: every window starts at column 0
      static_cast<const int32_t*>(out_offsets.data_ptr()),
      static_cast<int32_t*>(out_indices.data_ptr()),
      n_cols,
      static_cast<uint32_t>(top_k),
  };

  cudaLaunchConfig_t config;
  config.gridDim = static_cast<unsigned int>(num_rows);
  config.blockDim = sglang::kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaError_t status;
  if (enable_pdl) {
    status = cudaLaunchKernelEx(&config, sglang::topk_ragged_kernel<true>, params);
  } else {
    status = cudaLaunchKernelEx(&config, sglang::topk_ragged_kernel<false>, params);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sglang_dsv4_topk_ragged launch failed: " << cudaGetErrorString(status);
}

// top_k_varlen entry: request-level seq_lens in, per-row lengths derived on device.
// page_size > 0 selects the paged output (page_table: int32 (num_requests,
// max_pages) covering each request's valid length -- a row past the covered
// pages is clamped to them -- page_size a power of two); page_size == 0 emits
// raw columns and ignores page_table (callers pass any int32 tensor).
void sglang_dsv4_topk_varlen(TensorView scores, TensorView seq_lens, TensorView out_indices,
                             int64_t next_n, int64_t compress_ratio, TensorView page_table,
                             int64_t page_size, bool enable_pdl) {
  CHECK_INPUT(scores);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(out_indices);
  CHECK_DEVICE(seq_lens, scores);
  CHECK_DEVICE(out_indices, scores);
  CHECK_DIM(2, scores);       // (num_rows, N)
  CHECK_DIM(1, seq_lens);     // (num_rows / next_n,) int32, request-level
  CHECK_DIM(2, out_indices);  // (num_rows, top_k)
  TVM_FFI_ICHECK_EQ(scores.dtype(), dl_float32) << "sglang dsv4 topk supports fp32 scores only";
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(out_indices.dtype(), dl_int32);

  const int64_t num_rows = scores.size(0);
  const int64_t n_cols = scores.size(1);
  const int64_t top_k = out_indices.size(1);
  TVM_FFI_ICHECK(next_n >= 1 && compress_ratio >= 1) << "next_n and compress_ratio must be >= 1";
  TVM_FFI_ICHECK_EQ(seq_lens.size(0) * next_n, num_rows)
      << "num_rows must equal seq_lens.numel() * next_n";
  TVM_FFI_ICHECK_EQ(out_indices.size(0), num_rows);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= sglang::kMaxTopK)
      << "top_k must be in (0, " << sglang::kMaxTopK << "]";
  TVM_FFI_ICHECK_EQ(n_cols % 4, 0)
      << "score row stride must be a multiple of 4 (16-byte vectorized load)";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0)
      << "scores must be 16-byte aligned (vectorized loads)";
  TVM_FFI_ICHECK(n_cols <= std::numeric_limits<int32_t>::max());
  const bool paged = page_size > 0;
  const int32_t* page_table_ptr = nullptr;
  int64_t page_table_stride = 0;
  uint32_t page_bits = 0;
  if (paged) {
    CHECK_INPUT(page_table);
    CHECK_DEVICE(page_table, scores);
    CHECK_DIM(2, page_table);  // (num_requests, max_pages)
    TVM_FFI_ICHECK_EQ(page_table.dtype(), dl_int32);
    TVM_FFI_ICHECK_EQ(page_table.size(0), seq_lens.size(0)) << "one page-table row per request";
    TVM_FFI_ICHECK((page_size & (page_size - 1)) == 0) << "page_size must be a power of two";
    TVM_FFI_ICHECK(page_table.size(1) >= 1) << "page_table must have at least one page";
    TVM_FFI_ICHECK(page_table.size(1) * page_size <= std::numeric_limits<int32_t>::max())
        << "max_pages * page_size must fit int32 (physical slots are int32)";
    page_table_ptr = static_cast<const int32_t*>(page_table.data_ptr());
    page_table_stride = page_table.size(1);  // contiguous rows
    page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
  }
  if (num_rows == 0) return;

  cudaSetDevice(scores.device().device_id);
  const cudaStream_t stream = get_stream(scores.device());

  const sglang::TopKVarlenParams params{
      static_cast<const float*>(scores.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<int32_t*>(out_indices.data_ptr()),
      n_cols,
      static_cast<uint32_t>(top_k),
      static_cast<int32_t>(next_n),
      static_cast<int32_t>(compress_ratio),
      static_cast<int32_t>(n_cols),
      page_table_ptr,
      page_table_stride,
      page_bits,
  };

  cudaLaunchConfig_t config;
  config.gridDim = static_cast<unsigned int>(num_rows);
  config.blockDim = sglang::kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaError_t status;
  if (paged) {
    if (enable_pdl) {
      status = cudaLaunchKernelEx(&config, sglang::topk_varlen_kernel<true, true>, params);
    } else {
      status = cudaLaunchKernelEx(&config, sglang::topk_varlen_kernel<false, true>, params);
    }
  } else {
    if (enable_pdl) {
      status = cudaLaunchKernelEx(&config, sglang::topk_varlen_kernel<true, false>, params);
    } else {
      status = cudaLaunchKernelEx(&config, sglang::topk_varlen_kernel<false, false>, params);
    }
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sglang_dsv4_topk_varlen launch failed: " << cudaGetErrorString(status);
}
