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
 *
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, MIT licensed;
 * see LICENSE.DeepGEMM at the root of this package.
 */

// Launcher of one MOSCA MegaMoE kernel configuration (experimental, SM100).
//
// One JIT module per launch configuration: the generated header
// `mosca_megamoe_kernel_config.h` (written by the Python loader into this
// module's generated-source directory) fixes every template argument of
// `deep_gemm::sm100_fp8_fp4_mega_moe_impl` (the MOSCA kernel header
// `sm100_fp8_fp4_mega_moe.cuh` in this directory) and the MOSCA feature macros
// (DGX_SMALL_T, DGX_SPARSE_COUNT, DGX_BCAST) before that header is included,
// so this translation unit instantiates exactly one kernel.
//
// `launch` reproduces the host side of upstream
// `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` for this backend's
// contract (FP8 x FP4, no shared experts, BF16 combine, no x scales): the
// symmetric-buffer views come from the same `layout::MegaMoEBuffer` the
// kernel derives, every tensor map is encoded from those views and the weight
// tensors per call (tensor maps travel by value as __grid_constant__
// parameters, so a launch is self-contained and CUDA-graph capturable), and
// the kernel is launched on `num_sms` CTAs in 2-CTA clusters with the
// configuration's dynamic shared memory, no PDL and no cooperative attribute
// (upstream's DeepJIT launch defaults). With DGX_BCAST, this rank's first row
// in the gathered output rides in the high bits of the token-count argument
// (the kernel ABI is unchanged).

// clang-format off
// Order matters: the generated config header defines the template arguments and
// the MOSCA feature macros, and the common shim defines DG_UNIFIED_ASSERT; both
// must precede the kernel header and every vendored deep_gemm header.
#include "mosca_megamoe_kernel_config.h"
#include "common/deep_gemm_host_shim.h"
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>

#include <cstdint>

#include "megamoe/sm100_fp8_fp4_mega_moe.cuh"
#include "tvm_ffi_utils.h"

namespace mosca_megamoe_kernel {

using deep_gemm::MmaKind;
namespace layout = deep_gemm::layout;

constexpr uint32_t kNumMaxTokensPerRank = MOSCA_NUM_MAX_TOKENS_PER_RANK;
constexpr uint32_t kHidden = MOSCA_HIDDEN;
constexpr uint32_t kIntermediateHidden = MOSCA_INTERMEDIATE_HIDDEN;
constexpr uint32_t kNumExperts = MOSCA_NUM_EXPERTS;
constexpr uint32_t kNumTopk = MOSCA_NUM_TOPK;
constexpr uint32_t kNumRanks = MOSCA_NUM_RANKS;
constexpr uint32_t kNumSMs = MOSCA_NUM_SMS;
constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
constexpr uint32_t kBlockM = MOSCA_BLOCK_M;
constexpr uint32_t kBlockN = MOSCA_BLOCK_N;
constexpr uint32_t kBlockK = MOSCA_BLOCK_K;
constexpr uint32_t kLoadBlockM = kBlockM / 2;
constexpr uint32_t kLoadBlockN = kBlockN;
constexpr uint32_t kStoreBlockM = MOSCA_STORE_BLOCK_M;
constexpr uint32_t kSFBlockM = MOSCA_SF_BLOCK_M;
constexpr uint32_t kSFBlockN = MOSCA_SF_BLOCK_N;
constexpr uint32_t kGranK = MOSCA_GRAN_K;
constexpr uint32_t kNumRingTokens = MOSCA_NUM_RING_TOKENS;
constexpr uint32_t kNumSFRingTokens = MOSCA_NUM_SF_RING_TOKENS;
constexpr uint32_t kNumStages = MOSCA_NUM_STAGES;
constexpr uint32_t kSmemSize = MOSCA_SMEM_SIZE;
constexpr uint32_t kNumBytesPerPull = MOSCA_NUM_BYTES_PER_PULL;
constexpr uint32_t kNumDispatchThreads = MOSCA_NUM_DISPATCH_THREADS;
constexpr uint32_t kNumNonEpilogueThreads = MOSCA_NUM_NON_EPILOGUE_THREADS;
constexpr uint32_t kNumEpilogueThreads = MOSCA_NUM_EPILOGUE_THREADS;
constexpr int kSwizzleActsMode = MOSCA_SWIZZLE_ACTS_MODE;
constexpr int kSwizzleWeightsMode = MOSCA_SWIZZLE_WEIGHTS_MODE;
constexpr MmaKind kMmaKind = MmaKind::MXFP8FP4;
static_assert(kNumExperts % kNumRanks == 0, "experts must split evenly over ranks");
static_assert(kNumRingTokens % kBlockM == 0, "ring must hold whole M blocks");

// The single kernel this module instantiates (template order of the kernel header).
constexpr auto kKernel = &deep_gemm::sm100_fp8_fp4_mega_moe_impl<
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, /*kNumSharedExperts=*/0u,
    kNumTopk, kBlockM, kBlockN, kBlockK, kStoreBlockM, kSFBlockM, kSFBlockN, kMmaKind,
    kNumRingTokens, kNumSFRingTokens, kNumStages, kNumBytesPerPull, kNumDispatchThreads,
    kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, MOSCA_ACTIVATION_CLAMP,
    /*kSwiGLUAlpha=*/0.0f, /*kUseSitu=*/false, MOSCA_FAST_MATH, /*kUseXScales=*/false,
    /*kWithL1Alphas=*/false, /*kWithL2Alphas=*/false, /*kWithL2ActScales=*/false,
    /*kUseFp8Combine=*/false, cutlass::detail::float_e2m1_unpacksmem_t>;

// ---------------------------------------------------------------------------------------------
// Tensor maps (upstream `make_tma_2d_desc` / `make_tma_sf_desc`, runtime_utils.hpp).
// ---------------------------------------------------------------------------------------------
enum class Elem { kUInt8, kInt32, kPackedFP4Unpacked };

CUtensorMapSwizzle swizzle_of(int mode) {
  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TVM_FFI_THROW(ValueError) << "unsupported swizzle mode " << mode;
  }
  return CU_TENSOR_MAP_SWIZZLE_NONE;
}

// `ptr` points at a 2-D region whose rows are `gmem_outer_stride` elements of `elem_size` bytes
// apart. For packed FP4 the element is one 4-bit value stored two per byte (`elem_size` stays 1, as
// upstream).
CUtensorMap make_tma_2d_desc(void* ptr, Elem elem, int gmem_inner_dim, int gmem_outer_dim,
                             int smem_inner_dim, int smem_outer_dim, int gmem_outer_stride,
                             int swizzle_mode) {
  const int elem_size = elem == Elem::kInt32 ? 4 : 1;
  const uint64_t num_gmem_outer_stride_bytes = static_cast<uint64_t>(gmem_outer_stride) * elem_size;
  TVM_FFI_CHECK(reinterpret_cast<std::uintptr_t>(ptr) % 16u == 0u, ValueError)
      << "tensor map base must be 16-byte aligned";
  TVM_FFI_CHECK(num_gmem_outer_stride_bytes % 16u == 0u, ValueError)
      << "tensor map row stride must be a multiple of 16 bytes, got "
      << num_gmem_outer_stride_bytes;
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (elem == Elem::kInt32) dtype = CU_TENSOR_MAP_DATA_TYPE_INT32;
  if (elem == Elem::kPackedFP4Unpacked) {
    // `.b4x16_p64`: the inner dim must be a multiple of 64 bytes (128 values).
    TVM_FFI_CHECK(gmem_inner_dim % 128 == 0, ValueError)
        << "packed FP4 inner dim must be a multiple of 128, got " << gmem_inner_dim;
    dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
  }
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(num_gmem_outer_stride_bytes)};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult status =
      cuTensorMapEncodeTiled(&tensor_map, dtype, 2, ptr, gmem_dims, gmem_strides, smem_dims,
                             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_of(swizzle_mode),
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    TVM_FFI_THROW(RuntimeError) << "cuTensorMapEncodeTiled failed: "
                                << (message ? message : "unknown error");
  }
  return tensor_map;
}

int tma_aligned_size(int x, int element_size) {
  constexpr int kNumTMAAlignmentBytes = 16;
  const int alignment = kNumTMAAlignmentBytes / element_size;
  return (x + alignment - 1) / alignment * alignment;
}

// MN-major packed UE8M0 scale factors (int32 words, 4 K-blocks per word), `num_groups` groups
// stacked in K.
CUtensorMap make_tma_sf_desc(void* ptr, int shape_mn, int shape_k, int block_mn, int num_groups,
                             int smem_outer_dim) {
  shape_mn = tma_aligned_size(shape_mn, 4);
  const int num_k_words = (shape_k + kGranK * 4 - 1) / (kGranK * 4);
  return make_tma_2d_desc(ptr, Elem::kInt32, shape_mn, num_k_words * num_groups, block_mn,
                          smem_outer_dim, shape_mn,
                          /*swizzle_mode=*/0);
}

// Tensor maps are encoded from raw pointers and the kernel runs on y's device,
// so a tensor on another device would fault (or read peer memory) at launch.
void check_same_device(const TensorView& t, const DLDevice& dev, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == dev.device_type && t.device().device_id == dev.device_id,
                ValueError)
      << name << " must be on the same device as y";
}

void check_weight(const TensorView& w, int64_t d1, int64_t d2, const char* name) {
  check_cuda_tensor(w, name);
  check_contiguous(w, name);
  TVM_FFI_CHECK(w.dtype() == dl_int8 || w.dtype() == dl_uint8, ValueError)
      << name << " must be packed FP4 bytes (int8/uint8)";
  TVM_FFI_CHECK(
      w.ndim() == 3 && w.size(0) == kNumExpertsPerRank && w.size(1) == d1 && w.size(2) == d2,
      ValueError)
      << name << " must be [" << kNumExpertsPerRank << ", " << d1 << ", " << d2 << "]";
}

void check_weight_sf(const TensorView& sf, int64_t mn, int64_t k, const char* name) {
  check_cuda_tensor(sf, name);
  TVM_FFI_CHECK(sf.dtype() == dl_int32, ValueError) << name << " must be int32 (packed UE8M0)";
  const int64_t num_k_words = (k + kGranK * 4 - 1) / (kGranK * 4);
  TVM_FFI_CHECK(sf.ndim() == 3 && sf.size(0) == kNumExpertsPerRank && sf.size(1) == mn &&
                    sf.size(2) == num_k_words,
                ValueError)
      << name << " must be [" << kNumExpertsPerRank << ", " << mn << ", " << num_k_words
      << "] (MN-major)";
  TVM_FFI_CHECK(sf.stride(1) == 1 && sf.stride(2) == tma_aligned_size(static_cast<int>(mn), 4),
                ValueError)
      << name << " must be MN-major with a TMA-aligned MN stride";
  // The tensor map folds the experts into one outer dimension with one stride
  // (as upstream's check_sf_layout: stride(-3) == stride(-1) * size(-1)), so
  // the expert groups must be packed back to back; a sliced or padded group
  // would make every later expert read another expert's scales.
  TVM_FFI_CHECK(sf.stride(0) == sf.stride(2) * sf.size(2), ValueError)
      << name << " expert groups must be packed back to back (stride(0) == stride(2) * size(2))";
}

// Launch one MegaMoE call.
//
//   y               bf16 [num_tokens, hidden]: this rank's output rows (num_tokens = y.size(0) is
//   the
//                   per-rank token count, as upstream). The kernel writes it in every mode.
//   l1/l2 weights   transformed weights and MN-major packed scale factors (``weights.py``).
//   sym_buffer      this rank's symmetric buffer (uint8/int8, at least the layout's total bytes),
//                   its inputs already staged for `num_tokens` rows.
//   peer_buffers    every rank's base address of the same symmetric buffer (slot r = rank r).
//   gather_row_offset  this rank's first row in the gathered output (DGX_BCAST builds only; -1
//   otherwise). stream          the CUDA stream to launch on
//   (``torch.cuda.current_stream().cuda_stream``).
void Launch(TensorView y, TensorView l1_weights, TensorView l1_weights_sf, TensorView l2_weights,
            TensorView l2_weights_sf, TensorView sym_buffer, tvm::ffi::Shape peer_buffers,
            int64_t rank_idx, int64_t gather_row_offset, int64_t stream_ptr) {
  check_cuda_tensor(y, "y");
  check_dtype(y, dl_bfloat16, "y");
  check_contiguous(y, "y");
  const DLDevice dev = y.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_CHECK(y.ndim() == 2 && y.size(1) == kHidden, ValueError)
      << "y must be [num_tokens, " << kHidden << "]";
  const int64_t num_tokens = y.size(0);
  TVM_FFI_CHECK(num_tokens >= 0 && num_tokens <= kNumMaxTokensPerRank, ValueError)
      << "num_tokens must be in [0, " << kNumMaxTokensPerRank << "], got " << num_tokens;
  check_weight(l1_weights, 2 * kIntermediateHidden, kHidden / 2, "l1_weights");
  check_weight(l2_weights, kHidden, kIntermediateHidden / 2, "l2_weights");
  check_weight_sf(l1_weights_sf, 2 * kIntermediateHidden, kHidden, "l1_weights_sf");
  check_weight_sf(l2_weights_sf, kHidden, kIntermediateHidden, "l2_weights_sf");
  check_cuda_tensor(sym_buffer, "sym_buffer");
  check_contiguous(sym_buffer, "sym_buffer");
  check_same_device(l1_weights, dev, "l1_weights");
  check_same_device(l1_weights_sf, dev, "l1_weights_sf");
  check_same_device(l2_weights, dev, "l2_weights");
  check_same_device(l2_weights_sf, dev, "l2_weights_sf");
  check_same_device(sym_buffer, dev, "sym_buffer");
  TVM_FFI_CHECK(static_cast<int64_t>(peer_buffers.size()) == kNumRanks, ValueError)
      << "peer_buffers must carry " << kNumRanks << " addresses, got " << peer_buffers.size();
  TVM_FFI_CHECK(rank_idx >= 0 && rank_idx < kNumRanks, ValueError)
      << "rank_idx out of range: " << rank_idx;
  TVM_FFI_CHECK(static_cast<uint64_t>(peer_buffers[rank_idx]) ==
                    reinterpret_cast<uint64_t>(sym_buffer.data_ptr()),
                ValueError)
      << "peer_buffers[rank_idx] must be this rank's sym_buffer address";

  // Symmetric-buffer views: the same layout the kernel derives from its template arguments.
  uint8_t* base = static_cast<uint8_t*>(sym_buffer.data_ptr());
  const auto mega_buffer =
      layout::MegaMoEBuffer(nullptr, kHidden, kIntermediateHidden, kNumRanks, kNumExperts,
                            kNumMaxTokensPerRank, kNumTopk, kNumRingTokens, kNumSFRingTokens,
                            kMmaKind, /*num_shared_experts=*/0u, /*use_fp8_combine=*/false);
  int64_t required_bytes = mega_buffer.get_num_bytes();
#ifdef DGX_BCAST
  required_bytes = (required_bytes + 255) / 256 * 256 +
                   static_cast<int64_t>(kNumRanks) * kNumMaxTokensPerRank * kHidden * 2;
#endif
  TVM_FFI_CHECK(
      sym_buffer.numel() * static_cast<int64_t>(sym_buffer.dtype().bits / 8) >= required_bytes,
      ValueError)
      << "sym_buffer holds fewer than the " << required_bytes << " bytes this configuration needs";
  const auto at = [&](const layout::Buffer& b) { return base + reinterpret_cast<int64_t>(b.base); };
  void* l1_acts = at(mega_buffer.l1_token_buffer);
  void* l1_acts_sf = at(mega_buffer.l1_sf_buffer);
  void* l2_acts = at(mega_buffer.l2_token_buffer);
  void* l2_acts_sf = at(mega_buffer.l2_sf_buffer);
  const int l1_acts_stride = kHidden;  // FP8 bytes per ring row
  const int l2_acts_stride = kIntermediateHidden;

  // Tensor maps (upstream launcher, MXFP8FP4: weights stay packed FP4 in gmem, unpacked to 8 bit in
  // smem).
  const int block_k_inner = kBlockK;  // 8-bit activations
  const int sf_smem_outer_dim = kBlockK / (kGranK * 4);
  const auto tensor_map_l1_acts =
      make_tma_2d_desc(l1_acts, Elem::kUInt8, kHidden, kNumRingTokens, block_k_inner, kLoadBlockM,
                       l1_acts_stride, kSwizzleActsMode);
  const auto tensor_map_l1_acts_sf =
      make_tma_sf_desc(l1_acts_sf, kNumSFRingTokens, kHidden, kSFBlockM, 1, sf_smem_outer_dim);
  const auto tensor_map_l1_weights =
      make_tma_2d_desc(l1_weights.data_ptr(), Elem::kPackedFP4Unpacked, kHidden,
                       kNumExpertsPerRank * kIntermediateHidden * 2, block_k_inner, kLoadBlockN,
                       static_cast<int>(l1_weights.stride(1)), kSwizzleWeightsMode);
  const auto tensor_map_l1_weights_sf =
      make_tma_sf_desc(l1_weights_sf.data_ptr(), kIntermediateHidden * 2, kHidden, kBlockN,
                       kNumExpertsPerRank, sf_smem_outer_dim);
  // L1 output and L2 activations are the same tensor; post-SwiGLU output has half the N width.
  const int l1_out_block_n_bytes = kBlockN / 2;
  const auto tensor_map_l1_output =
      make_tma_2d_desc(l2_acts, Elem::kUInt8, kIntermediateHidden, kNumRingTokens,
                       l1_out_block_n_bytes, kStoreBlockM, l2_acts_stride, l1_out_block_n_bytes);
  const auto tensor_map_l2_acts =
      make_tma_2d_desc(l2_acts, Elem::kUInt8, kIntermediateHidden, kNumRingTokens, block_k_inner,
                       kLoadBlockM, l2_acts_stride, kSwizzleActsMode);
  const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
      l2_acts_sf, kNumSFRingTokens, kIntermediateHidden, kSFBlockM, 1, sf_smem_outer_dim);
  const auto tensor_map_l2_weights =
      make_tma_2d_desc(l2_weights.data_ptr(), Elem::kPackedFP4Unpacked, kIntermediateHidden,
                       kNumExpertsPerRank * kHidden, block_k_inner, kLoadBlockN,
                       static_cast<int>(l2_weights.stride(1)), kSwizzleWeightsMode);
  const auto tensor_map_l2_weights_sf =
      make_tma_sf_desc(l2_weights_sf.data_ptr(), kHidden, kIntermediateHidden, kBlockN,
                       kNumExpertsPerRank, sf_smem_outer_dim);

  // Packed token count (DGX_BCAST: this rank's first gathered row in bits [14, 32)).
  uint32_t num_tokens_arg = static_cast<uint32_t>(num_tokens);
#ifdef DGX_BCAST
  TVM_FFI_CHECK(
      gather_row_offset >= 0 && gather_row_offset + num_tokens <= kNumRanks * kNumMaxTokensPerRank,
      ValueError)
      << "gather_row_offset must place this rank's rows inside the gathered output, got "
      << gather_row_offset;
  TVM_FFI_CHECK(num_tokens < (1 << 14) && gather_row_offset < (1 << 18), ValueError)
      << "gathered launch packs num_tokens into 14 bits and the row offset into 18 bits";
  num_tokens_arg |= static_cast<uint32_t>(gather_row_offset) << 14;
#else
  TVM_FFI_CHECK(gather_row_offset < 0, ValueError)
      << "this kernel was not built with the in-kernel all-gather";
#endif

  std::vector<int64_t> peers(kNumRanks);
  for (uint32_t i = 0; i < kNumRanks; ++i) peers[i] = peer_buffers[i];
  const auto sym = layout::SymBuffer<>(peers, static_cast<uint32_t>(rank_idx));
  void* y_ptr = y.data_ptr();
  int* stats_ptr = nullptr;
  const float* null_alphas = nullptr;

  // Per launch, as upstream: the attribute is per device, and setting it is cheap.
  {
    const cudaError_t status =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(kKernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=" << kSmemSize
        << ") failed: " << cudaGetErrorString(status);
  }

  void* kargs[] = {&y_ptr, &stats_ptr, &null_alphas, &null_alphas, &null_alphas, &num_tokens_arg,
                   const_cast<layout::SymBuffer<>*>(&sym),
                   const_cast<CUtensorMap*>(&tensor_map_l1_acts),
                   const_cast<CUtensorMap*>(&tensor_map_l1_acts_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l1_weights),
                   const_cast<CUtensorMap*>(&tensor_map_l1_weights_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l1_output),
                   const_cast<CUtensorMap*>(&tensor_map_l2_acts),
                   const_cast<CUtensorMap*>(&tensor_map_l2_acts_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l2_weights),
                   const_cast<CUtensorMap*>(&tensor_map_l2_weights_sf),
                   // No shared experts: the shared tensor maps alias the routed ones (as upstream).
                   const_cast<CUtensorMap*>(&tensor_map_l1_acts),
                   const_cast<CUtensorMap*>(&tensor_map_l1_acts_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l1_weights),
                   const_cast<CUtensorMap*>(&tensor_map_l1_weights_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l1_output),
                   const_cast<CUtensorMap*>(&tensor_map_l2_acts),
                   const_cast<CUtensorMap*>(&tensor_map_l2_acts_sf),
                   const_cast<CUtensorMap*>(&tensor_map_l2_weights),
                   const_cast<CUtensorMap*>(&tensor_map_l2_weights_sf)};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSMs, 1, 1);
  config.blockDim = dim3(kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1);
  config.dynamicSmemBytes = kSmemSize;
  config.stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status =
      cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kKernel), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(MOSCA MegaMoE) failed: " << cudaGetErrorString(status);
}

// Static facts of this module, for the loader's consistency check.
tvm::ffi::Array<int64_t> Describe() {
  int64_t gather = 0, small_t = 0, sparse_count = 0;
#ifdef DGX_BCAST
  gather = 1;
#endif
#ifdef DGX_SMALL_T
  small_t = 1;
#endif
#ifdef DGX_SPARSE_COUNT
  sparse_count = 1;
#endif
  return tvm::ffi::Array<int64_t>(std::vector<int64_t>{
      kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, kNumRanks, kNumSMs,
      kBlockM, kStoreBlockM, kNumStages, kSmemSize, gather, small_t, sparse_count});
}

// How many 2-CTA clusters of this configuration (dynamic shared memory, threads,
// cluster shape) the current device can hold resident at once. The kernel is
// persistent and synchronizes its whole grid, so the loader requires
// num_sms / 2 of them: fewer would leave CTAs waiting for CTAs that never start.
int64_t MaxActiveClusters() {
  cudaError_t status = cudaFuncSetAttribute(reinterpret_cast<const void*>(kKernel),
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=" << kSmemSize
      << ") failed: " << cudaGetErrorString(status);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSMs, 1, 1);
  config.blockDim = dim3(kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1);
  config.dynamicSmemBytes = kSmemSize;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  int clusters = 0;
  status =
      cudaOccupancyMaxActiveClusters(&clusters, reinterpret_cast<const void*>(kKernel), &config);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaOccupancyMaxActiveClusters failed: " << cudaGetErrorString(status);
  return clusters;
}

}  // namespace mosca_megamoe_kernel

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, mosca_megamoe_kernel::Launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(describe, mosca_megamoe_kernel::Describe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_active_clusters, mosca_megamoe_kernel::MaxActiveClusters);
