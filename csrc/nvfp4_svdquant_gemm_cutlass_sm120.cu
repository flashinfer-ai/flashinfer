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
// TVM-FFI binding for the SM120 SVDQuant fused NVFP4 GEMM:
//   nvfp4_svdquant_gemm : out = alpha * (A @ Bᵀ) + (D @ L1ᵀ) [+ bias], the residual NVFP4 GEMM
//                         fused with the rank-32 LoRA-up via a 2nd bf16 mma.sync into the same
//                         fp32 register accumulator (custom CUTLASS SM120 block-scaled
//                         collective). The LoRA rank is a build parameter of the module
//                         (SVDQ_SM120_LORA_RANK, default 32); see nvfp4_svdquant_gemm_lora_rank.
//   nvfp4_svdquant_gemm_tactic_num : length of the flattened runtime tactic table (compiled
//                                    kernels x scheduler variants) for the autotuner.
//   nvfp4_svdquant_gemm_can_implement : per-tactic host-side feasibility query (includes the
//                                       shape-dependent Split-K bound).
//   nvfp4_svdquant_gemm_fallback_tactic : first feasible runtime row for tactic=-1.
//   nvfp4_svdquant_gemm_workspace_size : per-tactic workspace query for pre-provisioning.
//   nvfp4_svdquant_gemm_tactic_row : decoded runtime-tactic row (packed int64).

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
#include "flashinfer/gemm/svdquant_sm120_prefix_route.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer {
namespace gemm {

namespace sd = svdquant_sm120;

// Dispatch a callable template over the compiled kernel configs by kernel id
// (the decoded runtime-tactic row selects the kernel; scheduler parameters ride
// along in the row).
#define SVDQ_SM120_DISPATCH(kernel_id, EXPR)              \
  switch (static_cast<sd::KernelShapeSm120>(kernel_id)) { \
    case sd::KernelShapeSm120::k128x128x128: {            \
      using C = sd::Tactic128x128x128Config;              \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x128Swap: {        \
      using C = sd::Tactic128x128x128SwapConfig;          \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x128Sk: {          \
      using C = sd::Tactic128x128x128SkConfig;            \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x128SwapSk: {      \
      using C = sd::Tactic128x128x128SwapSkConfig;        \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x256: {            \
      using C = sd::Tactic128x128x256Config;              \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x256Swap: {        \
      using C = sd::Tactic128x128x256SwapConfig;          \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x256Sk: {          \
      using C = sd::Tactic128x128x256SkConfig;            \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x128x256SwapSk: {      \
      using C = sd::Tactic128x128x256SwapSkConfig;        \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x128x128: {            \
      using C = sd::Tactic256x128x128Config;              \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x128x128Swap: {        \
      using C = sd::Tactic256x128x128SwapConfig;          \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x128x128Sk: {          \
      using C = sd::Tactic256x128x128SkConfig;            \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x128x128SwapSk: {      \
      using C = sd::Tactic256x128x128SwapSkConfig;        \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x256x128: {            \
      using C = sd::Tactic128x256x128Config;              \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x256x128Swap: {        \
      using C = sd::Tactic128x256x128SwapConfig;          \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x256x128Sk: {          \
      using C = sd::Tactic128x256x128SkConfig;            \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x256x128SwapSk: {      \
      using C = sd::Tactic128x256x128SwapSkConfig;        \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x64x128Swap: {         \
      using C = sd::Tactic128x64x128SwapConfig;           \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x64x128SwapSk: {       \
      using C = sd::Tactic128x64x128SwapSkConfig;         \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x64x256Swap: {         \
      using C = sd::Tactic128x64x256SwapConfig;           \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x32x128Swap: {         \
      using C = sd::Tactic128x32x128SwapConfig;           \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k64x128x128: {             \
      using C = sd::Tactic64x128x128Config;               \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k64x128x256: {             \
      using C = sd::Tactic64x128x256Config;               \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x64x256SwapSk: {       \
      using C = sd::Tactic128x64x256SwapSkConfig;         \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x64x256SwapStatic: {   \
      using C = sd::Tactic128x64x256SwapStaticConfig;     \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x32x128SwapStatic: {   \
      using C = sd::Tactic128x32x128SwapStaticConfig;     \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k64x128x128Static: {       \
      using C = sd::Tactic64x128x128StaticConfig;         \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x32x256Swap: {         \
      using C = sd::Tactic128x32x256SwapConfig;           \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k128x32x256SwapStatic: {   \
      using C = sd::Tactic128x32x256SwapStaticConfig;     \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k64x32x256SwapStatic: {    \
      using C = sd::Tactic64x32x256SwapStaticConfig;      \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x128x128SwapStatic: {  \
      using C = sd::Tactic256x128x128SwapStaticConfig;    \
      EXPR;                                               \
    }                                                     \
    case sd::KernelShapeSm120::k256x64x128SwapStatic: {   \
      using C = sd::Tactic256x64x128SwapStaticConfig;     \
      EXPR;                                               \
    }                                                     \
  }                                                       \
  throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): invalid kernel shape");

size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic) {
  auto const rt = sd::decode_runtime_tactic(tactic);
  SVDQ_SM120_DISPATCH(rt.kernel_id, return sd::workspace_size_for_tactic<C>(rt, m, n, k))
}

size_t nvfp4_svdquant_gemm_shared_storage_size(int tactic) {
  auto const rt = sd::decode_runtime_tactic(tactic);
  SVDQ_SM120_DISPATCH(rt.kernel_id, return sd::shared_storage_size_for_tactic<C>())
}

bool nvfp4_svdquant_gemm_can_implement(int m, int n, int k, int lora_rank, int tactic) {
  auto const rt = sd::decode_runtime_tactic(tactic);
  SVDQ_SM120_DISPATCH(rt.kernel_id, return sd::can_implement_tactic<C>(rt, m, n, k, lora_rank))
}

int nvfp4_svdquant_gemm_fallback_tactic(int m, int n, int k, int lora_rank) {
  for (int tactic = 0; tactic < sd::kNvfp4SvdquantGemmSm120NumTactics; ++tactic) {
    if (nvfp4_svdquant_gemm_can_implement(m, n, k, lora_rank, tactic)) {
      return tactic;
    }
  }
  return -1;
}

// Fused residual NVFP4 GEMM + rank-32 LoRA-up: D @ L1ᵀ via the 2nd bf16 mma.sync in the custom
// collective. 1/alpha folded into L1 so the epilogue yields alpha*residual + D@L1ᵀ + bias.
void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa,
                             void const* sfb, float const* alpha, void const* D, void const* L1,
                             void const* bias, int m, int n, int k, int lora_rank, char* ws,
                             size_t wsBytes, cudaStream_t stream, int tactic, bool enable_pdl) {
  auto const rt = sd::decode_runtime_tactic(tactic);
  // The production entry never supplies the inline-down pointers, so every config
  // selected by kCompileInlineDownSm120 runs the variant that compiles the dead
  // path out. The selection is per config because compiling it out regresses the
  // Stream-K and static-scheduler siblings; see the measurement table next to
  // SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST.
  SVDQ_SM120_DISPATCH(
      rt.kernel_id,
      if constexpr (sd::kCompileInlineDownSm120<C>) {
        return sd::run_tactic<C>(rt, out, A, B, sfa, sfb, alpha, D, L1, bias, nullptr, nullptr, m,
                                 n, k, lora_rank, ws, wsBytes, stream, enable_pdl);
      } else {
        return sd::run_tactic_no_inline_down<C>(rt, out, A, B, sfa, sfb, alpha, D, L1, bias, m, n,
                                                k, lora_rank, ws, wsBytes, stream, enable_pdl);
      })
}

void nvfp4_svdquant_gemm_run_inline_down(void* out, void const* A, void const* B, void const* sfa,
                                         void const* sfb, float const* alpha, void const* D,
                                         void const* L1, void const* bias,
                                         void const* inline_down_x, void const* inline_down_l2t,
                                         int m, int n, int k, int lora_rank, char* ws,
                                         size_t wsBytes, cudaStream_t stream, int tactic,
                                         bool enable_pdl) {
  auto const rt = sd::decode_runtime_tactic(tactic);
  if (rt.kernel_id != static_cast<int>(sd::KernelShapeSm120::k256x128x128SwapStatic)) {
    throw std::invalid_argument("nvfp4_svdquant_gemm (sm120): inline LoRA-down requires tactic 81");
  }
  using C = sd::Tactic256x128x128SwapStaticConfig;
  return sd::run_tactic<C>(rt, out, A, B, sfa, sfb, alpha, D, L1, bias, inline_down_x,
                           inline_down_l2t, m, n, k, lora_rank, ws, wsBytes, stream, enable_pdl);
}

}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

constexpr auto FLOAT4_E2M1X2 = dl_uint8;  // packed e2m1
constexpr auto SF_DTYPE = dl_uint8;       // swizzled ue4m3 block scales

// ceil(rows / 128) * 128 * ceil(cols / 4) * 4, the 128x4-swizzled block-scale layout size.
inline int64_t swizzled_sf_size(int64_t rows, int64_t sfCols) {
  auto pad = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  return pad(rows, 128) * pad(sfCols, 4);
}

cudaError_t configure_case7_xq_persisting_l2(int device_index, cudaStream_t stream, void* base,
                                             std::size_t window_extent_bytes,
                                             std::size_t persisting_budget_bytes) {
  int persisting_l2_bytes = 0;
  cudaError_t status = cudaDeviceGetAttribute(&persisting_l2_bytes,
                                              cudaDevAttrMaxPersistingL2CacheSize, device_index);
  if (status != cudaSuccess) {
    return status;
  }
  int access_window_bytes = 0;
  status = cudaDeviceGetAttribute(&access_window_bytes, cudaDevAttrMaxAccessPolicyWindowSize,
                                  device_index);
  if (status != cudaSuccess) {
    return status;
  }
  if (persisting_l2_bytes <= 0 || access_window_bytes <= 0 || window_extent_bytes == 0 ||
      persisting_budget_bytes == 0) {
    return cudaSuccess;
  }
  status = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                              static_cast<std::size_t>(persisting_l2_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  std::size_t const window_bytes =
      window_extent_bytes < static_cast<std::size_t>(access_window_bytes)
          ? window_extent_bytes
          : static_cast<std::size_t>(access_window_bytes);
  std::size_t const budget_bytes =
      persisting_budget_bytes < static_cast<std::size_t>(persisting_l2_bytes)
          ? persisting_budget_bytes
          : static_cast<std::size_t>(persisting_l2_bytes);
  float const hit_ratio = window_bytes <= budget_bytes
                              ? 1.0f
                              : static_cast<float>(budget_bytes) / static_cast<float>(window_bytes);
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr = base;
  attribute.accessPolicyWindow.num_bytes = window_bytes;
  attribute.accessPolicyWindow.hitRatio = hit_ratio;
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attribute);
}

cudaError_t clear_xq_persisting_l2(cudaStream_t stream) {
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.num_bytes = 0;
  return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attribute);
}

cudaError_t query_stream_capture(cudaStream_t stream, bool& is_capturing) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t const status = cudaStreamIsCapturing(stream, &capture_status);
  if (status == cudaSuccess) {
    is_capturing = capture_status != cudaStreamCaptureStatusNone;
  }
  return status;
}

struct LoraDownCublasLtState {
  static constexpr int kMaxHeuristicResults = 32;
  static constexpr std::size_t kMaxCachedPlans = 8;

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t l2_layout = nullptr;
  cublasLtMatrixLayout_t x_layout = nullptr;
  cublasLtMatrixLayout_t down_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::array<cublasLtMatmulHeuristicResult_t, kMaxHeuristicResults> heuristics{};
  int heuristic_index = 0;
  int device_id = -1;
  int m = 0;
  int k = 0;
  void const* persisting_l2t_ptr = nullptr;
  cudaStream_t persisting_l2t_stream = nullptr;
  bool initialized = false;

  LoraDownCublasLtState() = default;
  LoraDownCublasLtState(LoraDownCublasLtState const&) = delete;
  LoraDownCublasLtState& operator=(LoraDownCublasLtState const&) = delete;

  ~LoraDownCublasLtState() { reset(); }

  void reset() noexcept {
    if (preference != nullptr) {
      static_cast<void>(cublasLtMatmulPreferenceDestroy(preference));
      preference = nullptr;
    }
    if (down_layout != nullptr) {
      static_cast<void>(cublasLtMatrixLayoutDestroy(down_layout));
      down_layout = nullptr;
    }
    if (x_layout != nullptr) {
      static_cast<void>(cublasLtMatrixLayoutDestroy(x_layout));
      x_layout = nullptr;
    }
    if (l2_layout != nullptr) {
      static_cast<void>(cublasLtMatrixLayoutDestroy(l2_layout));
      l2_layout = nullptr;
    }
    if (operation != nullptr) {
      static_cast<void>(cublasLtMatmulDescDestroy(operation));
      operation = nullptr;
    }
    if (handle != nullptr) {
      static_cast<void>(cublasLtDestroy(handle));
      handle = nullptr;
    }
    heuristics = {};
    heuristic_index = 0;
    device_id = -1;
    m = 0;
    k = 0;
    persisting_l2t_ptr = nullptr;
    persisting_l2t_stream = nullptr;
    initialized = false;
  }
};

struct LoraDownCublasLtInitGuard {
  LoraDownCublasLtState& state;

  ~LoraDownCublasLtInitGuard() {
    if (!state.initialized) {
      state.reset();
    }
  }
};

LoraDownCublasLtState& lora_down_cublaslt_state(int device_id, int m, int k) {
  static thread_local std::array<LoraDownCublasLtState, LoraDownCublasLtState::kMaxCachedPlans>
      states{};
  for (auto& state : states) {
    if (state.initialized && state.device_id == device_id && state.m == m && state.k == k) {
      return state;
    }
  }
  for (auto& state : states) {
    if (state.initialized) {
      continue;
    }
    state.reset();
    LoraDownCublasLtInitGuard init_guard{state};
    constexpr int kRank = 32;
    constexpr std::size_t kMaxWorkspaceBytes = 32 * 1024 * 1024;
    int current_device = -1;
    cudaError_t cuda_status = cudaGetDevice(&current_device);
    TVM_FFI_ICHECK_EQ(cuda_status, cudaSuccess)
        << "failed to query CUDA device for SM120 LoRA-down";
    TVM_FFI_ICHECK_EQ(current_device, device_id)
        << "SM120 LoRA-down initialized on the wrong CUDA device";
    cublasStatus_t status = cublasLtCreate(&state.handle);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt handle for SM120 LoRA-down";
    status = cublasLtMatmulDescCreate(&state.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt operation for SM120 LoRA-down";
    status = cublasLtMatrixLayoutCreate(&state.l2_layout, CUDA_R_16BF, kRank, k, kRank);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt L2 layout for SM120 LoRA-down";
    status = cublasLtMatrixLayoutCreate(&state.x_layout, CUDA_R_16BF, k, m, k);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt X layout for SM120 LoRA-down";
    status = cublasLtMatrixLayoutCreate(&state.down_layout, CUDA_R_16BF, kRank, m, kRank);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt output layout for SM120 LoRA-down";
    status = cublasLtMatmulPreferenceCreate(&state.preference);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to create cuBLASLt preference for SM120 LoRA-down";
    status = cublasLtMatmulPreferenceSetAttribute(state.preference,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &kMaxWorkspaceBytes, sizeof(kMaxWorkspaceBytes));
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to set cuBLASLt workspace limit for SM120 LoRA-down";
    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        state.handle, state.operation, state.l2_layout, state.x_layout, state.down_layout,
        state.down_layout, state.preference, LoraDownCublasLtState::kMaxHeuristicResults,
        state.heuristics.data(), &returned_results);
    TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
        << "failed to query cuBLASLt heuristic for SM120 LoRA-down";
    TVM_FFI_ICHECK_GT(returned_results, 0) << "cuBLASLt returned no heuristic for SM120 LoRA-down";
    // CTK 13.2 returns eight candidates for the admitted M537 and M1935
    // problems. Cold-L2 sweeps select the candidates below
    // independently for each exact (M, K); keep every other shape on
    // cuBLASLt's first-ranked result.
    int heuristic_index = 0;
    if (m == 537) {
      heuristic_index = (k == 5120 || k == 7168 || k == 14336) ? 2 : 0;
    } else if (m == 1935) {
      heuristic_index = k == 5120 ? 2 : k == 5376 ? 4 : k == 7168 ? 5 : 0;
    }
    TVM_FFI_ICHECK(heuristic_index >= 0 && heuristic_index < returned_results)
        << "SM120 LoRA-down heuristic index must be in [0, " << returned_results << "), got "
        << heuristic_index;
    TVM_FFI_ICHECK_EQ(state.heuristics[heuristic_index].state, CUBLAS_STATUS_SUCCESS)
        << "cuBLASLt heuristic is invalid for SM120 LoRA-down";
    state.heuristic_index = heuristic_index;
    state.device_id = device_id;
    state.m = m;
    state.k = k;
    state.initialized = true;
    return state;
  }
  throw std::runtime_error("SM120 LoRA-down cuBLASLt plan cache is full");
}

void run_lora_down_cublaslt(TensorView x, TensorView l2t_smoothed, TensorView down,
                            TensorView workspace_buffer, cudaStream_t stream) {
  int const m = static_cast<int>(x.size(0));
  int const k = static_cast<int>(x.size(1));
  int const rank = static_cast<int>(l2t_smoothed.size(1));
  TVM_FFI_ICHECK(l2t_smoothed.ndim() == 2 && l2t_smoothed.size(0) == k)
      << "l2t_smoothed must have shape [k, rank]";
  TVM_FFI_ICHECK(down.ndim() == 2 && down.size(0) == m && down.size(1) == rank)
      << "down must have shape [m, rank]";

  LoraDownCublasLtState& state = lora_down_cublaslt_state(x.device().device_id, m, k);
  if (m == 537 && (state.persisting_l2t_ptr != l2t_smoothed.data_ptr() ||
                   state.persisting_l2t_stream != stream)) {
    bool is_capturing = false;
    cudaError_t cuda_status = query_stream_capture(stream, is_capturing);
    TVM_FFI_ICHECK_EQ(cuda_status, cudaSuccess)
        << "failed to query M537 LoRA-down stream capture status: "
        << cudaGetErrorString(cuda_status);
    if (!is_capturing) {
      std::size_t const l2t_bytes =
          static_cast<std::size_t>(l2t_smoothed.numel()) * get_element_size(l2t_smoothed);
      cuda_status = configure_case7_xq_persisting_l2(x.device().device_id, stream,
                                                     l2t_smoothed.data_ptr(), l2t_bytes, l2t_bytes);
      TVM_FFI_ICHECK_EQ(cuda_status, cudaSuccess)
          << "failed to configure M537 L2T persisting-L2 window: "
          << cudaGetErrorString(cuda_status);
      state.persisting_l2t_ptr = l2t_smoothed.data_ptr();
      state.persisting_l2t_stream = stream;
    }
  }
  auto const& heuristic = state.heuristics[state.heuristic_index];
  std::size_t const workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(workspace_bytes, heuristic.workspaceSize)
      << "workspace buffer is too small for cuBLASLt SM120 LoRA-down";
  float const one = 1.0f;
  float const zero = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      state.handle, state.operation, &one, l2t_smoothed.data_ptr(), state.l2_layout, x.data_ptr(),
      state.x_layout, &zero, down.data_ptr(), state.down_layout, down.data_ptr(), state.down_layout,
      &heuristic.algo, workspace_buffer.data_ptr(), workspace_bytes, stream);
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cuBLASLt BF16 LoRA-down launch failed";
}

}  // namespace

#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
void nvfp4_quantize_smooth(TensorView x, TensorView pqs, TensorView global_scale, TensorView xq,
                           TensorView sf, bool enable_pdl);

void nvfp4_quantize_smooth_lora_down_sm120(TensorView x, TensorView pqs, TensorView global_scale,
                                           TensorView l2t_smoothed, TensorView xq, TensorView sf,
                                           TensorView down);
// Geometry-selecting producer entry. Separate symbol so the seven-argument FFI
// export keeps its arity; see nvfp4_smooth_quantize_sm100.cu.
void nvfp4_quantize_smooth_lora_down_geometry_sm120(TensorView x, TensorView pqs,
                                                    TensorView global_scale,
                                                    TensorView l2t_smoothed, TensorView xq,
                                                    TensorView sf, TensorView down,
                                                    int geometry_variant);
// Producer entry that takes the geometry as numbers rather than as an index
// into a ladder both languages have to agree on; see nvfp4_smooth_quantize_sm100.cu.
void nvfp4_quantize_smooth_lora_down_dyn_sm120_impl(TensorView x, TensorView pqs,
                                                    TensorView global_scale,
                                                    TensorView l2t_smoothed, TensorView xq,
                                                    TensorView sf, TensorView down, int family,
                                                    int tiling0, int tiling1, int tiling2,
                                                    int address_policy, bool signal_launch);

void nvfp4_quantize_smooth_lora_down_overlap_sm120(TensorView x, TensorView pqs,
                                                   TensorView global_scale, TensorView l2t_smoothed,
                                                   TensorView xq, TensorView sf, TensorView down);

void nvfp4_quantize_smooth_lora_down_cublaslt_sm120(TensorView x, TensorView pqs,
                                                    TensorView global_scale,
                                                    TensorView l2t_smoothed, TensorView xq,
                                                    TensorView sf, TensorView down,
                                                    TensorView workspace_buffer) {
  CHECK_INPUT_AND_TYPE(l2t_smoothed, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(down, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace_buffer, dl_uint8);
  CHECK_DEVICE(l2t_smoothed, x);
  CHECK_DEVICE(down, x);
  CHECK_DEVICE(workspace_buffer, x);
  // No shape list. This prefix is a cuBLASLt GEMM plus the same quantizer every
  // other producer uses, so what it can run is what they can run; the nine
  // shapes named here were the ones it had been measured on, which is a
  // different question and belongs to the autotuner.
  TVM_FFI_ICHECK(x.ndim() == 2) << "x must be 2-D [m, k]";

  cudaStream_t stream = get_stream(x.device());
  // Two shapes used to swap these calls, to keep XQ/SF hot for the residual GEMM
  // that follows. That was worth measuring and it was measured -- on two values
  // of M, on one card. It is an ordering hint, not a correctness property, and
  // naming shapes inside a kernel entry is what this backend is removing
  // everywhere else. If the ordering is worth having it should come back as a
  // tactic axis the autotuner can rank on the card in front of it.
  nvfp4_quantize_smooth(x, pqs, global_scale, xq, sf, false);
  run_lora_down_cublaslt(x, l2t_smoothed, down, workspace_buffer, stream);
}
#endif

// out = alpha * (A @ Bᵀ) + (D @ L1ᵀ) [+ bias]. a = quant(x_hat) [m, k/2] uint8, b [n, k/2] uint8
// (packed e2m1), a_sf/b_sf swizzled UE4M3 block scales, alpha f32[1] (residual dequant scale).
// D [m, 32] = x_hat @ L2ᵀ (bf16) and L1 [n, 32] = svdquant_lora_b / alpha (bf16; 1/alpha folded so
// the epilogue out = alpha * acc yields the LoRA). One module serves one LoRA rank;
// any other rank is rejected. out [m, n] bf16, allocated by the caller.
void nvfp4_svdquant_gemm_impl(TensorView a, TensorView b, TensorView a_sf, TensorView b_sf,
                              TensorView alpha, TensorView d, TensorView l1,
                              Optional<TensorView> const& bias, TensorView out,
                              TensorView workspace_buffer, int64_t tactic, bool enable_pdl,
                              void const* inline_down_x, void const* inline_down_l2t) {
  CHECK_INPUT_AND_TYPE(a, FLOAT4_E2M1X2);
  CHECK_INPUT_AND_TYPE(b, FLOAT4_E2M1X2);
  CHECK_INPUT_AND_TYPE(a_sf, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(b_sf, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(d, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(l1, dl_bfloat16);
  CHECK_DEVICE(b, a);
  CHECK_DEVICE(a_sf, a);
  CHECK_DEVICE(b_sf, a);
  CHECK_DEVICE(alpha, a);
  CHECK_DEVICE(d, a);
  CHECK_DEVICE(l1, a);
  CHECK_DEVICE(out, a);
  CHECK_DEVICE(workspace_buffer, a);

  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be [n, k/2]";
  int64_t const m = a.size(0);
  int64_t const kPacked = a.size(1);
  int64_t const k = kPacked * 2;
  int64_t const n = b.size(0);
  TVM_FFI_ICHECK_EQ(b.size(1), kPacked) << "a and b inner dimensions mismatch";
  TVM_FFI_ICHECK(n > 0 && k > 0) << "n and k must be positive";
  TVM_FFI_ICHECK(n % 32 == 0 && k % 32 == 0) << "n and k must be divisible by 32";
  TVM_FFI_ICHECK_GE(alpha.numel(), 1) << "alpha must contain at least one element";
  TVM_FFI_ICHECK_GE(a_sf.numel(), swizzled_sf_size(m, k / 16))
      << "a_sf is smaller than the required swizzled scale layout";
  TVM_FFI_ICHECK_GE(b_sf.numel(), swizzled_sf_size(n, k / 16))
      << "b_sf is smaller than the required swizzled scale layout";
  TVM_FFI_ICHECK(d.ndim() == 2 && d.size(0) == m)
      << "d must have shape [m, r] (rank-r LoRA-down output)";
  int64_t const loraRank = d.size(1);
  // Legality only. Which ranks this module can actually stage is a per-tactic
  // property -- a module built for rank 64 still serves rank 32 on its K128
  // tactics -- so the decision belongs to can_implement, not to a module-wide
  // equality test. Infeasibility surfaces below as "no feasible fallback
  // tactic", or from run_tactic_impl when an explicit tactic is passed.
  TVM_FFI_ICHECK(loraRank > 0 && loraRank % 32 == 0)
      << "the LoRA rank (d.shape[1]) must be a positive multiple of 32, got " << loraRank;
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(d.data_ptr()) % 16, 0)
      << "d must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(l1.ndim() == 2 && l1.size(0) == n && l1.size(1) == loraRank)
      << "l1 must have shape [n, r] with the same LoRA rank as d (pre-divided by alpha)";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(l1.data_ptr()) % 16, 0)
      << "l1 must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == m && out.size(1) == n)
      << "out must have shape [m, n]";
  // The epilogue writes a dense row-major buffer; a strided view (e.g. a transpose)
  // would be silently rearranged and could clobber unrelated storage.
  CHECK_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), bfloat16_code)
      << "nvfp4_svdquant_gemm currently supports bf16 output only";

  void const* biasPtr = nullptr;
  if (bias.has_value()) {
    auto const& biasTensor = bias.value();
    CHECK_INPUT_AND_TYPE(biasTensor, dl_bfloat16);
    CHECK_DEVICE(biasTensor, a);
    TVM_FFI_ICHECK(biasTensor.ndim() == 1 && biasTensor.size(0) == n) << "bias must have shape [n]";
    biasPtr = biasTensor.data_ptr();
  }

  TVM_FFI_ICHECK(tactic >= -1 &&
                 tactic < flashinfer::gemm::svdquant_sm120::kNvfp4SvdquantGemmSm120NumTactics)
      << "invalid NVFP4 SVDQuant SM120 tactic: " << tactic;

  // Empty batch: out is [0, n], nothing to compute.
  if (m == 0) return;

  int const tacticId = tactic < 0 ? flashinfer::gemm::nvfp4_svdquant_gemm_fallback_tactic(
                                        static_cast<int>(m), static_cast<int>(n),
                                        static_cast<int>(k), static_cast<int>(loraRank))
                                  : static_cast<int>(tactic);
  TVM_FFI_ICHECK_GE(tacticId, 0) << "no feasible NVFP4 SVDQuant SM120 fallback tactic for m=" << m
                                 << ", n=" << n << ", k=" << k << ", rank=" << loraRank
                                 << " (this module was built for LoRA rank "
                                 << flashinfer::gemm::svdquant_sm120::kSvdqSm120ModuleLoRaRank
                                 << "; a rank above what a tile can stage has no tactic here)";

  size_t const requiredWorkspaceBytes = flashinfer::gemm::nvfp4_svdquant_gemm_workspace_size(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), tacticId);
  auto stream = get_stream(a.device());

  // Never allocate here: the caller pre-provisions the workspace (the Python
  // wrapper sizes it to the maximum over candidate tactics via
  // nvfp4_svdquant_gemm_workspace_size), so this path stays safe under
  // CUDA-graph capture, where an implicit allocation would corrupt the graph.
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(provided_workspace_size, static_cast<int64_t>(requiredWorkspaceBytes))
      << "workspace buffer too small for tactic " << tacticId << ": need " << requiredWorkspaceBytes
      << " bytes, got " << provided_workspace_size
      << " (query nvfp4_svdquant_gemm_workspace_size and pre-provision before launch)";
  if (inline_down_x != nullptr && inline_down_l2t != nullptr) {
    flashinfer::gemm::nvfp4_svdquant_gemm_run_inline_down(
        out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(),
        static_cast<float const*>(alpha.data_ptr()), d.data_ptr(), l1.data_ptr(), biasPtr,
        inline_down_x, inline_down_l2t, static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(loraRank),
        reinterpret_cast<char*>(workspace_buffer.data_ptr()), requiredWorkspaceBytes, stream,
        tacticId, enable_pdl);
  } else {
    flashinfer::gemm::nvfp4_svdquant_gemm_run(
        out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(),
        static_cast<float const*>(alpha.data_ptr()), d.data_ptr(), l1.data_ptr(), biasPtr,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(loraRank),
        reinterpret_cast<char*>(workspace_buffer.data_ptr()), requiredWorkspaceBytes, stream,
        tacticId, enable_pdl);
  }
}

void nvfp4_svdquant_gemm(TensorView a, TensorView b, TensorView a_sf, TensorView b_sf,
                         TensorView alpha, TensorView d, TensorView l1,
                         Optional<TensorView> const& bias, TensorView out,
                         TensorView workspace_buffer, int64_t tactic, bool enable_pdl) {
  nvfp4_svdquant_gemm_impl(a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer, tactic,
                           enable_pdl, nullptr, nullptr);
}

#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
// Both the L2T layout and the producer kernel are selected by the producer half
// of the runtime tactic.
void dispatch_svdquant_producer_sm120(TensorView x, TensorView pqs, TensorView global_scale,
                                      TensorView l2t_smoothed, TensorView xq, TensorView a_sf,
                                      TensorView down, TensorView workspace_buffer,
                                      int producer_family = -1, int tiling0 = 0, int tiling1 = 0,
                                      int tiling2 = 0, int address_policy = 0,
                                      bool enable_pdl = false) {
  if (producer_family == 3) {
    nvfp4_quantize_smooth_lora_down_cublaslt_sm120(x, pqs, global_scale, l2t_smoothed, xq, a_sf,
                                                   down, workspace_buffer);
    return;
  }
  if (producer_family >= 0) {
    nvfp4_quantize_smooth_lora_down_dyn_sm120_impl(
        x, pqs, global_scale, l2t_smoothed, xq, a_sf, down, producer_family, tiling0, tiling1,
        tiling2, address_policy, producer_family == 4 && enable_pdl);
    return;
  }
  nvfp4_quantize_smooth_lora_down_geometry_sm120(x, pqs, global_scale, l2t_smoothed, xq, a_sf, down,
                                                 0);
}

void nvfp4_svdquant_linear_sm120(TensorView x, TensorView pqs, TensorView global_scale,
                                 TensorView l2t_smoothed, TensorView b, TensorView b_sf,
                                 TensorView alpha, TensorView l1, Optional<TensorView> const& bias,
                                 TensorView xq, TensorView a_sf, TensorView down, TensorView out,
                                 TensorView workspace_buffer, int64_t tactic, bool enable_pdl,
                                 int64_t producer_family, int64_t tiling0, int64_t tiling1,
                                 int64_t tiling2, int64_t address_policy) {
  // Was: `x.size(0) == 27280 && x.size(1) == 14336`. One shape, one card, wired
  // into the entry point as a literal. Same objection as every other table this
  // backend has shed -- a persisting-L2 window that helps on one part is not
  // known to help on another, and nothing here can tell. Off for everyone until
  // it is something the tuner selects.
  bool const use_case7_persisting_l2 = false;
  cudaStream_t const stream = get_stream(x.device());
  std::size_t const sf_bytes = static_cast<std::size_t>(a_sf.numel()) * get_element_size(a_sf);
  int max_persisting_l2_bytes = 0;
  bool configure_case7_persisting_l2 = false;
  if (use_case7_persisting_l2) {
    bool is_capturing = false;
    cudaError_t status = query_stream_capture(stream, is_capturing);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to query case-7 stream capture status: " << cudaGetErrorString(status);
    configure_case7_persisting_l2 = !is_capturing;
  }
  if (configure_case7_persisting_l2) {
    cudaError_t status = cudaDeviceGetAttribute(
        &max_persisting_l2_bytes, cudaDevAttrMaxPersistingL2CacheSize, x.device().device_id);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to query case-7 persisting-L2 capacity: " << cudaGetErrorString(status);
    std::size_t const sf_budget = sf_bytes < static_cast<std::size_t>(max_persisting_l2_bytes)
                                      ? sf_bytes
                                      : static_cast<std::size_t>(max_persisting_l2_bytes);
    status = configure_case7_xq_persisting_l2(x.device().device_id, stream, a_sf.data_ptr(),
                                              sf_bytes, sf_budget);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to configure case-7 SF persisting-L2 window: " << cudaGetErrorString(status);
  }
  // The tactic is a packed pair: the producer half selects the prefix and the
  // consumer half is the K3 row. Python resolved the same producer half to pick
  // the L2T layout before this call, so layout and dispatch cannot disagree.
  namespace prefix_route = flashinfer::gemm::svdquant_sm120_prefix_route;
  int const k3_row = prefix_route::k3_row_of(tactic);
  dispatch_svdquant_producer_sm120(
      x, pqs, global_scale, l2t_smoothed, xq, a_sf, down, workspace_buffer,
      static_cast<int>(producer_family), static_cast<int>(tiling0), static_cast<int>(tiling1),
      static_cast<int>(tiling2), static_cast<int>(address_policy), enable_pdl);
  if (configure_case7_persisting_l2) {
    std::size_t const max_budget = static_cast<std::size_t>(max_persisting_l2_bytes);
    std::size_t const xq_budget = max_budget > sf_bytes ? max_budget - sf_bytes : 0;
    cudaError_t const status = configure_case7_xq_persisting_l2(
        x.device().device_id, stream, xq.data_ptr(), xq_budget, xq_budget);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to configure case-7 XQ persisting-L2 window: " << cudaGetErrorString(status);
  }
  nvfp4_svdquant_gemm_impl(xq, b, a_sf, b_sf, alpha, down, l1, bias, out, workspace_buffer, k3_row,
                           enable_pdl, nullptr, nullptr);
  if (configure_case7_persisting_l2) {
    cudaError_t const status = clear_xq_persisting_l2(stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to clear case-7 XQ persisting-L2 window: " << cudaGetErrorString(status);
  }
}

#endif

int64_t nvfp4_svdquant_gemm_tactic_num() {
  return flashinfer::gemm::svdquant_sm120::kNvfp4SvdquantGemmSm120NumTactics;
}

// The LoRA rank this module was built for. Callers use it to tell "this backend
// cannot do rank r" apart from "no tactic fits this shape", and to key a tuned
// cache, since a different rank is a different module.
int64_t nvfp4_svdquant_gemm_lora_rank() {
  return flashinfer::gemm::svdquant_sm120::kSvdqSm120ModuleLoRaRank;
}

bool nvfp4_svdquant_gemm_can_implement(int64_t m, int64_t n, int64_t k, int64_t lora_rank,
                                       int64_t tactic) {
  return flashinfer::gemm::nvfp4_svdquant_gemm_can_implement(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(lora_rank),
      static_cast<int>(tactic));
}

int64_t nvfp4_svdquant_gemm_fallback_tactic(int64_t m, int64_t n, int64_t k, int64_t lora_rank) {
  return flashinfer::gemm::nvfp4_svdquant_gemm_fallback_tactic(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(lora_rank));
}

int64_t nvfp4_svdquant_gemm_shared_storage_size(int64_t tactic) {
  return static_cast<int64_t>(
      flashinfer::gemm::nvfp4_svdquant_gemm_shared_storage_size(static_cast<int>(tactic)));
}

int64_t nvfp4_svdquant_gemm_workspace_size(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  return static_cast<int64_t>(flashinfer::gemm::nvfp4_svdquant_gemm_workspace_size(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(tactic)));
}

// Decoded runtime-tactic row for host-side introspection (manifest, tests),
// packed into one int64: kernel_id | splits<<8 | raster<<16 | swizzle<<24 |
// streamk<<32, with raster 0=Heuristic 1=AlongM 2=AlongN.
int64_t nvfp4_svdquant_gemm_tactic_row(int64_t tactic) {
  auto const rt = flashinfer::gemm::svdquant_sm120::decode_runtime_tactic(static_cast<int>(tactic));
  int64_t const streamk =
      flashinfer::gemm::svdquant_sm120::kernel_is_streamk_sm120(rt.kernel_id) ? 1 : 0;
  return static_cast<int64_t>(rt.kernel_id) | (static_cast<int64_t>(rt.splits) << 8) |
         (static_cast<int64_t>(rt.raster) << 16) | (static_cast<int64_t>(rt.swizzle) << 24) |
         (streamk << 32);
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm, torch_ext::nvfp4_svdquant_gemm);
#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_quantize_smooth_lora_down_cublaslt_sm120,
                              torch_ext::nvfp4_quantize_smooth_lora_down_cublaslt_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_linear_sm120, torch_ext::nvfp4_svdquant_linear_sm120);
#endif
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_tactic_num,
                              torch_ext::nvfp4_svdquant_gemm_tactic_num);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_lora_rank,
                              torch_ext::nvfp4_svdquant_gemm_lora_rank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_can_implement,
                              torch_ext::nvfp4_svdquant_gemm_can_implement);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_fallback_tactic,
                              torch_ext::nvfp4_svdquant_gemm_fallback_tactic);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_shared_storage_size,
                              torch_ext::nvfp4_svdquant_gemm_shared_storage_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_workspace_size,
                              torch_ext::nvfp4_svdquant_gemm_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_tactic_row,
                              torch_ext::nvfp4_svdquant_gemm_tactic_row);
