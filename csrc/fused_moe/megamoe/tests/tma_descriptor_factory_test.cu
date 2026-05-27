// ============================================================================
// Standalone host-side unit test for the `CUtensorMap` descriptor factories
// declared in `vllm/csrc/moe/moe_monokernel/src/moe_tma.h` and defined in
// `vllm/csrc/moe/moe_monokernel/src/moe_tma.cu` (Task 2.4 of the
// tma-wgmma-weight-load spec and Task 6.4 of the
// tma-wgmma-down-projection spec).
//
// Goal (R5.6, R6.3 from up-proj spec; R6.6, R9.3 from down-proj spec):
//   1. Allocate dummy device pointers for the up-projection weight and
//      activation tensors and for the down-projection weight and
//      intermediate-activation tensors via `cudaMalloc`.  The descriptors
//      only store the pointer value; no real data is needed.
//   2. Call all four factories with representative shapes:
//        - `create_up_weight_tma_desc(weights_ptr, E=256, N=1408, K=3584)`
//        - `create_activations_tma_desc(acts_ptr, BS=8, K_hidden=3584)`
//        - `create_down_weight_tma_desc(weights_ptr, E=256, K=3584, N=512)`
//        - `create_down_activation_tma_desc(acts_ptr, TEMP_ROWS=64, N=512)`
//   3. Assert `sizeof(CUtensorMap) == 128` so a layout change in the CUDA
//      Driver API surface shows up loudly instead of silently.
//   4. Assert that each returned descriptor has at least one non-zero byte
//      out of its 128 — catches the failure mode where a factory silently
//      returns a zero-initialized POD without calling
//      `cuTensorMapEncodeTiled` (e.g., a stubbed implementation or a
//      misreported `CUresult`).
//
// Build (requires CUDA 12.0+ and a pytorch install that matches `torch`'s
// cxx11-ABI; the moe_tma.cu factories use `TORCH_CHECK` on encode failure):
//
//   TORCH_DIR=$(python -c 'import torch, os;
//   print(os.path.dirname(torch.__file__))')
//   nvcc -std=c++17 -arch=sm_90a -O2 \
//        -D_GLIBCXX_USE_CXX11_ABI=1 \
//        -I vllm/csrc/moe/moe_monokernel/src \
//        -I "$TORCH_DIR/include" \
//        -I "$TORCH_DIR/include/torch/csrc/api/include" \
//        -L "$TORCH_DIR/lib" \
//        -Xlinker -rpath="$TORCH_DIR/lib" \
//        -lcuda -ltorch -ltorch_cpu -lc10 \
//        vllm/csrc/moe/moe_monokernel/tests/tma_descriptor_factory_test.cu \
//        vllm/csrc/moe/moe_monokernel/src/moe_tma.cu \
//        -o /tmp/tma_descriptor_factory_test
//   /tmp/tma_descriptor_factory_test
//
// Exit codes:
//   0  - test passed
//   1  - factory produced an invalid descriptor (wrong size or all-zeros)
//   2  - CUDA runtime error (printed to stderr)
//  77  - device is pre-SM90 or CUDA driver is unavailable; skipped
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../src/moe_tma.h"

namespace {

// Representative shapes from the task description:
//   E=256, N=1408, K=3584 (Qwen3.5-35B up-projection)
//   BS=8, K_hidden=3584
constexpr uint32_t kNumExperts = 256u;
constexpr uint32_t kN = 1408u;
constexpr uint32_t kK = 3584u;
constexpr uint32_t kBatchSizeCap = 8u;
constexpr uint32_t kKHidden = 3584u;

// Representative shapes for the down-projection factories (Task 6.4):
//   E=256, K=3584 (hidden = down-proj output), N=512 (reduction = up-proj N)
//   TEMP_ROWS = BS * MAX_TOPK = 8 * 8 = 64
// Note: `kDownK` uses the same numerical hidden size as `kK` above but is
// named separately to reflect its role as the down-projection OUTPUT rather
// than the up-projection REDUCTION axis.
constexpr uint32_t kDownK = 3584u;
constexpr uint32_t kDownN = 512u;
constexpr uint32_t kTempRows = 64u;

// sizeof(CUtensorMap) is a stable 128 B on CUDA 12.x; hard-code it here so
// the test breaks loudly if the Driver API POD ever grows/shrinks.
constexpr size_t kExpectedCUtensorMapSize = 128u;

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t _err = (expr);                                                   \
    if (_err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                   cudaGetErrorName(_err), cudaGetErrorString(_err));            \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

// Count non-zero bytes in a fixed-size POD to defend against a silent
// zero-init bypass — if a factory returned the default-constructed
// `CUtensorMap{}` without calling `cuTensorMapEncodeTiled`, every byte
// would be zero and the kernel would dispatch TMAs to `nullptr`.
size_t count_nonzero_bytes(const CUtensorMap& desc) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&desc);
  size_t nz = 0;
  for (size_t i = 0; i < sizeof(CUtensorMap); ++i) {
    if (bytes[i] != 0u) ++nz;
  }
  return nz;
}

// Emit the first 32 bytes of a descriptor as hex for diagnosis on failure.
void dump_descriptor_head(const char* label, const CUtensorMap& desc) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&desc);
  std::fprintf(stderr, "  %s[0..32] =", label);
  for (size_t i = 0; i < 32 && i < sizeof(CUtensorMap); ++i) {
    std::fprintf(stderr, " %02X", bytes[i]);
  }
  std::fprintf(stderr, "\n");
}

}  // namespace

int main() {
  // --- Device sanity ------------------------------------------------------
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major < 9) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test: device %d (%s) has compute "
                 "capability %d.%d; this test requires SM90 (H100) or "
                 "newer.\n",
                 device, props.name, props.major, props.minor);
    return 77;
  }

  // --- Compile-time / ABI check (R5.6) -----------------------------------
  //
  // The factory returns `CUtensorMap` by value, and the kernel receives it
  // as a `__grid_constant__ CUtensorMap const` parameter.  Both rely on the
  // POD being exactly 128 bytes; the Driver API header pins this size.  If
  // this ever changes we want a loud, immediate failure.
  static_assert(sizeof(CUtensorMap) == kExpectedCUtensorMapSize,
                "sizeof(CUtensorMap) must be 128 bytes for the __grid_constant__ "
                "kernel-parameter contract to hold (spec R5.6).");
  if (sizeof(CUtensorMap) != kExpectedCUtensorMapSize) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test FAILED: sizeof(CUtensorMap) = %zu, "
                 "expected %zu.\n",
                 sizeof(CUtensorMap), kExpectedCUtensorMapSize);
    return 1;
  }

  // --- Allocate dummy device pointers ------------------------------------
  //
  // The descriptor only stores the pointer value; no writes go through it
  // at descriptor-build time.  Allocating real storage (rather than passing
  // a fake address) means `cuTensorMapEncodeTiled`'s optional alignment /
  // addressability checks — if any — still pass.
  const size_t weights_bytes = static_cast<size_t>(kNumExperts) * 2ULL * kN * kK;  // fp8 = 1 B/elem
  const size_t acts_bytes =
      static_cast<size_t>(kBatchSizeCap) * kKHidden * 2ULL;  // bf16 = 2 B/elem
  // Down-projection weights: `[E, K, N]` fp8.  For `E=256, K=3584, N=512`
  // this is 256 * 3584 * 512 ≈ 469 MB — well within a typical SM90 device's
  // GM budget.  `cuTensorMapEncodeTiled` only records the pointer value and
  // does not read through it, but we allocate real storage to make any
  // addressability checks happy.
  const size_t down_weights_bytes =
      static_cast<size_t>(kNumExperts) * kDownK * kDownN;  // fp8 = 1 B/elem
  // Down-projection intermediate activations: `[TEMP_ROWS, N]` fp8.
  // TEMP_ROWS = 64, N = 512 → 32 KB; fits in any device.
  const size_t down_acts_bytes = static_cast<size_t>(kTempRows) * kDownN;  // fp8 = 1 B/elem

  void* d_weights = nullptr;
  void* d_acts = nullptr;
  void* d_down_weights = nullptr;
  void* d_down_acts = nullptr;
  CUDA_CHECK(cudaMalloc(&d_weights, weights_bytes));
  CUDA_CHECK(cudaMalloc(&d_acts, acts_bytes));
  CUDA_CHECK(cudaMalloc(&d_down_weights, down_weights_bytes));
  CUDA_CHECK(cudaMalloc(&d_down_acts, down_acts_bytes));

  // --- Build descriptors --------------------------------------------------
  CUtensorMap weight_desc =
      moe_monokernel::create_up_weight_tma_desc(d_weights, kNumExperts, kN, kK);
  CUtensorMap act_desc =
      moe_monokernel::create_activations_tma_desc(d_acts, kBatchSizeCap, kKHidden);
  // Down-projection factories (Task 6.4).  The two new descriptors share
  // the same 128-B POD shape as the up-proj descriptors; the non-zero-byte
  // check below guards against a silent zero-init bypass just as it does
  // for the up-proj pair.
  CUtensorMap down_weight_desc =
      moe_monokernel::create_down_weight_tma_desc(d_down_weights, kNumExperts, kDownK, kDownN);
  CUtensorMap down_act_desc =
      moe_monokernel::create_down_activation_tma_desc(d_down_acts, kTempRows, kDownN);

  // --- Non-zero-byte check (R6.3 intent: guard against silent zero-init) --
  const size_t weight_nz = count_nonzero_bytes(weight_desc);
  const size_t act_nz = count_nonzero_bytes(act_desc);
  const size_t down_weight_nz = count_nonzero_bytes(down_weight_desc);
  const size_t down_act_nz = count_nonzero_bytes(down_act_desc);

  int rc = 0;
  if (weight_nz == 0) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test FAILED: up-weight descriptor "
                 "has 0 non-zero bytes (expected > 0; a zero-initialized "
                 "descriptor would silently dispatch TMAs to nullptr).\n");
    dump_descriptor_head("weight_desc", weight_desc);
    rc = 1;
  }
  if (act_nz == 0) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test FAILED: activations descriptor "
                 "has 0 non-zero bytes (expected > 0; a zero-initialized "
                 "descriptor would silently dispatch TMAs to nullptr).\n");
    dump_descriptor_head("act_desc", act_desc);
    rc = 1;
  }
  if (down_weight_nz == 0) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test FAILED: down-weight descriptor "
                 "has 0 non-zero bytes (expected > 0; a zero-initialized "
                 "descriptor would silently dispatch TMAs to nullptr).\n");
    dump_descriptor_head("down_weight_desc", down_weight_desc);
    rc = 1;
  }
  if (down_act_nz == 0) {
    std::fprintf(stderr,
                 "tma_descriptor_factory_test FAILED: down-activations descriptor "
                 "has 0 non-zero bytes (expected > 0; a zero-initialized "
                 "descriptor would silently dispatch TMAs to nullptr).\n");
    dump_descriptor_head("down_act_desc", down_act_desc);
    rc = 1;
  }

  if (rc == 0) {
    std::printf(
        "tma_descriptor_factory_test OK: sizeof(CUtensorMap)=%zu, "
        "weight_desc non-zero bytes=%zu/%zu, "
        "act_desc non-zero bytes=%zu/%zu, "
        "down_weight_desc non-zero bytes=%zu/%zu, "
        "down_act_desc non-zero bytes=%zu/%zu.\n",
        sizeof(CUtensorMap), weight_nz, sizeof(CUtensorMap), act_nz, sizeof(CUtensorMap),
        down_weight_nz, sizeof(CUtensorMap), down_act_nz, sizeof(CUtensorMap));
  }

  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_acts));
  CUDA_CHECK(cudaFree(d_down_weights));
  CUDA_CHECK(cudaFree(d_down_acts));

  return rc;
}
