// ============================================================================
// Standalone device-side sanity test for the mbarrier PTX wrappers defined in
// `vllm/csrc/moe/moe_monokernel/src/ptx_utils.h` (Task 1.3 of the
// tma-wgmma-weight-load spec).
//
// Goal (R3.1, R3.2, R3.5):
//   1. Allocate a single 64-bit mbarrier in SHM.
//   2. One thread inits it with `arrival_count = 1`.
//   3. Publish with `fence.mbarrier_init.release.cluster`.
//   4. One thread calls `mbarrier_arrive_expect_tx(bar, /*tx=*/0)`. With
//      `tx_bytes == 0` both internal counters (arrival and transaction-bytes)
//      reach zero in a single arrive, so the barrier must complete
//      immediately without any TMA traffic.
//   5. Loop on `mbarrier_try_wait_parity(bar, /*parity=*/0)` until it
//      returns true, then write a success flag to GM.
//   6. Host verifies the flag.
//
// Build (requires CUDA 12.0+ and an SM90a target):
//   nvcc -std=c++17 -arch=sm_90a -O2 \
//        -I vllm/csrc/moe/moe_monokernel/src \
//        vllm/csrc/moe/moe_monokernel/tests/mbarrier_sanity_test.cu \
//        -o mbarrier_sanity_test
//
// Run on an H100 / H200:
//   ./mbarrier_sanity_test
//
// The kernel uses a short bounded spin count so that a broken wrapper cannot
// hang the GPU — on failure the test surfaces a clear diagnostic and returns
// a non-zero exit status.
// ============================================================================

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../src/ptx_utils.h"

namespace {

// Arbitrary "magic" success sentinel — any non-zero value would do; we pick a
// recognizable one to make mismatches easy to spot in failure output.
constexpr std::uint32_t kSuccessSentinel = 0xC0DE0001u;

// Maximum number of `try_wait.parity` polls before the test gives up. With
// `arrival_count == 1` and `tx == 0` the barrier should complete on the first
// poll, so this bound is deliberately loose.
constexpr std::uint32_t kMaxSpinIters = 1'000'000u;

// Kernel: launches a single block with one warp; thread 0 drives the whole
// test, all other threads are idle.
__global__ void mbarrier_sanity_kernel(std::uint32_t* out_flag, std::uint32_t* out_iters) {
  __shared__ alignas(16) std::uint64_t bar;

  if (threadIdx.x == 0) {
    // R3.1: init a single mbarrier with arrival_count = 1.
    moe_monokernel::mbarrier_init(&bar, /*arrival_count=*/1);
    // R3.2: publish the init before any arrive / try_wait.
    moe_monokernel::fence_mbarrier_init_release_cluster();

    // R3.5: arm with tx = 0 and arrive. With no TMA traffic, the barrier
    // must flip parity to 1 immediately, so try_wait.parity(0) should
    // succeed on (or very near) the first poll.
    moe_monokernel::mbarrier_arrive_expect_tx(&bar, /*tx_bytes=*/0);

    std::uint32_t iters = 0;
    bool done = false;
    while (iters < kMaxSpinIters) {
      done = moe_monokernel::mbarrier_try_wait_parity(&bar, /*parity=*/0);
      ++iters;
      if (done) break;
    }

    *out_iters = iters;
    *out_flag = done ? kSuccessSentinel : 0u;
  }
}

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t _err = (expr);                                                   \
    if (_err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                   cudaGetErrorName(_err), cudaGetErrorString(_err));            \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

}  // namespace

int main() {
  // Sanity-check the device is SM90+.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major < 9) {
    std::fprintf(stderr,
                 "mbarrier_sanity_test: device %d (%s) has compute "
                 "capability %d.%d; this test requires SM90 (H100) or "
                 "newer.\n",
                 device, props.name, props.major, props.minor);
    return 77;  // conventional "skip" exit code for autotools-style runners.
  }

  std::uint32_t* d_flag = nullptr;
  std::uint32_t* d_iters = nullptr;
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_iters, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(d_iters, 0, sizeof(std::uint32_t)));

  mbarrier_sanity_kernel<<<1, 32>>>(d_flag, d_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::uint32_t h_flag = 0;
  std::uint32_t h_iters = 0;
  CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_iters, d_iters, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_flag));
  CUDA_CHECK(cudaFree(d_iters));

  int rc = 0;
  if (h_flag != kSuccessSentinel) {
    std::fprintf(stderr,
                 "mbarrier_sanity_test FAILED: try_wait.parity never "
                 "reported complete (flag=0x%08X, iters=%u / %u).\n",
                 h_flag, h_iters, kMaxSpinIters);
    rc = 1;
  } else {
    std::printf(
        "mbarrier_sanity_test OK: barrier completed after %u poll(s) "
        "(flag=0x%08X).\n",
        h_iters, h_flag);
  }

  return rc;
}
