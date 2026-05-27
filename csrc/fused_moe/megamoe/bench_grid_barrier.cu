// ============================================================================
// Standalone microbenchmark for the software grid / expert / col-stripe
// barriers defined in `src/moe_grid_barrier.h` (spec
// `.kiro/specs/moe-monokernel-software-grid-sync/`, Task 16.1).
//
// Goal
// ----
// Measure per-call latency of:
//
//   * `moe_monokernel::grid_barrier<GRID_SIZE_STATIC>`   — 128 arrivals
//     (Req 7.1, 7.2: overhead budget ≤ 10 µs at GRID_SIZE = 128).
//
//   * `moe_monokernel::expert_barrier` — arrival_count = UP_GRID = 8,
//     one barrier id (id = 0), only the first 8 blocks participate
//     (matches the production BS8 TMA+WGMMA Expert_Barrier at site #2
//     for a single expert group at UP_GRID = 8).
//     (Req 10.1: latency ≤ 50 % of Grid_Barrier at the same grid.)
//
//   * `moe_monokernel::colstripe_barrier` — arrival_count = DOWN_GROUPS = 16,
//     8 concurrent barriers (id = blockIdx.x % 8), so every block of the
//     128-block grid calls the barrier with its col stripe id. Matches the
//     production ColStripe_Barrier at site #3 with DOWN_GRID = 8 and
//     DOWN_GROUPS = 16 after the Phase-2a layout alignment.
//     (Req 10.1: latency ≤ 50 % of Grid_Barrier at the same grid.)
//
// The Grid_Barrier kernel uses `__launch_bounds__(BLOCK_SIZE, 1)` to match
// the monokernel's one-block-per-SM occupancy invariant (Req 4.4) — the
// microbenchmark would otherwise pack two measurement blocks onto a single
// SM and report artificially low latency.
//
// Cooperative-groups baseline
// ---------------------------
// The spec's task item originally asked for a fourth kernel,
// `cooperative_group_sync_microbench`, as a reference point. That kernel
// requires `cudaLaunchCooperativeKernel` + `-rdc=true`, neither of which
// the production monokernel build uses (by design — spec Requirement 1
// removes cooperative launch entirely so CUDA Graph capture can work).
// Including it here would force this benchmark to also adopt `-rdc=true`,
// complicating what is otherwise a self-contained single-TU build. The
// baseline comparison is therefore out-of-band: run a separate
// cooperative-launch kernel in its own TU if a direct
// `cooperative_groups::this_grid().sync()` reference number is needed.
//
// Timing methodology
// ------------------
// Per-kernel measurement uses `cudaEventRecord` / `cudaEventElapsedTime`
// on the default stream. The host driver:
//
//   1. Sweeps `N_BARRIERS ∈ {1, 10, 100, 1000}` per primitive.
//   2. Warms up by running the kernel ≥ 10 times at the largest
//      N_BARRIERS to prime the instruction cache and L2.
//   3. Reports `elapsed_ms * 1000 / N_BARRIERS` as per-call µs, matching
//      the Req 7.2 definition of "per-call overhead".
//
// Launch overhead (≈ 5–10 µs on H200) shows up the same way for every
// primitive at every N, so subtracting it is unnecessary for the relative
// ranking the spec cares about (Grid vs. Expert vs. ColStripe). The
// absolute per-call numbers at N = 1000 are however already dominated by
// the barriers themselves, so the launch-overhead term is negligible.
//
// Build
// -----
//   nvcc -std=c++17 -arch=sm_90 -O3
//        -I vllm/csrc/moe/moe_monokernel/src
//        vllm/csrc/moe/moe_monokernel/bench_grid_barrier.cu
//        -o /tmp/bench_grid_barrier
//
// Run (on H200)
// -------------
//   /tmp/bench_grid_barrier
//
// Expected output shape:
//
//   Grid_Barrier (128 arrivals):
//     N=1     latency = <x> μs
//     N=10    latency = <x> μs
//     N=100   latency = <x> μs
//     N=1000  latency = <x> μs
//   Expert_Barrier (8 arrivals, 1 concurrent):
//     ...
//   ColStripe_Barrier (16 arrivals, 8 concurrent):
//     ...
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// `moe_grid_barrier.h` is guarded with
// `#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION #error` so it can't be
// included from outside the kernel TU by accident. For the microbench
// we intentionally flip the guard — we are the implementation here, in
// the sense that we consume the device primitive directly.
#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "src/moe_grid_barrier.h"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

// ── Configuration ────────────────────────────────────────────────────────

// Must match the production BS8 TMA+WGMMA variant so the microbench
// result is directly comparable to the end-to-end overhead budget in
// Req 7.1. 128 blocks × 384 threads = one-block-per-SM on H200 (132
// SMs); see Design "Target compile-time configuration".
constexpr uint32_t GRID_SIZE = 128u;
constexpr uint32_t BLOCK_SIZE = 384u;

// Arrival-set sizes for the sub-grid barriers.
constexpr uint32_t UP_GRID = 8u;       // Expert_Barrier arrival count
constexpr uint32_t DOWN_GRID = 8u;     // number of concurrent col-stripe
                                       // barriers
constexpr uint32_t DOWN_GROUPS = 16u;  // ColStripe_Barrier arrival count

// Counter-region sizing: one Counter_Pair (2 × uint32_t) per barrier id.
constexpr uint32_t GRID_COUNTERS = 2u;            // one pair
constexpr uint32_t EXPERT_COUNTERS = 16u * 2u;    // 16 Counter_Pairs
                                                  // — spec R13.1 says
                                                  // "16 Counter_Pairs
                                                  // for expert"; we
                                                  // only hit id=0 but
                                                  // provision the full
                                                  // set to match the
                                                  // real scratchpad.
constexpr uint32_t COLSTRIPE_COUNTERS = 8u * 2u;  // 8 Counter_Pairs
                                                  // — one per col
                                                  // stripe.

// H200 shader clock (2.11 GHz). Unused in the current event-based timing
// path but kept as a documented constant for anyone who wants to cross-
// check against `clock64()` reads inside the device code (spec Task 16.1
// "Compute per-call µs using H200's 2.11 GHz shader clock").
[[maybe_unused]] constexpr double H200_SHADER_GHZ = 2.11;

constexpr int WARMUP_LAUNCHES = 10;

// ── CUDA error helper ────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                               \
  do {                                                                 \
    cudaError_t _err = (expr);                                         \
    if (_err != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, \
                   __LINE__, cudaGetErrorName(_err),                   \
                   cudaGetErrorString(_err));                          \
      std::exit(2);                                                    \
    }                                                                  \
  } while (0)

// ── Device-side microbench kernels ──────────────────────────────────────

/**
 * @brief Grid_Barrier microbench — every block in the grid arrives at
 *        every iteration. Mirrors spec "Microbenchmark design" snippet.
 *
 * `GRID_SIZE_STATIC` is a template non-type parameter so `grid_barrier`'s
 * seed value and degenerate-case gate fold at compile time (same shape
 * as the production call sites).
 */
template <uint32_t GRID_SIZE_STATIC>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void grid_barrier_microbench(
    uint32_t* __restrict__ counters, uint32_t N_BARRIERS) {
  uint32_t phase = 0u;
  for (uint32_t i = 0; i < N_BARRIERS; ++i) {
    moe_monokernel::grid_barrier<GRID_SIZE_STATIC>(counters, phase);
  }
}

/**
 * @brief Expert_Barrier microbench — only the first `UP_GRID = 8` blocks
 *        participate, all calling with the same barrier id (= 0).
 *
 * Blocks outside the arrival set must NOT call `expert_barrier` for this
 * id (caller contract in `moe_grid_barrier.h`), so we gate the call on
 * `blockIdx.x < UP_GRID`. The remaining 120 blocks spin waiting for the
 * kernel to end but do not touch the counter.
 *
 * Seed block is blockIdx.x == 0 (lowest blockIdx.x in the arrival set),
 * matching the production recipe `up_group * UP_GRID` with up_group = 0.
 */
__global__ __launch_bounds__(BLOCK_SIZE, 1) void expert_barrier_microbench(
    uint32_t* __restrict__ expert_counters, uint32_t N_BARRIERS) {
  uint32_t phase = 0u;
  if (blockIdx.x < UP_GRID) {
    for (uint32_t i = 0; i < N_BARRIERS; ++i) {
      moe_monokernel::expert_barrier(expert_counters,
                                     /*expert_id=*/0u,
                                     /*arrival_count=*/UP_GRID,
                                     /*seed_thread_blockidx=*/0u, phase);
    }
  }
}

/**
 * @brief ColStripe_Barrier microbench — all 128 blocks call the barrier
 *        every iteration, using `col_stripe = blockIdx.x % DOWN_GRID`
 *        as the id. `DOWN_GRID = 8` concurrent barriers run, each with
 *        `DOWN_GROUPS = 16` arrivals.
 *
 * Seed block for col stripe c is the block whose `blockIdx.x == c`
 * (matches the production recipe where the Phase-5 writer for col
 * stripe c has blockIdx.x == c).
 */
__global__ __launch_bounds__(BLOCK_SIZE, 1) void colstripe_barrier_microbench(
    uint32_t* __restrict__ colstripe_counters, uint32_t N_BARRIERS) {
  const uint32_t col_stripe = blockIdx.x % DOWN_GRID;
  uint32_t phase = 0u;
  for (uint32_t i = 0; i < N_BARRIERS; ++i) {
    moe_monokernel::colstripe_barrier(colstripe_counters,
                                      /*col_stripe=*/col_stripe,
                                      /*arrival_count=*/DOWN_GROUPS,
                                      /*seed_thread_blockidx=*/col_stripe,
                                      phase);
  }
}

// ── Host-side timing helpers ────────────────────────────────────────────

/// Time a single kernel launch (on the default stream) using
/// `cudaEventRecord` / `cudaEventElapsedTime`. Returns elapsed
/// milliseconds.
template <typename Launch>
static float time_kernel_ms(Launch&& launch) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

struct BenchResult {
  uint32_t n_barriers;
  double per_call_us;
};

/// Sweep `N_BARRIERS ∈ {1, 10, 100, 1000}` for a given launch closure.
/// Performs `WARMUP_LAUNCHES` warmup iterations at the largest N before
/// timing.
template <typename Launch>
static void sweep_bench(const char* label, Launch&& launch,
                        BenchResult results[4]) {
  const uint32_t Ns[4] = {1u, 10u, 100u, 1000u};

  // Warm up on the largest N to prime the instruction cache and let the
  // ping-pong slot reach steady-state before the timed runs.
  for (int w = 0; w < WARMUP_LAUNCHES; ++w) {
    launch(Ns[3]);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("%s:\n", label);
  for (int i = 0; i < 4; ++i) {
    const uint32_t N = Ns[i];
    const float ms = time_kernel_ms([&]() { launch(N); });
    const double us_per_call =
        (static_cast<double>(ms) * 1000.0) / static_cast<double>(N);
    results[i] = {N, us_per_call};
    std::printf("  N=%-5u latency = %.3f μs\n", N, us_per_call);
  }
}

// ── main ────────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/) {
  // Sanity-check the device is SM90+ (H100 / H200). Matches the
  // production monokernel's target. The microbench itself doesn't use
  // SM90-specific features beyond what `grid_barrier.h` uses (device-
  // scope atomics + threadfences, available from SM60), but the latency
  // numbers are only meaningful against the SM90 co-residency
  // invariant, so we require it.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::printf("Device %d: %s (compute %d.%d, %d SMs)\n", device, props.name,
              props.major, props.minor, props.multiProcessorCount);
  if (props.major < 9) {
    std::fprintf(stderr,
                 "bench_grid_barrier: device has compute capability %d.%d; "
                 "this benchmark expects SM90 (H100) or newer.\n",
                 props.major, props.minor);
    return 77;  // conventional "skip" exit code.
  }
  if (static_cast<uint32_t>(props.multiProcessorCount) < GRID_SIZE) {
    std::fprintf(stderr,
                 "bench_grid_barrier: device has %d SMs but GRID_SIZE = %u. "
                 "The software-barrier co-residency invariant requires "
                 "GRID_SIZE <= SM count, otherwise the kernel can deadlock.\n",
                 props.multiProcessorCount, GRID_SIZE);
    return 2;
  }

  // Allocate + zero-init the three counter regions.
  //   grid:       1  Counter_Pair  = 2 × uint32_t       (8 B)
  //   expert:     16 Counter_Pairs = 32 × uint32_t      (128 B)
  //   colstripe:  8  Counter_Pairs = 16 × uint32_t      (64 B)
  // Spec Task 16.1 calls for "16 Counter_Pairs for expert, 8
  // Counter_Pairs for colstripe" to mirror the scratchpad layout from
  // `MoEGemmSpec<Dims>::partial_barrier`. Zero-init is required on
  // first use (spec R13.2, `moe_grid_barrier.h` caller contract).
  uint32_t* d_grid_counters = nullptr;
  uint32_t* d_expert_counters = nullptr;
  uint32_t* d_colstripe_counters = nullptr;
  CUDA_CHECK(cudaMalloc(&d_grid_counters, GRID_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_expert_counters, EXPERT_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_colstripe_counters, COLSTRIPE_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_grid_counters, 0, GRID_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMemset(d_expert_counters, 0, EXPERT_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_colstripe_counters, 0,
                        COLSTRIPE_COUNTERS * sizeof(uint32_t)));

  std::printf("GRID_SIZE=%u, BLOCK_SIZE=%u\n", GRID_SIZE, BLOCK_SIZE);
  std::printf("UP_GRID=%u, DOWN_GRID=%u, DOWN_GROUPS=%u\n\n", UP_GRID,
              DOWN_GRID, DOWN_GROUPS);

  BenchResult grid_results[4]{};
  BenchResult expert_results[4]{};
  BenchResult colstripe_results[4]{};

  sweep_bench(
      "Grid_Barrier (128 arrivals)",
      [&](uint32_t N) {
        grid_barrier_microbench<GRID_SIZE>
            <<<GRID_SIZE, BLOCK_SIZE>>>(d_grid_counters, N);
        CUDA_CHECK(cudaGetLastError());
      },
      grid_results);

  sweep_bench(
      "Expert_Barrier (8 arrivals, 1 concurrent)",
      [&](uint32_t N) {
        expert_barrier_microbench<<<GRID_SIZE, BLOCK_SIZE>>>(d_expert_counters,
                                                             N);
        CUDA_CHECK(cudaGetLastError());
      },
      expert_results);

  sweep_bench(
      "ColStripe_Barrier (16 arrivals, 8 concurrent)",
      [&](uint32_t N) {
        colstripe_barrier_microbench<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_colstripe_counters, N);
        CUDA_CHECK(cudaGetLastError());
      },
      colstripe_results);

  // Summary — compare each sub-grid barrier to the Grid_Barrier at
  // the same N (Req 10.1). Use N = 1000 as the reference point since
  // per-launch overhead is amortized there.
  const double grid_us = grid_results[3].per_call_us;
  const double expert_us = expert_results[3].per_call_us;
  const double colstripe_us = colstripe_results[3].per_call_us;

  std::printf("\nSummary (N=1000):\n");
  std::printf(
      "  Grid_Barrier       : %.3f μs   (R7.1 budget: ≤ 10 μs   — %s)\n",
      grid_us, grid_us <= 10.0 ? "PASS" : "FAIL");
  std::printf(
      "  Expert_Barrier     : %.3f μs   (%.1f%% of Grid, R10.1: ≤ 50%% — %s)\n",
      expert_us, grid_us > 0.0 ? (100.0 * expert_us / grid_us) : 0.0,
      (grid_us > 0.0 && expert_us <= 0.5 * grid_us) ? "PASS" : "FAIL");
  std::printf(
      "  ColStripe_Barrier  : %.3f μs   (%.1f%% of Grid, R10.1: ≤ 50%% — %s)\n",
      colstripe_us, grid_us > 0.0 ? (100.0 * colstripe_us / grid_us) : 0.0,
      (grid_us > 0.0 && colstripe_us <= 0.5 * grid_us) ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_grid_counters));
  CUDA_CHECK(cudaFree(d_expert_counters));
  CUDA_CHECK(cudaFree(d_colstripe_counters));
  return 0;
}
