# MoE Monokernel

Single-kernel Mixture-of-Experts implementation for Qwen3-Coder / Qwen3.5-35B
on NVIDIA Hopper (SM90+). The full top-K pipeline (routing, up-projection,
SiLU, down-projection, reduction) runs inside one `__global__` function
(`moe_kernel_topk<Dims>`) to maximize on-chip reuse of activations and
weight tiles.

- Host wrapper: `moe_wrapper.cu`
- Device code: `src/`
- Standalone device tests: `tests/`

## Software Grid / Partial Barriers

### Why software barriers

Phase boundaries inside `moe_kernel_topk` used to synchronize through
`cudaLaunchCooperativeKernel` + `cooperative_groups::this_grid().sync()`.
Cooperative launch blocks `cudaStreamBeginCapture`, so the kernel could
not be captured into a CUDA Graph and paid a per-decode-step host launch
overhead. The primitives in `src/moe_grid_barrier.h` replace that with a
standard `<<<grid, block, smem, stream>>>` launch plus a software barrier
built on global-memory atomic counters. See
`.kiro/specs/moe-monokernel-software-grid-sync/` for the full spec.

### Co-residency invariant

The software barrier is a spin on a global-memory flag, so it can only
deadlock if a block is waiting on a block that has not yet been scheduled.
The monokernel pins one block per SM so every block is co-resident from
launch:

- `Dims::KernelConfig::GRID_SIZE = 128` on H200 (132 SMs).
- `__launch_bounds__(BLOCK_SIZE, 1)` on `moe_kernel_topk<Dims>`
  (BS8 TMA+WGMMA uses `BLOCK_SIZE = 384`).
- Opt-in dynamic SHM > per-SM/2 (`sizeof(MoE_SHM<Dims>) ≤ 228 KB`) so a
  second block cannot fit alongside.

The host launcher (`moe_wrapper.cu`) enforces the runtime half via
`TORCH_CHECK`:

- `Dims::KernelConfig::GRID_SIZE <= cudaDevAttrMultiProcessorCount`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(...) == 1`

Both checks run once per process (gated behind a `_diag_printed`
static). A future configuration that breaks either invariant will fail
loudly at launch rather than deadlock at runtime.

### Primitive API

From `src/moe_grid_barrier.h`:

```cpp
namespace moe_monokernel {

// Full-grid barrier. GRID_SIZE_STATIC arrivals; compile-time seed value.
template <uint32_t GRID_SIZE_STATIC>
__device__ __forceinline__ void grid_barrier(
    uint32_t* __restrict__ counters,  // &spec->grid_barrier.slot[0]
    uint32_t&              phase);    // in/out, register-resident

// Generic sub-grid barrier. Underpins both expert_barrier and
// colstripe_barrier; disjoint counter regions make their id
// namespaces non-interfering.
__device__ __forceinline__ void partial_barrier(
    uint32_t* __restrict__ counter_region,
    uint32_t               id,
    uint32_t               arrival_count,
    uint32_t               seed_thread_blockidx,
    uint32_t&              phase);

// Phase 2b site #2 (BS8 Phase 3 → 4). Thin alias for partial_barrier
// keyed by up_group; arrival_count = UP_GRID = 8.
__device__ __forceinline__ void expert_barrier(
    uint32_t* __restrict__ expert_counters,
    uint32_t expert_id, uint32_t arrival_count,
    uint32_t seed_thread_blockidx, uint32_t& phase);

// Phase 2b site #3 (BS8 Phase 4 → 5). Thin alias for partial_barrier
// keyed by blockIdx.x % DOWN_GRID; arrival_count = DOWN_GROUPS = 16.
__device__ __forceinline__ void colstripe_barrier(
    uint32_t* __restrict__ colstripe_counters,
    uint32_t col_stripe, uint32_t arrival_count,
    uint32_t seed_thread_blockidx, uint32_t& phase);

}  // namespace moe_monokernel
```

`expert_barrier` and `colstripe_barrier` add no logic — they exist for
call-site clarity and inline through `partial_barrier` at zero runtime
cost.

### Protocol

Every primitive follows the same seed-atomicAdd-spin-on-high-bit
discipline:

1. `__syncthreads()` publishes pre-barrier thread-local writes across
   the block.
2. Thread 0 issues `__threadfence()` so the block's pre-barrier global
   writes are visible to other SMs before its arrival.
3. The seed block writes `0x80000000u - (arrival_count - 1)` with
   `atomicExch`. The previous value may be a mix of the prior call's
   `0x80000000u` exit-state marker and any early arrivals for the
   current call, so the seed thread masks off bit 31 and folds back
   the low-31-bit arrival count via `atomicAdd(c, prior & 0x7FFFFFFFu)`.
4. Non-seed blocks issue exactly one `atomicAdd(c, 1u)`.
5. Every thread spins on `atomicAdd(c, 0u)` (uncached device-scope
   read) until bit 31 flips set. The last arrival drives the slot to
   exactly `0x80000000u`.
6. `__threadfence()` + `__syncthreads()` establish cross-block
   happens-before and re-gather the block before returning.

The high-bit invariant holds because `arrival_count ≤ GRID_SIZE ≤ 132`
on H200, so `SEED + j` has bit 31 clear for every partial arrival
`j ∈ [0, arrival_count - 1)`.

### Ping-pong reset discipline

Each counter region is a pair of slots; `phase & 1` selects the active
one and `++phase` on barrier exit rotates them. The reset is
**self-maintaining** across kernel invocations because the next call
that lands on a given slot re-seeds it with `atomicExch`, overwriting
the previous `0x80000000u` in one atomic step and folding any stray
low-31-bit arrivals back into the fresh seed. The host zero-initializes
the counter region once per process via `cudaMemsetAsync` on the first
launch — no per-call memset is required.

### Migrated call sites

Five cooperative-groups sync sites were migrated. Site #1 is eliminated
for BS8 because its Phase-5 reduction `=`-writes every element of
`activations_out` (for tokens in `[0, batch_size)` it assigns the
reduced sum; for tokens in `[batch_size, Dims::BS)` it explicitly
zeros the block's col stripe), making the pre-zero loop dead work.

| # | Variant       | Location                               | Arrival count | Primitive                                    | Notes                                                    |
|---|---------------|----------------------------------------|---------------|----------------------------------------------|----------------------------------------------------------|
| 1 | BS64 only     | top-of-kernel (BS8 eliminated)         | 128           | `grid_barrier`                               | BS8 Phase 5 `=`-writes obviate the pre-zero              |
| 2 | BS8 TMA+WGMMA | Phase 3 → 4                            | 8             | `expert_barrier(up_group)`                   | Phase 2b; 16 concurrent barriers (one per expert group)  |
| 3 | BS8 TMA+WGMMA | Phase 4 → 5                            | 16            | `colstripe_barrier(blockIdx.x % DOWN_GRID)`  | Phase 2b; 8 concurrent barriers (one per col stripe)     |
| 4 | BS64          | up → down                              | 128           | `grid_barrier`                               | BS64 not Phase-2 scope                                   |
| 5 | BS64          | `act_scale` visibility                 | 128           | `grid_barrier`                               | BS64 not Phase-2 scope; `moe_scale_activation_BSx`       |

### Counter storage

Appended at the **tail** of `MoEGemmSpec<Dims>` (after `act_scale`) so
`TEMP_FP8_OFFSET = offsetof(MoEGemmSpec<Dims>, temp_fp8)` stays
byte-accurate — the host-side down-activation TMA descriptor in
`moe_wrapper.cu` depends on that constant:

```cpp
struct {
  uint32_t slot[2];
} grid_barrier;                                   // 8 B — Counter_Pair

struct {
  uint32_t expert_slot[Dims::NUM_EXPERTS][2];     // 2 KB @ NUM_EXPERTS=256
  uint32_t colstripe_slot[DOWN_GRID][2];          // 64 B @ DOWN_GRID=8
} partial_barrier;
```

For the BS8 TMA+WGMMA variant (`NUM_EXPERTS = 256`, `DOWN_GRID = 8`)
the total barrier counter footprint is ~2120 B — negligible vs. the
MB-scale scratchpad. Any new field added to `MoEGemmSpec<Dims>` must
go **after** `temp_fp8` or the down-activation TMA descriptor will
silently address the wrong memory.

### CUDA Graph capture

The migrated kernel launches via standard `cudaLaunchKernel` and can
now be captured into a `torch.cuda.CUDAGraph`. See the
`_bench_cudagraph` harness in `test_monokernel_accuracy.py` for an
end-to-end capture / replay example:

```python
# test_monokernel_accuracy.py
cuda_cg_ms = _bench_cudagraph(cuda_fn)
```

Capture was impossible under the previous cooperative launch path.

### Further reading

The full design — seed correctness argument, fence discipline,
per-site producer/consumer-set analysis, and the Phase-2a layout
alignment that enables `expert_barrier` at site #2 — lives in
`.kiro/specs/moe-monokernel-software-grid-sync/design.md`.
