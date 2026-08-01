# MonoMoe kernel — design

A single-kernel ("mono") top-K MoE for the Qwen3.5-35B block-FP8 shape on
Hopper (SM90a). One `cudaLaunchKernel` of `moe_kernel_topk<Dims>` runs the whole
MoE — routing, up-projection + SiLU, down-projection, reduction — with grid-wide
ordering between phases provided by software barriers (no cooperative-groups
launch, so it is CUDA-Graph-capturable).

This document is the single home for the cross-cutting concepts. Source comments
in `csrc/fused_moe/monomoe/` describe what a given line does and link here
(`see docs/design_docs/monomoe_kernel.md §N`) for the shared "why". When you
change a mechanism, update the section here rather than re-explaining it in
another file.

Fixed shape (`Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA`, see
`csrc/fused_moe/monomoe/src/moe_interface.h`): `E=256` experts,
`K = HIDDEN_STATES = 2048`, `N = 512`
(gate and up each have `N` rows), `BS ≤ 8` tokens, block-FP8 (128×128) weights,
`GRID_SIZE = 128`, `BLOCK_SIZE = 384` (12 warps). The kernel is hard-specialized
to this shape; `BS ≤ 8` is enforced by `static_assert`.

---

## §1 — Five-phase pipeline

`moe_kernel_topk` → `moe_kernel_topk_BS8` runs five phases separated by two
grid-wide syncs. Warps within a block split into **calc warps** `[0, 8)` and
**prefetch warps** `[8, 12)`; the TMA launcher thread is warp 8 lane 0 (see §6).

| Phase | What | Sync after |
|-------|------|------------|
| 1 | Routing (softmax/sigmoid top-K + optional renormalize) on calc warps, **in parallel** with the prefetch warps issuing the routing-window TMA that pulls the full per-block BF16 input tile into `bf16_in_full`. | block `__syncthreads()` |
| 2 | warp 0 runs `prepare_moe_topk_BS8` (expert ids, `sorted_slot`, `expert_slot_start[]`); warps 1..11 wait on `bar_rwin` then quantize BF16→FP8 into `fp8_act_full` + `act_scale`. | block `__syncthreads()` |
| 3 | Up-projection: streaming dual-warpgroup K=128 WGMMA reads FP8 directly from `fp8_act_full` → SiLU → FP8 writeback to `spec->temp_fp8`. | **grid:** Expert_Barrier (§2, site #2) |
| 4 | Down-projection: streaming WGMMA; each block `atomicAdd`s its fp32 partial into the single-buffer `spec->down_partial_out[tok][col]`. | **grid:** ColStripe_Barrier (§2, site #3) |
| 5 | Cast fp32 → bf16 into `activations_out`. | — |

The two grid syncs are Partial_Barriers, not full `grid_barrier`s: phase 3→4 only
needs the producer/consumer blocks for a given expert group to rendezvous, and
4→5 only the blocks sharing a column stripe. See §2 for why that is sufficient
and §6 for the group geometry (`UP_GRID`, `UP_GROUPS`, `DOWN_GRID`,
`DOWN_GROUPS`).

---

## §2 — Software grid / partial barriers (seed + high-bit protocol)

Defined in `csrc/fused_moe/monomoe/src/moe_grid_barrier.h`. `grid_barrier` (all blocks) and
`partial_barrier` (a caller-specified arrival set) give
cooperative-groups-equivalent happens-before with a standard launch. The thin
aliases `expert_barrier` / `colstripe_barrier` are `partial_barrier` with the
per-site argument recipe baked in.

**Counter layout.** Each barrier owns a *Counter_Pair* — two `uint32_t` slots
selected by `phase & 1` (ping-pong). `grid_barrier` uses one pair
(`spec->grid_barrier.slot[2]`); `partial_barrier` uses one pair per id
(`expert_slot[NUM_EXPERTS][2]`, `colstripe_slot[DOWN_GRID][2]`).

**Protocol** (`N` = arrival count):

1. `__syncthreads()` — publish pre-barrier per-thread writes within the block.
2. thread 0 issues `__threadfence()` — release the block's pre-barrier *global*
   writes so other SMs observe them before our arrival.
3. The **seed block** `atomicExch`es the slot to
   `SEED = 0x80000000u - (N - 1)`, then folds back any early arrivals
   (`prior & 0x7FFFFFFFu`). Non-seed blocks `atomicAdd(c, 1)`. When all `N`
   blocks have contributed, the slot equals `0x80000000u` (bit 31 set).
4. thread 0 spins on `atomicAdd(c, 0u)` (an uncached device-scope read, same
   codegen as `ld.acquire.gpu` on SM90) until bit 31 flips. Only thread 0 spins;
   the rest of the block waits at the step-6 `__syncthreads()`.
5. `__threadfence()` — post-barrier reads/writes happen-after the bit-31
   observation.
6. `__syncthreads()` — re-gather the block and release the non-spinning threads.
7. `++phase` — next call targets the other ping-pong slot.

**Why the high bit, and why two slots.** The high bit `0x80000000u` doubles as
the *exit-state marker* left by the previous call on a slot. On entry a slot may
be `0`, `k` (early arrivals), `0x80000000u` (leftover marker), or
`0x80000000u + k`. Masking the seeder's `prior` with `0x7FFFFFFFu` strips the
leftover marker and keeps only the real early-arrival count, so the invariant
`SEED + (N - 1) = 0x80000000u` holds regardless of `atomicExch`/`atomicAdd`
interleaving. The two ping-pong slots guarantee call `N+2`'s seeder cannot
observe a stale high bit from call `N`. This is why the slots are
**self-maintaining**: only the *first* use of a buffer needs host zero-init
(§4); every call afterward resets its slot via the step-3 `atomicExch`.

**Bound.** `N ≤ GRID_SIZE ≤ 132` on H200, so `SEED ≥ 0x80000000u - 131` never
wraps. The protocol is correct for any `1 ≤ N ≤ 2³¹`.

**Degenerate `N == 1`** short-circuits (a lone block has nothing to order) but
still `++phase`. `partial_barrier` takes `arrival_count` as a runtime argument
(not a template param) so the primitive isn't instantiated twice; every call
site passes a compile-time constant (`UP_GRID`, `DOWN_GROUPS`), so the compiler
still folds `SEED` and the degenerate gate.

**Caller contract.** A block MUST NOT call `partial_barrier` for an `id` whose
arrival set it isn't in. Each `phase` is per-(region, id-at-call-site) register
state, initialized to 0 at kernel entry. Seed-block / arrival-set recipe per
site:

- **Site #2 Expert_Barrier** (phase 3→4): `id = up_group`,
  `arrival_count = UP_GRID = 8`, `seed_blockidx = up_group * UP_GRID`. Arrival
  set = the 8 blocks with `blockIdx.x / UP_GRID == up_group` — exactly the
  blocks that wrote expert group `g`'s `temp_fp8` rows and will read them back.
- **Site #3 ColStripe_Barrier** (phase 4→5): `id = blockIdx.x % DOWN_GRID`,
  `arrival_count = DOWN_GROUPS = 16`, `seed_blockidx = id`. The Phase-5 writer
  for col stripe `c` is `blockIdx.x == c`, which is in the arrival set and is
  the seed.

---

## §3 — Co-residency invariant (why the spin can't deadlock)

The barrier spin (§2 step 4) is only deadlock-free if every participating block
is resident on the GPU for the kernel's whole lifetime — a block must never spin
waiting on a block that hasn't been scheduled yet. The kernel guarantees this:

- `GRID_SIZE <= SM_count` so every block gets a slot. The host checks this once
  before launch (`csrc/fused_moe/monomoe/monomoe_wrapper.cuh`); `GRID_SIZE` is a
  compile-time constant and SM count is a static device property.
- `__launch_bounds__(BLOCK_SIZE, 1)` pins one block per SM.
- Opt-in dynamic SHM (> half the per-SM budget) reinforces one-block-per-SM
  occupancy.

---

## §4 — Scratchpad (`MoEGemmSpec<Dims>`) and zero-init

The global scratchpad is reinterpreted as a `MoEGemmSpec<Dims>` (layout in
`csrc/fused_moe/monomoe/src/moe_internal.h`). It holds the inter-phase tensors (`temp_bf16`,
`temp_fp8`, `down_partial_out`) and, at its **tail**, the barrier Counter_Pairs
(`grid_barrier`, `partial_barrier`). The host sizes the buffer from
`sizeof(MoEGemmSpec<Dims>)` (exported via `monomoe_scratchpad_size`) so Python
never re-derives the layout.

**Zero-init discipline.** The barrier counters must start at 0 so the first
seeder's `atomicExch` commits the seed cleanly (§2). After that the ping-pong
reset makes the slots self-maintaining, so a given buffer pays the zero-init
*once*. The host keys a guard on `(ptr, size, device)` and `cudaMemsetAsync`es
the whole scratchpad whenever the buffer identity changes — a process-wide
one-shot flag would be wrong, because it would leave a second, distinct
scratchpad (different stream/device, or a freshly-malloc'd buffer) with
uninitialized counters and deadlock the spin. Zeroing the entire scratchpad
(not just the counter tail) is simpler and costs a few hundred µs once on H200.

**`TEMP_FP8_OFFSET` invariant.** The host builds the down-activation TMA
descriptor (§5) by adding the compile-time `MoEGemmSpec<Dims>::TEMP_FP8_OFFSET`
to the scratchpad base. That constant MUST equal
`offsetof(MoEGemmSpec<Dims>, temp_fp8)`, enforced by a `static_assert` in
`csrc/fused_moe/monomoe/monomoe_wrapper.cuh`. New fields (e.g. extra barrier
counters) MUST be appended
*after* `temp_fp8` — anything inserted before it shifts the offset and silently
corrupts TMA fetches.

---

## §5 — TMA descriptors and the tile-major SHM layout

The BS8 path loads weights and activations with TMA
(`cp.async.bulk.tensor.2d`). Host-side `create_*_tma_desc` factories
(`csrc/fused_moe/monomoe/src/moe_tma.cu`, declared in `…/src/moe_tma.h`) build
the `CUtensorMap`s; the kernel takes them as `__grid_constant__ CUtensorMap
const` params.

**Caller contracts** (encoded once in the factory doc comments):

- **Up-projection weights** MUST be pre-interleaved in Python via
  `interleave_for_tma_wgmma_up` (gate/up row interleave) so a single 128×128 TMA
  fetches one full WGMMA A-tile. SWIZZLE_128B.
- **Down-projection weights** MUST be passed *raw* row-major `[E, K, N]` — the
  TMA hardware applies the 8-row × 128-byte core-matrix XOR swizzle at write
  time. SWIZZLE_128B, `row_box = DOWN_COL_TILE`.
- **Activations** use SWIZZLE_NONE. The down-activation descriptor reads
  `spec->temp_fp8` inside the scratchpad (§4).

**Tile-major SHM (`bf16_in_full`, `fp8_act_full`).** The BF16 input buffer is
shaped `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]`, not the natural
`[BS][HIDDEN_STATES]`. The activation TMA descriptor is configured with
`boxDim = (128, 8)` (innermost = K, outer = tokens), SWIZZLE_NONE, so each
`cp.async.bulk.tensor.2d` writes a compact 8×128 BF16 box (2 KB) whose outer-row
stride equals the inner box dim (256 B), not the logical row stride. With a
`[BS][HIDDEN_STATES]` layout (row stride 4096 B for Qwen3.5) successive K-substep
writes would overlap and corrupt each other. The tile-major layout gives each
K-substep its own self-contained 2 KB slab matching exactly the bytes the TMA
writes, and consumers read `bf16_in_full[kblk][token]` as a natural
`[K_STEP_WGMMA]` row. Total size is unchanged (16 × 8 × 128 × 2 = 32 KB), so the
union with the weight tiles and the ≤ 228 KB per-block SHM budget are unaffected.

Why 16 × 2 KB issues instead of one 32 KB issue: `cuTensorMapEncodeTiled` caps
per-axis `boxDim` at 256 elements, so a single-issue load with innermost axis
= `HIDDEN_STATES = 2048` is rejected by the Driver API.

The `temp_fp8` tensor (down-projection input) is written by the Phase-3 epilogue
in a reorganized `[expert, token]` layout, so each expert's routed tokens form a
contiguous slab `[expert_slot_start[id], + routed_token_count[id])`.

---

## §6 — Warp roles and group geometry

**Warps** (`BLOCK_SIZE = 384` = 12 warps): calc warps `[0, 8)` do routing/top-K;
prefetch warps `[8, 12)` drive TMA and BF16→FP8 quantization. The TMA launcher
thread is warp 8 lane 0 (`is_tma_launcher_thread<Dims>()`); mbarrier arms and
TMA issues run on it alone, gated so other threads never block on an
uninitialized mbarrier parity.

**Up-projection groups** (Phase 3): `UP_GRID = 2*N / W_UP_TILE_EFFECTIVE = 8`
blocks cover one expert's `2*N` weight rows;
`UP_GROUPS = GRID_SIZE / UP_GRID = 16` expert groups run in parallel. Group `g`
(blocks `[g*UP_GRID, (g+1)*UP_GRID)`) iterates experts from index `g` stepping by
`UP_GROUPS`. Groups write disjoint `temp_fp8` slabs, so no write conflict.

**Down-projection groups** (Phase 4): blocks partition into
`DOWN_GROUPS = 16` expert groups × `DOWN_GRID = 8` column blocks; each block owns
`DOWN_COL_TILE = 256` output cols. Phase 2a aligned `DOWN_GROUPS == UP_GROUPS` so
each Expert_Barrier producer set equals its consumer set (§2 site #2). Every
contributing block `atomicAdd`s into the single-buffer `down_partial_out`; Phase
5 reads each cell once (no cross-group reduction).

---

## File map

Paths are relative to `csrc/fused_moe/monomoe/`.

| File | Role |
|------|------|
| `monomoe_wrapper.cuh` | host launcher template `monomoe_topk_launcher<Dims>` (checks §3, zero-inits §4, builds §5 descriptors, launches §1) |
| `monomoe_binding.cu` | TVM-FFI exports (`monomoe_topk`, `monomoe_scratchpad_size`) |
| `src/moe_interface.h` | `Dims` shape, `MoEDimensions`, public kernel decl |
| `src/moe.cuh` | `moe_kernel_topk` / `moe_kernel_topk_BS8` — the §1 pipeline |
| `src/moe_grid_barrier.h` | §2 barriers |
| `src/moe_internal.h` | `MoEGemmSpec` (§4) and `MoE_SHM` layouts, `MoECoreDims` |
| `src/moe_routing.cuh` | Phase 1 routing / top-K (`topK_BS8`, `prepare_moe_topk_BS8`) |
| `src/moe_scale_inputs.cuh` | Phase 2 BF16→FP8 quantization |
| `src/moe_up_projection.cuh` | Phase 3 up-proj + SiLU |
| `src/moe_down_projection.cuh` | Phase 4 down-proj |
| `src/moe_tma.{h,cu}` | §5 TMA descriptor factories |
| `src/ptx_utils.h` | WGMMA / mbarrier / TMA inline-PTX wrappers |
