# Incremental SSU with MTP Replay — Equations & Kernel Description

## 1. Equations (per head, dropping batch/head indices)

### Index conventions

- `t`, `j` — token index within T (sequence position)
- `d` — index into `dim` (head dimension)
- `n` — index into `dstate` (SSM state dimension)

### Variables

| Symbol | Shape | Description |
|--------|-------|-------------|
| `h_0` | `[dim, dstate]` | Initial SSM state (before this call) |
| `x` | `[T, dim]` | New token inputs |
| `dt` | `[T]` | Raw timestep (scalar per token, tie_hdim) |
| `A` | scalar | Decay rate |
| `B` | `[T, dstate]` | Input projection |
| `C` | `[T, dstate]` | Output projection |
| `D` | `[dim]` | Skip connection |
| `z` | `[T, dim]` | Gate |
| `y` | `[T, dim]` | Output |

### Derived quantities

```
dt_proc[t]    = softplus(dt[t] + dt_bias)                          — [T]
cumAdt[t]     = sum_{s=0}^{t} (A * dt_proc[s])                    — [T], cumulative sum
decay_vec[t]  = exp(cumAdt[t])                                     — [T]
```

### Sequential recurrence (what we're computing, token by token)

```
h_t[d,n] = h_{t-1}[d,n] * exp(A * dt_proc[t])  +  B[t,n] * dt_proc[t] * x[t,d]
y[t,d]   = sum_n( C[t,n] * h_t[d,n] )  +  D[d] * x[t,d]
y[t,d]   = y[t,d] * z[t,d] * sigmoid(z[t,d])
```

### Closed-form unroll

Unrolling the recurrence from `h_0`:

```
h_t[d,n] = h_0[d,n] * exp(cumAdt[t])
         + sum_{j=0}^{t} exp(cumAdt[t] - cumAdt[j]) * dt_proc[j] * B[j,n] * x[j,d]
```

Substituting into the output equation and factoring out `x[j,d]`:

```
y[t,d] = decay_vec[t] * sum_n( C[t,n] * h_0[d,n] )
       + sum_{j=0}^{t} [ exp(cumAdt[t]-cumAdt[j]) * dt_proc[j] * sum_n(C[t,n]*B[j,n]) ] * x[j,d]
       + D[d] * x[t,d]
```

Define the `T x T` lower-triangular matrix:

```
CB_scaled[t,j] = exp(cumAdt[t]-cumAdt[j]) * dt_proc[j] * dot(C[t,:], B[j,:])    for j <= t
                 0                                                                 for j > t
```

### Final matrix form

```
init_out[T, dim]  =  (C[T, dstate] @ h_0[dim, dstate]^T)  *  decay_vec[T, 1]
cb_out[T, dim]    =  CB_scaled[T, T]  @  x[T, dim]
y[T, dim]         =  init_out  +  cb_out  +  D[1, dim] * x[T, dim]
```

---

## 2. Kernel implementation

### Kernel 1: Precompute (one thread block per `(batch, head)`)

Computes everything that doesn't depend on `dim`:

1. Load `dt[T]`, compute `dt_proc[T]` (add bias, softplus)
2. `cumAdt[T] = cumsum(A * dt_proc)` and `decay_vec[T] = exp(cumAdt)`
3. `decay_matrix[T, T]` where `[t,j] = exp(cumAdt[t] - cumAdt[j])`
4. Load `C[T, dstate]` and `B[T, dstate]`, compute `CB[T, T] = C @ B^T` via `tl.dot`
5. `CB_scaled[T, T] = CB * decay_matrix * dt_proc[1, T]` masked by causal
6. Write `B`, `dt_proc`, `cumAdt` to double-buffered cache (for next step's replay)

**Outputs**: `CB_scaled[T, T]` and `decay_vec[T]`

### Kernel 2: Main (one thread block per `(dim_tile, batch, head)`)

Each block handles `DIM_PER_CTA` elements of `dim`. Two phases.
(Following FlashInfer convention: `DIM_PER_CTA = DIM / CTAS_PER_HEAD`.)

#### Phase 1 — Replay (update `h_0` with `K` old accepted tokens from previous MTP step)

The same closed-form, applied to old cached tokens to fast-forward the state:

```
total_decay          = exp(old_cumAdt[K-1])                                     — scalar
coeff[t]             = exp(old_cumAdt[K-1] - old_cumAdt[t]) * old_dt_proc[t]    — [K]
dB_scaled[T, dstate] = coeff[T, 1] * old_B[T, dstate]

h_0[dim, dstate]  =  h_0[dim, dstate] * total_decay
                   +  old_x[T, dim]^T  @  dB_scaled[T, dstate]
                      ─────────────────────────────────────────
                      [dim, T]          @  [T, dstate]  →  [dim, dstate]
```

This is the batched rank-1 update `sum_t x[t] outer (B[t] * coeff[t])` done as one matmul.

#### Phase 2 — Output (compute `y` for the new T tokens)

Using `D` = `DIM_PER_CTA` for brevity in the shapes below:

```
init_out[T, D]  =  C[T, dstate]  @  h_0[D, dstate]^T   *   decay_vec[T, 1]
                   ──────────────────────────────────
                   [T, dstate]   @  [dstate, D]  →  [T, D]

cb_out[T, D]    =  CB_scaled[T, T]  @  x[T, D]
                   ─────────────────────────────
                   [T, T]           @  [T, D]  →  [T, D]

y[T, D]         =  init_out  +  cb_out  +  D_skip[D] * x[T, D]
y[T, D]         =  y[T, D] * z[T, D] * sigmoid(z[T, D])
```

Then writes current `x` into the `old_x` cache for next step's replay.

### Key insight

The sequential `T`-step recurrence is converted into three matrix multiplies
(`C @ B^T`, `C @ h^T`, `CB_scaled @ x`) which map well to `tl.dot` on the GPU,
avoiding a serial loop over tokens.

---

## 2b. Merged single-kernel description

One CTA per `(batch, head)`. Each CTA handles the full `dim`.
`D` = `dim`, `K` = `prev_num_accepted_tokens` (0..T, per cache slot). All arithmetic in f32.

### Inputs read from global memory

```
state[D, dstate]          — initial SSM state
old_x[K, D]               — cached x from previous MTP step (single-buffered)
old_B[K, dstate]           — cached B from READ buffer
old_dt_proc[K]             — cached processed dt from READ buffer (f32)
old_cumAdt[K]              — cached cumulative A*dt from READ buffer (f32)
x[T, D]                   — new token inputs
dt[T]                      — raw timestep (scalar per token, tie_hdim)
A                          — decay rate (scalar per head, tie_hdim)
B[T, dstate]               — input projection (new tokens)
C[T, dstate]               — output projection (new tokens)
D_skip[D]                  — skip connection (optional, tie_hdim → scalar per head)
z[T, D]                    — gate (optional)
dt_bias                    — bias added to dt before softplus (optional, scalar per head)
```

### Phase 0 — Load data into smem (all warps cooperate)

All warps cooperatively load from global memory into shared memory:
```
B[T, dstate], C[T, dstate]               — cp.async / LDG
x[T, D], z[T, D]                          — cp.async
old_x[T, D], old_B[T, dstate]             — LDG
old_dt_proc[T], old_cumAdt[T]              — LDG (small, single thread)
dt[T] → dt_proc[T] (add bias, softplus)   — LDG + compute, store to smem
```

`cp.async.commit + cp.async.wait_group 0 + __syncthreads()`

### Phase 1a — Precompute CB_scaled and decay_vec (warp 0, does NOT depend on dim)

```
cumAdt[t]         = cumsum(A * dt_proc[t])                                     — [T], serial
decay_vec[t]      = exp(cumAdt[t])                                             — [T]

CB[T, T]          = C[T, dstate] @ B[T, dstate]^T                             — matmul
                    ──────────────────────────────
                    [T, dstate]  @  [dstate, T]  →  [T, T]

CB_scaled[T, T]   = CB * decay_matrix * dt_proc[1, T] * causal_mask           — elementwise
```

> **Implementation note:** `decay_matrix` and `causal_mask` are never materialized as arrays.
> Each CB_scaled element is computed inline as:
> `CB_scaled[t,j] = (t >= j) ? CB[t,j] * exp(cumAdt[t] - cumAdt[j]) * dt_proc[j] : 0`

**Cache writes:**
```
WRITE buffer old_B[T, dstate]    ← B[T, dstate]       (new B for next step's replay)
WRITE buffer old_dt_proc[T]      ← dt_proc[T]
WRITE buffer old_cumAdt[T]       ← cumAdt[T]
```

### Phase 1b — Replay (warps 1-3, fast-forward state with K old cached tokens)

> **Possible optimization:** `old_cumAdt` can be recomputed from `old_dt_proc` and `A`
> (`cumAdt = cumsum(A * dt_proc)`, serial scan over T≤16 elements). This would eliminate
> one cached tensor and one global memory round-trip. Keeping it cached for now.

```
total_cumAdt         = old_cumAdt[K-1]                                          — scalar
total_decay          = exp(total_cumAdt)          (1.0 if K == 0)               — scalar
coeff[t]             = exp(total_cumAdt - old_cumAdt[t]) * old_dt_proc[t]       — [K]
                       (zeroed for t >= K)
dB_scaled[K, dstate] = coeff[K, 1] * old_B[K, dstate]                          — elementwise

state[D, dstate]    =  state[D, dstate] * total_decay                          — scale
                     +  old_x[K, D]^T  @  dB_scaled[K, dstate]                 — matmul
                        ───────────────────────────────────────
                        [D, K]          @  [K, dstate]  →  [D, dstate]
```

### __syncthreads()

Phase 1a results (CB_scaled, decay_vec) are in smem. Phase 1b updated state in registers.

> **State writeback can overlap with Phase 2.** State stores (`ST.GLOBAL`) are non-blocking.
> They can be issued right after the sync — state is in registers, and Phase 2 only reads
> those same registers for `C @ state^T`. No conflict. This gives maximum overlap: stores
> fly in the background while Phase 2 computes.

### Phase 2 — Output (all 4 warps, compute y for T new tokens)

```
init_out[T, D]  =  ( C[T, dstate]  @  state[D, dstate]^T )  *  decay_vec[T, 1]
                     ───────────────────────────────────────
                     [T, dstate]   @  [dstate, D]  →  [T, D]

cb_out[T, D]    =  CB_scaled[T, T]  @  x[T, D]                                — matmul
                   ─────────────────────────────
                   [T, T]           @  [T, D]  →  [T, D]

y[T, D]         =  init_out  +  cb_out  +  D_skip * x[T, D]                   — elementwise

if z:
    y[T, D]     =  y[T, D] * z[T, D] * sigmoid(z[T, D])                       — gating
```

**Cache write:**
```
old_x[T, D]  ← x[T, D]           (single-buffered, replay already read it)
```

### Phase 2 — State writeback

```
state[D, dstate]  → global memory   (encode: direct for f32/bf16, SR for f16, quantize for int16)
```

### Output write

```
y[T, D]  → output[T, D]           (in input_t dtype)
```

### Summary of matmuls

| # | Operation | Shape | When |
|---|-----------|-------|------|
| 1 | `C @ B^T` → CB | `[T, dstate] @ [dstate, T] → [T, T]` | Phase 0a |
| 2 | `old_x^T @ dB_scaled` → state update | `[D, K] @ [K, dstate] → [D, dstate]` | Phase 0b |
| 3 | `C @ state^T` → init_out | `[T, dstate] @ [dstate, D] → [T, D]` | Phase 1 |
| 4 | `CB_scaled @ x` → cb_out | `[T, T] @ [T, D] → [T, D]` | Phase 1 |

---

## 3. CUDA implementation design decisions

### Single kernel (no precompute/main split)

The Triton reference uses two kernels: precompute (`CB_scaled`, `decay_vec`) and main
(replay + output). This was done so precompute can overlap with replay via PDL.

For the CUDA version: **merge into a single kernel**. The overlap benefit is minimal
(both phases are tiny), and merging eliminates:
- Kernel launch overhead (~1-2 µs, significant when total is ~4-8 µs)
- Global memory round-trip for `CB_scaled[T,T]` and `decay_vec[T]`
- PDL synchronization complexity

Within a single CTA, the CB computation and replay can still overlap at the
warp level (different warps do independent work before a `__syncthreads()`).

### MMA instruction choice: Ampere `mma.sync.m16n8k16`

Blackwell's `tcgen05.mma` has minimum tile 64x64 (1SM) or 128x64 (2SM), with
accumulator in TMEM. Hopper's `wgmma` has minimum M=64. Our matmul dimensions
(T=4-16, M=4-64) are smaller than these minimums in every case.

Ampere `mma.sync.m16n8k16.f32.bf16` is the right fit:
- 16x8 output tile, k=16 — matches our small dimensions
- Both operands from registers (no TMEM/smem requirement)
- Accumulators stay in f32 registers for subsequent element-wise ops
- Backward-compatible on Blackwell

Matmul sizes and mma.sync tile counts (T=16, DIM_PER_CTA=32, dstate=128):

| Operation | Shape | Output tiles | K-steps | Total mma insts |
|-----------|-------|-------------|---------|-----------------|
| `C @ B^T` | `[16,128] @ [128,16]` | 1x2 | 8 | 16 |
| `old_x^T @ dB_scaled` | `[32,16] @ [16,128]` | 2x16 | 1 | 32 |
| `C @ h^T` | `[16,128] @ [128,32]` | 1x4 | 8 | 32 |
| `CB_scaled @ x` | `[16,16] @ [16,32]` | 1x4 | 1 | 4 |

### Swizzled smem for mma.sync

`mma.sync` takes operands from registers, but to load them from shared memory
without bank conflicts, the smem layout must be swizzled. Data flow:

1. Load from global memory → swizzled smem
2. `ldmatrix` from smem → register fragments (bank-conflict-free)
3. `mma.sync.m16n8k16` on register fragments
4. Accumulators stay in f32 registers

### SIMT-first, then tensor ops

Implementation strategy: start with SIMT matmul functions, then swap in mma.sync
via separate function variants (not `if constexpr` inside one function):

```cpp
// SIMT path
compute_cb_scaled_simt(...)
replay_state_simt(...)
compute_output_simt(...)

// Tensor core path
compute_cb_scaled_mma(...)
replay_state_mma(...)
compute_output_mma(...)
```

The caller dispatches based on a template parameter:

```cpp
if constexpr (UseTensorOps)
    compute_cb_scaled_mma(...)
else
    compute_cb_scaled_simt(...)
```

This keeps each function clean and testable independently.

### No TMA

Benchmarking showed TMA only helps at batch >= 64. At small batch, the data per
CTA is tiny (state: 8KB, B/C: 4KB each, x: 1KB). Regular loads or `cp.async`
are sufficient. TMA's descriptor setup cost dominates at small batch.

### State dtype handling

| State dtype | Decode (read) | Encode (write) |
|-------------|--------------|----------------|
| `float32` | Direct | Direct |
| `bfloat16` | Direct | Direct |

All matmul accumulation happens in **f32**. State is converted at load/store boundaries.

> **Future work:** fp16 state with stochastic rounding, int16 block-scaled state
> (`val * decode_scale` on read, `quantize(val) + encode_scale` on write).
> These require `rand_seed` and per-dim-row scale factors respectively.

### CTA structure

Grid: `(batch, nheads)`, NUM_WARPS=4 warps (128 threads) per CTA.
Each CTA processes the full `dim` for one (batch, head). CTAS_PER_HEAD optimization deferred.

```
Phase 0 — Load data into smem (all warps cooperate):
  All warps: cp.async / LDG B[T,dstate], C[T,dstate], x[T,dim], z[T,dim],
             old_x[T,dim], old_B[T,dstate] into smem.
             LDG dt[T] → compute dt_proc[T] → smem.
             LDG old_dt_proc[T], old_cumAdt[T] → smem.
  cp.async.commit + wait + __syncthreads()

Phase 1 — CB + Replay + Prefetch (overlap across warps):
  Warp 0:    CB_scaled[T,T], decay_vec[T] → smem (from B,C,dt_proc already in smem)
             Cache writes (old_B, old_dt_proc, old_cumAdt) to WRITE buffer
  Warps 1-2: Replay state — load state from global, apply decay + old_x^T @ dB_scaled
  Warp 3:    (idle or assist replay)
  __syncthreads()

Phase 2 — Output + State writeback (all 4 warps, partition dim):
  Issue state writeback stores (non-blocking, overlaps with output compute)
  All 4 warps: C @ state^T, CB_scaled @ x, combine, z-gate, store output
               Write old_x cache
```

---

## 4. Implementation plan

### Goal

Create a standalone `ssu_incremental` kernel in CUDA that implements the matmul-based
incremental SSU algorithm. Initially uses SIMT matmuls, with mma.sync as a future
optimization.

### Interface (matches Triton reference)

```
ssu_incremental(
    state,                      # (cache, nheads, dim, dstate)  — in-place
    old_x,                      # (cache, T, nheads, dim)       — single-buffered
    old_B,                      # (cache, 2, T, ngroups, dstate) — double-buffered
    old_dt_proc,                # (cache, 2, nheads, T)         — double-buffered
    old_cumAdt,                 # (cache, 2, nheads, T)         — double-buffered
    cache_buf_idx,              # (cache,) int32
    prev_num_accepted_tokens,   # (cache,) int32
    x,                          # (batch, T, nheads, dim)
    dt,                         # (batch, T, nheads, dim) tie_hdim
    A,                          # (nheads, dim, dstate) tie_hdim
    B,                          # (batch, T, ngroups, dstate)
    C,                          # (batch, T, ngroups, dstate)
    out,                        # (batch, T, nheads, dim)
    D=None,                     # (nheads, dim)
    z=None,                     # (batch, T, nheads, dim)
    dt_bias=None,               # (nheads, dim) tie_hdim
    dt_softplus=False,
    state_batch_indices=None,   # (batch,) int32
)
```

### Files to create/modify

#### New files

| # | File | Description |
|---|------|-------------|
| 1 | `include/flashinfer/mamba/ssu_incremental.cuh` | Params struct + kernel dispatcher |
| 2 | `include/flashinfer/mamba/kernel_ssu_incremental.cuh` | CUDA kernel implementation |
| 3 | `csrc/ssu_incremental.cu` | C++ launcher (tensor validation + params) |
| 4 | `csrc/ssu_incremental_kernel_inst.cu` | Template instantiation |
| 5 | `csrc/ssu_incremental_jit_binding.cu` | TVM-FFI export |
| 6 | `csrc/ssu_incremental_customize_config.jinja` | Jinja config template |
| 7 | `flashinfer/jit/mamba/ssu_incremental.py` | JIT module generator |
| 8 | `flashinfer/mamba/ssu_incremental.py` | Python API |
| 9 | `tests/mamba/test_ssu_incremental.py` | Test (mirrors TRT-LLM test pattern) |

#### Files to modify

| File | Change |
|------|--------|
| `flashinfer/mamba/__init__.py` | Export `ssu_incremental` |
| `flashinfer/__init__.py` | Export from mamba (if needed) |

### Implementation steps

#### Step 1 — Scaffolding: Jinja config, JIT generator, binding, Python API
**EASY** — Boilerplate following existing `selective_state_update` pattern.

- [x] 1.1 Create `csrc/ssu_incremental_customize_config.jinja`
  - Template parameters: `state_dtype`, `input_dtype`, `dt_dtype`, `weight_dtype`, `dim`, `dstate`,
    `ntokens_mtp`, `philox_rounds`, `state_scale_type`
  - `dt_t` is a separate type so the kernel accepts dt in its native dtype (e.g. bf16)
    and converts to f32 internally — eliminates the host-side dtype conversion kernel launch
    that the old FI kernel requires.
  - Ref: `csrc/selective_state_update_customize_config.jinja`

- [ ] 1.2 Create `csrc/ssu_incremental_jit_binding.cu`
  - Single TVM-FFI export: `TVM_FFI_DLL_EXPORT_TYPED_FUNC(ssu_incremental, ...)`
  - Ref: `csrc/flashinfer_mamba_binding.cu:58`

- [ ] 1.3 Create `flashinfer/jit/mamba/ssu_incremental.py`
  - `get_ssu_incremental_uri(...)` — unique cache key
  - `gen_ssu_incremental_module(...)` — render jinja, copy sources, return JitSpec
  - Sources: `ssu_incremental.cu`, `ssu_incremental_kernel_inst.cu`, `ssu_incremental_jit_binding.cu`
  - Ref: `flashinfer/jit/mamba/selective_state_update.py`

- [ ] 1.4 Create `flashinfer/mamba/ssu_incremental.py`
  - `ssu_incremental(...)` — public API, validates inputs, calls JIT module
  - `_ssu_incremental(...)` — `@register_custom_op`
  - `get_ssu_incremental_module(...)` — `@functools.cache`, builds/loads JIT module
  - Ref: `flashinfer/mamba/selective_state_update.py`

- [ ] 1.5 Update `flashinfer/mamba/__init__.py` — export `ssu_incremental`

#### Step 2 — Params struct and C++ launcher
**EASY** — Tensor validation and params population.

- [ ] 2.1 Create `include/flashinfer/mamba/ssu_incremental.cuh`
  - `SsuIncrementalParams` struct with all tensor pointers, strides, dimensions
  - Forward-declare `launchSsuIncremental<>()`
  - Ref: `include/flashinfer/mamba/selective_state_update.cuh` (SelectiveStateMTPParams)

- [ ] 2.2 Create `csrc/ssu_incremental.cu`
  - `ssu_incremental(...)` — validate tensors, populate params, call dispatcher
  - Validate: shapes, dtypes, tie_hdim strides, cache dimensions
  - Ref: `csrc/selective_state_update.cu` (run_selective_state_update_mtp)

- [ ] 2.3 Create `csrc/ssu_incremental_kernel_inst.cu`
  - `#include "ssu_incremental_config.inc"`
  - Explicit instantiation of `launchSsuIncremental<>()`

#### Step 3 — CUDA kernel (SIMT matmuls)
**HARD** — Core algorithm implementation.

Grid: `(batch, nheads, CTAS_PER_HEAD)`, 4 warps per CTA.

- [ ] 3.1 Create `include/flashinfer/mamba/kernel_ssu_incremental.cuh`
  - Shared memory layout struct
  - Data loading functions (state, old cache, new inputs)

- [ ] 3.2 Implement Phase 0a — Compute `CB_scaled[T,T]` and `decay_vec[T]`
  - Load dt[T], apply bias + softplus → dt_proc[T]
  - cumAdt[T] = cumsum(A * dt_proc) — single-thread serial loop (T <= 20)
  - decay_vec[T] = exp(cumAdt[T])
  - decay_matrix[T,T] = exp(cumAdt[t] - cumAdt[j])
  - CB[T,T] = C @ B^T — SIMT matmul across warp
  - CB_scaled[T,T] = CB * decay_matrix * dt_proc * causal_mask
  - Store CB_scaled and decay_vec to smem

- [ ] 3.3 Implement Phase 0b — Replay (state fast-forward from old cache)
  - Load old_dt_proc, old_cumAdt from READ buffer (cache_buf_idx)
  - coeff[t] = exp(old_cumAdt[K-1] - old_cumAdt[t]) * old_dt_proc[t]
  - total_decay = exp(old_cumAdt[K-1])
  - state *= total_decay
  - dB_scaled = coeff * old_B
  - state += old_x^T @ dB_scaled — SIMT matmul
  - Can reuse helpers from existing kernel: `cp_async_16B`, conversion functions

- [ ] 3.4 Implement Phase 1 — Output
  - init_out[T, DIM_PER_CTA] = C @ h^T * decay_vec — SIMT matmul
  - cb_out[T, DIM_PER_CTA] = CB_scaled @ x — SIMT matmul
  - y = init_out + cb_out + D*x
  - y *= z * sigmoid(z) (optional gating)
  - Store output

- [ ] 3.5 Implement cache writes
  - Write old_x (single-buffered): current x → old_x cache
  - Write old_B, old_dt_proc, old_cumAdt to WRITE buffer (1 - cache_buf_idx)
  - Only done by dim_tile == 0 (one dim tile writes cache, others skip)

- [ ] 3.6 Implement state encode/decode at load/store boundary
  - float32 state: direct load/store
  - bfloat16 state: direct load/store
  - Reuse: `toFloat`, `convertAndStore` from existing kernel
  - Future: fp16 stochastic rounding, int16 block-scaled state

- [ ] 3.7 Implement `launchSsuIncremental<>()` dispatcher
  - Dispatch CTAS_PER_HEAD via `dispatchCtasPerHead` (same pattern as simple MTP kernel)
  - Dispatch HEADS_PER_GROUP via `dispatchRatio`
  - Compute grid `(batch, nheads, CTAS_PER_HEAD)`, shared memory size
  - Launch kernel
  - Ref: `invoke_selective_state_update_mtp.cuh` (simple kernel section)

#### Step 4 — Test
**MEDIUM** — Port TRT-LLM test pattern.

- [ ] 4.1 Create `tests/mamba/test_ssu_incremental.py`
  - Validate CUDA kernel against the Triton reference (`incremental_selective_state_update`)
  - For each k in 0..T: run both with identical inputs, compare output and state
  - Configs: (nheads=16, dim=64, dstate=128, ngroups=1), (nheads=32, dim=64, dstate=128, ngroups=2)
  - Parametrize: state_dtype (float32, bfloat16, float16), paged_cache (True/False)
  - Tolerances: rtol=2e-2, atol=5e-1 (bf16 matmul precision)
  - Ref: Triton reference at `tests/mamba/triton_reference/incremental_selective_state_update.py`

#### Step 5 — Validation and benchmarking
**MEDIUM**

- [ ] 5.1 Run test, verify correctness across all configs
- [ ] 5.2 Benchmark against Triton incremental and existing FI kernel
  - Measure at batch=1, mtp=4,8,12,16 with prev_k=0,half,full
  - Compare median_us

### Reusable code from existing kernels

From `kernel_selective_state_update_mtp_simple.cuh`:
- `cp_async_16B()` — 16-byte async copy (line 90)
- `cp_async_state_cooperative()` — cooperative state prefetch (line 103)
- `toFloat()`, `toFloat2()` — type conversion helpers
- `convertAndStore()`, `convertAndStoreSRHorizontal()` — state encode with stochastic rounding
- `computeBlockScaleEncode()` — block scaling for int16 state
- `PackedAligned<>` — vectorized load/store wrapper
- Shared memory bank-conflict-avoidance patterns

From `selective_state_update.cu` (launcher):
- `validate_state_tensor()`, `validate_A_tensor()`, etc. — tensor validation helpers
- `validate_dtype_consistency()` — dtype consistency checks

### Notes

- The kernel is standalone: separate binding, separate Python module, separate JIT spec
- No intermediate state writes — only final state after replay
- The state is updated in-place (replay fast-forward + decay through new T tokens is NOT done;
  only replay is applied to state, matching the Triton reference behavior)
- cache_buf_idx is NOT flipped by the kernel — caller manages buffer flipping
- **Finishing touch:** kernel uses `cp.async` (SM80+). The JIT Python module
  (`flashinfer/jit/mamba/ssu_incremental.py`) must set `supported_major_versions=[8, 9, 10, 11, 12]`
  to exclude SM75 and below.

---

## 5. Lessons learned from v0 kernel (ncu profiling)

**v0 kernel:** 273-384 µs at batch=128, mtp=8 vs 88 µs for the simple MTP kernel (3-4x slower).
Profile: `ssu_replay_03.ncu-rep`, kernel index 3.

### What went wrong

1. **571 MB register spill traffic** (17.9M local loads vs 1.6M global loads = 11x more spill
   than actual DRAM work). Root cause: `compute_y()` is a monolithic function with too many
   live variables — C @ state^T reduction (acc + shuffle chain), cb_out loop (T iterations
   of smem reads), decay_vec, D*x, all simultaneously live. Compiler caps at 37 regs/thread
   and spills everything else.

2. **state[DIM][DSTATE] in smem** (16KB for bf16). Every thread reads state through smem for
   `C @ state^T` with strided access → 26-way bank conflicts on `LDS.128` at line 273.
   31M shared load wavefronts total.

3. **cb_out + D*x computed by lane 0 only** after warp reduction for init_out. 31/32 lanes
   idle during T × ROWS_PER_WARP iterations of cb_out accumulation.

4. **1.6% DRAM throughput** — the kernel is doing almost zero useful memory work. Entirely
   bottlenecked on spills and smem bank conflicts.

5. **Barrier stall at __syncthreads() = 17.4%** — warps 0 and 3 finish Phase 1 fast (CB_scaled
   and idle) while warps 1-2 are still doing replay through smem.

6. **FMA pipe only 19%** — most time spent on integer address math (36% INT) and data movement.

### Key takeaways for v1 redesign

- **Keep functions small** to avoid register spill. Process one (t, d) output at a time,
  or split into `__noinline__` phases so the compiler can reuse registers.

- **All lanes must work** during output computation. cb_out and D*x should be distributed
  across lanes, not gated behind `lane == 0`.

- **No cross-lane reductions.** See section 6 below.

---

## 6. v1 redesign: fused-accumulator, no reductions

### Core insight: independent matmuls, shared accumulator

Both output matmuls produce the same `[T, D]` shape and get summed:

```
y[t, d]  = 0
y[t, d] += sum_n( C[t,n] * state[d,n] ) * decay_vec[t]   // matmul 1, K = dstate
y[t, d] += sum_j( CB_scaled[t,j] * x[j,d] )               // matmul 2, K = T
y[t, d] += D * x[t,d]                                      // elementwise
```

These are **completely independent terms of a sum**. No intertwined dependency (unlike
flash attention where softmax couples QK^T and P@V). Just two matmuls accumulated into
the same output register, one after the other.

Each thread owns specific `(t, d)` output elements and does the full dot products locally.
**No warp shuffles. No cross-lane reductions.** Just sequential accumulation.

### Thread-to-output mapping

Output `y[T, D]` has `T × D` elements. Each thread owns `(T × D) / 128` elements.

**Assignment:** each thread owns a fixed `t` and a contiguous chunk of `d` values.

```
128 threads / T rows = 128/T threads per row
Each thread handles D / (128/T) = D*T/128 consecutive d values
```

Example (T=16, D=64): 8 threads per row, 8 d-values per thread.

```
thread_t      = flat_tid / (128 / T)         // which T row
thread_d_base = (flat_tid % (128 / T)) * (D*T/128)  // starting d index
```

This means:
- Threads sharing the same `t` broadcast-read the same `C[t,:]` and `CB_scaled[t,:]` from smem
- Threads with different `d` read different `state[d,:]` and `x[:,d]` rows — no conflicts
- Future mma.sync: the thread-to-output mapping matches the accumulator fragment layout
  (M=T tiles along output rows, N=D tiles along output columns)

### State in smem (staging buffer)

State lives in smem as a staging buffer `state[D][DSTATE_PAD]` (same as v0). This is
intentional — the global→smem load pattern is different from the register fragment layout
needed for future mma.sync (`ldmatrix` with swizzled smem). The SIMT version loads from
smem with scalar reads; the mma.sync version will use `ldmatrix`.

### Replay — register-local, no state register file

Each thread updates state in smem for its owned `d` rows:

```cpp
for (int d : my_d_values) {
  for (int n = 0; n < DSTATE; n++) {
    float s = toFloat(smem.state[d][n]) * total_decay;
    for (int t = 0; t < prev_k; t++) {
      s += toFloat(smem.old_x[t][d]) * coeff[t] * toFloat(smem.old_B[t][n]);
    }
    convertAndStore(&smem.state[d][n], s);
  }
}
```

Register pressure: `s` (1 float), `coeff[t]` (1), `old_x` (1), `old_B` (1), loop counters.
Very light — no risk of spills. State is read once from smem, updated, written back.
The inner loop over `prev_k` is at most T iterations (≤16).

### Output computation — fused, no reduction

```cpp
for (int d : my_d_values) {
  for (int t = 0; t < T; t++) {
    // --- init_out: K-loop over dstate ---
    float y = 0.f;
    for (int n = 0; n < DSTATE; n++) {
      y += toFloat(smem.C[t][n]) * toFloat(smem.state[d][n]);
    }
    y *= decay_vec[t];

    // --- cb_out: K-loop over T ---
    for (int j = 0; j < T; j++) {
      y += smem.CB_scaled[t][j] * toFloat(smem.x[j][d]);
    }

    // --- skip connection ---
    y += D_val * toFloat(smem.x[t][d]);

    smem.out[t][d] = y;
  }
}
```

Register pressure per output element: `y` (1 float), `C[t][n]` (1), `state[d][n]` (1),
`CB_scaled[t][j]` (1), `x[j][d]` (1), `decay_vec[t]` (1). About 6 live registers.
Processed one element at a time — **impossible to spill**.

### CB_scaled — all 128 threads cooperate

v0 assigned CB_scaled to warp 0 (32 threads), creating warp imbalance.
v1 distributes `T × T` = 256 elements across all 128 threads (~2 per thread).

```cpp
for (int idx = flat_tid; idx < T * T; idx += 128) {
  int t = idx / T, j = idx % T;
  if (j <= t) {
    float acc = 0.f;
    for (int n = 0; n < DSTATE; n++)
      acc += toFloat(smem.C[t][n]) * toFloat(smem.B[j][n]);
    smem.CB_scaled[t][j] = acc * expf(cumAdt[t] - cumAdt[j]) * dt_proc[j];
  } else {
    smem.CB_scaled[t][j] = 0.f;
  }
}
```

### Execution phases

```
Phase 0 — Load smem (all warps, cp.async + LDG)
  B, C, x, z, old_x, old_B, dt→dt_proc, old_dt_proc, old_cumAdt, state
  __syncthreads()

Phase 1 — Precompute (all threads)
  cumAdt, decay_vec, coeff — every thread computes same values (cheap, T≤16)
  CB_scaled[T,T] — distributed across 128 threads
  __syncthreads()

Phase 2 — Replay (each thread updates its d-rows of state in smem)
  state[d][n] = state[d][n] * total_decay + old_x^T @ dB_scaled
  __syncthreads()

Phase 3 — Output (each thread computes its (t,d) elements, no reduction)
  y = C·state * decay + CB_scaled·x + D*x → smem.out
  __syncthreads()

Phase 4 — z-gate + global stores (all threads cooperate)
  Apply silu gate, write output
  State writeback, cache writes (old_x, old_B, old_dt_proc, old_cumAdt)
```

### Why this fixes v0

| v0 Problem | v1 Fix |
|------------|--------|
| 571 MB register spill | 6 live regs per output element. Impossible to spill. |
| Lane 0 bottleneck (31 idle) | Every thread computes its own (t,d) elements. Full utilization. |
| Warp-reduce over dstate | No reductions. Each thread does full dot product locally. |
| Warp imbalance (CB phase) | CB_scaled distributed across all 128 threads. |
| smem bank conflicts on state | Deferred — address with padding/swizzle after correctness. |

### Smem budget (T=16, D=64, DSTATE=128, bf16)

| Array | Size |
|-------|------|
| state[64][128] (bf16) | 16 KB |
| B[16][128], C[16][128] (bf16) | 4+4 KB |
| x[16][64], z[16][64] (bf16) | 2+2 KB |
| old_x[16][64], old_B[16][128] (bf16) | 2+4 KB |
| CB_scaled[16][16] (f32) | 1 KB |
| out[16][64] (f32) | 4 KB |
| decay_vec, dt_proc, old_dt_proc, old_cumAdt (f32) | ~256 B |
| **Total** | **~39 KB** |

### Notes

- The d-loop and t-loop order can be swapped to optimize smem reuse (process all t for
  one d before moving to next d → reuse `state[d,:]` across T tokens).
- State smem layout will need DSTATE_PAD for bank conflicts eventually, but correctness first.
- The thread-to-output mapping is chosen to match mma.sync.m16n8k16 accumulator layout
  for a smooth transition to tensor cores.

---

## 7. v1 profiling results

### Benchmark

```bash
# Timing sweep (batch=64, fp16 state, bf16 activations)
docker exec flashinfer-cu130-dev-ishovkun python \
    /home/scratch.ishovkun_gpu/code/flashinfer-dev/benchmarks/bench_incremental_ssu.py \
    --batch-sizes 64 --mtp-lengths 4,6,8,10,12,14,16 \
    --state-dtype float16 --warmup 10 --iters 30

# ncu profile (single config for detailed analysis)
ncu --target-processes all \
    python benchmarks/bench_incremental_ssu.py --profile \
    --batch-sizes 128 --mtp-lengths 8 --warmup 5 --iters 5
```

Benchmark results: `benchmarks/img/incremental_ssu_b64_f16.csv`

### Timing comparison (batch=64, mtp=8, fp16 state, bf16 activations)

| kernel | prev_k=0 | prev_k=4 | prev_k=8 | trend |
|--------|----------|----------|----------|-------|
| **cuda-incr (v0/v1)** | 61 µs | 84 µs | 94 µs | +54% slower with more accepted |
| **incremental (Triton)** | 20.5 µs | 20.5 µs | 21.5 µs | flat |
| **fi-replay (simple MTP)** | 29 µs | 27 µs | 25 µs | faster (fewer output steps) |
| **flashinfer (baseline)** | 35 µs | — | — | no replay |

### ncu profile: v1 kernel (batch=128, mtp=8, prev_k=4)

Profile: `/home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_replay_05.ncu-rep`, kernel index 1.

```
Duration:              271.9 µs
Registers/thread:      36
Dynamic smem:          27.4 KB
Achieved occupancy:    43.7%
```

| Metric | v0 | v1 | Target |
|--------|-----|-----|--------|
| Duration | 273-384 µs | 272 µs | ~20 µs |
| Register spill traffic | 571 MB | 260 MB | 0 |
| FMA pipe | 19% | 12% | >50% |
| INT compute (time) | 36% | 53.5% | <20% |
| Smem excessive wavefronts | 63% (26-way) | 63% (32-way) | <10% |
| Barrier stall | 17.4% | 20.5% | <5% |
| DRAM throughput | 1.6% | 2.2% | >30% |

### What v1 changed vs v0

- Split monolithic `compute_y` into `add_init_out`, `add_cb_out`, `add_D_skip`, `compute_z_gating`
- Per-warp row ownership throughout (no cross-warp syncs for output)
- No warp-level reductions — each lane does full dot products
- `compute_cumAdt` as warp-level prefix sum in smem
- `compute_CB_scaled` uses `simt_mma_nt_set` + elementwise decay/causal scaling
- CB_scaled still computed by warp 0 only (not yet distributed to all threads)

### What v1 did NOT fix

1. **32-way bank conflicts on state reads** — `add_init_out` reads `smem.state[dd][n]`
   where all lanes in a warp read different `dd` at the same `n`. Row stride
   `DSTATE * sizeof(bf16) = 256 bytes` = exact multiple of 128-byte bank cycle.
   Worse than v0 because each thread now reads all 128 dstate elements
   (vs v0 where 32 threads split the work).

2. **260 MB register spills** — still in `replay_state` (warps 1-2). The loop
   `for t in 0..prev_k` with old_x, coeff, old_B keeps too many variables live.

3. **20.5% barrier stall** — warp 0 does CB_scaled (T×T×DSTATE FMAs ≈ 32K),
   warps 1-2 do replay, warp 3 is idle. Massive imbalance.

4. **53.5% INT** — bf16→float conversions dominate. Each `toFloat(__nv_bfloat16)`
   generates IMAD.U32 + SHF.L.U32. The "no reduction" approach means MORE
   toFloat calls per thread (full dstate loop per output element).

---

## 8. v2: eliminate register spills

### Change

Removed the `coeff[NTOKENS]` register array from `replay_state`. Instead of
precomputing all coefficients into a local array, each coefficient is computed
inline inside the inner `t` loop:

```cpp
// v1: coeff[NTOKENS] precomputed, then indexed
coeff[t] = __expf(total_cumAdt - smem.old_cumAdt[t]) * smem.old_dt_proc[t];
...
val += toFloat(smem.old_x[t][dd]) * coeff[t] * toFloat(smem.old_B[t][n]);

// v2: compute on the fly, no array
for (int t = 0; t < prev_k; t++) {
    float coeff = __expf(total_cumAdt - smem.old_cumAdt[t]) * smem.old_dt_proc[t];
    val += toFloat(smem.old_x[t][dd]) * coeff * toFloat(smem.old_B[t][n]);
}
```

Trade-off: NTOKENS extra `__expf` calls per (dd, n) iteration, but eliminates
16 floats (64 bytes) of register pressure that was causing 260 MB of spill traffic.

### Timing comparison (batch=64, mtp=8, fp16 state, bf16 activations)

| kernel | prev_k | v1 (µs) | v2 (µs) | delta |
|--------|--------|---------|---------|-------|
| cuda-incr | 0 | 61.4 | 63.5 | ~same |
| cuda-incr | 4 | 83.9 | 81.9 | +2% |
| cuda-incr | 8 | 94.3 | 92.2 | +2% |

At mtp=12:

| kernel | prev_k | v1 (µs) | v2 (µs) | delta |
|--------|--------|---------|---------|-------|
| cuda-incr | 0 | 112.6 | 120.7 | -7% |
| cuda-incr | 6 | 174.1 | 162.8 | +6% |
| cuda-incr | 12 | 205.8 | 185.3 | +10% |

Win grows with prev_k (more replay work → more spill savings). Slight regression
at prev_k=0 is noise (replay loop doesn't execute).

### ncu profile: v2 kernel (batch=128, mtp=8, prev_k=4)

Profile: `/home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_replay_06.ncu-rep`, kernel index 1.

```
Duration:              257.5 µs
Registers/thread:      40
Dynamic smem:          27.4 KB
Achieved occupancy:    41.6%
```

| Metric | v1 | v2 | Target |
|--------|-----|-----|--------|
| Duration | 272 µs | 258 µs | ~20 µs |
| Register spill traffic | 260 MB | **0** | 0 |
| FMA pipe | 12% | 19.1% | >50% |
| INT compute (time) | 53.5% | 39.1% | <20% |
| Smem excessive wavefronts | 63% | 58.7% | <10% |
| Barrier stall | 20.5% | 28.5% | <5% |
| DRAM throughput | 2.2% | 2.3% | >30% |

### What v2 fixed

- **Register spills eliminated entirely** (260 MB → 0). Registers/thread went from 36 to 40
  (compiler uses more regs now that it doesn't have to spill).
- **INT compute down** from 53.5% to 39.1% (fewer spill-related integer address math).
- **FMA pipe up** from 12% to 19.1%.

### What v2 did NOT fix

1. **28.5% barrier stall** (worse than v1's 20.5%) — 97% of barrier stalls are at
   `__syncthreads()` line 679 (between replay/CB_scaled and output). Warp 0 (CB_scaled)
   and warp 3 (idle) finish early while warps 1-2 grind through replay.

2. **32.8% short_scoreboard** — now the top stall reason. Hotspots in `replay_state`:
   line 426 (inline `__expf` for coeff) and line 430 (FMA accumulation). Computing
   coeff on the fly adds a `MUFU.EX2` + dependent smem reads per inner-loop iteration.

3. **32-way bank conflicts on state reads** — still 58.7% excessive smem wavefronts,
   all at `add_init_out` line 463 reading `smem.state[dd][n]`.

---

## 9. v3 plan: CuTe mma.sync + swizzled smem

### Goal

Replace SIMT matmuls in the output phase with Ampere `mma.sync.m16n8k16` via CuTe,
and use swizzled smem layouts to eliminate the 32-way bank conflicts on state reads.

### Design decisions

1. **Ampere `mma.sync.m16n8k16` only** — M=16 (NTOKENS) fits a single m16 tile.
   Hopper wgmma (minimum M=64) and Blackwell tcgen05 (minimum M=64/128) are too large.
   `mma.sync` is backward-compatible on all SM80+ GPUs.

2. **CuTe C++ API** — already used in FlashInfer (`decode_mla_cute_sm80.cuh`).
   Provides swizzled smem layouts, ldmatrix-based s2r copies, and `cute::gemm()` dispatch
   to mma.sync. We follow the established patterns from that file.

3. **No TMA** — loads use cp.async and plain LDG as before. Data per CTA is small
   (state: 16KB, B/C: 4KB each). TMA descriptor setup cost dominates at small batch.
   Keeping data in original (non-transposed) layout in smem for future TMA compatibility.

4. **Swizzled smem for mma operands** — bank-conflict-free ldmatrix access.
   Non-mma arrays (CB_scaled f32, old_x, old_B, scalars) stay flat row-major.

5. **MMA atom**: `SM80_16x8x16_F32BF16BF16F32_TN` — bf16 operands, f32 accumulator.
   Triton reference casts all dot operands to bf16 (including f32 state and CB_scaled).
   We match that: all mma A/B operands are bf16, accumulator is f32.

### Matmul inventory

| # | Operation | Shape (M,N,K) | mma tiles | K-steps | Notes |
|---|-----------|---------------|-----------|---------|-------|
| 1 | `C @ B^T` → CB | (16,16,128) | 1×2 | 8 | output f32, then scale → CB_scaled |
| 2 | `old_x^T @ dB_scaled` → state | (64,128,K) | — | — | **keep SIMT** (dynamic K=0..16) |
| 3 | `C @ state^T` → init_out | (16,64,128) | 1×8 | 8 | biggest matmul, 32-way bank conflicts in SIMT |
| 4 | `CB_scaled @ x` → cb_out | (16,64,16) | 1×8 | 1 | CB_scaled must be cast to bf16 first |

Matmuls 3 and 4 share the same f32 accumulator fragment `[16, 64]`:
```
clear(accum)
gemm(mma, accum, C_frag, state_frag, accum)       // matmul 3
accum[i] *= decay_vec[t_i]                         // elementwise
gemm(mma, accum, CB_frag, x_frag, accum)           // matmul 4, accumulates
accum[i] += D_val * x[t_i, d_i]                   // skip connection
accum[i] *= z[t_i, d_i] * sigmoid(z[t_i, d_i])   // gating
store(accum → output)
```

### What stays SIMT

- **Matmul 2 (replay)**: K=prev_k is dynamic (0..16), not always a multiple of 16.
  Would need zero-padding to use mma, wasteful for small prev_k.
  Current SIMT replay has no spills after v2 and is not the bank-conflict bottleneck.

- **Matmul 1 (CB)**: Tiny output (16×16 = 256 elements). mma overhead (atom setup,
  ldmatrix) not worth it for 16 mma instructions. Keep warp 0 SIMT computation.
  Write result as bf16 to a small swizzled smem buffer for matmul 4's A operand.

- **CB_scaled elementwise scaling**: apply `exp(cumAdt[t]-cumAdt[j]) * dt_proc[j] * causal_mask`
  after SIMT matmul 1, then convert f32 → bf16 and write to smem.

### Swizzle atom

Following the existing MLA kernel pattern (`decode_mla_cute_sm80.cuh:150-151`):
```cpp
using SwizzleAtom = decltype(composition(
    Swizzle<3,3,3>{},
    make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{}))));
```

This is an 8-row × 64-column bf16 atom (128 bytes per row = one full bank cycle).
Tiles to full matrix size via `tile_to_shape(SwizzleAtom{}, target_shape)`.

### Swizzled smem layouts

| Array | Logical shape | Swizzled? | Atom tiling | Role |
|-------|---------------|-----------|-------------|------|
| C[16, 128] | (T, dstate) | **yes** | 2×2 atoms | A operand matmul 1 & 3 |
| B[16, 128] | (T, dstate) | **yes** | 2×2 atoms | B operand matmul 1 (TN) |
| state[64, 128] | (dim, dstate) | **yes** | 8×2 atoms | B operand matmul 3 (TN), replay read/write |
| x[16, 64] | (T, dim) | **yes** | 2×1 atoms | B operand matmul 4 (via transposed view) |
| z[16, 64] | (T, dim) | **yes** | 2×1 atoms | epilogue only, matches x layout |
| CB_scaled_bf16[16, 16] | (T, T) | small swizzle or none | — | A operand matmul 4 (converted from f32) |
| CB_scaled_f32[16, 16] | (T, T) | no | flat row-major | SIMT computation buffer |
| old_x[16, 64] | (T, dim) | no | flat row-major | SIMT replay |
| old_B[16, 128] | (T, dstate) | no | flat row-major | SIMT replay |
| out[16, 64] (f32) | (T, dim) | no | flat row-major | output staging (may be replaced by direct reg→gmem) |
| scalars | various | no | flat | cumAdt, dt_proc, decay_vec, etc. |

Total smem ≈ same as before (~39 KB). Swizzle only changes address mapping, not footprint.

### TiledMMA configuration

For matmuls 3+4 (output phase), all 4 warps (128 threads) cooperate:

```cpp
using TiledMmaOutput = decltype(make_tiled_mma(
    MMA_Atom<MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>>{},
    Layout<Shape<_1, _4, _1>>{} ));  // 1 in M, 4 in N → 128 threads
```

Covers 16 × 32 output per tile step (4 atoms × 8 columns each).
For output [16, 64]: 2 N-tile iterations.

For matmul 1 (CB, warp 0 only):

```cpp
using TiledMmaCB = decltype(make_tiled_mma(
    MMA_Atom<MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>>{},
    Layout<Shape<_1, _1, _1>>{} ));  // 1 atom → 32 threads (warp 0)
```

Covers 16 × 8 per step. For output [16, 16]: 2 N-tile iterations, 8 K-steps each.

### S2R copy atoms (smem → register via ldmatrix)

Following MLA kernel patterns:

```cpp
// A operand (C, CB_scaled): row-major access
auto s2r_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bf16>{}, tiled_mma);

// B operand (B, state): stored row-major (N,K), accessed via ldmatrix
auto s2r_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bf16>{}, tiled_mma);
```

For matmul 4, x is stored as [T, D] row-major but used as B operand with shape (D, T).
Use transposed ldmatrix + layout composition (same trick as MLA kernel lines 296-303):

```cpp
// Transposed logical view of x: (D=64, T=16) over x's (T=16, D=64) physical layout
auto layout_x_trans = make_layout(make_shape(_64, _16), make_stride(_1, _64));
auto layout_x_trans_cps = composition(smem_x.layout(), layout_x_trans);
Tensor smem_x_trans = make_tensor(smem_x.data(), layout_x_trans_cps);

auto s2r_copy_B_x = make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, bf16>{}, tiled_mma);
```

### Elementwise ops on accumulator fragments

Following MLA pattern (lines 400-401, 437-438):

**Decay scaling** after matmul 3 — use broadcast matrix in smem:
```cpp
// decay_vec[T] broadcast to [T, D] via stride-0 layout
auto decay_bcast = make_tensor(make_smem_ptr(decay_buf),
    make_layout(make_shape(_16, _64), make_stride(_1, _0)));
Tensor decay_part = thr_mma.partition_C(decay_bcast);

for (int i = 0; i < cute::size(accum); ++i)
    accum(i) *= decay_part(i);
```

**D*x skip connection** — partition x (row-major [T, D]) with partition_C:
```cpp
auto x_row_major = make_tensor(make_smem_ptr<bf16>(x_buf),
    make_layout(make_shape(_16, _64), make_stride(_64, _1)));
Tensor x_part = thr_mma.partition_C(x_row_major);

for (int i = 0; i < cute::size(accum); ++i)
    accum(i) += D_val * toFloat(x_part(i));
```

**z-gating** — same partition_C on z:
```cpp
Tensor z_part = thr_mma.partition_C(z_row_major);
for (int i = 0; i < cute::size(accum); ++i) {
    float zv = toFloat(z_part(i));
    accum(i) *= zv / (1.f + __expf(-zv));
}
```

### Kernel execution phases (v3)

```
Phase 0 — Load smem (all warps cooperate)
  cp.async / LDG → swizzled smem for C, B, state, x, z
  Plain LDG → flat smem for old_x, old_B, scalars, dt→dt_proc
  cp.async.commit + wait + __syncthreads()

Phase 1 — CB + Replay (warp-parallel, overlapped)
  Warp 0:   cumAdt (prefix sum), CB[T,T] via mma or SIMT, elementwise → CB_scaled (f32)
            Convert CB_scaled f32→bf16, write to smem
            Cache writes (old_B, old_dt_proc, old_cumAdt)
  Warps 1-3: Replay state via SIMT (reads/writes swizzled state smem through CuTe tensor view)
  __syncthreads()

Phase 2 — Output (all 4 warps via CuTe mma.sync)
  Matmul 3: gemm(C, state^T) → accum [16, 64] f32 registers
  Scale: accum *= decay_vec (broadcast partition_C)
  Matmul 4: gemm(CB_scaled_bf16, x^T) → same accum (accumulate)
  D*x: accum += D * x (partition_C on x)
  z-gate: accum *= z * sigmoid(z) (partition_C on z)
  Store: accum → global output (convert f32 → bf16)

Phase 3 — State + cache writeback (all warps)
  State: swizzled smem → global (all warps cooperate)
  old_x cache: from x in smem → global (warp 0)
```

### Changes vs v2

| Component | v2 (current) | v3 (planned) |
|-----------|-------------|-------------|
| Smem layout for C, B, state, x, z | flat 2D arrays | CuTe swizzled layouts |
| Matmul 3 (C @ state^T) | SIMT, 32-way bank conflicts | `cute::gemm` + ldmatrix, bank-conflict-free |
| Matmul 4 (CB_scaled @ x) | SIMT | `cute::gemm`, transposed B via LDSM_T |
| Matmul 1 (CB) | SIMT (warp 0) | keep SIMT (warp 0), convert result to bf16 |
| Matmul 2 (replay) | SIMT (warps 1-2) | keep SIMT, access swizzled state via CuTe view |
| Elementwise on output | SIMT loops over smem.out[] | elementwise on mma register fragment |
| Output store | smem.out → global (SIMT) | register fragment → global (via partition_C) |
| Replay warp assignment | warps 1-2 (warp 3 idle) | warps 1-3 (3 replay warps, better balance) |

### Expected improvements

| v2 Issue | How v3 addresses it |
|----------|-------------------|
| 58.7% excessive smem wavefronts (32-way bank conflict) | Swizzled layouts + ldmatrix eliminate bank conflicts entirely |
| 39.1% INT compute (bf16→f32 conversions) | mma.sync natively consumes bf16 operands — no explicit toFloat() calls |
| 28.5% barrier stall (warp imbalance) | Move replay to warps 1-3 (3 warps instead of 2). Output uses all 4 warps via tiled_mma |
| 19.1% FMA pipe utilization | mma.sync issues 16×8×16 FMAs per instruction — massively higher throughput |

### Reference code

- **Existing CuTe kernel**: `include/flashinfer/attention/decode_mla_cute_sm80.cuh`
  - Swizzle atom: lines 66-67, 150-151
  - TiledMMA: lines 237-239, 286-288
  - S2R copy: lines 252-261, 302-304
  - Fragment creation: lines 245, 250, 278-279, 293-294, 316-317
  - Gemm: lines 387-388, 396-397, 442
  - Elementwise on accum: lines 400-401, 437-438
  - Transposed B via composition: lines 296-300

- **CuTe test examples**: `3rdparty/cutlass/test/unit/cute/ampere/cooperative_gemm.cu`
- **CuTe MMA atoms**: `3rdparty/cutlass/include/cute/atom/mma_traits_sm80.hpp`

### Implementation steps

- [ ] 9.1 **EASY** — Add CuTe include, define swizzle layout types
  - `#include "cute/tensor.hpp"`, `using namespace cute;`
  - Define `SwizzleAtom`, `LayoutSwizzleC`, `LayoutSwizzleB`, `LayoutSwizzleState`,
    `LayoutSwizzleX`, `LayoutSwizzleZ` as `using` type aliases
  - Ref: `decode_mla_cute_sm80.cuh:66-73`

- [ ] 9.2 **MEDIUM** — Refactor smem struct to use raw byte buffers + CuTe tensor views
  - Replace `input_t C[T][DSTATE]` etc. with aligned byte arrays
  - Create CuTe tensor views in kernel body: `make_tensor(make_smem_ptr<T>(buf), Layout{})`
  - Keep flat arrays for non-mma data (old_x, old_B, CB_scaled_f32, scalars)
  - Ref: `decode_mla_cute_sm80.cuh:170-190`

- [ ] 9.3 **MEDIUM** — Update `load_data` to write to swizzled smem
  - cp.async destinations computed via CuTe tensor views
  - Verify: `&smem_tensor(row, col)` gives correct swizzled address for cp_async_16B
  - Plain LDG for flat arrays unchanged

- [ ] 9.4 **MEDIUM** — Update `replay_state` to read/write swizzled state
  - Replace `smem.state[dd][n]` with `smem_state(dd, n)` (CuTe tensor view)
  - Read: `toFloat(smem_state(dd, n))`
  - Write: `convertAndStore(&smem_state(dd, n), val)`
  - Access pattern is per-thread (dd fixed per warp, n strided across lanes) — swizzle
    may slightly change bank conflict pattern but shouldn't be worse

- [ ] 9.5 **MEDIUM** — Update `compute_CB_scaled` to read swizzled C, B
  - SIMT matmul reads from `smem_C(t, n)` and `smem_B(j, n)` via CuTe views
  - Write result as f32 to flat `smem.CB_scaled_f32[t][j]`
  - After scaling: convert to bf16 and write to small `CB_scaled_bf16` buffer
    (512 bytes, with appropriate swizzle or flat — TBD at implementation time)

- [ ] 9.6 **HARD** — Implement CuTe mma output phase
  - Define `TiledMmaOutput` with `SM80_16x8x16_F32BF16BF16F32_TN` + `Layout<Shape<_1,_4>>{}`
  - Define s2r copy atoms for A (C, CB_scaled) and B (state, x)
  - Partition smem tensors: `local_tile`, `partition_S`, `partition_fragment_A/B/C`
  - Create accumulator fragment: `make_fragment_C`
  - K-loop for matmul 3: `copy(s2r, smem_part, reg_view)` + `gemm(mma, accum, A, B, accum)`
  - Decay_vec scaling via broadcast partition_C
  - Matmul 4: same pattern with transposed x (LDSM_T) and CB_scaled_bf16
  - Elementwise: D*x, z-gate on register fragment
  - Store: register fragment → global output
  - Ref: `decode_mla_cute_sm80.cuh:236-442`

- [ ] 9.7 **EASY** — Update replay warp assignment
  - Change from warps 1-2 to warps 1-3 (3 replay warps)
  - `ROWS_PER_WARP = DIM / 3` (requires DIM divisible by 3... if DIM=64 that's not clean)
  - Alternative: warps 1-2-3 with 22-21-21 rows, or keep 2 warps and optimize elsewhere
  - Evaluate after profiling v3

- [ ] 9.8 **EASY** — Test and validate
  - Run existing test: `pytest -k test_ssu_incremental -v -s -x`
  - Compare output/state against Triton reference at same tolerances

- [ ] 9.9 **EASY** — Benchmark and profile
  - Timing sweep: `collect_incremental_ssu_runs.py --output-prefix incremental_ssu_b64_f16_v3`
  - ncu profile: check bank conflicts eliminated, FMA pipe utilization, INT reduction

---

## 10. v3 results: CuTe mma.sync for CB matmul (matmul 1 only)

### Change

Replaced the SIMT `simt_mma_nt_set` in `compute_CB_scaled` with CuTe `mma.sync.m16n8k16`
via `SM80_16x8x16_F32BF16BF16F32_TN`. The elementwise scaling (causal mask, decay, dt_proc)
stays SIMT — applied after the mma writes CB to smem.

Key implementation details:
- **Padding**: B and C arrays padded from `[NTOKENS][DSTATE]` to `[NTOKENS_PAD][DSTATE]`
  where `NTOKENS_PAD = next_multiple_of_16(NTOKENS)`. Required because `mma.sync.m16n8k16`
  always reads 16 rows via ldmatrix — no predicated variant.
- **CB_scaled also padded** to `[NTOKENS_PAD][NTOKENS_PAD]` to avoid OOB writes from
  `partition_C`. Padding rows/columns contain garbage but are never read.
- No zeroing of padding needed — valid CB[t][j] for t,j < NTOKENS uses only valid
  C[t,:] and B[j,:] rows in the K-reduction.
- CuTe setup: `TiledMMA` with 1 atom (32 threads = warp 0), `SM75_U32x4_LDSM_N` for
  A (C) s2r copy, `SM75_U32x2_LDSM_N` for B s2r copy. No swizzle yet (flat row-major smem).

### Timing comparison (batch=64, fp16 state, bf16 activations)

mtp=4 (NTOKENS_PAD=16, 4× padding waste):

| prev_k | v2 (µs) | v3 (µs) | delta |
|--------|---------|---------|-------|
| 0 | 40.9 | 43.0 | -5% |
| 2 | 55.3 | 57.3 | -4% |
| 4 | 59.4 | 61.4 | -3% |

mtp=8 (NTOKENS_PAD=16, 2× padding waste):

| prev_k | v2 (µs) | v3 (µs) | delta |
|--------|---------|---------|-------|
| 0 | 63.5 | 82.0 | -29% |
| 4 | 81.9 | 114.8 | -40% |
| 8 | 92.2 | 133.1 | -44% |

mtp=16 (NTOKENS_PAD=16, no padding waste):

| prev_k | v2 (µs) | v3 (µs) | delta |
|--------|---------|---------|-------|
| 0 | 143.3 | 133.1 | +8% |
| 8 | 196.6 | 182.3 | +7% |
| 16 | 235.5 | 215.0 | +9% |

### Analysis

- **mtp=16 improved** ~8% — no padding waste, mma.sync is faster than SIMT for the CB matmul.
- **mtp<16 regressed** significantly (up to -44% at mtp=8). Root cause: padding B, C, CB_scaled
  to 16 rows regardless of NTOKENS wastes smem and computation:
  - B and C doubled in size for mtp=8 (2KB→4KB each)
  - CB_scaled quadrupled for mtp=8 (256B→1KB)
  - Total ~5KB extra smem may reduce occupancy
  - CB_scaled stride changed from NTOKENS to NTOKENS_PAD, hurting `add_cb_out` cache behavior
  - mma computes 16×16 output but only 8×8 is useful — 4× wasted work

### What v3 did NOT fix

The padding overhead dominates at small mtp. Need to separate the padded mma workspace
from the main CB_scaled array so that the rest of the kernel (add_cb_out, etc.) still uses
the original NTOKENS-sized arrays.

---

## 11. v3.1: Swizzled smem for B and C arrays

### Change

Added `Swizzle<3,3,3>` on 8×64 row-major atom to B and C arrays in shared memory.
Both B and C are operands of the CB matmul (`C @ B^T` via `SM80_16x8x16_F32BF16BF16F32_TN`);
in TN mode both are row-major `[NTOKENS_PAD, DSTATE]`, so they share the same swizzle.

Implementation:
- **`make_swizzled_layout_rc<ROWS, COLS>()`** — helper returning swizzled CuTe layout via
  `composition(Swizzle<3,3,3>{}, Layout<Shape<_8,_64>, Stride<_64,_1>>{})`
  tiled to `(ROWS, COLS)`.
- **`load_B_cute` / `load_C_cute`** — cp.async 16B into swizzled smem. Destination address
  computed as `base + layout(t, col)`. 8 consecutive bf16 elements (128-bit chunk) remain
  physically contiguous under `Swizzle<3,3,3>` (only bits ≥ 3 are XOR'd), so cp.async works.
- **`load_B` / `load_C`** — original flat versions kept for fallback.
- **`compute_CB_scaled`** — `layout_bc` now uses the swizzled layout; ldmatrix reads
  bank-conflict-free.
- **`store_old_B`** — reads B through `B_base[layout_B(t, n)]`.
- **`add_init_out`** — reads C through `C_base[layout_C(t, n)]`.

### Timing comparison (batch=64, fp16 state, bf16 activations)

Ratios = `cuda-incr / flashinfer` (different server from v3, only ratios comparable):

| mtp | prev_k | v3 ratio | v3.1 ratio | delta |
|-----|--------|----------|------------|-------|
| 4   | 0      | 1.91x    | 2.58x     | **-35%** |
| 4   | 2      | 2.55x    | 4.07x     | **-60%** |
| 4   | 4      | 2.73x    | 4.17x     | **-53%** |
| 8   | 0      | 2.35x    | 2.30x     | +2% |
| 8   | 4      | 3.30x    | 3.30x     | 0% |
| 8   | 8      | 3.82x    | 3.47x     | **+9%** |
| 10  | 10     | 4.35x    | 3.75x     | **+14%** |
| 12  | 12     | 3.91x    | 3.43x     | **+12%** |
| 16  | 0      | 2.24x    | 2.25x     | 0% |
| 16  | 8      | 3.07x    | 2.87x     | **+7%** |
| 16  | 16     | 3.62x    | 3.62x     | 0% |

### ncu profile: v3.1 kernel (batch=128, mtp=8, prev_k=4)

```
Duration:          296.51 µs
Registers/thread:  72
Shared memory:     33.16 KB
Occupancy:         31.2% (theoretical 37.5%)

Compute (SM):      47.2%
Memory:            62.3% (L1/TEX)
DRAM:              2.0%

FMA pipe:          20.7%
ALU pipe:          30.6%
Tensor pipe:       0.1%
LSU pipe:          28.7%

Excessive smem wavefronts: 51.0%
  All from line 615 (add_init_out inner loop): 32-way bank conflicts on LDS.128
  → smem.state[dd][n] — all threads read same n, different dd → same bank

Warp stalls:
  barrier:          31.1%  (line 828: __syncthreads between phases)
  short_scoreboard: 20.4%  (lines 574-577: replay_state swizzled state + old_B access)
  wait:             11.8%

Instruction mix:
  INT:    55.8%  (IMAD: 19.2M, LOP3/XOR: 9.0M, SHF: 6.6M)
  FP:     18.9%  (FMUL: 6.9M, FFMA: 6.6M)
  LD/ST:  15.3%  (LDS: 12.7M)
```

### Analysis

- **mtp≥8 with prev_k>0 improved** (+7–14%): ldmatrix in `compute_CB_scaled` is now
  bank-conflict-free.
- **mtp=4 regressed badly** (-35–60%): CB matmul is tiny (4×4 useful output), so mma benefit
  is negligible. Meanwhile `add_init_out` and `store_old_B` now pay the swizzle overhead
  (9M LOP3 instructions for XOR index computation, plus scattered smem reads that break
  vectorization).
- **The dominant bottleneck remains `add_init_out`** (matmul 3: `C @ state^T`): 51% excessive
  smem wavefronts, all 32-way bank conflicts on `smem.state[dd][n]`. The swizzle on B/C
  did not help here — **state** needs swizzling, and the matmul needs mma.sync.

### What v3.1 did NOT fix

1. **32-way bank conflicts on state** in `add_init_out` — the #1 bottleneck (51% excess wavefronts).
   Requires swizzling the state array and converting matmul 3 to mma.sync (plan step 9.6).
2. **9M LOP3 instructions** from SIMT code accessing swizzled B/C. Will disappear once
   `add_init_out` uses mma.sync (ldmatrix handles swizzle natively) and `store_old_B` is
   refactored.
3. **31% barrier stall** from warp imbalance between phases.
4. **55.8% INT instruction mix** — still dominated by address computation, not useful FP work.

---

## 12. v3.2: mma.sync for matmul 3 (C @ state^T) + swizzled state

### Change

Replaced SIMT `add_init_out` with CuTe `add_init_out_cute` using mma.sync for the largest
matmul: `C[T, DSTATE] @ state[DIM, DSTATE]^T → out[T, DIM]`. Also swizzled the state array
in shared memory (`Swizzle<3,3,3>` same as B/C).

Implementation:
- **`add_init_out_cute`**: all 4 warps via `TiledMMA Layout<_1, _4>` (128 threads, 16×32 output
  per step). 2 N-tile iterations for DIM=64. K-loop of 8 tiles (DSTATE=128 / K_TILE=16).
  Decay applied as `accum[i] *= __expf(cumAdt[t_i])` via broadcast `partition_C`.
- **MMA type dispatch**: `mma_type = __half` if either operand is f16, else `__nv_bfloat16`.
  After s2r copy, fragment conversion via `memcpy` + `toFloat` + `mma_type()` if source differs.
- **`load_state_cute`**: cp.async 16B into swizzled smem. All warps cooperate.
- **`replay_state`**: reads/writes state through swizzled layout via `if constexpr (sizeof(state_t)==2)`.
- **`store_state`**: reads from swizzled state via same dispatch.
- **`smem.out`**: padded from `[NTOKENS][DIM]` to `[NTOKENS_PAD][DIM]` so `partition_C` writes
  don't go OOB.
- **Dispatch**: `sizeof(state_t) == 2` → cute path, else SIMT fallback (for f32 state).

### Timing comparison (batch=64, fp16 state, bf16 activations)

Ratios = `cuda-incr / flashinfer` (v3.2 on same server as v3):

| mtp | prev_k | v3 ratio | v3.1 ratio | **v3.2 ratio** | v3→v3.2 |
|-----|--------|----------|------------|----------------|---------|
| 4   | 0      | 1.91x    | 2.58x     | **1.92x**      | ~same |
| 4   | 2      | 2.55x    | 4.07x     | **3.01x**      | -18% |
| 4   | 4      | 2.73x    | 4.17x     | **3.29x**      | -21% |
| 8   | 0      | 2.35x    | 2.30x     | **1.41x**      | **+40%** |
| 8   | 4      | 3.30x    | 3.30x     | **2.35x**      | **+29%** |
| 8   | 8      | 3.82x    | 3.47x     | **2.94x**      | **+23%** |
| 10  | 0      | 2.45x    | —         | **1.25x**      | **+49%** |
| 10  | 10     | 4.35x    | —         | **3.00x**      | **+31%** |
| 12  | 0      | 2.43x    | —         | **1.13x**      | **+53%** |
| 12  | 12     | 3.91x    | —         | **2.69x**      | **+31%** |
| 16  | 0      | 2.24x    | 2.25x     | **1.00x**      | **+55%** |
| 16  | 8      | 3.07x    | 2.87x     | **1.93x**      | **+37%** |
| 16  | 16     | 3.62x    | 3.62x     | **2.76x**      | **+24%** |

### ncu profile: v3.2 kernel (batch=128, mtp=8, prev_k=4)

```
Duration:          190.56 µs  (v3.1: 296.51 µs, -36%)
Registers/thread:  43         (v3.1: 72, -40%)
Shared memory:     35.16 KB   (v3.1: 33.16 KB)
Occupancy:         30.4%      (v3.1: 31.2%)

Compute (SM):      48.6%
Memory (L1/TEX):   34.6%   (v3.1: 62.3% — no longer memory-bound)
DRAM:              3.1%

FMA pipe:          20.8%
ALU pipe:          28.9%
Tensor pipe:       1.0%    (v3.1: 0.1% — now using tensor cores)
LSU pipe:          31.0%

Excessive smem wavefronts: 7.7%  (v3.1: 51.0% — 32-way conflicts eliminated!)
  Remaining sources:
    34.2%: LDGSTS (cp.async to swizzled smem) — 8-way conflicts in state load
     6.4%: STS.64 (smem.out store from partition_C) — 8-way
     4.3%: LDS/STS in replay_state and add_cb_out — 2-way

Warp stalls:
  barrier:          40.7%  (line 1034: __syncthreads between Phase 1 and Phase 2)
  wait:             20.2%  (lines 627-639: replay_state)
  short_scoreboard:  6.6%  (v3.1: 20.4%)

Instruction mix:
  Total:  62.1M   (v3.1: 94.2M, -34%)
  INT:    52.0%   (v3.1: 55.8%)
  FP:     22.2%   (v3.1: 18.9%)
  LD/ST:  16.0%   (v3.1: 15.3%)

Top opcodes:
  IMAD: 10.1M   (v3.1: 19.2M, -47%)
  LDS:   8.0M   (v3.1: 12.7M, -37%)
  LOP3:  5.8M   (v3.1: 9.0M, -35%)
  FMUL:  6.9M   (same)
  FFMA:  2.4M   (v3.1: 6.6M, -64%)
```

### Analysis

- **Duration -36%**: replacing SIMT dot products with mma.sync eliminates redundant
  address computation and bank-conflicted loads.
- **Registers -40%** (72 → 43): SIMT `add_init_out` accumulated full [T, DSTATE] dot products
  per thread, requiring many live registers. mma.sync uses small 16×8 fragment registers.
  Max blocks by registers increased from 7 to 10 (though smem still limits to 6).
- **32-way bank conflicts eliminated**: 51.0% → 7.7% excessive wavefronts. ldmatrix reads from
  swizzled smem are bank-conflict-free by design.
- **Tensor pipe 0.1% → 1.0%**: tensor cores now contribute, though still a small fraction since
  mma only runs in Phase 2 (the output epilogue is still SIMT).
- **Barrier stall grew to 40.7%**: now the dominant bottleneck. Warp 0 (CB_scaled: mma K-loop
  over DSTATE=128) is much slower than warps 1-2 (SIMT replay with dynamic prev_k). Warp 3 is
  completely idle during Phase 1. All warps wait at __syncthreads before Phase 2.
- **Wait stall 20.2%**: replay_state (lines 627-639) is the second bottleneck. Swizzled state
  access adds XOR overhead, and the inner loop over prev_k has data dependencies.
- **mtp=4 still degraded** vs v3: NTOKENS_PAD=16 forces 16-row mma tiles for only 4 useful
  rows — 75% wasted work in the CB and output matmuls.

### What v3.2 did NOT fix

1. **40.7% barrier stall** — Phase 1 warp imbalance. Warp 0 (CB_scaled) and warps 1-2 (replay)
   have very different runtimes. Warp 3 is idle. Options:
   - Move warp 3 to help with replay (3 replay warps instead of 2)
   - Overlap CB_scaled computation with replay more effectively
2. **20.2% wait stall in replay** — the SIMT replay loop is latency-bound on smem accesses
   with data dependencies (state read → compute → state write per element).
3. **mtp=4 regression** — padding to 16 rows wastes 75% of mma compute.
4. **SIMT add_cb_out, add_D_skip, z_gating, store_output** — these Phase 2 epilogue functions
   still use per-warp SIMT loops through `smem.out[]`. Converting matmul 4 (`CB_scaled @ x`)
   to mma.sync and fusing all epilogue ops on the register accumulator (as planned in section 9)
   would eliminate the smem.out round-trip entirely.

---

## 13. v3.3: Fused mma output phase (matmul 4 + epilogue)

### Change

Fused matmul 4 (`CB_scaled @ x`), D*x skip connection, z-gating, and smem.out store into a
single function `compute_output_cute`, sharing the register accumulator with matmul 3.
Also introduced `USE_TENSOR_MMA` pattern (inferred from `sizeof(state_t) == 2`) to replace
scattered `sizeof(state_t) == 2` checks.

Implementation:
- **`compute_output_cute`** — replaces `add_init_out_cute` + `add_cb_out` + `add_D_skip` +
  `compute_z_gating`. Per N-tile: matmul 3 (C @ state^T, K-loop of 8) → decay → matmul 4
  (CB_scaled @ x, 1 K-tile) → D*x → z-gate → store to smem.out. All on register accumulator.
- **CB_scaled_mma buffer** — `input_t CB_scaled_mma[NTOKENS_PAD][NTOKENS_PAD]` in smem struct
  (512 bytes). Warp 0 converts f32 CB_scaled → input_t after elementwise scaling. Padding
  elements zeroed to prevent 0 × NaN = NaN in mma.
- **x as B operand** — flat row-major [NTOKENS_PAD, DIM], transposed view [DIM, NTOKENS_PAD]
  for TN mma B operand. Uses `SM75_U16x2_LDSM_T` (transposed ldmatrix). Not swizzled — K=16
  is only 1 K-tile, so bank conflict cost is negligible.
- **x padding zeroed** — rows NTOKENS..NTOKENS_PAD-1 zeroed after cp.async wait to prevent
  NaN propagation (IEEE 754: 0 × NaN = NaN).
- **z via partition_C** — flat row-major, elementwise on accumulator.
- **`USE_TENSOR_MMA`** — `constexpr bool` computed locally as `sizeof(state_t) == 2` in each
  function that needs it. Replaces all prior `sizeof(state_t) == 2` checks.
- **Smem changes** — x and z padded to `[NTOKENS_PAD][DIM]`, added `CB_scaled_mma` buffer.

### Timing comparison (batch=64, fp16 state, bf16 activations)

| mtp | prev_k | v3.2 ratio | v3.3 ratio | delta |
|-----|--------|-----------|-----------|-------|
| 4   | 0      | 1.92x     | 2.09x     | ~noise |
| 4   | 4      | 3.29x     | 3.58x     | ~noise |
| 8   | 0      | 1.41x     | 1.36x     | +4% |
| 8   | 4      | 2.35x     | 2.24x     | +5% |
| 8   | 8      | 2.94x     | 2.77x     | +6% |
| 12  | 0      | 1.13x     | 1.13x     | same |
| 12  | 12     | 2.69x     | 2.65x     | ~same |
| 16  | 0      | 1.00x     | 1.00x     | same |
| 16  | 8      | 1.93x     | 1.97x     | ~same |
| 16  | 16     | 2.76x     | 2.86x     | ~same |

### Analysis

Perf is essentially flat vs v3.2. Modest 5-6% improvement at mtp=8, within noise elsewhere.
Not surprising:

- **Matmul 4 is tiny** (K=16, 1 K-tile). The SIMT `add_cb_out` was already fast.
- **D*x and z-gating** were trivial SIMT operations, cost barely visible in the profile.
- **smem.out round-trip not fully eliminated** — `compute_output_cute` still writes to smem.out,
  and `store_output` still reads from it. Direct register → global store would skip this, but
  requires extracting (t, d) coordinates from partition_C (deferred).
- **Dominant bottleneck unchanged**: 40.7% barrier stall from Phase 1 warp imbalance.

The value of v3.3 is **structural**: the fused `compute_output_cute` keeps the accumulator in
registers across 4 operations (matmul 3 → decay → matmul 4 → epilogue), eliminating 3 separate
smem.out passes. This is the right architecture for future optimizations (direct register → gmem
store, overlapping Phase 1 with Phase 2, etc.).

### What v3.3 did NOT fix

1. **40.7% barrier stall** — still the #1 bottleneck. Phase 1 warp imbalance unchanged.
2. **smem.out round-trip** — still write to smem.out then read in store_output.
   Needs direct register → global store with (t, d) coordinate extraction.
3. **mtp=4 regression** — 75% wasted mma compute from padding to 16 rows.

---

## Matmul status summary (as of v3.3)

| # | Operation | Shape (M,N,K) | Implementation | Since |
|---|-----------|---------------|----------------|-------|
| 1 | `C @ B^T` → CB | (16,16,128) | mma.sync (warp 0, swizzled B/C) | v3 |
| 2 | `old_x^T @ dB_scaled` → state replay | (64,128,prev_k) | **SIMT** (dynamic K, stays SIMT) | v1 |
| 3 | `C @ state^T` → init_out | (16,64,128) | mma.sync (all warps, swizzled C/state) | v3.2 |
| 4 | `CB_scaled @ x` → cb_out | (16,64,16) | mma.sync (all warps, CB_scaled_mma + LDSM_T on flat x) | v3.3 |

All matmuls that can benefit from tensor cores are now tensorized. Matmul 2 (replay) stays
SIMT by design — `prev_k` is dynamic (0..16) and not always a multiple of 16.

---

## 14. v4 (WIP): Swizzle x buffer — bank conflict removal

### ncu report analysis (ssu_replay_09.ncu-rep, batch=128, 16 heads)

| Metric | Value |
|--------|-------|
| Duration | 195.58 µs |
| Registers/thread | 44 |
| Achieved occupancy | 30.4% (theoretical 37.5%, smem-limited) |
| Compute (SM) | 48.9% |
| Tensor pipe | 1.1% (barely used despite 3 mma.sync matmuls) |
| #1 stall: barrier | 40.7% — Phase 1 warp imbalance (warp 3 idle) |
| #2 stall: wait | 19.6% — smem latency in replay inner loop |
| Instruction mix | Integer 51.8%, FP 21.3%, Load/Store 15.4% |

Shared memory excessive wavefronts (11.1% total):

| Source | Excess % | Conflict |
|--------|----------|----------|
| `cp_async_16B` (line 164, LDGSTS.128) — aggregated across all cp.async calls | 23.3% | 8-way |
| `LDSM.16.MT88` (ldmatrix in S2R copies) — 4 instances | 5.1% each | 8-way |
| `LDS` (line 916, D*x skip via partition_C on flat x) — 4 instances | 5.1% each | 8-way |
| `STS.64` (line 940, smem.out store via partition_C) | 4.4% | 8-way |

### Root cause of smem conflicts

Buffers B, C, and state are swizzled (Swizzle<3,3,3>) → conflict-free ldmatrix. But:

1. **x** is loaded flat (`load_x`, not `load_x_cute`) → LDS conflicts on D*x skip reads,
   LDSM_T conflicts on matmul 4 B operand
2. **CB_scaled_mma** is flat [16,16] → LDSM_N conflicts on matmul 4 A operand
3. **smem.out** is flat float[16][64] → STS conflicts on partition_C writes

### What we implemented

Switched x to swizzled storage when `USE_TENSOR_MMA` (bf16 state):

1. **`load_x_cute`** (swizzled cp.async) replaces `load_x` (flat). Conditional on
   `sizeof(state_t) == 2` — the SIMT path (`!USE_TENSOR_MMA`) still reads `smem.x[t][d]`
   with flat array indexing, so it must keep flat storage.

2. **`make_swizzled_layout_rc_transpose<ROWS, COLS>()`** — new helper that constructs a
   transposed swizzled layout: maps `(col, row)` → same physical offset as
   `make_swizzled_layout_rc` maps `(row, col)`. Uses column-major inner layout
   `(COLS, 8):(1, COLS)` composed with Swizzle<3,3,3>, tiled to `(COLS, ROWS)`.
   Bank-conflict-free for ldmatrix.trans (verified analytically: 8 threads in each
   group hit banks 0,4,8,12,16,20,24,28 — all distinct).

3. **`compute_output_cute`** updated:
   - `smem_x` uses `make_swizzled_layout_rc<NTOKENS_PAD, DIM>()` (was flat row-major)
   - `smem_x_trans` uses `make_swizzled_layout_rc_transpose<NTOKENS_PAD, DIM>()` (was flat col-major)

4. **`store_old_x`** updated to read x via swizzled layout (USE_TENSOR_MMA branch).

5. **x padding zeroing** updated to write via swizzled layout.

### Status: tests pass, needs re-profiling

All 8 `test_ssu_incremental` tests pass. The changes are in the kernel code, not yet profiled.

### Bug found and fixed during implementation

The initial attempt used `load_x_cute` unconditionally, which broke the
`state_dtype=float32` path. When `USE_TENSOR_MMA=false`, the SIMT functions
(`add_cb_out`, `add_D_skip`) read `smem.x[t][d]` with flat array indexing — storing
swizzled data there produces garbage. Fixed by making the load conditional on
`sizeof(state_t) == 2`.

### Remaining bank conflict sources (not yet addressed)

1. **CB_scaled_mma** [16,16] flat — LDSM_N conflicts for matmul 4 A operand.
   Small matrix (512 bytes), could swizzle but atom size (8×64) doesn't tile nicely
   into 16×16. Might need a different swizzle atom or just accept the cost (1 K-tile).

2. **smem.out** float[16][64] flat — STS conflicts from partition_C writes.
   Row = 64×4 = 256 bytes = 2 bank cycles. Could swizzle with a float-width atom
   (Swizzle for 4-byte elements), but needs a different swizzle configuration than
   the bf16 one.

3. **cp_async conflicts** (23.3% aggregated) — unclear which specific buffer(s) contribute
   most since ncu aggregates all calls through `cp_async_16B` at the same PC.
