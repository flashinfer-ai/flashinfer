# SSU two-kernel split — selected by caller-provided scratch

> **Dispatch decision:** there is **no `algorithm` flag**. The two-kernel
> (precompute + main) path runs **iff** the caller provides the scratch
> tensors `cb_scaled`/`decay_vec`; otherwise the monolithic kernel runs.
> Presence of the (graph-safe, caller-allocated) workspace *is* the switch —
> so the fast path can't be requested without the buffer that makes it
> CUDA-graph-safe.

## Motivation

Profiling (CUPTI per-kernel timeline, bf16, conv1d + PDL) localized the
batch-16 latency cliff to the **monolithic** `checkpointing_ssu_kernel`'s
post-`griddepcontrol.wait` phase ballooning **3.9 → 10.3 µs** (b8 → b16),
while Triton's two-kernel replay path stays flat (**5.06 → 5.31 µs**).

Root cause: the cuda kernel is global-load-latency bound and runs at
**6–10% occupancy**; with one under-occupied grid there aren't enough warps
in flight to hide the loads, so doubling the batch just stalls longer.
Triton hides the same latency by running **two overlapping persistent
kernels** (`_dynamic_precompute` + `_persistent_main`) that keep the GPU
busy. This plan ports that structure, selected by the presence of the
caller-provided scratch tensors (no `algorithm` flag).

## Why two kernels (and why scratch buffers are required)

Triton allocates two per-call scratch tensors
(`replay_selective_state_update.py:4197-4202`):

- `cb_scaled` — **bf16**, fragA-native `[batch, nheads, lane(32), reg(8)]` (= matmul-4 fragA)
- `decay_vec` — **fp32**, `(batch, nheads, NPREDICTED_PAD_MMA_M)`

(Triton used fp32 + a logical layout; we improve on it — see "Scratch layout" below.)

The precompute **writes** them; the main **reads** them (main kernel
"Phase 2: Output using precomputed CB_scaled and decay_vec", line 1577).
A two-kernel split has nowhere else to stash the precompute's results, so
the scratch hand-off is fundamental — we need the same. (The existing
`old_B/old_dt/old_dA` cache is also written by precompute and is *not* new.)

It is **not** a conv1d-independence split: both kernels consume conv1d
output (precompute needs `B/C`, main needs `x`). The win is **concurrency /
effective occupancy**, not avoiding the conv1d dependency.

## The op split (by equation — see `ssu_checkpoint_equations.md`)

Dividing principle: **precompute = the coefficient/matrix-build block**
(computable from the projections `dt, A, B, C` alone); **main = the recurrence
+ output** (touches `state` and `x`).

### Precompute owns (the "C" block)

| eq | what | monolithic helper |
|----|------|-------------------|
| C1 | `dt_proc = softplus(dt+bias)` | `load_*_data` |
| C2 | `cumAdt` scan + `decay = exp(cumAdt)` | `compute_cumAdt` |
| C5 | `CB[t,j] = (C·B)·exp(cumAdt[t]-cumAdt[j])·dt_proc[j]` | `compute_CB_scaled_2warp` (`:1057`) |
| C6 | `CB_old[t,i]` (no-write; folded into the rectangular CB over `[old ∪ new]`) | `compute_CB_old_2warp` (`:1060`) |
| C7 | cumsum continuity for the cache write | Phase-3 (`:1130`) |
| cache | write `old_B`, `old_dt`, `old_cumAdt` (from `B/dt/A`) | `store_old_B` (`:1047`), `store_old_dt`/`store_old_cumAdt` (`:1113`/`:1120`) |

Emits to scratch: `cb_scaled` = C5 (CB_scaled), `decay_vec` = `exp(cumAdt)` (C2).

### Scratch layout (fragA-native — the key store optimization)

`cb_scaled` is **bf16** stored in **matmul-4 fragA layout**: the two `m16n8`
N-tiles a thread accumulates for CB have the *same per-thread element order* as
`fragA` for `mma.m16n8k16` (`CB @ x`). So the precompute writes each lane's 8
scaled bf16 with **one `STG.128`** straight from the MMA accumulator registers
(no swizzled-smem round-trip, no de-swizzle), and the main reads them with **one
`LDG.128` straight into `fragA`** (no smem / LDSM / swizzle). Layout
`[batch, nheads, lane(0..31), reg(0..7)]`, 512 B/head; `(t,j)` per register is
computed on the fly from the register index. `decay_vec` (C2's `exp(cumAdt)`)
stays f32 — it's the per-head β for OUT.1, not an MMA operand.

### Main owns (STATE + OUT + writeback)

| eq | what | monolithic helper |
|----|------|-------------------|
| C3 | totals & β-factor (`total_old_cumAdt`, `β(t)`) | epilogue |
| STATE | replay `state <- total_decay·state + old_xᵀ@(coeff⊙old_B)` (checkpoint) | `replay_state_mma` (via `ssu_checkpoint` `:1089`) |
| C8 | state writeback (stochastic-round) | `store_state` |
| OUT.1 | `β·C@stateᵀ` | `add_init_out` |
| OUT.2 | `CB@x` (reads `cb_scaled`) | `add_cb_x` |
| OUT.3 | `CB_old@old_x` (no-write; reads `cb_scaled` rectangle) | `add_cb_old_x` |
| OUT.4/5 | `+D·x`, z-gate | `add_D_skip`, `compute_z_gating` |
| cache | write `old_x` (from `x`) | `store_old_x` (`:1110`) |

Reads from scratch: `cb_scaled`, `decay_vec`.

**Cut point = the monolithic kernel's single `__syncthreads()` (`:1087`)**:
precompute = Phase0 + the CB half of Phase1 (C5/C6) + the `old_B/dt/cumAdt`
writes; main = the STATE half of Phase1 + all of Phase2 (C3/OUT/C8) + the
`old_x` write.

**C4 (`coeff`/`dB_old`) is shared**: precompute uses it to build CB_old (C6);
main re-derives it (cheap scalar on `k ≤ 16` values, from `old_dt`/`old_cumAdt`
it already loads) for the checkpoint STATE replay — no third scratch tensor.

## Granularity: precompute is per-GROUP, main is per-head

`decay_vec`/`cb_scaled` are **per-head** outputs (decay = `exp(cumsum(A·dt))`,
and `A`/`dt` are per-head). But the expensive **C·B contraction (C5, over
DSTATE) is per-GROUP** — `B`/`C` are `(batch, T, ngroups, d_state)`, so that
`T×window` matrix is shared by every head in the group, then scaled per-head.

The monolithic kernel runs one CTA per `(batch, head)` (equations §0), so it
recomputes C·B once per head — at the default **ngroups=1, nheads=16** that's
the **same C·B 16× redundantly**. Triton's precompute instead processes a
**block of `HEADS_PER_BLOCK` heads** sharing one group's `B`/`C`
(`replay_selective_state_update.py:344` + `nheads_ngroups_ratio`): C·B once,
scaled per-head.

→ Our precompute grid is **`(batch, ngroups)`** with a head loop/tile over
`HEADS_PER_GROUP` (C·B once per group, scale into per-head
`cb_scaled`/`decay_vec`), **not** `(batch, nheads)`. This is both correct and a
real saving (16× fewer C·B contractions at ngroups=1) — the reason the
precompute is cheap enough to overlap conv1d. The **main** kernel stays
per-`(batch, head)` (it's the per-head state recurrence).

## PDL topology

Clean **chain** (single-predecessor PDL each — no fork-join):

```
conv1d  ──PDL──▶  precompute  ──PDL──▶  main
```

precompute overlaps conv1d's tail (loads `dt/A` pre-wait, `B/C` post-wait);
main overlaps precompute's tail (loads `state/x/old_x` pre-wait,
`cb_scaled/decay_vec` post-wait). Matches the Triton timeline
(precompute starts ~1.2 µs into conv1d; main overlaps precompute).

## Decisions

Made:
- [x] Approach: two overlapping kernels (precompute + main), Triton-style.
- [x] **Dispatch by scratch presence** (no `algorithm` flag): two-kernel path
      iff `cb_scaled`/`decay_vec` are passed; else monolithic. Runtime
      launcher choice, **not** a JIT constexpr.
- [x] Scratch **caller-provided** (like `out`) — graph-safe under any capture
      path.  `cb_scaled` = **bf16 fragA-native** `[b, h, 32, 8]` (matmul-4
      fragA; `STG.128` store / `LDG.128` load, no smem on either side);
      `decay_vec` = f32 `(b, h, NPREDICTED_PAD_MMA_M)`.
- [x] PDL chain `conv1d → precompute → main`.

Open (default chosen, change if desired):
- [ ] First-cut scope: **bf16 state, write + nowrite, non-varlen** (the
      profiled regime, b8–128). fp8/int8 + varlen deferred to a follow-up.

## Test (xfail first)

- [ ] **T0** *EASY* — `tests/mamba/test_checkpointing_ssu.py`: add
      `test_two_kernel_matches_monolithic` — call `checkpointing_ssu` twice on
      identical inputs (bf16, a write case and a nowrite case): once
      **without** scratch (monolithic) and once **with** caller-allocated
      `cb_scaled`/`decay_vec` (two-kernel). Assert bit-for-bit (or `atol`
      matching the monolithic's own determinism) on `out`, `state`, and the
      `old_x/old_B/old_dt/old_dA` cache. Initially `xfail` (two-kernel
      unimplemented → mismatch/raise).

## Implementation steps

- [ ] **S1** *EASY* — Python API: add optional
      `cb_scaled: Optional[torch.Tensor] = None` and
      `decay_vec: Optional[torch.Tensor] = None` to `checkpointing_ssu`
      (`flashinfer/mamba/checkpointing_ssu.py:200`). Dispatch:
      `two_kernel = cb_scaled is not None`; require both-or-neither with a
      clear value error; thread to `_checkpointing_ssu` (`:74`).
- [ ] **S2** *EASY* — Wire scratch (no wrapper allocation): the **caller**
      pre-allocates `cb_scaled` (bf16 `[b,h,32,8]`, fragA-native) + `decay_vec`
      (f32 `[b,h,NPREDICTED_PAD_MMA_M]`) — exactly like `out` (graph-safe). The
      wrapper only views + passes
      pointers. Add the two pointers (+ strides) to `CheckpointingSsuParams`
      (`csrc/checkpointing_ssu_customize_config.jinja` + the params struct);
      consumed only on the two-kernel launch.
- [ ] **S3** *HARD* — Precompute kernel: new
      `checkpointing_ssu_precompute_kernel` (new
      `include/flashinfer/mamba/kernel_checkpointing_ssu_precompute.cuh`).
      **Grid `(batch, ngroups)`** with a head loop/tile over `HEADS_PER_GROUP`
      — compute the per-group C·B once, scale into per-head
      `cb_scaled`/`decay_vec` (see "Granularity" above) — **not**
      `(batch, nheads)` like the monolithic kernel. Reuse the `dt/dA/decay`
      + `compute_CB_scaled`/`compute_CB_old` + `store_old_B/dt/dA` paths; add
      gmem stores of `cb_scaled` + `decay_vec`. Sub-step: read
      `compute_CB_scaled_2warp` / `compute_CB_old_2warp` to pin the exact
      smem→gmem boundary and the per-group B/C reuse across the group's heads.
- [ ] **S4** *HARD* — Main kernel: new `checkpointing_ssu_main_kernel`
      (new `kernel_checkpointing_ssu_main.cuh`). **Grid `(batch, nheads)`**
      (per-head state recurrence — unlike precompute's `(batch, ngroups)`).
      **`cb_scaled` loads via one `LDG.128`/thread straight into matmul-4's
      `fragA`** (fragA-native layout — no smem / LDSM / swizzle); `decay_vec`
      gives β for OUT.1.  Run `ssu_checkpoint`/`ssu_nocheckpoint` (replay +
      output) reading those instead of the smem-computed versions; `store_old_x`
      + final state.
- [ ] **S5** *MEDIUM* — Launcher: in the `csrc` launcher (the `.cu` that
      launches `checkpointing_ssu_kernel`), when the scratch pointers are
      non-null launch precompute then main with the PDL chain
      (`cudaLaunchAttributeProgrammaticStreamSerialization`), each
      `enable_pdl`-stamped so the `griddepcontrol` wait/signal is present;
      else launch the monolithic kernel as today.
- [ ] **S6** *MEDIUM* — JIT: extend `gen_checkpointing_ssu_module`
      (`flashinfer/jit/mamba/checkpointing_ssu.py`) to compile the two new
      kernels (new sources). Both paths live in one module — dispatch is
      runtime (scratch presence), so no new URI key needed beyond the sources.
- [ ] **S7** *EASY* — Bench: add a `--two-kernel` flag to
      `bench_checkpointing_ssu.py` that pre-allocates `cb_scaled`/`decay_vec`
      in `build_kernel_inputs` and passes them through the `cuda-incr`
      dispatch (`:814`/`:839`) (absence ⇒ monolithic); surface in
      `bench_ssu_checkpoint_mixed.py`.
- [ ] **S8** *EASY* — Flip the T0 test from `xfail` to a real assert; run
      `uv run pytest tests/mamba/test_checkpointing_ssu.py -k two_kernel`.
- [ ] **S9** *EASY* — Bench: `--with-conv1d` sweep b8–128, bf16, compare
      two-kernel (scratch passed) vs monolithic post-conv1d span. Use the
      `SSU_CUPTI_DEBUG` per-kernel dump to confirm the post-signal phase is
      flat (the cliff is gone).

## Risks / open questions

- **Extraction correctness (S3/S4)**: the monolithic kernel keeps
  `CB_scaled`/`decay`/state in smem and consumes them in-place; splitting
  means materializing `cb_scaled`/`decay_vec` to gmem and reloading. The
  per-warp partitioning + `__syncthreads` (`:1087`) cross-warp visibility
  notes must be preserved across the kernel boundary.
- **PDL chain vs fork-join**: if the timeline shows precompute serialized
  too far behind conv1d (precompute needs `B/C`, so it can't start until
  conv1d signals), escalate to the fork-join variant. Measure first.
- **Per-group writer ↔ per-head reader layout**: precompute writes
  `cb_scaled`/`decay_vec` per-head from a `(batch, ngroups)` grid (head loop);
  main reads them per-head from `(batch, nheads)`. The fragA-native
  `[batch, nheads, lane, 8]` indexing (and `decay_vec[batch, nheads, t]`) must
  match exactly on both sides — the lane/register → `(t,j)` mapping is the
  contract; get it identical in writer and reader.
- **Scratch dtype/precision**: `cb_scaled` is **bf16** = the monolithic's smem
  CB `operand_t`, so the bf16→gmem→bf16 round-trip is bitwise lossless and T0's
  equivalence holds.  (Triton's fp32 scratch was for its `tl.dot` path; our MMA
  wants the bf16 operand.)  `decay_vec` stays f32.
- **Determinism**: the monolithic kernel has known per-launch state-write
  nondeterminism in fp16 with `must_checkpoint`; pick the T0 tolerance to
  match the monolithic's own run-to-run spread, not zero.
- **CUDA-graph safety (why caller-provided)**: an internal `torch.empty` *is*
  graph-safe under PyTorch + `torch.cuda.graph` (the graph-private mempool
  allocates once at capture, replay reuses the address — no per-replay
  footprint; this is why the Triton reference is fine). But that rescue is
  PyTorch-allocator + `torch.cuda.graph` specific. FlashInfer is
  framework-agnostic (TVM-FFI) and is captured by serving stacks (TRT-LLM,
  raw `cudaGraph` API) with no such pool — so the scratch is **caller-provided
  and never allocated in the wrapper**, matching `out`/`float_workspace_buffer`
  and graph-safe under any capture path. The eager (non-graph) path also pays
  no per-call alloc this way.
