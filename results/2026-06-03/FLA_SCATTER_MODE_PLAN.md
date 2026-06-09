# Plan: FLA-style per-token pool scatter for FlashInfer GDN MTP

**Date:** 2026-06-03
**Branch target:** atop `ameyn/fused_recovery_decode` (the 7-commit stack documented in `GDN_BF16_KERNEL_MODES.md`)
**Scope:** BF16-state kernel only — `flashinfer/gdn_kernels/gdn_decode_bf16_state.py`. FP32-state (`gdn_decode_mtp.py`) deferred.

---

## 1. Why this is a real win

vLLM (Vadim Gimpelson) needs FlashInfer's `gated_delta_rule_mtp` to expose the same API shape as FLA's Triton kernel — per-token intermediate states scattered directly into pool slots via an `ssm_state_indices[B, T]` int32 tensor, rather than written to a dense per-call `intermediate_states_buffer`.

End-to-end per-iter comparison at Qwen3.5 TP=4 / B=256 / T=8 / 24 GDN layers:

| | Speculation kernel | Post-verify scatter | **Total** |
|---|---|---|---|
| Path A (dense + fused scatter) | ~7.5 ms (contiguous writes, ~85% peak BW) | ~0.95 ms (`fused_mamba_state_scatter_with_mask`) | **~8.5 ms** |
| **FLA mode (this plan)** | ~7.8 ms (scattered pool writes, ~80% peak BW) | **0** | **~7.8 ms** |

State writes are ~30–40 % of the speculation kernel's memory traffic (Q/K/V/gate compute, state reads, output emission take the rest). A 10–25 % write-phase slowdown on the scattered writes translates to a 3–8 % whole-kernel slowdown — well under the 948 µs saved by eliminating the post-verify scatter.

**Three wins:**

1. **~5–10 % faster end-to-end** per spec-decode iter at the production cell.
2. **One launch instead of two.** vLLM doesn't have to port or maintain sglang's fused Triton scatter; the kernel does the right thing on its own.
3. **One fewer CUDA graph node per layer.** Path A's post-verify scatter is a separate launch that has to be captured/replayed; FLA mode collapses it.

Memory budget is unchanged (~27 GB at the production cell either way — the 24 GB just moves from `intermediate_states_buffer` into the persistent pool). The real memory win is still Path B/C; this mode is for API/perf compat with vLLM's existing architecture.

---

## 2. Goals and non-goals

### Goals
1. Add optional `ssm_state_indices: Tensor[B, T] int32` to the public `gated_delta_rule_mtp` wrapper.
2. When provided, the kernel writes `h_{t+1}` directly to `pool[ssm_state_indices[i, t]]` for each `(i, t)`.
3. Bit-identical state values to the existing dense-buffer mode at the corresponding `(i, t)`.
4. Compose cleanly with `accepted_steps` (per-request K), `output_state_indices` (split-pool final write), `disable_output=True` (state-only mode).
5. Bench shows FLA mode **beats Path A on total wallclock** at production shapes (BS=256, T=8) — that's the ship gate.
6. Zero regression on the existing 529-case test grid.

### Non-goals
- Replace dense-buffer mode. Both ship; caller selects via parameter presence.
- FP32-state kernel (`gdn_decode_mtp.py`) — follow-up.
- Combination with `recovery_steps > 0` in MVP — Phase A semantics complicate the t-loop; deferred.
- vLLM-side integration wiring — downstream of this plan.

---

## 3. API design

### 3.1 Wrapper signature

```python
def gated_delta_rule_mtp(
    A_log, a, dt_bias,
    softplus_beta=1.0, softplus_threshold=20.0,
    q=None, k=None, v=None, b=None,
    initial_state_source=None,
    initial_state_indices=None,
    output_state_indices=None,
    intermediate_states_buffer=None,
    accepted_steps=None,
    # NEW:
    ssm_state_indices=None,            # int32[B, T] — per-token pool slots
    disable_state_update=False,
    use_qk_l2norm_in_kernel=True,
    scale=None,
    output=None,
    tile_v=128,
    disable_output=False,
    recovery_steps=0,
) -> torch.Tensor: ...
```

### 3.2 Semantics

`ssm_state_indices[i, t]` is a slot index into `initial_state_source`'s pool. After step `t`, the kernel writes `h_{t+1}` to `initial_state_source[ssm_state_indices[i, t], :]`. The caller pre-allocates T fresh slots per request from its pool free-list and sizes the pool for at least `B × (T+1)` concurrent states.

### 3.3 Mutual exclusion / legal combinations

| Other parameter | Legal with `ssm_state_indices`? | Reason |
|---|---|---|
| `intermediate_states_buffer` | **No** — mutex | Two destinations for the same writes |
| `disable_state_update=True` | **No** — mutex | Nothing to scatter |
| `recovery_steps > 0` | **No** (MVP) | Phase A boundary semantics deferred |
| `accepted_steps` (per-request K) | **Yes** | Kernel writes only first `accepted_steps[i]+1` slots; rest unused |
| `output_state_indices` | **Yes** | Final-state write independent of per-token writes |
| `disable_output=True` | **Yes** | State-only mode with per-token scatter is valid |

### 3.4 Python wrapper validation

```python
if ssm_state_indices is not None:
    assert intermediate_states_buffer is None, "mutually exclusive"
    assert not disable_state_update, "requires state writes"
    assert recovery_steps == 0, "not yet supported"
    assert initial_state_source is not None and initial_state_indices is not None
    assert ssm_state_indices.shape == (B, T)
    assert ssm_state_indices.dtype == torch.int32
    assert ssm_state_indices.device == q.device
    # NOT asserted (perf): in-bounds slot indices, no duplicates → caller's responsibility
```

---

## 4. Kernel changes

### 4.1 The three write sites

`grep -n "cache_intermediate_states" flashinfer/gdn_kernels/gdn_decode_bf16_state.py`:

| Kernel family | Write-site lines | Notes |
|---|---|---|
| `gdn_decode_bf16state_mtp_ilp4_kernel` | L670–706 | Small-batch ILP4 path |
| `gdn_wide_vec_kernel_t1` | L1204–1216 | T=1 dedicated kernel |
| `gdn_wide_vec_kernel` (T≥2) | L1399–1435 (cache mode); L1653–1670 (stateless path) | Two write paths in one kernel |

All three families take `intermediate_states: cute.Tensor` and `cache_intermediate_states: cutlass.Constexpr[bool]`.

### 4.2 New constexpr

Add `per_token_pool_scatter: cutlass.Constexpr[bool]` to each kernel signature, alongside `cache_intermediate_states`. The two are mutually exclusive at the wrapper layer but are independent flags at the kernel level. Both gate independent write paths.

### 4.3 The write substitution

**Current pattern** (paraphrased from L1399–1435):

```python
if cutlass.const_expr(cache_intermediate_states):
    # flat_idx = i_n * T * HV + i_t * HV + i_hv
    # intermediate_states is reshaped to [B*T*HV, V, K]
    cute.copy(intermediate_states, (vec, v0, lane_in_group), offset=(flat_idx, ...))
```

**New pattern** routes by mode:

```python
if cutlass.const_expr(per_token_pool_scatter):
    pool_slot = cutlass.Int64(smem_ssm_slot[i_t])     # cached at CTA prelude; mul.wide.s32
    pool_slot_view = cute.slice_(
        initial_state_source, (pool_slot, None, None, None)
    )
    cute.copy(pool_slot_view, (vec, v0, lane_in_group), ...)
elif cutlass.const_expr(cache_intermediate_states):
    # Existing dense-buffer write — unchanged.
    cute.copy(intermediate_states, (vec, v0, lane_in_group), offset=(flat_idx, ...))
```

The pool-slot view uses the **slot-slice idiom** Amey introduced in commit `bb2cb8ea` (`mul.wide.s32` for the Int64 slot base offset; Int32 strides within the slot). This pattern is already proven on the final-state write — we apply it at every `i_t` instead of once at the end.

### 4.4 Per-step slot-index loading

Strategy: **shmem-cache the T indices at CTA entry.**

```python
if cutlass.const_expr(per_token_pool_scatter):
    # Load T int32s (≤64 bytes for T≤16) into shmem once per CTA.
    for t in cutlass.range_constexpr(T):
        smem_ssm_slot[t] = cutlass.Int32(ssm_state_indices[i_n, t])
```

Per-step reads in the t-loop become 1-cycle shmem loads. Negligible shmem pressure.

### 4.5 Final-state writeback interaction

When `ssm_state_indices` is provided:

- **`output_state_indices is None`:** the per-token scatter at `t == K-1` IS the final state. Skip the existing final-state writeback to avoid a double-write to the same slot.
- **`output_state_indices is not None`:** split-pool semantics. Per-token scatter writes to `ssm_state_indices[i, t]` for all valid t; final state is *additionally* written to `pool[output_state_indices[i]]` (consistent with existing split-pool semantics).

Implementation: gate the existing final-state writeback site behind `(not per_token_pool_scatter) or (output_state_indices is not None)`.

### 4.6 Combination with `accepted_steps`

No additional kernel change. The t-loop already early-exits at `t == accepted_steps[i] + 1` (commit `019a66cb`); the per-token scatter rides along. Slots `ssm_state_indices[i, accepted_steps[i]+1:]` go unused — caller's responsibility to reclaim or pad.

### 4.7 Files touched

| Function / location | File | Change |
|---|---|---|
| `gdn_decode_bf16state_mtp_ilp4_kernel` | `flashinfer/gdn_kernels/gdn_decode_bf16_state.py` (L67–L730) | Add constexpr; route write site at L670 |
| `gdn_wide_vec_kernel_t1` | same file (L760+) | Add constexpr; route write site at L1204 |
| `gdn_wide_vec_kernel` (T≥2) | same file (~L770–1700) | Add constexpr; route write sites at L1399 + L1653 |
| `gated_delta_rule_mtp` wrapper | same file, end | Add kwarg + validation per §3.4; thread `per_token_pool_scatter` to kernel launchers |
| CTA prelude (each kernel) | same file | Shmem-cache slot indices per §4.4 |

---

## 5. Test plan

### 5.1 Gating test — bit-equivalence vs dense mode

**`test_gdn_decode_bf16_state_fla_scatter_vs_dense`** (new in `tests/gdn/test_decode_delta_rule.py`).

Grid (75 cells):

| Axis | Values |
|---|---|
| `dtype` | `"bfloat16"` |
| `head_size` | `128` |
| `(num_q_heads, num_k_heads, num_v_heads)` | `(16, 16, 64)` (Qwen3.5) |
| `batch_size` | `1, 4, 16, 64, 256` |
| `T` | `2, 4, 8` |
| `split_pool` | `False, True` |

Logic:

```python
# 1. Random inputs + initial pool.
# 2. Allocate B*T fresh, unique slots from a free-list:
ssm_state_indices = generate_unique_pool_slots(B, T, pool_capacity, exclude=h0_indices)

# 3. Run FLA mode on pool_fla.
pool_fla = pool_init.clone()
gated_delta_rule_mtp(
    q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
    initial_state_source=pool_fla, initial_state_indices=h0_indices,
    ssm_state_indices=ssm_state_indices,
    output_state_indices=out_idx if split_pool else None,
    use_qk_l2norm_in_kernel=True, scale=K_dim ** -0.5,
)

# 4. Run dense mode as reference on pool_dense.
intermediate_buffer = torch.empty((B, T, HV, K_dim, V), dtype=...)
pool_dense = pool_init.clone()
gated_delta_rule_mtp(
    q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
    initial_state_source=pool_dense, initial_state_indices=h0_indices,
    intermediate_states_buffer=intermediate_buffer,
    output_state_indices=out_idx if split_pool else None,
    use_qk_l2norm_in_kernel=True, scale=K_dim ** -0.5,
)

# 5. Compare slot-by-slot:
for i in range(B):
    for t in range(T):
        slot = int(ssm_state_indices[i, t].item())
        diff = (pool_fla[slot] - intermediate_buffer[i, t]).abs().max().item()
        assert diff <= 0.06

if split_pool:
    assert (pool_fla[out_idx] - pool_dense[out_idx]).abs().max() <= 0.06
```

Tolerance 0.06 (BF16 envelope, matches existing tests).

### 5.2 Combination tests

- `test_..._with_accepted_steps` (10 cells) — verify slots `> accepted_steps[i]` are unchanged from `pool_init`.
- `test_..._state_only` (5 cells) — `disable_output=True` + `ssm_state_indices`; no output materialized, per-token scatter correct.

### 5.3 Negative tests

Parametrized — confirm each illegal combination raises:

```python
@pytest.mark.parametrize("combo,expected_msg", [
    ({"intermediate_states_buffer": ...},       "mutually exclusive"),
    ({"disable_state_update": True},            "requires state writes"),
    ({"recovery_steps": 1},                     "not yet supported"),
    ({"ssm_state_indices_dtype": torch.int64},  "int32"),
    ({"ssm_state_indices_shape": (B, T+1)},     "shape"),
])
def test_fla_scatter_validation(combo, expected_msg): ...
```

### 5.4 Optional: cross-validate against FLA's Triton kernel

Gate on `FLASHINFER_TEST_FLA_REF=1`. If a working FLA Triton kernel is available locally (`flash-linear-attention` package or vLLM bundle), compare pool contents at each `ssm_state_indices[i, t]` slot. Tolerance 0.06.

---

## 6. Bench plan

### 6.1 Bench infra

Add `--ssm-state-indices` flag to `benchmarks/bench_gdn_decode.py`. When set, the bench generates unique pool slots (`randperm(pool_capacity)[:B*T].reshape(B, T)`) and enlarges the bench pool to `B × (T+1)` slots.

### 6.2 Kernel-only sweep — isolates the write-pattern penalty

| Grid | Values |
|---|---|
| `T` | 1, 2, 4, 8, 16 |
| `BS` | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |

Same Qwen3.5 config (HV=64, K=V=128, BF16). Expected: FLA mode ~3–8 % slower on the speculation kernel than dense mode.

### 6.3 Total-time sweep — the actual ship-gate bench

Compare two configurations end-to-end (speculation + post-verify):

- **Path A:** dense-mode kernel + sglang's `fused_mamba_state_scatter_with_mask` (or equivalent), measured at scatter cost ≈ 948 µs/iter at B=256.
- **FLA mode:** speculation kernel only (no post-verify scatter).

**Ship gate:** FLA mode total wall-clock < Path A total wall-clock at the production cell (BS=256, T=8). Expected delta: FLA wins by ~5–10 % (~0.7 ms / iter).

If FLA loses at the production cell, this plan changes — we'd ship as opt-in but recommend Path A as default. **That is not the expected outcome.**

### 6.4 ncu analysis at BS=256, T=8

| Metric | Expected FLA vs dense |
|---|---|
| Duration | +3–8 % |
| Achieved bandwidth | −5–10 % |
| L2 hit rate (this kernel) | similar |
| L2 hit rate (next kernel's reads) | possibly −few % (pollution) |
| DRAM bytes written | ≈ equal |
| Registers / thread | +0–2 (shmem-cached indices) |
| Local memory spills | **0** (hard gate) |

### 6.5 Output file

`results/2026-06-03/fla_scatter_mode_bench.md` — committed alongside the code changes. Contains:
- Kernel-only table (T × BS, FLA slowdown %).
- Total-time table (T × BS, FLA total vs Path A total, winner column).
- ncu metrics for BS=256/T=8.
- Conclusion: ship-as-default, ship-as-opt-in-with-Path-A-default, or do-not-ship.

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Register pressure from per-step slot reads → spills | Medium | High | Shmem-cache T indices (§4.4); ncu register gate ≤ +6 regs/thread; spills must be 0 |
| L2 pollution degrades downstream kernel reads | Medium | Medium | Measure L2 hit rate on next-kernel reads in bench; Qwen3.5 end-to-end smoke if available |
| Mutex matrix bugs (untested combo) | High | Medium | Explicit validation table (§3.4) + parametrized test grid covering each documented combo |
| Caller's `ssm_state_indices` has out-of-range slots → silent corruption | Low | Critical | Document the requirement clearly; optionally add `validate_indices=True` debug flag (~+1 µs/call, off by default) |
| Total-time bench reveals FLA loses at production cell | Low | Medium | Ship as opt-in only; doc Path A as default. (Not expected.) |
| Vadim asks for `recovery_steps > 0` combination before MVP ships | Low | Low | Document the MVP exclusion; punt to a follow-up plan |

---

## 8. Ship gate

The FLA mode ships iff:

1. ✅ Bit-equivalence test (§5.1) passes at all 75 cells.
2. ✅ Combination tests (§5.2) pass.
3. ✅ Negative tests (§5.3) all raise as expected.
4. ✅ **Total-time bench (§6.3) shows FLA wins at the production cell (BS=256, T=8).**
5. ✅ Kernel-only bench (§6.2) shows ≤ +8 % slowdown vs dense mode at the production cell.
6. ✅ ncu (§6.4) shows zero local-mem spills and ≤ +6 regs/thread.
7. ✅ All 529 existing tests still pass.
8. ✅ `results/2026-06-03/fla_scatter_mode_bench.md` committed.

If (4) fails: pause and re-evaluate. Don't reflexively ship as opt-in — figure out why first (CTA scheduling, L2 churn, something else), since the perf model predicts a win.

---

## 9. Estimate

| Phase | Work | Time |
|---|---|---|
| 1 | Kernel changes (3 kernel families × write-site routing), wrapper + validation, CTA prelude slot loads | 3 days |
| 2 | Bit-equivalence + combination + negative tests (§5.1–5.3) | 1 day |
| 3 | Bench infra (`--ssm-state-indices` flag, total-time grid, ncu sweep) | 1 day |
| 4 | Bench result write-up + ship/no-ship decision | 1 day |
| 5 | (Stretch) FLA-Triton cross-validation, if reference available | 0.5 day |
| **Total** | | **~1 week** |

---

## 10. Commit stack

Four commits on top of the current `ameyn/fused_recovery_decode` tip (plus a fifth if Phase 5 lands):

1. **`feat(gdn): per-token pool scatter (FLA-style) in BF16-state kernel`** — kernel + wrapper + validation + tests.
2. **`bench(gdn): --ssm-state-indices flag + FLA-vs-dense kernel-only sweep`** — bench infra.
3. **`bench(gdn): FLA-mode total-time vs Path A across (BS, T) grid`** — result file under `results/2026-06-03/fla_scatter_mode_bench.md`.
4. **`docs(gdn): add FLA-mode (Mode 2.5) to GDN_BF16_KERNEL_MODES.md`** — extend the existing kernel-modes doc; update §7 follow-ups.
5. *(optional)* **`test(gdn): cross-validate FLA-mode against flash-linear-attention reference`** — gated on env var.

---

## 11. Open questions for confirmation

Before starting Phase 1, please confirm:

1. **Scope confirmed:** BF16-state only (`gdn_decode_bf16_state.py`)? Defer `gdn_decode_mtp.py` to a follow-up?
2. **FLA reference for cross-validation:** Is there a known-good FLA Triton kernel locally to compare against, or self-compare to dense-mode only?
3. **Index validation flag:** Add the optional `validate_indices=True` debug flag, or skip and rely on caller?
4. **Path C interaction:** OK to defer `recovery_steps > 0` + `ssm_state_indices` to a follow-up plan?
5. **Bench shapes:** Qwen3.5 config only, or sweep other models too?
