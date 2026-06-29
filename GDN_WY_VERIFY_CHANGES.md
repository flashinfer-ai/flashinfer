# GDN WY output-only verify kernel — change log

Scope: GDN MTP speculative-decode **verify** path. Validated on Qwen3.5-397B-A17B-NVFP4,
GB300 (sm_103), TP4, conc 256, draft-len 3 (T=4), `gdn-mtp-cache-mode=none`, with Triton
recovery (the FlashInfer state kernel is **not** on this runtime path). All changes are
**bit-exact** (verified by `max|Δ|=0` microbenches) and **lossless** end-to-end (gsm8k
unchanged, speculative accept-length unchanged).

## File changed (1)
- `flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` — kernel + host wrapper.

(An earlier GB300 stride-compaction tweak to `gdn_decode_bf16_state.py` was **dropped** — that
state kernel is only used under FlashInfer-recovery / full-mode, not in the Triton-recovery
output-only WY path shipped here.)

## API compatibility
The public wrapper `gated_delta_rule_mtp(...)` signature is **unchanged** (byte-identical
params/defaults). The two optimizations are gated by **default-off env flags**; the internal
`GdnDecodeKernel.__init__` gained 3 optional kwargs with backward-compatible defaults
(`n_valid=16, qkv_row_stride=0, ab_native=False`). No caller change required.

---

## Change 1 — NaN-safe decay matrix (correctness fix, always on)

**Problem.** The decay matrix element for (row r, col c) was `exp(cumsum_r)·exp(−cumsum_c)`.
For strong real decay (large `A_log`) the log-domain cumsum is large, so `exp(−cumsum_c)`
overflows to `+inf` while `exp(cumsum_r)` underflows to `0` → `0·inf = NaN`, silently breaking
MTP verify (gsm8k → 0). (The bench masked it with weak `A_log ~ randn`.)

**Fix.** Store the log-domain cumsum and form the decay as `exp(cumsum_r − cumsum_c)` (bounded
≤1 in the causal r≥c region). Mathematically identical, NaN-free.

```python
# warp-3 gamma compute: also stash the log-domain cumsum in the free sGamma[0:T] slots
exp_g = _exp_approx_f32(cumsum)
sGamma.iterator[T + lane_id] = exp_g
sGamma.iterator[lane_id]     = cumsum          # <-- added (log-domain cumsum)

# decay-matrix build (consumed only for r >= c):
# before: f32(1.0) if r == c else sGamma.iterator[T + r] * sBeta.iterator[T + c]
#                                  # = exp(cumsum_r) * exp(-cumsum_c)  -> overflows to inf
# after:
_exp_approx_f32(sGamma.iterator[r] - sGamma.iterator[c])   # = exp(cumsum_r - cumsum_c), <=1
```
Validated: gsm8k 0.000 → 0.585 on the real verify inputs.

---

## Change 2 — strided q/k/v read (drop `.contiguous()`)
Env flag: **`SGLANG_GDN_WY_STRIDED_QKV`** (default off). Native path (T∈{4,8}) only.

**Problem.** The kernel assumed canonical-compact q/k/v (token stride = `H*K`). In the verify
path q/k/v are **column slices of the fused conv output** (`mixed_qkv[:, slice]`), so their
token stride is `conv_dim`, not compact — the host `.contiguous()`-materialized all three
(~11µs, v is ~8MB) before the launch.

**Change.** The kernel already had per-token stride vars (`sq_t/sk_t/sv_t`); make them accept
the real conv row stride. Head/element strides unchanged (features stay contiguous within a
token → smem/ldmatrix/MMA untouched).

```python
# kernel strides (host-supplied via ctor qkv_row_stride; 0 => compact as before):
_rs = self._qkv_row_stride
_qt = _rs if _rs > 0 else H * K_DIM
_vt = _rs if _rs > 0 else HV * V_DIM
sq_t = _qt; sq_b = self._n_valid * _qt        # k analogous; v uses _vt
```
```python
# host wrapper (native branch): pass the row stride and skip the 3 copies
if _STRIDED_QKV and q.stride(1) == k.stride(1) == v.stride(1):
    _qkv_rs = q.stride(1)                      # read strided slices in place
else:
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()   # fallback
```
**Bit-exact:** same values, just strided gmem addressing (`max|Δ|=0`).
Result: q/k/v copies removed; conc256 13,036 → 13,483 tok/s (+3.4%).

---

## Change 3 — native a/b read (drop a/b staging)
Env flag: **`SGLANG_GDN_WY_NATIVE_AB`** (default off). Native path (T∈{4,8}) only.

**Problem.** Even on the native path the wrapper still zero-pad-staged a/b into `T_KERNEL`(=16)
buffers and copied the T valid rows (~4µs, 2 copies).

**Change.** Add an `ab_native` ctor kwarg: a/b batch stride uses `n_valid*HV` and the warp-3
a/b load + gamma compute gate by `_ab_rows` (= `n_valid` when native) so tail lanes
`[n_valid:T]` are not loaded.

```python
# kernel: a/b rows present = n_valid (native) else T
_ab_rows = self._n_valid if self._ab_native else T
sa_b = _ab_rows * HV; sb_b = _ab_rows * HV     # token/head strides unchanged
...
if warp_id == 3 and lane_id < _ab_rows:        # load gate (was: lane_id < T)
    _v7e_a_bf16 = gA.iterator[pid_b*sa_b + lane_id*sa_t + pid_hv*sa_hv].to(f32)
...
if lane_id < _ab_rows:                          # gamma-compute gate (was: lane_id < T)
    log_alpha = ...
```
```python
# host wrapper: pass native a/b, skip staging
if _native and _NATIVE_AB and a.is_contiguous() and b.is_contiguous():
    _ab_native_flag = True                      # a, b stay [B, n_valid, HV]
else:
    ... stage a/b into T_KERNEL zero-padded buffers (original) ...
```
**Bit-exact (on the real output):** gamma is a causal prefix-sum and the native path returns a
compact `[B,T]` output, so the unloaded tail rows (`log_alpha=0`) cannot affect rows
0..n_valid-1 and their output is discarded (`max|Δ|=0` on the compact output).
Result: a/b copies removed; full stack conc256 13,036 → 13,744 tok/s (+5.4%); verify region
~53µs → ~37µs.

---

## Env flags (default off; set on the serving process)
| flag | change | effect |
|---|---|---|
| `SGLANG_GDN_WY_NATIVE_T` | (pre-existing) | q/k/v native load + in-kernel smem-tail zero (no q/k staging) |
| `SGLANG_GDN_WY_STRIDED_QKV` | 2 | read q/k/v from strided conv-output slices (no `.contiguous()`) |
| `SGLANG_GDN_WY_NATIVE_AB` | 3 | read a/b native `[B,n_valid,HV]` (no staging) |

The NaN fix (Change 1) has no flag — it is always on.

## Validation
- Bit-exact microbenches (`max|Δ| = 0.000e+00`): `microbench_strided_qkv.py`, `microbench_native_ab.py`.
- Shape sweep via the repo's own `benchmarks/bench_gdn_decode.py --version bf16_wy_output_only`
  (built-in WY-vs-state `max|d|` cross-check + timing): T∈{4,8,16} × B∈{1..256}, HV=64.
- e2e: gsm8k 0.572/500 & 0.554/1000, accept-length 3.39–3.43, conc256 throughput, decode-only
  nsys/perfetto traces.
