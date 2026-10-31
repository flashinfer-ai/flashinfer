# Qwen 3.5 GDN block (experimental)

**Status: experimental API and experimental backend.** The names, signatures, forms and backends may change or be
removed without deprecation. Calling an entry point of `flashinfer.qwen_3_5_gdn_block` is the opt-in (it emits the
standard `ExperimentalWarning` once); no environment variable is read, nothing is reachable from a stable API's
`backend="auto"`, and nothing is registered for ahead-of-time compilation (JIT only). **No performance claim is made
in this draft**: the tests establish correctness only.

## What it computes

The gated-delta-net sub-layer of a Qwen3.5-style model between its QKVZ projection and its out projection, in one
launch per step. With `Hv, Hk, Dv, Dk, K` = `num_v_heads, num_k_heads, head_v_dim, head_k_dim, conv_kernel`,
`C = 2 Hk Dk + Hv Dv`, V head `h` reading K head `h // (Hv / Hk)`, per token:

```
qkv [C] --causal conv over the slot's history (K taps, each tap product rounded to BF16, FP32 sum) + SiLU--> q | k | v (BF16)
hidden [d_model] --BA projection (ba_weight rows 0..Hv-1 -> b, Hv..2Hv-1 -> a), FP32 dot, BF16--> b, a
beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)            (softplus threshold 20)
q = q / sqrt(|q|^2 + 1e-6) / sqrt(Dk);  k = k / sqrt(|k|^2 + 1e-6)
S = exp(g) S;  S += beta (v - S k) k^T;  o = S q                       (S: [Dv, Dk] per V head, FP32)
o -> BF16;  y = BF16(o * rsqrt(mean_Dv(o^2) + norm_eps) * norm_weight * silu(z))
out = FP8_E4M3(satfinite(y * fp32(1 / out_scale)))                      (static per-tensor scale)
```

Two forms, each its own plan:

| form | entry point | step | state semantics |
|---|---|---|---|
| decode | `qwen_3_5_gdn_block_decode` | one token per request, `rows = requests` | row `r` reads and writes slot `state_indices[r]` of both pools; the conv history is the slot's `K - 1` columns (oldest first) and becomes `[hist[1:], x]` |
| verify | `qwen_3_5_gdn_block_verify` | MTP verify, `T = 1 + drafts` tokens per request, `rows = requests * T` | per-token state layout: request `n` with `a = num_accepted_tokens[n]` starts from slot `state_indices[n, a - 1]` and writes the state after token `t` to slot `state_indices[n, t]`; its conv slot `state_indices[n, 0]` holds `T + K - 2` columns, the history is columns `a - 1 .. a + K - 3`, and the slot becomes `[old[a : a + K - 2], x_0 .. x_{T-1}]` |

A decode row with slot `<= 0`, or a verify request whose source slot `state_indices[n, a - 1]` is `<= 0`, is padding:
zero outputs, no state touched. A verify slot `state_indices[n, t] <= 0` (`t > 0`) skips that state write. A live
verify request needs `state_indices[n, 0] > 0` and every request carries exactly `T` tokens (`cu_seqlens[n] = n T`).

## Arguments and layouts

| argument | shape | dtype | layout |
|---|---|---|---|
| `qkv` | `[rows, C]` | bf16 | the projected `q \| k \| v` before the conv; channels contiguous, any 16-byte-multiple row stride |
| `z` | `[rows, Hv, Dv]` | bf16 | the output gate; heads contiguous, any 16-byte-multiple row stride |
| `hidden` | `[rows, d_model]` | bf16 | contiguous; the normed layer input the BA projection reads |
| `ba_weight` | `[2 Hv, d_model]` | bf16 | contiguous; `b` rows then `a` rows |
| `A_log`, `dt_bias` | `[Hv]` | fp32 or bf16 | contiguous |
| `ssm_state` | `[num_slots, Hv, Dv, Dk]` | fp32 | each slot contiguous, any 16-byte-multiple slot stride |
| `state_indices` | decode `[rows]`; verify `[requests, T]` | int32 | verify: any row stride |
| `cu_seqlens` (verify) | `[requests + 1]` | int32 | contiguous |
| `num_accepted_tokens` (verify) | `[requests]` | int32 | contiguous, each in `[1, T]` |
| `norm_weight` | `[Dv]` | fp32 or bf16 | contiguous |
| `out_scale` | one element | fp32 | the static FP8 scale (`out = fp8(y / out_scale)`) |
| `conv_state` | decode `[num_slots, C, K - 1]`; verify `[num_slots, C, T + K - 2]` | bf16 | a view, oldest column first; FROST verify needs it channel-contiguous (stored `[num_slots, L, C]`) |
| `conv_weight` | `[C, K]` | bf16 | contiguous; tap `K - 1` multiplies the newest input |
| `out` (optional) | `[rows, Hv, Dv]` | float8_e4m3fn | contiguous; default: the plan's buffer, overwritten by its next run |
| `out_bf16` (optional) | `[rows, Hv, Dv]` | bf16 | contiguous; also receive `y` before quantization |

Every tensor lives on the plan's device (the current CUDA device) and is 16-byte aligned. Device-side values (slot
ids, offsets, accepted counts) are not read on the host; slot ids must be valid in both pools. The dtypes of `A_log`,
`dt_bias` and `norm_weight` are part of a plan.

## Plans, the cache and CUDA graphs

`prepare_qwen_3_5_gdn_block(form, geometry=, device=, requests=, tokens_per_request=, backend=, param_dtypes=)` returns
a `Qwen35GdnPlan` keyed by its `Qwen35GdnPlanSpec` (backend, form, requests, tokens per request, geometry, device,
parameter dtypes). `plan.run(**tensors)` checks the tensors against the spec (typed `ValueError` naming the operand),
lets the backend specialize for them when eager, and launches on the current stream. Pass `plan=` to the entry points,
or let them use their LRU (32 plans; `flashinfer.experimental.qwen_3_5_gdn_block.api.clear_plan_cache()` drops it).

CUDA graphs: build the plan and run each binding once eagerly, then capture `plan.run` (or the entry point with a
warm cache). A capture of a cold binding raises `RuntimeError`; a cached plan a graph was captured over is retained past
its LRU eviction until `clear_plan_cache()`. Runs of one plan must be ordered on one stream (the plan's default output
and the backend's scratch buffers are shared by its runs).

`qwen_3_5_gdn_block_support(device, geometry, form, backend="frost", requests=None, tokens_per_request=None)` is the
init-time predicate: it never raises, and a declined answer carries `kind` (`argument`, `backend`, `form`, `device`,
`dsl`, `geometry`, `shape`) and `reason`. `prepare` applies the same gate: `ValueError` for `argument`,
`NotImplementedError` for the rest.

## The backend interface

A backend registers two callables (`flashinfer.experimental.qwen_3_5_gdn_block.api`):

```python
from flashinfer.experimental.qwen_3_5_gdn_block import api
from flashinfer.experimental.qwen_3_5_gdn_block.support import Qwen35GdnDecline

def support_fn(device, geometry, form, requests, tokens_per_request):
    """None = served, else Qwen35GdnDecline(kind, reason). Cheap, no kernel import, should not raise.
    requests / tokens_per_request are None when the caller asks without a step shape; decode passes T = 1."""

def build_fn(spec):  # spec: api.Qwen35GdnPlanSpec
    """Called once per plan, outside capture, after support_fn admitted the spec; may compile and allocate.
    Returns a runner with prepare(inputs) and run(inputs)."""

api.register_backend("my_backend", support_fn, build_fn)
```

The runner receives an `api.Qwen35GdnInputs` (the checked tensors of the table above; `out` always set, `out_bf16`
`None` unless asked, `cu_seqlens` / `num_accepted_tokens` on verify only):

- `prepare(inputs)`: before every eager run, never under capture; compile or specialize for this binding (strides,
  alignment) and be cheap when already done. Raise `NotImplementedError` for a binding the backend does not serve.
- `run(inputs)`: launch on `torch.cuda.current_stream()`, write `inputs.out` (and `inputs.out_bf16` when set), update
  `conv_state` and `ssm_state` in place with the semantics above (including padding rows / requests). Capturable once
  `prepare` saw an equivalent binding; raise `RuntimeError` under capture otherwise.

A new backend is validated against the same reference: `tests/experimental/qwen_3_5_gdn_block/reference.py`
(`make_case`, `reference`) is backend-independent, and the checks of `test_frost_*.py` apply unchanged to a call with
`backend="my_backend"`.

## The FROST backend

`backends/frost/`: CuTe DSL kernels written for SM100 (cc 10.0) at one geometry, Qwen3.8-27B at TP 1 (`d_model` 5120,
48 V heads, 16 K heads, head dims 128, conv kernel 4). `gdn.py` (`FrostGdnConsumer`, decode: one launch, a cluster of 2
CTAs per V head at one row, 1 from two rows on) and `gdn_verify.py` (`FrostGdnVerify`, verify: the fused speculative
conv, the per-token state layout, no programmatic dependent launch) are ported byte-identical in code from the serving
branch they were validated in (only headers, docstrings and comments changed); `admission.py` is the lightweight gate
and `runner.py` binds a plan's inputs to the two wrappers (whose compiled kernels are shared by every plan that compiles
the same code).

Served: decode at up to 65535 rows (one grid row per request, CUDA's grid Y limit); verify at `2 <= T <= 10` tokens per request for one request and `2 <= T <= 8` from
two requests on, up to 4 requests (the fused conv's envelope; a grid within one wave would admit more, which no current
part reaches), with a channel-contiguous conv pool. The launch policies (CTAs per head, resident CTAs per SM, state-copy
ring depth) follow the device's SM count and were tuned on a 148-SM B200.

## What is validated

`tests/experimental/qwen_3_5_gdn_block/` (GPU modules skip unless the device is cc 10.0 and the probe serves it):

- `test_support.py` (CPU): geometry form checks, typed declines of the probe (which never raises), the FROST
  step-shape envelope, typed plan-time errors, the registry and plan cache with a fake backend, operand checks, and
  that importing the API and probing load no kernel.
- `test_frost_decode.py`: rows 1, 2, 3, 4, 8, 16, 64 (the last row a padding row from two rows on); FP32 `dt_bias` /
  norm weight with a dim-first conv pool; a CUDA-graph replay bitwise equal to the eager run.
- `test_frost_verify.py`: requests 1, 2, 4 x T 4, 6 (the last request a padding request from two on, random accepted
  counts), each under the device's launch policy and the 148-SM policy; a CUDA-graph replay; the typed decline of a conv
  pool the fused conv cannot read.

The reference (`reference.py`) is pure torch: the whole chain with the kernels' rounding points, the recurrence in
FP64. Inputs are small-magnitude and seeded; `hidden` in {-1, 0, 1} and `ba_weight` in k/256 make the BA projection
exact before its BF16 rounding, and conv inputs whose exact SiLU lies within 2^-16 of a BF16 rounding midpoint are
moved (the kernels' SiLU uses `ex2.approx`), so the reference's q / k / v equal the kernels'. Tolerances are the ones
the kernels were validated with against the unfused op chain: FP8 bytes agree on more than 99.9% of live outputs, the
BF16 `y` within 2^-6 relative + 1e-3, the conv pool bitwise, written SSM slots within 1e-4 of the FP64 recurrence,
other slots untouched, padding outputs zero; and the FP8 output bitwise equal to the static quantization of the
kernel's own BF16 `y`. The tests pass on a B200 (cc 10.0, 148 SMs) with nvidia-cutlass-dsl 4.8.0 and torch 2.14
(cu130).

## Not in this draft (future work)

- The **journaled verify** (`FrostGdnVerify(journal=True)`: one base state per request plus per-token update terms,
  replayed bit-identically by the next verify) and its materialize kernel, which restores the per-token layout for
  readers of that layout. The kernel code is in `gdn_verify.py` but not reachable from the API; exposing it needs the
  materialize kernel ported and a journal-mode tensor in the verify signature.
- Programmatic dependent launch / early release of the next kernel (the verify wrapper's `pdl` / `release` knobs are
  compiled off).
- Verify beyond 4 requests (served elsewhere by a separate speculative conv followed by the verify on convolved
  inputs), other geometries or tensor-parallel shards, SM103, and requests with fewer than `T` tokens.
- A second backend: register it with `api.register_backend` (see "The backend interface").
