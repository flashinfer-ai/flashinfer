# Experimental fused GDN decode step

`gdn_fused_decode_step` fuses one decode step of a gated-delta-net
linear-attention layer — the b/a projection GEMV, the causal conv1d state
update, the q/k/v split and the gated delta-rule decode — into a single op,
and ships with the specialized SM120 kernels that serve its registered
layer geometries.

The op is **exported at the top level**, like the other GDN APIs
(`flashinfer.chunk_gated_delta_rule` from `flashinfer/gdn_prefill.py`,
`flashinfer.gated_delta_rule_decode_pretranspose` from
`flashinfer/gdn_decode.py`).  *Experimental* describes where this code
lives, not how it is called:

```python
import flashinfer

if flashinfer.gdn_fused_decode_step_supported(batch_size, ..., conv_state_layout="SD"):
    flashinfer.gdn_fused_decode_step(hidden_states, w_ba, ..., out=core_attn_out)
else:
    ...  # keep your own composition
```

Its input contract is one architecture's layer geometry rather than a
reusable tensor primitive (the registry below is what makes that concrete),
which is why the registered surface is narrow and the routing probe is part
of the API rather than an afterthought.

Layout — interface here, kernels under `kernel/`:

- `__init__.py` — re-exports the two public names, and nothing else.
  Importing it pulls in `torch` and no more: the dispatch module, the
  kernels, the JIT machinery and the optional CuTe-DSL dependency are all
  imported lazily, so `flashinfer/__init__.py` re-exports the op
  unconditionally and a consumer's capability check
  (`getattr(flashinfer, "gdn_fused_decode_step", None)`) costs nothing and
  never compiles.
- `gdn_fused_decode.py` — the public API and the composable torch
  implementation (the executable specification of the op, works on any CUDA
  arch).
- `gdn_fused_decode_specialized.py` — registry loading, signature matching,
  and the single specialized dispatch entry point.
- `gdn_fused_decode_registry.json` — the workload registry (schema below).
- `kernel/gdn_fused_decode_<impl>.py`, `kernel/gdn_fused_decode_sm120.cu` —
  kernel implementation modules and their in-package JIT source.
- `kernel/_stream_order.py` — the cross-stream ordering every impl needs for
  the per-device state it shares between calls.

## No backend option

`gdn_fused_decode_step` takes **no `backend` argument**.  This is one fused
operation, not a family of interchangeable backends: which implementation
runs — one of the specialized SM120 kernels or the composable torch path —
is decided by the library from the workload registry and the device.
Callers do not have the information to make that choice per call, and an
override would be a second, unmeasured configuration surface.

What callers do get is the decision, in advance and for free:
`gdn_fused_decode_step_supported(...)` is a host-side, capture-safe probe
that answers "would this call hit a specialized kernel?" without running
anything, so a framework can keep its own optimized composition for shapes
this op does not accelerate.  (The composable path inside the op is a
correctness path, not a fast one.)

Internally the registry's `impl` values are grouped into preference
families — `cute_dsl` before `cuda`, then the composable path — but those
names are an implementation detail; they are not accepted or returned by
any public function.

## No environment gate: support is ours, policy is the framework's

**There is no `FLASHINFER_*` variable that turns this op on or off**, and
adding one would be a mistake.  A kill switch is what you ship when an
integration *replaces* an existing FlashInfer implementation: unsetting it
has something to fall back to, and the variable is the operator's escape
hatch. `gdn_fused_decode_step` is a **new** API — before this package,
FlashInfer had no fused Qwen GDN decode step at all — so there is nothing
inside FlashInfer for such a variable to fall back to. Its only effect
would be to add a second policy surface that nobody measures.

The split this package does implement:

| question | who answers it | how |
| --- | --- | --- |
| *Can* this call be served fast? | FlashInfer | the registry + the device, reported by `gdn_fused_decode_step_supported(...)` |
| *Should* this operation be used at all? | the calling framework | the framework's own configuration (vLLM: `additional_config`/env), which decides whether to call this API |

So a framework that wants the stock computation does not ask this library
to pretend the op is unavailable — it does not call it. That keeps the
A/B honest too: the arm that is "off" runs the framework's own chain,
which is the thing the fused op is supposed to replace.

| call | behavior |
| --- | --- |
| `gdn_fused_decode_step(...)`, signature registered, impl importable | specialized kernel (preference order `cute_dsl`, then `cuda`) |
| `gdn_fused_decode_step(...)`, signature not registered, or non-SM120 device | composable path |
| `gdn_fused_decode_step_supported(...)`, signature registered + impl importable | `True` |
| `gdn_fused_decode_step_supported(...)`, anything else | `False` |

A specialized-kernel failure can never break the op: it warns once, latches
that impl off for the rest of the process (and drops the probe memo, so the
probe stops advertising it), and the call is served by the composable path.
There is no path on which choosing an implementation raises.

### Which impl served: attestation

Because that fallback is invisible to the caller, "the op returned a result"
does not identify the kernel that produced it — a benchmark or an accuracy
run can keep passing every gate while a *different* implementation serves the
workload. Dispatch therefore attests:

- each impl logs one line the first time it serves,
  `Fused GDN decode step is being served by specialized impl '<impl>'.`;
- `gdn_fused_decode_stats()` reports `served_impls` next to `failed_impls`;
- the latch warning states that a different implementation now serves, so
  any measurement taken after it describes that one.

A harness that pins a specific impl — a kernel A/B, an e2e arm, an accuracy
probe — should assert on those (`failed_impls == []` and the expected name in
`served_impls`), not on the call succeeding.

### Probe cost

`gdn_fused_decode_step_supported(...)` is on the framework's per-layer,
per-decode-step path, outside the CUDA graph, and it answers the same
question every time. It is therefore memoized on `(compute capability,
geometry)`: the first call resolves the device capability, indexes the
registry and imports the winning impl module; every later call is a dict
lookup. The memo is dropped whenever the registry object is substituted
(tests and benchmark harnesses do this) or an impl is latched off, so it
cannot advertise a surface that has gone away. Measured on a
48-layer decode step, this is the difference between ~0.12 ms and ~0.01 ms
of host time per step — small, but one-directional and paid at exactly the
shapes the registry declines.

CUDA-graph contract: each impl compiles **lazily, per variant** — the first
eager (non-capturing) dispatch of a (batch size, scale, conv-state layout)
variant compiles and warms it (vLLM's profile run precedes its capture
phase and plays this role).  During capture an impl is recorded only when
`ready_for_graph_capture` confirms this exact variant is already warm —
never compiling, synchronizing, or allocating persistent state under
capture — falling through to the next impl and finally to the
(capture-safe) composable path, which then gets baked for that shape.  This
differs from `flashinfer/gemm/specialized`'s eager
`ensure_precompiled(rows)` pass on purpose: the query `scale` is part of
the CuTe-DSL compile key but is a runtime value, so the full variant set
cannot be precompiled from the registry alone.

## Registry: `gdn_fused_decode_registry.json`

The registry maps complete workload signatures to kernel implementations.
A signature may appear once per capable impl — this op ships **two** — and
dispatch selects among the impls registered for the signature in the
preference order above.

```json
{
  "op": "gdn_fused_decode_step",
  "schema_version": 1,
  "workloads": [
    {
      "impl": "cutedsl_sm120_pdl",
      "cc": 120,
      "b": 1, "hidden": 5120, "n_ba": 96, "qkv_dim": 10240,
      "h_q": 16, "hv": 48, "d": 128,
      "conv_width": 4, "conv_state_len": 3,
      "conv_layout": "SD"
    }
  ]
}
```

Field semantics (schema_version 1):

- `impl` — selects the kernel module `kernel/gdn_fused_decode_<impl>.py`.
  Shipped impls and the internal preference family each belongs
  to (neither name is public API):
  - `cutedsl_sm120_pdl` (family `cute_dsl`, preferred) — two-launch
    CuTe-DSL kernel with PDL overlap; compiled per (batch, scale, conv
    layout).
  - `cuda_sm120_persistent` (family `cuda`) — single-launch persistent
    CUDA kernel behind one B-dynamic JIT module
    (`kernel/gdn_fused_decode_sm120.cu`).
- `cc` — compute capability as `major * 10 + minor` (SM120 → `120`).
- `b` — exact decode batch size (`hidden_states` is `(b, hidden)`).
- `hidden`, `n_ba`, `qkv_dim`, `h_q`, `hv`, `d`, `conv_width`,
  `conv_state_len` — exact layer geometry (`w_ba` is `(hidden, n_ba)`,
  `mixed_qkv` is `(b, qkv_dim)` with `qkv_dim = (2*h_q + hv) * d`, the conv
  pool view is `(P, qkv_dim, conv_state_len)`, `ssm_state` is
  `(P, hv, d, d)`).
- `conv_layout` — the physical conv-state pool layout consumed as the
  logical `[P, qkv_dim, state_len]` view: `"SD"` = `(state_len, dim)` rows
  passed as their transposed view (vLLM's default allocation), `"DS"` =
  dense `(dim, state_len)` rows.  The kernels take the conv-state strides
  as runtime parameters and are hardware-validated on both layouts, but a
  layout dispatches only where a row lists it — no unbenched surface ships.
- Dtypes and stride patterns are **not** per-row: the op contract is fixed
  (bf16 activations/weights/conv pool, fp32 `A_log`/`ssm_state`, int32
  `state_indices`; dense inner layouts with row-strided `mixed_qkv` views
  and padded conv/ssm pool page strides allowed; `use_qk_l2norm=True`) and
  is checked by the dispatch guard
  (`signature_from_tensors`).  Kernels needing different dtypes or scale
  semantics need a `schema_version` bump.

Matching is exact equality of every signature field plus the device's `cc`.
Rows must describe the *complete* dispatch surface: anything not matched is
served by the composable path (and reported unsupported by the probe).

Keep the registry trimmed to the measured-win surface, and let the
**end-to-end** measurement decide it — a row is a promise about serving
behaviour, not about the kernel in isolation.  The shipped surface is
decode batches `1/2/4/8` in the `SD` layout on SM120, per impl, for exactly
one GDN layer geometry: `hidden=5120`, `n_ba=96`, `qkv_dim=10240`, 16 qk /
48 v heads, `d=128`, `conv_width=4`, `conv_state_len=3`.

**Which checkpoints those rows are valid for.** The geometry was captured
from `nvidia/Qwen3.6-27B-NVFP4`, and the end-to-end serving sweep that
chose the batch window ran on that checkpoint.  Matching is on the numbers
and the device `cc` alone — no row names a model, and nothing in the
dispatch path reads a model name — so *any* checkpoint whose GDN layer has
exactly these sizes dispatches here, and a checkpoint from a neighbouring
release (Qwen3.5-class GDN layers, other 27B variants) does so if and only
if its layer sizes are identical, which is a property of that checkpoint
rather than of the family name.  Anything else falls through to the
composable path.  Batches 16/24/32 are deliberately absent even
though the kernel is faster than the stock chain there in a kernel A/B
(1.49x at 16, 1.16x at 32 under CUDA-graph replay): the serving sweep that
covered them did not reproduce a win at the engine level, so those batches
keep the stock path and the registry claims only what was measured.  Adding
them back is a registry-only change once an end-to-end win is measured; no
dispatch, kernel or consumer code has to move.

## Adding a new specialized fused-GDN-decode kernel

1. Add `kernel/gdn_fused_decode_<impl>.py`.  The module may import heavy or
   optional dependencies (e.g. the CuTe DSL) at module import time — it is
   only imported when a registry row names it — and must raise
   `ImportError`/`RuntimeError` at import when they are missing.
2. Implement the impl-module interface (reference implementations:
   `kernel/gdn_fused_decode_cutedsl_sm120_pdl.py`,
   `kernel/gdn_fused_decode_cuda_sm120_persistent.py`).  `rows` below is the list
   of registry rows whose `impl` is this module:
   - `execute(hidden_states, w_ba, mixed_qkv, conv_weight, conv_bias,
     conv_state, A_log, dt_bias, scale, ssm_state, state_indices, out=None)
     -> (output, conv_state, ssm_state)` — run the fused step on the
     caller's current stream, updating both pools in place; compile lazily
     on eager calls; raise on failure.
   - `ready_for_graph_capture(hidden_states, conv_state, scale) -> bool` —
     True only when this exact call can be recorded into a CUDA graph
     without compiling, synchronizing, or allocating persistent state.
   - `variant_plan(rows) -> set` — distinct compiled-kernel descriptors the
     rows require (host-side planning only).
   - `launch_count() -> int`, `compiled_variant_keys() -> list[str]` —
     introspection for benchmarks and tests (a CUDA-graph capture counts
     once; replays do not re-run host code).

   `execute` is called with the tensors' device already current (dispatch
   enters `torch.cuda.device(hidden_states.device)`), because a call is not
   guaranteed to arrive on the ambient device — TP > 1 serving drives rank
   *r*'s layers on `cuda:r`.  Take the launch stream from that device by
   name anyway: `torch.cuda.current_stream(device)`, not
   `torch.cuda.current_stream()`.  A test parses the impl modules and fails
   on the bare form, since a single-GPU box cannot tell the two apart.

   Both state pools are updated **in place**, so the same buffer is read and
   written within one call.  A kernel that spells that as separate input and
   output pointers (the CUDA impl does, to keep the phases readable) must not
   mark those pointers `__restrict__`: they alias by construction, and the
   promise would let the compiler hoist a pool load across a pool store.

   Any per-device state an impl keeps across calls (scratch buffers, a
   persistent grid barrier) is shared by every call on that device, so
   `execute` must order a call that arrives on a different stream after the
   previous one — call `kernel/_stream_order.py`'s
   `order_after_previous_stream(<your per-device stream dict>, device)`, as
   both shipped impls do.  It records an event on the previous stream and
   costs a stream compare in the steady state.  Two *replays* of captured
   graphs running concurrently on different streams remain a caller-side
   serialization requirement, exactly as the in-place conv/ssm pools already
   are.
3. Match the composable path's numerics, which is what the correctness tests
   assert: round the fp32 `b`/`a` GEMV sums through bf16 before the gates
   (the reference materializes `ba` as a bf16 tensor, so skipping this makes
   a kernel *more* precise than the op it implements), and use an
   overflow-free `softplus` — `log(1 + exp(x))` returns `+inf` for `x` above
   ~88.7 in fp32 and silently zeroes the decay gate.

   A CuTe-DSL impl must also stay inside the DSL surface its *deployed*
   version offers, not the newest one installed while developing: the
   `cute.math` module was hand-written up to nvidia-cutlass-dsl 4.5 and
   became a re-export of the much larger `cutlass._mlir_helpers.math` in
   4.6, so e.g. `cute.math.max` resolves on 4.6+ and raises `AttributeError`
   inside `cute.compile` on 4.5.  That error is caught by the dispatch
   layer, which latches the impl off and serves the next one — a silent
   substitution, not a crash.  `PORTABLE_CUTE_MATH_PRIMITIVES` in
   `tests/gdn/test_fused_decode.py` pins the supported surface and is
   checked without a GPU or cutlass; raise it deliberately if the floor
   moves.
4. Map the impl to an internal preference family in `BACKEND_IMPLS` (an
   existing family to extend its impl set, or a new one, placed in
   `_AUTO_BACKEND_ORDER`) and add the workload rows to
   `gdn_fused_decode_registry.json`.  No public surface changes.
5. Add tests in `tests/gdn/test_fused_decode.py` (correctness vs the
   composable reference on random *and* gate-saturating inputs,
   registry-driven dispatch and rejection, internal preference order,
   off-registry fallback, conv layouts, CUDA-graph capture) and extend the
   benchmark (`benchmarks/bench_gdn_fused_decode.py`).

Beyond `BACKEND_IMPLS`, no dispatch-code changes are needed:
`gdn_fused_decode.py` only forwards to the dispatch module, and
`gdn_fused_decode_specialized.py` resolves `impl` to a module.
