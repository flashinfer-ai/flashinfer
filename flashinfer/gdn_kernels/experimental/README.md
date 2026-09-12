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
eager (non-capturing) dispatch of a (layer geometry, batch size, scale,
conv-state layout) variant compiles and warms it (vLLM's profile run
precedes its capture phase and plays this role).  During capture an impl is
recorded only when
`ready_for_graph_capture` confirms this exact variant is already warm —
never compiling, synchronizing, or allocating persistent state under
capture — falling through to the next impl and finally to the
(capture-safe) composable path, which then gets baked for that shape.  This
differs from `flashinfer/gemm/specialized`'s eager
`ensure_precompiled(rows)` pass on purpose: the query `scale` is part of
the CuTe-DSL compile key but is a runtime value, so the full variant set
cannot be precompiled from the registry alone.

### Not in the AOT build

Neither impl is registered in `flashinfer/aot.py`, so neither ships in
`flashinfer-jit-cache`. That is a deliberate trade against a shared,
size-limited cache:

- The **CuTe-DSL** impl — the one that actually serves every registered
  geometry — is outside the AOT pass entirely. `aot.py` drives `gen_*_module()`
  nvcc specs; CuTe-DSL kernels cache through `JitSpecCuteDsl` / `cached_ops/`
  instead. There was never an AOT entry to keep for it.
- The **CUDA** impl is second in `_AUTO_BACKEND_ORDER`, so on any install where
  the CuTe-DSL impl loads it is dispatched only after that one is latched off
  by a failure. Its AOT entry cost one sm120a translation unit in every
  jit-cache wheel to pre-build a kernel that, in the normal case, never runs.

Both therefore JIT-compile on first eager dispatch — which is already the path
the CUDA-graph contract above depends on, so this removes a build cost rather
than changing when compilation happens.

The one deployment this changes is `FLASHINFER_DISABLE_JIT=1` with no JIT cache
present: neither impl can build, both latch off after their first attempt, and
the op serves the composable path. It stays **correct**, and the latch clears
the probe memo, so `gdn_fused_decode_step_supported(...)` starts answering
`False` and a framework goes back to its own composition after at most one
declined call per impl. If a JIT-disabled deployment needs the CUDA kernel,
re-adding the spec under `has_sm120` restores it — one per registered layer
geometry, since the geometry is compiled in:
`gen_gdn_fused_decode_module(*geometry) for geometry in registry_geometries()`.
Nothing else depends on its absence.

### `@flashinfer_api` on both; `trace=` only on the op

Both public entry points carry `@flashinfer_api`. The op adds
`trace=gdn_fused_decode_trace` — template in `flashinfer/trace/templates/gdn.py`,
row in `docs/fi_trace.rst`, example definition in `tests/trace/fi_trace_out/`.
The probe takes the bare decorator, and is **not** cached at that layer. Both
halves are deliberate:

- **Nothing to trace.** `fi_trace` records the shapes of a *kernel* call:
  `docs/fi_trace.rst` presents `trace=` as the thing you attach when adding a
  new kernel, and its output is the input format for flashinfer-bench. The
  probe launches no kernel — it is host-side and capture-safe by contract — so
  a template here would emit a benchmark definition for a function that does
  no device work. Precedent for a bare decorator on a capability query:
  `has_monomoe` (`flashinfer/fused_moe/monomoe.py`).
- **No cache at the API layer**, which is why this is not the `has_monomoe`
  shape. `has_monomoe()` is nullary, so `@functools.cache` above the decorator
  is trivially safe and its logging fires once per process. This probe takes a
  `device`, and `None` / an index-less `"cuda"` mean *whatever device is
  current now*. A cache on the public signature would pin the answer under the
  key `None`; a later `torch.cuda.set_device()` onto a device of a different
  compute capability would keep reading it — routing a call into a kernel
  built for another architecture, or declining one it could serve. So the memo
  sits one layer in, keyed on the **resolved** capability
  (`gdn_fused_decode_supported_geometry`), and the resolution runs before the
  memo is consulted.
- **Cost of the decorator.** None at the default log level: `flashinfer_api`
  reads `FLASHINFER_LOGLEVEL` once at import and returns the original function
  unchanged when it is `0` — no wrapper frame, no per-call branch. At
  `FLASHINFER_LOGLEVEL>=1` it emits one line per GDN layer per decode step (48
  for this model). That is intended rather than tolerated: anyone at that
  level is debugging dispatch, and which calls the probe declined is the
  question they are asking. Note that `FLASHINFER_DUMP_EXCLUDE` does not
  suppress those lines — it gates the heavy stat/dump path at level 3+ only.

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
    CuTe-DSL kernel with PDL overlap; compiled per (layer geometry, batch,
    scale, conv layout).
  - `cuda_sm120_persistent` (family `cuda`) — single-launch persistent
    CUDA kernel behind one B-dynamic JIT module per layer geometry
    (`kernel/gdn_fused_decode_sm120.cu`).
- `cc` — compute capability as `major * 10 + minor` (SM120 → `120`).
- `b` — exact decode batch size (`hidden_states` is `(b, hidden)`).
- `hidden`, `n_ba`, `qkv_dim`, `h_q`, `hv`, `d`, `conv_width`,
  `conv_state_len` — exact layer geometry, and a **compile-time parameter**
  of both impls: a new model geometry is new registry rows plus a recompile,
  not new kernel code (`w_ba` is `(hidden, n_ba)`,
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
- `state_indices` **values** are not per-row either, and are not checked by
  the dispatch guard at all — they live on the device, and reading them
  host-side would cost a device-to-host sync per layer per decode step and
  is impossible under CUDA-graph capture.  A **negative** index (vLLM's
  `PAD_SLOT_ID = -1`) marks a padded batch row: every implementation must
  leave both pools untouched for it and write its output row as zero, the
  same contract `gated_delta_rule_decode_pretranspose`'s float32 path
  documents.  An index `>= P` is a caller bug, is deliberately neither
  clamped nor skipped, and is undefined.

Matching is exact equality of every signature field plus the device's `cc`.
Rows must describe the *complete* dispatch surface: anything not matched is
served by the composable path (and reported unsupported by the probe).

Keep the registry trimmed to the measured-win surface, and let the
**end-to-end** measurement decide it — a row is a promise about serving
behaviour, not about the kernel in isolation.  The shipped surface, all in
the `SD` layout on SM120:

| validation provenance | `hidden` | `n_ba` | `qkv_dim` | `h_q` | `hv` | `d` | CuTe-DSL batches | persistent CUDA batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B | 5120 | 96 | 10240 | 16 | 48 | 128 | 1/2/4/8 | 1/2/4/8 |
| Qwen3.6-35B-A3B | 2048 | 64 | 8192 | 16 | 32 | 128 | 1/2/4 | 1/2/4 |
| Qwen3.5-27B TP2 | 5120 | 48 | 5120 | 8 | 24 | 128 | — | 1/2/4/8/16 |
| Qwen3.5-35B-A3B TP2 | 2048 | 32 | 4096 | 8 | 16 | 128 | — | 1/2/4/8/16/24/32 |

All geometries have `conv_width=4` and `conv_state_len=3`.  The persistent CUDA
kernel requires `n_ba == 2*hv`, `hv % h_q == 0`,
`qkv_dim == (2*h_q + hv)*d`, and `d == 128`, enforced by named
`static_assert`s.  Geometries registered for CuTe-DSL must additionally satisfy
its K-split and convolution-tile divisibility checks.

**Which checkpoints those rows are valid for.** The first two geometries were
captured from `nvidia/Qwen3.6-27B-NVFP4` and
`nvidia/Qwen3.6-35B-A3B-NVFP4`.  The rank-local TP2 geometries were validated
on Qwen3.5-27B and Qwen3.5-35B-A3B.  Matching uses the complete numerical
signature and device `cc`; model and TP names are validation provenance, not
dispatch keys.  Anything else falls through to the composable path.

For Qwen3.6-27B, batches 16/24/32 remain absent even though the kernel is faster
than the stock chain there in a kernel A/B (1.49x at 16, 1.16x at 32 under
CUDA-graph replay), because the serving sweep did not reproduce a win.  For
Qwen3.5-27B TP2, batches 24/32 are absent because correctness passed but the
fixed full-model-graph serving screen did not show an end-to-end win.  The TP2
CuTe-DSL rows are absent because that implementation has not been validated for
either rank-local geometry.

## Adding a new layer geometry (a new model)

No kernel or dispatch code changes:

1. Add the rows to `gdn_fused_decode_registry.json` — one per (batch, impl)
   you have an end-to-end measurement for.
2. Add the geometry to `GEOMETRIES` in `tests/gdn/test_fused_decode.py`; the
   correctness, tiling and registry tests are parameterized over it, and
   `benchmarks/bench_gdn_fused_decode.py` picks the rows up from the
   registry with no edit at all.

A geometry that does not satisfy the tiling relations above cannot silently
mis-tile: the CUDA kernel fails to compile with a named `static_assert`, and
the CuTe-DSL impl raises at dispatch, which the dispatch layer turns into
the composable path.

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
     on eager calls; raise on failure.  **Padded rows**: for any batch row
     whose `state_indices` entry is negative, touch neither pool (no read
     and no write) and write that row's `output` as zero.  The guard belongs
     in the kernel — the host cannot look at index values without a sync,
     and the shape that carries padding is the CUDA-graph one.  Make the
     predicate uniform over whatever unit shares a batch row (block or warp)
     so it costs a branch rather than divergence.
   - `ready_for_graph_capture(signature, hidden_states, conv_state, scale)
     -> bool` — True only when this exact call can be recorded into a CUDA
     graph without compiling, synchronizing, or allocating persistent
     state.  `signature` is the matched dispatch signature, so readiness is
     checked against the exact compiled variant *including its layer
     geometry*: a process warm for one model must not make a
     differently-shaped model's call look capture-ready.
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

   **If an impl launches with PDL** (`use_pdl=True` /
   `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`), every kernel it
   launches that way must issue `griddepcontrol_wait()` on *every* path before
   its first read of anything a stream predecessor produced — the op's inputs
   are all predecessor-produced, including a workspace the host memsets just
   before the launch.  The attribute frees the driver from waiting on the
   predecessor's completion *and* its memory flush, so the wait is the only
   thing that orders those reads, and omitting it makes correctness depend on
   whether some unrelated upstream kernel happens to fire a trigger.  A kernel
   that also calls `griddepcontrol_launch_dependents()` must do so *after* its
   own wait: the trigger is a scheduling gate that places the dependent's CTAs
   on the SMs, and a dependent released by an unordered kernel inherits that
   disorder for any load it issues before its own wait.  The trigger is *not*
   what publishes data — a dependent's wait blocks until the whole prerequisite
   grid has completed and flushed — so where the trigger sits is a performance
   decision (earlier = more overlap, more SM competition) rather than a
   correctness one.  Two AST tests in `tests/gdn/test_fused_decode.py` pin both
   rules; the worked example is the contract block at the top of
   `kernel/gdn_fused_decode_cutedsl_sm120_pdl.py`.

   Any per-device state an impl keeps across calls (scratch buffers, a
   persistent grid barrier) is shared by every call on that device, so
   `execute` must order a call that arrives on a different stream after the
   previous one — call `kernel/_stream_order.py`'s
   `order_after_previous_stream(<your per-device stream dict>, device)`, as
   both shipped impls do.  It records an event on the previous stream and
   costs a stream compare in the steady state.  What it covers is *one host
   thread* switching streams, which is the reachable case.  It does **not**
   make the shared state safe against genuinely concurrent callers, and two
   cases stay a caller-side serialization requirement, exactly as the
   in-place conv/ssm pools already are:

   - two host threads calling the op for the same device at the same time —
     the "record an event on the previous stream" step and the launch it
     protects are not one atomic action, so a second thread can slip between
     them;
   - two *replays* of captured graphs running concurrently on different
     streams.

   Keying the state by stream instead would not fix either, and would cost
   the CUDA-graph contract: `torch.cuda.graph` captures on a fresh side
   stream, so a stream-keyed cache is always cold at capture time and
   `ready_for_graph_capture` would decline.
3. Match the composable path's numerics, which is what the correctness tests
   assert: round the fp32 `b`/`a` GEMV sums through bf16 before the gates
   (the reference materializes `ba` as a bf16 tensor, so skipping this makes
   a kernel *more* precise than the op it implements), and use an
   overflow-free `softplus` — `log(1 + exp(x))` returns `+inf` for `x` above
   ~88.7 in fp32 and silently zeroes the decay gate.

   A CuTe-DSL impl must also stay inside the DSL surface its *deployed*
   version offers, not the newest one installed while developing.  **The
   runtime floor for this package is nvidia-cutlass-dsl 4.5**, which is
   deliberately lower than the `==4.7.0` FlashInfer itself pins in
   `requirements.txt` and the `cu12`/`cu13` extras.  The two answer different
   questions: the pin says which DSL a FlashInfer install brings along, the
   floor says which DSL the shipped kernel must still compile under — and
   they differ because this op is consumed from serving stacks that resolve
   the DSL themselves (the vLLM nightly image these kernels were validated
   in downgrades to 4.5.2, vLLM's own pin, *after* FlashInfer is installed;
   the pt2605 container ships 4.5.0).  Concretely: the `cute.math` module was
   hand-written up to 4.5 and became a re-export of the much larger
   `cutlass._mlir_helpers.math` in 4.6, so e.g. `cute.math.max` resolves on
   4.6+ and raises `AttributeError` inside `cute.compile` on 4.5.  That error
   is caught by the dispatch layer, which latches the impl off and serves the
   next one — a silent substitution, not a crash, which is why the floor is
   enforced by a test rather than left to review.  The authoritative
   statement is `CUTE_DSL_RUNTIME_FLOOR` plus
   `PORTABLE_CUTE_MATH_PRIMITIVES` in `tests/gdn/test_fused_decode.py`,
   checked without a GPU or cutlass (a third test asserts the floor stays at
   or below the repo pin).  Raise it deliberately, in both places and here.
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
