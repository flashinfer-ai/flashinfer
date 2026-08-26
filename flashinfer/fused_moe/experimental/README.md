# Experimental fused MoE routing

Three entry points covering the non-GEMM glue a serving engine runs around the
routed-expert GEMMs of one MoE block, plus the specialized SM120 kernels that
serve the allowlisted decode shapes:

| entry point | runs | replaces |
| --- | --- | --- |
| `moe_routing_prologue` | before the expert GEMMs | router GEMV + softmax top-k + block-aligned descriptor build + shared-expert gate |
| `moe_routing_align` | before the expert GEMMs | the descriptor build alone (`moe_align_block_size` and its counting/sorting chain), for an engine that runs its own router |
| `moe_routing_finalize` | after the routed-expert w2 GEMM | the top-k weighted reduce (`moe_sum` / `torch.sum`), optionally plus the gated shared expert |

They are **exported at the top level**, like the other fused-MoE APIs
(`flashinfer.cutlass_fused_moe`, `flashinfer.trtllm_fp4_block_scale_moe`).
*Experimental* describes where this code lives, not how it is called:

```python
import flashinfer

if flashinfer.moe_routing_supported(m, hidden_size, num_experts, top_k):
    sorted_token_ids, expert_ids, num_tokens_post_pad = flashinfer.moe_routing_align(
        topk_ids, num_experts
    )
    ...  # w13 GEMM -> activation -> w2 GEMM
    out = flashinfer.moe_routing_finalize(expert_out, None, topk_weights, None)
else:
    ...  # keep your own composition
```

Layout — interface here, kernel under `kernel/`:

- `__init__.py` — re-exports the public names, and nothing else. Importing it
  pulls in `torch` and no more: the dispatch allowlist, the JIT toolchain and
  the kernel are all reached lazily, so `flashinfer/__init__.py` re-exports the
  op unconditionally and a consumer's capability check
  (`getattr(flashinfer, "moe_routing_finalize", None)`) costs nothing and never
  compiles.
- `moe_routing.py` — the three public APIs, the exact-match dispatch guards,
  and the portable torch composition that is the executable specification of
  each entry point (works on any arch, and is what an unallowlisted shape
  runs).
- `moe_routing_sm120_workloads.json` — the dispatch allowlist (below).
- `kernel/moe_routing_sm120.cu` — one translation unit holding all three
  kernels and all three tvm_ffi entry points, compiled on demand by
  `flashinfer/jit/moe_routing.py`. It lives here rather than in `csrc/` because
  it is only ever built for this one op; it is registered as package data in
  `pyproject.toml`, like the experimental fused-GDN-decode source.

Tests: `tests/moe/test_moe_routing.py`.

## The finalize owns the routing weights

`expert_out` is the routed-expert down-projection output with `topk_weights`
**not** yet applied. A caller whose expert GEMM folds the routing weights into
its own epilogue must turn that off (vLLM's Marlin MoE:
`mul_topk_weights=False`), or they are applied twice — which presents as an
accuracy regression, not as a crash. Derive both the fold and the choice of
reduce from **one** boolean so they cannot disagree.

`shared_out`/`shared_gate` are all-or-nothing and may both be `None`, for an
engine that combines the shared expert one level up (vLLM does); the result is
then exactly the routed weighted sum. That is advertised as the capability flag
`finalize_optional_shared_expert` in `moe_routing_stats()`, because an optional
*value* is invisible to `inspect.signature` and sniffing a version would be
wrong.

## Dispatch: `moe_routing_sm120_workloads.json`

```json
{
  "fields": ["m", "hidden_size", "num_experts", "top_k"],
  "workloads": [[1, 2048, 256, 8], [2, 2048, 256, 8], [4, 2048, 256, 8]]
}
```

Matching is exact equality on every field the entry point can *observe* plus
compute capability `(12, 0)`: the prologue sees all four, `moe_routing_align`
takes `topk_ids` and so never sees `hidden_size`, `moe_routing_finalize` takes
`expert_out` and so never sees `num_experts`. Each accepts a size that some
allowlisted row covers on the axes it can check. Anything else — a different
shape, a non-bfloat16 activation, a non-contiguous operand, another device —
takes the composable path.

This is the **measured win surface, not the support surface**: the prologue
serves any token count up to 32 and the finalize any token count at all, but
only these decode sizes have a measured graph-replay win, so larger capture
sizes deliberately keep the composable path until they are measured. Widening
the file is a measurement decision, not a code change, and a test asserts the
shipped bound so re-widening has to be deliberate.

`FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1` is read at *call* time and restores
the composable path for all three entry points. Unlike a brand-new op, this one
*replaces* computation an engine already had, so the kill switch has something
to fall back to and is the operator's escape hatch.

### Which path served: attestation

A guard that declines is invisible to the caller — the op returns the right
answer either way — so "the benchmark passed" does not tell you which
implementation produced it. Dispatch therefore attests:

- each entry point logs one line the first time it dispatches,
  `flashinfer: specialized SM120 MoE routing <which> kernel dispatched (...)`;
- `moe_routing_stats()` reports a per-entry-point launch counter
  (`prologue_launch_count`, `align_launch_count`, `finalize_launch_count`).
  These count **host-side dispatches**: a CUDA-graph capture counts once and
  replays do not count, so they prove which implementation was *recorded* into
  a graph rather than how often the graph ran;
- every fallback logs `str(exc)`, not just the exception class. A build failure
  is otherwise indistinguishable from a shape that was never allowlisted.

A harness that pins the specialized path — a kernel A/B, an e2e arm, an
accuracy probe — should assert on those counters, not on the call succeeding.

## CUDA-graph contract

A dispatch made **under capture** never compiles, queries a device or
synchronizes: it dispatches only when the module is already compiled and
loaded, and otherwise falls through to the composable path, which is
capture-safe and gets baked into the graph for that shape.

The module is warmed by the first *non-capturing* call, or explicitly by
`moe_routing_precompile()`. A serving engine's eager profile run precedes its
capture phase and plays that role. One translation unit holds all three entry
points and every allowlisted size, so there is exactly one compiled variant and
readiness is a single check (`moe_routing_ready_for_graph_capture()`).

The kernels hold **no persistent device state** (`moe_routing_stats()` reports
`persistent_device_state_bytes: 0`) and contain no inter-CTA rendezvous, so a
launch's result cannot depend on any earlier launch. A test asserts that
directly, by interleaving problem sizes and demanding bit-identical results
against the same calls made in isolation.

## Not in the AOT build

There is no entry in `flashinfer/aot.py`, so this op does not ship in
`flashinfer-jit-cache`. The experimental fused GDN decode step has no entry
either, but for a *different* reason, and copying its rationale here would be
wrong: that op's preferred impl is CuTe-DSL, which the AOT pass does not cover
at all, so its entry could only ever pre-build a second-choice kernel that
never runs on an install where the CuTe-DSL one loads. Here
`moe_routing_sm120.cu` is the **only** compiled implementation — the fallback
is a Python-level torch composition, not a second backend — so the entry did
pre-build the kernel that actually runs.

Dropping it is therefore a real trade, not a cleanup: one fewer `sm120a`
translation unit in every jit-cache wheel, out of a shared and size-limited
budget, for an op whose allowlist is three decode shapes on one architecture —
paid for by a compile on the first eager call.

**This does not change when compilation happens relative to CUDA-graph
capture.** An AOT artifact still has to be *loaded* host-side before a dispatch
can be recorded, and loading is no more legal under capture than compiling is;
`moe_routing_ready_for_graph_capture()` is `False` until something warms the
module either way. So with or without AOT, a cold capture falls back to the
composable path and a warm one records the kernels. Dropping the entry removes
a build cost and lengthens the first eager call, and nothing else.

The one deployment it changes is `FLASHINFER_DISABLE_JIT=1`. That mode refuses
to compile but still *loads* a module the cache or a `flashinfer-jit-cache`
wheel already carries — and with no AOT entry, no wheel carries this one. So
unless the on-disk JIT cache was populated by an earlier unrestricted run, the
module cannot be obtained at all: `moe_routing_precompile()` returns `False`
after logging the loader's own message, and all three entry points serve the
composable path. The failure is **latched**: the JIT is now the op's only build
path, and a build that fails once in a process fails for a reason that does not
resolve itself, so the attempt is made once and later dispatches take the
fallback directly rather than re-entering the file lock and `ninja` on every
call. That is **correct**, just not accelerated — and
`moe_routing_supported()` keeps answering `True` there, because it reports what
this build's *allowlist and device* support, not whether a toolchain happens to
be available. A consumer that needs the stronger statement should call
`moe_routing_precompile()` once at startup and check the result, which is what
warming the module ahead of capture requires anyway.

## Adding a shape or a kernel

1. Measure it **end to end**. An allowlist row is a promise about serving
   behaviour, not about the kernel in isolation; per-call microbenchmarks of
   these entry points are dominated by a fixed launch-and-harness cost and do
   not settle a graph-replay win.
2. Add the row to `moe_routing_sm120_workloads.json` and raise `SHIPPED_MAX_M`
   in `tests/moe/test_moe_routing.py` — deliberately, in both places.
3. Kernel changes go in `kernel/moe_routing_sm120.cu`, keeping all three entry
   points in the one translation unit so the capture-readiness check stays a
   single boolean. `moe_routing_align` and `moe_routing_prologue` share the
   *same* descriptor kernel (`logits != null` ⇒ score and describe,
   `in_tid != null` ⇒ describe only): two copies could drift, and a descriptor
   that disagrees with the one the expert GEMM was fed is a silent wrong
   answer. A test asserts the two entry points produce byte-identical
   descriptors at the shipped sizes.
4. Numerics are the contract, and the composable path in `moe_routing.py` is
   its executable statement: softmax scoring over all experts (only the shared
   gate is a sigmoid), router logits rounded to bfloat16 *between* the GEMV and
   the scoring, selection by descending score with ties broken toward the lower
   expert id, fp32 renormalisation, and a finalize that accumulates in fp32 and
   rounds to bfloat16 exactly once.
