# GC MoE (experimental)

**Status: experimental API and experimental backend.** `flashinfer.gc_moe`,
`flashinfer.GCMoEWorkspace` and `flashinfer.transform_gc_moe_weights` may
change or be removed without deprecation. Calling them is the opt-in (each
emits the standard `ExperimentalWarning` once); no environment variable gates
the feature, it never participates in `backend="auto"` routing (it is a
standalone API family, not a backend of a stable API), and nothing is
registered for ahead-of-time compilation: it is JIT-only and never ships in
`flashinfer-jit-cache` / `flashinfer-cubin`. The drop's own `GC_MOE_*` tuning
switches are separate (see Tuning switches).

The kernels are vendored from [Bruce-Lee-LY/gc-moe](https://github.com/Bruce-Lee-LY/gc-moe)
0.6.2; `flashinfer/experimental/gc_moe/kernel_src/gc_moe/VENDOR.md` records the
pinned commit, and `flashinfer.experimental.gc_moe.UPSTREAM_VERSION` /
`UPSTREAM_COMMIT` expose the same values at runtime.

## What it computes

One fused MoE step for serving: dispatch of the staged tokens to their routed
experts, MXFP4 routed GEMMs over FP8 activations, SwiGLU, the per-expert
combine across the expert-parallel group, and the overlapping of all of it
across Green Contexts (CUDA SM partitions), in three per-architecture kernels:

```
x [T, H] fp8 + x_sf                      (staged by the caller)
  -> dispatch to the routed expert peers (topk_idx / topk_weights)
  -> FC1 (MXFP4 x FP8), SwiGLU, requantize
  -> FC2 (MXFP4 x FP8)
  -> combine (routing weights) across ranks
  -> y [T, H] bf16                       (written in place)
```

| backend | GPUs | entry in the drop |
|---|---|---|
| `sm90` | Hopper SM90 (H100/H200) | `sm90_fp8_fp4_gc_moe` |
| `sm100` | Blackwell SM100/SM103 (B200/B300) | `fp8_fp4_gc_moe` |
| `sm120` | Blackwell consumer SM120 (RTX 50 / RTX Pro) | `sm120_fp8_fp4_gc_moe` |

The backend is selected from the device compute capability; `backend=` (on
the workspace, the weight transform and the call) overrides it and must match
the device.

## Requirements

- CUDA GPU of one of the three architectures above, `nvidia-cutlass-dsl >=
  4.8.1` (pure-Python CuTe DSL; kernels are JIT-compiled on first use).
- A PTX assembler from CUDA >= 13.2: resolved from the `ptxas=` argument,
  then `$GC_MOE_PTXAS`, then `$CUDA_HOME`/`$CUDA_PATH`, then `PATH`, then
  `/usr/local/cuda/bin/ptxas`. A named assembler that is missing or too old
  is reported, never silently replaced.
- Multi-rank calls use `torch.distributed._symmetric_memory` (a private
  PyTorch API) to allocate and rendezvous the workspace; a single-rank group
  allocates with plain `torch.empty`. The module must be importable either
  way, because the vendored trees import it at load time; the wrapper checks
  and reports a build that lacks it.
- `num_experts` must divide evenly across the group.

## API

```python
import flashinfer

with flashinfer.GCMoEWorkspace(group, num_experts, num_max_tokens_per_rank,
                               num_topk, hidden, intermediate_hidden) as ws:
    l1, l2 = flashinfer.transform_gc_moe_weights((w1_payload, w1_sf),
                                                 (w2_payload, w2_sf))
    ws.x[:tokens].copy_(x_fp8)               # staged by the caller
    ws.x_sf[:tokens].copy_(x_sf)
    ws.topk_idx[:tokens].copy_(topk_idx)
    ws.topk_weights[:tokens].copy_(topk_weights)
    y = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    flashinfer.gc_moe(y, l1, l2, ws)
```

`GCMoEWorkspace` owns one staging allocation (symmetric across the group when
it has more than one rank) holding the staging views (`x`, `x_sf`,
`topk_idx`, `topk_weights`, the shared-expert views, the routing workspace),
exposes them by delegation, and must be destroyed with `destroy()` or by
leaving a `with` block. Keep the returned weight tensors alive and
address-stable for as long as any launch or captured graph references them.

`transform_gc_moe_weights` accepts the checkpoint-format `(payload, scales)`
pairs; the accepted layout follows the architecture:

| backend | routed payload | routed scales | extra |
|---|---|---|---|
| `sm90` | int8 `[E, N, K/2]` | FP32 `[E, N, K/32]` (powers of two) or UE8M0 bytes | optional FP32 `[E]` secondary scales (`l1_scale_2` / `l2_scale_2`) |
| `sm100` | int8 `[E, N, K/2]` | int32 `[E, N, K/128]`, MN-major (`stride(-2) == 1`) | — |
| `sm120` | int8 `[E, N, K/2]` | int32 `[E, N, K/128]` | FP8 shared weights take `shared_recipe` |

MXFP4 nibbles are packed even-K-in-low-nibble on every backend. Call the
transform outside graph capture.

### Shared experts

Create the workspace with `num_shared_experts=S` and pass the dense shared
pair (`shared_l1_weights`, `shared_l2_weights`) to `gc_moe`; the pair is
all-or-nothing and must agree with the workspace. Shared weights are FP8.
SM90 and SM120 take a `shared_recipe` for them (K32 or K128); SM100 has no
`shared_recipe` parameter and instead expects the K128 packed scales already
expanded to one packed row per N.

The shared activations are the staged `x` rows (`shared_l1_acts` is an alias
of `x`), but their scales have their own staging view. Write the input scale
words to `shared_l1_acts_sf` — int32, MN-major (`stride[0] == 1`), four
packed scale bytes per 128 elements — at row

```
(t // BM) * align(BM, 128) + (u // 128) * 128 + (u % 32) * 4 + (u % 128) // 32,
u = t % BM
```

where `BM = flashinfer.experimental.gc_moe.block_m(num_ranks, num_experts,
num_max_tokens_per_rank, tokens_this_rank, num_topk)` for the rank's actual
token count. Shared activations always use K32 scales; `x_sf` stays K-major
while the shared scale views are MN-major.

### Graph capture

The call launches on the current stream and nests its internal role graph
into an enclosing `torch.cuda.graph`, so replay runs the whole kernel
sequence without re-entering the host path. Warm the exact call once eagerly,
then capture; every captured graph must be retired before `destroy()`.
Re-staging the workspace between replays is supported and is how a serving
stack feeds new tokens.

## Tuning switches

The vendored runtime is configured through process environment variables:
`GC_MOE_PTXAS` is read when the kernels are first imported, the rest when the
first launch creates its Green Context pair. Those pairs are cached
thread-locally, keyed by `(CUDA context, dispatch_sms)` on SM100 but by the
CUDA context alone on SM90/SM120, so **switches with a `GcMoePlan` field
cannot change for the lifetime of a workspace** — a later call that finds a
different environment raises instead of silently mismatching. Use
`flashinfer.experimental.gc_moe.GcMoePlan` to set them explicitly, keeping
the block open through the first `gc_moe` call:

```python
from flashinfer.experimental.gc_moe import GcMoePlan

plan = GcMoePlan(dispatch_sms=12)
with plan.applied("sm100"):
    workspace = flashinfer.GCMoEWorkspace(...)
    flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)  # first launch
```

| switch | field | default |
|---|---|---|
| `GC_MOE_DISPATCH_SMS` | `dispatch_sms` (SM100) | 8, or 16 with more than 8 ranks |
| `GC_MOE_SM90_DISPATCH_SMS` | `dispatch_sms` (SM90) | 4 |
| `GC_MOE_SM120_DISPATCH_SMS` | `dispatch_sms` (SM120) | 6 |
| `GC_MOE_BALANCED_WORKQUEUES` | — (inert; see below) | on (`1`) |
| `GC_MOE_SM90_BALANCED_WORKQUEUES` | — (inert; see below) | on (`1`) |
| `GC_MOE_PRIMARY_CTX` | — (pinned by the integration) | on (`1`) |
| `GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS` | `nvlink_barrier_timeout_secs` (SM100) | `3600` |
| `GC_MOE_PTXAS` | `ptxas` | discovered assembler |

The integration pins `GC_MOE_PRIMARY_CTX=1` before the vendored sources are
imported: every role kernel runs in the primary context and the SM split only
sizes grids. The balanced-workqueue switches build on the green contexts that
this mode never creates, so they are inert; none of the three is a
`GcMoePlan` field.

`GC_MOE_PTXAS` is read at import time by the vendored JIT; the integration
resolves and validates the assembler before that import and refuses to switch
it afterwards.

### Runtime lifetime

The Green Context runtime is a process-global singleton and registers an
`atexit` hook, so the usual path needs no explicit cleanup.
`flashinfer.experimental.gc_moe.close_runtime()` retires every Green Context
pair and compiled kernel created through the integration; it is terminal for
launches in this process, so call it only after every workspace and captured
graph is retired. It returns a list of per-backend failures instead of
raising.

## Limitations

- `activation="swiglu"` only; `activation_clamp` is supported on all three
  backends. `activation_alpha` is rejected: it only parameterizes the
  unsupported `swiglu_oai` activation. `num_tokens_bound` (a capture hint) is
  accepted on SM90/SM120 only.
- One call at a time per workspace: calls that share a workspace must be
  serialized (a concurrent call raises rather than silently interleaving).
- Staging is the caller's job: the operator reads `workspace.x`, `x_sf`,
  `topk_idx`, `topk_weights`; it does not quantize activations or compute
  routing itself.
- The workspace layer uses PyTorch's private
  `torch.distributed._symmetric_memory`; the vendored trees import it
  unconditionally, so a PyTorch build without it cannot run GC MoE even on a
  single rank (the wrapper checks first and reports it).

## Validation

`tests/experimental/gc_moe/` runs the public API against a pure-Torch MXFP4
reference (cosine > 0.97 and normalized L2 error < 0.05, so a magnitude-only
error cannot pass on direction alone), covers the workspace lifecycle, the
validation paths and the environment-fingerprint guard, and captures/replays
a CUDA graph bitwise-identically to the eager run. The SM90 and SM120 cases
skip on other architectures. Measured on 8x B300 (SM103, `sm100` backend,
single-rank group, E=4/H=512/I=256): cosine 0.9996-0.9997 and normalized L2
error 0.0255-0.0283 over 1/8/64 tokens against the decoded reference,
eager-vs-replay delta 0.0.

Latency: `python benchmarks/bench_gc_moe.py --tokens 1,8,64,256
[--cuda-graph]` sweeps token counts over one staged workspace.
