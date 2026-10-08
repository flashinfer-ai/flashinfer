# Experimental BF16 MoK training adapter

This is a **synthetic toy setting**, not a full-model training result.
The explicit `flashinfer.mok.prepare_mok_bf16` API returns a BF16 Mixture of
Kittens functional adapter. It covers scheduling, forward, and backward,
including input gradients, router-score gradients, and all six expert-weight
gradients. Shared-expert weight gradients remain source-rank local.

The implementation uses standalone CUDA sources and FlashInfer's TVM-FFI JIT.
It preserves the original launch boundaries: three scheduler kernels,
metadata gathering and peer barriers, one fused forward, one fused backward,
separate output/input-gradient epilogues, and empty-expert gradient zeroing.
Dispatch, GEMMs, SwiGLU, combine, and backward ring replay stay inside the
fused kernels. Framework copies, resets, and the router-gradient clone remain
part of a complete iteration.

## Use

The package includes its caller-owned workspace and metadata layer; no separate
MoK installation or extension is needed. A CUDA toolkit supporting the selected
device and PyTorch with symmetric-memory multicast support are required.
Distributed ranks must share one peer-accessible NVLink domain.

```python
from flashinfer.mok import prepare_mok_bf16, create_mok_bf16_workspace

config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=1024, hidden_size=6144, topk=8,
)
backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8)
# Using caller-owned inputs and weights:
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=16)
y, context = backend.forward(config, workspace, schedule, x, scores, *weights)
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)
```

`weights` contains shared gate/up/down followed by routed gate/up/down.
`grads` contains input, router scores, routed gate/up/down, then shared
gate/up/down gradients. Prepare and warm up before CUDA Graph capture.
All ranks must execute matching collective calls. Keep workspace allocations
and peer mappings alive while the adapter or a captured graph can use them.
Use a separate workspace and adapter for concurrent executions.

The forward context holds the dispatched routed rows, gate, up and hidden of
the macrobatch ring plus the real-row shared activations; routed outputs are
not retained. The BF16 backward dispatches the upstream routed gradient once,
score-scaled, and derives every routed gradient from that single ring; the
score gradient is the FP32 dot product of the recomputed hidden activation
with the scaled hidden gradient divided by the score, as in MoK (a route whose
score is exactly zero receives a zero score gradient). The MXFP8 backward
quantizes the unscaled gradient for the hidden gradient and the score-scaled
transposed copy for the weight gradient, so its zero scores are exact. Context
memory follows
the macrobatch ring and the real rows, not `schedule_capacity`. Pass the
matching context and schedule to backward; retain both while any captured
graph uses them.

`recompute_forward_context` rebuilds the context from `x` for activation
checkpointing: dispatch, the gate/up expert GEMMs and the SwiGLU only, no down
projections, combine or output. It is bitwise identical to the context
`forward` saved for the same inputs and schedule, so a backward from it
reproduces the saved-context backward bitwise. `swiglu_limit=L` (GLM-5.3-Flash
uses `L = 10`) selects the clamped activation `silu(min(gate, L)) * clamp(up,
-L, L)` with the exact masked backward for both expert kinds:

```python
context = backend.recompute_forward_context(
    config, workspace, schedule, x, *weights[:2], *weights[3:5], swiglu_limit=10.0
)
grads = backend.backward(
    config, workspace, schedule, context, dy, x, scores, *weights, swiglu_limit=10.0
)
```

Calling the experimental API is the opt-in; it emits an experimental warning.
There is no automatic dispatch or AOT registration.

## Unequal source inputs

`create_mok_bf16_workspace` collectively negotiates capacity from each rank's
actual `num_local_tokens`. Supply optional `source_capacity` with the same
value on every rank to reserve room for future inputs. The returned workspace
contains common physical `storage`; schedules and outputs use logical source
lengths. Invalid expert IDs mask padding, which contributes no routed work.
Empty source ranks still participate in every collective; their local shared
expert weight gradients are zero.

To change counts within capacity, create a new schedule and forward context.
Recapture CUDA Graphs when tensor shapes or addresses change. Same-shape value
updates can replay directly. Earlier graphs can still replay while their
inputs and workspace remain alive. Run graphs sharing a workspace serially
and in the same order across ranks. Growth beyond capacity requires all ranks
to recreate the workspace. Reserved capacity determines storage only; compute
and the retained shared activations follow the real input lengths (shared
expert tiles cover whole 256-row blocks).

Run `torchrun --standalone --nproc-per-node=4 examples/mok_bf16_unequal.py`
for unequal lengths, empty ranks, count changes and graph reuse.

## Toy contract

- Scheduler layouts `(EP, local experts, top-k)`: the toy layouts `(1, 4, 2)`,
  `(4, 4, 2)`, `(16, 16, 8)` and `(64, 4, 8)` (EP1/EP4 are small diagnostic
  settings), the 256-expert GLM-5.2 layouts `(4, 64, 8)`, `(8, 32, 8)` and
  `(32, 8, 8)`, and the 288-expert GLM-5.3-Flash layouts `(8, 36, 8)` and
  `(32, 9, 8)`.
- BF16 inputs, expert weights, outputs and weight gradients; FP32 positive
  router scores; contiguous int64 expert IDs. Each token selects distinct,
  valid expert IDs. Scores are already normalized/scaled by the caller.
- Logical source token counts may differ across ranks and may be zero. Hidden
  and intermediate widths are positive multiples of 256. Common physical
  source capacity is aligned for tiles/metadata and is at least 512 rows. Communication-SM
  counts are positive and even, leaving compute SMs available. When they are not given,
  `create_mok_bf16_workspace` uses the measured per-precision defaults: 20 forward / 24
  backward communication SMs for BF16 and 40 / 32 for MXFP8 (B200, EP4 GLM-5.2 shape,
  complete training step, 3 counterbalanced groups; BF16 sweep 16/24 35.56 ms, 20/20 34.35,
  20/24 34.18, 20/28 34.35, 24/24 34.45-34.53, 24/28 34.57, 28/24 34.71, 24/32 34.88,
  24/36 35.14; MXFP8 sweep 32/32 26.56, 36/32 25.93, 40/24 26.71, 40/28 26.04, 40/32 25.98,
  40/40 26.17, 40/48 26.44, 40/56 26.76. The MXFP8 kernels move half the activation
  bytes per compute tile and shift the optimum towards communication). The defaults are
  keyed by the compute capability of the workspace device (`comm_sms_defaults`): the
  212-SM compute-capability-10.7 part ships 40 / 32 for BF16 and 72 / 64 for MXFP8 (same
  protocol; BF16 full step 20/24 19.54 ms, 32/28 16.95, 40/32 16.81, 48/40 16.82, 56/40
  16.96, 64/48 17.18; MXFP8 full 72/64 13.05, 80/64 13.31, 80/56 13.52, 96/64 13.81,
  112/64 14.54, checkpoint 72/64 16.50, 80/64 16.62, 80/56 16.92). Generations without
  their own row use the B200 values. Pass `precision="mxfp8"` to the factory for an
  MXFP8 workload, or set the counts explicitly.
- Mini-batches are multiples of 256; macro-batches are multiples of the
  mini-batch. Schedule capacity must hold every padded route; overflow traps.
- An EP16 workload can use 16,384 source tokens per rank (262,144 global), hidden width
  6,144, intermediate width 2,048, 256 routed experts, and one shared expert.
  For this workload, explicitly choose mini/macro sizes 4,096/393,216 and
  sufficient schedule capacity; the smaller default ring is a diagnostic setting. Sequence length here is only a way to
  specify token count: no attention or sequence-dependent operation is tested.
- A larger macro ring uses more memory and can avoid forward-context
  recomputation inside backward. A comparison that changes this setting must
  report it, alongside peak memory, and time both complete iterations with
  the same CUDA Graph policy.

SwiGLU clipping, a separate recompute API, optimizer steps, router-network backpropagation, and shared-gradient all-reduce are outside this adapter.

## MXFP8 routed experts

`prepare_mok_bf16(..., mxfp8=True)` selects kernels whose routed experts run on
tcgen05 block-scaled MMA (E4M3 data, E8M0 scales, FP32 accumulation). The shared
experts, the dispatch and combine of `x`, `y` and `dx`, and the router-score
gradient stay BF16. The recipe is MoK's: one E8M0 scale per 32-element block
along the contraction axis, `scale = max(amax / 448, 1e-12)` rounded up to a
power of two, values rounded to E4M3 with saturation, scale bytes stored in
128 x 128 tiles (`[rows / 128, cols / 128, 32, 16]`). The caller quantizes the
routed weights once:

```python
from flashinfer.mok import mxfp8_quantize, prepare_mok_bf16

backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8, mxfp8=True)
wg, wu, wd = (mxfp8_quantize(w, True, True) for w in (w_gate, w_up, w_down))
# each tuple: (w_fp8, w_sc, w_t_fp8, w_t_sc), normal and transposed layouts
y, ctx = backend.forward(
    config, workspace, schedule, x, scores,
    sg, su, sd,  # shared experts, BF16
    (wg[0], wg[1]), (wu[0], wu[1]), (wd[0], wd[1]),
)
ctx = backend.recompute_forward_context(
    config, workspace, schedule, x, sg, su, (wg[0], wg[1]), (wu[0], wu[1])
)
grads = backend.backward(
    config, workspace, schedule, ctx, dy, x, scores,
    sg, su, sd, wg, wu, (wd[2], wd[3]), wgrad_f32=False,
)
```

Forward takes the routed `(w_fp8, w_sc)` pairs, recompute the gate/up pairs,
and backward the gate/up 4-tuples plus the down `(w_t_fp8, w_t_sc)` pair (its
transposed layout contracts over the hidden axis). `wgrad_f32=True` returns the
routed weight gradients in FP32 instead of BF16. The context stores the
dispatched rows, gate, up and hidden as `(E4M3, scale tiles)` pairs. The fused
kernels quantize the dispatched rows, the saved gate/up (from their BF16
values), the hidden activation and the backward `dy`, `dg` and `du` tiles in
both layouts, so there is no separate quantization pass and no compute on
padded rows. A context produced in one precision cannot be consumed by the
other.

Numerics: `mxfp8_reference.py` holds the recipe in FP32 PyTorch arithmetic and
the CUDA quantizer matches it bitwise. `tests/experimental/test_mok_bf16_mxfp8.py`
checks the fused kernels against fake-quant FP32 references of the same recipe
(BF16 rule `atol = rtol = 1e-2` with at most `max(4, 2e-7 * numel)` exceptions),
the saved tiles bitwise, repeated and recomputed executions bitwise, and reports
the end-to-end error against a plain FP32 oracle: about 7% relative L1 on `dx`,
the router-score gradient and the routed weight gradients for Gaussian inputs
(the E4M3 block quantization error of both GEMM operands), larger with a small
SwiGLU clamp because quantized gate values cross the clamp boundary.
`MOK_TOY_PRECISION=mxfp8` runs `examples/mok_bf16_toy.py` on the MXFP8 kernels
with the same fake-quant reference.

## Validation and lifecycle

Run `pytest tests/experimental/test_mok_bf16.py` for the independent
BF16-rounding reference (and `test_mok_bf16_mxfp8.py` for the MXFP8 path), empty experts, multiple ring lengths, variable
widths, repeated eager execution, and changed-input CUDA Graph replay.
The runnable distributed examples are `examples/mok_bf16_toy.py` and
`examples/mok_bf16_unequal.py`. The focused validation passes six single-GPU
tests and three tests on each of four ranks. Unequal inputs include empty
ranks, all-empty inputs, changed counts, three bitwise-identical graph
replays, earlier-graph reuse, and same-shape input updates.

Owner: Haozheng Fan. Tracking: [#6052](https://github.com/flashinfer-ai/flashinfer/issues/6052).
Experimental admission needs a maintainer-agreed release target. Graduation requires
resolving the remaining sanitizer qualification described below;
numerical checks alone do not clear that gate. No whole-model or comparative
performance claim is made.

The CUDA schedules derive from Cursor Research's Apache-2.0 Mixture of Kittens
implementation at revision `caeb2963f855c7ad53bb50c8bbf211086405cb98`. Copyright and modification notices are
retained in the generated source files.


## EP16 and EP64 validation

Launch one process per GPU in a single peer-accessible NVLink domain. The
full-shape driver uses H=6144, I=2048, 256 routed experts, top-8, and 16,384
source input tokens per rank (262,144 total at EP16; 1,048,576 at EP64).
It checks all nine outputs against an independent BF16-rounding reference,
routing and padding, three identical graph replays, changed source counts,
earlier-graph replay, and same-shape input updates.

On each participating node, with the same rendezvous address and a distinct
node rank:

```bash
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  examples/mok_bf16_validate.py --output results/fixed-uniform \
  --layout fixed --routing uniform --save-outputs --benchmark
```

Run each pair of `--layout fixed|near|strong` and
`--routing uniform|imbalanced` with a separate output directory. `near`
preserves the total token count with distinct neighboring rank lengths;
`strong` includes empty ranks and lengths up to 32,768. The `empty`
layout additionally exercises a globally empty batch.

The optional benchmark reports three groups of full captured iterations,
using completed CUDA-event intervals, at least 100 ms warmup and 1,000 ms
measurement on every rank per group, and the slowest rank's latency.
It includes recurring copies, scheduling, resets, and recomputation. Reference
checks and compilation are excluded. This is a warm-input timing protocol;
it does not establish a speedup or a whole-model training result.

`--sanitizer-smoke` runs the initial, changed-count and earlier graphs with
route/padding checks for per-rank Compute Sanitizer instrumentation. It omits
the reference and must accompany a separate full numerical pass. A successful
workload exit alone does not establish sanitizer acceptance: inspect every
rank's tool report.


Numerical acceptance now requires every element of all nine output/gradient
tensors on every rank to satisfy
`abs(actual - reference) <= 1e-2 + 1e-2 * abs(reference)`, with finite values.
Maximum absolute error and global relative L1 remain diagnostics only.
The driver records mismatch counts and worst normalized errors per rank and
globally, and preserves failed reports before returning a failing exit status.

Before the router-gradient repair, strict requalification completed all seven
cases at EP16 and EP64: **EP16 5/7 and EP64 2/7 passed**. Router-weight
gradients exceeded the elementwise tolerance in the cases below; the other
eight tensors passed. All cases retained finite outputs, three bitwise-identical
replays and earlier-Graph replay. This historical matrix does not qualify
the changed runtime; full EP16/EP64 requalification is pending.

| Source input lengths | Expert routing | EP16 | EP64 |
| --- | --- | --- | --- |
| fixed | uniform | PASS | FAIL |
| fixed | imbalanced | PASS | PASS |
| near | uniform | PASS | FAIL |
| near | imbalanced | FAIL | FAIL |
| strong | uniform | FAIL | FAIL |
| strong | imbalanced | PASS | FAIL |
| empty | uniform | PASS | PASS |

The previous aggregate-error qualification does not establish a strict pass.
The distributed BF16-rounding reference is unchanged. The repaired runtime
uses saved BF16 forward outputs for the score derivative; FP32-reference
runs remain separate diagnostics.

The previous runtime passed full-shape memcheck on all six nonempty
shape/routing pairs at each scale. The changed runtime requires new checks.
The previous full-shape synccheck passed those six pairs at EP16 and EP64
with the recorded communication configuration. A separate default-collective
diagnostic occurred before the adapter ran and reproduced in a collective-only
control.

Reduced-shape racecheck passes uniform and imbalanced routing at EP16/EP64
with 512 source tokens per rank, H=I=256, eight CPU workers per rank and
`--racecheck-deadlock-timeout 0`. This disables deadlock detection. Positive
windows of 10,000 and 60,000 ms report forward barrier deadlocks in diagnostic
runs, including a controlled eight-worker run. The cause remains unresolved;
these results do not establish full sanitizer acceptance.

Instrument one worker per GPU so each rank produces its own tool report. For
example, on each participating node, choose a shared output directory and run:

```bash
export MOK_VALIDATION_OUTPUT=results/synccheck-fixed-uniform
mkdir -p "$MOK_VALIDATION_OUTPUT"
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  --no-python bash -c 'exec compute-sanitizer --tool synccheck \
    --error-exitcode 99 --log-file "$MOK_VALIDATION_OUTPUT/rank-$RANK.log" \
    python examples/mok_bf16_validate.py --sanitizer-smoke \
    --output "$MOK_VALIDATION_OUTPUT/workload" --layout fixed --routing uniform'
```

Use distinct output directories for each layout/routing/tool combination.
Check the exit status and tool summary for every rank. The complete numerical
driver must run separately from `--sanitizer-smoke`.
