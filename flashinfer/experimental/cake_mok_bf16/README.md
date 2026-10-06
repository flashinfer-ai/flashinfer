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

Each forward context retains the BF16 routed expert outputs across all
macrobatches. Backward computes each supplied-score gradient as their FP32
dot product with the original upstream gradient, inside the fused kernel.
This requires `2 * schedule_capacity * hidden_size` bytes per live context
for routed outputs. Other routed activations keep the macrobatch ring and
recompute policy. Pass the matching context and schedule to backward; retain
both while any captured graph uses them.

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
to recreate the workspace. Padding increases storage and work according to
the maximum reserved capacity, not only the sum of real input lengths.

Run `torchrun --standalone --nproc-per-node=4 examples/mok_bf16_unequal.py`
for unequal lengths, empty ranks, count changes and graph reuse.

## Toy contract

- Scheduler layouts `(EP, local experts, top-k)`: `(1, 4, 2)`, `(4, 4, 2)`,
  `(16, 16, 8)`, and `(64, 4, 8)`. EP1/EP4 are small diagnostic settings.
- BF16 inputs, expert weights, outputs and weight gradients; FP32 positive
  router scores; contiguous int64 expert IDs. Each token selects distinct,
  valid expert IDs. Scores are already normalized/scaled by the caller.
- Logical source token counts may differ across ranks and may be zero. Hidden
  and intermediate widths are positive multiples of 256. Common physical
  source capacity is aligned for tiles/metadata and is at least 512 rows. Communication-SM
  counts are positive and even, leaving compute SMs available.
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

SwiGLU clipping, MXFP8, a separate recompute API, optimizer steps, router-network
backpropagation, and shared-gradient all-reduce are outside this adapter.

## Validation and lifecycle

Run `pytest tests/experimental/test_mok_bf16.py` for the independent
BF16-rounding reference, empty experts, multiple ring lengths, variable
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
