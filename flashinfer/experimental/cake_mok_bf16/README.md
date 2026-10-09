# Experimental MoK training adapter

This is a **synthetic training experiment**, not a full-model training result.
The explicit `flashinfer.mok.prepare_mok_bf16` API returns a Mixture of Kittens
(MoK) functional adapter for expert-parallel MoE training. It covers
scheduling, forward, context-only recompute and backward, including input
gradients, router-score gradients, and all six expert-weight gradients.
Shared-expert weight gradients remain source-rank local.

- **Routed experts** run in BF16, or natively in **MXFP8** (MoK recipe:
  caller-prequantized weights, activations and gradients quantized inside the
  fused kernels). The shared expert always runs in BF16.
- **Clamped SwiGLU** (`swiglu_limit=L`): `silu(min(gate, L)) * clamp(up, -L, L)`
  for routed and shared experts, with the matching backward masks. The default
  `swiglu_limit=None` is plain SwiGLU.
- **Context-only recompute** (`recompute_forward_context`) for activation
  checkpointing: it rebuilds the backward context without down projections,
  combine or the output epilogue.
- **FP32 weight-gradient accumulation** (`fp32_wgrad=True`): backward adds
  every weight-gradient contribution into caller-owned FP32 accumulators.
- **Unequal per-rank source counts**, including zero, as runtime inputs.

The implementation uses standalone generated CUDA sources and FlashInfer's
TVM-FFI JIT. It preserves the original launch boundaries: three scheduler
kernels, metadata gathering and peer barriers, one fused forward (or
recompute), one fused backward, separate output/input-gradient epilogues, and
empty-expert gradient zeroing. Dispatch, GEMMs, SwiGLU, quantization, combine
and backward ring replay stay inside the fused kernels. Framework copies,
resets and the router-gradient clone remain part of a complete iteration.

## Use

The package includes its caller-owned workspace and metadata layer; no separate
MoK installation or extension is needed. A CUDA toolkit supporting the selected
device and PyTorch with symmetric-memory multicast support are required.
Distributed ranks must share one peer-accessible NVLink domain (peers may span
hosts of one NVL72 domain).

```python
from flashinfer.mok import (
    create_mok_bf16_workspace,
    prepare_mok_bf16,
    quantize_mok_mxfp8_weights,
)

config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=6144, topk=8,
)
backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8)
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=16)
y, context = backend.forward(config, workspace, schedule, x, scores, *weights)
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)

# Native MXFP8 routed experts: prequantize after every weight update.
fwd_routed, bwd_routed = quantize_mok_mxfp8_weights(gate_w, up_w, down_w)
backend.prepare(mxfp8=True)  # compile before CUDA Graph capture
y, context = backend.forward(config, workspace, schedule, x, scores,
                             *shared, *fwd_routed)
# Activation checkpointing: drop the context, rebuild it before backward.
del context
context = backend.recompute_forward_context(
    config, workspace, schedule, x, shared[0], shared[1], *fwd_routed[:2])
grads = backend.backward(config, workspace, schedule, context, dy, x, scores,
                         *shared, *bwd_routed)
```

`weights` contains shared gate/up/down followed by routed gate/up/down; gate
and up are `[I, H]` (shared) or `[E_local, I, H]` (routed) and down is
`[H, I]` or `[E_local, H, I]`. `grads` contains input, router scores, routed
gate/up/down, then shared gate/up/down gradients. MXFP8 routed weights follow
MoK's functional API: forward and recompute take `(data, scales)` pairs;
backward takes `(data, scales, data_t, scales_t)` for gate and up and the
transposed `(data_t, scales_t)` pair for down. `quantize_mok_mxfp8_weights`
produces both sets (E4M3 data, UE8M0 scales for 32-element blocks along the
reduction dimension, tensor-core scale tiles `[E * rows / 128, cols / 128, 32,
16]`). MXFP8 weight gradients are BF16 (or the FP32 accumulators).

`swiglu_limit` is a per-call runtime value; plain and clamped SwiGLU are
separate kernel builds. `prepare_mok_bf16(clamped_swiglu=..., fp32_wgrad=...)`
compiles one BF16 variant immediately; call `prepare(swiglu_limit, mxfp8)` and
`prepare_recompute(swiglu_limit, mxfp8)` for every other variant before CUDA
Graph capture. With `fp32_wgrad=True`, pass six FP32 tensors (shared gate, up,
down, routed gate, up, down) as `weight_grad_accumulators`; backward adds into
them and returns them.

Prepare and warm up before CUDA Graph capture. All ranks must execute
matching collective calls. Keep workspace allocations and peer mappings alive
while the adapter or a captured graph can use them. Use a separate workspace
and adapter for concurrent executions. Pass the matching context and schedule
to backward; retain both while any captured graph uses them.

Calling the experimental API is the opt-in; it emits an experimental warning.
There is no automatic dispatch or AOT registration.

## Context, memory and recompute

A forward context holds the shared-expert activations and the routed
activations of the first macrobatch, which stays resident in the forward ring
(MXFP8 keeps the routed x and activation transposed in MXFP8 and gate/up in
E4M3). Backward replays later macrobatches inside the fused kernel. No routed
expert output is saved: the router-score gradient is fused into SwiGLU
backward as `dot(d_hidden, hidden) / score`, as in native MoK. Scores must be
positive; a zero score yields a zero score gradient. A larger macrobatch ring
uses more memory and replays less in backward.

`recompute_forward_context(config, workspace, schedule, x, shared_gate,
shared_up, routed_gate, routed_up, swiglu_limit=None)` dispatches and
recomputes the gate/up GEMMs and SwiGLU of exactly that context. Pass the same
schedule, inputs, weights and limit as the checkpointed forward; the returned
context is bitwise identical to the forward's and interchangeable with it.

Local work follows the real source rows: shared-expert GEMMs, their weight
gradients and both epilogues read only the active prefix rounded up to 256
rows; peers read only routed rows. Reserved source or schedule capacity adds
memory, not work.

## Targets

`sources.json` lists every kernel role with the exact targets each source
variant was generated for. The BF16 kernels, scheduler, communication and
epilogue sources are shared by SM100a, SM103a and SM107a. The MXFP8 fused
kernels have target-specific schedules: SM100a/SM103a (512 tensor-memory
columns) use 256 x 256 routed tiles with a six-stage operand ring, and SM107a
(576 columns) uses 256 x 512 tiles that compute and save gate and up in one
tile, with the K=64 block-scaled MMA. The adapter selects the variant from the
device compute capability (10.0, 10.3 or 10.7) and compiles it for that exact
target; SM107a requires a CUDA toolkit that supports it. Every generated
source is checksum-verified before compilation.

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
to recreate the workspace.

## Contract

- Scheduler layouts `(EP, local experts)`: `(1, 4)` and `(4, 4)` diagnostic
  settings; 256 routed experts at EP 4/8/16/32/64 (`(4, 64)`, `(8, 32)`,
  `(16, 16)`, `(32, 8)`, `(64, 4)`); 288 routed experts (GLM-5.3-Flash) at EP8
  and EP32 (`(8, 36)`, `(32, 9)`). Top-k is 2 or 8.
- BF16 inputs, shared weights and outputs; FP32 positive router scores;
  contiguous int64 expert IDs. Each token selects distinct, valid expert IDs.
  Scores are already normalized/scaled by the caller.
- Logical source token counts may differ across ranks and may be zero. Hidden
  and intermediate widths are positive multiples of 256; MXFP8 also requires a
  hidden width that is a multiple of 512. Common physical source capacity is
  aligned for tiles/metadata and is at least 512 rows. Communication-SM counts
  are positive and even, leaving compute SMs available.
- Mini-batches are multiples of 256; macro-batches are multiples of the
  mini-batch. Schedule capacity must hold every padded route; overflow traps.
- Routed weight gradients are summed across macrobatches in BF16 (or in the
  FP32 accumulators). MXFP8 adds quantization error to the routed outputs and
  gradients; the shared expert path is unchanged.

Optimizer steps, router-network backpropagation and shared-gradient
all-reduce are outside this adapter.

## Validation and lifecycle

Single-GPU tests:

- `tests/experimental/test_mok_bf16.py`: fused BF16 forward/backward against
  an independent BF16-rounding reference (empty experts, multiple ring lengths,
  variable widths, CUDA Graph replay, changed and zero scores), clamped SwiGLU,
  FP32 accumulation, GLM-5.3-Flash local expert counts, recompute versus
  saved context, every scheduler layout and the epilogues.
- `tests/experimental/test_mok_mxfp8.py`: the weight quantizer, the fused MXFP8
  forward and forward+backward against a quantization-aware reference
  (plain and clamped), GLM-5.3-Flash local expert counts and MXFP8 recompute.

Distributed tests (`torchrun --standalone --nproc-per-node=4 -m pytest ...`):
`test_mok_bf16_distributed.py` (toy), `test_mok_bf16_unequal.py` (unequal and
empty ranks, graph reuse), `test_mok_bf16_capacity.py` (API guards) and
`test_mok_training_distributed.py` (clamped SwiGLU, recompute versus saved
context, FP32 accumulation and MXFP8 over unequal ranks; 288 experts at EP8 or
EP32). `test_mok_bf16_tolerance.py` checks the tolerance reporter. The
runnable examples are `examples/mok_bf16_toy.py`,
`examples/mok_bf16_unequal.py` and `examples/mok_training_features.py`.

Owner: Haozheng Fan. Tracking: [#6052](https://github.com/flashinfer-ai/flashinfer/issues/6052).
Experimental admission needs a maintainer-agreed release target. Graduation
requires completing the sanitizer qualification below; numerical checks alone
do not clear that gate. No whole-model performance claim is made.

The CUDA schedules derive from Cursor Research's Apache-2.0 Mixture of Kittens
implementation at revision `caeb2963f855c7ad53bb50c8bbf211086405cb98`. Copyright and modification notices are
retained in the generated source files.

## Full-scale validation

Launch one process per GPU in a single peer-accessible NVLink domain. The
full-shape driver defaults to H=6144, I=2048, 256 routed experts, top-8, and
16,384 source input tokens per rank; `--experts 288 --hidden 4096
--swiglu-limit 10` selects GLM-5.3-Flash at EP8 or EP32 and `--mxfp8` runs the
routed experts in MXFP8. It checks the outputs against independent
references, routing and padding, three identical graph replays, changed source
counts, earlier-graph replay, and same-shape input updates.

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

BF16 acceptance requires every element of all nine output/gradient tensors on
every rank to satisfy `abs(actual - reference) <= 1e-2 + 1e-2 * abs(reference)`
against the BF16-rounding reference, with finite values. The fused score
gradient follows native MoK (`dot(d_hidden, hidden) / score`), so at full
scale a handful of the millions of score-gradient elements can exceed this
gate against the BF16 reference while matching it as closely as MoK does; the
driver reports mismatch counts and worst normalized errors per rank and
globally. MXFP8 runs gate the global relative L2 error of the routed-dependent
outputs against an FP32 reference and the shared expert weight gradients
elementwise.

The optional benchmark reports three groups of full captured iterations,
using completed CUDA-event intervals, at least 100 ms warmup and 1,000 ms
measurement on every rank per group, and the slowest rank's latency. It
includes recurring copies, scheduling, resets, and recomputation. Reference
checks and compilation are excluded. This is a warm-input timing protocol;
it does not establish a speedup or a whole-model training result.

`--sanitizer-smoke` runs the initial, changed-count and earlier graphs with
route/padding checks for per-rank Compute Sanitizer instrumentation. It omits
the reference and must accompany a separate full numerical pass. A successful
workload exit alone does not establish sanitizer acceptance: inspect every
rank's tool report. Instrument one worker per GPU so each rank produces its
own tool report. For example, on each participating node, choose a shared
output directory and run:

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
