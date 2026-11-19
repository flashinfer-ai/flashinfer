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

The implementation uses standalone generated CUDA sources, loaded prebuilt
from `aot/` when the package ships them and otherwise built by FlashInfer's
TVM-FFI JIT from `build.json`. It preserves the original launch boundaries: three scheduler
kernels, metadata gathering and peer barriers, one fused forward (or
recompute), one fused backward, separate output/input-gradient epilogues, and
empty-expert gradient zeroing. Dispatch, GEMMs, SwiGLU, quantization, combine
and backward ring replay stay inside the fused kernels. Framework copies,
resets and the router-gradient clone remain part of a complete iteration.

## Use

The package includes its caller-owned workspace and metadata layer; no separate
MoK installation or extension is needed. PyTorch with symmetric-memory
multicast support is required. Minimum CUDA toolkit (nvcc) per target, the
`min_cuda` of every library in `build.json`: 12.8 for SM100a, 12.9 for SM103a
and 13.4 for SM107a (checked by compiling every SM100a and SM103a library with
12.8, 12.9, 13.0 and 13.1: each compiles from its floor up, and 12.8 rejects
`compute_103a`).
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

# BF16 activation checkpointing without storing the context at all.
backend.prepare(save_context=False)  # compile before CUDA Graph capture
y, _ = backend.forward(config, workspace, schedule, x, scores, *weights,
                       save_context=False)
context = backend.recompute_forward_context(
    config, workspace, schedule, x, *weights[:2], *weights[3:5])
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)
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
16]`). On CUDA tensors it launches one fused Triton kernel per operand that
reads the BF16 weights once and writes both layouts, without host
synchronization, so it can be captured in a CUDA Graph;
without Triton, PyTorch operations produce the same bytes. MXFP8 weight
gradients are BF16 (or the FP32 accumulators).

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

`backward` returns `dx`, `dscores` and the six weight gradients. The fused
kernel's routed and shared scratch is released as soon as the launch is
enqueued, before the output epilogue and the score-gradient copy allocate,
so a training step's peak allocation is inside the fused backward launch for
both precisions, except when the intermediate size is below a third of the
hidden size and a rank's source rows exceed twice the macrobatch.

`recompute_forward_context(config, workspace, schedule, x, shared_gate,
shared_up, routed_gate, routed_up, swiglu_limit=None)` dispatches and
recomputes the gate/up GEMMs and SwiGLU of exactly that context. Pass the same
schedule, inputs, weights and limit as the checkpointed forward; the returned
context is bitwise identical to the forward's and interchangeable with it.

### Context-free forward

With BF16 routed experts, `forward(..., save_context=False)` runs the
context-free forward of an activation-checkpointed step and returns
`(y, None)`. Its fused tiles store hidden, which the down projection reads,
and skip the shared and routed gate/up stores that only backward reads; it
also allocates no gate/up tensors. `y` is bitwise identical to the context
forward's `y`. Before `backward`, rebuild the context with
`recompute_forward_context` (same schedule, inputs, weights and limit);
backward then returns the same bits as after a context forward. Compile it
with `prepare(swiglu_limit, save_context=False)` before CUDA Graph capture.
The MXFP8 forward always saves its context and rejects `save_context=False`.
The kernels are the `forward_noctx` and `forward_clamped_noctx` roles; they
take the forward's arguments and launcher.

Local work follows the real source rows: shared-expert GEMMs, their weight
gradients and both epilogues read only the active prefix rounded up to 256
rows; peers read only routed rows. Reserved source or schedule capacity adds
memory, not work.

## Targets

`sources.json` lists every kernel role with the targets it is generated
for, the argument names of its entry and its architecture-free `library`
stem (`cake_mok_<role>`); `build.json` (schema `cake-build/1`) records the
sources, digests and nvcc flags of every `<library>_<arch>` build
(`cake_mok_<role>_<arch>`, for example `cake_mok_forward_sm_100a`), and the
shared `cake_build.py` loader loads it (prebuilt from `aot/`, else JIT).
`jit.variant(role, capability)` returns the record selected by the device
compute capability and `jit.library_name(role, capability)` the name of its
`build.json` entry, a plain string like the other Cake loaders. The bindings include the
package's `csrc/cake_ffi_checks.h`, no FlashInfer header. The fused bindings
(`csrc/sm_100a/*_binding.cu`), the BF16 fused kernels (`csrc/sm_100a/*.cu`,
which select the UMMA shared-memory descriptor address mask from
`__CUDA_ARCH__`: 15 bits on SM107a, 14 bits otherwise) and the scheduler,
communication and epilogue sources are shared by SM100a, SM103a and SM107a.
Launchers of the same shape share one TU: the plain fused launcher also
serves the `_clamped` kernel of its role (and the forward launcher the
`_noctx` kernels), and `csrc/cake_mok_count_binding.cu`,
`cake_mok_rows_binding.cu`, `cake_mok_pad_binding.cu`,
`cake_mok_epilogue_2_binding.cu` and `cake_mok_epilogue_8_binding.cu` serve
their kernel families; the entry's `nvcc_flags` name the kernel with
`-DCAKE_MOK_KERNEL=<symbol>` (and the count/rows launch's dynamic shared
memory with `-DCAKE_MOK_DYN_SMEM=<bytes>`).
The MXFP8 fused kernels have target-specific schedules: SM100a/SM103a (512 tensor-memory
columns) use 256 x 256 routed tiles with a six-stage operand ring, and SM107a
(576 columns) uses 256 x 512 tiles that compute and save gate and up in one
tile, with the K=64 block-scaled MMA. The adapter selects the variant from the
device compute capability (10.0, 10.3 or 10.7) and loads its build for that
exact target; on the JIT path SM107a requires a CUDA toolkit that supports it
and every generated source is checksum-verified before compilation.

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

`build_schedule` derives the expert layout once per schedule
(`MoKSchedule.layout`, int32 `[3 * local experts + schedule_capacity / 256]`:
padded counts, row offsets, real rows and the expert of each 256-row block) and
every fused launch of that schedule reads it; a schedule built without it
(`layout=None`) makes each launch derive its own. The layout is a function of
`tokens_per_expert` and `peer_rank`, whose real rows come first within each
expert's padded rows; a schedule copied with `dataclasses.replace` that changes
`peer_rank`, `tokens_per_expert` or the capacity must pass `layout=None`. The
launchers memoise their launch-invariant geometry per distinct launch
arguments (integers and, for MXFP8, the device); the memo holds no tensors
and does not change the launch sequence.

## Repeatability

Within the scope below, every output repeats bit for bit:

| Output | Repeats bit for bit |
| --- | --- |
| `forward`: `y` | yes; identical with `save_context=False` |
| `forward`: context | yes |
| `recompute_forward_context`: context | yes, identical to the forward's |
| `backward`: `dx` | yes |
| `backward`: `dscores` | yes |
| `backward`: six weight gradients | yes |

The context is `x_routed`, `gate_shared`, `gate_routed`, `up_shared`,
`up_routed`, `hidden_shared` and `hidden_routed` (MXFP8 routed fields as data
and scales); the recomputed context is bitwise identical to the forward's (see
Context, memory and recompute). The weight gradients are shared and routed
gate/up/down, or the six FP32 accumulators with `fp32_wgrad=True`. This holds
for BF16 and MXFP8 routed experts and for plain and clamped SwiGLU.

Scope: repeated calls with the same inputs (source tokens `x`, expert IDs,
scores, per-rank source counts, `dy`), the same weights (and the same MXFP8
prequantized weights), the same `MoKConfig` values other than the
communication-SM split (minibatch and macrobatch sizes, schedule capacity), the
same precision, `swiglu_limit` and `fp32_wgrad`, the same EP group (size and
rank order), the same package build and the same GPU model (compute capability
and SM count). Results may differ bitwise across GPU architectures, EP sizes,
minibatch or macrobatch sizes, precisions, package versions and CUDA
toolkits.

No output bit depends on:

- the communication-SM split: a fixed pair, the `"auto"` rows or the
  imbalance-aware switch, including ranks running different splits in one
  step;
- other kernels resident on other streams that occupy SMs while MoK runs;
- CUDA Graph replay versus eager launches;
- on SM107a MXFP8, the choice of the two-macro-batch forward program (see
  Contract).

Every reduction runs in a fixed order: the top-k combine adds each token's
slots in slot order, each routed weight-gradient tile has one writer over its
full reduction, router-score gradient partials are summed in a fixed order and
later macrobatches are added in macrobatch order. Kernels use atomics only on
integers, to count rows and claim work, never to add floating-point values.

Checked on GB300 (SM103a) by repeating the checkpointed training step (schedule,
forward, context recompute, backward) five times and comparing every output
bitwise on every rank: GLM-5.2 shapes (H 6144, I 2048, 256 experts, top 8) at
EP4/8/16/64 and GLM-5.3-Flash shapes (H 4096, I 2048, 288 experts, top 8,
clamped SwiGLU limit 10) at EP8/32; 16,384 tokens per rank, FP32 accumulators,
BF16 and MXFP8; uniform, ragged (1 to 32,767 tokens, one empty rank) and
1.5x-hot routing; with a resident side-stream kernel holding up to 16 SMs,
across communication-SM splits and under CUDA Graph replay.

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
  hidden width that is a multiple of 512. Below CUDA 13.1 the MXFP8 forward
  and recompute roles convert to E4M3 through FP32
  (`cvt.rn.satfinite.e4m3x2.f32`) instead of `cvt.rn.satfinite.e4m3x2.bf16x2`,
  which older ptxas rejects. Every sm_107a (Rubin) library needs CUDA 13.4, the
  first nvcc that accepts `compute_107a`.
  Common physical source capacity is
  aligned for tiles/metadata and is at least 512 rows. Communication-SM counts
  are positive and even, leaving compute SMs available.
- Imbalance-aware split (`MoKConfig.imbalance_comm_sms` / `imbalance_threshold`;
  factory default `imbalance_comm_sms="auto"`, `None` keeps the fixed split): every
  `build_schedule` counts the gathered expert IDs per owning rank on device
  inside the scheduler's count kernel (the same pass that counts this rank's
  experts; no extra launch) and copies the counts into pinned host memory
  behind a CUDA event (no host synchronization; skipped during graph capture).
  The gathered IDs and the counts are identical on every rank; each rank
  applies its own latest completed observation, so with back-to-back
  steps a launch may use different splits on different ranks for a step - the
  split is a per-rank launch parameter and the protocol does not require
  agreement. While the hottest rank's load is at least `imbalance_threshold`
  x the mean load (facade default `"auto"`: the device's value in
  `IMBALANCE_DEVICE_THRESHOLDS`, else the precision's value in
  `IMBALANCE_THRESHOLDS`, BF16 4, MXFP8 2.7), the later launches use the
  (forward, backward) imbalance split; balanced steps are unchanged. Measured
  rows (`IMBALANCE_COMM_SMS`, keyed by compute capability and SM count, so B200
  at 148 SMs only - GB200 is also (10, 0) at 152 SMs and keeps the fixed split,
  where the B200 BF16 row measured slower under skew at EP16-64), from EP8
  complete training steps, two rounds on one node, step time against the fixed
  split of this package's programs. B200 BF16 24 / 28 against 20 / 24: hottest
  rank at 5.6x / 7.4x the mean routed load -1.0 / -1.7 % and -1.2 / -1.3 %;
  hottest rank at 3.0x (one rank's experts at twice the routing probability)
  +0.4 / +0.7 % (full step), +0.5 / +0.8 % (checkpoint step) and -0.1 ... +1.6 %
  with ragged source rows (8 cells, 7 slower): no gain, within or just above the
  ~0.8 % spread of identical arms, so the fixed split is kept there; the BF16
  threshold sits between the measured 3.0x and 5.6x (the crossover in between is
  not measured; with 4 ranks the ratio reaches 4 only when one rank carries
  every routed row). B200 MXFP8 44 / 36 against 40 / 32: hottest rank at 3.0x
  -0.8 / -1.2 % (full step) and -1.1 / -1.0 % (checkpoint step), at 5.6x
  -2.6 / -2.9 %; balanced steps +0.6 / +0.3 % (full) and +0.9 / +0.8 %
  (checkpoint) under an unconditional 44 / 36, which is why only imbalanced
  steps switch; with 4 ranks at 2.4x +0.8 % (one screening round). The MXFP8
  threshold sits between the measured 2.4x and 3.0x. Rubin (212 SMs) MXFP8
  72 / 60 over its fixed 60 / 60 from a device threshold of 1.5
  (`IMBALANCE_DEVICE_THRESHOLDS`, which replaces the precision's threshold for
  that device), two-node EP8, 27116 tokens per rank, configurations rotated on
  one workspace, four rounds, hottest-rank medians, MoE training-update proxy:
  hottest rank at 1.51x the mean routed load forward -6.9 % / update -3.8 %,
  2.05x -5.7 / -2.8 %, 3.05x -10.1 / -5.1 %, 4.66x -11.1 / -5.5 %, 6.47x
  -10.6 / -5.5 %; balanced steps do not switch, where an unconditional 72 / 60
  forward is 4.1 % slower. No row elsewhere.
  `MoKConfig.comm_split_log=True` (default `False`; `dataclasses.replace(config,
  comm_split_log=True)` on the returned config) prints each split change per
  rank to stderr; the `CAKE_MOK_COMM_SPLIT_LOG` environment variable that used
  to select this no longer exists.
- Two-macro-batch forward program (sm_107a MXFP8 only): the
  `forward_mxfp8[_clamped]_multimacro` build streams macro batch 0's
  unaliased rows to the experts while macro batch 1 combines, instead of
  after it. Each rank chooses it from the same lagged observation as the
  split, when its padded schedule rows (`num_tokens`) filled exactly two
  macro batches; otherwise, before the first observation and at a schedule
  capacity of at most one macro batch the default build runs (and no
  observation is queued for this alone). Both builds give bitwise the same
  outputs, context and gradients for any schedule, so ranks may run different
  builds in one step and a stale choice costs time only. `comm_split_log`
  also prints each change of the choice.
- Fixed split per device and group size: `fwd_num_comm_sms` / `bwd_num_comm_sms`
  default to `"auto"` in `create_mok_bf16_workspace` (package and
  `flashinfer.mok` wrapper): the measured `COMM_SMS` row (compute capability, SM
  count, precision, EP group size), else the factory 40 / 40. Integers fix the
  split; `"auto"` for only one of the two is an error. With MXFP8 routed experts
  pass `precision="mxfp8"` for the MXFP8 row (`precision` only selects rows and
  thresholds). The split changes time, never bits (see Repeatability). GB300
  (152 SMs): BF16 20 / 24 at EP4/8/16/32/64; MXFP8 44 / 36 at EP8/32/64, none
  at EP4/16 (no gain measured). Checkpointed training steps captured in CUDA
  Graphs, 40 / 40 and the row interleaved in one process on the same nodes,
  slowest-rank median of 12 steps, 16,384 tokens per rank, FP32 accumulators:

  | EP | shape | precision | routing | 40 / 40 (ms) | row (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- |
  | 4 | GLM-5.2 | BF16 | uniform | 31.95 | 29.83 | -6.6 % |
  | 8 | GLM-5.2 | BF16 | uniform | 30.89 | 28.50 | -7.7 % |
  | 8 | GLM-5.3-Flash | BF16 | uniform | 21.17 | 19.31 | -8.8 % |
  | 16 | GLM-5.2 | BF16 | uniform | 30.81 | 28.19 | -8.5 % |
  | 32 | GLM-5.3-Flash | BF16 | uniform | 21.11 | 18.84 | -10.7 % |
  | 32 | GLM-5.3-Flash | BF16 | ragged | 26.83 | 24.25 | -9.6 % |
  | 32 | GLM-5.3-Flash | BF16 | 1.5x hot rank | 29.51 | 26.52 | -10.2 % |
  | 64 | GLM-5.2 | BF16 | uniform | 30.91 | 30.12 | -2.6 % |
  | 64 | GLM-5.2 | BF16 | ragged | 39.66 | 38.29 | -3.5 % |
  | 64 | GLM-5.2 | BF16 | 1.5x hot rank | 42.52 | 39.70 | -6.6 % |
  | 8 | GLM-5.2 | MXFP8 | uniform | 21.38 | 21.46 | +0.4 % |
  | 8 | GLM-5.3-Flash | MXFP8 | uniform | 15.60 | 15.54 | -0.4 % |
  | 32 | GLM-5.3-Flash | MXFP8 | uniform | 15.35 | 14.80 | -3.6 % |
  | 32 | GLM-5.3-Flash | MXFP8 | ragged | 20.42 | 20.13 | -1.4 % |
  | 32 | GLM-5.3-Flash | MXFP8 | 1.5x hot rank | 20.77 | 20.40 | -1.8 % |
  | 64 | GLM-5.2 | MXFP8 | uniform | 25.71 | 25.25 | -1.8 % |
  | 64 | GLM-5.2 | MXFP8 | ragged | 36.58 | 35.99 | -1.6 % |
  | 64 | GLM-5.2 | MXFP8 | 1.5x hot rank | 32.67 | 32.89 | +0.7 % |

  The MXFP8 EP8 row first measured -3.3 ... -1.8 % against 40 / 32 with the
  hottest rank at 3.0x-7.4x the mean routed load; on balanced steps it is
  within -0.4 ... +0.4 % of 40 / 40 (five comparisons, EP8, both shapes).
  Rubin (212 SMs) MXFP8 EP8 60 / 60 against 40 / 40, balanced routing,
  hottest-rank medians of four rounds, forward / backward at 8192 / 27116 /
  32768 tokens per rank: -7.4 / -4.6, -13.3 / -8.7, -18.1 / -2.4 % (round
  spread at most 1.2 %); Rubin BF16 had no consistent gain at EP8.
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
  saved context, the context-free forward (`y`, combine rows, hidden and
  routed x against the context forward, and backward after recompute against
  backward from the saved context, all bitwise), every scheduler layout (with the owner-rank load counts
  against the host reference and the expert layout against a torch reference,
  explicit and launcher-derived), the counter slab, the launch-geometry memo
  and the epilogues.
- `tests/experimental/test_mok_mxfp8.py`: the weight quantizer (fused kernel,
  PyTorch path and reference byte-identical, NaN / infinity / zero blocks
  included), the fused MXFP8
  forward and forward+backward against a quantization-aware reference (plain
  and clamped; routed and shared-expert gradients; BF16 or FP32
  weight-gradient accumulation), GLM-5.3-Flash local expert counts and MXFP8
  recompute.

Distributed tests (`torchrun --standalone --nproc-per-node=4 -m pytest ...`):
`test_mok_bf16_distributed.py` (toy), `test_mok_bf16_unequal.py` (unequal and
empty ranks, graph reuse), `test_mok_bf16_capacity.py` (adapter guards) and
`test_mok_training_distributed.py` (clamped SwiGLU, recompute versus saved
context, the context-free BF16 forward with recompute versus saved context,
FP32 accumulation with BF16 and MXFP8 routed experts, and MXFP8
over unequal ranks; 288 experts at EP8 or EP32). `test_mok_bf16_tolerance.py` checks the tolerance reporter.
`test_mok_bf16.py`, `test_mok_mxfp8.py` and `test_mok_bf16_capacity.py` import the
package only through `tests/cake/test_helpers/cake_package.py`; the torchrun suites and
`test_mok_bf16_tolerance.py` drive the `examples/mok_*.py` scripts, which use the
`flashinfer.mok` wrapper;
`test_mok_bf16_api.py` checks the `flashinfer.mok` wrapper itself (experimental
marking, signatures, re-exports and argument forwarding). The
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
