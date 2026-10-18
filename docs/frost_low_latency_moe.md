# FROST low-latency routed MoE

`FrostLowLatencyConfig` adds an explicit low-token candidate to `MoELayer` on
SM100/SM103. It uses CuTe DSL and the existing FlashInfer low-latency blockscaled
GEMM's TMA producers and epilogue. It does not depend on cuDNN.

The backend owns both scalar FMA and Tensor Core tactics. FMA reuses the device
implementation previously embedded in `cudnn_frost`, while reading the same
prepared weights as the Tensor Core tactic. Switching tactics needs no second
weight copy, requantization, or per-call layout conversion. Autotuning compares
the complete pipelines; the untuned default remains Tensor Core.

When Frost is an eligible candidate, `MoELayer` tunes every candidate's tactics
and the cross-backend choice using individual-call CUDA Graph GPU latency.
An explicit autotune measurement policy can still override cache or execution
mode. External event nodes bracket each invocation inside the graph; the median
excludes Python submission and first-replay initialization. This objective can
rank FMA and Tensor Core differently from amortized back-to-back replay. The
autotuner records the timing mode as provenance, so a tuning request remeasures
a winner selected under the other mode. Layers without an eligible Frost
candidate retain their existing timing behavior. This remains kernel GPU time,
not model decode-step latency or throughput.

```mermaid
flowchart TD
  W[One prepared MX weight view] --> A[Low-latency runner and per-call workspace]
  A --> T[Tensor Core tactic: tile K 128 or 256]
  A --> F[FMA tactic: top-k at most 32]
  T --> S[Shared FP32 SwiGLU and MXFP8 intermediate contract]
  F --> S
  S --> O[BF16 expert results and ordered FP32 FMA combine]
  L[Legacy cudnn_frost callers] --> C[Compatibility adapters: original numerics]
  C --> D[Shared FMA device source]
  F --> D
```

Code sharing does not make the old and new numerical contracts interchangeable:

| Boundary | Low-latency FMA and TC | Legacy MX FMA adapter |
|---|---|---|
| Activation before MXFP8 quantization | FP32 | Rounded to BF16 first |
| Activation parameters | Layer's SwiGLU alpha/beta/limit | Existing admitted defaults |
| Expert result | BF16 | BF16 |
| Weighted combine | Slot-ordered FP32 FMA | Separate FP32 multiply and slot-ordered add |

Legacy entry points preserve these behaviors. Their compatibility mode is fixed
by the adapter and is never an autotuning choice in the low-latency backend.
Dot-product accumulation order may differ between FMA and Tensor Core; the
independent reference tests enforce the documented numerical tolerances.

```mermaid
flowchart LR
  X[MXFP8 input + linear UE8M0 scales] --> S[Pack input scales]
  R[Precomputed expert IDs] --> GU
  S --> GU[Gate/up GEMM + SwiGLU + MXFP8 quantization]
  GU -->|PDL dependency| D[Down GEMM]
  D --> C[Ordered weighted combine]
  C --> Y[BF16 output]
```

Compared with composing standalone low-latency GEMMs and an activation/quantizer,
this pipeline removes the separate activation/quantization launch. GU's FP32
accumulators still pass through per-call global scratch before activation; this
is not an all-on-chip pipeline. Native MoE backends can already fuse activation,
so fusion alone does not explain the comparison with them.
Runtime expert IDs select weights directly, avoiding
sorting tokens into expert groups for tiny batches. Eight warps per CTA and a
smaller tensor-memory allocation reduce resources per independent route tile.
PDL lets the Down weight producer start early while its activation producer
waits for the preceding kernel to finish. This remains a multi-kernel pipeline.

## Supported contract

| Property | Support |
|---|---|
| Input | E4M3 MXFP8, group-32 UE8M0 scales in the common linear layout |
| Weights | Packed E2M1 MXFP4 or E4M3 MXFP8, group-32 UE8M0 |
| Output | BF16, finalized or raw token/slot-ordered expert rows |
| Activation | `SwiGLU(alpha, beta, limit)` |
| Geometry | H divisible by 128; logical I divisible by 32, padded to 128 |
| Token rows | 1–2 by default; explicitly configurable up to 8 |
| Routing | Packed or unpacked precomputed IDs; valid IDs in `[0, E)` |
| Parallelism | Local expert computation, including TP weight slices; no EP shards or communication |
| Compiler | `nvidia-cutlass-dsl >= 4.8` |

This candidate is omitted from the default backend list. Include a prepared
native backend for larger inputs and for shapes where it wins. Correctness
coverage at T=8 is not a claim that this schedule is faster at T=8. Separate
shared experts, input quantization, router computation, communication, attention
and full decode-forward fusion are outside this backend.

## Use with the unified API

The weight helper accepts **prequantized** canonical tensors, preserving their
bytes. `w1` is `[E, 2I, H/pack]` in **[up, gate]** order; `w2` is
`[E, H, I/pack]`. `pack=2` for MXFP4 and 1 for MXFP8. Weight scales are
`[E, rows, K/32]` UE8M0 bytes. Preparation pads I and rearranges gate/up tiles and
scales; it does not dequantize or requantize the checkpoint.

```python
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions, ExpertConfig, FrostLowLatencyConfig, MoEActivationPack,
    MoEConfig, MoELayer, MoEWeightPack, QuantConfig, QuantFormat, RoutingConfig,
    SwiGLU,
)

weights = MoEWeightPack()
weights.prepare_for(
    "frost_low_latency",
    FrostLowLatencyConfig.prepare_weights(w1_q, w1_sf, w2_q, w2_sf),
)
config = MoEConfig(
    routing=RoutingConfig(num_experts=E, top_k=K),
    quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
    experts=ExpertConfig(intermediate_size=I),  # logical I, before padding
    activation=SwiGLU(alpha=1.0, beta=0.0, limit=10.0),
    backend=BackendOptions((FrostLowLatencyConfig(),)),
)
layer = MoELayer(config)
act = MoEActivationPack(x_q, x_linear_ue8m0_scale, topk_ids, topk_weights)
with autotune(True):
    y = layer(act, weights)  # select tactic and warm up before CUDA Graph capture
```

For cross-backend selection, include e.g. `TrtllmFp4Config()` in `BackendOptions`
and prepare its weight view too. All eligible candidates then use the same
warm-cache latency objective by default, including native runners whose own
default tuning policy uses cold L2. An explicit `MeasurementPolicy` overrides
that common default for all candidates. A missing fallback does not silently choose an
unprepared backend; the layer rejects calls with no eligible candidate. The
same activation pack is reusable across these candidates.
Both candidates must also accept the shared `MoEConfig` geometry. If a native
candidate requires I divisible by 128, prepare both views from the same
explicitly zero-padded canonical weights and use that padded I in their shared
config. A FROST-only configuration can retain the smaller logical I.

Selection is cached by token bucket, routing input mode and eligible candidates.
Changing expert IDs within that bucket does not automatically select a new
backend. Warm up with representative expert reuse and cache conditions; the
benchmark's condition-specific sweeps do not imply a distribution-aware
dispatcher on every decode step.
Enter `autotune(True)` on the layer's first call for the target bucket. Re-entering
that context does not invalidate an existing `MoELayer` backend/tactic winner.
A fresh `MoELayer` re-runs cross-backend selection before graph capture and can
reuse the prepared weight pack, but it may reuse per-runner autotuner records.
To reprofile the same metadata with different routing values, also use fresh
tuning state, for example an isolated process without loading previous tuning
records. Autotuner measurement provenance applies when selection runs, not as
automatic invalidation of this layer-level cache.

The compile cache stores executable metadata. `pack_inputs` owns each call's
scratch tensors, and execution binds current pointers and stream. The packed
routing contract rounds weights to BF16; the unpacked contract preserves BF16
or FP32 weights. Raw outputs preserve this distinction for caller-side combine.
For finalized packed output, BF16 narrowing happens inside the existing combine
kernel. This removes a separate conversion launch while retaining the same
rounding before the ordered FP32 weighted sum. Producers that already supply
BF16 routing weights do not have that launch to remove.

## Reproduce correctness and performance

```bash
pytest tests/moe/test_frost_low_latency.py
python benchmarks/bench_frost_low_latency.py \
  --hidden 5120 --intermediate 1152 --experts 384 --top-k 6 --tokens 1 \
  --include-cutedsl --include-native-default --output first.json
python benchmarks/bench_frost_low_latency.py \
  --hidden 5120 --intermediate 1152 --experts 384 --top-k 6 --tokens 1 \
  --include-cutedsl --include-native-default --reverse --output reverse.json
```

The benchmark prepares all backends from the same canonical weights and uses
the same quantizer, expert IDs, routing-weight inputs and activation parameters.
FROST and TRT-LLM narrow packed routing weights to BF16; the native CuTe DSL
interface retains FP32 weights. Comparisons allow the numerical tolerances below
rather than requiring bitwise equality across those rounding boundaries.
It autotunes each original backend, checks six changing inputs after graph
replay, and verifies outputs again after timing. The default
`--timing-mode latency` brackets each complete call with external event nodes
inside the CUDA Graph. `--timing-mode amortized --cache-mode warm` instead times
128 calls together and divides the replay time by 128. Both retain repeated
original A/A controls. Amortized replay throughput and individual-call latency
are different measurements and can rank the pipelines differently; compare
candidates under the same mode.
Use `--route-pattern disjoint` for distinct experts across tokens or `shared`
for the same top-k set per token. Both modes retain distinct experts within a
token. Each timing graph repeats that routed input; it is not a serving trace.

`--cache-mode cold` sweeps a separate read/write buffer of at least 512 MiB and
twice the device's L2 capacity before each measured call. External CUDA event
nodes inside the graph bracket only MoE, excluding eviction and host enqueue.
The result records the buffer size and queried L2 capacity. Cold and warm
results must be compared separately.

For a selection run, `--retune-cache` ranks every tactic advertised by each
runner with the selected cache and routing pattern, including the native
heuristic and the initial warm-autotuned winner. Every candidate is checked
against the reference. The JSON records candidate timings in `cache_tuning`.
Freeze the selected tactics with the flags below for independent normal and
reverse-order measurements. This enumerates the runner's exposed candidates,
not every configuration its underlying kernel could theoretically implement.
For factorized backends such as CUTLASS and cuDNN grouped GEMM, enumeration
must run inside `autotune(True)` to obtain the stage-ranked candidate pairs;
outside tuning mode their inner tuner returns only a cached or default pair.
Optional `--include-cutlass` and `--include-prims-ts` add the corresponding
original MX backends. `--include-cudnn-grouped` adds the cuDNN grouped-GEMM
backend for `--weight-format mxfp8 --activation-limit 0`; its default SwiGLU
contract cannot replace a clamped-SwiGLU comparison. Requested backends must
pass their architecture, dependency and numerical checks; they are not silently
omitted. `--include-native-default` measures the default of every included
original backend. Pin their measured winners with `--cutlass-tactic`,
`--prims-ts-tactic` or `--cudnn-grouped-tactic`, alongside the existing TRT/CuTe
flags. These options expand reproducible baseline coverage; the tables below
retain their explicitly named original-backend sets.
When a tactic is pinned, initial preparation avoids a redundant full autotune;
the default and requested configuration are still initialized before capture.
`--retune-cache` explicitly requests the candidate sweep even if pins are given.

Ordinary MX SwiGLU in CUTLASS and cuDNN grouped GEMM rounds GEMM1 and
post-activation values to BF16 before intermediate MXFP8 quantization; the
low-latency paths keep FP32 through that quantizer input. By default, all
included backends must pass the same TRT-LLM comparison. To study the former
contracts, explicitly add `--backend-references`: the benchmark computes their
BF16-intermediate references outside timed regions, using the same 1% relative
L2 and 3% normalized-max limits, and records disagreement with TRT separately.
The JSON labels these comparisons as different intermediate rounding. Compare
their absolute kernel times with that qualification; they are not interchangeable
Frost autotuning tactics. Changing-input graph and post-timing checks use each
backend's declared reference, including its default-tactic control.

Prims-TS uses another MXFP8 intermediate scale: `2**(floor(log2(amax)) - 8)`
with saturating E4M3 conversion, rather than `2**ceil(log2(amax / 448))`.
For example, a block maximum of 500 gives scales 1 and 2 respectively; the
former clips its largest value to 448. `--backend-references` models this
source-defined scale with FP32 activation for Prims, again outside timing and
with unchanged tolerances. This is a separate numerical contract, not an
interchangeable Frost tactic or evidence of a kernel correctness failure.
With `--retune-cache`, the benchmark ranks the full advertised Prims tactic
list once; it skips the redundant preliminary autotune. Factorized backends
retain their native per-GEMM search before the cache-specific comparison.

It includes scale conversion and combine. By default the precomputed routing
weights are FP32; use `--routing-weights-dtype bf16` to measure a producer that
already emits BF16 weights. Removing a conversion helps only when that
conversion was required. For MXFP4 with BF16 routing weights, omit
`--include-cutedsl`: that native W4A8 interface requires FP32 routing weights.
This control compares the supported TRT-LLM and FROST paths; it does not insert
an extra dtype conversion for CuTe DSL or claim an exhaustive original baseline.
Weight preparation, input
quantization and routing selection are excluded. Check A/A and cross-process
stability before interpreting a small difference. These are kernel GPU times
at model-relevant shapes, not model throughput measurements.

## Individual-call B300 latency

These paired measurements bracket each complete call with external events
inside a CUDA Graph. They are separate from the historical amortized table
below. H/I/E/K denotes hidden size, physical intermediate size, expert count
and top-k. The current matrix uses H=2048, I=512, E=512 and K=10 unless shown
otherwise; rows with failed controls are omitted. These results qualify the
stated capture configuration. Alternate graph granularities remain under
investigation; no unqualified timing-method change is included.

| H / I / E / K | Weights | T | Cache / expert sets | Best original | Original µs | Frost µs | Speedup |
|---|---|---:|---|---|---:|---:|---:|
| 2048 / 512 / 512 / 10 | MXFP8 | 1 | warm / disjoint | trtllm_default | 23.68 | 13.44 | 1.762×† |
| 2048 / 512 / 512 / 10 | MXFP8 | 2 | warm / disjoint | trtllm_default | 27.74 | 17.55 | 1.581× |
| 2048 / 512 / 512 / 10 | MXFP8 | 4 | warm / disjoint | trtllm | 42.08 | 31.86 | 1.321× |
| 2048 / 512 / 512 / 10 | MXFP4 | 8 | warm / disjoint | cute_dsl | 45.86 | 38.02 | 1.206× |
| 2048 / 512 / 512 / 10 | MXFP8 | 8 | warm / disjoint | trtllm | 62.52 | 54.22 | 1.153×† |
| 2048 / 512 / 512 / 10 | MXFP4 | 1 | cold / disjoint | cute_dsl | 26.62 | 18.43 | 1.444×† |
| 2048 / 512 / 512 / 10 | MXFP4 | 4 | cold / shared | cute_dsl | 22.53 | 22.54 | 1.000× |
| 2048 / 512 / 512 / 10 | MXFP8 | 4 | cold / shared | trtllm | 30.72 | 24.58 | 1.250× |
| 2048 / 512 / 512 / 10 | MXFP4 | 4 | cold / disjoint | cute_dsl | 36.88 | 30.72 | 1.201× |
| 2048 / 512 / 512 / 10 | MXFP8 | 4 | cold / disjoint | trtllm | 51.20 | 40.96 | 1.250× |
| 2048 / 512 / 512 / 10 | MXFP8 | 4 | warm / shared | trtllm | 25.70 | 21.62 | 1.189× |

The original set includes tuned and default TRT-LLM, plus native CuTe DSL for
MXFP4. All selected Frost tactics in this table use Tensor Cores. † The displayed
pair passed; another candidate had unstable timing. Values close to 1× indicate
parity, not a meaningful gain. Cold shared-expert cases demonstrate why a native
fallback remains necessary.

B300/SM103, PyTorch 2.13.0+cu130, CUDA 13.0, CuTe DSL 4.8.0, FI base
`3184bbea`. The corrected device kernels match the current implementation;
tactics were explicitly retuned and pinned with benchmark source `bed1b997e`.
The subsequent production dispatcher fixes are validated separately. See the
[results and control metrics](frost_low_latency_b300_latency.json) for exact
tactics, cache policy, numerical tolerances and excluded cases.

## Historical B300 performance: amortized warm replay

Single B300/SM103, FlashInfer 0.7.2 at base `3184bbea`, PyTorch 2.13.0+cu130,
CUDA 13.0 and CuTe DSL 4.8.0. These are **synthetic MX quantized model shapes**,
not original-checkpoint or model-throughput measurements. TP labels describe
local weight-slice geometry; these rows do not include communication.
The table averages 128 complete calls in a warm CUDA Graph. It does not measure
event-delimited single-call latency. Use the recorded source and commands in
the linked result artifact to reproduce this historical measurement; new
decode-selection results must state which timing mode they use.

| Workload (T=1) | H / physical I / E / top-k | Best measured original | Original µs | FROST µs | Speedup |
|---|---|---|---:|---:|---:|
| DSv4.1-Flash TP2 slice shape | 5120 / 1152 / 384 / 6 | CuTeDSL | 19.51 | 13.68 | 1.425× |
| DSv4.1-Flash TP4 slice shape | 5120 / 640 / 384 / 6 | CuTeDSL | 17.91 | 12.11 | 1.479× |
| Generic MXFP8 | 4096 / 1024 / 64 / 8 | TRT-LLM tuned | 24.70 | 17.66 | 1.398× |
| Generic large MXFP4 | 7168 / 2048 / 64 / 8 | TRT-LLM tuned | 47.80 | 41.83 | 1.143× |
| Qwen3-Next shape / MXFP4 | 2048 / 512 / 512 / 10 | CuTeDSL | 16.49 | 9.22 | 1.787× |
| Qwen3-Next shape / MXFP8 | 2048 / 512 / 512 / 10 | TRT-LLM tuned | 19.14 | 9.37 | 2.043× |
| Qwen3-235B shape / MXFP4 | 4096 / 1536 / 128 / 8 | CuTeDSL | 24.27 | 17.70 | 1.372× |

All rows use MXFP8 activations and BF16 output. The two DS-shaped rows use MXFP4
weights and SwiGLU limit 10. TP4 has logical I=576 with zero padding to 640 for
both backends. Generic rows also use limit 10; the Qwen-shaped rows use standard
SwiGLU. The Qwen source configurations are BF16: only their routed-expert
geometry is used here.

The original comparison includes TRT-LLM tuned and default heuristic; MXFP4 also
includes the existing CuTeDSL backend. Independent selection measurements first
choose the best observed configuration per implementation. Those configurations
are frozen for the reported opposite-order runs; each row uses the fastest
original in those runs. This is not an exhaustive comparison with all backends
or all possible configurations. All per-backend sweep/cross-process and original
A/A spreads are below 1%. All six changing-input graph checks pass in both
processes (relative L2 < 0.01 and normalized maximum error < 0.03).

[Machine-readable results and exact reproduction commands](frost_low_latency_b300.json)
include every compared backend, both process medians, selected tactics and the
numerical/stability controls. Use the commands with the recorded source/compiler
versions: native tactic IDs are version dependent. `--trtllm-tactic`,
`--cute-tactic` and `--frost-tile-k` pin the recorded configurations; omitting them
allows fresh autotuning. `--include-native-default` retains the original heuristic.
Use `--frost-ffma` to pin the FMA tactic for a paired comparison with a pinned
Tensor Core tile. The performance table above records the original Tensor Core
pipeline; it does not attribute those gains to the newly unified FMA tactic.

## Extension opportunities

The same route-wise schedule supports MXFP8 weights without changing the
pipeline. Hidden size, expert count, top-k and logical intermediate size are
parameters; they are not tied to one checkpoint. A model can use it only when
its activation and quantization semantics match this contract. SiTU, BF16 and
NVFP4 need separate implementation and qualification. They are not dtype aliases
for the MX path.

| Extension | What carries over | Remaining work |
|---|---|---|
| Other MXFP4/MXFP8 SwiGLU models | Route addressing, fused activation/quantization, PDL, weight preparation | Qualify the checkpoint's scale conventions, activation parameters and workload |
| SiTU activation | Route schedule, weight loads and Down dependency | Implement and validate its distinct activation arithmetic |
| NVFP4 | Route schedule and dependency design | Group-16 E4M3 scales, global scales, FP4 activation quantization and MMA layout |
| BF16 | Shared FFMA device bodies, route schedule and fusion boundaries | Qualify runtime bindings, activation parameters and rounding; a TC path also needs non-blockscaled MMA |
| More token rows | API and correctness contract | Expert reuse/grouping; the current per-route schedule can lose to native |

The shared BF16 FFMA kernels widen expert indices before stride multiplication.
Device-body regressions use actual 32-GiB-plus FC1 and 16-GiB-plus FC2 weight
tables and replay changing expert IDs and inputs. The old FC1 indexing wrapped
to expert zero in this test. This fixes the shared implementation's addressing;
it does not add BF16 admission to `FrostLowLatencyConfig` or change legacy
activation and combine rounding.

For concrete shape candidates, the published
[Qwen3-235B-A22B configuration](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)
has H=4096, expert I=1536, E=128 and top-k=8; the
[Qwen3-Next-80B-A3B configuration](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
has H=2048, expert I=512, E=512 and top-k=10. Their dimensions fit this schedule.
These configurations declare BF16, so this is a compatibility opportunity for
a separately qualified MX quantized variant, not support for those original
checkpoints or a measured model speedup. Qwen3-Next's shared expert also remains
outside this routed-only backend.
