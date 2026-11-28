# SM120 NVFP4 decode defaults and integration validation

This publication updates the previous FlashInfer `72a6079e` integration with
standalone kernel `f6f6b5ae0f7d16fc7856d1af1d633d421590f44d` and the qualified
FlashInfer GIN host/graph adaptations. The delivery cache ABI is
`2814542011117`. It does not publish SGLang adapters or benchmark harnesses.

## Default selection

T means compile-time tokens per EP rank, not global serving concurrency.
The rules are continuous intervals, including both endpoints. They require
the calibrated model geometry (H=4096, gate/up intermediate=4096;
DSV top-k=6/E=256, GLM top-k=8/E=288), clamp=10, DP=TP=1 at the MoE
kernel boundary, owner_bf16 numerical permission, default build/JIT eligibility,
and the qualified 110-SM topology. EP4 is same-NUMA; EP8 has three local and
four cross-NUMA peers per rank. Explicit overrides and tuning legality are
checked first. Other shapes/contracts retain ordinary selection.

| Model | EP | M128 K1/K2 range | Tile M/N | K1/K2/TX/RX SMs | Utility |
|---|---:|---|---|---|---|
| DSV4-Flash | 4 | T2–128 | 128/16 | 80/30/0/0 | on |
| GLM5.3-Flash | 4 | T2–95 | 128/16 | 80/30/0/0 | off |
| DSV4-Flash | 8 | T1–256 | 128/32 | 56/38/8/8 | on |
| GLM5.3-Flash | 8 | T1–128 | 128/32 | 64/30/8/8 | off |

EP8 resource allocations also apply through T512 for both models. Above the
M128 ranges, ordinary tiles remain and utility is off. EP4 M128/N16 K2 uses
eight compute warps and two serial N8 accumulator groups; its launch has
352 physical threads including producer/scheduler warps. Queue and workspace
geometry follow the selected tiles. EP4 tail reclaim remains enabled in the
qualified M128 policy.

DATA TMA hints are selected separately; scale TMA retains its default policy:

| Model | EP | T range | Weight DATA | Activation DATA |
|---|---:|---|---|---|
| DSV | 4 | 1–16 | EVICT_FIRST | EVICT_LAST |
| DSV | 4 | 17–128 | default | EVICT_LAST |
| GLM | 4 | 1–8 | EVICT_FIRST | EVICT_LAST |
| GLM | 4 | 9–95 | default | EVICT_LAST |
| DSV | 8 | 1–16 | EVICT_FIRST | default |
| DSV | 8 | 17–64, 257–512 | EVICT_FIRST | EVICT_LAST |
| DSV | 8 | 65–256 | default | EVICT_LAST |
| GLM | 8 | 1–16 | EVICT_FIRST | default |
| GLM | 8 | 17–64, 129–512 | EVICT_FIRST | EVICT_LAST |
| GLM | 8 | 65–128 | default | EVICT_LAST |

These are cache eviction hints, not additional prefetches. Outside the admitted
intervals, ordinary selection remains authoritative. Historical utility profiles
are a separate fallback; these tables describe the continuous decode policy.

## Dependencies and integration

The qualification environment used CUDA 13, CuTe DSL 4.8.1 and pinned NCCL
2.32.3, with `NCCL_CUMEM_ENABLE=1`. Cross-NUMA deployment additionally requires
working GIN/RDMA and the appropriate device-to-HCA mapping. The optional
`MEGAMOE_GIN_HCA_MAP` is indexed by visible CUDA device, not EP subgroup rank.
The package does not silently fall back from an invalid transport deployment.

The GIN artifact manifest authenticates the host binary, host/device source,
device bitcode and NCCL library. The source is preserved together with its
qualified binary. Runtime transport end, finalizer and K3 retain their stream
ordering. No synchronization is removed for these defaults.

## Validation scope (2026-11-15/16)

- On the isolated publication tree, 76 CPU tests and 103 subtests passed:
  FlashInfer host selection, continuous decode intervals, EP8 resource policy,
  utility policy and the four gin/gin_hybrid bridge tests. These run with no
  visible GPU and import the publication tree itself.
- Earlier EP4 and EP8 GPU checks used both model shapes, uneven per-rank rows,
  changing compile buckets and 20 outer CUDA Graph replays per bucket. Finite
  output, eager/replay byte equivalence and Green Context binding passed.
- Final same-host serving qualification covered 27 EP4 points and 36 EP8
  points across MegaMoE DEP, native DEP and pure TP. EP4 GLM covers CC4/8/28;
  DSV and the EP8 campaigns cover CC4/8/28/64/128/256.
- This publication reuses those GPU results for runtime files with identical Python ASTs (formatting-only publication
  cleanup is recorded separately);
  it does not claim a new GPU run, independent arithmetic-reference test,
  independent transport test or semantic-quality evaluation. Historical
  arithmetic qualification of the older integration remains historical.

The numerical contract remains NVFP4 x NVFP4 with E4M3 block scales, FP32
accumulation and BF16 output. owner_bf16 permits rounding owner partials before
final accumulation; it is not bit-identical to ordinary_topk reduction.

Serving throughput includes target and draft execution and is not a kernel-only
speedup. DSV used the checkpoint MXFP4 DSpark draft and fake-prefill 8K/1K,
native client throughput including ramp/tail and request-averaged TPOT. GLM
used its BF16 NEXTN draft and semantic 8K/1K, with three qualified windows of
150 full-batch decode cycles. These two metric definitions must not be mixed.

The selected policy is not uniformly faster than native DEP or pure TP.
For example, on the same SM120 host, DSV EP4 CC64 measured 4997.05 token/s
with MegaMoE versus 3724.54 token/s with native DEP4, while CC128 measured
6024.82 versus 6125.03 token/s. In EP8, MegaMoE was slower than native DEP8
at DSV CC4–128 and all tested GLM points; DSV CC256 improved. These are
serving comparisons, not a matched before/after kernel comparison.

GLM EP4 uses CAP32 after TP CAP128 failed to start; MegaMoE also needed a
warm-cache retry after cold graph initialization OOM. Successful warmed
measurements do not establish cold-start reliability. DSV pure TP4 CC256 had
sampled active occupancy at most 170 requests, so submitted concurrency is
not proof of full device residency. Clocks/power were recorded but not locked
identically. GPU process and cache checks are sampled evidence.

The source/delivery manifests and final measurement qualification are archived
with campaigns `sm120_nvfp4_e2e_latest_m16_20261115`,
`sm120_nvfp4_dep8_serving_m9_20261115` and
`sm120_nvfp4_dep4_serving_m9_20261116`. Runtime, evidence freshness, telemetry,
cache and exact-wave qualification was applied to the final accepted results.
Private SGLang compatibility patches and checkpoint data are not in this commit.
