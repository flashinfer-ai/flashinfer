# SM120 materialized FP8 prefill research integration

This is an explicit local experiment for the SGLang GLM serving workload. It
does not register a public FlashInfer API, alter automatic dispatch, install a
package, or replace the existing absorbed-MLA implementation in this directory.
Upstream ownership and admission are not established; no upstream PR is implied.

The integration requires the matching modified SGLang checkout, CuTe DSL 4.6.0,
PyTorch, Triton, and SM120. Validation uses CUDA 12.9 on an RTX PRO 5000 72 GB.
The existing installed FlashInfer 0.6.15 remains separate from this checkout.

The final local HTTP runs measured **6.45–6.49 QPS at concurrency 9**, mean E2E
**1395.14/1385.80 ms**, and 100% success: **+51.05–51.99%** vs the original
4.27 QPS baseline. Both stop at concurrency 10 on the 1500 ms mean-latency SLA.
These are experimental low-precision results, with quality limits below.

## Explicit entry and files

The experimental directory can be added to the Python module search path;
the SGLang forward-hook factory is `materialized_fp8_hook:create_hook`. It
targets `model.layers.0.self_attn.attn_mha` and installs the prefill replacements
after decode graph capture. `minimum_query_tokens` defaults to 1024. This
document records configuration only, in accordance with the user's request not
to write launch or benchmark commands to files.

- `materialized_fp8_cute.py`: derived from SGLang's CuTe SM80 forward path,
  uses QK/PV FP8 warp MMA, FP32 softmax/accumulators, BF16 output, and shared
  memory reuse. PV K32 byte-transpose addressing and register order follow
  the existing `mla_fp8.cu` reference. The latest candidate uses TMA for Q/K/V
  global-to-shared transfers; the measured 6.11 QPS K32 snapshot used cp.async.
- `materialized_fp8_interface.py`: isolated derivative of the SGLang FA4
  interface, importing only the experimental forward class.
- `materialized_fp8_quantize.py`: query-tile quantization and fused
  K-nope/K-rope concatenation plus K/V quantization; tile counts are runtime
  arguments to avoid recompilation for every request length.
- `materialized_fp8_hook.py`: retains RoPE, KV writes, prefix collection and
  projections; eligible one-shot prefill uses the experimental kernel, short
  and otherwise ineligible inputs retain the original path.
- `tests/experimental/mla_fp8_sm120/test_materialized_prefill.py`: FP32
  mathematical reference, right-aligned causal masks, partial/ragged tiles,
  sharply changing V scales, empty causal rows, and CUDA graph mutation tests.

## Scope and limitations

The serving adapter assumes BF16 source activations, 20 local heads,
K-nope=192, RoPE=64 and V=256. This materialized representation differs from
absorbed MLA's Q/K=576 and latent V=512; kernel timings are not interchangeable.
The adapter supports ordinary one-shot prefill and intentionally excludes
speculative modes, paged attention, local attention and distributed variants.
The copied low-level interface is an internal harness, not a claim that every
signature option or GPU architecture is supported.

Per sequence/head/tile scales reconstruct QK. Changes in V block scale rescale
the running O accumulator together with online softmax. P is multiplied by
448 before FP8 conversion; the final BF16 epilogue compensates this factor.

The original generic CuTe B loader provided 8-byte alignment for K32, below
the 16-byte requirement of `ldmatrix.m16n16.x2.trans.b8`. Explicit source
coordinates use K32 rows and N16-aligned columns; their alignment is preserved
by the shared swizzle. The code does not suppress the generic loader's check.
Shared storage can become BF16 output only after all input reads finish at the
epilogue barrier.

## Initial evidence

On captured layers 0/23/46, Q=K=5093, H=20, D=256, K32 core times were
0.96060/0.96358/0.96571 ms; preparation plus core was
1.11029/1.11364/1.11620 ms. Same-run BF16 preparation plus attention was
1.54498/1.55468/1.55609 ms. CUDA graph event timing excludes service overhead;
relative output L2 vs BF16 was 4.309%/6.006%/5.602%.

The previous K16 fusion produced 5.78 QPS in a service experiment; that result
must not be attributed to this new K32 implementation before another capacity
run. Business quality acceptance remains outstanding. New variants need their
own numerical, sanitizer and service evidence; historical tests do not validate
new source automatically.

## K32 service measurement

A subsequent local HTTP run used case version 202609111724, cache perturbation
target 24%, a KV flush after JIT warmup, 180 seconds per concurrency, and the
unchanged output/sampling protocol. C8: 6.05 QPS, mean E2E 1321.30 ms; C9:
6.11 QPS, 1472.25 ms; C10: 6.20 QPS, 1612.30 ms. All requests succeeded.
C9 is the highest latency-SLA-compliant round; C10 stopped the run normally.
The highest met throughput is 43.09% above the original 4.27 QPS baseline
at C6. This is not a same-concurrency A/B and business quality is unaccepted.

The new tests passed memcheck with zero errors and racecheck with zero errors
and warnings. The copied source was formatted with AST equivalence checked.

## Follow-up TMA candidate

Q/K/V use the 128-byte TMA-compatible `Swizzle(3,4,3)` layout. Three independent
mbarriers track Q, K and V transfers; K/V alternate parity on every reuse.
An entire warp enters CuTe's TMA copy because the generated instruction elects
its issuing lane internally. Barrier arrival is separately elected once.
The CTA barrier before overwriting K also waits for V; the next K transfer
overlaps current softmax/PV. Output reuse remains protected by a CTA barrier.

On the same three captured layers, core time is approximately 0.898–0.900 ms,
and preparation plus core 1.047–1.049 ms. A 14-configuration comparison retains
64×64 tiles. Ragged/prefix/empty-row and graph tests pass; memcheck reports zero
errors and racecheck zero hazards. The same three real outputs are bitwise
equal to the preserved K32 implementation.

The new service run achieved C9: 6.23 QPS, mean E2E 1442.94 ms, 1124/1124
successful; C10: 6.34 QPS, 1576.06 ms, 1143/1143 successful. C10 stopped on
the mean-latency SLA. This is +1.96% vs K32 at C9 and +45.90% vs the original
highest-met 4.27 QPS; the 50% target is still unmet. The service was stopped
after collecting evidence. No new full-model quality acceptance is implied.

## Current instruction/layout candidate

The current files additionally combine PRMT probability packing, approximate
reciprocal of positive V block scales, block reconstruction scales fused into
log2 softmax, and head-major temporary FP8 Q/K/V storage. The quantization
blocks and scale definitions are unchanged; FP32 rounding order can differ.
The BF16 output keeps its original token-major layout.

Twelve captured-layer/prefix shapes passed, with preparation-plus-core time
3.60–4.36% lower than the preceding TMA version. Added relative output L2 is
at most 0.00007997. Ragged/empty-row/graph tests, memcheck and racecheck pass.
These measurements do not establish service throughput or business quality.
The exact 6.23 QPS source is preserved separately in the delivery snapshot.

A subsequent service run of the combined source measured C9: 6.32 QPS,
mean E2E 1424.08 ms, 1138/1138 successful; C10: 6.40 QPS, 1560.59 ms,
1155/1155 successful. C10 stopped on the mean-latency SLA. Highest met
throughput is +48.01% vs 4.27 QPS and +1.44% vs the preceding TMA C9 run.
The 50% target remains unmet, and full-model/business acceptance is separate.

Fixed-reference full-model scoring was subsequently collected for 100 cases,
1460 tokens, before forcing identical reference outputs. NLL is 2.159617 vs
BF16 2.152681; paired case bootstrap 95% CI for the difference is
[-0.017033, 0.030543]. All logits are finite. Decode top-1 agreement is
87.4265%, with mean KL 0.072916. This sample does not show a statistically
significant NLL increase, but does not establish equivalence or business
accuracy. Diagnostic capture disables overlap and is excluded from capacity.

## Causal full tiles, LPT, and short-K projections

The current kernel skips repeated causal comparisons only in iterations already
known to contain complete visible KV tiles. Boundary masks remain active.
It also enables the existing varlen scheduler's LPT order for causal attention,
with a one-byte storage estimate for head grouping. Each CTA keeps the same
arithmetic and writes its original output tile; only CTA order changes.

Twelve captured layer/prefix shapes remain bitwise equal to the instruction/layout
combination, with preparation-plus-core time reduced by 3.75–7.50%. At layer 23,
Q=K=5093, four interleaved samples give 1.012416 → 0.937069 ms including
preparation, and 0.856308 → 0.795241 ms for the core. These are GPU microbenchmarks;
service throughput needs a separate completed capacity run. The final repository
source is AST-equivalent to this candidate; ragged/prefix/empty-row and graph
tests pass, with zero memcheck errors and zero racecheck hazards.

`projection_tuning.py` and `projection_policy.cuh` add an explicit SGLang factory,
`projection_tuning:create_hook`, which installs the prefill hook plus a short-K
FP8 GEMM scheduling policy. The C++ header derives from SGLang's existing SM120
block-FP8 GEMM. M >= 1024 and K <= 2048 retain ordinary scheduling, avoiding
Stream-K reduction overhead. Weight, activation quantization, scale layout,
and output dtype remain unchanged. Small batches and long-K projections use
the existing implementation. The standalone test compares actual projection
shapes with that implementation on SM120.

The preceding policy-only service run reached C9: 6.37 QPS, 1412.40 ms,
1149/1149 successful. QKV-A zero padding was separately tested after repairing
its hook and also reached 6.37 QPS (1410.96 ms); it is not included here because
it showed no additional service throughput. Rejected prototypes remain in the
delivery evidence. The 100-case quality numbers above belong to the earlier
attention combination, not automatically to every subsequent source revision.


The first service run from these repository files reached C9: **6.49 QPS**,
mean E2E **1385.80 ms**, 1171/1171 successful; C10: 6.55 QPS, 1525.03 ms,
1182/1182 successful, stopping normally on the mean-latency SLA. The highest
met throughput is **51.99% above 4.27 QPS** and 1.88% above the preceding
6.37 QPS C9 policy run. C10 does not satisfy the 1500 ms condition. Case
version, 24% cache perturbation target, output limits, sampling, and 180-second
rounds are unchanged; KV was flushed after nine warmup requests. An independent
KV-flush repeat subsequently measured C9: 6.45 QPS, 1395.14 ms, 1162/1162
successful; C10: 6.54 QPS, 1525.25 ms, 1181/1181 successful. Both runs stop
on C10 latency; their highest-met throughput gains are 51.05–51.99%. The repeat
uses the same unmodified process after another nine-request warmup and KV flush.
Final-source fixed-reference scoring is complete: 100 cases, 1460 tokens,
all finite logits and identical reference prefixes. NLL is 2.13990939 vs
2.15268077; delta −0.01277137 with paired case bootstrap 95% CI
[−0.03808404, 0.01323553]. Decode top-1 agreement is 86.1765%, KL 0.07530448.
The interval includes zero: this is not proof of equivalence, improvement, or
business accuracy. Both sides use the same online-NVFP4 routed MoE, so this
comparison does not validate NVFP4 weight quality against the original FP8
checkpoint. All owned capacity and diagnostic services have been stopped.

## Per-item commits (2026-10-10)

Delivery target: sjtushenhai/flashinfer, branch fp8_mla_sm120_dev.
The commit sequence preserves the measured implementation stages; smaller
PRMT/reciprocal/softmax/layout commits are separately reviewable but only have
combined service measurements. Their percentages must not be added.

| Item | Commit |
| --- | --- |
| 已有 SHARD_QK 实验（未计收益） | [6483a18e3a](https://github.com/sjtushenhai/flashinfer/commit/6483a18e3a9cc1f361acaa45a1efd40badbbaf75) |
| 融合 FP8 prefill（K16） | [2d81fdc43f](https://github.com/sjtushenhai/flashinfer/commit/2d81fdc43f4d6b6dce365caa57310f524c3aa93a) |
| PV K32 指令与对齐加载 | [3e25c2fb95](https://github.com/sjtushenhai/flashinfer/commit/3e25c2fb9533843d84b0eb747563eb621304f3f0) |
| TMA 搬运与共享内存 swizzle | [28385bf2f0](https://github.com/sjtushenhai/flashinfer/commit/28385bf2f0ff55c41ac2231ef822b66de43aed6c) |
| PRMT 概率寄存器重排 | [ffad90f628](https://github.com/sjtushenhai/flashinfer/commit/ffad90f628dc96b1a32283fcc1eaf7240c242211) |
| V scale 近似倒数 | [c896ad61cb](https://github.com/sjtushenhai/flashinfer/commit/c896ad61cbbfff907f73ce7ba035a08fe57975f3) |
| log2 在线 softmax 与 scale 融合 | [7cfe4cf382](https://github.com/sjtushenhai/flashinfer/commit/7cfe4cf3821de2fed7414fe936ee602935058090) |
| head-major 临时 QKV 布局 | [d27368d7b6](https://github.com/sjtushenhai/flashinfer/commit/d27368d7b6cd8f58aaf8b3f73adfde205dbd96b4) |
| 短 K dense FP8 GEMM 调度 | [f05215d47a](https://github.com/sjtushenhai/flashinfer/commit/f05215d47a4208011553fe5a93af0725d8bfd6fa) |
| 完整块 mask 与 LPT 调度 | [438ee55eec](https://github.com/sjtushenhai/flashinfer/commit/438ee55eecc598a842178f122429a8c3d05a7b10) |

All final runtime and test files match the previously validated workspace
byte for byte. The paired SGLang implementation is based on b1478c4293,
with SM120 online NVFP4 support, FA4 one-shot prefix attention and the opt-in
native FP8 decode changes. A newer SGLang base is not covered by these results.

Historical final-source numerical/sanitizer and projection validations are
retained under tests/experimental/mla_fp8_sm120/results/. Recorded absolute
paths identify the original test workspace; the included hashes identify
the matching source files. This commit-only delivery does not rerun GPU tests
while an existing SGLang service occupies the device. Every changed Python
revision in the split sequence passed AST parsing, and the final diff passed
whitespace checks. Intermediate code splits do not imply new performance runs.

## Commit identity correction

Author and committer are sjtushenhai <1730536718@qq.com>.
This includes the two earlier SM120 research commits. The links above use the
rewritten commit IDs; runtime code and historical validation results are unchanged.
