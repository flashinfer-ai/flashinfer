# MiniMax M3 paged index selection

This API replaces index scoring, partial top-k, and top-k merge. It excludes
attention-output merging. The vLLM consumer must resolve all three public APIs from flashinfer.msa_ops:
msa_index_decode, msa_index_decode_supported, and
msa_index_decode_warmup. Older libraries and unsupported calls retain
the framework stock implementation.

## Dispatch contract

The JSON allowlist is flashinfer/msa_ops/msa_index_decode_workloads.json: Hopper SM90,
B8/B16 single-token decode, one BF16 head of width 128, 128-token pages,
128-column int32 table, int32 lengths/output, topk16/init0/local1, and host bound
through 16384. Input/output inner strides must be one, lengths contiguous,
devices identical, and score_out absent. Output can be an oversized buffer.
The kill switch is FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1, evaluated on each
call; FlashInfer autotuning also bypasses specialization.

For bounds <=2048 a CUDA kernel writes the visible logical prefix and -1 padding.
For conservative FULL bounds, device-side short-row branches skip score/sort work
while preserving the stock algorithm for longer rows. The selected int32 multiset
and trailing padding match stock; the valid prefix may be permuted. There is no
dtype downcast, reduced accumulation, approximate math, host tensor read, or graph
mode change.

## Compilation and graph readiness

There are four kernel definitions: one dynamic-batch CUDA kernel and three Triton
kernels. The CUDA JIT key is msa_index_decode_sm90, with SM90a and
--fmad=false; the process module cache is keyed by device. Triton compiles stock
and device-gated variants. Readiness keys include device, B, Q/cache/table/output
strides, pointer alignment modulo16, and host bound. Thus four definitions are NOT
a claim of four compiled variants. B8/B16 and layout/bound combinations can create
additional Triton signatures.

The public warmup runs before capture on the actual layout and prepares both stock
and gated variants at bound16384, plus the current larger bound where needed.
Capture does not query device properties or compile. Cold/unready callers retain
stock. Introspection _msa_index_decode_stats reports CUDA module count,
four definitions, chain signatures, dispatch counts, and precompile errors.

## Evidence scope

The measured feature revision is 806c7a4d, based on FlashInfer main 188bdd76
(version 0.7.1). Validation used the vLLM 0.31.0 container with a framework-side
consumer integration, MiniMax-M3-NVFP4, and eight H200s with TP8+EP.

H200 kernel validation passed 36 correctness cases. Six production FULL-graph
rows improved from 13.39 to 6.94 microseconds (1.93x). A same-node model comparison
with synthetic 1024-input/256-output requests measured throughput gains of 2.25%
at concurrency 8 and 1.78% at concurrency 16; p99 normalized interactivity
improved 1.86% at both. Paired accuracy gates passed, including 19/20 GSM8K
answers on both sides. See the PR description for the protocol, finite accuracy
scope, and original first-trial tail regression that follow-ups did not reproduce.

Subsequent commits update test registration, package-data metadata, and
documentation, and relocate the helper into flashinfer.msa_ops. Review fixes add a query-device guard to the stock fallback and
an order-insensitive trace checker with regression tests. Kernel algorithms and
the optimized dispatch path are unchanged; no additional model measurement is
claimed for these descendants.
