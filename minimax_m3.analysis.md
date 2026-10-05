# MiniMax M3 paged index selection

This API replaces index scoring, partial top-k, and top-k merge. It excludes
attention-output merging. The vLLM consumer must resolve all three public APIs:
minimax_m3_index_decode, minimax_m3_index_decode_supported, and
minimax_m3_index_decode_warmup. Older libraries and unsupported calls retain
the framework stock implementation.

## Dispatch contract

The JSON allowlist is flashinfer/minimax_m3_workloads.json: Hopper SM90,
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
kernels. The CUDA JIT key is minimax_m3_index_decode_sm90, with SM90a and
--fmad=false; the process module cache is keyed by device. Triton compiles stock
and device-gated variants. Readiness keys include device, B, Q/cache/table/output
strides, pointer alignment modulo16, and host bound. Thus four definitions are NOT
a claim of four compiled variants. B8/B16 and layout/bound combinations can create
additional Triton signatures.

The public warmup runs before capture on the actual layout and prepares both stock
and gated variants at bound16384, plus the current larger bound where needed.
Capture does not query device properties or compile. Cold/unready callers retain
stock. Introspection _minimax_m3_index_decode_stats reports CUDA module count,
four definitions, chain signatures, dispatch counts, and precompile errors.

## Evidence scope

This is a feature-only forward port onto FlashInfer main (0.7.1). The CUDA and
Triton kernel bodies are unchanged from the release-based integration in Draft
!1941. That release head has H200 and model-level evidence, but those results
do not establish performance or accuracy of this main-based package.

Actual-main package installation, shipped tests, consumer dispatch/fallback,
and the artifact harness are checked on Ballast H100 NVL before publication.
The corresponding receipt identifies the exact source revision and dependency
versions. H200 kernel benchmarks, model performance A/B, and paired model
accuracy must be repeated on this main-based head before promotion.
No main-head end-to-end speedup is claimed here.
