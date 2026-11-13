# Backend implementation notes

Read the relevant contract before changing dispatch, captured-buffer lifetimes,
or dtypes. General workflow rules are in [AGENTS.md](../../AGENTS.md).

- [cuDNN linear attention](#cudnn-linear-attention)
- [cuDNN query sparse attention](#cudnn-query-sparse-attention)
- [Attention plans and CUDA graphs](#attention-plan-and-cuda-graph-lifetime)
- [Grouped FP8 GEMM](#grouped-fp8-gemm)
- [SM90 FP4 MegaMoE](#sm90-fp4-megamoe)
- [Prims-TS decode dtypes](#prims-ts-decode-dtypes)

## cuDNN linear attention

Use `nvidia-cudnn-frontend>=1.32.0`; automatic selection also requires CuTe DSL
4.7+. KDA/GDN normalization adds `1e-6`, without an epsilon argument. KDA's
SM100/SM103 policies live in `flashinfer/kda.py`; GDN's SM100 policy lives in
`flashinfer/gdn_prefill.py`. See [KDA shape regions](../api/kda.rst).

Auto-selected cuDNN ignores the native KDA workspace; native fallback uses it.
Explicit state-pool calls require unique, in-range slots, while automatic
selection keeps pools on their native route. Validate outputs and final states
when changing these policies.

Use `benchmarks/bench_cudnn_linear_attention.py` for public-call comparisons.
Its raw-gate GDN baseline includes Torch transforms, unlike a vLLM fused-producer
baseline. Report host CPU, completed wall, and reset-inclusive CUDA graph times
separately.

## cuDNN query sparse attention

Both APIs take BF16 Q/K/V. `flashinfer.qsa_forward` selects blocks with a packed
`uint8` QSA block bitmap; `flashinfer.qsa_indexed_forward` accepts existing
causal token IDs arranged in non-overlapping four-slot groups (a complete block,
four `-1` padding slots, or a partial causal tail followed by `-1`). These
device-value rules are a caller contract and are not checked on the host; see
the docstrings in `flashinfer/qsa.py`. Both call
`cudnn.ops`, return BF16 output, FP32 output, and natural-log LSE, and do not
register autograd. Keep imports lazy and preserve these input/output contracts.

Validate dispatch changes during CUDA graph replay. Indexed forward is not
bitwise repeatable because index compaction can change accumulation order.
On cuDNN Frontend 1.32, cover the bitmap boundary at 512/513 tokens.

## Attention plan and CUDA Graph lifetime

FlashInfer requires cuDNN Frontend >=1.32.0. Attention code can rely on its
cumulative-length graph arguments, shape-override constructor, short-cache
placement, and CUDA-graph replay hint. Do not retain compatibility branches for
older FE releases. Backend versions and native/runtime capabilities remain
separate: token offsets require backend 9.24 (mixed masks/FP8 require 9.25), and
the single-token GQA Stats bug is guarded until 9.27. Keep checks that ensure
both backend and FROST remain usable through the common graph contract.

### Buffer ownership and graph bounds

Old captures can retain a previous provider's metadata pointers after `auto`
changes providers. Keep wrapper-owned tables and length buffers alive and
refresh them in `plan()`, even when the page table is caller-owned. `run()` must
not rebuild or repack metadata. Replay only with compatible buffers and lengths
within the capture's declared bounds.

A reserved table can exceed a capture's KV range. Declare only that range,
preserving its storage row stride and pointer. Intersect the explicit KV maximum
with caller-table capacity; use the same bound for the attention attribute and
table slice. Device lengths still mask live tokens without host readback.
Capacity must cover initialized indices and rounded graph bounds.

Test changed page IDs and lengths, replacement by a caller table of another
width, and Q-partition changes after a caller-owned first plan. Shortening a KV
prefix alone can hide stale metadata. Include fallback beyond cuDNN admission
and a backend-only provider control: FROST can accept oversized descriptors the
backend rejects. Provider controls need an isolated real graph cache and a
selected-engine check. Keep lifetime/numerical tests independent of heuristic
winners and timing thresholds.

### Ragged lengths and masks

Explicit paged cuDNN plans pass `window_left` through the common graph contract.
The prepared and scaled execution paths use the same window. A value of `-1`
disables it. For a nonnegative W, let p = query_index + KV_length - Q_length
denote the bottom-right aligned query position. With `causal=True`, eligible keys
range from p-W through p. With `causal=False`, the window only excludes keys
below p-W; keys above p remain eligible up to the valid KV length. Both cases
also respect the valid sequence bounds. The low level converts W to cuDNN's W+1
inclusive left bound.
The window remains a plan-time attribute and cannot change in a retained capture.
Paged automatic admission still excludes windows pending separate qualification.

Valid attention lengths differ from physical ragged offsets: `seq_lens_q` can
be shorter than its slot. Infer lengths from adjacent offsets only when omitted.
Keep token-offset units consistent in graph descriptors and override shapes,
regardless of whether masks use lengths or cumulative lengths.

From the first graph-mode cuDNN plan, stage both mask lengths into owned
storage, even when inferred. Replan with fresh caller tensors, alternating
Q-only, KV-only, both, and omitted lengths. Replay the original capture against
a compacted reference; checking only the latest eager call misses stale
bindings and mask-representation transitions.

### FA2 scheduling and runtime scales

Standard FA2/FA3 paged adapters reject `sinks` instead of silently dropping the
tensor. This also applies to tensor-core decode when it uses the paged prefill
adapter, including a fallback from another provider. Custom sink JIT modules
retain their existing positional-argument interface; their generated kernel
must actually include the sink variant.

FA2 captures bind CPU-generated output/merge prefixes as well as indptr tensors.
Updating indptrs after switching to cuDNN leaves schedules stale when requests
exchange query rows. In graph mode, keep FA2 once planned; eager auto selection
can still change providers. Test FA2-first and cuDNN-first sequences with changed
Q partitions and page IDs.

Preserve runtime calibration scales for half-precision and FP8 inputs. Test
non-unit Q/K scales with nontrivial logits, V scaling of O with unchanged LSE,
and GPU scalar updates before replay. Plan-time predicates cannot exclude
arguments supplied only to `run()`.

### Admission without host readback

Declining cuDNN can make FA2 stage KV metadata back to the host. Use per-request
work for admission only when its lengths are already on CPU; otherwise retain
the bound-based policy. Check with a busy GPU stream and synchronization
debugging, as well as predicate-level tests.

Full-prefill admission requires each request's Q length to equal its KV length;
equal batch maxima do not prove this. Use existing CPU metadata to establish
that condition. Decline the extension if independent query-length overrides or
device-only KV lengths prevent proof. Preserve the existing prefix policy
without introducing a dispatch readback.

SM100 BF16 paged D128 auto retains its original bounded-prefix/full domains.
With existing CPU Q offsets and KV lengths, it additionally admits larger
batches and longer prefill at `num_qo_heads * sum(q_i * kv_i) >= 16 Mi`.
Full prefill requires every `q_i == kv_i`, with maximum Q in 1024–16384.
Other admitted prefixes require every Q length above 128, maximum Q at most
8192, and maximum KV in 2048–32768. HND/page16 and the existing head/dtype,
feature and runtime guards still apply. Short requests mixed with long ones
do not qualify for this extension merely because their declaration is large.
Independent Q overrides and device-only metadata keep the original policy.

### SM100 D256 paged GQA admission

The BF16 HND/page-16 prefix extension requires cuDNN 9.24+, FE 1.33+,
and native packed-split and physical-origin capability flags for the requested
width. It uses the same public cuDNN graph contract and never forces an engine.
D128 admission is independent of this extension; D64 remains on the native route.

Admission requires CPU query offsets and KV lengths, plus CPU query masks when
provided. Device-only metadata retains the native route without an admission
readback. Compute work from paired actual lengths, `Hq * sum(Qi * KVi)`;
physical slots and maximum lengths must not overestimate this work. The floor
is 16 Mi. The measured domain is batch 1–4,
Q 64–1024, KV 2048–32768, 8–64 query heads, and GQA groups 2/4/8/16, with each
KV length at least four times its query length. Every request must stay in the
same Q<=128 or Q>128 class as the declared maximum. Small-work and mixed-class
cases remain on the native route.

Test the automatic route with strided Q, independent actual query masks,
physical token origins, changed lengths/page tables and retained graph replay.
The version and native flags are necessary dependencies; benchmark evidence
from a development FE build does not establish released-wheel qualification.

## Grouped FP8 GEMM

`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous` is the standalone
CuTe-DSL API; `group_deepgemm_fp8_nt_groupwise` retains its DeepGEMM-only
signature. Both use per-row expert indices, unlike the indptr-based
`group_gemm_fp8_nt_groupwise` API.

Use `python benchmarks/bench_grouped_fp8.py --production-shapes` to compare
DeepGEMM/CuTe-DSL with preallocated outputs. `--cache-probe` separates first-call
and new-token-count compilation. CuTe-DSL reuses kernels across token counts
with the same 128-row alignment class, at most two variants per device/weight
shape. Concurrent misses share compilation; warm hits avoid the compile lock.

With `skip_check=False`, `validate_indices=True` checks routing values for
non-empty inputs and synchronizes with the CPU. Validate new routing data before
capture. The default checks metadata only; `skip_check=True` bypasses both
metadata and routing-value checks, even when `validate_indices=True`.

## SM90 FP4 MegaMoE

For SM90 NVFP4 fused/split and MXFP4 fused MegaMoE, use
[bench_megamoe_fp4_sm90.py](../../benchmarks/bench_megamoe_fp4_sm90.py) for
timing and [test_megamoe_fp4_sm90.py](../../tests/moe/test_megamoe_fp4_sm90.py)
for correctness checks.

## Prims-TS decode dtypes

Keep storage (`q_dtype`, `k_dtype`, `v_dtype`) separate from MMA precision
(`qk_dtype`, `pv_dtype`). Transformed FP8/NVFP4 K/V uses Q's compute dtype;
BF16 Q/K with FP8 V uses FP8 PV without that transform. Kernel predicates,
P packing, and attention-sink scaling follow compute precision; TMA descriptors
and allocations follow storage precision.

`BatchDecodePagedTSWrapper.plan()` accepts `k_dtype`, `v_dtype`, and the common-KV
alias `kv_dtype`. `BlockSparseTSWrapper.plan()` instead accepts `k_data_type`,
`v_data_type`, and the common-KV alias `kv_data_type`. Use the names for the
specific wrapper.
