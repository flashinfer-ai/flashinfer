# Experimental MQA TopK

`FP8MQATopKWrapper` accepts contiguous K and scales for selective prefill.
It uses `backend="cutedsl"`, the only supported backend.

Selective prefill accepts
H4/H32/H64, D128, TopK 512/1024/2048, compression ratio 1, 2 or 4, and FP8,
MXFP8 or MXFP4 inputs. Full-domain prefill supports a single request and at
most 1,048,576 keys after pooling; decode is unsupported. Candidate consumption
allows 1,048,576 + max_chunk_size keys in the source pool. Q capacity must be a
multiple of 64, 8 or 4 for H4, H32 or H64 respectively. Output indices are
request-local and padded with -1; their order is unspecified.

FP8 uses E4M3FN Q/K and FP32 per-key scales. MXFP8 uses E4M3FN Q/K and
UE8M0 scales per 32 elements. MXFP4 packs two values per byte and uses the
same scale granularity. For block-scaled input, pass Q scales as uint8[R,H,4]
or packed int32[R,H], and K scales as uint8[N,4]. Block-scaled K storage has
a multiple of 256 rows; the plan describes the live extent. FP16, BF16 and
FP32 logits storage and FP32 or BF16 weights are supported by selective.
`native_bf16` and `native_fp16` select native head accumulation independently
of FP32/FP16/BF16 logits storage. `relu_strict` uses FP32 head accumulation. The selective default is `relu_fast` for FP8 and
MXFP4, and `relu_strict` for MXFP8. Candidate publication and consumption
default to `relu_strict` and reject explicit `relu_fast`. MXFP8 also rejects
`relu_fast`.

`model_config` may select a named model policy. Its shape must match the
wrapper configuration. Qwen's fixed-weight and unit-scale numerical contract
applies only when `model_config="Qwen3.8-Flash-Next"` is explicitly supplied.
Other inputs with the same shape retain the general numerical contract.

> [!NOTE]
> Kimi-K3.1-Pro and GLM-5.3 share H32/D128/K2048/ratio-1. Shape inference
> selects GLM-5.3. Set `model_config="Kimi-K3.1-Pro"` for the Kimi-specific
> token-scaled FP8 `relu_strict` policy. Other encoding and mode combinations
> use the same policy as GLM-5.3.

Construct the wrapper outside CUDA graph capture, plan each step and prepare
every layer pool before capture. Calls sharing a wrapper must be serialized;
the caller owns its graphs and keeps their inputs alive. Selective trace export is unsupported.

The selective tests cover independent selection oracles, supported encodings,
padded capacity, planning, graph replay and lifetime behavior.

## DS-V4.1 candidate workflow

DS-V4.1 ordinary layers use H32, D128, TopK 512 and compression ratio 2.
Candidate publication and consumption use ratio 1 with
`model_config="DS-V4.1-Flash"` and `qkv_dtype="mxfp4"`. All three paths
support FP32/FP16/BF16 logits combined with FP32 (`relu_strict`),
native BF16 (`native_bf16`) or native FP16 (`native_fp16`) head reduction.
Kimi-K3.1-Pro full-domain indexing supports the same nine combinations.

> [!NOTE]
> `logits_dtype` defines score ordering, including conversion ties, but does
> not always select internal storage. Kimi FP8 `relu_strict` uses FP32;
> full-domain DS-V4.1-Flash MXFP4 `native_bf16` uses BF16. Budget scratch using
> the effective dtype, exposed by the raw engine as `effective_logits_dtype`
> alongside `requested_logits_dtype`. An FP32 tensor uses twice the bytes of
> an equally sized FP16/BF16 tensor; total workspace need not double.
> These full-domain storage rules do not apply to candidate consumption.

- Set `publish_candidate_blocks=True` to produce token TopK and candidate
  blocks. After `plan`, pool preparation and `run`, read `candidate_blocks`
  and `candidate_valid_lens`. `run` still returns token indices.
- Set `candidate_capacity=2048` to consume blocks through
  `run(..., candidate_blocks=blocks)`. Publication and consumption cannot
  be enabled on the same wrapper.

Candidate blocks are contiguous CUDA int32 `[Q, 2048]`, with eight keys per
block. Each row's live prefix contains `min(ceil(visible_keys / 8), 2048)`
sorted, unique block IDs, including the latest visible block; the remaining
entries are INT32_MAX. IDs address the supplied K pool. Consumers require
these device-value conditions from the caller; publishers produce this format.
K1 checks range, order, uniqueness, the latest block and padding before
reading candidate KV on every run, including graph replay. Invalid active rows
return -1 and set the `0x20` flag in `repair_error`, without host synchronization
or a separate validation launch.
`candidate_valid_lens` contains live **token counts**, not block counts.

Candidate contents and causal bounds may change between graph replays within
the prepared capacities. `input_buffers()["candidate_blocks"]` exposes stable
consumer storage. An aligned table of the prepared shape, dtype and device can
also be supplied to `run` during capture. Keep it alive for replay and use
storage separate from the consumer wrapper’s other buffers.
Published buffers are shared across calls; consume or copy them before the next
publisher call. Candidate selection returns request-local token indices, not
positions in the candidate table.

## User contract

Calling `FP8MQATopKWrapper` is explicit opt-in and emits `ExperimentalWarning`.
It is not chosen by automatic routing; this API accepts `backend="cutedsl"`,
not `backend="auto"`. It supports batch-1 prefill on SM100/SM103, H4/H32/H64,
D128, TopK 512/1024/2048, and compression ratios 1, 2 or 4. The full-domain
capacity limit is 1,048,576 keys after pooling; candidate consumption allows
an additional `max_chunk_size` keys in the source pool. A table produced by this
wrapper is limited to the publisher capacity. A consumer can also use an externally
supplied table over a larger source pool.

`next_n` optionally asserts an exact query-row count. `host_length_hints`
provides optional host upper bounds and failures set error bit 2. Read `error`
and `repair_error` outside capture; fast-mode or Qwen numerical violations
set bit 16 in `repair_error`. Retired modules drain at `plan`, `prepare_pool`
and `clear_pools`, except while the caller's stream is capturing. Do not call
these methods while any other thread is in global capture.

### Workspace and capture

Pools on the same wrapper share workspace. Serialize their calls. A captured
`run` binds the supplied Q extent: capture another graph for a different
`num_query_rows`. The live prefix can change within that extent. Keys tied at the
TopK boundary may yield different indices with the same selected score multiset.

A candidate consumer reserves 8 KiB per query row for the stable int32
candidate table, including when a full aligned caller table is used directly.
Its two largest additional buffers are:

| Consumer path | Candidate storage per row | Recovery scores per row |
| --- | ---: | ---: |
| Packed native BF16 with BF16 weights | 8,224 × 4 bytes | 16,384 × 2 bytes |
| Typed | 4,096 × 8 bytes | 16,384 × sizeof(score dtype) |

At 32,768 rows these two buffers together occupy about 2 GiB in the packed
path and 3 GiB in the typed FP32 path. The staging table adds 256 MiB.
These figures exclude samples, outputs, other scratch and allocator overhead.

## Publication workspace and bounds

The raw publisher limits its output to 65,535 query rows. The wrapper additionally
requires an H32 query capacity divisible by eight, so its maximum is 65,528.
Live query counts may be smaller than that prepared capacity.
Block scores occupy `Q * ceil(N / 256) * 32 * sizeof(effective_logits_dtype)` bytes
(the key dimension is padded to 256)
in addition to the candidate/recovery arena and other metadata. For Q=8192
and N=1,048,576 this is 4 GiB with FP32 or 2 GiB with FP16/BF16. The engine's
`workspace` tensor alone is not a total-memory or peak-memory measurement.
The block scores remain live through K3 block selection and are allocated
separately from the token candidate arena.

Candidate validation status is sticky across layers in a step: `repair_error`
bit 0x20 is cleared by `plan()`, not by preparing another pool. Pool warmup
uses inactive candidate bounds and does not validate a stale candidate table.
Model tuning uses prepared capacity so launch templates remain fixed during
CUDA Graph replay; a short live step may therefore use a large-capacity template.
