# Experimental MQA TopK

`FP8MQATopKWrapper` accepts contiguous K and scales for selective prefill.
It uses `backend="cutedsl"`, the only supported backend.

Selective prefill accepts
H4/H32/H64, D128, TopK 512/1024/2048, compression ratio 1 or 4, and FP8,
MXFP8 or MXFP4 inputs. Selective prefill supports a single request and at most
1,048,576 keys after pooling; decode is unsupported. Q capacity must be a
multiple of 64, 8 or 4 for H4, H32 or H64 respectively. Output indices are
request-local and padded with -1; their order is unspecified.

FP8 uses E4M3FN Q/K and FP32 per-key scales. MXFP8 uses E4M3FN Q/K and
UE8M0 scales per 32 elements. MXFP4 packs two values per byte and uses the
same scale granularity. For block-scaled input, pass Q scales as uint8[R,H,4]
or packed int32[R,H], and K scales as uint8[N,4]. Block-scaled K storage has
a multiple of 256 rows; the plan describes the live extent. FP16, BF16 and
FP32 logits storage and FP32 or BF16 weights are supported by selective.
`native_bf16` and `native_fp16` select native head accumulation and require
matching logits storage. The selective default is `relu_fast` for FP8 and
MXFP4, and `relu_strict` for MXFP8. MXFP8 rejects `relu_fast`.

`model_config` may select a named model policy. Its shape must match the
wrapper configuration. Qwen's fixed-weight and unit-scale numerical contract
applies only when `model_config="Qwen3.8-Flash-Next"` is explicitly supplied.
Other inputs with the same shape retain the general numerical contract.

Construct the wrapper outside CUDA graph capture, plan each step and prepare
every layer pool before capture. Calls sharing a wrapper must be serialized;
the caller owns its graphs and keeps their inputs alive. Selective trace export is unsupported.

The selective tests cover independent selection oracles, supported encodings,
padded capacity, planning, graph replay and lifetime behavior.

## User contract

Calling `FP8MQATopKWrapper` is explicit opt-in and emits `ExperimentalWarning`.
It is not chosen by automatic routing; this API accepts `backend="cutedsl"`,
not `backend="auto"`. It supports batch-1 prefill on SM100/SM103, H4/H32/H64,
D128, TopK 512/1024/2048, and compression ratios 1 or 4. The capacity limit
is 1,048,576 keys after pooling.

`next_n` optionally asserts an exact query-row count. `host_length_hints`
provides optional host upper bounds and failures set error bit 2. Read `error`
and `repair_error` outside capture; fast-mode or Qwen numerical violations
set bit 16 in `repair_error`. Retired modules drain at `plan`, `prepare_pool`
and `clear_pools`, except while the caller's stream is capturing. Do not call
these methods while any other thread is in global capture.
