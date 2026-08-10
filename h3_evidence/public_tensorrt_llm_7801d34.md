# H3 exact public TensorRT-LLM FMHA ABI receipt

This branch adapts FlashInfer commit
`7f33b20840d00f5521400d23e3a75c62056e94dc` to the public TensorRT-LLM
source commit `7801d34fcffbefcb42781765191e2dc0637a08ce` for the H3 Ulysses-8
BF16 and SageAttention context kernels.

## Kernel parameter ABI

Relative to the pinned FlashInfer header, TensorRT-LLM's
`trtllmGen_fmha_export/KernelParamsDecl.h` inserts these fields:

- `float const* ptrDsv4InvRopeCosSinCache` after `ptrDebugO`.
- `float* ptrDsv4OScaleFp32` after `ptrDsv4InvRopeCosSinCache`.
- `int64_t mDsv4ScaleBufM` after `mChunkedAttentionSizeLog2`.

These three fields add 24 bytes before the Sage runtime fields. They must be
present even when DSv4 is disabled because the cubin consumes the structure by
value. `KernelParams::setKernelParams` zero-initializes the whole structure, so
all three fields remain zero/null for the two H3 kernels.

The public metadata structure also inserts, immediately before `sha256`:

- `bool mEnablesBf16QFp8KvKOnlyTransform`
- `bool mSeparateTransformedKv`
- `bool mFusesDsv4InvRopeFp8Quant`

All three values are false for the H3 rows.

## Native dispatcher rule

`fmhaKernels.h::run` in the source commit above overrides the autotuner to
`TileScheduler::Static` when
`mNumEltsPerSageAttnBlkP + mNumEltsPerSageAttnBlkV > 0`. H3 Sage uses
Q/K/P/V block sizes `1/16/0/1`, so its official peer is StaticContext. BF16
has no Sage blocks and remains PersistentContext.

## Exact public artifacts

| Path suffix | archive SHA-256 | cubin SHA-256 | cubin bytes | SM | shared bytes | threads |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `QkvBfloat16OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext.cubin.tar.zst` | `afdf798dfea52b29cd5dce2d17a0a4b75cef936f7e10f8d1531922323dde3cfd` | `7ab70b75f16a0264cd9fdb368958ec9c1f095778f02b836af677243283a366bd` | 110104 | `sm_100f` | 164480 | 512 |
| `QkInt8VE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext.cubin.tar.zst` | `dd99a5a4d8e6cb2a51aa72331b89826931e0f212a5e9c45a2714355edab78944` | `3c33caa41b808afe20186f4fcecf1d20cf70ec4509460408f7f212b96a198acd` | 98296 | `sm_100a` | 84688 | 512 |

The minimal local artifact manifest SHA-256 is
`58f630d79b510b913215fa100d6cd3284be486d0663453bf4a7ae5ddcbd578fb`;
its metadata header SHA-256 is
`11d3285c9f609538c1f7d558e2e39d156fc93211ed1fc13419bf73e6f8512536`.

## Launch contract

Both metadata rows use SeparateQkv, Dense mask, Context kernel, no paged KV,
no multi-CTA reduction, Q/KV tiles 128/128, Q/KV steps 256/128, head dimensions
128/128/128, and 512 threads. The launcher passes one by-value `KernelParams`
object to `cuLaunchKernelEx`, with cluster dimension `(1,1,1)`, default cluster
scheduling policy, and programmatic stream serialization controlled by the
runtime PDL flag. No multi-CTA workspace is selected for these rows. The H3
adapter's reusable 256 MiB byte workspace is an API-level FlashInfer argument;
it is not consumed as multi-CTA scratch by these kernels.

Runtime launch/correctness eligibility is recorded separately and must remain
false until exact-ABI BF16 and Sage smoke tests complete without a CUDA error.
