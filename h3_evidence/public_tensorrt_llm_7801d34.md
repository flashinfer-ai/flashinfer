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

## Public BF16 layout/scheduler matrix

The child diagnostic branch restores FlashInfer's immutable public FMHA
artifact directory
`158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/` and manifest
SHA-256 `c2d9399b2537be785882354a4f9902ed6c03136c0ea341e201eac40c3923e1dc`.
That artifact publishes all four H128 BF16 Dense VarSeq combinations needed
for a bounded H3 structural comparison:

- SeparateQkv Static: `7bb1c7081725d4884296c6071705d9744768f0b4eb909cce4d7f5e2932727c3a`;
- SeparateQkv Persistent: `8ce2d53fa98a6138b3a888433c23dc06d133dc6082b1f35fe4a799ba98f70800`;
- PackedQkv Static: `99aea57238cee34596d237ad773ab5a3b432ebb13badcb28053daef52719b56a`;
- PackedQkv Persistent: `d7d03f4c4a1c77e7eb05a8f72202fd96d693550e7d32339b30e9fb64caa437c3`.

For this diagnostic only, the ragged launcher accepts two fail-closed runtime
selectors: `FLASHINFER_TRTLLM_RAGGED_QKV_LAYOUT={separate,packed}` and
`FLASHINFER_TRTLLM_RAGGED_TILE_SCHEDULER={static,persistent}`. Defaults remain
SeparateQkv/Persistent. Packed mode additionally proves that Q, K, and V are
the exact head-major subviews of one fused allocation before setting `qkvPtr`;
it therefore cannot silently reinterpret unrelated tensors.

The public FlashInfer host wrapper must also populate
`KernelParams.logicalGridDim{X,Y,Z}` from the actual launch grid. TensorRT-LLM
7801d34 passes these values to `KernelParamsSetup::setKernelParams`; its Static
context cubins consume them to map CTAs to logical tiles. FlashInfer previously
left all three zero after value-initializing `KernelParams`, which is invisible
to Persistent scheduling but makes Static output invalid. Both the initial
kernel selection and the Cga-to-Gmem fallback parameter rebuild now copy the
resolved CTA grid dimensions before launch.

## Immutable 158f descriptor ABI

The same artifact path was introduced by FlashInfer commit
`9035311e975a6aeb2d229f5162e999dfb7c9a733`. That commit changed the artifact
path and `KernelParams` together because the cubins consume the structure by
value. Its descriptor byte layout is:

| slot | byte offset | size | field |
| ---: | ---: | ---: | --- |
| 0 | 0 | 128 | `tmaQ_` |
| 1 | 128 | 128 | `tmaK_` |
| 2 | 256 | 128 | `tmaO_` |
| 3 | 384 | 128 | `tmaV_` |
| 4 | 512 | 128 | `tmaOSf_` |
| 5 | 640 | 128 | `tmaKSf_` |
| 6 | 768 | 128 | `tmaVSf_` |

`logicalGridDimX` begins at byte 896. Compile-time `sizeof`/`offsetof`
assertions now enforce every entry. FlashInfer commit `9c76c994` later changed
slots 2 and 4 to sliding-window K and O, respectively, while leaving the
158f6fa artifact path and checksum unchanged. That mismatch is invisible to
the Persistent H3 cubin's direct output store, but the Static H3 cubin reads
slot 2 as its O TMA descriptor. The H3 branch therefore restores the immutable
pack's descriptor order and rejects sliding-window-K and dynamic sparse-MLA
launches that require the newer host ABI.

The descriptor-only repair was tested by B200 job `1662912` and reproduced
the prior C48 failure bit-for-bit, including the first damaged QKV element and
the exact mismatch counts. This falsifies descriptor order as the active cause
for this H3 cubin. A complete declaration diff against `9035311e` found the
remaining byte drift: two later DSv4 pointers inserted after `ptrDebugO` and
one DSv4 `int64_t` inserted after `mChunkedAttentionSizeLog2`. Those additions
shift the static scheduler's scalar block by 28 bytes even though DSv4 is
disabled. They belong to the separate TensorRT-LLM 7801d34 artifact lane, not
the immutable 158f pack, and are removed in this diagnostic.

Additional compile-time offsets now pin
`ptrFirstSparseMaskOffsetsKv=976`, reserved state `=1104`,
`mAttentionWindowSize=1112`, `mInflateMax=1124`, `mNumHeadsQ=1168`, and
`mNumTokensPerCtaQ=1204`. The full-ABI patched header SHA-256 is
`e9cce119067e50655589876821af36024276b1038bed2d18fe280bea2f6d95df`.
The target Static BF16 cubin remains immutable at
`7bb1c7081725d4884296c6071705d9744768f0b4eb909cce4d7f5e2932727c3a`.
