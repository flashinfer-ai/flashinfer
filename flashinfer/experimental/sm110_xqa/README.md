# Experimental SM110 XQA attention

This opt-in module provides native attention for NVIDIA Thor GPUs with the
exact SM110a target: the D512 speculative-decode (tree) attention in two
`tcgen05` schedules and the D128 single-token decode. Call
`flashinfer.sm110_xqa.prepare` for prepared replay or `attention` for a single
invocation. Both entry points use FlashInfer's standard experimental API
decorator: calling either is explicit opt-in and emits `ExperimentalWarning`
once per API. The backend implementation and the generated sources remain
under `flashinfer.experimental.sm110_xqa`. Experimental APIs may change or be
removed without deprecation.

The package is a self-contained Cake package (FlashInfer issue #6553):

| File | Role |
| --- | --- |
| `cake_jit.py` | The registry: one record per exported program (`MODULES`), the host plan (route selection, grid rules, the decode workspace and partition default) and the loader glue (`select_record`, `select_decode_record`, `decode_partitions`, `launch_grid`, `argument_list`, `entry`). |
| `build.json` | The build manifest (schema `cake-build/1`): one library per program, its two sources under `csrc/` with their SHA-256 digests, the nvcc flags of the source build and the `sm_110a` gencode. |
| `cake_build.py` | The loader shared by every Cake package: a prebuilt `aot/<library>.so` when a builder compiled it, else FlashInfer's JIT from the manifest's sources and flags (`use_fast_math=False`; the entry's own `--use_fast_math` is in `nvcc_flags`). |
| `csrc/cake_ffi_checks.h` | The tvm-ffi checks the bindings use; no FlashInfer header is included. |
| `csrc/cake_sm110_xqa_<digest>_{kernel,binding}.cu` | One device translation unit and one tvm-ffi binding (`run`, `build_cuda_version`) per program. |
| `csrc/cake_sm110_xqa_tree_fp16_contig_tmem_even_kernel.cuh` | The device translation unit the `tree_fp16_contig_g<ratio>_tmem_even` records share (GQA ratios 2, 4 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_contig_tmem_even_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_contig_tmem_odd_kernel.cuh` | The device translation unit the `tree_fp16_contig_g<ratio>_tmem_odd` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_contig_tmem_odd_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_page128_tmem_even_kernel.cuh` | The device translation unit the `tree_fp16_page128_g<ratio>_tmem_even` records share (GQA ratios 2, 4 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_page128_tmem_even_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_page128_tmem_odd_kernel.cuh` | The device translation unit the `tree_fp16_page128_g<ratio>_tmem_odd` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp16_page128_tmem_odd_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_tmem_even_kernel.cuh` | The device translation unit the `tree_fp8_contig_g<ratio>_tmem_even` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_tmem_even_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_tmem_odd_kernel.cuh` | The device translation unit the `tree_fp8_contig_g<ratio>_tmem_odd` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_tmem_odd_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_tmem_even_kernel.cuh` | The device translation unit the `tree_fp8_page128_g<ratio>_tmem_even` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_tmem_even_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_tmem_odd_kernel.cuh` | The device translation unit the `tree_fp8_page128_g<ratio>_tmem_odd` records share (GQA ratios 2, 4, 8 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_tmem_odd_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_pair_kernel.cuh` | The device translation unit the `tree_fp8_contig_g<ratio>_pair` records share (GQA ratios 2, 4 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_contig_pair_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_pair_kernel.cuh` | The device translation unit the `tree_fp8_page128_g<ratio>_pair` records share (GQA ratios 2, 4 and 16); each record's kernel source defines the ratio and the kernel symbol and includes it. |
| `csrc/cake_sm110_xqa_tree_fp8_page128_pair_binding.h` | The tvm-ffi binding those records share; each record's binding source defines the ratio, the Q box height, the kernel symbol and name and the binding namespace and includes it. |
| `csrc/cake_sm110_xqa_device_abi.cuh` | The device ABI block every kernel translation unit includes first: fixed-width typedefs, the `CakeTensorMap64` / `CUtensorMap` size and alignment asserts, the CUDA headers, `make_warp_uniform` and `CAKE_INF`. |
| `csrc/cake_sm110_xqa_device_helpers.cuh` | The device helper library the kernel translation units share; every kernel translation unit includes the whole library, and the helpers a unit does not call emit no code. |
| `csrc/cake_sm110_xqa_kv_cache_list_contig.h` | The `KVCacheList` record of the contiguous cache layout, included by each tree kernel of that layout and by its binding. |
| `csrc/cake_sm110_xqa_kv_cache_list_page128.h` | The `KVCacheList` record of the `page128` cache layout, included by each tree kernel of that layout and by its binding. |
| `csrc/cake_sm110_xqa_binding_helpers_tma_record.h` | The host helpers the tree bindings share (CUDA and driver error checks, tensor-view accessors, TMA extent and stride conversion, the cache record writer), included once per binding inside its namespace after `KERNEL_NAME`. |
| `csrc/cake_sm110_xqa_binding_helpers_decode.h` | The host helpers the decode bindings share, included the same way. |
| `backend.py` | Argument validation, routing, the workspace and the prepared launch. |

Every generated file is written by the Cake exporter; `cake_jit.py` is read as
a literal by `scripts/check_cake_build_manifests.py`, which checks the manifest
against the registry and the sources. Edit the exporter, not the generated
files.

The `csrc/cake_sm110_xqa_*` headers (`.cuh` and `.h`) are a hand fold of text
the generated files shared, pending regeneration from the exporter. Like the
other shared headers under `csrc/`, the `cake_sm110_xqa_tree_*` body headers
are rebuilt through the build system's header dependency tracking and are not
covered by a `build.json` digest.

Native compilation and correctness checks have run with CUDA 13.4 on physical
SM110 hardware. The interface supports two attention families:

| Family | Q | KV | Other metadata |
| --- | --- | --- | --- |
| Decode, D128 | FP16 `[B,Hq,128]` or `[B,1,Hq,128]` | FP16 `[B,2,Hkv,C,128]` | int32 lengths `[B]`; Hq/Hkv = 4, 8, or 16 |
| Tree, D512 | FP16 `[B,Q,Hq,512]` | FP16/E4M3 `[B,2,Hkv,C,512]` | int32 lengths `[B]`, packed mask; Hq/Hkv = 2, 4, 8, or 16 |
| Paged tree | Same Q, or packed Q below | FP16/E4M3 `[numPages,128,Hkv,512]` | int32 page table `[B,2,maxPages]` |
| Packed tree | FP16 `[totalQ,Hq,512]` | Either tree layout | int32 prefix offsets `[B+1]`, explicit maximum query length |

All tensors must be contiguous CUDA tensors on one device. The two components
of the page table address K and V independently. The draft mask contains
int32 packed words, with bit 1 permitting attention. It covers only the final
draft-token block; all earlier tokens are visible. Its shape is
`[B,Q,ceil(Q/32)]`, or `[totalQ,ceil(maxQ/32)]` for packed queries. Packed rows
retain the maximum-length mask stride even when a request is shorter.

Device metadata must satisfy these preconditions: sequence lengths fit cache
capacity; each tree length includes that request's draft block; page indices
address the pool; query offsets are nondecreasing, start at zero, end at
totalQ, and each request length is at most maxQ. Preparation validates tensor
metadata without copying device contents to the CPU.

```python
from flashinfer.sm110_xqa import prepare

plan = prepare(
    q, kv, sequence_lengths,
    mask=mask,                         # omitted for D128 decode
    out=output,                        # optional caller-owned tensor
    page_table=page_table,             # None for contiguous KV
    page_size=128,                     # 0 for contiguous KV
    k_scale=k_scale,
    v_scale=v_scale,
    kernel="auto",                     # D512: "auto", "tmem" or "pair"
)
output = plan.run()
```

`sm_scale` defaults to `1/sqrt(D)`. `k_scale` and `v_scale` independently
dequantize FP8 storage; decode requires both to be 1. Output is FP16 and may
not alias any input.

## D512 tree schedules

`kernel` selects the schedule of a D512 call. Both schedules accumulate S, P
and O in tensor memory on `tcgen05.mma`, apply the same tensor layouts, mask
contract, dequantization scales and tolerances, and are instantiated per cache
precision (FP16 / E4M3), cache layout (contiguous / page128) and GQA ratio
(2 / 4 / 8 / 16):

- `"tmem"`: one CTA per 128-row Q tile of a KV head and 256-column output half
  (512 threads, grid `(2, Hkv * ceil(Q * ratio / 128), B)`). The two
  output-half CTAs of a Q tile and the two Q tiles of a KV head form a
  `(2, 2, 1)` thread-block cluster in which Q is fetched once per output-half
  pair and every K/V tile once per cluster by TMA multicast; a `(2, 1, 1)`
  Q-multicast form serves KV heads with an odd number of Q tiles. Every cache
  mode.
- `"pair"`: the same schedule as two `cta_group::2` CTA pairs in a `(4, 1, 1)`
  cluster per two Q tiles of a KV head (grid
  `(4, Hkv * ceil(Q * ratio / 128) / 2, B)`): the pair leader issues one M256
  MMA stream for both Q tiles, each CTA holds one token half of every K chunk
  and one column half of every V chunk, and the raw E4M3 V rows are multicast
  to the pair. E4M3 KV whose heads have an even number of 128-row Q tiles;
  `kernel="pair"` rejects FP16 KV and odd Q-tile counts.
- `"auto"` (default): `pair` for E4M3 KV with an even Q-tile count per head,
  `tmem` otherwise.

The ratio-8 records of the `(2, 2, 1)` FP16 `tmem` form and of the E4M3
`pair` form (both cache layouts) are tuned builds of the same schedule: eight
softmax warps, four convert warps and a K(0)-first prologue, with a partial
tensor-memory allocation for `tmem` and the follower-Q wait for `pair`. The
other records are the base builds; every record keeps the layouts, mask
contract, scales and tolerances above.

The retired names `tcgen05`, `register_mma`, `register_mma_split` and
`register_mma_auto` (the frozen tree and the register `mma.sync` families,
superseded by the two schedules above) raise `ValueError` naming the
retirement; any other name raises `ValueError("kernel must be one of ...")`.

The selected record is available as `plan.route` (`cake_jit.MODULES` key,
`tree_<fp16|fp8>_<contig|page128>_g<ratio>_<tmem_even|tmem_odd|pair>`), its
exported program as `plan.program` and the launch grid as `plan.grid`; replay
does not redo dispatch. The generated binding of every tree program binds Q
and the KV cache as TMA descriptor sources (an E4M3 cache as its bytes), the
draft mask as unsigned 32-bit words and the cache record member by member.
The kernels read the query offsets through a non-null pointer: for uniform
queries the backend synthesizes the prefix `0, Q, 2Q, ...` once at prepare
(it is part of `plan.inputs`); a packed call passes the caller's offsets.

## D128 decode

The decode route serves every batch, capacity and GQA ratio 4 / 8 / 16 over
a contiguous FP16 cache; it takes `kernel="auto"` (the default) or the
route's previous name `"tcgen05"`, and rejects the tree schedule names with
`ValueError("D128 decode has only the tcgen05 kernel family")`. One
128-thread CTA per (partition, KV head, request) streams the partition's keys
through a TMA ring into a tensor-memory tile whose live rows are the KV
head's query heads, and the partitions of a (request, KV head) are merged by
the last CTA to finish; the merge order is fixed, so a launch is bitwise
reproducible. D128 accepts a runtime `partition_tokens` argument that is a
positive multiple of 64; omitting it selects the registry default
(`cake_jit.DEFAULT_PARTITION_TOKENS`, 256). The grid is
`(partitions, Hkv, B)` with `partitions = ceil(C / partition_tokens)`.

The record is a pure function of the partition count
(`cake_jit.select_decode_record(ratio=..., partitions=...)`): a decode record
whose variant carries `cluster_partitions` equal to the count serves it with a
thread-block cluster of that many CTAs merging through distributed shared
memory; every other count takes `decode_fp16_contig`, the program that merges
any count through the workspace. This registry ships three decode records:
`decode_fp16_contig_c2` and `decode_fp16_contig_c4` serve two and four
partitions (a capacity of 257..512 and 769..1024 tokens at the default
`partition_tokens`) with clusters of two and four CTAs, `decode_fp16_contig`
every other count; `plan.route` names the record a plan launches.

D128 workspace is `(partial, statistics, counters)`: FP32 tensors of shapes
`[B*Hq,partitions,128]` and `[B*Hq,partitions,2]`, plus int32 counters
`[B,Hkv]`. It is available as `plan.workspace`; a compatible, disjoint
workspace tuple can be passed to `prepare`. Q, KV, output and partial-output
base addresses must be 16-byte aligned for the TMA descriptors and vector
memory accesses, statistics 8-byte aligned; `prepare` rejects insufficient
alignment (`ValueError("<name> has insufficient address alignment")`) before
any launch. Preparation initializes counters to zero once (the kernel reads
them as unsigned 32-bit values). The fused kernel wraps them back to zero at
the end of each completed replay. Workspace must remain private to ordered
launches: do not launch two plans sharing it concurrently or re-prepare it
while an earlier launch is running.

Prepare before timing or graph capture. `plan.run()` submits the complete
kernel sequence on the caller's current stream and performs no tensor
allocation. `attention(...)` is a convenience wrapper that prepares and runs
once. Standard PyTorch stream dependencies apply to inputs and workspace
initialization. A launch on another stream must wait for preparation and any
previous replay using the same workspace; output consumers must wait for the
launch stream. No host synchronization or stream selection is hidden in the
prepared plan. For example:

```python
preparation_stream = torch.cuda.current_stream(q.device)
plan = prepare(q, kv, sequence_lengths, partition_tokens=256)  # D128
execution_stream = torch.cuda.Stream(device=q.device)
execution_stream.wait_stream(preparation_stream)
with torch.cuda.stream(execution_stream):
    output = plan.run()
preparation_stream.wait_stream(execution_stream)
```

Neither API substitutes a kernel for another GPU architecture. The module
requires CUDA 13.0 or newer and physical capability 11.0; the tested toolchain
is CUDA 13.4.

## Validation

Commands from a FlashInfer checkout:

```bash
python scripts/check_cake_build_manifests.py --in-tree --package sm110_xqa
python -m pytest tests/experimental/test_cake_sm110_xqa.py -q
python -m pytest tests/experimental/sm110_xqa/test_sm110_xqa.py -q
python benchmarks/bench_sm110_xqa.py --warmup-ms 250 --output sm110-xqa-native.json
```

`tests/experimental/test_cake_sm110_xqa.py` is the backend suite: it imports
the package alone (through `tests/test_helpers/cake_package.py`), checks the
`build.json` contract against the registry, the route and grid rules on the
CPU, and on an SM110 device the FP32-oracle correctness, the bitwise
determinism of repeated launches, side-stream launches and CUDA-graph
replays, and the counter reset of the decode route. The public suite keeps the
numerical ledger over the public API (an FP32 oracle at `atol=rtol=1e-2`,
quantized caches included), the runtime partition and repeated-counter checks,
caller-provided workspace initialization, cross-stream graph replay and the
output-alias rejection. The performance rows are specified in
`benchmarks/sm110_xqa_shapes.json`; the standalone benchmark uses
`flashinfer.testing.bench_gpu_time` with CUPTI, cold L2, a default 250 ms
warmup target and 256 measured iterations, and `--kernel` runs every tree row
under one schedule for a comparison. It measures the complete prepared replay
and reports native latency only.
