# Experimental SM110 GQA decode

This package provides an explicit opt-in FP16 decode path for NVIDIA SM110
(Jetson AGX Thor). It is specialized for 32 query heads, 8 key/value heads,
head dimension 128, and one query token per request. The kernels therefore
have a fixed 4:1 query-to-KV head ratio.

The backend is reached only through explicit experimental APIs. It is
not registered for AOT packaging or automatic decode routing. The experimental
lifecycle is owned by `@yyihuang` and tracked in issue #5051.

The input contract is:

- `q`: contiguous FP16 `[batch, 32, 128]`
- `kv`: contiguous FP16 `[batch, 2, 8, capacity, 128]` with `capacity >= 64`
  (`backend.MIN_CAPACITY`, the 64-token TMA box of the short route; a smaller
  cache raises `ValueError` as before), with K at index 0 and
  V at index 1
- `sequence_lengths`: contiguous CUDA int32 `[batch]`; every value must be in
  the inclusive range `[1, capacity]` whenever a launch runs. The lengths are
  read on the device only; no API call copies them to the host.
- `out`: optional caller-owned contiguous FP16 `[batch, 32, 128]`

```python
import torch
from flashinfer import sm110_gqa_decode

q = torch.randn(1, 32, 128, dtype=torch.float16, device="cuda")
kv = torch.randn(1, 2, 8, 1024, 128, dtype=torch.float16, device="cuda")
sequence_lengths = torch.tensor([1024], dtype=torch.int32, device="cuda")
out = sm110_gqa_decode(q, kv, sequence_lengths)
```

## Package form

The package is a self-contained Cake package (FlashInfer issue #6553):

| File | Role |
| --- | --- |
| `cake_jit.py` | The registry: one record per exported program (`MODULES`), the host plan as data (the `select_rule` and `grid_rule` of every record, the `workspace` leaf of a split record) and the loader glue (`select_record`, `launch_grid`, `argument_list`, `entry`). |
| `build.json` | The build manifest (schema `cake-build/1`): one library per program, its two sources under `csrc/` with their SHA-256 digests, the nvcc flags of the source build and the `sm_110a` gencode. |
| `cake_build.py` | The loader shared by every Cake package: a prebuilt `aot/<library>.so` when a builder compiled it, else FlashInfer's JIT from the manifest's sources and flags (`use_fast_math=False`; the entry's own `--use_fast_math` is in `nvcc_flags`). |
| `csrc/cake_ffi_checks.h` | The tvm-ffi checks the bindings use; no FlashInfer header is included. |
| `csrc/cake_sm110_gqa_decode_<digest>_{kernel,binding}.cu` | One device translation unit and one tvm-ffi binding (`run`, `build_cuda_version`) per program. |
| `csrc/cake_sm110_gqa_decode_device_abi.cuh` | The device ABI block every kernel translation unit includes first: fixed-width typedefs, the `CakeTensorMap64` / `CUtensorMap` size and alignment asserts, the CUDA headers, `make_warp_uniform` and `CAKE_INF`. |
| `csrc/cake_sm110_gqa_decode_device_helpers.cuh` | The device helper library the kernel translation units share; every kernel translation unit includes the whole library, and the helpers a unit does not call emit no code. |
| `csrc/cake_sm110_gqa_decode_binding_helpers_tma.h` | The host helpers the six TMA bindings share (CUDA and driver error checks, tensor-view accessors, TMA extent and stride conversion), included once per binding inside its namespace after `KERNEL_NAME`. |
| `backend.py` | Argument validation, routing, the workspace and the launches of both public APIs. |

Every generated file is written by the Cake exporter; the package tests check
`build.json` against the registry in `cake_jit.py` and against the sources.
Edit the exporter, not the generated files.

The three `csrc/cake_sm110_gqa_decode_*` headers (`.cuh` and `.h`) are a hand
fold of blocks the generated files shared, pending regeneration from the
exporter.

## Routes

Seven programs ship for compute capability 11.0 and CUDA 13.0 or newer:
`short` is a CUDA-core program (256 threads, no tensor memory, no TMA); the
other six are `tcgen05` / TMA / tensor-memory kernels. The route is a pure
function of the batch `B`, the cache capacity `C` and the caller's
`num_splits` (`cake_jit.select_route` answers the record and the split count
it launches with, `cake_jit.select_record` the record alone); no rule reads a
tensor. Every row of a record's `regions` rule carries its own split count:
an integer on the size-based rows (`None` / `1`; one split on every row
today) and `"runtime"` -- the caller's count -- on the explicit-split rows, so
a size-based row can route to a run-time-split program with several splits
by a row edit, without a new rule kind:

| `num_splits` | Shape | Record | Splits | Grid |
| --- | --- | --- | --- | --- |
| `None` / `1` | `B = 1`, `C = 64` | `short` (CUDA cores, 256 threads, one 64-token block) | 1 | `(B * 8, 1, 1)` |
| `None` / `1` | `B = 1`, `65 <= C <= 128` | `long` (384 threads, two-slot K/V ring) | 1 | `(B * 8, 1, 1)` |
| `None` / `1` | `B = 1`, `129 <= C <= 256`; `B = 2`, `C <= 512` | `split_n64_direct` (64-token tiles, three-slot ring, one split, direct output) | 1 | `(1, B * 8, 1)` |
| `None` / `1` | `B = 1`, `C >= 257`; `B = 2`, `C >= 513` | `split_n64_r4_direct` (64-token tiles, four-slot ring, one split, direct output) | 1 | `(1, B * 8, 1)` |
| `None` / `1` | `B >= 3` | `split_n32_direct` (32-token tiles, one split, direct output) | 1 | `(1, B * 8, 1)` |
| `2, 4, 8, 10, 16` | `B <= 2`, `C <= 4095` | `split_n64` (the split count from the grid) | `S` (the caller's) | `(S, B * 8, 1)` |
| `2, 4, 8, 10, 16` | every other shape | `split_n32` | `S` (the caller's) | `(S, B * 8, 1)` |

The split-KV programs run one 128-thread CTA per (request, KV head, split)
and merge the splits in the last CTA of each (request, KV head) through the
workspace. The accepted `num_splits` values are `cake_jit.ACCEPTED_SPLITS`
(`{1, 2, 4, 8, 10, 16}`); any other value raises `ValueError`.

A split record names a workspace (`cake_jit.MODULES[record]["workspace"]`):
FP32 `partial_O [B, 32, S, 128]`, `partial_max` / `partial_sum [B, 32, S]` and
`completed` counters `uint32 [B * 8]`, zero at prepare. The counters reset
themselves after every merge, so a prepared launch replays from a CUDA graph
without a reset node; the one-split builds keep the pointers in their binding
and never read them. The convenience API allocates the workspace per call;
the prepared API allocates it once.

## Prepared sustained decode

Use `prepare_sm110_gqa_decode` and `launch_sm110_gqa_decode_prepared` to reuse
caller-owned output and workspace across ordinary launches or CUDA Graph replays:

```python
from flashinfer import prepare_sm110_gqa_decode, launch_sm110_gqa_decode_prepared

output = torch.empty_like(q)
prepared = prepare_sm110_gqa_decode(
    {"Q": q, "KV": kv, "O": output, "sequence_lengths": sequence_lengths}
)
result = launch_sm110_gqa_decode_prepared(prepared)  # result is output
```

Preparation validates tensor metadata, selects a route and compiles outside
Graph capture. Every length must remain in `[1, capacity]` at launch; GPU
length values may change, but this precondition is not checked with a host
read. Unlike the convenience API, the prepared API requires caller output that
does not alias any input storage, and finite positive `q_scale` (default 1).
Launch uses the current PyTorch stream and returns asynchronously. Keep the
prepared object and tensors alive until all work and captured Graphs finish.
Order tensor updates before launch, and allocate separate prepared instances
for concurrently executing streams or Graphs because workspace is mutable.
The prepared state records `route` (the registry record), `num_splits` (the
run-time split count), `grid`, `workspace` and `workspace_bytes`.

This API is experimental because its tensor layout and fixed head geometry are
serving-workload specific. Graduation requires broader workload validation,
stable packaging coverage, and an agreed integration point with FlashInfer's
decode APIs.

## Validation

```bash
python -m pytest tests/experimental/test_cake_sm110_gqa_decode.py -q
python -m pytest tests/experimental/test_sm110_gqa_decode.py tests/experimental/test_sm110_gqa_decode_prepared.py -q
python examples/experimental/sm110_gqa_decode_prepared.py --graph
python benchmarks/bench_sm110_gqa_decode.py
```

The package is not registered with `scripts/check_cake_build_manifests.py`
yet; `cake_build` applies the same digest rule at JIT load: the SHA-256 of
each source listed in `build.json` must equal the manifest's value.

`tests/experimental/test_cake_sm110_gqa_decode.py` is the backend suite: it
imports the package alone (its `PACKAGE` constant names the import path),
checks the `build.json` contract against the registry and the route table on
the CPU, and on an SM110 device the FP32-oracle correctness of every record,
the explicit split counts, the bitwise determinism of repeated launches,
side-stream launches and CUDA-graph replays, and the counter reset. The public
suites keep the FP32-oracle cases, the host-path checks (no device
synchronization), dynamic-length graph replay, independent prepared objects on
other streams and the alias rejection. The benchmark reports the same-process
cold-L2 CUPTI median against PyTorch SDPA on an SM110 device.
