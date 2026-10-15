<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# B300 small-head FP8 DSv4 decode

This implementation uses CuTeDSL and CUTLASS primitives behind the existing
`trtllm_batch_decode_sparse_mla_dsv4(..., backend="auto")` API. Explicit backend
selection is preserved. Automatic selection is limited to the qualified SM103
geometry; unsupported geometry, explicit PDL-off, and an unwarmed CUDA Graph
capture retain the native path.

| Property | Supported automatic route |
| --- | --- |
| GPU / DSL | B300 SM103 / CuTeDSL 4.8 or newer |
| Query | FP8 E4M3, ordinary decode, one query per request |
| Heads / rows | H16 with 1, 2 or 4 rows; H32 with 1 or 2 rows |
| Head dimension | 512 |
| Sparse table | 640 columns: 128 SWA slots plus 512 compressed-cache slots |
| Cache | FP8 E4M3, physical token-row offsets; padded page strides supported |
| Output / sinks | BF16 output / FP32 sinks |
| Storage | Caller-owned output and uint8 workspace; serialized workspace use |

The public query/output shape is `[batch, 1, heads, 512]`. HND, NHD and rank3
refer to cache layouts, not to the ordinary query rank. The existing API
normalizes caches before handing them to the internal kernel adapter. Logical
index remapping, ragged/padded requests, native counter buffers and packed KV
formats keep their existing routes.

The partial kernel uses a 512-thread CTA for cooperative copies and softmax;
four of those warps perform CUTLASS FP8 QK and BF16 PV MMA. Vectorized copies,
shared-memory layouts and warp-cooperative merging target the gather and
reduction costs of small query batches. Address multiplication uses 64-bit
arithmetic. PDL allows merge and an eligible downstream consumer to launch
earlier while preserving dependency waits before reading predecessor outputs.
A PDL kernel interval can include waiting, so overlapping launch intervals are
not themselves a measurement of useful work.

Compiled caches own immutable geometry and code, not caller tensors. Each call
rebinds its input, output and workspace pointers. Warm an eligible geometry
outside CUDA Graph capture before expecting the automatic CuTeDSL route inside
a graph; a cache miss during capture falls back without compiling there.

## Reproduce the GPU comparison

From the FlashInfer checkout under test:

```bash
pytest tests/mla/test_dsv4_smallhead.py -v
python benchmarks/bench_dsv4_smallhead.py --output /tmp/dsv4-smallhead-first.json
python benchmarks/bench_dsv4_smallhead.py --reverse --output /tmp/dsv4-smallhead-reverse.json
```

The standalone benchmark covers both admitted shapes and neighboring native
fallback shapes. It checks changing inputs against an FP64 reference, then
measures warm-cache GPU intervals using FlashInfer's standard CUDA Graph event
timer with 128 calls per graph. Captured replay output and native counter reset
are checked after timing. It is independent of campaign files and test-module
imports.

The original controls include the default native path and ten original-kernel
configurations: five logical scheduling budgets with PDL off/on. Only an
isolated copy of the native launcher's `get_device_sm_count` lookup changes;
this is a research scheduling control, not a physical GPU partition or a
public tuning API. Native implementations and mathematical inputs are shared.

Each result gives absolute microseconds for default native, the best measured
native control, and automatic dispatch. The comparison uses the fastest native
repeat and slower candidate repeat, then requires agreement in the independent
reverse-order process. A fallback row must not be called a new-kernel gain.

These synthetic padded-cache measurements are component GPU comparisons, not
model traffic or serving token throughput. The benchmark reports warm-cache
results only; cold-cache and full-model results require their own matched
protocols and must not be combined with these absolute times.
