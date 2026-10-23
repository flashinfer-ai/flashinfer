# Experimental SM120 CuTe GDN prefill

Owner: @cuihangbin. Tracking issue: https://github.com/flashinfer-ai/flashinfer/issues/7118.

This JIT-only backend is selected explicitly through the existing public API:

```python
from flashinfer.gdn_prefill import chunk_gated_delta_rule
output, final_state = chunk_gated_delta_rule(
    q, k, v, g=log_gate, beta=beta, initial_state=initial_state,
    output_final_state=True, cu_seqlens=cu_seqlens,
    cu_seqlens_cpu=cpu_offsets, gate_domain="log", backend="sm120_cute",
)
```

The call emits the standard experimental warning. Stable `backend="auto"` and
`backend="flashinfer"` never select this backend. The old pipeline32 environment
variable and stable-path implementation are removed.

## Qualified fast domain

RTX 6000D, exact SM120/156 SMs, D=128, contiguous BF16/FP16 QKV, Hq=Hk and
Hv divisible by Hq; finite FP32 native log gates and FP32 beta/state. CPU offsets
must match the GPU offsets and remain immutable for each captured graph.
Layouts are cached by CPU values, dtype, device and CP choice, not GPU tensor
identity. Up to 64 layouts are retained without eviction to preserve captured
pointer lifetimes; new layouts beyond that use the official backend. Warm a
layout before graph capture. Empty sequences use the official backend.

Linear/raw gates, absent gates, logits, checkpoints, state pools, non-FP32 state,
other GPU configurations and unsupported shapes fall back to the official
FlashInfer implementation. Checkpoint capacity retains the official contract
`sum(seq_len_i // interval)` and the `checkpoint_cu_starts` caller precondition;
there is no aggregate ceil bound. In particular `[100]`, `[64,32]`, `[200,200]`
at interval64 require 1, 1 and 6 checkpoints respectively.

The consumer uses 32-token chunks, with CP boundary generation for low
parallelism and V64/V128 consumers. Proven full chunks specialize away tail
handling; the flag is part of each compile key. This is the measured CuTe auto
rework, not the previous pipeline32 implementation. The separate local-fusion
prototype is not included.

## Validation and measurement

```bash
pytest -q tests/experimental/gdn_sm120_cute
python benchmarks/bench_gdn_sm120_cute.py --tp 1 --preallocated --output tp1.json
python benchmarks/bench_gdn_sm120_cute.py --tp 2 --preallocated --output tp2.json
```

The benchmark compares the actual public `chunk_gated_delta_rule` default route
(`backend="flashinfer", use_cp="auto"`) with this explicit backend. Both receive
the same native log gates, CPU offsets and optional caller buffers. GPU gate
materialization and output/state copies performed by either call are included.
CUPTI, cold L2, three alternating rounds, 10 warmups and 100 samples per round;
JIT and CPU overhead are excluded. TP means per-rank head dimensions, without
communication. Results for a source-hashed commit are stored beside the benchmark;
older internal-kernel measurements are not evidence for this public adapter.

Correctness tests include an independent FP64 recurrence, full/tail and index
dtype cache switching, CUDA graphs, checkpoint remainders and low-precision state
fallback. Only measured RTX6000D workloads are claimed; no other SKU, training,
inter-card CP or all-shape performance guarantee is implied.

## Provenance and graduation

The CuTe forward/boundary algorithms and host CP policy are adapted from Qwen
FlashQLA commit5538c35f0df33ef804eacd90c77d9d9087c1d4e4; its MIT notice is retained
in LICENSE.FlashQLA. This backend does not import FlashQLA, TileLang or FLA.

Review target: 2026-11-20 (four weeks). Graduation requires reproducible public-call
gains against current stable default routing, documented production usage,
correctness/CI approval and a maintainable specialization policy. Otherwise keep
it experimental only with a maintainer-approved extension, or remove it. This date
is a review target, not a promise of stable graduation.
