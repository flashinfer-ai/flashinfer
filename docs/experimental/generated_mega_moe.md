# Prepared MegaMoE on SM103a

The generated catalogs require a GPU with compute capability 10.3 and 152
physical SMs. They contain a fixed set of shapes; an unsupported shape has no
fallback. Input preparation and JIT compilation occur before repeated `run()`.

`flashinfer.source_mega_moe.prepare_mega_moe` executes dispatch, two projections,
clamped SwiGLU, one shared FP8 expert, and combine in one submission. Routed
weights may use packed FP4 E2M1 or FP8 E4M3. It accepts packed activations and
scale words, int64 expert indices and float32 route weights, and explicit packed
weight tensors. `MegaMoEPlan` documents the tensor extents. Its workspace is
zero-initialized once. Subsequent calls rely on device cleanup. Use
`update_inputs()` between serialized calls to copy new activation/routing inputs
into the same workspace; do not reset its counters on the host.

`flashinfer.mega_moe_v3.prepare_pipeline` accepts logical gate/up weight halves
and float power-of-two scales. Preparation packs scales and interleaves the
weights. `run()` includes its required counter resets. The separately exported
`prepare_grouped_l2` accepts E4M3 activation values, packed E2M1 weights and
natural row-major packed scale words; every `run()` repacks both scale tensors
before the clustered GEMM. Do not time the GEMM alone.

FP4 packs the earlier K element into the low nibble. The scale granularity is
32 values. A UE8M0 byte contains the biased exponent of a power-of-two scale;
four consecutive bytes make an int32 scale word. Source weight scales additionally
use a 128-row permutation and group-folded order. The example input helper
shows these layouts using synthetic values without an external quantizer.

L1 projection rounds to BF16 before gate/up clamping. Gate values have an upper
bound of 10, while up values are clamped to [-10, 10]. SwiGLU and route weighting
produce FP32 values; these are directly quantized to E4M3 with per-32 scales.
Each L2 contribution rounds to BF16 before FP32 combine and BF16 output.

A plan retains its tensors and descriptor storage. Reuse it on serialized
streams; do not run one plan concurrently. The caller may capture `plan.run()`
in a CUDA graph after warmup. Preparing a plan or replacing its storage during
capture is unsupported.

From the repository root:

```bash
python examples/experimental/mega_moe.py --family source --precision fp4
python examples/experimental/mega_moe.py --family v3 --precision fp8
python examples/experimental/mega_moe.py --family grouped-l2
pytest -q tests/experimental/test_generated_mega_moe.py
python examples/experimental/bench_mega_moe.py --family v3 --precision fp4
```

The public tests use independent PyTorch math on the exact packed inputs,
check valid dispatch metadata and reusable counters, poison outputs between
launches, and exercise direct and graph replay. They include both routed
precisions on the two smoke routes and the catalog's grouped L2 route. These
examples do not enumerate every model shape or claim model-wide performance.

The benchmark requires `cupti-python >= 13`, flushes L2, and sums all kernel
activities belonging to the complete `plan.run()` call. Missing CUPTI is an
error; no CUDA-event fallback is permitted. The reported median excludes JIT
and input preparation. It does not establish comparative speedup or an export
regression gate.
