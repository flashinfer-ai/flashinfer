# Prepared MegaMoE on SM100a and SM103a

The generated catalogs require a GPU with compute capability 10.0 (148
physical SMs) or 10.3 (152 physical SMs); the route is selected from the
device, there is no SM-count override. They contain a fixed set of model
shapes (384 experts, top-6, hidden 5120, intermediate 2304); an unsupported
shape has no fallback. Input preparation and JIT compilation occur before
repeated `run()`.

`flashinfer.source_mega_moe.prepare_mega_moe` executes dispatch, two projections,
clamped SwiGLU, one shared FP8 expert, and combine in one submission. Routed
weights may use packed FP4 E2M1 or FP8 E4M3. It accepts packed activations and
scale words, int64 expert indices and float32 route weights, and explicit packed
weight tensors. `MegaMoEPlan` documents the tensor extents. Its workspace is
zero-initialized once. Subsequent calls rely on device cleanup. Use
`update_inputs()` between serialized calls to copy new activation/routing inputs
into the same workspace (four copies and one scatter with a permutation index
computed at preparation); do not reset its counters on the host.

`flashinfer.mega_moe_v3.prepare_pipeline` accepts logical gate/up weight halves
and float power-of-two scales. Its FP4 model routes cover 1, 16, 128, 512, 1024
and 4096 tokens (FP8 routed weights at 16 tokens); the 1024- and 4096-token
routes use the 32- and 128-row source tile heights. Preparation packs scales and interleaves the
weights. Every exported pipeline route is `self_cleaning`: it launches exactly
one kernel per `run()`, the kernel zeroes its per-launch workspace words before
it exits and its grid gates are phase-toggling words, so the plan zeroes the
counter workspace once when it is bound, never inside `run()` (a captured
`run()` contains only the kernel node), and `plan.reset()` is rejected.
`plan.self_cleaning` reports the contract; the grouped fused route keeps its
L1 arrival reset inside `run()`. The separately exported `prepare_grouped_l2`
accepts E4M3 activation values, packed E2M1 weights and natural row-major
packed scale words; it repacks both scale tensors into the group-folded layout
at preparation, `run()` launches only the clustered GEMM, and
`plan.update_scales()` repacks again after the caller changed the scale words
in place.

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
pytest -q tests/experimental/test_generated_mega_moe.py
```

The public tests use independent PyTorch math on the exact packed inputs,
check valid dispatch metadata and reusable counters, poison outputs between
launches, and exercise direct and graph replay. They cover both routed
precisions of both families on the 16-token model route, the FP4 single-token
route's workspace lifecycle, the FP4 1024- and 4096-token routes, input refresh
on a reused source workspace, and the catalog's grouped L2 route, and they check that a captured `run()` is one
kernel node. The 384-expert inputs need about 48 GiB of free device memory;
the tests skip below that. They do not enumerate every model shape or claim
model-wide performance.
