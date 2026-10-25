# Prepared MegaMoE on SM100a and SM103a

The generated catalog requires a GPU with compute capability 10.0 (148
physical SMs) or 10.3 (152 physical SMs); the route is selected from the
device, there is no SM-count override. It contains a fixed set of shapes of
the 384-expert, top-6, hidden-5120 model: the single-rank routes
(intermediate 2304; routed FP4 at 1, 16, 128, 512, 1024 and 4096 tokens,
routed FP8 at 16 tokens) on both architectures, and the tensor-parallel and
expert-parallel routes of the model (384 experts at the physical intermediate
widths 640 and 384, 96 and 48 experts per rank at 2304; routed FP4 at 1, 4,
16, 32, 128, 512, 1024 and 4096 tokens) as the catalog of an architecture
lists them. An unsupported shape has no fallback: `prepare_pipeline` raises
an error that names the requested shape and the shapes the architecture's
catalog carries. Input preparation and JIT compilation occur before
repeated `run()`.

The catalog (`mega_moe_v3.v3`) lists one program per schedule, compiled for
every architecture it names from one source: the pipeline kernels read the
SM count from the compile line (`-DNUM_CTAS=<physical SMs>`, supplied by the
loader from the route), the grouped kernels carry no SM count at all. Two
FP4 pipeline schedules (16 and 128 tokens) select different producer
instructions per architecture and stay as per-architecture pairs. The
routes table is keyed by architecture, so a route may ship for one
architecture only. Every route names the SM counts it admits (one, or a
list for a program that takes the count from its compile line; the runtime
picks the device's count and refuses any other) and its launch grid; a
route's metadata names the program's source-schedule form (`block_m`, the
source tile height; `readiness`, `"block"` or `"grid"`) and whether its
program carries the local expert filter (`local_expert_filter`). The
bindings are thin `tvm_ffi_utils.h` launchers.

`flashinfer.mega_moe_v3.prepare_pipeline` executes dispatch, two projections,
clamped SwiGLU and combine in one submission. Routed weights may use packed
FP4 E2M1 or FP8 E4M3. It accepts logical gate/up weight halves and float
power-of-two scales. The 1024- and 4096-token routes use the 32- and
128-row source tile heights, the others 16 rows. A `topk_idx` of -1 routes
its slot to no expert, as a routing gate marks a masked token: the slot takes
no row and adds nothing to the output, whatever its weight. Preparation packs
scales and interleaves the weights. Every exported pipeline route is `self_cleaning`: it launches exactly
one kernel per `run()`, the kernel zeroes its per-launch workspace words before
it exits and its grid gates are phase-toggling words, so the plan zeroes the
counter workspace once when it is bound, never inside `run()` (a captured
`run()` contains only the kernel node), and `plan.reset()` is rejected.

`intermediate` is the model's logical width, a multiple of the 32-value
scale group (576 and 288 for the TP4 and TP8 shardings of the model). The
programs compute the intermediate in 128-wide steps, so the route is the
one generated for the next multiple of 128 (640, 384) and preparation pads
the weights and scales to it once: zero rows after each W1 half (before the
gate/up interleave), zero W2 columns, scale 1.0 in the padded groups. The
padded columns activate to exactly 0 and add exact zeros, so the output is
bit-identical to the unpadded arithmetic; a width that already is a
multiple of 128 is packed as it is. Weights pre-padded by the caller (as
the TensorRT-LLM path takes them) are accepted with `intermediate` set to
the padded width.

Expert parallelism is a window of the global expert space. `num_experts` is
the number of experts this rank holds (the leading dimension of its
weights); `local_expert_offset` (default 0) is the global id of its first
expert and `num_global_experts` (default `num_experts`) the size of the
global space. `topk_idx` carries global ids in `[0, num_global_experts)` or
-1. A slot whose id falls outside `[local_expert_offset, local_expert_offset
+ num_experts)` routes to no expert on this rank, exactly like a -1 slot,
so the output is the rank's partial sum over its local routes, with zero
rows for tokens without one; the combine of the partials across ranks is
the caller's. The filter runs on the device (the `local_expert_offset`
kernel scalar of the routes that carry `local_expert_filter`), so the
caller's `topk_idx` tensor stays bound as it is and a captured `run()` is
still one kernel node. A route without the filter accepts only a
single-rank window. Preparation validates the window, the weight and scale
shapes, the route dtypes and the id range before it packs anything.
`plan.self_cleaning` reports the contract; the grouped fused route keeps its
L1 arrival reset inside `run()`. The separately exported `prepare_grouped_l2`
accepts E4M3 activation values, packed E2M1 weights and natural row-major
packed scale words; it repacks both scale tensors into the group-folded layout
at preparation, `run()` launches only the clustered GEMM, and
`plan.update_scales()` repacks again after the caller changed the scale words
in place.

FP4 packs the earlier K element into the low nibble. The scale granularity is
32 values. A UE8M0 byte contains the biased exponent of a power-of-two scale;
four consecutive bytes make an int32 scale word. The example input helper
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
pytest -q tests/experimental/test_mega_moe_v3_catalog.py tests/experimental/test_mega_moe_v3_preparation.py
```

The second line needs no GPU: `test_mega_moe_v3_catalog.py` checks the
catalog (every route's programs, grid and SM counts, the sources and their
seals, and the TP/EP route matrix with its metadata) and
`test_mega_moe_v3_preparation.py` the host contract (logical width padding,
the expert window, the source-schedule form and the dispatch reference).

The public tests use independent PyTorch math on the exact packed inputs,
check valid dispatch metadata and reusable counters, poison outputs between
launches, and exercise direct and graph replay. They cover both routed
precisions on the 16-token model route, the FP4 single-token route's workspace
lifecycle, the FP4 1024- and 4096-token routes, -1 routes on every pipeline
route, and the catalog's grouped L2 route, and they check that a captured
`run()` is one kernel node. For the tensor-parallel and expert-parallel
routes (skipped while the architecture's catalog lacks one) they check the
reference at the same FP4 tolerance and, bit for bit: direct and graph
replay, a fresh preparation, the logical width against pre-padded weights,
a nonzero rank with -1 slots against the same call remapped to local ids,
an all-remote batch (zeros), and a rank's program against the 384-expert
program on the global weight set with the remote slots set to -1. The
384-expert, 2304-wide inputs need about 48 GiB of free device memory, the
other shapes proportionally less; the tests skip below that. They do not
enumerate every model shape or claim model-wide performance.
