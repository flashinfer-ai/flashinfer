# Fused routing gate

`from flashinfer.mega_gate import prepare_mega_gate` prepares the fused BF16
routing GEMM `x[M,K] @ weight[E,K]^T`, scoring, expert-group top-k selection,
unbiased weight normalization and logical-to-physical expert mapping on SM100a,
SM103a and SM107a. The prepared call returns a plan; `plan.run()` submits one
fused kernel on the current PyTorch stream and returns `(expert_indices,
weights)`: int64 `[M,topk+shared]` physical expert ids and FP32 normalized
weights scaled by `routed_scaling_factor`.

The exported programs use K=5120, E=384, top-k 6, `sqrtsoftplus` scoring and
an FP32 expert bias. Any token count is accepted: the token count, the
SM-count-derived worker stride and the route flags (physical map, logical
output, token mask, image bias) are kernel arguments, and the DeepGEMM
configuration for `M` is snapped onto the exported schedule templates
(`mega_gate.TEMPLATES`, one compiled program each: block tokens, MMA CTA pair,
split-K, expert groups, pipeline stages, gate warpgroups, K-block merge, the
single-token cluster-reduction route, the 2..4-token cluster-reduction route of
SM100a and the small-M idle L2 touch; the single-tile routes of 1..4 tokens run
16 K-splits over a 16-CTA cluster and combine the FP32 partials with a
compensated pairwise sum). The 193..960-token rows run DeepGEMM's own 48, 64
and 80-token single-CTA blocks and 112, 128 and 160-token CTA-pair blocks. One
source per program serves every architecture it is delivered for.
The small physical-map production token counts (1, 3, 16, 128, 512) and the
2..16-token logical-output route keep an exact-shape program of their schedule
(tile geometry, route and worker stride compiled in; one 16-token tile serves
2..16 tokens, one program per route), selected on an exact geometry match;
every other token count runs the runtime program.
Physical mapping (`to_physical_map`, `logical_count`), a logical
`unmapped_topk_idx` output, the token mask (`mask`, bool `[M]`: a masked token
takes expert -1 with weight 0 in every slot), the image bias (`image_bias`,
FP32 `[E]`, with `image_token_mask`, bool `[M]`: an image token ranks experts by
its score plus the image bias instead of the bias) and deterministic routing
(the split-K-free templates: up to 16 tokens run one paired-CTA 32-token block
with two MMA CTAs per expert group; larger counts keep the MMA CTA pairing
DeepGEMM selects for them and run that pairing's split-K-free block whatever
block size DeepGEMM chose) combine freely. Other scoring functions, fixed and
random routing are part of the kernel ABI but have no exported program.

`enable_pdl` (default: when the device supports it) launches with programmatic
dependent launch: the kernel waits on the preceding kernel before its first
global read and lets the following kernel launch right after.

Optional `scratch` (FP32 split-K partials) and `score_barriers` (uint64
launch-epoch barriers, zeroed once before first use) are caller-owned.
Preparation owns all allocations and device queries; `run()` submits the
prepared kernel with launch overhead only, without allocation, and is safe to
capture in a CUDA Graph. Operand values, bias, masks and mapping tables may
change in place between runs. Do not concurrently reuse one plan's outputs or
workspace.

## Package layout

The backend is a self-contained package. `cake_jit.py` is the registry of the
generated programs: one `MODULES` record per program carrying the host plan
(its schedule template, the architectures it is delivered for, its launch
record and the argument plan of its `run` entry) and the loader glue
(`template`, `load`). What to compile is recorded once in `build.json`: the
generated CUDA sources under `csrc/`, their digests and nvcc flags, one entry
per (program, architecture). `cake_build.py` loads a prebuilt library from
`aot/` when a builder compiled one and otherwise JIT-builds the manifest entry
through FlashInfer from the same sources and flags.
`scripts/check_cake_build_manifests.py` checks the manifest against the
registry and the sources without importing the package. `DEEPGEMM_NOTICE.txt`
at the package root carries the MIT notice of the DeepGEMM portions the
generated kernels derive from.

`mega_gate.py` keeps the template selection (`deepgemm_config`,
`select_template`) and the prepared plan; the public API is
`flashinfer/mega_gate.py`; see `examples/experimental/mega_gate.py`.

```bash
pytest tests/experimental/test_mega_gate_generated.py -q
```

The test covers the production rows (M in 1, 3, 16, 128, 512, 1024, 2048,
4096, 8192 with physical mapping; deterministic M=16 with a logical output;
logical M=16 without a map) with independent expected indices and weights,
physical duplicate mapping, changed bias values, graph replay and a
non-default stream; held-out token counts (2, 7, 33, 100, 400, 777, 3000,
5000) on every route against an FP32 reference; the token mask, the image
bias and both PDL launch modes against the same reference; the host plan of
every registry record and the `build.json` contract.
