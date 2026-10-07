# Fused routing gate

`from flashinfer.mega_gate import prepare_mega_gate` prepares the fused BF16
routing GEMM `x[M,K] @ weight[E,K]^T`, scoring, expert-group top-k selection,
unbiased weight normalization and logical-to-physical expert mapping on SM100a
and SM103a. The prepared call returns a plan; `plan.run()` submits one fused
kernel on the current PyTorch stream and returns `(expert_indices, weights)`:
int64 `[M,topk+shared]` physical expert ids and FP32 normalized weights scaled
by `routed_scaling_factor`.

The exported programs use K=5120, E=384, top-k 6, `sqrtsoftplus` scoring and
an FP32 expert bias. Any token count is accepted: the token count, the
SM-count-derived worker stride and the physical-map / logical-output flags are
kernel arguments, and the DeepGEMM configuration for `M` is snapped onto the
exported schedule templates (`mega_gate.TEMPLATES`, one compiled program each:
block tokens, MMA CTA pair, split-K, expert groups, pipeline stages, gate
warpgroups, K-block merge, the single-token cluster-reduction route, the
2..4-token cluster-reduction route of SM100a and the small-M idle L2 touch; the
single-tile routes of 1..4 tokens run 16 K-splits over a 16-CTA cluster and
combine the FP32 partials with a compensated pairwise sum). One source per program serves both architectures.
The small physical-map production token counts (1, 3, 16, 128, 512) and the
2..16-token logical-output route keep an exact-shape program of their schedule
(tile geometry, route and worker stride compiled in; one 16-token tile serves
2..16 tokens, one program per route), selected on an exact geometry match;
every other token count runs the runtime program.
Physical mapping (`to_physical_map`, `logical_count`), a logical
`unmapped_topk_idx` output and deterministic routing (the split-K-free
template) combine freely. Other scoring functions, image bias, token masks,
fixed and random routing are part of the kernel ABI but have no exported
program.

Optional `scratch` (FP32 split-K partials) and `score_barriers` (uint64
launch-epoch barriers, zeroed once before first use) are caller-owned.
Preparation owns all allocations and device queries; `run()` submits the
prepared kernel with launch overhead only, without allocation, and is safe to
capture in a CUDA Graph. Operand values, bias and mapping tables may change in
place between runs. Do not concurrently reuse one plan's outputs or workspace.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_mega_gate/generated`.
The runtime (template selection, JIT loader, plan) is `mega_gate.py` in this
directory; the public API is `flashinfer/mega_gate.py`.

```bash
pytest tests/experimental/test_mega_gate_generated.py -q
```

The test covers the eleven production rows (M in 1, 3, 16, 128, 512, 1024,
2048, 4096, 8192 with physical mapping; deterministic M=16 with a logical
output; logical M=16 without a map) with independent expected indices and
weights, physical duplicate mapping, changed bias values, graph replay and a
non-default stream, plus held-out token counts (2, 7, 33, 100, 777, 3000,
5000) on every route against an FP32 reference.
