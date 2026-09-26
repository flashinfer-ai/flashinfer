# Fused routing gate

`from flashinfer.mega_gate import prepare_mega_gate` prepares the fused BF16
routing GEMM `x[M,K] @ weight[E,K]^T`, scoring, expert-group top-k selection,
unbiased weight normalization and logical-to-physical expert mapping on SM100a
(148 SMs) and SM103a (152 SMs). The prepared call returns a plan; `plan.run()`
submits one fused kernel on the current PyTorch stream and returns
`(expert_indices, weights)`: int64 `[M,topk+shared]` physical expert ids and
FP32 normalized weights scaled by `routed_scaling_factor`.

The exported routes use K=5120, E=384, top-k 6, `sqrtsoftplus` scoring and an
FP32 expert bias. With physical mapping (`to_physical_map`, `logical_count`)
they cover M=1, 3, 16, 128, 512, 1024, 2048, 4096 and 8192. Two feature
routes cover M=16: deterministic routing with a logical `unmapped_topk_idx`
output (`ep_rank=7`), and logical routing without a physical map. Only
configurations present in the exported physical catalog are accepted; other
scoring functions, image bias, token masks, fixed and random routing are
part of the kernel ABI but have no exported route.

Optional `scratch` (FP32 split-K partials), `score_barriers` (uint64
launch-epoch barriers, zeroed once before first use) and descriptor workspace
storage are caller-owned. Preparation owns all allocations; `run()` submits the
prepared kernel with launch overhead only, without allocation, and is safe to
capture in a CUDA Graph. Operand values, bias and mapping tables may change in
place between runs. Do not concurrently reuse one plan's outputs or workspace.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_mega_gate/generated`.
The runtime and catalog are in this directory. The public API is
`flashinfer/mega_gate.py`; see `examples/experimental/mega_gate.py`.
Build/install FlashInfer with its supported CUDA toolchain before running:

```bash
python examples/experimental/mega_gate.py
pytest tests/experimental/test_mega_gate_generated.py -q
```

The test covers all eleven exported routes: independent expected indices and
weights, physical duplicate mapping, changed bias values, graph replay and a
non-default stream. Performance qualification measures complete prepared calls
with prepared inputs and retains all nine model rows plus the two feature rows.
Per-row exported execution must remain within 3% of its source route.
Performance results accompany the generated bundle; preparation is excluded
from both timed arms.
