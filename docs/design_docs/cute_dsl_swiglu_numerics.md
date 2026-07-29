# CuTe DSL SwiGLU numerical-alignment experiment

## Status

The activation-only experiment and production integration are complete. The
experiment file is
`benchmarks/bench_cute_dsl_swiglu_numerics.py`; it contains the current fast
CuTe arithmetic and the strict CuTe implementation in one plain-linear kernel.
The strict expression is now shared with the production SM100/SM103 fused-MoE
epilogue behind
`FLASHINFER_DISABLE_CUTE_DSL_FUSED_MOE_NVFP4_SWIGLU_FAST_MATH=1`. The default
remains the existing fast path, with no layout or non-activation arithmetic
changes.

Local setup on 2026-07-28:

- FlashInfer branch: `agent-cutedsl-strict-swiglu-experiment`
- FlashInfer base: upstream `main` at
  `1ca039eb6992f365788f149fbb1dfae993ce920d`
- Local Megatron branch: `miles-main` at
  `4716f7547` (the locally recorded `origin/miles-main` is one checkpointing-only
  commit ahead; the activation files are identical)
- Devbox: `cutedsl-strict-swiglu`
- Required image: `radixark/miles:dev-202607281246`
- Requested hardware: one exclusive 8x B200 devbox in queue `hell`
- Successful final activation-only run:
  `cutedsl_swiglu_deepseek_v3_final_20260728_193500`
- Successful final production-integration run:
  `cutedsl_strict_swiglu_final_20260728_211500`
- Final experiment script SHA-256:
  `9e5d1ebc1e883caf107c2bae01d82bfd70f6a53575422a7a9a039373d69a28ac`
- Final production kernel module SHA-256:
  `60c72e5b37734ec4d63da2fd8c7081de681044b5258dee126c56229381772582`

## Scope

The only quantity under test is elementwise SwiGLU arithmetic.

Included:

- A dependency-free mirror of Megatron's owning forward expression:
  `F.silu(gate) * up`
- The current SM100 CuTe DSL MoE arithmetic:
  fast `exp2`, approximate reciprocal, and FP32 multiplies
- A strict CuTe DSL candidate:
  non-fast `exp` plus explicit round-to-nearest FP32 add, divide, and multiply
- FP32, which matches the FlashInfer epilogue accumulator domain
- BF16, as a separate source-expression/storage diagnostic

Excluded:

- GEMM and its accumulation differences
- routing weights
- bias
- quantization
- MoE permutation, unpermutation, and combine
- any FlashInfer interleave, swizzle, scale-factor, or expert layout
- NaN and infinity contract differences
- backward arithmetic
- performance

The closure claim is FP32-only. With BF16 inputs, eager
`F.silu(gate) * up` materializes and rounds the BF16 SiLU result before the
separate multiply. Both CuTe paths promote the BF16 inputs to FP32, keep SiLU
and the up multiply in FP32, then round once at the output store. Consequently,
the BF16 comparison includes a staging difference unrelated to fast
exponential or reciprocal accuracy. It is reported to make that boundary
visible, but it is not used to decide whether strict activation arithmetic
closes the fast-math gap.

Per the experiment constraint, all implementations consume one ordinary
contiguous tensor of shape `[7168, 2 * 2048]`. The first half is gate and the
second half is up, producing `[7168, 2048]`. The model-hidden value 7168 is a
requested workload extent in this pointwise-only grid. In the real MoE path,
model hidden is the excluded GEMM K dimension and the post-FC1 leading axis is
token/expert rows. The numerical comparison remains per-element and introduces
no new memory format. The CuTe kernel performs only scalar linear loads,
elementwise arithmetic, and scalar linear stores.

## Owning source paths

Megatron's activation is
`megatron/core/fusions/fused_bias_swiglu.py::swiglu`:

```python
y_1, y_2 = torch.chunk(y, 2, -1)
return F.silu(y_1) * y_2
```

The experiment mirrors these two lines locally and deliberately does not import
Megatron. This keeps the file standalone while preserving the source-level
activation contract. Compiler-wrapper behavior and the router-probability
multiplication in Megatron's weighted MoE wrapper are outside this experiment.

The B200 FlashInfer path is the vectorized activation epilogue in
`flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`.
For standard SwiGLU defaults (alpha 1, beta 0, finite values), its activation is:

```text
exp_value = exp2(-gate * log2(e), fastmath=True)
sigmoid = rcp_approx(1 + exp_value)
output = (sigmoid * gate) * up
```

The activation receives FP32 accumulators. Conversion to BF16 or FP4 happens
after this arithmetic and is not part of the primary FP32 comparison.

The strict candidate is:

```text
denominator = fadd_rn(1, exp(-gate, fastmath=False))
silu = fdiv_rn(gate, denominator)
output = fmul_rn(silu, up)
```

`fadd_rn`, `fdiv_rn`, and `fmul_rn` are existing FlashInfer CuTe DSL helpers in
`flashinfer/cute_dsl/fp4_common.py`; the experiment introduces no new numeric
helper.

## Production integration design

The production toggle is resolved into an immutable boolean when
`CuteDslMoEWrapper`, the functional API runner, or the unified
`CuteDslNvfp4Runner` is constructed:

```bash
FLASHINFER_DISABLE_CUTE_DSL_FUSED_MOE_NVFP4_SWIGLU_FAST_MATH=1
```

Only the exact string `1` enables strict mode. Unset, `0`, and other strings
retain the existing fast arithmetic. A wrapper captures the setting at
construction so its autotune identity, compiled kernel specialization, and
CUDA graph all agree. Changing the setting therefore requires constructing a
new wrapper and recapturing its CUDA graph. Every distributed rank must receive
the same environment setting.

The concrete boolean participates in both layers of relevant caching:

- `CuteDslFusedMoENvfp4Runner.__hash__` and its persistent autotune cache extras
- the process-local `_gather_kernel_cache` key used around `cute.compile`

The strict branch changes only the gated elementwise sigmoid and final
activation multiplies. It uses the shared `_strict_swiglu_f32` helper with
non-fast `exp` plus explicit round-to-nearest FP32 add, divide, and multiply.
The default branch retains its existing packed-FP32 `exp2(..., fastmath=True)`
and `rcp_approx` sequence. ReLU2, GEMM, output quantization reciprocals,
routing, permutation, combine, and tensor layouts are unchanged.

This phase is intentionally scoped to the standard SM100/SM103
`cute_dsl_fused_moe_nvfp4` backend used by SGLang's CuTe-DSL v2 rollout path.
The environment variable includes `FUSED_MOE_NVFP4` to make that ownership
boundary explicit rather than implying a cross-backend global policy.
The separate MegaMoE and SM12x implementations, and SGLang's DeepEP v1
activation-plus-quantization path, do not consume this flag. The
Megatron-alignment claim remains limited to ordinary SwiGLU defaults
(`alpha=1`, `beta=0`, no effective clamp); the strict arithmetic is also
available to parameterized/clamped SwiGLU but is not described as a Megatron
equivalence result.

## Prediction and closure rule

Setting only `fastmath=False` on the exponential would be insufficient because
the current sigmoid also uses `rcp_approx`. The strict path therefore replaces
both approximations while preserving FP32 operation order.

The script makes a deliberately strong FP32 closure claim:

1. At least one FP32 case must contain a value mismatch from current-fast CuTe.
2. Strict CuTe must value-match the reference on every selected FP32 case.
3. Strict CuTe must close every observed current-fast FP32 gap.

Numeric equality treats `+0` and `-0` as the same value. ULP reporting also
collapses signed zero and uses monotonic FP32/BF16 bit ordering.

## Deterministic input matrix and metrics

The default DeepSeek-V3-sized grid has 14,680,064 gate/up pairs
(`7168 x 2048`) for each case:

- `edge`: repeats exact signed zeros, `2^-20`, and representative positive and
  negative gates with exact signed up values
- `sweep`: gate uniformly covers `[-20, 20]`; up cycles through exact signed
  powers-of-two-like values
- `normal`: independent standard-normal gate and up
- `wide`: gate standard deviation 6 and up standard deviation 3

Random values are generated on CPU with seed `20260728`, then converted to the
selected dtype and copied to the B200.

For current fast CuTe and strict CuTe comparisons against the local Megatron
mirror, the script reports exact-match fraction/count, mean absolute error,
RMSE, maximum absolute error, relative L2 error, absolute-error quantiles, and
the worst input. It also reports maximum/mean ULP distance, fraction within one
ULP, exact elements recovered/regressed by strict arithmetic, fast-to-strict
error ratios, and the FP32 closure verdict.

## Environment setup and command log

Local FlashInfer branch setup:

```bash
cd /Users/ziangli/playground/cute-dsl-nvfp4/flashinfer
git fetch upstream main
git switch -c agent-cutedsl-strict-swiglu-experiment upstream/main
```

Local devbox request:

```bash
cd ~/rl/hai
uv run h devbox -c c1 \
  --size exclusive \
  --gpu-type b200 \
  --gpus-per-pod 8 \
  --image radixark/miles:dev-202607281246 \
  --bare \
  --queue hell \
  cutedsl-strict-swiglu
```

Before every sync, run the checks from the local FlashInfer checkout. The
explicit `--files` invocation includes the two new, initially untracked files:

```bash
pre-commit run --files \
  benchmarks/bench_cute_dsl_swiglu_numerics.py \
  docs/design_docs/cute_dsl_swiglu_numerics.md
pre-commit run --all-files
```

Sync the checked source:

```bash
cd ~/rl/hai
uv run h sync -c c1 \
  --source-dir /Users/ziangli/playground/cute-dsl-nvfp4/flashinfer \
  --remote-path /sgl-workspace/flashinfer \
  cutedsl-strict-swiglu
```

The following consolidates the traced setup and final run into one reproducible
sequence. The optional-package uninstall was performed in an earlier traced
smoke setup; those packages were already absent for the final run. The workflow
reinstalls the synced FlashInfer checkout without dependency resolution,
records the Megatron source revision and expression, and stages its complete log
directory from the same shell. It allocates a fresh run ID and refuses to merge
with an existing local or staging directory:

```bash
set -euo pipefail
RUN_ID="${RUN_ID:-cutedsl_swiglu_deepseek_v3_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="/sgl-workspace/logs/$RUN_ID"
STAGE_ROOT=/data/home/ziangli/agent/warp_logs
if [[ -e "$LOG_DIR" || -e "$STAGE_ROOT/$RUN_ID" ]]; then
  echo "refusing to reuse run ID: $RUN_ID" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" "$STAGE_ROOT"
LOG="$LOG_DIR/run.log"

stage_logs() {
  set +x
  cp -a "$LOG_DIR" "$STAGE_ROOT/"
}
trap stage_logs EXIT

exec > >(tee "$LOG") 2>&1
set -x
echo "RUN_ID=$RUN_ID"

cd /sgl-workspace/flashinfer
python -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache
python -m pip install --no-build-isolation --no-deps -e . -v
flashinfer show-config
flashinfer module-status --detailed
python -m flashinfer.collect_env
git -C /root/Megatron-LM rev-parse HEAD
git -C /root/Megatron-LM status --short --branch
sed -n '15,27p' \
  /root/Megatron-LM/megatron/core/fusions/fused_bias_swiglu.py
python benchmarks/bench_cute_dsl_swiglu_numerics.py \
  --dtype all \
  --case all \
  --json-out "$LOG_DIR/results.json"
```

The exact-source production validation used these focused commands after the
editable reinstall and environment capture:

```bash
STRICT_SWIGLU_ENV=FLASHINFER_DISABLE_CUTE_DSL_FUSED_MOE_NVFP4_SWIGLU_FAST_MATH
TEST=tests/moe/test_cute_dsl_fused_moe.py

python -m pytest -q \
  "$TEST::TestStrictSwiGLUConfiguration"
python benchmarks/bench_cute_dsl_swiglu_numerics.py \
  --dtype float32 \
  --case all \
  --json-out "$LOG_DIR/activation_results.json"
env -u "$STRICT_SWIGLU_ENV" python -m pytest -q \
  "$TEST::TestCuteDslFusedMoeFunctional::test_deterministic_finalize_numerical_accuracy" \
  -k swiglu-per-tensor
env "$STRICT_SWIGLU_ENV=1" python -m pytest -q \
  "$TEST::TestCuteDslFusedMoeFunctional::test_deterministic_finalize_numerical_accuracy" \
  -k swiglu-per-tensor
env "$STRICT_SWIGLU_ENV=1" python -m pytest -q \
  "$TEST::TestCuteDslMoEWrapper::test_wrapper_cuda_graph[256-64-False-False]"
```

The trap performs the remote staging. Download the explicitly named final run
from the local HAI checkout:

```bash
cd ~/rl/hai
uv run h warp \
  c1:/data/home/ziangli/agent/warp_logs/cutedsl_swiglu_deepseek_v3_final_20260728_193500 \
  /Users/ziangli/playground/cute-dsl-nvfp4/logs/
```

The final environment was:

- NVIDIA B200, SM100, driver `580.126.09`
- PyTorch `2.11.0+cu130`, CUDA toolkit/runtime `13.0`
- `nvidia-cutlass-dsl 4.5.2`
- editable `flashinfer-python 0.6.15`; `flashinfer-cubin` and
  `flashinfer-jit-cache` absent
- Megatron `miles-main` at
  `4716f75475c78e2fc2c6f0d3af095f1681b770b4`

The editable build emitted best-effort warnings when its build hook tried to
install optional CUDA/NCCL dependencies into the externally managed system
interpreter. The required packages were already installed, the editable wheel
installed successfully, and both environment inspection and the experiment
completed. The first setup attempt used dependency resolution and then hit a
stale `flashinfer-cubin 0.6.12` versus editable `flashinfer-python 0.6.15`
version check. Removing the optional binary packages and reinstalling with
`--no-deps` resolved that issue.

## Results

The final JSON reports:

```text
fast_gap_demonstrated_and_closed = true
observed_fast_gap_case_count = 4
strict_value_matches_reference_on_all_cases = true
all_observed_fast_gaps_closed = true
```

FP32 results, with 14,680,064 values per case:

| Case | Current-fast mismatches | Fast RMSE | Fast max abs | Fast max ULP | Fast within 1 ULP | Strict mismatches | Strict max ULP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| edge | 2,097,152 (14.2857%) | 1.593e-08 | 5.960e-08 | 1 | 100.0000% | 0 | 0 |
| sweep | 6,810,336 (46.3917%) | 2.668e-07 | 7.629e-06 | 17 | 71.0913% | 0 | 0 |
| normal | 4,605,313 (31.3712%) | 3.205e-08 | 1.907e-06 | 6 | 93.6277% | 0 | 0 |
| wide | 6,179,697 (42.0958%) | 4.099e-07 | 1.526e-05 | 25 | 80.7675% | 0 | 0 |

Across the four FP32 cases, current-fast CuTe mismatched
19,692,498 / 58,720,256 values (33.5361%). Strict CuTe value-matched the local
Megatron source-expression mirror for all 58,720,256 values: zero RMSE, zero
maximum absolute error, and zero maximum ULP distance. This closes the observed
activation-arithmetic gap under the experiment's finite-value, alpha-1,
beta-0, FP32 contract.

The final production-integration run used the exact script and production
kernel hashes recorded above and passed:

- 9 strict-mode configuration, capture, autotune-key, and compile-cache tests
- 1 default-fast full fused-MoE functional case
- 1 strict full fused-MoE functional case
- 1 strict wrapper CUDA-graph capture/replay case

The default-fast functional run included a cold build of FlashInfer's existing
MoE routing extension and completed in 537.22 seconds. With that cache warm,
the strict functional and CUDA-graph cases each completed in 9.68 seconds.
These timings are validation context, not a strict-versus-fast performance
comparison.

BF16 confirms the expected staging boundary rather than serving as a closure
test. `edge` and `sweep` round both CuTe variants and the eager source expression
to the same BF16 values. For `normal` and `wide`, both CuTe variants remain
within one BF16 ULP of the eager expression, but the intermediate BF16 rounding
inside eager `F.silu(gate) * up` dominates: strict does not reduce the aggregate
error. On `wide`, only 21 of 14,680,064 exact-match positions differ between
fast and strict, while both have the same RMSE and maximum error.

Artifacts are under
`/Users/ziangli/playground/cute-dsl-nvfp4/logs`:

- Final machine-readable result:
  `cutedsl_swiglu_deepseek_v3_final_20260728_193500/results.json`
- Final traced environment and run log:
  `cutedsl_swiglu_deepseek_v3_final_20260728_193500/run.log`
- Final production-integration result:
  `cutedsl_strict_swiglu_final_20260728_211500/activation_results.json`
- Final production-integration traced log:
  `cutedsl_strict_swiglu_final_20260728_211500/run.log`
- Initial package-conflict log:
  `cutedsl_swiglu_smoke_20260728_185210/run.log`
- Reporting-layer failure log:
  `cutedsl_swiglu_smoke_20260728_185446/run.log`
- Successful small smoke:
  `cutedsl_swiglu_smoke_20260728_190115/`

These results do not claim trainer/rollout equality. The full fused-MoE cases
are integration smoke tests; the numerical closure claim remains
activation-only. GEMM, quantization, routing, permutation, combine, and BF16
intermediate-staging differences remain intentionally outside this phase.
