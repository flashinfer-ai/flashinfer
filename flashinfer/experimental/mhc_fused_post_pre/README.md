# Experimental fused mHC post -> pre (SM103)

`flashinfer.mhc.mhc_fused_post_pre` computes the boundary between two mHC
blocks in one call: the post map of block `i` followed by the pre map
(projection, RMS scaling, sigmoid gates, Sinkhorn) and the input RMSNorm of
block `i + 1`. Unfused, this is three kernels: `mhc_post`, the TF32 pre-norm
projection GEMM and `mhc_pre_big_fuse_with_prenorm`.

Both the **API** and the **backend** are experimental: they provide no
compatibility guarantees and may change or be removed without deprecation.

## Opt-in and routing

- Calling `mhc_fused_post_pre`, `mhc_fused_post_pre_supported` or
  `mhc_fused_post_pre_warmup` (all marked `@flashinfer_experimental_api`) is
  the opt-in; each emits one `ExperimentalWarning` per process. No environment
  variable is involved.
- The op takes no `backend` argument and is not reachable from any stable
  API: it takes no part in `backend="auto"` dispatch, autotuning or
  trace-apply (it has no trace template).
- JIT-only: the module is compiled on first use and is not part of the
  `flashinfer-jit-cache` / `flashinfer-cubin` packages.

## Supported surface and limitations

One specialized kernel launch computes all four outputs when every check
passes:

- compute capability 10.3 (`sm_103a`, CUDA 13), one CUDA device;
- hidden size 4096, four residual streams (`hc_mult = 4`), 24 mixes;
- `rms_eps = 1e-5`, `hc_pre_eps = hc_sinkhorn_eps = 1e-6`,
  `hc_post_mult_value = 2.0`, `sinkhorn_repeat = 20`, `norm_eps = 1e-5`;
- 33 <= tokens <= 16384;
- bf16 `x` / `residual` / `norm_weight` and outputs, fp32 mixes / `fn` /
  `hc_scale` / `hc_base`; dense row-major operands, 16-byte aligned (`fn`:
  32-byte), outputs not aliasing inputs or each other.

Anything else runs the composable PyTorch implementation in `reference.py`
(a correctness path, not a fast one), or, with `fallback=False`, computes
nothing and returns `None` so that a caller keeps its own implementation.
`mhc_fused_post_pre_supported(...)` answers "would this call run the
specialized kernel?" without launching or allocating anything.

The module is built, loaded and prepared (kernel attributes and co-resident
cluster capacity per device) on the first call outside CUDA graph capture,
never under capture. Call `mhc_fused_post_pre_warmup()` before capturing
graphs whose first call may be this op; an unprepared call under capture is
served by the fallback (or declined), and that is what the graph records.

Numerics: no fast-math. `expf` is the accurate libdevice `expf`, divisions are
IEEE round-to-nearest, the projection is TF32 (`fn` rounded to nearest, bf16
`residual_cur` exact) with FP32 accumulation, and the RMS statistic of
`layer_input_cur` is taken from the pre-rounding FP32 weighted sum.

## Example

```python
import torch
from flashinfer.mhc import mhc_fused_post_pre, mhc_fused_post_pre_warmup

mhc_fused_post_pre_warmup()  # optional; needed before CUDA graph capture
residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = mhc_fused_post_pre(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm_weight
)
```

`examples/experimental/mhc_fused_post_pre.py` is a runnable version, and
`tests/experimental/test_mhc_fused_post_pre.py` holds the tests (host-only
checks everywhere, kernel-vs-reference checks on SM103).

## Layout

- `__init__.py`: the names the core entry points hand off to.
- `backend.py`: support checks (`SUPPORTED_SURFACE`), module loading and
  per-device preparation, dispatch, the fallback, and `stats()` (dispatch /
  fallback / decline counters, launches per kernel variant, tensor-map cache
  hits, and the token-count span each variant serves on a prepared device).
- `reference.py`: the composable PyTorch implementation.
- `jit.py`: the JIT specification (`sm_103a`, no fast-math).
- `csrc/`: the TVM-FFI binding (operand checks that decline instead of
  throwing, regime selection, per-thread TMA descriptor cache) and the kernels.
  One thread-block cluster of 16 CTAs owns a tile of up to 16 tokens; CTA `r`
  owns hidden slice `[256r, 256r + 256)` of all four streams and keeps its
  24 x 1024 slice of `fn` in registers as TF32 MMA fragments. Partial mixes,
  RMS statistics and pre-mix weights move through distributed shared memory,
  the big tiles through TMA. Five translation units and seven kernel variants
  (decode narrow / wide, stream with two register splits, short-stream,
  crossover, long-stream) cover the whole token range in one module.

## Graduation

Graduation needs a finalized API contract (keyword-only constants), a decision
on wider geometry and architecture coverage, a trace template, AOT
registration, and moving the tests to `tests/mhc/`.
