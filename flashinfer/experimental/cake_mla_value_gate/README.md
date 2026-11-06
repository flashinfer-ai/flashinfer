# Cake MLA value projection + sigmoid output gate (experimental, SM100 / SM103)

Generated-program backend for the decode tail of gated multi-head latent
attention (Kimi-K3): the per-head value up-projection of the attention's
latent output fused with the sigmoid output gate, in one launch.

Both the API and the backend are experimental (see
[`../README.md`](../README.md)).  Calling
`flashinfer.cake_mla_value_gate.prepare_mla_value_gate` is the opt-in; no
environment variable is involved and the backend takes part in no automatic
routing (dispatch, autotuning or trace-apply).

## Operation

For the token rows `t < B` of one rank with `H` heads, `h < H`, `v < 128`:

```text
S[t, h, v]        = sum_k latent[t, h, k] * w_uv[h, k, v]                 (k < 512, FP32 accumulation)
out[t, h*128 + v] = bf16( f32(bf16(S[t, h, v])) * sigmoid_fp32(f32(gate[t, h*128 + v])) )   if valid_lengths[t // q] > 0
out[t, h*128 + v] = +0.0                                                    if valid_lengths[t // q] <= 0
```

```python
from flashinfer.cake_mla_value_gate import prepare_mla_value_gate

runner = prepare_mla_value_gate(w_uv, max_rows=64, sigmoid_precision="fp32")  # outside capture
runner.launch(latent, gate, valid_lengths, out)  # returns out; graph-capturable
```

| tensor | shape / dtype | layout |
| --- | --- | --- |
| `w_uv` (bound at prepare) | `[H, 512, 128]` BF16 | strides `(65536, 1, 512)`: the view of a contiguous `[H, 128, 512]` tensor (latent axis contiguous, e.g. `w_vc.contiguous().transpose(1, 2)`); 16-byte aligned |
| `latent` | `[B, H, 512]` BF16 | contiguous, 16-byte aligned (a `[:B]` slice of a persistent buffer is fine) |
| `gate` | `[B, H * 128]` BF16 logits | unit column stride, even row pitch `gate.stride(0) >= H * 128`, rows 4-byte aligned; e.g. `projected[:, 2112:]` of a `[B, 3648]` fused projection output |
| `valid_lengths` | `[B // q]` int32 | contiguous |
| `out` | `[B, H * 128]` BF16, caller-owned | contiguous, 16-byte aligned, no overlap with any input |

`1 <= B <= max_rows <= 64`, `B % q == 0` (`q = rows_per_request`), all tensors on
the device of `w_uv`.  The tests cover `H` in {1, 6, 12, 16} (12 is the
Kimi-K3 TP8 rank) and `B` in {1..9, 16, 17, 24, 25, 32, 33, 48, 64}, on both
programs (see below).

## Numerics

- The products are BF16 x BF16 (exact in FP32), accumulated in FP32 on the
  tensor cores; the projection is rounded to BF16 (round to nearest even) once.
- `sigmoid_precision="fp32"` is the only mode, and it is a required argument so
  that the numerics are explicit at every call site.  The sigmoid is evaluated
  in FP32 and multiplied unrounded with the BF16 projection; the product is
  rounded to BF16 once.  It follows SGLang `MlaValue`'s FP32 sigmoid
  `1 / (1 + exp(-g))`: a fast-math exponential, an approximate reciprocal
  refined by one Newton step, and IEEE division when `1 + exp(-g)` is at least
  `2**126` or not finite.  It is therefore within a few FP32 units in the last
  place of the exact sigmoid, not correctly rounded.
- The path that rounds the sigmoid to BF16 before the multiply (as a separate
  `x * torch.sigmoid(gate)` in BF16 does) produces different bits; this API
  does not offer it.
- No atomics, no split-K: results are bitwise reproducible, and each row does
  not depend on `B` (batch-invariant).
- The tests compare against a float64 model with the same rounding boundaries
  (BF16 projection, FP32-range sigmoid, BF16 output), with a per-element bound
  derived from the FP32 accumulation and sigmoid error and a cap on the
  fraction of elements that differ from the model's rounded value.

## Valid and padded rows

Row `t` is valid if and only if `valid_lengths[t // q] > 0`; a zero or negative
length marks an invalid (padded) request.  An invalid row is
written as `+0.0` (bits `0x0000`) whatever its latent or gate holds (padded
requests may carry NaN gates).  A valid row with a NaN input gives NaN.  Rows
`t >= B` of the caller's buffers, and gate bytes between rows (the pitch
padding), are never read or written.  A caller whose attention shards the KV
cache passes the global lengths: a request with an empty local shard is still
a valid row.

## Programs: 16-row and 8-row tiles

Each architecture has two programs.  They differ only in the token rows one
CTA owns; both give the same bits for every input, so the choice affects time
only.

| program | grid | CTA owns | dynamic shared memory |
| --- | --- | --- | --- |
| 16-row | `(4, H, ceil(B / 16))` | 32 value columns x 16 rows | 17408 B |
| 8-row | `(4, H, ceil(B / 8))` | 32 value columns x 8 rows | 8704 B |

For every `B` in `[1, max_rows]`, `prepare_mla_value_gate` decides once
(outside capture) which program a launch of `B` rows runs:

```text
8-row program   if enable_pdl  or  4 * H * ceil(B / 8) <= SM count of the device
16-row program  otherwise
```

At `H = 12` the 8-row grid fits one wave for `B <= 24` on B200, B300 (148 SMs)
and GB300 (152 SMs).  `runner.program_for(B)` and `runner.tile_rows_for(B)`
report the choice, `runner.programs` lists the loaded programs and
`runner.sm_count` the SM count used.  Only the programs some `B` uses are
loaded.

## Prepare, launch and CUDA graphs

- `prepare_mla_value_gate` checks the device (exact SM100a / SM103a), the
  weight's dtype, shape, strides and alignment, decides the program of every
  `B`, loads or JIT-builds those programs and caches the device facts.  It raises `RuntimeError` during
  CUDA graph capture: prepare before capturing.
- `runner.launch` checks host metadata only (shapes, strides, dtypes, devices,
  alignment, overlap), looks up the program decided for `B`, computes its grid
  on the host and submits one kernel on the current torch stream without
  allocating or synchronizing.  It may be
  captured in a CUDA graph.
- A replay re-reads the contents of the captured `latent`, `gate`,
  `valid_lengths` and `w_uv` and rewrites `out`; keep their storage alive and
  in place.  Every distinct `B` or tensor address is its own capture.
- The runner keeps a reference to `w_uv`.  In-place updates of its contents are
  seen by later launches.  A new or re-aliased weight tensor (for example after
  a weight reload that replaces the storage) is not seen: call
  `prepare_mla_value_gate` again and re-capture.

## Programmatic dependent launch

With `enable_pdl=True` (the default) the kernel may start while the preceding
kernel of the stream still runs:

- it reads `w_uv` **before** it waits on that kernel (`griddepcontrol.wait`),
  so the preceding kernel must not write `w_uv`;
- it reads `latent`, `gate` and `valid_lengths`, and writes `out`, only after
  the wait;
- it lets its dependents launch early (before the projection); a following
  kernel launched with programmatic dependent launch must wait before it reads
  `out`.

With `enable_pdl=False` the launch is an ordinary stream-ordered launch.

## Limitations

- Exact SM100a (compute capability 10.0) and SM103a (10.3) devices only;
  `prepare_mla_value_gate` raises `NotImplementedError` elsewhere.
- `K = 512` latent and `N = 128` value columns per head are fixed.
- Release wheels on PyPI omit the generated sources; the first load raises
  there (use a nightly wheel or a source install).

## Files

| file | content |
| --- | --- |
| `cake_backend.py` | weight and launch checks, the runner, the logical argument values (hand-written) |
| `cake_jit.py` | the host vocabulary, registry and loader glue (hand-written); the `MODULES` records (one per program; `grid_rule["rows"]` is the row tile the host selects on) are written by the exporter |
| `build.json`, `cake_build.py` | the build manifest and the pinned loader (generated) |
| `csrc/` | the generated kernels (`<program>_kernel.cu`), their TVM-FFI bindings (`<program>_binding.cu`) and `cake_ffi_checks.h` (generated; digest-bound by `build.json`) |

Program names are `cake_mla_value_gate_<20 hex>`.  Tests:
`pytest tests/experimental/test_cake_mla_value_gate.py`; example:
`python examples/cake_mla_value_gate.py`; benchmark:
`python benchmarks/bench_cake_mla_value_gate.py`.

## Ownership and graduation

- Owner: @yyihuang.
- Tracking issue: flashinfer-ai/flashinfer#8795.
- Graduation plan: the feature leaves the experimental track when (a) an
  adapter in a real engine consumer is qualified on its target GPU, and (b) the
  maintainers agree on the API name and on how its shape is generalized (head
  count, decode widths, sigmoid modes).  Until then it is reviewed on the
  experimental policy's cadence and may change or be removed.

## Attribution

The computation, its tiling and the valid-row and programmatic-dependent-launch
contract are derived from SGLang's `MlaValue`
(`python/sglang/kernels/ops/attention/kimi_k3_specialized/mla_value.py` and the
sigmoid of `python/sglang/kernels/jit/kimi_k3/primitives.py` at
sgl-project/sglang commit `1b9493001ba7a8aa023c761dcac0db0ab3847f72`, pull
request sgl-project/sglang#48510), Copyright contributors to the SGLang
project, licensed under the Apache License, Version 2.0.  That kernel was
originally written by Lee Nau in sgl-project/sgl-nvidia-mega pull request #17.
The kernels here are a re-implementation, not a copy of that source.
