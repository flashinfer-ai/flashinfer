# Deterministic fused SwiGLU + MXFP8 quantization training backend (SM100 / SM103 / SM107)

Backend of `flashinfer.cake_gated_act_mxfp8_det_train`: the forward and backward of SwiGLU
fused with MXFP8 quantization, for training.  The input `gated_input` `[M, 2K]` BF16 holds
the gate columns `[0, K)` and the up columns `[K, 2K)`.  The forward quantizes
`silu(gate) * up` (`[M, K]`).  The backward quantizes the gate and up gradients (`[M, 2K]`)
from `grad_output` `[M, K]`.  Each value is rounded to BF16 and quantized to E4M3 per
32-element block with an RCEIL E8M0 scale, rowwise, colwise or both.

The facade `flashinfer.cake_gated_act_mxfp8_det_train` is a stable API.  This backend package
is experimental, and its internal names may change.  The facade imports the backend on first
use; `import flashinfer` does not load it.  The supported architectures are SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Another device raises `RuntimeError`.
The package loads its libraries prebuilt from `aot/` when a builder compiled them.  Otherwise
FlashInfer JIT-builds them from the sources and flags that `build.json` records.  The package
reads no environment variable.

## Usage

```python
from flashinfer.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_backward,
    cake_gated_act_mxfp8_det_train_forward,
    is_cake_gated_act_mxfp8_det_train_supported,
)

if is_cake_gated_act_mxfp8_det_train_supported(gated_input.device):
    # gated_input: BF16 [M, 2K]; defaults: rowwise=True, colwise=False
    row, col, row_scales, col_scales = cake_gated_act_mxfp8_det_train_forward(
        gated_input, rowwise=True, colwise=True
    )
    # grad_output: BF16 [M, K]; outputs cover [M, 2K]: gate gradient, then up gradient
    drow, dcol, drow_scales, dcol_scales = cake_gated_act_mxfp8_det_train_backward(
        gated_input, grad_output, rowwise=True, colwise=True
    )
```

`cake_gated_act_mxfp8_det_train_register_custom_ops()` registers the two entries as
`torch.ops.flashinfer.cake_gated_act_mxfp8_det_train_forward` and
`torch.ops.flashinfer.cake_gated_act_mxfp8_det_train_backward`, with fake ops for
`torch.compile`.  Their signatures are `(gated_input, rowwise, colwise)` and
`(gated_input, grad_output, rowwise, colwise)`.  The facade also exposes `ARCHES` and
`SUPPORTED_COMPUTE_CAPABILITIES`.  The backend package exposes the forward, backward and
support-check entries too.  It also exposes `cake_gated_act_mxfp8_det_train_check_inputs`,
which returns `(M, K)` or raises, and `cake_gated_act_mxfp8_det_train_empty_outputs`, which
returns the unwritten outputs.

## Inputs and outputs

- `gated_input` `[M, 2K]` and `grad_output` `[M, K]` (backward only) are BF16, contiguous,
  32-byte aligned CUDA tensors on one device.
- `M` and `K` are positive multiples of 128.
- `M * 2K - 1` fits in a signed int32.
- At least one of `rowwise` and `colwise` is set.

A wrong dtype raises `TypeError`.  Any other violation raises `ValueError`.  An unsupported
device raises `RuntimeError`.  Each entry returns `(row_output, col_output, row_scales,
col_scales)`:

| output | shape | dtype | layout |
|---|---|---|---|
| `row_output` | `[M, N]` | `float8_e4m3fn` | row-major |
| `col_output` | `[M, N]` | `float8_e4m3fn` | column-major, strides `(1, M)`: stored as its `[N, M]` transpose |
| `row_scales` | `[M, N / 32]` | `float8_e8m0fnu` | swizzled 128 x 4 tiles |
| `col_scales` | flat, `N * M / 32` bytes | `float8_e8m0fnu` | swizzled 128 x 4 tiles |

`N` is `K` for the forward and `2K` for the backward.  In the backward, columns `[0, K)` hold
the gate gradient and `[K, 2K)` hold the up gradient.  Both scale tensors use the swizzled
128 x 4 tile layout (512-byte tiles) that Blackwell block-scaled MMA reads.  For a disabled
orientation, the values and scales are zero-size tensors on the input device.

## Programs

Each architecture has six routes, named by direction and outputs: `fwd_row`, `fwd_col`,
`fwd_both`, `bwd_row`, `bwd_col` and `bwd_both`.  A route holds 4 to 7 programs.  The package
has 81 distinct programs and 93 per-architecture libraries in `build.json`.  Each library is
compiled with the exact `-gencode` flag of `sm_100a`, `sm_103a` or `sm_107a`.  There is no
family fallback.

A call runs one program of its route.  `M * K` alone chooses the program.  A program's
`elements_at_least` rule gives a threshold per architecture.  Among the programs whose
threshold is at most `M * K`, the one with the highest threshold runs.  When no threshold
admits the call, the route's single fallback runs.  The fallback is the program with no
threshold on that architecture.  `cake_jit.selected_module(arch, kind, M=M, K=K)` names the
chosen program.  The grid has one CTA per tile: `(K / tile_cols, M / tile_rows, 1)`.  Every
program of a route writes the same bytes, so the program choice never changes the result.

## Numerics

Let `x` be a gate value, `s = 1 / (1 + 2^(-x log2 e))` and `act = x * s`.  The forward
quantizes `act * up`.  The backward quantizes `dgate = (fma(act, 1 - s, s) * grad) * up` in
columns `[0, K)` and `dup = act * grad` in columns `[K, 2K)`.  Each step is one
round-to-nearest FP32 operation.  `2^t` is `ex2.approx.f32`.  The backward does not write a
BF16 intermediate to global memory.

Each value is rounded to BF16 and quantized per 32-element block.  The row outputs use blocks
along rows, and the column outputs use blocks along columns.  The E8M0 scale follows RCEIL.
It starts from the exponent of the block's BF16 absolute maximum.  It adds one when the
mantissa of that maximum is above 1.75.  It then subtracts 8 and clamps at 0.  A block whose
maximum is Inf or NaN gets scale 255, and all its codes are NaN (`0x7F`).  A code is the BF16
value times `2^(127 - scale)`, converted to E4M3 with round to nearest even.

## Determinism

This is the `det` package of the op.  For identical inputs, each output repeats bit for bit
on every supported architecture and for every program of the route:

| entry | output | identical inputs |
|---|---|---|
| forward | `row_output` (E4M3 row values) | repeats bit for bit |
| forward | `col_output` (E4M3 column values) | repeats bit for bit |
| forward | `row_scales` (E8M0 row scales) | repeats bit for bit |
| forward | `col_scales` (E8M0 column scales) | repeats bit for bit |
| backward | `row_output` (E4M3 row values) | repeats bit for bit |
| backward | `col_output` (E4M3 column values) | repeats bit for bit |
| backward | `row_scales` (E8M0 row scales) | repeats bit for bit |
| backward | `col_scales` (E8M0 column scales) | repeats bit for bit |

The op is elementwise plus a maximum over each 32-element block.  That maximum is exact in
any order.  The kernels use no atomics.  Nothing is summed across threads, CTAs or calls.  No
entry takes a `deterministic` keyword.  No call alerts under
`torch.use_deterministic_algorithms(True)`.  A non-deterministic package of this op is not
planned, since determinism costs nothing here.

`tests/experimental/test_cake_gated_act_mxfp8_det_train.py` checks five repeated calls per route on
every shipped program.  It also checks byte-exact agreement with an independent plain-torch
reference, for every program, on five input sets:

- normal;
- scaled (power-of-two block factors from 2^-8 to 2^8);
- special (NaN, ±Inf, zeros, subnormals, overflow);
- boundary (the E4M3 maximum and rounding ties, the RCEIL scale bump, power-of-two scale
  boundaries);
- bits (every BF16 bit pattern as a gate).

## Limitations

- BF16 input only.
- E4M3 values with RCEIL E8M0 scales only: no E5M2, no FLOOR scales, no NVFP4.
- SwiGLU (`silu(gate) * up`) only.
- `M` and `K` must be multiples of 128.
- Contiguous, 32-byte aligned inputs only; strided inputs are rejected.
- `sm_100a`, `sm_103a` and `sm_107a` only.
