# ResServe ResAttn (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
Source: **ResServe**, public commit
[51b906836a69fa43915040314df57bc9f23030ad](https://github.com/ShawRong/resserve/commit/51b906836a69fa43915040314df57bc9f23030ad).
This is a port of ResServe's existing method and kernels, not an independent
implementation.

## Original code and integration changes

- `csrc/resattn/{host,kernel_paged_decode,qmul4}.py`: grouped Q residue,
  pre-softmax QK folding, paged decode, QMUL4 V conversion, and the
  split/reduction path. The donor's `plan`, `kv_layout` and `output` modules
  are not carried over: this port fixes the configuration to what the public
  API exposes.
- `csrc/nvfp4/cuda/nvfp4_quant_kernels.cu`: grouped Q producer.
- `csrc/common/fp4/nvfp4_utils.cuh`: production pack16 residue arithmetic,
  shared with this fork's MEXT branch.

All original copyright and license notices are retained. ResAttn reuses
MEXT's quantization helpers. Integration changes
replace the external Torch extension with FlashInfer JIT/TVM-FFI, introduce
an explicit experimental API, add tensor validation, and allocate grouped Q
scratch per invocation to avoid shared mutable buffers across streams/graphs.
The per-call allocation cost is part of the integrated pipeline.

## Scope

The public entry point is imported as:

```python
from flashinfer.resserve import paged_decode_residue
```

It uses `@flashinfer_experimental_api`; calling it is the explicit opt-in.
The public function validates arguments and defers to this backend.
Modules in this directory are internal implementation details. The public
signature remains experimental and may change without compatibility guarantees.

`flashinfer.resserve.paged_decode_residue` accepts BF16 single-token queries,
prequantized NVFP4 K/V pages and E4M3 block scales. K scales are linear
`[pages, heads, 128, D/16]`; V scales are four-token interleaved
`[pages, heads, 32, D/16, 4]`, with indices
`[page, head, token//4, channel//16, token%4]`. It supports SM100,
page size 128, GQA4/6/8, head dimension 128/256, BF16 output, and split counts
1 or 2 (deterministic FP32-partial reducer for split2). The donor's packed
NVFP4 output path, its fused last-CTA and cluster reducers, the automatic
split selection and the caller-owned reduction workspace are not ported in
this PR; the kernel emits BF16 only. No serving-framework monkeypatches are
included.

Requires `nvidia-cutlass-dsl[cu13]==4.8.0` and its CUDA 13.4 PTX compiler;
install the compiler dependencies in your development environment.
The API is opt-in and does not participate in automatic backend selection.
See its docstring for layout, reconstruction scale and metadata preconditions.

Prepare V scales in this layout when writing the cache. For a pre-existing
linear scale tensor, the one-time conversion is:

```python
pages, heads, _, blocks = v_scale_linear.shape
v_scale = (
    v_scale_linear.reshape(pages, heads, 32, 4, blocks)
    .transpose(-1, -2)
    .contiguous()
)
```

Decode only creates a view of those bytes; it does not copy or rearrange the
cache. The initial fork API incorrectly described V scales as linear.
Nonuniform-scale regression testing exposed that mismatch; the explicit 5D
shape now rejects the old ambiguous input. The donor kernel is unchanged.

## Runnable example

```sh
python examples/resserve/resattn.py
```

The example constructs a synthetic prequantized KV cache with shuffled page
tables and nonuniform scales, prepares the V-scale layout, and calls the
public API with split1 and split2. A zero-query check compares the output
against uniform attention over decoded V. It requires no checkpoint or test
helpers. Both split modes are exercised by the example.

## Tests and numerical behavior

```sh
pytest -q tests/experimental/test_resserve_resattn.py
```

The kernel converts unnormalized probabilities to E4M3 for PV while retaining
an unrounded denominator; QMUL4 also converts V to E4M3 before the global scale.
Tests model these conversions and check nonuniform V-scale layouts, split
reduction and changed-query CUDA Graph replay. This is not bitwise equivalence
to FP32 attention or evidence of full-model quality or performance.
