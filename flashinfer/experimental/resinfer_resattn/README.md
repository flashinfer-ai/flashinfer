# ResInfer ResAttn (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
Source: **ResInfer**, public commit
[51b906836a69fa43915040314df57bc9f23030ad](https://github.com/ShawRong/resinfer/commit/51b906836a69fa43915040314df57bc9f23030ad).
This is a port of ResInfer's existing method and kernels, not an independent
implementation.

## Original code and integration changes

- `csrc/resattn/{host,kernel_paged_decode,plan,kv_layout,qmul4,output}.py`:
  grouped Q residue, pre-softmax QK folding, paged decode, QMUL4 V conversion,
  output and split/reduction paths.
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
from flashinfer.resinfer import paged_decode_residue
```

It uses `@flashinfer_experimental_api`; calling it is the explicit opt-in.
The public function validates arguments and defers to this backend.
Modules in this directory are internal implementation details. The public
signature remains experimental and may change without compatibility guarantees.

`flashinfer.resinfer.paged_decode_residue` accepts BF16 single-token queries,
prequantized NVFP4 K/V pages and E4M3 block scales. K scales are linear
`[pages, heads, 128, D/16]`; V scales are four-token interleaved
`[pages, heads, 32, D/16, 4]`, with indices
`[page, head, token//4, channel//16, token%4]`. It supports SM100,
page size 128, GQA4/6/8, head dimension 128/256, BF16 output, and split counts
1 or 2 (deterministic FP32-partial reducer for split2). The donor's packed
output and other reducer implementations remain internal and are not exposed
through this initial API. No serving-framework monkeypatches are included.

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
python examples/resinfer/resattn.py
```

The example constructs a synthetic prequantized KV cache with shuffled page
tables and nonuniform scales, prepares the V-scale layout, and calls the
public API with split1 and split2. A zero-query check compares the output
against uniform attention over decoded V. It requires no checkpoint or test
helpers. Both split modes passed this check on B200 with DSL 4.8.

## Validation and graduation

On 2026-10-21, **10 tests passed in 21.98 seconds** on one B200 / SM100,
with Torch 2.13.0+cu130, CUTLASS DSL 4.8.0, its CUDA 13.4 PTX compiler,
and TVM-FFI 0.1.11. Run:

```sh
pytest -q tests/experimental/test_resinfer_resattn.py
```

The MEXT suite also passed all 18 tests in 6.90 seconds under this same
DSL 4.8 environment, validating coexistence of the two integrated features
within those test scopes.

Coverage: GQA4/6/8 at D128 and GQA4 at D256, split1/split2, batch 2 with two
KV heads, lengths 601/519, shuffled page tables, non-unit cache reconstruction
scales, changed-Q graph replay, and a 127-token single-tile FP8-P reference.

The first run used a plain FP32-probability reference and failed one of 4096
GQA8 output elements at an absolute error of 0.00235049. The kernel converts
unnormalized probabilities to E4M3 for PV while retaining an unrounded
denominator. The revised protocol explicitly models that conversion for
the single-tile case. Multi-tile comparisons use the E4M3 rounding envelope
(normal relative error at most 1/16, subnormal absolute error at most 1/1024),
propagated through decoded absolute V, plus BF16 output rounding. This is
a conservative numerical bound, not bitwise equivalence to FP32 attention.
The kernel was not changed to obtain the second result.

The subsequent review replaced constant KV block scales with nonuniform
per-token/per-block scales. This exposed the V-layout mismatch described
above: the original interface produced 73 violations of the existing error
bound in the first GQA4 case. Correctly interleaving V scales removed those
violations without widening tolerances or changing kernel arithmetic.
The reference also models QMUL4's E4M3 V conversion before the global scale.

The expanded suite has 15 tests, including lengths 1/127 with split1/split2,
rejection of linear V scales, and a host-only regression for the retained
packed-output gmem workspace validation. That last test stubs the decode
launch; it does not validate packed-output numerics. The original four-tensor
workspace raised `ValueError` because three partials and the completion
counter were zipped against three shapes; they are now checked separately.
All 15 passed on the same SM100/DSL 4.8 runtime during review.

These tests do not establish model E2E quality or speed. Other shapes, packed
output, other reducers and concurrent streams remain unvalidated. Graduation
requires matched performance records, API review, and completion of the
tracking issue.
