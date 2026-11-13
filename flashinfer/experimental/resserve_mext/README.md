# ResServe MEXT (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
These SM100 APIs are experimental. Calling them explicitly opts in; they do not
participate in automatic backend selection and provide no compatibility guarantee.

## API and constraints

Import `nvfp4_quantize_residue` and `mm_nvfp4_residue` from `flashinfer.resserve`.
The quantizer produces paired base/residue activation rows. Fold GEMM uses shared
weights and adds the paired accumulators in FP32 before FP16/BF16 conversion.

- FP16/BF16 input; finite inputs and positive finite global multipliers.
- GEMM requires K divisible by 64 and N divisible by 8.
- `mm_nvfp4_residue` selects a row-pair tile by M: `(128, 24)` through M 8,
  `(128, 32)` through M 16, `(128, 64)` below M 256, and `(256, 128)` on a
  2-SM cluster from M 256. Narrower tma tiles are available to direct host
  calls only when `2*M` fits in one tile.
- Packed data is uint8. Scales are the 128-row, 4-scale E4M3 swizzle, returned as a `(padded_rows, K_tiles * 4)` float8 view. `mm_nvfp4_residue` accepts that view or the same bytes flattened.
- Packed weights may be base-K views with nonoverlapping K-contiguous rows and
  a 32-byte-aligned row stride. Base-K scales must be prepared separately;
  slicing extended swizzled scales is not equivalent.
- Checkpoint parsing, weight/scale preparation and projection fusion belong to
  the caller. See API docstrings for tensor shapes and alignment requirements.
- Warm up before CUDA Graph capture. The producer uses ordinary stream ordering,
  without PDL.

## Example and tests

```bash
python examples/resserve/bench_mext.py --m 32 --n 128 --k 256
pytest tests/experimental/test_resserve_mext.py
```

## Source

Ported from [ResServe commit 51b9068](https://github.com/ShawRong/resserve/commit/51b906836a69fa43915040314df57bc9f23030ad).
Original copyright and license notices are retained in the source files.

- `csrc/common/fold/sm10x/{host,kernel_sm100_fold}.py`: Fold GEMM host and kernel.
- `csrc/nvfp4/cuda/nvfp4_quant_kernels.cu`: row-pair activation quantizer.
- `csrc/common/fp4/nvfp4_utils.cuh`: shared FP4 quantization helpers.
