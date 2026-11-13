# ResServe KEXT (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
These SM100 APIs are experimental. Calling them explicitly opts in; they do not
participate in automatic backend selection and provide no compatibility guarantee.

## API and constraints

Import `nvfp4_quantize_kext` from `flashinfer.resserve`. It appends quantized
reconstruction errors for selected channels after each base activation row.

- Contiguous, 32-byte-aligned FP16/BF16 input `[M, K]`; positive dimensions and
  K divisible by 128. Automatic mode uses pack16 for widths not divisible by
  256; explicit pack8 requires K divisible by 256.
- `residue_per_8` is 1, 2 or 4. CUDA uint8 `channel_masks[K/8]` must contain
  exactly that many set bits per byte; bit zero selects the first channel.
- Inputs must be finite; the FP32 global multiplier must be positive and finite.
  Mask values and multiplier contents are caller preconditions.
- Packed outputs are uint8 E2M1. E4M3 scales use the 128-row, 4-scale swizzle
  with zero padding, returned as a `(padded_rows, K_tiles * 4)` float8 view.
  Pass that view to `mm_fp4`. See the API docstring for output sizes.
- Before quantizing weights, append the same selected columns in ascending
  order. Use the extended tensors with upstream `mm_fp4`; downstream GEMM
  alignment still applies and may require extra padding.
- `elts_per_thread=0` selects the producer automatically; 8 or 16 forces it.
  Warm up the quantizer and GEMM before CUDA Graph capture.

## Example and tests

```bash
python examples/resserve/kext.py
pytest tests/experimental/test_resserve_kext.py
```

## Source

Ported from [ResServe commit 51b9068](https://github.com/ShawRong/resserve/commit/51b906836a69fa43915040314df57bc9f23030ad).
Original copyright and license notices are retained in the source files.

- `csrc/nvfp4/quant.py` and `csrc/nvfp4/cuda/nvfp4_quant_entry.cu`: allocation
  and pack dispatch.
- `csrc/nvfp4/cuda/nvfp4_residue_pack{8,16}.cu`: selected-channel residue packing.
- `csrc/common/fp4/nvfp4_utils.cuh`: shared FP4 quantization helpers.
