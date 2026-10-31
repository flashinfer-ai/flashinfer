# ResInfer KEXT (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
The public entry point is `flashinfer.resinfer.nvfp4_quantize_kext`, decorated
with `@flashinfer_experimental_api`. Calling it explicitly opts in. Backend
imports are deferred; no AOT registration or default GEMM routing is changed.

## Source and attribution

This is a port of **ResInfer's production K-extension quantization**, not an
independent invention of its selected-channel residue method. Public donor:
[ResInfer 51b906836a69fa43915040314df57bc9f23030ad](https://github.com/ShawRong/resinfer/commit/51b906836a69fa43915040314df57bc9f23030ad).

| Donor source | Fork adaptation |
| --- | --- |
| `csrc/nvfp4/cuda/nvfp4_residue_pack8.cu` | `include/flashinfer/experimental/resinfer/kext_pack8.cuh` |
| `csrc/nvfp4/cuda/nvfp4_residue_pack16.cu` | `include/flashinfer/experimental/resinfer/kext_pack16.cuh` |
| `csrc/common/fp4/nvfp4_utils.cuh` | Shared `quant_utils.cuh` and KEXT-specific `kext_utils.cuh` |
| `csrc/nvfp4/cuda/nvfp4_quant_entry.cu` | SM100 shape selection in `quantization.py` |
| `csrc/nvfp4/quant.py` | Output allocation and mask/ratio contract |
| `quant/modelopt_integration/residue_linear.py` | Weight extension order in the example |
| `quant/vllm_integration/residue_linear_op.py` | Composition with ordinary FlashInfer FP4 GEMM |

The original NVIDIA and Rong Shuo notices and Apache-2.0 identifiers are
retained. CUDA arithmetic and the pack8/pack16 selection policy come from
ResInfer. Adaptation adds TVM-FFI/JIT bindings, an explicit API, device-aware
validation, zeroed scale padding, and tests. The donor's process-global
compute-capability cache and environment override are replaced by checking
the actual input device and an explicit `elts_per_thread` option.
Missing mask bits return zero instead of indexing before a local residue array;
valid masks follow the donor arithmetic unchanged. Malformed masks still
violate the numerical API contract.

KEXT reuses MEXT's licensed quantization helpers. It does not call Fold
or ResAttn.

## Numerical and layout contract

- SM100; contiguous, 32-byte-aligned FP16/BF16 activation `[M, K]`.
- Positive M and K; this first integration requires K divisible by 256.
  This keeps pack8 warp shuffles complete. Pack16's half-warp tail at K
  divisible by 256 but not 512 is retained.
- `residue_per_8` is 1, 2 or 4: select 12.5%, 25% or 50% of channels.
  CUDA uint8 `channel_masks[K/8]` must have exactly that many bits set in
  every byte. Bit zero selects the first channel in its group.
- One positive finite FP32 global quantization multiplier, on the input
  device, applies to base and residue. Finite inputs, valid masks and scale
  values are caller preconditions; there is no host synchronization to
  inspect their contents.
- Each output row contains the base followed by quantized reconstruction
  errors for selected channels, in ascending channel order. Packed E2M1
  bytes have shape `[M, (K + K/8*residue_per_8)/2]`.
- Scales are flat E4M3 bytes with the usual 128-row, 4-scale swizzle and
  deterministic zero padding.
- Before weight quantization, append the same selected weight columns in the
  same order. Feed the resulting extended activation/weight to upstream
  `mm_fp4`. Its backend alignment requirements still apply: for example,
  K=256 with one selected channel per group gives extended K=288, requiring
  additional padding for a GEMM that requires K divisible by 64.
- `elts_per_thread=0` selects ResInfer's production SM100 policy; 8 or 16
  forces the respective producer for diagnosis. No performance retuning
  has been done in this port.
- Swap is disabled in the donor's current production policy and is not
  exposed here. Ratio 1 uses the separate MEXT API; ratio 0 can use upstream
  `nvfp4_quantize`.

## Example and validation

Run `python examples/resinfer/kext.py` on SM100. It prepares extended weights,
quantizes activations with KEXT, and calls upstream `mm_fp4(backend="cute-dsl")`.
The example uses K=512, which satisfies the GEMM alignment for all three ratios.
Warm up before graph capture to compile the producer and GEMM.

Run `pytest tests/experimental/test_resinfer_kext.py`. The suite checks random
per-group masks, selected-channel reconstruction, pack8/pack16/default
agreement, zero scale padding, changed-input/mask/scale CUDA Graph replay,
upstream GEMM against decoded FP32 matmul, and invalid tensor metadata.

### Verified on 2026-10-21

On one B200 / SM100, the main suite completed with **41 passed in 13.61 s**.
A subsequent focused run of the added missing-mask-bit guard tests completed
with **2 passed in 1.26 s** (41 deselected). The runnable example also completed
successfully with BF16 output `[32, 128]` using upstream CuTe-DSL GEMM.
Runtime: Torch 2.13.0+cu130, CUTLASS DSL 4.8.0, TVM-FFI 0.1.11.

The numerical matrix covers FP16/BF16, all three residue counts, all three pack
settings, and paired shapes `(M, K)` of `(1, 256)`, `(7, 768)`, `(129, 1536)`,
`(1024, 512)`, `(4, 8192)`. Graph tests change activations, masks and global
scale. GEMM tests cover all three ratios at M=7, N=128, K=512, against decoded
FP32 matrix multiplication.

The first JIT attempt failed because the adapter defined/undefined a launch
macro inside another macro's argument. Moving the definition outside the
dtype dispatch fixed compilation; no kernel arithmetic or test tolerances
changed. Compiler/DSL deprecation warnings remain.

After source review widened the shared MEXT scale-offset arithmetic, all
43 KEXT tests passed again in a combined 88-test MEXT/ResAttn/KEXT run on
the same SM100/DSL 4.8 runtime (32.64 seconds). KEXT kernel arithmetic and
dispatch policy were unchanged.

This is bounded kernel correctness evidence. Multi-device execution, complete
quantizer-plus-GEMM graph capture, other architectures, model E2E, and speedup
claims remain unvalidated. The missing-mask-bit tests exercise one defensive
guard; they do not make malformed masks supported.

Performance measurements must separate quantizer-only, GEMM-only, and
activation-quantization-plus-GEMM timing, excluding offline weight preparation.
These are kernel pipelines, not model E2E or model quality evidence.

Graduation still requires matched performance evidence, broader supported
shapes/devices, API review, and completion of the tracking issue.
