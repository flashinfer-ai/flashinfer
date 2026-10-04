# DeepGEMM FP8 1D1D GEMM (generated CUDA source)

Prepared FP8 E4M3 x E4M3 GEMM with 1D1D UE8M0 block scaling (per-token A
scales, per-row B scales at K128 granularity), delivered as one generated CUDA
kernel source plus a thin TVM-FFI host binding per route. Two routes share the
DeepGEMM normal schedule (BM128 / BN224 / BK128, six TMA stages, `cta_group::2`,
persistent grouped raster): `forward` writes BF16 output, `wgrad` accumulates
into an FP32 output in place (`D = D + A @ B^T`). Problem sizes, tile counts
and the persistent CTA count are launch scalars, so one compiled program per
route serves every `M` and `N` and every `K` that is a multiple of 128.
`cake_jit.py` registers the programs and compiles each source for the exact
architecture of the device (`sm_100a`, `sm_103a`) through FlashInfer's JIT.

`a` `[M, K]` and `b` `[N, K]` are `uint8` tensors holding E4M3 bytes; `sfa`
`[ceil(K/512), M]` and `sfb` `[ceil(K/512), N]` are `uint32` words that pack
four adjacent K128 UE8M0 scale bytes, stored MN-major (`pack_ue8m0_words`
builds them from natural `[rows, K/128]` exponent bytes, broadcasting
per-128-row block scales, and padding the row pitch of the packed words to a
16-byte multiple so any `M` and `N` encode). Every operand moves through TMA,
so it needs a 16-byte aligned base and a row pitch that is a multiple of 16
bytes: a contiguous BF16 `out` needs `N % 8 == 0` and a contiguous FP32 `out`
needs `N % 4 == 0`; for any other `N` pass a column slice of a wider buffer
(`torch.empty(M, pitch, ...)[:, :N]`), whose pad columns are never written.
`prepare_fp8_gemm_1d1d(...)` returns a plan; `plan.run()` submits the launch on
the current PyTorch stream without any allocation, so it can be captured into
a CUDA graph. Restore the FP32 initializer before each independent accumulated
evaluation.

Generated sources live in `csrc/experimental/deepgemm_fp8_gemm/`; the runtime
and loader are in this directory.

```bash
pytest tests/experimental/test_fp8_gemm_1d1d_generated.py -q
```

The test runs on any registered architecture and skips elsewhere.
