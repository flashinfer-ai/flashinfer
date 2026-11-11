# DeepGEMM FP8 1D1D GEMM (generated CUDA source)

Prepared FP8 E4M3 x E4M3 GEMM with 1D1D UE8M0 block scaling (per-token A
scales, per-row B scales at K128 granularity), delivered as one generated CUDA
kernel source plus a thin TVM-FFI host binding per route. Two routes share the
DeepGEMM normal schedule (BM128 / BN224 / BK128, six TMA stages, `cta_group::2`,
persistent grouped raster): `forward` writes BF16 output, `wgrad` accumulates
into an FP32 output in place (`D = D + A @ B^T`). Problem sizes, tile counts
and the persistent CTA count are launch scalars, so one compiled program per
route serves every `M` and `N` and every `K` that is a multiple of 128. The
programs are compiled for the exact architecture of the device (`sm_100a`,
`sm_103a`). Portions of the generated device code derive from DeepGEMM (MIT
licensed); `DEEPGEMM_NOTICE.txt` in this directory carries the notice.

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

## Package layout

The package is self-contained. `cake_jit.py` is the registry of the generated
programs: one `MODULES` record per route carrying the host plan (the
architectures it is delivered for, its launch record, the argument plan of its
`run` entry, the grid rule the runtime evaluates, the workspace rule and the
kernel-parameter to logical-argument map), `KERNELS` mapping each route to its
record, and the loader glue (`library_name`, `launch_args`, `load`,
`run_module`). What to compile is recorded once in `build.json`: the generated
CUDA sources under `csrc/`, their digests and nvcc flags, one entry per
(program, architecture). `cake_build.py` loads a prebuilt library from `aot/`
when a builder compiled one and otherwise JIT-builds the manifest entry through
FlashInfer from the same sources and flags. `runtime.py` holds
`launch_geometry` (the grid rule), `pack_ue8m0_words` and
`prepare_fp8_gemm_1d1d`. `scripts/check_cake_build_manifests.py` checks the
manifest against the registry and the sources without importing the package.

```bash
pytest tests/experimental/test_fp8_gemm_1d1d_generated.py -q
```

The test runs on any registered architecture and skips its GPU cases
elsewhere; the host-plan and manifest cases need no GPU.
