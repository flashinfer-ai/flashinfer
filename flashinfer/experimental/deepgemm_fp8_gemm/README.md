# DeepGEMM FP8 1D1D GEMM (PTX source export)

Prepared FP8 E4M3 x E4M3 GEMM with 1D1D UE8M0 block scaling (per-token A
scales, per-row B scales at K128 granularity), exported as Cake-generated PTX
source plus a generated TVM-FFI host binding. Two routes exist per exported
architecture: `forward` writes BF16 output, `wgrad` accumulates into an FP32
output in place (`D = D + A @ B^T`). Every program is specialized to one
architecture and SM count; `catalog.json` (schema `fp8_gemm_1d1d.ptx.v2`)
carries one `arches[<arch>]` section per exported architecture (`sm_100a`
with 148 SMs, `sm_103a` with 152 SMs). The runtime selects the section from
the device's compute capability and refuses devices without an exact program.

The exported specialization is M4096 / N7168 / K4096. `a` `[4096, 4096]` and
`b` `[7168, 4096]` are `uint8` tensors holding E4M3 bytes; `sfa` `[8, 4096]`
and `sfb` `[8, 7168]` are `uint32` words that pack four adjacent K128 UE8M0
scale bytes, stored MN-major. `prepare_fp8_gemm_1d1d(...)` returns a plan;
`plan.run()` submits the launch on the current PyTorch stream without any
allocation, so it can be captured into a CUDA graph. Restore the FP32
initializer before each independent accumulated evaluation.

Building requires an installed CUDA toolkit (`ptxas`) and Apache TVM FFI with
`embed_cubin` support; no precompiled GPU binary ships with the sources.
`ptx_builder.build_from_ptx` freshly assembles the exported `.ptx` for the
catalogued architecture with the catalogued `ptxas` options, compiles the
generated binding, and records a build receipt in the caller's `cache_dir`.

Generated PTX and bindings live in `csrc/experimental/deepgemm_fp8_gemm/generated/<arch>/`;
the runtime, builder and catalog are in this directory.

```bash
pytest tests/experimental/test_fp8_ptx_source_export.py -q
```

The test runs on any catalogued architecture whose SM count matches an
exported route and skips elsewhere. Performance qualification measures the
complete prepared launch against the Cake production source route on the same
device; per-row exported execution must remain within 3% of its source route.
