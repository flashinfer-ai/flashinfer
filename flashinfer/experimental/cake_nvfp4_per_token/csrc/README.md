# Generated sources

`cake_nvfp4_per_token/*.cu` is written by the Cake generated-program export
(`tools/export-generated-programs run` with `exports/nvfp4_per_token/export.py`
in the Cake repository), which populates `../cake_jit.py` (`MODULES` and
`KERNELS`) at the same time.  One `<program>_kernel.cu` / `<program>_binding.cu`
pair serves every registered architecture (SM100 / SM103; the lowering lines
that differ sit under `__CUDA_ARCH__` guards); the device helpers and the
launch helpers every program uses live once in
`cake_nvfp4_per_token_device_common.cuh` / `cake_nvfp4_per_token_host_common.cuh`.
Nothing under this directory is hand-written; do not edit the generated files.
