# Generated sources

`cake_nvfp4_per_token/*.cu` is written by the Cake generated-program export
(`tools/export-generated-programs run` with `exports/nvfp4_per_token/export.py`
in the Cake repository), which populates `../cake_jit.py` (`MODULES` and
`KERNELS`) at the same time.  One `<program>_kernel.cu` / `<program>_binding.cu`
pair serves every registered architecture (SM100 / SM103; the lowering lines
that differ sit under `__CUDA_ARCH__` guards); the device helpers and the
launch helpers every program uses live once in
`cake_nvfp4_per_token_device_common.cuh` / `cake_nvfp4_per_token_host_common.cuh`.
The kernel sources are exporter output hand-folded here pending exporter
support: a program that differs from a sibling only by constant knobs (output
dtype, TMA cache-hint policies, raster group, quantizer CTA shape, folded
output scale or store policy) builds the sibling's `<program>_kernel.cu`,
selected by its own `-D` flags (`-DCAKE_PROGRAM_KERNEL` naming its kernel
symbol and the `-DCAKE_*` knob values, `compile_flags` in `../cake_jit.py`),
and the shared kernel sources carry `#ifndef` defaults for the program each
was exported for.
