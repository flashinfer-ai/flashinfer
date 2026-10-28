# Generated sources

`cake_nvfp4_per_token/*.cu` is written by the Cake generated-program export
(`tools/export-generated-programs run` with `exports/nvfp4_per_token/export.py`
in the Cake repository), which populates `../cake_jit.py` (`MODULES` and
`KERNELS`) at the same time.  The directory is flat: the kernel and binding
translation units are named by the 20-hex digest of the program they were
exported for, and one `<program>_kernel.cu` / `<program>_binding.cu` pair
serves every registered architecture (SM100 / SM103; the lowering lines that
differ sit under `__CUDA_ARCH__` guards).  A program whose exported text
differs from a sibling's only by a knob value (kernels) or by the kernel
symbol and host-shim namespace (bindings; the bf16 quantizer bindings also by
the launch block width, `CAKE_Q_BLOCK`) builds the sibling's translation unit
with its own `-D` flags (`cake_jit.py`: the `kernel` / `binding` row fields and
`compile_flags`).

- `cake_nvfp4_per_token_<hash>_kernel.cu` defines the device entry
  `kernel_cake_nvfp4_per_token_<hash>` (one `extern "C" __global__` per file;
  a shared source names it `CAKE_PROGRAM_KERNEL` with that default and selects
  the knob values through `#ifndef` defaults and `#if` arms).
- `cake_nvfp4_per_token_<hash>_binding.cu` declares the kernel
  `CAKE_PROGRAM_KERNEL` and exports its tvm-ffi entry point `run` from the
  host-shim namespace `CAKE_PROGRAM_SHIM`; the file's defaults are the symbol
  and the namespace `cake_host_shim_<id>` (a 16-hex id distinct from the
  program digest) of the program it was exported for.  A GEMM binding includes
  the host header a second time from inside its namespace with `CAKE_TMA_A` /
  `CAKE_TMA_B` / `CAKE_TMA_SFA` / `CAKE_TMA_SFB` / `CAKE_TMA_OUT` selecting its
  five TMA descriptor encoders.
- `cake_nvfp4_per_token_device_common.cuh` holds the device helpers every
  kernel includes; `cake_nvfp4_per_token_host_common.cuh` holds the launch
  helpers every binding includes (`cake_host_shim_common`, `CAKE_STRINGIZE`)
  and, selected by the `CAKE_TMA_*` macros, the exported TMA descriptor
  encoder texts.

The kernel and binding sources under this directory, the two shared headers
and the registry rows in `../cake_jit.py` are exporter output
hand-folded here pending exporter support: the shared translation units
with their `#ifndef` defaults and `#if` arms, the per-record `-D`
selection, the shared-memory address helper (`shared_base_addr`) of the
device header, and the include guard, `CAKE_STRINGIZE` and the encoder
section of the host header.
