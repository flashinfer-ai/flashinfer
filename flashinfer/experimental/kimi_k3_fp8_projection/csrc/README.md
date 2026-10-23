# Generated sources

`cake_kimi_k3_fp8_projection/` is flat: the kernel and launcher translation
units are named by the 20-hex digest of the program they were exported for.
A program whose exported text differs from a sibling's only by a lever value
(kernels) or by the kernel symbol and launcher namespace (launchers) builds
the sibling's translation unit with its own `-D` flags (`cake_jit.py`: the
`kernel` / `binding` row fields and `compile_flags`).

- `cake_kimi_k3_fp8_projection_<hash>_kernel.cu` defines the device entry
  `kernel_cake_kimi_k3_fp8_projection_<hash>` (one `extern "C" __global__`
  per file; a shared source names it `CAKE_PROGRAM_KERNEL` with that
  default).  One source serves sm_100a and sm_103a; `__CUDA_ARCH__` guards
  select the per-architecture lowering.
- `cake_kimi_k3_fp8_projection_<hash>_binding.cu` declares the kernel
  `CAKE_PROGRAM_KERNEL` and exports its tvm-ffi entry points `run`,
  `run_plan`, `run_launch` and `__tvm_ffi_run_chain` from the launcher
  namespace `CAKE_PROGRAM_SHIM`; the file's defaults are the symbol and the
  namespace `cake_host_shim_<id>` (a 16-hex id distinct from the program
  digest) of the program it was exported for.
- `cake_kimi_k3_fp8_projection_device_common.cuh` is the device preamble every
  kernel includes (typedefs, tensor-map ABI, the device-helper includes);
  `cake_kimi_k3_fp8_projection_host_common.cuh` holds the launch helpers every
  binding includes (`cake_host_shim_common`, the `CAKE_STRINGIZE` and
  `CAKE_CHECK_EXPANDED` macros) and, selected by `CAKE_HOST_COMMON_PART`, the
  receipt / chain text every launcher includes verbatim inside its namespace.

The device preamble's compiler helper headers (`cake_<helper>_<16-hex>.cuh`)
come from the repository's top-level `csrc/cake_device_helpers/` directory,
which the loader puts on the include path after the package's own `csrc/`.

The shared kernel and launcher sources, their `#ifndef` defaults and `#if`
arms and the hoisted parts of the host header are a hand fold of the exported
files, pending regeneration by the export.
