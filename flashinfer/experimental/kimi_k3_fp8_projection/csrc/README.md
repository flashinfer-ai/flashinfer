# Generated sources

`cake_kimi_k3_fp8_projection/` is flat: the kernel and launcher translation
units are named by the 20-hex digest of the program they were exported for.
A program whose exported text differs from a sibling's only by a lever value
(kernels) or by the kernel symbol, the launcher namespace and the launch
constants (launchers: TMA box dimensions, dynamic shared memory, block and
cluster dimensions) builds the sibling's translation unit with its own `-D`
flags (`cake_jit.py`: the `kernel` / `binding` row fields and
`compile_flags`).

- `cake_kimi_k3_fp8_projection_<hash>_kernel.cu` defines the device entry
  `kernel_cake_kimi_k3_fp8_projection_<hash>` (one `extern "C" __global__`
  per file; a shared source names it `CAKE_PROGRAM_KERNEL` with that
  default).  One source serves sm_100a and sm_103a; `__CUDA_ARCH__` guards
  select the per-architecture lowering.
- `cake_kimi_k3_fp8_projection_<hash>_binding.cu` is the launcher of one
  argument plan (`decode`, `decode_tso`, `gemm`, `gemm_skp`, `quant`), named
  by the digest of the program it was exported for; it declares the kernel
  `CAKE_PROGRAM_KERNEL`, exports its tvm-ffi entry points `run`, `run_plan`,
  `run_launch` and `__tvm_ffi_run_chain` from the launcher namespace
  `CAKE_PROGRAM_SHIM`, and takes the program's launch constants from `-D`
  macros (`CAKE_BOX_<T>_<i>`, `CAKE_DYN_SMEM`, `CAKE_BLOCK_X`,
  `CAKE_CLUSTER_X/Y/Z`, `CAKE_NONPORTABLE_CLUSTER`); the file's defaults are
  the symbol, the namespace `cake_host_shim_<id>` (a 16-hex id distinct from
  the program digest) and the launch constants of the program it was
  exported for.
- `cake_kimi_k3_fp8_projection_device_common.cuh` is the device preamble every
  kernel includes (typedefs, tensor-map ABI, the device-helper includes);
  `cake_kimi_k3_fp8_projection_host_common.cuh` holds the launch helpers every
  binding includes (`cake_host_shim_common`, the `CAKE_STRINGIZE`,
  `CAKE_CHECK_EXPANDED` and `CAKE_U` macros) and, selected by
  `CAKE_HOST_COMMON_PART`, the receipt / chain text every launcher includes
  verbatim inside its namespace.

The device preamble's compiler helper headers (`cake_<helper>_<16-hex>.cuh`)
come from the repository's top-level `csrc/cake_device_helpers/` directory,
which the loader puts on the include path after the package's own `csrc/`.

The shared kernel and launcher sources, their `#ifndef` defaults (lever values,
launch constants) and `#if` arms and the hoisted parts of the host header are
a hand fold of the exported files, pending regeneration by the export.
