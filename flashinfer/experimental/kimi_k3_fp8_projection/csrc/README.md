# Generated sources

`cake_kimi_k3_fp8_projection/` is flat: every program has one kernel and one
launcher translation unit, both named by the program's 20-hex digest.

- `cake_kimi_k3_fp8_projection_<hash>_kernel.cu` defines the device entry
  `kernel_cake_kimi_k3_fp8_projection_<hash>` (one `extern "C" __global__`
  per file).  One source serves sm_100a and sm_103a; `__CUDA_ARCH__` guards
  select the per-architecture lowering.
- `cake_kimi_k3_fp8_projection_<hash>_binding.cu` declares that kernel and
  exports its tvm-ffi entry points `run`, `run_plan`, `run_launch` and
  `__tvm_ffi_run_chain` from the launcher namespace `cake_host_shim_<id>`
  (a 16-hex id distinct from the program digest).
- `cake_kimi_k3_fp8_projection_device_common.cuh` is the device preamble every
  kernel includes (typedefs, tensor-map ABI, the device-helper includes);
  `cake_kimi_k3_fp8_projection_host_common.cuh` holds the launch helpers every
  binding includes (`cake_host_shim_common`).

`cake_device_helpers/` holds the compiler helper headers
(`cake_<helper>_<16-hex>.cuh`) that the device preamble includes.
