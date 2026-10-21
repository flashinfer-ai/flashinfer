# Generated sources

`cake_kimi_k3_vision_tower/*_kernel.cu` (one per program) and the two shared
headers (`cake_kimi_k3_vision_tower_device_common.cuh`,
`cake_kimi_k3_vision_tower_host_common.cuh`) are written by the Cake
generated-program export, which populates `../cake_jit.py` (`MODULES`,
`ARG_PLANS` and `KERNELS`) at the same time.  One `*_kernel.cu` plus the
shared launcher `cake_kimi_k3_vision_tower_launcher.cu` is one program:
`MODULES[name]["arches"]` lists the architectures it is built for, the loader
compiles it with the exact `sm100a` / `sm103a` flag set of the device it runs
on, and the compile definitions of `cake_jit.launch_definitions(name)` select
the program's kernel symbol, host-shim namespace, argument plan and launch
values in the launcher.  The launcher and its TMA-descriptor encoder
`cake_kimi_k3_vision_tower_tma_encode.cuh` are the export's per-program host
bindings, hand-folded here pending exporter support, as are the five attention
kernels built for both architectures: the SM100 and the SM103 export of each
form are one source whose differing regions are selected by
`#if __CUDA_ARCH__ == 1000` (SM100 text first).  Nothing else under this
directory is hand-written (`.clang-format` disables formatting here so the
sources stay byte-faithful to the export).
