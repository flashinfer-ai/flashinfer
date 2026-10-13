# Generated sources

`cake_kimi_k3_vision_tower/*.cu` and the two shared headers
(`cake_kimi_k3_vision_tower_device_common.cuh`,
`cake_kimi_k3_vision_tower_host_common.cuh`) are written by the Cake
generated-program export, which populates `../cake_jit.py` (`MODULES`,
`ARG_PLANS` and `KERNELS`) at the same time.  One `*_kernel.cu` /
`*_binding.cu` pair is one program; `MODULES[name]["arches"]` lists the
architectures it is built for, and the loader compiles it with the exact
`sm100a` / `sm103a` flag set of the device it runs on.  Nothing under this
directory is hand-written; do not edit the generated files (`.clang-format`
disables formatting here so the sources stay byte-faithful to the export).
