# Generated sources

`cake_kimi_k3_vision_tower/*.cu` and the two shared headers
(`cake_kimi_k3_vision_tower_device_common.cuh`,
`cake_kimi_k3_vision_tower_host_common.cuh`) are written by the Cake
generated-program export, which populates `../cake_jit.py` (`MODULES`,
`ARG_PLANS` and `KERNELS`) at the same time.  One `*_kernel.cu` /
`*_binding.cu` pair is one program; `MODULES[name]["arches"]` lists the
architectures it is built for, and the loader compiles it with the exact
`sm100a` / `sm103a` flag set of the device it runs on.  The five attention
kernels built for both architectures are Cake exporter output, hand-folded
here pending exporter support: the SM100 and the SM103 export of each form are
one source whose differing regions are selected by `#if __CUDA_ARCH__ == 1000`
(SM100 text first).  Nothing else under this directory is hand-written
(`.clang-format` disables formatting here so the sources stay byte-faithful to
the export).
