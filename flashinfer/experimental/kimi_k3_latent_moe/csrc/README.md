# Generated sources

`cake_kimi_k3_latent_moe/*.cu` is written by the Cake generated-program export
(`tools/export-generated-programs run` with `exports/kimi_k3_latent_moe/export.py`
in the Cake repository), which populates `../cake_jit.py` (`MODULES`, `KERNELS`,
`SPECIALIZATIONS`, `KERNEL_SYMBOLS`) at the same time.  One `_kernel.cu` /
`_binding.cu` pair per program serves every key routed to it (the compile-line
definitions of `SPECIALIZATIONS` and `KERNEL_SYMBOLS` select a key's variant and
kernel symbol) and every listed architecture (SM100a and SM103a compile the same
text); the helper preambles every program shares live once in
`cake_kimi_k3_latent_moe_device_common.cuh` / `cake_kimi_k3_latent_moe_host_common.cuh`.
Nothing under this directory is hand-written except the `#if` arms of the files
marked "hand-edited here pending exporter support", which fold formerly separate
programs into one source; do not edit the generated files otherwise.
