# Generated sources

`cake_kimi_k3_latent_moe/*.cu` is written by the Cake generated-program export
(`tools/export-generated-programs run` with `exports/kimi_k3_latent_moe/export.py`
in the Cake repository), which populates `../cake_jit.py` (`MODULES`, `KERNELS`,
`SPECIALIZATIONS`, `KERNEL_SYMBOLS`) at the same time.  One `_kernel.cu` per
program serves every key routed to it (the compile-line definitions of
`SPECIALIZATIONS` and `KERNEL_SYMBOLS` select a key's variant and kernel symbol)
and every listed architecture (SM100a and SM103a compile the same text); one
`_binding.cu` per launcher shape: where a shape has several programs the
launcher is shared and the `CAKE_PROGRAM_KERNEL`, `CAKE_PROGRAM_SHIM`,
`CAKE_BINDING_SMEM`, `CAKE_BINDING_BOX_Z` and `CAKE_BINDING_N_PAD` definitions on
a record's compile line select the program (the shared tail GEMM launcher takes
its kernel symbol from `KERNEL_SYMBOLS` and its dynamic shared memory from the
`NUM_STAGES` knob); the helper preambles every program shares live once in
`cake_kimi_k3_latent_moe_device_common.cuh` / `cake_kimi_k3_latent_moe_host_common.cuh`.
Nothing under this directory is hand-written except the `#if` arms and the
compile-definition guards of the files marked "hand-edited here pending exporter
support", which fold formerly separate programs into one source; the other
generated files are reproduced by the exporter.
