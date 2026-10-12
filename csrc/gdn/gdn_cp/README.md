# Cake GDN CP-prefill kernels

Generated device sources (`*_kernel.cu`) and their thin TVM-FFI launchers
(`*_binding.cu`) for the context-parallel GDN prefill backend on SM100 and
SM103.  One source per physical program compiles for both architectures; the
registry in `flashinfer/jit/cake_gdn_cp_backend.py` maps each logical kernel
to its program and the host in
`flashinfer/gdn_kernels/blackwell/cake_gdn_cp_backend.py` owns planning,
workspaces and dispatch behind the public `flashinfer.gdn_prefill`
`chunk_gated_delta_rule` API.

These files are produced by the Cake exporter; do not edit them by hand.
