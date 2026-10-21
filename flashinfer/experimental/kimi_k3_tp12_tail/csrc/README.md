# Generated sources

The programs under `cake_kimi_k3_tp12_tail/` (one `*_kernel.cu` / `*_binding.cu`
pair per program plus the shared device and host headers) are written by the
Cake generated-program export (`tools/export-generated-programs run` with
`exports/kimi_k3_tp12_tail/export.py` in the Cake repository).  Each program is
one source compiled for SM100a and SM103a; the lowering differences between the
two are `__CUDA_ARCH__` guards inside the source.  The export populates
`../cake_jit.py` (`MODULES` and `KERNELS`) at the same time.  Nothing under this
directory is hand-written; do not edit the generated files.  Until the export
has run, no program is registered and the public entry points raise
`NotImplementedError` naming the missing architecture.
