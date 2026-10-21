# Generated sources

The programs under `cake_kimi_k3_tp12_tail/` (one `*_binding.cu` per program, one
`*_kernel.cu` per program except the two `k23` programs, which share
`cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0_kernel.cu` selected by the
`K23_CAPACITY` compile definition, plus the shared device and host headers) are
written by the
Cake generated-program export (`tools/export-generated-programs run` with
`exports/kimi_k3_tp12_tail/export.py` in the Cake repository).  Each program is
one source compiled for SM100a and SM103a; the lowering differences between the
two are `__CUDA_ARCH__` guards inside the source.  The export populates
`../cake_jit.py` (`MODULES` and `KERNELS`) at the same time.  Nothing under this
directory is hand-written except the `#if` arms and guarded defaults of
`cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0_kernel.cu` (marked
"hand-edited here pending exporter support"), which fold the two `k23` programs
into one source; do not hand-edit the generated files otherwise.  Until the export
has run, no program is registered and the public entry points raise
`NotImplementedError` naming the missing architecture.
