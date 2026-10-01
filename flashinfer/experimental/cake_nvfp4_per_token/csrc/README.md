# Generated sources (placeholder)

`cake_nvfp4_per_token/<arch>/*.cu` is written by the Cake generated-program
export (`tools/export-generated-programs run` with
`exports/nvfp4_per_token/export.py` in the Cake repository).  It populates
`../cake_jit.py` (`MODULES` and `KERNELS`) at the same time.  Nothing under
this directory is hand-written; do not edit the generated files.  Until the
export has run, no program is registered and `backend="cake"` raises
`NotImplementedError` naming the missing architecture.
