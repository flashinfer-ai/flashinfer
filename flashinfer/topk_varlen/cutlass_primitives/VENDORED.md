# Vendored from cutlass-primitives

Do not edit these files in FlashInfer; change the library and re-vendor.

- upstream: cutlass-primitives, an NVIDIA-internal CuTe-DSL primitives library (ask the
  FlashInfer top-k maintainers for access)
- license: Apache-2.0 (the upstream `LICENSE`, copied here as `LICENSE`)
- tag: v0.1.31
- commit: 3d28a5d
- date: 2026-09-18
- command: python tools/vendor_into_flashinfer.py <flashinfer>

Layout: `device/`, `block/`, `dispatch/` are the shared layers; `topk/` holds the phases,
the kernels and the router.  FlashInfer-specific glue lives in
`flashinfer/topk_varlen/kernels/cutlass_primitives_backend.py`, not here.

Comments inside this directory follow the upstream library's conventions, and their
`docs/...` references point at the upstream repository's docs tree, not FlashInfer's.
The vendoring tool rewrites the imports to relative ones and runs `ruff format`, so the
repository's formatting hooks leave these files unchanged.
