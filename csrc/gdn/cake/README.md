# FlashInfer GDN non-CP general prefill and decode source

This reproducible package contains only the exact optimized GDN schedule specializations admitted by the frozen general prefill, FP32-state T=1/MTP decode, and promoted BF16 serving contracts. The manifest records every row on SM100a and SM103a. Explicit GDN non-CP requests outside the listed routes must fail closed. Generated CUDA and TVM FFI host shims are source-only.

Indexed prefill requires every state slot to be in `[0, pool_size)`. Indexed decode requires every initial/output state slot to be `-1` or in `[0, state.size(0))`. The public adapters validate CUDA-resident slots asynchronously on the caller stream before dispatch, including during CUDA Graph capture and replay.


## Prefill performance refresh

The generated prefill kernels are validated on B200 and B300. See
[RESULTS.md](RESULTS.md) for the measured comparisons, correctness coverage,
reference revisions, and sanitizer status; [validation-header.json](validation-header.json)
records the qualified manifest and timing protocol.

Use the existing GDN prefill API with `backend="cake_gdn"` to select this backend
explicitly. Automatic prefill dispatch continues to select CuTe. The existing
19 generated decode variants and their public dispatch are preserved.
