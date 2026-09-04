# FlashInfer GDN non-CP general prefill and decode source

This reproducible package contains only the exact optimized GDN schedule specializations admitted by the frozen general prefill, FP32-state T=1/MTP decode, and promoted BF16 serving contracts. The manifest records every row on SM100a and SM103a. Explicit GDN non-CP requests outside the listed routes must fail closed. Generated CUDA and TVM FFI host shims are source-only.

Indexed prefill requires every state slot to be in `[0, pool_size)`. Indexed decode requires every initial/output state slot to be `-1` or in `[0, state.size(0))`. The public adapters validate CUDA-resident slots asynchronously on the caller stream before dispatch, including during CUDA Graph capture and replay.
