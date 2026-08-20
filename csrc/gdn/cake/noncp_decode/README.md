# Cake GDN non-CP prefill and decode source

This reproducible package contains only the exact Cake schedule specializations admitted by the frozen non-CP prefill, FP32-state T=1/MTP decode, and promoted BF16 serving contracts. The manifest records every row on SM100a and SM103a. Explicit Cake requests outside the listed routes must fail closed; context-parallel prefill is an independent delivery unit. Generated CUDA and TVM FFI host shims are source-only.
