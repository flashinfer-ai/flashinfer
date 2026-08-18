# Cake GDN non-CP prefill and T=1 decode source

This reproducible package contains only the exact Cake schedule specializations admitted by the frozen non-CP prefill and FP32-state T=1 decode child contracts. The manifest records every row on SM100a and SM103a. Explicit Cake requests outside the listed routes must fail closed; context-parallel prefill and multi-token/BF16 decode are independent delivery units. Generated CUDA and TVM FFI host shims are source-only.
