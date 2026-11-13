# `kda_vibecuda` — Experimental SM100a / SM103a Recurrent KDA Prefill Backend

- **Owner:** `@Edenzzzz`
- **Entry point:** `flashinfer.kda.recurrent_kda(..., backend="vibecuda")`
- **Targets:** NVIDIA Blackwell SM100a (CC 10.0, B200/GB200) and SM103a (CC 10.3, B300/GB300), JIT-compiled on first use.

## Overview

`flashinfer.experimental.kda_vibecuda` provides an opt-in JIT-only recurrent Kimi Delta Attention (KDA) prefill backend for `bfloat16` `D = 128` full-attention / linear-attention prefill workloads on SM100a and SM103a GPUs. It implements five specialized schedules under the same numerical and workspace contract as `backend="cake"`:

1. **M128 slab schedule** (`vibecuda_flashkda_bf16_fused_m128_slab.cu`) for short/mid fixed and packed prefill chains.
2. **Split-sequence affine prefix schedule** (`RunM128SplitFull` with `KDA_FUSE_FIXUP`, `P=2` carry-copy specialization, and dead-map progress-flag pruning) for ultra-long low-batch/low-head prefill (`N * H <= 32`, `max_chunks >= 256`).
3. **Head-family `fam2` schedule** (`fam2h12`, `fam2h64`, `fam2h96` on SM100a; `fam2h64`, `fam2h96` on SM103a) with compile-time head specialization and direct TMA descriptor reuse.
4. **Device-planned persistent M128** (`vibecuda_flashkda_bf16_persistent_m128.cu`) for deep packed workloads (`N * H >= 4 * sm_count`).
5. **Fused-N16 (`bt16`) schedule** for shallow packed `H = 12` (`N <= 5`) and short fixed `H = 12` (`B = 1, T <= 1024` on SM103a).

## Usage

```python
import torch
from flashinfer.kda import recurrent_kda

out, final_state = recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log=A_log,
    dt_bias=dt_bias,
    initial_state=initial_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    lower_bound=-5.0,
    cu_seqlens=cu_seqlens,
    backend="vibecuda",
)
```

## Graduation Criteria

- Consolidate overlapping M128/slab/fam2 translation units with the core `csrc/kda/` templates to keep wheel binary size and AOT build times bounded before registering in `flashinfer/aot.py`.
- Integrate winning schedule families (e.g. `fam2h64`/`fam2h96` and fused-fixup split-seq) into `backend="auto"` routing after multi-SKU serving validation.
