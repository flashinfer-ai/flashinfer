# Source provenance

Modified 2026-11-04/05 for a standalone FlashInfer SM100/SM103 HC backend.
This backend builds on SGLang's original HC implementation and ports the
subsequent HC optimizations developed in PR #40552 to FlashInfer.
Upstream notices below apply to inherited source components.

- Original HC source: SGLang commit `52fecfdf0908dca24f4c6799ff5967125cc4110e`,
  including `python/sglang/srt/layers/hyperconnection.py` and
  `python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh`.
  Source copyright: `Copyright 2023-2024 SGLang Team`, Apache-2.0.
- SGLang optimization branch revision: `9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2`.
  Based on upstream `eae134d61fa9ce2e532d7543062cb9c6a7e200d7`.
  Repository: https://github.com/sgl-project/sglang (HC optimization PR #40552).
  The exact local reviewed revision includes follow-up commits; it is not a claim
  that this revision is merged upstream.
- `api.py` derives preparation, packing, scratch constraints, state and buffer
  semantics from `python/sglang/kernels/ops/elementwise/hc_sum_state.py`,
  `hc_sum_state_large.py`, `hc_mix_raw.py`, and
  `python/sglang/srt/layers/hyperconnection_packed.py`. It replaces model-object
  constructors with explicit tensor APIs and a shared workspace contract.
- `runtime.py`, `weights.py` and the core `hyperconnection.py` state types
  extend those tensor APIs with the native
  packing and carried-state ownership semantics from `hyperconnection_packed.py`.
  They retain no serving-framework or model-specific imports. Communication
  policy, parameter/checkpoint ownership and model traversal remain in callers.
- `large.py` retains the mathematical/layout bodies of
  `hc_down_batched_sum.py` and `hc_sum_state_large.py`. Framework classes and
  imports are replaced by a standalone runner.
- `csrc/kernels.cuh` derives from
  `python/sglang/kernels/jit/csrc/elementwise/hc_apply_sum.cuh` and
  `hc_raw_transition.cuh`. Launch geometry, accumulation order and pre-store
  statistics are retained; vectors and bindings use FlashInfer/CUDA/TVM-FFI.
  SGLang's JIT helper/header framework is not bundled.
- `lowm.py` adapts the sum-state HC epilogues from the optimization branch's
  `dense_bf16_gemm_sm100_splitk_epilogue.py`. Its original GEMM was vendored from
  FlashInfer PR #4266, commit `629147317d4149a12e53bcef27808bac380c283f`,
  `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.`, Apache-2.0.
  It preserves this origin and reuses appropriate current FlashInfer primitives.
- `deterministic.py` extracts the fixed-order batched GEMM and tile mapping from
  SGLang's `srt/batch_invariant_ops/batch_invariant_ops.py`, itself adapted from
  https://github.com/thinking-machines-lab/batch_invariant_ops. That upstream
  work is `Copyright 2025 Thinking Machines Lab`, MIT. The complete MIT text is
  included in `licenses/LICENSE.batch-invariant-ops.txt`; the extracted file
  carries both applicable MIT and Apache notices. No vLLM RMS routine or
  DeepGEMM helper is copied into this backend.
  The migrated BMM additionally widens the batch offset when only its output
  exceeds the signed32-bit indexing range; the original input-only guard did
  not cover this legal shape family.

Files newly wiring the interface/JIT contain Apache-2.0 identifiers. The root
`NOTICE` records these origins; the packaging manifest includes the required
license/notice files and CUDA sources.
