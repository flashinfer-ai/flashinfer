.. _apicake_mxfp4_situ_moe:

flashinfer.fused_moe.cake / cake_cute (MXFP4 SiTU routed MoE)
==============================================================

The Cake MXFP4 x MXFP8 SiTU routed MoE is the Kimi K3 W4A8 expert layer (MXFP4
weights, MXFP8 activations, the SiTU activation with per-expert ``beta`` /
``linear_beta``, top-k finalize into a BF16 output) reproduced as one set of
Cake programs and lowered to two independent backends.  Both packages come from
the same pinned Cake revision and carry a content-addressed manifest (schema
``cake.library_export.v5``) whose ``producer_revision`` is that revision:

``backend="cake"`` -- :mod:`flashinfer.fused_moe.cake`
   CUDA C++ translation units with ``tvm_ffi_utils`` bindings under
   ``csrc/fused_moe/cake_mxfp4_situ_moe/cuda/sm_103a/``, JIT-built through
   FlashInfer's TVM-FFI JIT.  Keeps the pointer TMA ABI of the validated
   production build (a caller-owned descriptor workspace per launch, owned by
   the plan).
``backend="cake_cute"`` -- :mod:`flashinfer.fused_moe.cake_cute`
   CuTe DSL device modules with ``cutedsl_tvm_ffi`` bindings under
   ``csrc/fused_moe/cake_mxfp4_situ_moe/cute/sm_103a/``, compiled through the
   DSL's ``cute.compile`` on first use (``pip install nvidia-cutlass-dsl``).
   By-value grid-constant TMA ABI.

Both target ``sm_103a`` (B300 / GB300).  The kernels are the exact reproduction
of the hand-written CuTe DSL swap-AB MoE chain: same warp roles, barrier and
pipeline structure, tile forms, TMEM layout, epilogue instruction classes,
programmatic-dependent-launch placement, persistent scheduler protocol and
workspace layout; numerics are frozen (FP32 accumulation, SiTU FP32 sequence,
UE8M0 round-up requantization, BF16 route-weight scaling before the reduce-add).

Host plan
---------

Each package ships the same host plan, rendered per backend
(``cake_mxfp4_situ_moe_plan.py``).  Its decision table is the Cake plan's own
source, rendered verbatim from the pinned revision; it reproduces the
hand-written ``CuteDslMxfp4MoEWrapper`` / ``Mxfp4MoESwapAbPlan`` decision for
decision (``swapab_max_tokens``, tile policy, group rows, stage depths, weight
grouping, fused-routing cap, PDL chain, workspace layout, launch sequence).

.. code-block:: python

    import torch
    from flashinfer.fused_moe.cake import CakeMxfp4MoEWrapper, prepare_cake_mxfp4_weights
    # or: from flashinfer.fused_moe.cake_cute import ...

    # Once per rank (pure torch): canonical [up | gate] MXFP4 bank -> tile-major kernel operands.
    weights = prepare_cake_mxfp4_weights(w1, w1_scale, w2, w2_scale)

    runner = CakeMxfp4MoEWrapper(
        num_experts, top_k, hidden_size, intermediate_size,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,  # expert parallel
        # intermediate_shard=intermediate_size // tp_size,                            # or MoE tensor parallel
        backend="cake",                                                                # "cake_cute" in cake_cute
    )
    decision = runner.decide(num_tokens)          # host only; decision.supported / decision.reason
    workspace = torch.empty(runner.get_workspace_size(num_tokens), dtype=torch.uint8, device="cuda")

    # Outside CUDA-graph capture: binds the caller-owned buffers, loads the three kernels, runs one warmup.
    plan = runner.plan(x, x_sf, topk_ids, topk_weights,
                       weights["w1"], weights["w1_sf"], weights["w2"], weights["w2_sf"],
                       beta=beta, linear_beta=linear_beta, workspace=workspace, output=out)
    plan.run()                                    # graph-capturable; no allocation, no host sync

Inputs: ``x`` ``[T, H]`` ``float8_e4m3fn`` with ``x_sf`` ``[T, H/32]`` UE8M0 codes
(``uint8``); ``topk_ids`` ``[T, top_k]`` ``int32`` global expert ids;
``topk_weights`` ``[T, top_k]`` BF16 or FP32 (``None`` selects the packed
BF16-in-``int32`` route layout); ``beta`` / ``linear_beta`` FP32
``[num_local_experts]``; ``out`` ``[T, H]`` BF16.  The weight bank is the
canonical rank-local ``w1`` ``[L, 2 * I_shard, H/2]`` (up rows then gate rows)
with ``w1_scale`` ``[L, 2 * I_shard, H/32]`` and ``w2`` ``[L, H, I_shard/2]``
with ``w2_scale`` ``[L, H, I_shard/32]`` (``uint8``).

Rows the plan refuses (``decision.supported`` is ``False``) name the hand-written
selection that is not built by this chain: the dense grouped-GEMM path above
``swapab_max_tokens``, the wide 192-row form, the hybrid, mixed and split forms,
the two-stage finalize (MoE-TP rows above 16 tokens) and the generic
``moe_sort`` routing above the fused-routing cap.  ``plan`` raises
``NotImplementedError`` with that reason.

Generated kernels
-----------------

Kernels are selected through the trace-time constants recorded in each module's
``route.form`` (``kind``, ``n_tile``, ``kbps``, ``m_group`` for the GEMM forms,
the ``RoutingConfig`` fields for the routing forms); module file names are
content fingerprints.  ``gate`` names the validation each form went through in
Cake: ``e2e_registry`` (compile, static analysis and GPU correctness on both
backends in Cake's kernel registry), ``plan_gate`` (selected by the plan on the
contract rows and validated end to end through the plan), ``plan_variant`` (a
routing-mode variant of a plan-gated form, reachable through the public API but
not on a contract row).

.. list-table:: Generated kernel forms (``route.stage``)
   :header-rows: 1

   * - Stage
     - Kind
     - Backends
     - Gate
   * - ``gemm2_swapab_finalize_n8``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n16``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n32``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n32_k12``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n64``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n128``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_swapab_finalize_n192_2cta``
     - gemm2_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm1_swapab_situ_n32``
     - gemm1_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm1_swapab_situ_n8_k8``
     - gemm1_swapab
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_dense_finalize_n192``
     - gemm2_dense
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_dense_finalize_n128``
     - gemm2_dense
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_dense_finalize_n256``
     - gemm2_dense
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm2_dense_finalize_n256_2cta``
     - gemm2_dense
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm1_dense_situ_m128_n256``
     - gemm1_dense
     - cake, cake_cute
     - ``e2e_registry``
   * - ``routing_ep8_decode``
     - routing
     - cake, cake_cute
     - ``e2e_registry``
   * - ``routing_cluster8``
     - routing
     - cake, cake_cute
     - ``e2e_registry``
   * - ``routing_split``
     - routing
     - cake, cake_cute
     - ``e2e_registry``
   * - ``token_index_top8``
     - token_index
     - cake, cake_cute
     - ``e2e_registry``
   * - ``route_preprocess_packed``
     - route_preprocess
     - cake, cake_cute
     - ``e2e_registry``
   * - ``dispatch_mixed_lists``
     - dispatch
     - cake, cake_cute
     - ``e2e_registry``
   * - ``dispatch_mixed192_dual``
     - dispatch_mixed
     - cake, cake_cute
     - ``e2e_registry``
   * - ``moe_sort_init``
     - moe_sort_init
     - cake
     - ``e2e_registry``
   * - ``moe_sort_coop_dual_mixed``
     - moe_sort_coop
     - cake
     - ``e2e_registry``
   * - ``finalize_top8``
     - finalize
     - cake, cake_cute
     - ``e2e_registry``
   * - ``finalize_top8_split``
     - finalize
     - cake, cake_cute
     - ``e2e_registry``
   * - ``gemm1_swapab_situ_n16``
     - gemm1_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``gemm2_swapab_finalize_n8_m2``
     - gemm2_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``gemm2_swapab_finalize_n8_k8``
     - gemm2_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``gemm2_swapab_finalize_n8_k12``
     - gemm2_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``gemm2_swapab_finalize_n16_k8``
     - gemm2_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``gemm2_swapab_finalize_n32_k8``
     - gemm2_swapab
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c1``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c1_single``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c2``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c3``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c4``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c5``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c6``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c7``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c8``
     - routing
     - cake, cake_cute
     - ``plan_gate``
   * - ``routing_separate_bf16_c8_l16384``
     - routing
     - cake, cake_cute
     - ``plan_gate``

Forms this family has but a backend does not build are declared in that
package's manifest under ``contract.unsupported_forms`` with the reason
(``backend="cake_cute"``: the cooperative ``moe_sort`` pair, whose register
state crosses ``grid.sync``, which the CuTe DSL backend refuses; the CUDA
package carries both kernels).

Manifest
--------

``contract`` of each manifest records the operator, the backend, the Cake
revision, the architecture, the PDL launch state, the form table (with
``gate`` and ``backends`` per form), the served geometry and contract rows of
the plan, the unsupported forms, and the SHA-256 of every integration file of
the package (plan, kernel loader, ``__init__``, tests, this page).  The kernel
loaders authenticate every generated source against the manifest before it is
compiled.
