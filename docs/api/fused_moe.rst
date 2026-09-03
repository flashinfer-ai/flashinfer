.. _apifused_moe:

flashinfer.fused_moe
====================

.. currentmodule:: flashinfer.fused_moe

This module provides fused Mixture-of-Experts (MoE) operations optimized for different backends and data types.

Types and Enums
---------------

.. autosummary::
    :toctree: ../generated

    RoutingMethodType
    WeightLayout

Shared activation helpers live in :mod:`flashinfer.tllm_enums` and are used by
both the TRT-LLM and CuteDSL MoE paths.

.. currentmodule:: flashinfer.tllm_enums

.. autosummary::
    :toctree: ../generated

    is_gated_activation

.. currentmodule:: flashinfer.fused_moe

Utility Functions
-----------------

.. autosummary::
    :toctree: ../generated

    convert_to_block_layout
    reorder_rows_for_gated_act_gemm
    interleave_moe_weights_for_sm90_mixed_gemm
    interleave_moe_scales_for_sm90_mixed_gemm
    preprocess_moe_weights_for_sm90_mixed_gemm_humming
    fused_topk_deepseek
    hash_topk

The E8M0 range-clamping, residual-scale factorization, and FP4 payload-rewrite
scheme used by ``preprocess_moe_weights_for_sm90_mixed_gemm_humming`` is adapted
from `Humming <https://github.com/inclusionAI/humming>`_.

Multi-LoRA MoE (BGMV)
---------------------

Batched Gather-Matrix-Vector kernels for serving multiple LoRA adapters on
top of a Mixture-of-Experts layer (shrink + expand).

.. autosummary::
    :toctree: ../generated

    bgmv_moe
    prepare_bgmv_moe
    BGMVMoEBlackwellPlan
    bgmv_moe_shrink
    bgmv_moe_expand
    bgmv_moe_gemm1_lora_delta
    bgmv_moe_gemm2_lora_delta

CUTLASS Fused MoE
-----------------

.. autosummary::
    :toctree: ../generated

    cutlass_fused_moe

cuTile Fused MoE
----------------

.. autosummary::
    :toctree: ../generated

    CuTileBf16Config
    CuTileBf16Runner
    CuTileNvfp4Config
    CuTileNvfp4Runner

TensorRT-LLM Fused MoE
----------------------

.. autosummary::
    :toctree: ../generated

    trtllm_bf16_moe
    trtllm_bf16_routed_moe
    trtllm_fp4_block_scale_moe
    trtllm_fp4_block_scale_routed_moe
    trtllm_fp8_block_scale_moe
    trtllm_fp8_block_scale_routed_moe
    trtllm_fp8_per_channel_scale_moe
    trtllm_fp8_per_channel_scale_routed_moe
    trtllm_fp8_per_tensor_scale_moe
    trtllm_fp8_per_tensor_scale_routed_moe
    trtllm_mxint4_block_scale_moe
    trtllm_mxint4_block_scale_routed_moe

Cake NVFP4 Warp Decode (SM103)
------------------------------

The Cake warp-decode runner is an explicit unified-MoE backend for exact
SM103. Select it with ``CakeWarpDecodeConfig(backend="cake")``; it is not in
the default backend list. The current generated portfolio fails closed outside
these contracts:

* ``(hidden_size, intermediate_size, num_experts, top_k)`` is exactly
  ``(2048, 512, 512, 10)`` or ``(2048, 1536, 60, 4)``;
* the token count is 1--32, routing is ``UnpackedPrecomputed`` with contiguous
  int32 expert IDs and BF16 routing weights, and the activation is the default
  ``SwiGLU()``;
* quantization is NVFP4, finalization and PDL are enabled, and expert
  parallelism, fused shared experts, bias, and LoRA are disabled.

The backend reuses the physical weight and activation layouts prepared by
``TrtllmFp4Config``. One prepared weight dictionary can therefore be
registered for both backend keys without copying::

    cake = CakeWarpDecodeConfig(backend="cake")
    view = cake.prepare_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_experts,
        hidden_size=2048,
        intermediate_size=intermediate_size,
    )

    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    weights.prepare_for("trtllm_fp4_routed", view)

    x_q, x_scale = cake.prepare_activations(x_bf16)
    activations = MoEActivationPack(
        x_q,
        x_scale,
        topk_ids,
        topk_weights_bf16,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU(),
        backend=BackendOptions((cake,)),
        execution=ExecutionConfig(enable_pdl=True),
    )
    output = MoELayer(config)(activations, weights)

The runner prepares its route-map workspace before a timed launch or CUDA
Graph capture and reuses it for the same token count and geometry. Warm up each
shape and routing tensor before capturing it; an unseen workspace shape or an
unvalidated routing-tensor generation during capture is rejected instead of
initializing implicitly. Repeated calls reuse the routing validation receipt
until a normal tensor is modified in place. Inference tensors lack a version
counter, so their receipt is identity/storage based. Every later inference-mode
mutation and graph replay must keep expert IDs in the configured range because
neither path can be revalidated automatically. At most 64 live routing tensors
are retained; validating another distinct tensor fails explicitly, so construct
a new runner for another bounded lifetime.

The runner retains a bounded prepared-workspace cache keyed by execution stream
and geometry. Preparation issues a generation receipt, and the binding records
completion events so explicit re-preparation or release cannot overtake submitted
work. Completion-event handles are retained in a bounded process-lifetime pool
so a live CUDA graph cannot reference a destroyed handle; a generation whose
accepted work cannot be recorded is quarantined instead of being reused. A
recycled allocator address cannot inherit stale metadata. Ordinary
``MoELayer`` calls receive per-stream workspaces automatically. Keep the runner
and its workspaces alive for the lifetime of any captured graph, and do not
concurrently replay multiple low-level graph executables that share one receipt.
Workspace receipts are positive, generation-specific, and single-use; an
unknown, stale, or repeated release is rejected rather than treated as a
successful retirement.
The runner-owned receipt lease strongly retains a workspace until retirement;
if retirement cannot prove completion, the storage remains quarantined until
process exit rather than returning to PyTorch's allocator. The 4096-address
event pool is likewise process-lifetime and requires a process restart after
exhaustion.
The module is also registered in SM103 AOT builds when MoE kernels are enabled.

.. autosummary::
    :toctree: ../generated

    CakeWarpDecodeConfig
    CakeWarpDecodeRunner

Standalone TRT-LLM Gen Routing
------------------------------

The routing stage the TRT-LLM Gen fused MoE launchers run before their GEMMs,
exposed on its own so expert selection and the permutation/padding bookkeeping
can be used (and tested) independently of quantization and GEMM configuration.

.. autosummary::
    :toctree: ../generated

    trtllm_gen_routing
    TrtllmGenRoutingResult

CuteDSL Fused MoE
-----------------

The CuteDSL backends are conditionally available when the
``nvidia-cutlass-dsl`` package is installed.

.. autosummary::
    :toctree: ../generated

    cute_dsl_fused_moe
    cute_dsl_fused_moe_nvfp4
    cute_dsl_fused_moe_mxfp8_mxfp4
    b12x_fused_moe

.. autoclass:: CuteDslMoEWrapper
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: CuteDslMxfp8Mxfp4MoEWrapper
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: B12xMoEWrapper
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

MonoMoE (Single-Kernel Block-FP8, SM90a)
-----------------------------------------

Single-kernel top-K Mixture-of-Experts implementation specialized for the
Qwen3.5-35B block-FP8 shape on Hopper (SM90a). The full pipeline — routing,
up-projection, SiLU, down-projection and reduction — runs inside one kernel
launch. Use :func:`has_monomoe` to check availability before calling.

.. autosummary::
    :toctree: ../generated

    has_monomoe
    get_scratchpad_size_bytes
    alloc_scratchpad
    interleave_for_tma_wgmma_up
    mono_moe
