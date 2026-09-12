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

Unified MoE API
---------------

Backend-agnostic configuration and layer types. ``QuantConfig`` carries the MMA
weight / activation formats and the layer output format as ``QuantFormat`` axes;
``QuantVariant`` is a deprecated preset that expands to a ``(weight, activation)``
pair.

.. autosummary::
    :toctree: ../generated

    MoELayer
    MoEConfig
    RoutingConfig
    QuantConfig
    QuantFormat
    QuantVariant
    ExpertConfig
    ExecutionConfig
    MoEFinalizeConfig
    BackendOptions
    MoEActivationPack
    MoEWeightPack

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

Cake NVFP4 Warp Decode (SM100/SM103)
------------------------------------

The Cake warp-decode runner is an explicit unified-MoE backend for exact
SM100 and exact SM103. Select it with
``CakeWarpDecodeConfig(backend="cake")``; it is not in the default backend
list. Each device loads only its exact generated target. The current generated
portfolio fails closed outside these contracts:

* ``(activation, hidden_size, intermediate_size, num_experts, top_k)`` is
  exactly ``(SwiGLU(), 2048, 512, 512, 10)``,
  ``(SwiGLU(), 2048, 1536, 60, 4)``,
  ``(SwiGLU(), 2560, 768, 384, 4)``,
  ``(SiLU(), 6144, 1536, 192, 4)``,
  ``(SwiGLU(), 2048, 768, 128, 8)``,
  ``(SwiGLU(), 4096, 1536, 128, 8)``,
  ``(SwiGLU(), 2048, 512, 256, 8)``,
  ``(SwiGLU(), 4096, 1024, 512, 10)``,
  ``(SwiGLU(), 3072, 1536, 256, 8)``,
  ``(SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 3072, 128, 4)``, or
  ``(SiTU(gate_scale=4.0, linear_scale=25.0), 3584, 3072, 896, 16)``;
* the token count is 1--32, routing is ``UnpackedPrecomputed`` with contiguous
  int32 expert IDs and BF16 routing weights;
* quantization is NVFP4, finalization and PDL are enabled, and expert
  parallelism, fused shared experts, bias, and LoRA are disabled.

The backend reuses the physical weight and activation layouts prepared by
``TrtllmFp4Config``. Logical GEMM1 weights have
``intermediate_size * (2 if activation.is_gated else 1)`` rows: SwiGLU and SiTU use
``2 * intermediate_size`` gate/up rows while standalone SiLU uses
``intermediate_size`` rows. The packed E2M1 weights use the production
``MajorK`` 32-row MMA shuffle: physical row ``p`` is restored at logical row
``(p & ~31) + ((p & 7) << 2) + ((p & 31) >> 3)``. Their E4M3 block scales use
the production ``R128c4`` layout. Default SwiGLU and SiTU weight dictionaries
can be registered for both backend keys without copying. Parameterized SwiGLU consumes
the prepared per-expert ``gemm1_alpha``, ``gemm1_beta``, and ``gemm1_clamp_limit``
FP32 tensors. SiTU consumes ``gemm1_alpha`` as its gate scale and ``gemm1_beta``
as its linear scale, and requires no clamp tensor. The prepared values default to
the configured activation; per-expert tensor changes are consumed on every launch
and CUDA Graph replay.

Parameterized SwiGLU uses different beta/clamp units in the two backends.
Cake consumes the logical parameters. The official TRT-LLM runner consumes
``beta / d`` and ``clamp_limit / d``, where
``d = view["output1_scale_gate_scalar"]``. For non-unit gate scales, prepare a
separate official dictionary before launching; alpha and the physical weight
and scale tensors remain shared::

    official_view = dict(view)
    d = view["output1_scale_gate_scalar"]
    official_view["gemm1_beta"] = (view["gemm1_beta"] / d).contiguous()
    official_view["gemm1_clamp_limit"] = (view["gemm1_clamp_limit"] / d).contiguous()

Keep the logical dictionary registered for ``"cake"`` and register
``official_view`` for ``"trtllm_fp4_routed"``. The derived buffers are FP32
per-expert tensors. After changing logical beta, clamp limit, or gate scale,
refresh their contents with ``copy_`` before the next official launch or replay.
Perform conversion outside timing and CUDA Graph capture, preserving buffer
addresses already used by a captured graph.

Standalone SiLU has no supported TRT-LLM routed-MoE peer and its dictionary
must be registered only for ``"cake"``. The default-SwiGLU example below shares
its dictionary directly::

    cake = CakeWarpDecodeConfig(backend="cake")
    activation = SwiGLU()  # Or SiLU() for H6144/I1536/E192/top-k 4.
    view = cake.prepare_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )

    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    if activation == SwiGLU():
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
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
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
The matching module is also registered in exact SM100 and SM103 AOT builds when
MoE kernels are enabled. SM100 does not load or fall back to the SM103 module,
and other compute capabilities are rejected.

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

    cute_dsl_fused_moe_bf16
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
