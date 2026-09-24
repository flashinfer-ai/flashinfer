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
weight / activation formats and the layer output format as ``QuantFormat`` axes.

.. autosummary::
    :toctree: ../generated

    MoEConfig
    RoutingConfig
    QuantConfig
    QuantFormat
    ExpertConfig
    ExecutionConfig
    MoEFinalizeConfig
    BackendOptions
    MoEActivationPack
    MoEWeightPack

``MoELayer`` is the official entry point of this API: both its constructor and
its call operator are decorated with ``@flashinfer_api``, so they participate in
``FLASHINFER_LOGLEVEL`` logging and ``FLASHINFER_DUMP_*`` capture. The lower-level
per-backend functions above remain official in their own right — the two layers
are designed to co-exist, and neither supersedes the other.

.. autoclass:: MoELayer
    :members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __call__

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

AlphaMoE Router (SM100/SM103)
-----------------------------

The standalone AlphaMoE frontend converts FP32 logits into top-k weights and an
expert-grouped, block-aligned route plan. Selection and ordering among exactly
equal routed logits are unspecified. The resulting plan can feed either the
AlphaMoE W8A8 or NVFP4 compute path and can be reused across launches to avoid
steady-state allocation.

.. autosummary::
    :toctree: ../generated

    AlphaMoERoutePlan
    allocate_alphamoe_route_plan
    alphamoe_fused_router

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
    CuTileMxfp4Bf16Config
    CuTileMxfp4Bf16Runner
    CuTileMxfp4Config
    CuTileMxfp4Runner
    CuTileNvfp4Bf16Config
    CuTileNvfp4Bf16Runner
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

AlphaMoE FP8 Block-Scaled MoE (SM100/SM103)
--------------------------------------------

.. autosummary::
    :toctree: ../generated

    alphamoe_interleave_gated_weights
    alphamoe_fp8_block_scale_aligned_moe

AlphaMoE NVFP4 (SM100/SM103)
-----------------------------

The AlphaMoE path consumes packed E2M1 activations and weights with linear
per-16 E4M3 scales. The aligned entry consumes an existing routing plan; the
routed entry aligns the supplied expert IDs before compute. Shape-selected
stages perform gate/up projection, SwiGLU, NVFP4 requantization and down
projection. The aligned path uses FP32 scratch; selected routed paths use
FP32 or BF16 route storage before finalizing into caller-owned BF16 output.
Neither entry resets the caller's initial output. Three contiguous FP32
``[E]`` tensors provide the per-expert static ModelOpt scales: the gate scale
is applied before SiLU, the up scale before SwiGLU multiplication, and the down
scale before route weighting.

.. autosummary::
    :toctree: ../generated

    alphamoe_nvfp4_aligned_moe
    alphamoe_nvfp4_routed_moe
    prepare_nvfp4_w1_data
    prepare_nvfp4_w1_gate_up_data
    prepare_nvfp4_w1_gate_up_scales
    prepare_nvfp4_w1_scales
    prepare_nvfp4_w2_scales

AlphaMoE NVFP4 prepared weight scales
-----------------------------------

The aligned entry consumes an existing routing plan. The routed entry also
aligns raw expert IDs and initializes the accumulation buffer. Both preserve
caller-owned BF16 output and retain the raw E4M3 scale inputs. Shape-selected
compute stages perform gate/up projection, SwiGLU, NVFP4 requantization, down
projection and weighted accumulation.

``prepare_nvfp4_w1_scales`` and ``prepare_nvfp4_w2_scales`` permute existing
scale bytes into immutable uint8 tensor-map panels. They perform no scale
arithmetic or requantization. Call them after the final device-local weight
layout is established and before warmup or graph capture. Keep the raw scale
tensors and the resulting panels alive for the layer's current weight load.
A subsequent weight load must replace the panels; they are derived buffers and
need not be saved in the model checkpoint.

For raw gate/up scales ``[E,N,K/16]``, the prepared shape is
``[E*(N/128)*(K/256),16,128]``. For raw down scales
``[E,K,N/32]``, it is ``[E*(K/128)*(N/256),8,128]``. At local dimensions
``E=256, N=1024, K=6144`` these require an additional 96 MiB and 48 MiB,
respectively, per weight pair. Preparation and its memory cost are separate
from repeated-request kernel timing.

Supply the buffers through optional keywords on either entry::

    from flashinfer.fused_moe import (
        prepare_nvfp4_w1_scales,
        prepare_nvfp4_w2_scales,
    )

    # Once, after device-local weight loading:
    w1_scale_prepared = prepare_nvfp4_w1_scales(gemm1_weights_scale)
    w2_scale_prepared = prepare_nvfp4_w2_scales(gemm2_weights_scale)
    prepared_scales = {
        "w1_scale_prepared": w1_scale_prepared,
        "w2_scale_prepared": w2_scale_prepared,
    }

    # Existing aligned or routed call: retain every raw argument and append
    # **prepared_scales. Reuse these same tensors for subsequent requests.

Omitting optional prepared scale tensors retains the existing scale paths.
The routed prepared-data paths below use W1 scales; the adjacent gate/up path
has its own paired scale carrier. Supplied prepared tensors must satisfy the
API's dtype, device, shape and alignment checks. No weight preparation occurs
inside a routed request.

AlphaMoE NVFP4 prepared gate/up data
----------------------------------

``prepare_nvfp4_w1_data`` permutes contiguous uint8 packed W1 weights
``[E,N,K/2]`` into ``[E*(N/128)*(K/256),128,128]`` panels. It preserves every
packed byte without dequantization or requantization and requires
``N % 128 == 0`` and ``K % 256 == 0``. The buffer occupies the same number of
bytes as raw W1: an additional 768 MiB at ``E=256, N=1024, K=6144``.

With compatible ``w1_scale_prepared`` and ``w1_data_prepared``, the routed
entry selects prepared-data paths for ``M=8``, ``M=128`` and ``M=512`` at
exactly ``N=1024, K=6144, E=256, top_k=8, block_m=8``. Other dimensions retain
their existing selection. Omitting ``w1_data_prepared`` retains the scale-only
or raw path. These data keywords belong to the routed entry, not the aligned
entry. The paths preserve caller output and use BF16 expert-route storage.
The eight-token path also keeps a separate FP32 initial-output seed; the
128-token and 512-token prepared-data paths finalize directly into caller
output in route order.

Prepare and pass ordinary data panels to the routed entry::

    from flashinfer.fused_moe import prepare_nvfp4_w1_data

    # Once, after the final device-local weight load:
    w1_data_prepared = prepare_nvfp4_w1_data(gemm1_weights)
    prepared_routed = dict(
        prepared_scales, w1_data_prepared=w1_data_prepared
    )
    # Append **prepared_routed to the existing routed call, retaining raw inputs.

AlphaMoE NVFP4 adjacent gate/up panels
------------------------------------

For the exact 512-token shape above, the routed entry also accepts
``w1_gate_up_data_prepared`` and ``w1_gate_up_scale_prepared``. These two
carriers place each gate panel next to its matching up panel. Supply both or
neither: supplying only one raises ``ValueError``. Both must be contiguous
uint8 tensors on the raw W1 device, with 16-byte-aligned addresses and these
shapes, where ``R = E*(N/256)*(K/256)``:

* data: ``[R,256,128]`` from ``prepare_nvfp4_w1_gate_up_data(gemm1_weights)``;
* scales: ``[R,32,128]`` from ``prepare_nvfp4_w1_gate_up_scales(
  w1_scale_prepared, gemm1_weights.shape)``.

Here ``N`` counts the combined gate/up rows in raw W1 ``[E,N,K/2]``. The
preparation requires ``N % 256 == 0`` and ``K % 256 == 0``. The scale helper
consumes the output of ``prepare_nvfp4_w1_scales``, not raw E4M3 scales. Both
helpers only permute bytes. At the supported local dimensions, the adjacent
data and scale buffers add 768 MiB and 96 MiB, respectively. Retaining ordinary
prepared-data panels as well incurs their separate memory cost.

A valid adjacent pair takes precedence at exactly
``M=512, N=1024, K=6144, E=256, top_k=8, block_m=8`` with supported route
metadata and 4-byte-aligned activation, W1 and W2 scale addresses. It does not
require the separate
``w1_scale_prepared``, ``w1_data_prepared`` or ``w2_scale_prepared`` arguments
at call time. Raw weights and raw scales remain required arguments. For other
shapes the adjacent pair does not select this specialization. Omitting both
retains the existing prepared-data, scale-only or raw selection. Invalid
supplied carriers are rejected rather than silently replaced.

Prepare the pair once and reuse it across routed requests::

    from flashinfer.fused_moe import (
        prepare_nvfp4_w1_scales,
        prepare_nvfp4_w1_gate_up_data,
        prepare_nvfp4_w1_gate_up_scales,
    )

    w1_scales = prepare_nvfp4_w1_scales(gemm1_weights_scale)
    adjacent_routed = {
        "w1_gate_up_data_prepared": prepare_nvfp4_w1_gate_up_data(gemm1_weights),
        "w1_gate_up_scale_prepared": prepare_nvfp4_w1_gate_up_scales(
            w1_scales, gemm1_weights.shape
        ),
    }
    # Append **adjacent_routed to the existing routed call; retain all raw inputs.

Prepare after the final device-local load or shard layout is established and
before warmup or CUDA graph capture. Keep the raw tensors and selected panels
alive and immutable; rebuild the panels whenever those weights or scales are
replaced. Shape checks cannot establish that a panel belongs to the current
weight values. The helpers do not maintain an automatic cache or fetch model
files. Reuse existing read-only model files as loading inputs, and keep any
optional derived-panel cache separate from the original checkpoint. Reuse
layer-owned device panels across requests instead of rebuilding them per call.
Prepared panels are derived buffers and need not be saved in the model
checkpoint. Preparation time and memory remain outside routed kernel timing.

The aligned entry keeps its existing ``None`` return value. The routed entry
returns the caller's output tensor. Neither entry resets caller output; provide
a zeroed output when a fresh result is wanted. The chosen implementation may
use separate compute launches and scratch storage. It does not promise a
single-kernel or entirely on-chip intermediate implementation.

Cake NVFP4 Warp Decode (SM100/SM103)
------------------------------------

The Cake warp-decode runner is an explicit unified-MoE backend for exact
SM100 and exact SM103. Select it with
``CakeWarpDecodeConfig(backend="cake")``; it is not in the default backend
list. Each device loads only its exact generated target. The current generated
portfolio fails closed outside these contracts:

* ``(activation, hidden_size, intermediate_size, num_experts, top_k)`` is
  exactly ``(SwiGLU(), 2048, 512, 512, 10)``,
  ``(SwiGLU(), 2048, 1536, 60, 4)``, or
  ``(SiLU(), 6144, 1536, 192, 4)``;
* the token count is 1--32, routing is ``UnpackedPrecomputed`` with contiguous
  int32 expert IDs and BF16 routing weights;
* quantization is NVFP4, finalization and PDL are enabled, and expert
  parallelism, fused shared experts, bias, and LoRA are disabled.

The backend reuses the physical weight and activation layouts prepared by
``TrtllmFp4Config``. Logical GEMM1 weights have
``intermediate_size * (2 if activation.is_gated else 1)`` rows: SwiGLU uses
``2 * intermediate_size`` gate/up rows while standalone SiLU uses
``intermediate_size`` rows. The packed E2M1 weights use the production
``MajorK`` 32-row MMA shuffle: physical row ``p`` is restored at logical row
``(p & ~31) + ((p & 7) << 2) + ((p & 31) >> 3)``. Their E4M3 block scales use
the production ``R128c4`` layout. A SwiGLU weight dictionary can therefore be
registered for both backend keys without copying. Standalone SiLU has no
supported TRT-LLM routed-MoE peer and its dictionary must be registered only
for ``"cake"``::

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

Prims-TS Fused MoE
------------------

Experimental Blackwell (SM100) Prims-TS backends.  Public entry points match
the corresponding ``trtllm_*`` APIs.

.. autosummary::
    :toctree: ../generated

    prims_ts_bf16_moe
    prims_ts_bf16_routed_moe
    prims_ts_fp4_block_scale_moe
    prims_ts_fp4_block_scale_routed_moe
    prims_ts_fp8_block_scale_moe
    prims_ts_fp8_block_scale_routed_moe
    prims_ts_fp8_per_tensor_scale_moe

These symbols are defined in the backend modules and re-exported from
:mod:`flashinfer.fused_moe`:

* :func:`flashinfer.fused_moe.backends.prims_ts.bf16_op.prims_ts_bf16_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.bf16_op.prims_ts_bf16_routed_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.fp4_op.prims_ts_fp4_block_scale_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.fp4_op.prims_ts_fp4_block_scale_routed_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.fp8_op.prims_ts_fp8_block_scale_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.fp8_op.prims_ts_fp8_block_scale_routed_moe`
* :func:`flashinfer.fused_moe.backends.prims_ts.fp8_op.prims_ts_fp8_per_tensor_scale_moe`

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
