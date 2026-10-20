.. _apicake_stepfun_moe:

flashinfer.experimental.cake_stepfun_moe
========================================

.. currentmodule:: flashinfer.experimental.cake_stepfun_moe

``cake_stepfun_moe`` is an experimental Cake backend: generated programs for the
StepFun fused MoE -- ``SwiGLUStep(limit)`` with a per-expert clamp limit -- on SM100
(B200 / GB200) and SM103 (B300 / GB300).  Both the API and the backend are
experimental (see :ref:`experimental`); naming the backend in the unified MoE API or
calling one of the functions below is the explicit opt-in and emits an
``ExperimentalWarning`` once.

Every stage of the forward is an exported Cake kernel: routing (Renormalize top-k),
FC1 with the StepFun epilogue, the NVFP4 per-token requantization, FC2 and finalize.
Five weight families are served -- ``nvfp4``, ``nvfp4_bf16tok`` (bf16 FC1 output with
fp32 per-token scales), ``bf16``, ``fp8`` (per-tensor) and ``mxfp8`` -- in the physical
layouts of the matching trtllm-gen configs.

Unified MoE API opt-in
----------------------

The backend is reachable from :mod:`flashinfer.fused_moe` by naming it::

    config = MoEConfig(..., activation=SwiGLUStep(limit=7.0),
                       backend=CakeStepFunConfig(backend="cake_stepfun"))

The weight view must come from ``CakeStepFunConfig.prepare_weights`` (view
``cake_stepfun``, which carries the per-expert ``gemm1_clamp_limit``).  Routing reads
logits (``RoutingInputMode.FromLogits``) or unpacked pre-computed top-k ids and weights
(``RoutingInputMode.UnpackedPrecomputed``); fused shared experts and LoRA deltas are not
supported.  ``FLASHINFER_CAKE_STEPFUN_FULL_PATH`` is read by this entry only (``auto`` /
``1``; the FC1-only hybrid of #6046 was removed by the cake-build/1 migration, so ``0``
raises ``NotImplementedError``).

One-shot functions
------------------

.. autosummary::
    :toctree: ../generated

    stepfun_moe
    routing
    fc1
    requant
    fc2
    finalize

Prepared launches
-----------------

``prepare_stepfun_moe`` validates the arguments, selects the exported units of the
family and tile (``Plan``, ``plans_for``, ``default_plan``), allocates the buffer set
once and returns a ``StepFunMoeLaunch`` whose ``launch()`` performs no allocation and no
host synchronisation and is CUDA-graph capturable; ``outputs()`` is ``[output]`` or, with
``do_finalize=False``, ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
``prepare_routing`` / ``prepare_fc1`` / ``prepare_requant`` / ``prepare_fc2`` /
``prepare_finalize`` bind one stage on caller-owned buffers and return a
``StepFunStageLaunch``.
