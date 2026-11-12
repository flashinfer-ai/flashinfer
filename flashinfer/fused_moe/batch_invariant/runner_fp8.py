# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor batch-invariant MoE backend of the unified MoE layer (``BatchInvariantFp8PerTensorConfig`` ->
``BatchInvariantFp8PerTensorRunner``): the W8A8 classes ``moe.e4m3.<act>`` on SM100 / SM103 / SM107.

The surface is TRTLLM-Gen's fp8 per-tensor one: e4m3 activations (quantized by the caller with the static multiplier
``hidden_states_scale_global``, no activation scale tensor in the pack), the weights prepared from the bf16 experts with the
two calibrated multipliers, every routing input mode (``FromLogits`` runs the class's own routing kernel), every typed
activation of the layer. The combine is the slot-order fp32 fold (``do_finalize=True`` only: the unfinalized triple is
refused), the experts are the dense set (an expert-parallel shard is refused: the class's EP form is not part of this
backend), the tactics are the ranked fp8 grouped records (any tactic gives the same bits), and an unsupported configuration
raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back. As in the bf16 runner, a
routing term the from-logits mode has no class for is refused at the from-logits call, not at construction (the pre-routed
modes never consult it), and a NaN logit follows the shared route kernel's rule. The pack / plan / launch flow is the shared
:class:`.runner.BatchInvariantMoeRunnerBase`'s; this runner sets the fp8 format hooks (the e4m3 activations with 16-byte
rows, the quantized weight view with its scales, the W8A8 chains, the fused fp8 op).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from ...batch_invariant.support import unsupported
from ..api import QuantFormat
from .runner import MOE_BF16_ARCHS, BatchInvariantMoeRunnerBase

CLASS = "moe.e4m3"
#: The compute capabilities the classes are stated for (the dense fp8 GEMM class's parts).
MOE_FP8_ARCHS: Tuple[Tuple[int, int], ...] = MOE_BF16_ARCHS
#: Positions of the packed inputs: the e4m3 activations, the expert ids, the routing weights, the two quantized weights,
#: the four dequantization scales, the intermediate multiplier and the output.
_X, _IDS, _W, _W1, _W2, _SW1, _SW2, _SX, _SINTER, _MINTER, _OUT = range(11)


class BatchInvariantFp8PerTensorRunner(BatchInvariantMoeRunnerBase):
    """The unified layer's runner for the batch-invariant fp8 per-tensor MoE classes."""

    backend_key = "batch_invariant_fp8_per_tensor"
    supported_quant_variants = ((QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),)

    CLASS = CLASS
    ARCHS = MOE_FP8_ARCHS
    X_DTYPE = torch.float8_e4m3fn
    ROW_ALIGN = 16
    X_RULE = "an e4m3 [num_tokens, hidden_size] (the static quantization)"
    ROW_RULE = "a positive multiple of 16 (16-byte e4m3 rows)"
    SCALE_RULE = (
        "no activation scale tensor (the static multiplier lives in the weight view)"
    )
    TACTIC_TAG = "moe.e4m3-v2"  # v2: the tactic carries the swapped flag
    N_WEIGHT_TENSORS = 7

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_experts(self) -> None:
        experts, routing = self.config.experts, self.config.routing
        if experts.local_expert_offset or (
            experts.local_num_experts not in (None, 0, routing.num_experts)
        ):
            raise unsupported(
                CLASS,
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the class's EP form is not part of this backend)",
            )

    def _check_support_extra(self) -> None:
        if not self.config.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    # ---- the weight view -------------------------------------------------------------------------------------------
    def _weights_from_view(self, view):
        from .api_fp8 import MoEWeightsE4M3

        return MoEWeightsE4M3.from_view(view)

    def _weight_tensors(self, w) -> List[torch.Tensor]:
        return [w.w1q, w.w2q, w.s_w1, w.s_w2, w.s_x, w.s_inter, w.m_inter]

    def _weights_from_tensors(self, a, tensors: List[torch.Tensor]):
        from .api_fp8 import MoEWeightsE4M3

        w1q, w2q, s_w1, s_w2, s_x, s_inter, m_inter = tensors
        return MoEWeightsE4M3(
            w1q, w2q, s_w1, s_w2, s_x, s_inter, m_inter, a.gated, a.act, a.params
        )

    def _chain1(self, a):
        from .schedule_fp8 import gemm1_chain

        return gemm1_chain(a.act, a.params, a.gated)

    def _chain2(self):
        from .blocks.epilogue import GEMM2_CHAIN

        return GEMM2_CHAIN

    # ---- launch ----------------------------------------------------------------------------------------------------
    def _launch(self, x, ids, wts, w, out, c1, c2, plan, inputs):
        from .api_fp8 import fused_moe_fp8

        fused_moe_fp8(
            x,
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            plan=plan,
            w_round=getattr(inputs, "w_round", False),
        )
        return out
