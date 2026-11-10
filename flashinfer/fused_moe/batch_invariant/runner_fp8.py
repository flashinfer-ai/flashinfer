# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor batch-invariant MoE backend of the unified MoE layer (``BatchInvariantFp8PerTensorConfig`` ->
``BatchInvariantFp8PerTensorRunner``): the W8A8 classes ``moe.e4m3.<act>`` on SM100 / SM103 / SM107.

The surface is TRTLLM-Gen's fp8 per-tensor one: e4m3 activations (quantized by the caller with the static multiplier
``hidden_states_scale_global``, no activation scale tensor in the pack), the weights prepared from the bf16 experts with the
two calibrated multipliers, every routing input mode (``FromLogits`` runs the class's own routing kernel), every typed
activation of the layer. The combine is the slot-order fp32 fold (``do_finalize=True`` only), the tactics are the ranked fp8
grouped records (any tactic gives the same bits), and an unsupported configuration raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from ..api import MoEActivationPack, MoEWeightPack, QuantFormat, RoutingInputMode
from ..runners import _validate_logits_inputs, _validate_prerouted_inputs
from .runner import (
    MOE_BF16_ARCHS,
    BatchInvariantMoeRunner,
    _configs_from_tactic,
    _tactic,
    routing_term,
)

CLASS = "moe.e4m3"
#: The compute capabilities the classes are stated for (the dense fp8 GEMM class's parts).
MOE_FP8_ARCHS: Tuple[Tuple[int, int], ...] = MOE_BF16_ARCHS
#: Positions of the packed inputs: the e4m3 activations, the expert ids, the routing weights, the two quantized weights,
#: the four dequantization scales, the intermediate multiplier and the output.
_X, _IDS, _W, _W1, _W2, _SW1, _SW2, _SX, _SINTER, _MINTER, _OUT = range(11)


class BatchInvariantFp8PerTensorRunner(BatchInvariantMoeRunner):
    """The unified layer's runner for the batch-invariant fp8 per-tensor MoE classes."""

    backend_key = "batch_invariant_fp8_per_tensor"
    supported_quant_variants = ((QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),)

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        from ..runners import MoERunner

        MoERunner._check_support(self)
        cfg = self.config
        check_device(self.device, CLASS, MOE_FP8_ARCHS)
        try:
            self._activation()
        except ValueError as e:
            raise unsupported(
                CLASS,
                f"activation {cfg.activation!r}",
                "one of the layer's typed activations",
            ) from e
        experts = cfg.experts
        if experts.num_fused_shared_experts:
            raise unsupported(
                CLASS,
                "fused shared experts",
                "num_fused_shared_experts = 0 (a shared expert is one more routed slot)",
            )
        if experts.intermediate_size % 16 or experts.intermediate_size <= 0:
            raise unsupported(
                CLASS,
                f"intermediate_size {experts.intermediate_size}",
                "a positive multiple of 16 (16-byte e4m3 rows)",
            )
        routing = cfg.routing
        if not 1 <= routing.top_k <= min(routing.num_experts, 16):
            raise unsupported(
                CLASS, f"top_k {routing.top_k}", "1 <= top_k <= min(num_experts, 16)"
            )
        if not 1 <= routing.num_experts <= 1024:
            raise unsupported(
                CLASS, f"num_experts {routing.num_experts}", "1 <= num_experts <= 1024"
            )
        self._method, self._grouped_other = routing_term(routing, CLASS)
        if experts.local_expert_offset or (
            experts.local_num_experts not in (None, 0, routing.num_experts)
        ):
            raise unsupported(
                CLASS,
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the class's EP form is not part of this backend)",
            )
        if not cfg.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return x.ndim == 2 and x.dtype == torch.float8_e4m3fn and x.shape[1] % 16 == 0

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._validate_pack_contract(act)
        self._require_built()
        from .api import route, route_plan
        from .api_fp8 import MoEWeightsE4M3

        cfg = self.config
        plan: List[torch.Tensor] = []
        x = act.hidden_states_q
        if x.dtype != torch.float8_e4m3fn or x.ndim != 2:
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "an e4m3 [num_tokens, hidden_size] (the static quantization)",
            )
        if act.hidden_states_scale is not None:
            raise unsupported(
                CLASS,
                "hidden_states_scale",
                "no activation scale tensor (the static multiplier lives in the weight view)",
            )
        x = x.contiguous()
        T = x.shape[0]
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        w = MoEWeightsE4M3.from_view(weights.get_view(self.backend_key))
        if (
            x.shape[1] != w.hidden_size
            or w.num_experts != E
            or cfg.experts.intermediate_size != w.intermediate_size
        ):
            raise unsupported(
                CLASS,
                f"weights (E {w.num_experts}, I {w.intermediate_size}, H {w.hidden_size}) for hidden {x.shape[1]}, E {E}, I {cfg.experts.intermediate_size}",
                "weights prepared for this configuration",
            )
        if x.shape[1] % 16:
            raise unsupported(
                CLASS,
                f"hidden size {x.shape[1]}",
                "a multiple of 16 (16-byte e4m3 rows)",
            )
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"{type(cfg.activation).__name__} with weights prepared for another activation",
                "weights prepared with the layer's activation",
            )
        mode = act.routing_input_mode
        if mode is RoutingInputMode.FromLogits:
            _validate_logits_inputs(act, T, E, type(self).__name__)
            # the group-limited selection is a term of the DeepSeekV3 method; with another method the groups have no
            # meaning in this module (with pre-routed ids the group structure never enters the computation)
            if getattr(self, "_grouped_other", False):
                raise unsupported(
                    CLASS,
                    f"expert groups (n_group / topk_group) with {cfg.routing.method!r} from logits",
                    "the groups with DeepSeekV3, or pre-routed ids",
                )
            if self._method is None:
                raise unsupported(
                    CLASS,
                    f"routing method {cfg.routing.method!r} from logits",
                    "Default, Renormalize, RenormalizeNaive, Sigmoid, SigmoidRenorm, TopK, Llama4, TopKSigmoid or DeepSeekV3",
                )
            bias = act.routing_bias
            if (bias is not None) != self._method.bias:
                raise unsupported(
                    CLASS,
                    "a routing bias" if bias is not None else "no routing bias",
                    "a bias exactly for the methods that select on sigmoid(l) + bias",
                )
            scale = cfg.routing.routed_scaling_factor if self._method.scaled else None
            if self._method.scaled and scale is None:
                scale = 1.0
            from .blocks.scheduler import MAX_E
            from .route_plan import ROUTE_PLAN_MAX_SLOTS, ROUTE_PLAN_MAX_TOKENS

            if (
                0 < T <= ROUTE_PLAN_MAX_TOKENS
                and T * k <= ROUTE_PLAN_MAX_SLOTS
                and cfg.routing.num_experts <= MAX_E
            ):
                # routing and the permutation plan in one launch (a placement: the same bits and integers)
                ids, wts, counts, dest, src = route_plan(
                    act.routing_logits, k, self._method, bias, scale
                )
                plan = [counts, dest, src]
            else:
                ids, wts = route(act.routing_logits, k, self._method, bias, scale)
        else:
            _validate_prerouted_inputs(
                act,
                T,
                k,
                type(self).__name__,
                allowed_weights_dtypes=(torch.bfloat16, torch.float32),
                require_contiguous=True,
            )
            ids = act.topk_ids.contiguous()
            wts = act.topk_weights
            if mode is RoutingInputMode.PackedPrecomputed:
                wts = wts.to(
                    torch.bfloat16
                )  # the packed word carries the weight as bf16
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
        return [
            x,
            ids,
            wts,
            w.w1q,
            w.w2q,
            w.s_w1,
            w.s_w2,
            w.s_x,
            w.s_inter,
            w.m_inter,
            out,
        ] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int) -> Tuple[list, list]:
        from .blocks.fp8_epilogue import GEMM2_CHAIN
        from .schedule import TOP_K, active_experts, rows_per_expert
        from .schedule_fp8 import gemm1_chain, rank_grouped_fp8

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = self._hidden
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        n1 = 2 * I if a.gated else I
        return (
            rank_grouped_fp8(
                rows,
                n1,
                H,
                E,
                gemm1_chain(a.act, a.params, a.gated),
                sm_count,
                family,
                TOP_K,
                n_active=n_act,
            ),
            rank_grouped_fp8(
                rows, H, I, E, GEMM2_CHAIN, sm_count, family, TOP_K, n_active=n_act
            ),
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        self._hidden = int(inputs[_X].shape[1])
        r1, r2 = self._ranked(int(inputs[_X].shape[0]))
        return [_tactic(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        from ..runners import MoERunner

        return MoERunner._cache_key_extras(self) + (
            "moe.e4m3-v2",
        )  # v2: the tactic carries the swapped flag

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_fp8 import MoEWeightsE4M3, fused_moe_fp8

        x, ids, wts, w1q, w2q, s_w1, s_w2, s_x, s_inter, m_inter, out = inputs[:11]
        plan = tuple(inputs[11:14]) if len(inputs) >= 14 else None
        self._hidden = int(x.shape[1])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(int(x.shape[0]))
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic(tuple(tactic))
        a = self._activation()
        w = MoEWeightsE4M3(
            w1q, w2q, s_w1, s_w2, s_x, s_inter, m_inter, a.gated, a.act, a.params
        )
        fused_moe_fp8(x, ids, wts, w, out=out, config1=c1, config2=c2, plan=plan)
        return out
