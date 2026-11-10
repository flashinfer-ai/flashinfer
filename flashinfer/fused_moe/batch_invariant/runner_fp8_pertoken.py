# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-token fp8 batch-invariant MoE backend of the unified MoE layer (``BatchInvariantFp8PerTokenConfig`` ->
``BatchInvariantFp8PerTokenRunner``): the classes ``moe.e4m3.<act>.pertoken`` on SM100 / SM103 / SM107.

The surface is the per-token / per-channel fp8 recipe (``QuantConfig(weight=FP8PerChannel, activation=FP8PerToken)``): the
activations arrive as bf16 ``[num_tokens, H]`` (``hidden_states_q`` bf16, no scale tensor: the class's quantizer writes the codes
and the per-token scales straight into the grouped operand, fused with the scatter) or as the caller's per-token quantization
(e4m3 ``hidden_states_q`` with ``hidden_states_scale`` fp32 ``[num_tokens]``); the weights are the per-channel view of
``prepare_weights``; every routing input mode; every typed activation of the layer. The combine is the slot-order fp32 fold
(``do_finalize=True`` only), the tactics are the ranked fp8 records (any tactic gives the same bits), and an unsupported
configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch

from ...batch_invariant.support import unsupported
from ..api import MoEActivationPack, MoEWeightPack, QuantFormat, RoutingInputMode
from ..runners import _validate_logits_inputs, _validate_prerouted_inputs
from .runner import _configs_from_tactic
from .runner_fp8 import CLASS as CLASS_FP8
from .runner_fp8 import BatchInvariantFp8PerTensorRunner

CLASS = CLASS_FP8 + ".pertoken"
#: Positions of the packed inputs: the activations (bf16, or e4m3 with the scales), the per-token scales (or a 1-element
#: placeholder), the expert ids, the routing weights, the two quantized weights, the two per-channel scale tables and the output.
_X, _SX, _IDS, _W, _W1, _W2, _SW1, _SW2, _OUT = range(9)


class BatchInvariantFp8PerTokenRunner(BatchInvariantFp8PerTensorRunner):
    """The unified layer's runner for the batch-invariant per-token fp8 MoE classes (the per-tensor runner's support rules; its own
    inputs, ranking and launch)."""

    backend_key = "batch_invariant_fp8_per_token"
    supported_quant_variants = ((QuantFormat.FP8PerChannel, QuantFormat.FP8PerToken),)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return (
            x.ndim == 2
            and x.dtype in (torch.bfloat16, torch.float8_e4m3fn)
            and x.shape[1] % 16 == 0
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._validate_pack_contract(act)
        self._require_built()
        from .api import route, route_plan
        from .api_fp8 import MoEWeightsE4M3PerChannel

        cfg = self.config
        plan: List[torch.Tensor] = []
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "a bf16 [num_tokens, hidden_size] (quantized per token by the class) or an e4m3 one with hidden_states_scale",
            )
        pre = x.dtype == torch.float8_e4m3fn
        sx = act.hidden_states_scale
        if pre and (
            sx is None
            or sx.ndim != 1
            or sx.shape[0] != x.shape[0]
            or sx.dtype != torch.float32
        ):
            raise unsupported(
                CLASS,
                "e4m3 hidden_states_q without a per-token hidden_states_scale",
                "hidden_states_scale as a float32 [num_tokens] (the caller's per-token scales)",
            )
        if not pre and sx is not None:
            raise unsupported(
                CLASS,
                "hidden_states_scale with bf16 activations",
                "no scale tensor (the class quantizes the bf16 tokens itself)",
            )
        x = x.contiguous()
        T = x.shape[0]
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        w = MoEWeightsE4M3PerChannel.from_view(weights.get_view(self.backend_key))
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
            if self._method is None:
                raise unsupported(
                    CLASS,
                    f"routing method {cfg.routing.method!r} from logits",
                    "Default, Renormalize, RenormalizeNaive, Sigmoid, SigmoidRenorm or DeepSeekV3 with n_group <= 1",
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
                wts = wts.to(torch.bfloat16)
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
        sx_t = (
            sx.contiguous()
            if pre
            else torch.zeros(1, dtype=torch.float32, device=x.device)
        )
        return [x, sx_t, ids, wts, w.w1q, w.w2q, w.s_w1, w.s_w2, out] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int) -> Tuple[list, list]:
        from .api_fp8 import GEMM2_CHAIN_PERTOKEN
        from .blocks.fp8_epilogue import Fp8Chain
        from .schedule import TOP_K, active_experts, rows_per_expert
        from .schedule_fp8 import rank_grouped_fp8

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
        chain1 = Fp8Chain(
            "glu" if a.gated else "act", a.act, "bf16", a.params, True, True
        )
        return (
            rank_grouped_fp8(
                rows, n1, H, E, chain1, sm_count, family, TOP_K, n_active=n_act
            ),
            rank_grouped_fp8(
                rows,
                H,
                I,
                E,
                GEMM2_CHAIN_PERTOKEN,
                sm_count,
                family,
                TOP_K,
                n_active=n_act,
            ),
        )

    def _cache_key_extras(self) -> tuple:
        from ..runners import MoERunner

        return MoERunner._cache_key_extras(self) + ("moe.e4m3.pertoken-v1",)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_fp8 import MoEWeightsE4M3PerChannel, fused_moe_fp8_pertoken

        x, sx, ids, wts, w1q, w2q, s_w1, s_w2, out = inputs[:9]
        plan = tuple(inputs[9:12]) if len(inputs) >= 12 else None
        self._hidden = int(x.shape[1])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(int(x.shape[0]))
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic(tuple(tactic))
        a = self._activation()
        w = MoEWeightsE4M3PerChannel(w1q, w2q, s_w1, s_w2, a.gated, a.act, a.params)
        pre = x.dtype == torch.float8_e4m3fn
        fused_moe_fp8_pertoken(
            x,
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            plan=plan,
            x_scale=sx if pre else None,
        )
        return out
