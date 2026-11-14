# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 x MXFP4 pair of the batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` with
``QuantConfig(weight=MXFP4, activation=MXFP4)`` -> ``BatchInvariantFp4Runner`` -> ``BatchInvariantMoeMxfp4Runner``, this
module's implementation runner; :mod:`.runner_fp4` holds the backend).

The runner serves the classes ``moe.mxfp4.<act>`` (MXFP4 experts and MXFP4 activations: e2m1 elements with one UE8M0
scale per 32 elements on both operands, no global or per-token scale) for every typed activation of the layer, on SM100 /
SM103 / SM107 (the instruction class is the part's: the sm_107 K = 128 step on SM107, the K = 64 step elsewhere). The
activations come either as bf16 rows (``hidden_states_q`` bf16, no scale: the class quantizes them itself) or already
quantized (``hidden_states_q`` packed uint8 ``[T, H/2]``, ``hidden_states_scale`` the UE8M0 block scales -- the 128x4
swizzled buffer of ``mxfp4_quantize(..., sfLayout=layout_128x4, batch_invariant=True)`` or the linear ``[T, H/32]``
layout). The pair has no per-token scale (``supports_per_token_scale = False``: the layer refuses the configuration and
the pack scale), no static-intermediate class (``BatchInvariantFp4Config(intermediate="static")`` is refused). The three
routing input modes of the bf16 backend, the slot-order combine, the dispatcher's ranked records as tactics (any tactic gives
the same bits), and an unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`
instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from ..api import MoEActivationPack, MoEWeightPack, QuantFormat, RoutingInputMode
from ..runners import _validate_logits_inputs, _validate_prerouted_inputs
from .runner import routing_term
from .runner_nvfp4 import (
    MOE_NVFP4_ARCHS,
    _configs_from_tactic_nvfp4,
    _Fp4PairRunner,
    _no_scales,
)
from .spec_mxfp4 import CLASS, MX_BLOCK

#: The compute capabilities the classes are stated for (the dense 4-bit GEMM class's parts).
MOE_MXFP4_ARCHS: Tuple[Tuple[int, int], ...] = MOE_NVFP4_ARCHS
#: Positions of the packed inputs: the activation rows (bf16, or packed fp4), their scale factors (or a 1-element
#: placeholder), the expert ids, the routing weights, the four weight tensors and the output.
_X, _SF, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


class BatchInvariantMoeMxfp4Runner(_Fp4PairRunner):
    """The unified layer's runner for the batch-invariant MXFP4 x MXFP4 MoE classes."""

    CLASS = CLASS
    supported_quant_variants = ((QuantFormat.MXFP4, QuantFormat.MXFP4),)
    # the MX pair carries no per-token scale: the flag and a pack scale are refused by the layer's shared contract
    supports_per_token_scale = False

    def _check_support(self) -> None:
        self._check_experts()
        super()._check_support()
        from .api_mxfp4 import check_shape_mxfp4

        cfg = self.config
        # the configuration's class first (a rule of the pair, device-free), then the device
        if self._intermediate != "per_token":
            raise unsupported(
                CLASS,
                f"BatchInvariantFp4Config(intermediate={self._intermediate!r})",
                "the intermediate quantized per 32-element block by the class itself (the static class is the NVFP4 pair's)",
            )
        check_device(self.device, CLASS, MOE_MXFP4_ARCHS)
        try:
            a = self._activation()
        except ValueError as e:
            raise unsupported(
                CLASS,
                f"activation {cfg.activation!r}",
                "one of the layer's typed activations",
            ) from e
        experts, routing = cfg.experts, cfg.routing
        if experts.num_fused_shared_experts:
            raise unsupported(
                CLASS,
                "fused shared experts",
                "num_fused_shared_experts = 0 (a shared expert is one more routed slot)",
            )
        check_shape_mxfp4(
            None, experts.intermediate_size, routing.num_experts, routing.top_k, a.gated
        )
        self._method, self._grouped_other = routing_term(routing, CLASS)
        if not cfg.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    # ---- inputs ----------------------------------------------------------------------------------------------------

    def _activation_scales(
        self, act: MoEActivationPack, T: int, H: int
    ) -> torch.Tensor:
        """The pre-quantized pack's scale factors: the flat 128x4 swizzled buffer or the linear ``[T, H/32]`` bytes (the op's
        scatter re-tiles them itself)."""
        from .api_mxfp4 import mx_sf_numel

        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(
                CLASS,
                "packed fp4 activations without hidden_states_scale",
                "the UE8M0 block scales of the rows",
            )
        if sf.dtype == torch.float8_e8m0fnu:
            sf = sf.view(torch.uint8)
        if sf.dtype != torch.uint8:
            raise unsupported(
                CLASS,
                f"hidden_states_scale of dtype {sf.dtype}",
                "uint8 or float8_e8m0fnu scale bytes",
            )
        if act.per_token_scale is not None:
            raise unsupported(
                CLASS,
                "a per-token scale",
                "no per-token scale (the MX pair's scales are the block scales alone)",
            )
        if (
            sf.ndim == 2
            and tuple(sf.shape) == (T, H // MX_BLOCK)
            and sf.is_contiguous()
        ):
            return sf  # the linear layout: the op's scatter assembles the 128x4 atoms from the rows
        if sf.numel() >= mx_sf_numel(T, H) and (
            sf.is_contiguous() or (sf.ndim == 2 and sf.t().is_contiguous())
        ):
            return (sf if sf.is_contiguous() else sf.t()).reshape(-1)
        raise unsupported(
            CLASS,
            f"hidden_states_scale of shape {tuple(sf.shape)}",
            f"the 128x4 swizzled buffer of mxfp4_quantize(..., layout_128x4) ({mx_sf_numel(T, H)} bytes) or the linear [{T}, {H // MX_BLOCK}] layout",
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        from .api import route
        from .api_mxfp4 import MoEWeightsMXFP4, check_shape_mxfp4, route_plan_dual

        cfg = self.config
        plan: List[torch.Tensor] = []
        self._round_scales = False
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.uint8):
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "bf16 [num_tokens, hidden_size] rows, or packed fp4 uint8 [num_tokens, hidden_size / 2] with their scales",
            )
        x = x.contiguous()
        T = x.shape[0]
        H = x.shape[1] if x.dtype == torch.bfloat16 else 2 * x.shape[1]
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        view = weights.get_view(self.backend_key)
        if view.get("quant", "nvfp4") != "mxfp4":
            raise unsupported(
                CLASS,
                "weights prepared for another pair of the fp4 backend",
                "weights from BatchInvariantFp4Config.prepare_weights(..., quant=QuantConfig(weight=MXFP4, activation=MXFP4))",
            )
        w = MoEWeightsMXFP4.from_view(view)
        if (
            w.hidden_size != H
            or w.num_experts != E
            or w.intermediate_size != cfg.experts.intermediate_size
        ):
            raise unsupported(
                CLASS,
                f"weights (E {w.num_experts}, I {w.intermediate_size}, H {w.hidden_size}) for hidden {H}, E {E}, I {cfg.experts.intermediate_size}",
                "weights prepared for this configuration",
            )
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"{type(cfg.activation).__name__} with weights prepared for the chain {w.act!r} ({'gated' if w.gated else 'ungated'})",
                "weights prepared with the layer's activation",
            )
        check_shape_mxfp4(H, w.intermediate_size, E, k, w.gated)
        if x.dtype == torch.bfloat16:
            if act.hidden_states_scale is not None or act.per_token_scale is not None:
                raise unsupported(
                    CLASS,
                    "scales with bf16 activations",
                    "no scale (the class quantizes bf16 rows itself)",
                )
            sf = _no_scales(x.device)[0]
        else:
            sf = self._activation_scales(act, T, H)
        mode = act.routing_input_mode
        if mode is RoutingInputMode.FromLogits:
            _validate_logits_inputs(act, T, E, type(self).__name__)
            if self._grouped_other:
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
            scale = self._scale_tensor() if self._method.scaled else None
            from .blocks.scheduler import MAX_E
            from .route_plan import ROUTE_PLAN_MAX_SLOTS, ROUTE_PLAN_MAX_TOKENS

            if (
                0 < T <= ROUTE_PLAN_MAX_TOKENS
                and T * k <= ROUTE_PLAN_MAX_SLOTS
                and E <= MAX_E
            ):
                # routing and the dense packing's dual plan in one launch (a placement: the same bits and integers)
                ids, wts, counts, dest, src, dest_pad, pos_pad = route_plan_dual(
                    act.routing_logits, k, self._method, bias, scale
                )
                plan = [counts, dest, src, dest_pad, pos_pad]
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
            self._round_scales = (
                mode is RoutingInputMode.PackedPrecomputed
                and wts.dtype != torch.bfloat16
            )
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        return [x, sf, ids, wts, w.w1, w.w1_sf, w.w2, w.w2_sf, out] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _scatter_launch(self, x: torch.Tensor) -> bool:
        return True  # this pair's quantizer has no scattering pass: every input's box form scatters in a launch of its own

    def _ranked(self, T: int, H: int, scatter_launch: bool = True) -> Tuple[list, list]:
        from .blocks.epilogue import MXFP4_GEMM2_CHAIN, Mxfp4Chain
        from .schedule import active_experts, rows_per_expert
        from .schedule_nvfp4 import TOP_K_FP4, gather_form_rule
        from .schedule_mxfp4 import rank_grouped_mxfp4
        from .spec_mxfp4 import expert_classes_mx

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in1, in2 = expert_classes_mx(family)
        chain1 = Mxfp4Chain("glu" if a.gated else "act", a.act, "bf16", a.params)
        n1 = 2 * I if a.gated else I
        r1 = list(
            rank_grouped_mxfp4(
                rows, n1, H, E, chain1, in1, sm_count, family, TOP_K_FP4, n_active=n_act
            )
        )
        twins = [gather_form_rule(c, scatter_launch) for c in r1]
        r1 += [t for t in twins if t.gather_a and t not in r1]
        r2 = rank_grouped_mxfp4(
            rows,
            H,
            I,
            E,
            MXFP4_GEMM2_CHAIN,
            in2,
            sm_count,
            family,
            TOP_K_FP4,
            n_active=n_act,
        )
        return r1, list(r2)

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + ("moe.mxfp4-v2",)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_mxfp4 import MoEWeightsMXFP4, fused_moe_mxfp4

        x, sf, ids, wts, w1, w1sf, w2, w2sf, out = inputs[:9]
        plan = tuple(inputs[9:14]) if len(inputs) >= 14 else None
        H = self._hidden_of(x)
        if tactic == -1 or tactic is None:
            from .schedule_nvfp4 import gather_form_rule

            r1, r2 = self._ranked(int(x.shape[0]), H, True)
            c1, c2 = gather_form_rule(r1[0], True), r2[0]
        else:
            c1, c2 = _configs_from_tactic_nvfp4(tuple(tactic))
        a = self._activation()
        w = MoEWeightsMXFP4(
            w1,
            w1sf,
            w2,
            w2sf,
            H,
            self.config.experts.intermediate_size,
            a.gated,
            a.act,
            a.params,
        )
        # a pre-quantized pack: the 2-D scale tensor is the linear layout (the op's own placement: no re-tiling launch)
        xin = x if x.dtype == torch.bfloat16 else (x, sf)
        fused_moe_mxfp4(
            xin,
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            plan=plan,
            round_scales=self._round_scales,
        )
        return out
