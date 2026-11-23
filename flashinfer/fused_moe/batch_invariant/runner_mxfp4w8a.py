# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 x MXFP8 pair of the batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` with
``QuantConfig(weight=MXFP4, activation=MXFP8)`` -> ``BatchInvariantFp4Runner`` -> ``BatchInvariantMoeMxfp4w8aRunner``, this
module's implementation runner; :mod:`.runner_fp4` holds the backend).

The runner serves the classes ``moe.mxfp4w8a.<act>`` (MXFP4 experts, MXFP8 activations, every typed activation of the
layer) on SM100 / SM103 / SM107, through the dense mixed class of the part (the SM107 instruction folds 64 elements per
step, the SM100 / SM103 one 32 and reads the e2m1 weight through the TMA's unpacking format, which takes ``hidden_size``
and ``intermediate_size`` in multiples of 128 there: both dimensions of the specification). It takes the layer's canonical
MXFP8 activation pack (e4m3 ``hidden_states_q [T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as
``TrtllmFp4Config`` / ``TrtllmFp8BlockConfig`` / ``CutlassMxfp8Config`` / this config's ``prepare_activations`` produce
it) and every routing input mode: ``FromLogits`` runs the class's own routing kernel (with the permutation plan in the
same launch at small token counts, in the form of the packing the ranked records run on), ``PackedPrecomputed`` /
``UnpackedPrecomputed`` take the caller's expert ids and weights. The combine is the slot-order fp32 fold
(``use_fused_finalize`` is ignored, ``do_finalize=True`` only), the tactics are the ranked grouped-GEMM records of the
dense mixed heuristic over both record spaces -- the grouped body's and the swapped decode body's (the weights on the
MMA's M side, the group's tokens a narrow tile; `schedule_mxfp4w8a`) -- any tactic gives the same bits, and the packing of
the expanded rows follows the first picks (the dense packing when either is a swapped record, else the padded one: a
placement decided by the token count). An unsupported configuration raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ...batch_invariant.support import (
    BatchInvariantUnsupportedError,
    check_device,
    unsupported,
)
from ..api import MoEActivationPack, MoEWeightPack, QuantFormat, RoutingInputMode
from ..runners import _validate_logits_inputs, _validate_prerouted_inputs
from .runner import routing_term
from .runner_fp4 import MOE_FP4_ARCHS, _Fp4RunnerBase
from .spec_mxfp4w8a import BLOCK, CLASS


def _tactic_mx(c1, c2) -> tuple:
    """A pair of records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache): the bf16 runner's six
    per record plus the swapped-record flag and the record's k-block (the dispatcher's two record spaces; the swapped
    records of the sm_107 class carry their stage width)."""
    return tuple(
        int(v) if isinstance(v, bool) else v
        for cfg in (c1, c2)
        for v in (
            cfg.mma_tiler_mn[0],
            cfg.mma_tiler_mn[1],
            cfg.cluster_shape_mn[0],
            cfg.cluster_shape_mn[1],
            cfg.use_2cta_instrs,
            cfg.raster_along,
            cfg.swap_ab,
            cfg.k_block,
        )
    )


_TACTIC_FIELDS = 8


def _configs_from_tactic_mx(t: tuple):
    from ...gemm.batch_invariant.generator import Knobs

    def one(v):
        return Knobs(
            (int(v[0]), int(v[1])),
            (int(v[2]), int(v[3])),
            bool(v[4]),
            True,
            str(v[5]),
            1,
            swap_ab=bool(v[6]),
            k_block=int(v[7]),
        )

    return one(t[:_TACTIC_FIELDS]), one(t[_TACTIC_FIELDS : 2 * _TACTIC_FIELDS])


#: The compute capabilities the classes are stated for (the dense mixed GEMM class's parts).
MOE_MXFP4W8A_ARCHS: Tuple[Tuple[int, int], ...] = MOE_FP4_ARCHS
#: Positions of the packed inputs: the quantized activations and their scales, the expert ids, the routing weights, the
#: two weights with their scales and the output.
_XQ, _XSF, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


class BatchInvariantMoeMxfp4w8aRunner(_Fp4RunnerBase):
    """The unified layer's runner for the batch-invariant mixed MXFP4 x MXFP8 MoE classes."""

    CLASS = CLASS
    supported_quant_variants = ((QuantFormat.MXFP4, QuantFormat.MXFP8),)
    # the pair's scales are the two operands' block scales: no per-token scale (the flag and a pack scale are refused by the
    # layer's shared contract)
    supports_per_token_scale = False

    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        self._in_fmt: Optional[str] = None
        self._w_round = False

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        self._check_experts(CLASS)
        super()._check_support()
        from .api_mxfp4w8a import check_shape_mxfp4w8a, mx_class_of

        cfg = self.config
        # the configuration's class first (a rule of the pair, device-free), then the device
        if self._intermediate != "per_token":
            raise unsupported(
                CLASS,
                f"BatchInvariantFp4Config(intermediate={self._intermediate!r})",
                "the intermediate quantized per 32-element block by the class itself (the static class is the NVFP4 pair's)",
            )
        check_device(self.device, CLASS, MOE_MXFP4W8A_ARCHS)
        try:
            self._activation()
        except ValueError as e:
            raise unsupported(
                CLASS,
                f"activation {cfg.activation!r}",
                "one of the layer's typed activations",
            ) from e
        if cfg.quant.pair != (QuantFormat.MXFP4, QuantFormat.MXFP8):
            raise unsupported(
                CLASS,
                f"quant {cfg.quant!r}",
                "QuantConfig(weight=MXFP4, activation=MXFP8)",
            )
        experts, routing = cfg.experts, cfg.routing
        if experts.num_fused_shared_experts:
            raise unsupported(
                CLASS,
                "fused shared experts",
                "num_fused_shared_experts = 0 (a shared expert is one more routed slot)",
            )
        # the shape rule on the dimensions the config fixes, at the part's instruction class (H enters at the first call,
        # with the full rule)
        check_shape_mxfp4w8a(
            None,
            experts.intermediate_size,
            routing.num_experts,
            routing.top_k,
            mx_class_of(self.device),
        )
        # a routing term the from-logits mode has no class for (a group shape outside the rule, Llama4 with top_k > 1) is
        # refused at a from-logits call, not at construction: the pre-routed modes never consult the method
        self._logits_refusal: Optional[BatchInvariantUnsupportedError] = None
        try:
            self._method, self._grouped_other = routing_term(routing, CLASS)
        except BatchInvariantUnsupportedError as e:
            self._method, self._grouped_other, self._logits_refusal = None, False, e
        if not cfg.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    def _build(self) -> None:
        super()._build()
        from .api_mxfp4w8a import mx_class_of

        self._in_fmt = mx_class_of(self.device)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return (
            x.ndim == 2
            and x.dtype == torch.float8_e4m3fn
            and x.shape[1] % BLOCK == 0
            and act.hidden_states_scale is not None
        )

    @staticmethod
    def _hidden_of(x: torch.Tensor) -> int:
        return int(x.shape[1])

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        from .api import route, route_plan
        from .api_mxfp4w8a import (
            QUANT_TAG,
            MoEWeightsMXFP4W8A,
            check_shape_mxfp4w8a,
        )

        cfg = self.config
        self._w_round = False
        x = act.hidden_states_q
        if x.dtype != torch.float8_e4m3fn or x.ndim != 2:
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "the MXFP8 activation pack: an e4m3 [num_tokens, hidden_size] with its UE8M0 scales",
            )
        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(
                CLASS,
                "no hidden_states_scale",
                "the token-major UE8M0 scales [num_tokens, hidden_size / 32]",
            )
        if act.per_token_scale is not None:
            raise unsupported(
                CLASS,
                "a per-token scale",
                "no per-token scale (the pair's scales are the two operands' block scales alone)",
            )
        x = x.contiguous()
        T, H = x.shape
        if H % BLOCK:
            raise unsupported(
                CLASS,
                f"hidden size {H}",
                f"a positive multiple of {BLOCK} (whole UE8M0 scale blocks)",
            )
        if sf.dtype not in (torch.uint8, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            raise unsupported(
                CLASS,
                f"hidden_states_scale {sf.dtype}",
                "UE8M0 scale bytes (uint8, e8m0 or the e4m3-typed carrier of TrtllmFp4Config.prepare_activations)",
            )
        sf = (
            sf.view(torch.uint8).reshape(T, -1 if T else H // BLOCK).contiguous()
        )  # the zero-token pack has no inferable column count
        if sf.shape[1] != H // BLOCK:
            raise unsupported(
                CLASS,
                f"hidden_states_scale of {sf.shape[1]} columns",
                f"{H // BLOCK} token-major UE8M0 scales per row (the linear layout; a swizzled buffer is not this pack)",
            )
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        view = weights.get_view(self.backend_key)
        if view.get("quant", "nvfp4") != QUANT_TAG:
            raise unsupported(
                CLASS,
                "weights prepared for another pair of the fp4 backend",
                "weights from BatchInvariantFp4Config.prepare_weights(..., quant=QuantConfig(weight=MXFP4, activation=MXFP8))",
            )
        w = MoEWeightsMXFP4W8A.from_view(view)
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
        check_shape_mxfp4w8a(
            H, w.intermediate_size, E, k, self._in_fmt or "mxfp4w8a_k64"
        )
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"{type(cfg.activation).__name__} with {'gated' if w.gated else 'ungated'} weights of chain {w.act!r}",
                "weights prepared with the layer's activation",
            )
        plan: List[torch.Tensor] = []
        mode = act.routing_input_mode
        if mode is RoutingInputMode.FromLogits:
            _validate_logits_inputs(act, T, E, type(self).__name__)
            # the group-limited selection is a term of the DeepSeekV3 method; with another method the groups have no
            # meaning in this module (with pre-routed ids the group structure never enters the computation)
            if self._grouped_other:
                raise unsupported(
                    CLASS,
                    f"expert groups (n_group / topk_group) with {cfg.routing.method!r} from logits",
                    "the groups with DeepSeekV3, or pre-routed ids",
                )
            if self._method is None:
                if self._logits_refusal is not None:
                    raise self._logits_refusal  # deferred from the construction
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
                # routing and the permutation plan in one launch (a placement: the same bits and integers), in the form of the
                # packing the ranked records run on at this token count
                ids, wts, *plan = route_plan(
                    act.routing_logits,
                    k,
                    self._method,
                    bias,
                    scale,
                    **self._plan_form(T, H),
                )
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
            # the packed word carries the weight as bf16: the rounded weights are the class's inputs in this mode; f32
            # weights are rounded by the combine kernel itself (no host pass over the slots), bf16 ones widen exactly
            self._w_round = (
                mode is RoutingInputMode.PackedPrecomputed
                and wts.dtype != torch.bfloat16
            )
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        return [x, sf, ids, wts, w.w1, w.w1_sf, w.w2, w.w2_sf, out] + plan

    def _plan_form(self, T: int, H: int) -> dict:
        """The fused route + plan launch in the form of the packing the ranked records run on at this token count: the dual
        plan (the dense rows with the padded scale positions) or the padded plan (whole 128-row atoms per group)."""
        from .api_mxfp4w8a import GROUP_ALIGN

        if self._packing(T, H) == "dense":
            return dict(dual=True, pad_align=GROUP_ALIGN)
        return dict(align=GROUP_ALIGN)

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(
        self, T: int, H: int, scatter_launch: bool = False
    ) -> Tuple[list, list]:
        """The ranked records of both GEMMs over both record spaces at ``T`` tokens (`schedule_mxfp4w8a.ranked_mx`,
        memoised), trimmed to the packing the first picks run on: on the padded packing the swapped records cannot run, so
        they leave the lists. ``scatter_launch`` is the pair runner's gather-form switch: the MX chains read the token box
        (no gather twin), so the lists are the same either way."""
        from .schedule import TOP_K
        from .schedule_mxfp4w8a import packing_for, ranked_mx

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in_fmt = self._in_fmt or "mxfp4w8a"
        # both record spaces are ranked for the fused-requantization spelling GEMM1 runs (the same acceptance as the bf16
        # chain's on either body)
        chain1 = self._chain1(a)
        r1, r2 = ranked_mx(T, k, H, I, E, chain1, in_fmt, sm_count, family, TOP_K)
        if packing_for(r1[0], r2[0]) == "padded":
            r1 = [c for c in r1 if not c.swap_ab]
            r2 = [c for c in r2 if not c.swap_ab]
        return r1, r2

    def _chain1(self, a):
        """GEMM1's chain for the dispatcher: the fused requantization spelling of the layer's activation at the part's
        instruction class."""
        from .blocks.epilogue_mx import MxChain

        return MxChain(
            "glu" if a.gated else "act",
            a.act,
            "mxfp8",
            a.params,
            self._in_fmt or "mxfp4w8a",
        )

    def _packing(self, T: int, H: int) -> str:
        """The packing of the expanded rows at ``T`` tokens: the first picks' (`schedule_mxfp4w8a.packing_for`)."""
        from .schedule_mxfp4w8a import packing_for

        r1, r2 = self._ranked(T, H)
        return packing_for(r1[0], r2[0])

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        r1, r2 = self._ranked(int(inputs[_XQ].shape[0]), self._hidden_of(inputs[_XQ]))
        return [_tactic_mx(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + ("moe.mxfp4w8a-v2", self._in_fmt or "")

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_mxfp4w8a import MoEWeightsMXFP4W8A, fused_moe_mxfp4w8a

        xq, xsf, ids, wts, w1, w1_sf, w2, w2_sf, out = inputs[:9]
        plan = (
            tuple(inputs[9:]) or None
        )  # the plan's tensors, in the form the pack built them
        T, H = int(xq.shape[0]), int(xq.shape[1])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(T, H)
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic_mx(tuple(tactic))
        a = self._activation()
        w = MoEWeightsMXFP4W8A(w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params)
        # the packing follows the first picks at this token count (the plan, when given, was built in its form); a tactic
        # of the padded packing is a pair of grouped records, one of the dense packing any pair
        fused_moe_mxfp4w8a(
            (xq, xsf),
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            in_fmt=self._in_fmt,
            plan=plan,
            path=self._packing(T, H),
            w_round=self._w_round,
        )
        return out
