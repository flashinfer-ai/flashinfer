# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 x NVFP4 pair of the batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` with
``QuantConfig(weight=NVFP4, activation=NVFP4)`` -> ``BatchInvariantFp4Runner`` -> ``BatchInvariantMoeNvfp4Runner``, this
module's implementation runner; :mod:`.runner_fp4` holds the backend).

The runner serves the W4A4 classes ``moe.nvfp4.<act>`` (``QuantConfig(weight=NVFP4, activation=NVFP4)``: NVFP4 experts with
one per-tensor global scale per expert and GEMM, per-token NVFP4 activations) for every typed activation of the layer, on
SM100 / SM103 / SM107 (the instruction class per expert weight shape by the dense 4-bit class rule: the sm_107 K = 128 step
on SM107, the K = 64 step elsewhere). The activations come either as bf16 rows (``hidden_states_q`` bf16, no scale: the
class quantizes them per token itself) or already quantized per token (``hidden_states_q`` packed uint8 ``[T, H/2]``,
``hidden_states_scale`` the e4m3 block scales -- the 128x4 swizzled buffer of ``nvfp4_quantize(...,
sfLayout=layout_128x4, per_token_activation=True, batch_invariant=True)`` or the linear ``[T, H/16]`` layout -- and
``per_token_scale [T]`` f32; without a per-token scale the rows are taken at unit scale, i.e. the per-tensor pack of
``TrtllmFp4Config.prepare_activations``). The three routing input modes of the bf16 backend, the slot-order combine
(never an atomic finalize), the dispatcher's ranked records as tactics (any tactic gives the same bits), and an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import check_device, unsupported

from ..api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEWeightPack,
    PowLU,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from ..runners import MoERunner, _validate_logits_inputs, _validate_prerouted_inputs
from .runner import MOE_BF16_ARCHS, routing_term
from .spec_nvfp4 import CLASS

#: The compute capabilities the classes are stated for (the dense 4-bit GEMM class's parts).
MOE_NVFP4_ARCHS: Tuple[Tuple[int, int], ...] = MOE_BF16_ARCHS
#: Positions of the packed inputs: the activation rows (bf16, or packed fp4), their scale factors and per-token scales (or
#: 1-element placeholders), the expert ids, the routing weights, the six weight tensors and the output.
_X, _SF, _PTS, _IDS, _W, _W1, _W1SF, _W1A, _W2, _W2SF, _W2A, _OUT = range(12)


_NO_SCALES: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _no_scales(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The placeholder scale operands of a bf16 activation (never read: the class quantizes the rows itself), one pair
    per device so a call launches no fill. The empty per-token placeholder also marks a pre-quantized pack without per-token
    scales (the per-tensor pack of the TRTLLM backends): the op takes its rows at unit scale with nothing to read or fill."""
    key = str(device)
    t = _NO_SCALES.get(key)
    if t is None:
        t = _NO_SCALES[key] = (
            torch.empty(1, dtype=torch.uint8, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    return t


def _tactic_nvfp4(c1, c2) -> tuple:
    """A pair of records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache): the bf16 runner's six
    per record plus the swapped-record and the gather-form flags (the dispatcher's record spaces and forms)."""
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
            cfg.gather_a,
        )
    )


TACTIC_FIELDS = 8  # scalars per record in the tactic tuple


def _configs_from_tactic_nvfp4(t: tuple):
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
            gather_a=bool(v[7]),
        )

    return one(t[:TACTIC_FIELDS]), one(t[TACTIC_FIELDS : 2 * TACTIC_FIELDS])


class _Fp4PairRunner(MoERunner):
    """What the fp4 backend's pair implementations share: the backend's key, the routing modes, the output format, the layer's
    activation classes, the construction (the routed scale constant, the part, the intermediate's class field), the class's
    shard refusal and the shape helpers. ``CLASS`` names the pair's registry class for its refusals."""

    CLASS: str = CLASS
    backend_key = "batch_invariant_fp4"
    supported_routing_modes = (
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_output_formats = (QuantFormat.BF16,)
    supported_activation_classes = (
        SwiGLU,
        SwiGLUStep,
        GeGLU,
        GeGLUTanh,
        SiTU,
        PowLU,
        ReLU2,
        Identity,
        GELU,
        ReLU,
        SiLU,
    )
    supports_expert_parallelism = False

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self._round_scales = False
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        # exact shapes: the dispatcher's pick is a function of the token count (every tactic computes the same bits)
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        self._part: Optional[Tuple[int, str]] = None
        self._scale: Optional[torch.Tensor] = None
        # the intermediate's class (`per_token` / `static`): the bound backend config's field
        cands = [c for c in config.backend if hasattr(c, "intermediate")]
        self._intermediate: str = cands[0].intermediate if cands else "per_token"

    # ---- support ---------------------------------------------------------------------------------------------------
    def _activation(self):
        from .spec import activation_of

        return activation_of(self.config.activation)

    def _check_experts(self) -> None:
        """The class's rule on the expert range: the EP form is not part of this backend, so a shard is refused in the
        class's terms (before the layer's generic shard check)."""
        experts, routing = self.config.experts, self.config.routing
        if experts.local_expert_offset or (
            experts.local_num_experts not in (None, 0, routing.num_experts)
        ):
            raise unsupported(
                self.CLASS,
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the class's EP form is not part of this backend)",
            )

    def _build(self) -> None:
        from ...gemm.batch_invariant.api import part_of

        self._part = part_of(self.device)
        method = getattr(self, "_method", None)
        if (
            method is not None
            and method.scaled
            and not torch.cuda.is_current_stream_capturing()
        ):
            # the routed scaling factor is a constant of the configuration: its tensor is built here, at the layer's
            # construction, so no forward (a first one under CUDA graph capture included) copies it to the device
            self._scale = self._scale_constant()

    def _scale_constant(self) -> torch.Tensor:
        """The routed scaling factor as a one-element fp32 tensor on the runner's device: a blocking host copy outside CUDA
        graph capture (complete before any stream reads it), a device fill (a node of that capture alone) inside one."""
        v = self.config.routing.routed_scaling_factor
        v = 1.0 if v is None else float(v)
        if torch.cuda.is_current_stream_capturing():
            return torch.full((1,), v, dtype=torch.float32, device=self.device)
        return torch.tensor([v], dtype=torch.float32).to(self.device)

    def _scale_tensor(self) -> torch.Tensor:
        """The scale tensor the route kernel reads as given: the one built at construction; a runner built inside a capture
        takes a fill per call until it runs outside one, which then keeps the constant."""
        if self._scale is not None:
            return self._scale
        s = self._scale_constant()
        if not torch.cuda.is_current_stream_capturing():
            self._scale = s
        return s

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        if x.ndim != 2:
            return False
        if x.dtype == torch.bfloat16:
            return x.shape[1] % 32 == 0
        return x.dtype == torch.uint8 and (2 * x.shape[1]) % 32 == 0

    @staticmethod
    def _hidden_of(x: torch.Tensor) -> int:
        return int(x.shape[1]) if x.dtype == torch.bfloat16 else 2 * int(x.shape[1])

    def _ranked(
        self, T: int, H: int, scatter_launch: bool = False
    ) -> Tuple[list, list]:
        """The dispatcher's ranked records of the two GEMMs at ``T`` tokens (the pair's); with ``scatter_launch`` GEMM1's
        list carries the structural rule's gather twin (:func:`~.schedule_nvfp4.gather_form_rule`)."""
        raise NotImplementedError

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        x = inputs[_X]
        r1, r2 = self._ranked(
            int(x.shape[0]), self._hidden_of(x), self._scatter_launch(x)
        )
        return [_tactic_nvfp4(c1, c2) for c1 in r1 for c2 in r2]

    def _scatter_launch(self, x: torch.Tensor) -> bool:
        """Whether the box form scatters the quantized rows in a launch of its own: a pre-quantized input (the bf16 input's
        quantizer scatters in its own pass)."""
        return x.dtype != torch.bfloat16


class BatchInvariantMoeNvfp4Runner(_Fp4PairRunner):
    """The unified layer's runner for the batch-invariant NVFP4 (W4A4) MoE classes."""

    supported_quant_variants = ((QuantFormat.NVFP4, QuantFormat.NVFP4),)
    supports_per_token_scale = True

    def _check_support(self) -> None:
        self._check_experts()
        super()._check_support()
        from .api_nvfp4 import check_shape_nvfp4

        cfg = self.config
        check_device(self.device, CLASS, MOE_NVFP4_ARCHS)
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
        # the shape rule on the dimensions the config fixes (H enters at the first call, with the full rule)
        check_shape_nvfp4(
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
    def _validate_pack_contract(self, act: MoEActivationPack) -> None:
        """The routing mode must be one of the three; a pack's ``per_token_scale`` is consumed in every configuration (the
        class is per-token by construction: no tactic depends on the scale's presence), so ``QuantConfig.per_token_scale``
        gates nothing here."""
        if act.routing_input_mode not in self.supported_routing_modes:
            names = ", ".join(mode.name for mode in self.supported_routing_modes)
            raise NotImplementedError(
                f"{type(self).__name__} does not support routing_input_mode={act.routing_input_mode!r}; supported: {names}."
            )

    def _activation_scales(
        self, act: MoEActivationPack, T: int, H: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The pre-quantized pack's scale factors (the flat 128x4 swizzled buffer, or the linear ``[T, H/16]`` bytes of the
        per-tensor pack of the TRTLLM backends, which the op's scatter re-tiles itself) and its per-token scales (``None`` for a
        pack without them: the op takes the rows at unit scale with nothing to fill)."""
        from ...batch_invariant.quantization.nvfp4_schedule import BLOCK, sf_numel

        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(
                CLASS,
                "packed fp4 activations without hidden_states_scale",
                "the e4m3 block scales of the rows",
            )
        if sf.dtype == torch.float8_e4m3fn:
            sf = sf.view(torch.uint8)
        if sf.dtype != torch.uint8:
            raise unsupported(
                CLASS,
                f"hidden_states_scale of dtype {sf.dtype}",
                "uint8 or float8_e4m3fn scale bytes",
            )
        if sf.ndim == 2 and tuple(sf.shape) == (T, H // BLOCK) and sf.is_contiguous():
            sf_flat = sf  # the linear layout: the op's scatter assembles the 128x4 atoms from the rows
        elif sf.numel() >= sf_numel(T, H) and (
            sf.is_contiguous() or (sf.ndim == 2 and sf.t().is_contiguous())
        ):
            sf_flat = (sf if sf.is_contiguous() else sf.t()).reshape(-1)
        else:
            raise unsupported(
                CLASS,
                f"hidden_states_scale of shape {tuple(sf.shape)}",
                f"the 128x4 swizzled buffer of nvfp4_quantize(..., layout_128x4) ({sf_numel(T, H)} bytes) or the linear [{T}, {H // BLOCK}] layout",
            )
        pts = act.per_token_scale
        if pts is None:
            pts = _no_scales(act.hidden_states_q.device)[
                1
            ]  # the empty marker: unit scales
        else:
            if tuple(pts.shape) != (T,):
                raise unsupported(
                    CLASS,
                    f"per_token_scale of shape {tuple(pts.shape)}",
                    f"one f32 scale per row [{T}]",
                )
            pts = pts.to(torch.float32).contiguous()
        return sf_flat, pts

    # the fp4 backend's key (``runner_fp4.BACKEND_KEY``): the pair implementations share the weight view and the records
    # the per-token scale of a pre-quantized pack is consumed (the class's own ``pts``)
    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        from .api import route
        from .api_nvfp4 import route_plan_dual
        from .api_nvfp4 import MoEWeightsNVFP4, check_shape_nvfp4

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
        if view.get("quant", "nvfp4") != "nvfp4":
            raise unsupported(
                CLASS,
                "weights prepared for another pair of the fp4 backend",
                "weights from BatchInvariantFp4Config.prepare_weights(..., quant=QuantConfig(weight=NVFP4, activation=NVFP4))",
            )
        w = MoEWeightsNVFP4.from_view(view)
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
        if w.intermediate != self._intermediate:
            raise unsupported(
                CLASS,
                f"intermediate={self._intermediate!r} with weights prepared for intermediate={w.intermediate!r}",
                "weights prepared with the backend config's intermediate class",
            )
        check_shape_nvfp4(H, w.intermediate_size, E, k, w.gated)
        if x.dtype == torch.bfloat16:
            if act.hidden_states_scale is not None or act.per_token_scale is not None:
                raise unsupported(
                    CLASS,
                    "scales with bf16 activations",
                    "no scale (the class quantizes bf16 rows per token itself)",
                )
            sf, pts = _no_scales(x.device)
        else:
            sf, pts = self._activation_scales(act, T, H)
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
            # the packed word carries the weight as bf16: the rounded weights are the inputs -- f32 weights are rounded by
            # the combine itself (no host passes over the slots), bf16 weights widen exactly
            self._round_scales = (
                mode is RoutingInputMode.PackedPrecomputed
                and wts.dtype != torch.bfloat16
            )
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        return [
            x,
            sf,
            pts,
            ids,
            wts,
            w.w1,
            w.w1_sf,
            w.w1_alpha,
            w.w2,
            w.w2_sf,
            w.w2_alpha,
            out,
            # the static class's per-expert scales (gs2 for GEMM1, inv2 for GEMM2); the empty placeholder for the per-token class
            w.gs2 if w.gs2 is not None else _no_scales(x.device)[1],
            w.inv2 if w.inv2 is not None else _no_scales(x.device)[1],
        ] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(
        self, T: int, H: int, scatter_launch: bool = False
    ) -> Tuple[list, list]:
        """The ranked records of both GEMMs (the tactic space's head); with ``scatter_launch`` GEMM1's list carries the
        gather-form twin the structural rule places (:func:`~.schedule_nvfp4.gather_form_rule`)."""
        from .api_nvfp4 import GEMM2_CHAIN, GEMM2_STATIC_CHAIN
        from .blocks.epilogue import W4A4Chain
        from .schedule_nvfp4 import (
            TOP_K_FP4,
            active_experts,
            gather_form_rule,
            rank_grouped_nvfp4,
            rows_per_expert,
        )
        from .spec_nvfp4 import expert_classes

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in1, in2 = expert_classes(H, I, family, a.gated)
        static = self._intermediate == "static"
        chain1 = W4A4Chain(
            "glu" if a.gated else "act",
            a.act,
            "nvfp4" if static else "bf16",
            a.params,
            "w4a4",
        )
        chain2 = GEMM2_STATIC_CHAIN if static else GEMM2_CHAIN
        n1 = 2 * I if a.gated else I
        r1 = list(
            rank_grouped_nvfp4(
                rows, n1, H, E, chain1, in1, sm_count, family, TOP_K_FP4, n_active=n_act
            )
        )
        twins = [gather_form_rule(c, scatter_launch) for c in r1]
        r1 += [t for t in twins if t.gather_a and t not in r1]
        r2 = rank_grouped_nvfp4(
            rows, H, I, E, chain2, in2, sm_count, family, TOP_K_FP4, n_active=n_act
        )
        return r1, list(r2)

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + ("moe.nvfp4-v4", self._intermediate)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_nvfp4 import MoEWeightsNVFP4, fused_moe_nvfp4

        x, sf, pts, ids, wts, w1, w1sf, w1a, w2, w2sf, w2a, out, gs2, inv2 = inputs[:14]
        plan = tuple(inputs[14:19]) if len(inputs) >= 19 else None
        H = self._hidden_of(x)
        if tactic == -1 or tactic is None:
            from .schedule_nvfp4 import gather_form_rule

            r1, r2 = self._ranked(int(x.shape[0]), H, self._scatter_launch(x))
            c1, c2 = gather_form_rule(r1[0], self._scatter_launch(x)), r2[0]
        else:
            c1, c2 = _configs_from_tactic_nvfp4(tuple(tactic))
        a = self._activation()
        static = self._intermediate == "static"
        w = MoEWeightsNVFP4(
            w1,
            w1sf,
            w1a,
            w2,
            w2sf,
            w2a,
            H,
            self.config.experts.intermediate_size,
            a.gated,
            a.act,
            a.params,
            self._intermediate,
            gs2 if static else None,
            inv2 if static else None,
        )
        # a pre-quantized pack: the empty per-token placeholder means unit scales (the per-tensor pack), the 2-D scale tensor the
        # linear layout; both are the op's own placements (no fill, no re-tiling launch)
        xin = (
            x
            if x.dtype == torch.bfloat16
            else (x, sf, pts if pts.numel() == x.shape[0] else None)
        )
        fused_moe_nvfp4(
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
