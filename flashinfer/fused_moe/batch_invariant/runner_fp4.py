# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` -> ``BatchInvariantFp4Runner``).

One runner for the 4-bit-weight quantization pairs, grouped as ``TrtllmFp4Config`` groups them: ``QuantConfig(weight=...,
activation=...)`` selects the pair (:data:`PAIRS`: NVFP4 x NVFP4, the W4A4 classes ``moe.nvfp4.<act>`` and the static class
``moe.nvfp4.<act>.static``; MXFP4 x MXFP4, the classes ``moe.mxfp4.<act>``), the op is :func:`.api_fp4.fused_moe_fp4` on the
pair. The weight view and the autotuner records live under the backend's key; a pack prepared for the other pair is refused.
MXFP4 x MXFP8 arrives with the mixed MoE PR. A pair outside ``supported_quant_variants`` is skipped by ``MoELayer``'s
candidate filter (``supports_quant``) and refused by ``check_support`` and by the config's ``prepare_weights`` /
``prepare_activations`` with :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`, naming the served pairs.

The activations come as bf16 rows (``hidden_states_q`` bf16, no scale: the pair's class quantizes them itself) or already
quantized (``hidden_states_q`` packed uint8 ``[T, H/2]``, ``hidden_states_scale`` the block scales -- the 128x4 swizzled
buffer of the mode's quantizer or the linear ``[T, H/block]`` layout; NVFP4 also ``per_token_scale [T]`` f32 under
``QuantConfig(per_token_scale=True)``, unit scales when absent: the per-tensor pack of ``TrtllmFp4Config``; the MX pair has
no per-token scale and refuses the flag). The three routing input modes of the bf16 backend, the slot-order combine (never an
atomic finalize), the dispatcher's ranked records as tactics (any tactic gives the same bits); an unsupported configuration
raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import (
    BatchInvariantUnsupportedError,
    check_device,
    unsupported,
)
from ..api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEWeightPack,
    PowLU,
    QuantConfig,
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
from .api_fp4 import MXFP4, NVFP4, Fp4Pair
from .runner import MOE_BF16_ARCHS, _PackedInputs, routing_term

#: The backend's key: the name of its weight view and of its autotuner records.
BACKEND_KEY = "batch_invariant_fp4"
#: The compute capabilities the classes are stated for (the dense 4-bit GEMM class's parts).
MOE_FP4_ARCHS: Tuple[Tuple[int, int], ...] = MOE_BF16_ARCHS
#: The quantization pairs the backend serves, in the order they are advertised.
PAIRS: Dict[Tuple[QuantFormat, QuantFormat], Fp4Pair] = {
    (QuantFormat.NVFP4, QuantFormat.NVFP4): NVFP4,
    (QuantFormat.MXFP4, QuantFormat.MXFP4): MXFP4,
}
#: The pairs of the group that a later PR of the series brings (named by the refusal).
PENDING_PAIRS = "MXFP4 x MXFP8 arrives with the mixed MoE PR"
#: Positions of the packed inputs: the activation rows (bf16, or packed fp4), their scale factors and per-token scales (or
#: placeholders), the expert ids, the routing weights, the six weight tensors (the MX pair's alphas are placeholders), the
#: output, the static class's per-expert scales (placeholders otherwise); the dense plan follows when the router made it.
_X, _SF, _PTS, _IDS, _W, _W1, _W1SF, _W1A, _W2, _W2SF, _W2A, _OUT, _GS2, _INV2 = range(
    14
)
_PLAN_LEN = 5
TACTIC_FIELDS = 8  # scalars per record in the tactic tuple

#: The layer's typed activations, every one a class of either pair.
_ACTIVATIONS = (
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
_NO_SCALES: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def pair_name(pair: Tuple[QuantFormat, QuantFormat]) -> str:
    return f"{pair[0].name} x {pair[1].name}"


def unsupported_pair(quant: QuantConfig) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair the fp4 backend does not serve: it names the served pairs (and the pending ones)."""
    served = ", ".join(pair_name(p) for p in PAIRS)
    pending = f"; {PENDING_PAIRS}" if PENDING_PAIRS else ""
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the fp4 MoE backend `{BACKEND_KEY}` (BatchInvariantFp4Config) does not serve the "
        f"quantization weight={quant.weight.name}, activation={quant.activation.name}; it serves {served}{pending}. "
        "The batch-invariant mode never falls back to another kernel; call without batch_invariant=True to use the "
        "regular backends."
    )


def _no_scales(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The placeholder scale operands (never read), one pair per device so a call launches no fill: a 1-byte scale buffer
    and an empty f32 vector. The empty vector also marks a pre-quantized NVFP4 pack without per-token scales (the per-tensor
    pack of the TRTLLM backends: unit scales, nothing to read or fill) and stands for the MX pair's absent alphas."""
    key = str(device)
    t = _NO_SCALES.get(key)
    if t is None:
        t = _NO_SCALES[key] = (
            torch.empty(1, dtype=torch.uint8, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    return t


def _tactic_fp4(c1, c2) -> tuple:
    """A pair of records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache): the bf16 runner's six
    per record plus the swapped-record and the gather-form flags."""
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


def _configs_from_tactic_fp4(t: tuple):
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


class BatchInvariantFp4Runner(MoERunner):
    """The unified layer's runner for the batch-invariant fp4 MoE classes: the pair from the configuration's ``QuantConfig``,
    the routed scale constant and the part at build, the intermediate's class from the backend config."""

    backend_key = BACKEND_KEY
    supported_quant_variants = tuple(PAIRS)
    supported_output_formats = (QuantFormat.BF16,)
    supported_routing_modes = (
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_activation_classes = _ACTIVATIONS
    supported_activation_classes_by_quant = {pair: _ACTIVATIONS for pair in PAIRS}
    # the NVFP4 pair consumes a pre-quantized pack's per-token scale (the class's own ``pts``) under the layer's shared
    # contract; the MX pair has none and refuses the flag at support
    supports_per_token_scale = True
    supports_expert_parallelism = False

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        # an unserved pair is refused by check_support (the layer's filter never constructs this case)
        self._pair: Optional[Fp4Pair] = PAIRS.get(config.quant.pair)
        # exact shapes: the dispatcher's pick is a function of the token count (every tactic computes the same bits)
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        self._part: Optional[Tuple[int, str]] = None
        self._scale: Optional[torch.Tensor] = None
        # the intermediate's class (`per_token` / `static`): the bound backend config's field
        cands = [c for c in config.backend if hasattr(c, "intermediate")]
        self._intermediate: str = cands[0].intermediate if cands else "per_token"

    @property
    def pair(self) -> Fp4Pair:
        """The configured quantization pair."""
        if self._pair is None:
            raise unsupported_pair(self.config.quant)
        return self._pair

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
                self.pair.cls if self._pair is not None else "moe.nvfp4",
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the class's EP form is not part of this backend)",
            )

    def _check_support(self) -> None:
        if self._pair is None:
            raise unsupported_pair(self.config.quant)
        # the class's shard refusal first, then the layer's generic rules, then the pair's
        self._check_experts()
        super()._check_support()
        cfg, pair, cls = self.config, self.pair, self.pair.cls
        if not pair.scaled:
            # the MX pair carries no per-token scale and no static-intermediate class (both are the NVFP4 pair's)
            if cfg.quant.per_token_scale:
                raise unsupported(
                    cls,
                    "QuantConfig(per_token_scale=True)",
                    "no per-token scale (the MX pair's scales are the block scales alone)",
                )
            if self._intermediate != "per_token":
                raise unsupported(
                    cls,
                    f"BatchInvariantFp4Config(intermediate={self._intermediate!r})",
                    "the intermediate quantized per 32-element block by the class itself (the static class is the NVFP4 pair's)",
                )
        check_device(self.device, cls, MOE_FP4_ARCHS)
        try:
            a = self._activation()
        except ValueError as e:
            raise unsupported(
                cls,
                f"activation {cfg.activation!r}",
                "one of the layer's typed activations",
            ) from e
        experts, routing = cfg.experts, cfg.routing
        if experts.num_fused_shared_experts:
            raise unsupported(
                cls,
                "fused shared experts",
                "num_fused_shared_experts = 0 (a shared expert is one more routed slot)",
            )
        # the shape rule on the dimensions the config fixes (H enters at the first call, with the full rule)
        pair.check_shape(
            None, experts.intermediate_size, routing.num_experts, routing.top_k, a.gated
        )
        # a routing term the from-logits mode has no class for (Llama4 beyond top-1, a group shape outside the rule) is
        # refused at a from-logits call, not here: the pre-routed modes never consult it (the bf16 runner's rule)
        self._logits_refusal: Optional[BatchInvariantUnsupportedError] = None
        try:
            self._method, self._grouped_other = routing_term(routing, cls)
        except BatchInvariantUnsupportedError as e:
            self._method, self._grouped_other, self._logits_refusal = None, False, e
        if not cfg.finalize.do_finalize:
            raise unsupported(
                cls,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
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
            # the routed scaling factor is a constant of the configuration: built here, at the layer's construction, so no
            # forward (a first one under CUDA graph capture included) copies it to the device
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
    def accepts(self, act: MoEActivationPack, weights: MoEWeightPack) -> bool:
        """The layer's candidate filter: the shared rule (a view under the backend's key, the input geometry) and the view's
        intermediate class being this candidate's -- the two fp4 candidates of a layer share the view key, so a view prepared
        for the other class is the other candidate's pack and is declined here (the runner driven directly refuses it in
        ``pack_inputs``); a view of another pair stays a refusal, since no candidate of the layer serves it."""
        if not super().accepts(act, weights):
            return False
        view = weights.get_view(self.backend_key)
        return str(view.get("intermediate", "per_token")) == self._intermediate

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

    def _activation_scales(
        self, act: MoEActivationPack, T: int, H: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The pre-quantized pack's scale factors -- the 128x4 swizzled buffer of the mode's quantizer, returned flat, or the
        linear ``[T, H/block]`` bytes, returned 2-D (the op's scatter re-tiles them itself): the rank names the layout to
        ``forward`` -- and, for the NVFP4 pair, its per-token scales (the layer's shared contract admits them under
        ``QuantConfig(per_token_scale=True)``; the empty placeholder for a pack without them: unit scales, nothing to fill).
        The layout is ``QuantConfig(swizzled_scale_factors)``'s when set (``True``: the swizzled buffer, 2-D as the quantizer
        returns it or flattened; ``False``: the linear layout); unset, the shape names it where it can, and a 2-D
        ``[T, H/block]`` tensor at a row count that is a multiple of 128 (the quantizer's buffer has that shape too) is
        refused (:func:`~.api_fp4.gather_sf_layout`)."""
        from .api_fp4 import gather_sf_layout

        pair, cls = self.pair, self.pair.cls
        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(
                cls,
                "packed fp4 activations without hidden_states_scale",
                "the block scales of the rows",
            )
        sf_dtype = torch.float8_e4m3fn if pair.scaled else torch.float8_e8m0fnu
        if sf.dtype == sf_dtype:
            sf = sf.view(torch.uint8)
        if sf.dtype != torch.uint8:
            raise unsupported(
                cls,
                f"hidden_states_scale of dtype {sf.dtype}",
                f"uint8 or {sf_dtype} scale bytes",
            )
        flag = self.config.quant.swizzled_scale_factors
        named = None if flag is None else ("128x4" if flag else "linear")
        layout = gather_sf_layout(sf, T, H, pair.block, named, cls)
        if layout == "linear":
            if sf.ndim != 2 or tuple(sf.shape) != (T, H // pair.block):
                raise unsupported(
                    cls,
                    f"hidden_states_scale of shape {tuple(sf.shape)} under QuantConfig(swizzled_scale_factors=False)",
                    f"the linear [{T}, {H // pair.block}] layout",
                )
            sf_flat = (
                sf.contiguous()
            )  # the op's scatter assembles the 128x4 atoms from the rows
        elif sf.numel() >= pair.sf_numel(T, H) and (
            sf.is_contiguous() or (sf.ndim == 2 and sf.t().is_contiguous())
        ):
            sf_flat = (sf if sf.is_contiguous() else sf.t()).reshape(-1)
        else:
            raise unsupported(
                cls,
                f"hidden_states_scale of shape {tuple(sf.shape)}",
                f"the 128x4 swizzled buffer of the mode's quantizer ({pair.sf_numel(T, H)} bytes) or the linear "
                f"[{T}, {H // pair.block}] layout",
            )
        if sf_flat.data_ptr() % 16:
            raise unsupported(
                cls,
                "hidden_states_scale at a storage offset that is not a multiple of 16 bytes",
                "16-byte aligned scale bytes (the kernels' vector loads and TMA boxes)",
            )
        pts = act.per_token_scale
        if pts is None:
            pts = _no_scales(act.hidden_states_q.device)[
                1
            ]  # the empty marker: unit scales
        elif not pair.scaled:
            raise unsupported(
                cls,
                "a per-token scale",
                "no per-token scale (the MX pair's scales are the block scales alone)",
            )
        else:
            if tuple(pts.shape) != (T,):
                raise unsupported(
                    cls,
                    f"per_token_scale of shape {tuple(pts.shape)}",
                    f"one f32 scale per row [{T}]",
                )
            pts = pts.to(torch.float32).contiguous()
        return sf_flat, pts

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        from .api import route
        from .api_fp4 import MoEWeightsFp4, route_plan_dual

        cfg, pair, cls = self.config, self.pair, self.pair.cls
        plan: List[torch.Tensor] = []
        w_round = False  # bound to the pack (the bf16 runner's rule): a later pack never changes an earlier one's rounding
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.uint8):
            raise unsupported(
                cls,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "bf16 [num_tokens, hidden_size] rows, or packed fp4 uint8 [num_tokens, hidden_size / 2] with their scales",
            )
        x = x.contiguous()
        if x.data_ptr() % 16:
            raise unsupported(
                cls,
                "hidden_states_q at a storage offset that is not a multiple of 16 bytes",
                "16-byte aligned rows (the kernels' vector loads and TMA boxes)",
            )
        T = x.shape[0]
        H = self._hidden_of(x)
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        view = weights.get_view(self.backend_key)
        if view.get("quant", "nvfp4") != pair.name:
            raise unsupported(
                cls,
                "weights prepared for another pair of the fp4 backend",
                f"weights from BatchInvariantFp4Config.prepare_weights(..., quant=QuantConfig(weight={pair.name.upper()}, "
                f"activation={pair.name.upper()}))",
            )
        w = MoEWeightsFp4.from_view(view)
        if (
            w.hidden_size != H
            or w.num_experts != E
            or w.intermediate_size != cfg.experts.intermediate_size
        ):
            raise unsupported(
                cls,
                f"weights (E {w.num_experts}, I {w.intermediate_size}, H {w.hidden_size}) for hidden {H}, E {E}, I {cfg.experts.intermediate_size}",
                "weights prepared for this configuration",
            )
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                cls,
                f"{type(cfg.activation).__name__} with weights prepared for the chain {w.act!r} ({'gated' if w.gated else 'ungated'})",
                "weights prepared with the layer's activation",
            )
        if w.intermediate != self._intermediate:
            raise unsupported(
                cls,
                f"intermediate={self._intermediate!r} with weights prepared for intermediate={w.intermediate!r}",
                "weights prepared with the backend config's intermediate class",
            )
        pair.check_shape(H, w.intermediate_size, E, k, w.gated)
        if x.dtype == torch.bfloat16:
            if act.hidden_states_scale is not None or act.per_token_scale is not None:
                raise unsupported(
                    cls,
                    "scales with bf16 activations",
                    "no scale (the class quantizes bf16 rows itself)",
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
                    cls,
                    f"expert groups (n_group / topk_group) with {cfg.routing.method!r} from logits",
                    "the groups with DeepSeekV3, or pre-routed ids",
                )
            if self._method is None:
                # the refusal deferred at construction (the pre-routed modes never consulted it), else no term
                refusal = getattr(self, "_logits_refusal", None)
                if refusal is not None:
                    raise refusal
                raise unsupported(
                    cls,
                    f"routing method {cfg.routing.method!r} from logits",
                    "Default, Renormalize, RenormalizeNaive, Sigmoid, SigmoidRenorm, TopK, Llama4, TopKSigmoid or DeepSeekV3",
                )
            bias = act.routing_bias
            if (bias is not None) != self._method.bias:
                raise unsupported(
                    cls,
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
            w_round = (
                mode is RoutingInputMode.PackedPrecomputed
                and wts.dtype != torch.bfloat16
            )
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
        none = _no_scales(x.device)[1]

        def opt(t):
            return t if t is not None else none

        packed = _PackedInputs([
            x, sf, pts, ids, wts,
            w.w1, w.w1_sf, opt(w.w1_alpha), w.w2, w.w2_sf, opt(w.w2_alpha),
            out, opt(w.gs2), opt(w.inv2),
        ] + plan)  # fmt: skip
        packed.w_round = w_round
        return packed

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _scatter_launch(self, x: torch.Tensor) -> bool:
        """Whether the box form scatters the quantized rows in a launch of its own: a pre-quantized input, and every input of
        the MX pair (whose quantizer stores the class's layout); a bf16 NVFP4 input's quantizer scatters in its own pass."""
        return x.dtype != torch.bfloat16 or not self.pair.scaled

    def _ranked(
        self, T: int, H: int, scatter_launch: bool = False
    ) -> Tuple[list, list]:
        """The ranked records of both GEMMs (the tactic space's head); with ``scatter_launch`` GEMM1's list carries the
        gather-form twin the structural rule places (:func:`~.schedule_fp4.gather_form_rule`)."""
        from .api_fp4 import MoEWeightsFp4
        from .schedule_fp4 import (
            TOP_K_FP4,
            active_experts,
            gather_form_rule,
            rank_grouped_fp4,
            rows_per_expert,
        )

        cfg, pair = self.config, self.pair
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in1, in2 = pair.classes(H, I, family, a.gated)
        # the chains of the class (the weights' terms without the tensors)
        w = MoEWeightsFp4(
            pair,
            None,
            None,
            None,
            None,
            H,
            I,
            a.gated,
            a.act,
            a.params,
            intermediate=self._intermediate,
        )  # type: ignore[arg-type]
        chain1, chain2 = w.chain1, w.chain2
        n1 = 2 * I if a.gated else I
        r1 = list(
            rank_grouped_fp4(
                rows, n1, H, E, chain1, in1, sm_count, family, TOP_K_FP4, n_active=n_act
            )
        )
        twins = [gather_form_rule(c, scatter_launch) for c in r1]
        r1 += [t for t in twins if t.gather_a and t not in r1]
        r2 = rank_grouped_fp4(
            rows, H, I, E, chain2, in2, sm_count, family, TOP_K_FP4, n_active=n_act
        )
        return r1, list(r2)

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        x = inputs[_X]
        r1, r2 = self._ranked(
            int(x.shape[0]), self._hidden_of(x), self._scatter_launch(x)
        )
        return [_tactic_fp4(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            "moe.fp4-v1",
            self.pair.name,
            self._intermediate,
        )

    # ---- launch ----------------------------------------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._require_built()
        from .api_fp4 import MoEWeightsFp4, fused_moe_fp4

        pair = self.pair
        x, sf, pts, ids, wts, w1, w1sf, w1a, w2, w2sf, w2a, out, gs2, inv2 = inputs[:14]
        plan = (
            tuple(inputs[14 : 14 + _PLAN_LEN])
            if len(inputs) >= 14 + _PLAN_LEN
            else None
        )
        H = self._hidden_of(x)
        if tactic == -1 or tactic is None:
            from .schedule_fp4 import gather_form_rule

            r1, r2 = self._ranked(int(x.shape[0]), H, self._scatter_launch(x))
            c1, c2 = gather_form_rule(r1[0], self._scatter_launch(x)), r2[0]
        else:
            c1, c2 = _configs_from_tactic_fp4(tuple(tactic))
        a = self._activation()
        static = pair.scaled and self._intermediate == "static"
        w = MoEWeightsFp4(
            pair, w1, w1sf, w2, w2sf, H, self.config.experts.intermediate_size, a.gated, a.act, a.params,
            w1a if pair.scaled else None, w2a if pair.scaled else None,
            self._intermediate, gs2 if static else None, inv2 if static else None,
        )  # fmt: skip
        # a pre-quantized pack: the empty per-token placeholder means unit scales (the per-tensor pack); the scale tensor's
        # rank names its layout (flat: the 128x4 buffer, 2-D: the linear rows, as pack_inputs shaped it); both are the op's
        # own placements (no fill, no re-tiling launch)
        if x.dtype == torch.bfloat16:
            xin: Any = x
        elif pair.scaled:
            xin = (x, sf, pts if pts.numel() == x.shape[0] else None)
        else:
            xin = (x, sf)
        fused_moe_fp4(
            xin,
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            plan=plan,
            round_scales=getattr(inputs, "w_round", False),
            sf_layout=None
            if x.dtype == torch.bfloat16
            else ("linear" if sf.ndim == 2 else "128x4"),
        )
        return out
