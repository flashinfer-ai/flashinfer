# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp4 MoE backend of the unified MoE layer (``BatchInvariantFp4Config`` -> ``BatchInvariantFp4Runner``).

One runner for the 4-bit-weight quantization pairs, grouped as ``TrtllmFp4Config`` groups them: ``QuantConfig(weight=...,
activation=...)`` selects the pair (:data:`PAIRS`: NVFP4 x NVFP4, the W4A4 classes ``moe.nvfp4.<act>`` and the static class
``moe.nvfp4.<act>.static``; MXFP4 x MXFP4, the classes ``moe.mxfp4.<act>``; MXFP4 x MXFP8, the mixed classes
``moe.mxfp4w8a.<act>``), the op is :func:`.api_fp4.fused_moe_fp4` on the fp4 pairs and :func:`.api_mxfp4w8a.fused_moe_mxfp4w8a`
on the mixed pair. The weight view and the autotuner records live under the backend's key; a pack prepared for another pair
is refused. A pair outside ``supported_quant_variants`` is skipped by ``MoELayer``'s candidate filter (``supports_quant``) and
refused by ``check_support`` and by the config's ``prepare_weights`` / ``prepare_activations`` with
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`, naming the served pairs.

The fp4 pairs' activations come as bf16 rows (``hidden_states_q`` bf16, no scale: the pair's class quantizes them itself) or
already quantized (``hidden_states_q`` packed uint8 ``[T, H/2]``, ``hidden_states_scale`` the block scales -- the 128x4 swizzled
buffer of the mode's quantizer or the linear ``[T, H/block]`` layout; NVFP4 also ``per_token_scale [T]`` f32 under
``QuantConfig(per_token_scale=True)``, unit scales when absent: the per-tensor pack of ``TrtllmFp4Config``; the MX pairs have
no per-token scale and refuse the flag). The mixed pair's activations are the canonical MXFP8 pack (e4m3 ``hidden_states_q
[T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as ``TrtllmFp4Config`` / ``TrtllmFp8BlockConfig`` /
this config's ``prepare_activations`` produce it), taken as given. The three routing input modes of the bf16 backend, the
slot-order combine (never an atomic finalize), the dispatcher's ranked records as tactics (any tactic gives the same bits); an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...autotuner import TuningConfig
from ...batch_invariant.support import (
    BatchInvariantUnsupportedError,
    LazyNames,
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
from ...gemm.batch_invariant.generator import Knobs
from ..runners import MoERunner, _validate_logits_inputs, _validate_prerouted_inputs
from . import spec as S
from .api_fp4 import MXFP4, MXFP4W8A, NVFP4, Fp4Pair
from .runner import MOE_BF16_ARCHS, _PackedInputs, pick_generation, routing_term


def _load_names() -> Dict[str, Any]:
    from .api import route, route_plan
    from .api_fp4 import (
        GROUP_ALIGN,
        TE_GS_INV,
        MoEWeightsFp4,
        _fused_moe_fp4_run,
        _gs_inv,
        gather_form_ok,
        gather_sf_layout,
    )
    from .api_mxfp8 import _fused_moe_mxfp8_run, check_activation_pack
    from .blocks.scheduler import MAX_E
    from .route_plan import ROUTE_PLAN_MAX_SLOTS, ROUTE_PLAN_MAX_TOKENS
    from .schedule import TOP_K
    from .schedule_fp4 import (
        TOP_K_FP4,
        active_experts,
        gather_form_rule,
        rank_grouped_fp4,
        rows_per_expert,
    )
    from .schedule_mxfp4w8a import packing_for, ranked_mx

    return dict(
        route=route,
        route_plan=route_plan,
        GROUP_ALIGN=GROUP_ALIGN,
        MoEWeightsFp4=MoEWeightsFp4,
        TE_GS_INV=TE_GS_INV,
        fused_moe_fp4_run=_fused_moe_fp4_run,
        gs_inv=_gs_inv,
        gather_form_ok=gather_form_ok,
        gather_sf_layout=gather_sf_layout,
        check_activation_pack=check_activation_pack,
        fused_moe_mxfp8_run=_fused_moe_mxfp8_run,
        MAX_E=MAX_E,
        ROUTE_PLAN_MAX_SLOTS=ROUTE_PLAN_MAX_SLOTS,
        ROUTE_PLAN_MAX_TOKENS=ROUTE_PLAN_MAX_TOKENS,
        TOP_K=TOP_K,
        TOP_K_FP4=TOP_K_FP4,
        active_experts=active_experts,
        gather_form_rule=gather_form_rule,
        rank_grouped_fp4=rank_grouped_fp4,
        rows_per_expert=rows_per_expert,
        packing_for=packing_for,
        ranked_mx=ranked_mx,
    )


_names = LazyNames(_load_names)

#: The backend's key: the name of its weight view and of its autotuner records.
BACKEND_KEY = "batch_invariant_fp4"
#: The compute capabilities the classes are stated for (the dense 4-bit GEMM classes' parts).
MOE_FP4_ARCHS: Tuple[Tuple[int, int], ...] = MOE_BF16_ARCHS
#: The quantization pairs the backend serves, in the order they are advertised.
PAIRS: Dict[Tuple[QuantFormat, QuantFormat], Fp4Pair] = {
    (QuantFormat.NVFP4, QuantFormat.NVFP4): NVFP4,
    (QuantFormat.MXFP4, QuantFormat.MXFP4): MXFP4,
    (QuantFormat.MXFP4, QuantFormat.MXFP8): MXFP4W8A,
}
#: Positions of the packed inputs: the activation rows (bf16, packed fp4, or the mixed pair's e4m3), their scale factors and
#: per-token scales (or placeholders), the expert ids, the routing weights, the six weight tensors (the MX pairs' alphas are
#: placeholders), the output, the static class's per-expert scales (placeholders otherwise); the plan follows when the router
#: made it (the dual form for the fp4 pairs; the mixed pair's in the form of its packing).
_X, _SF, _PTS, _IDS, _W, _W1, _W1SF, _W1A, _W2, _W2SF, _W2A, _OUT, _GS2, _INV2 = range(
    14
)
#: Plan tensors after the ids and weights in the packed inputs (``route_plan_dual``'s outputs, appended by construction in
#: ``pack_inputs``) and scalars per record in the tactic tuple (``_tactic_fp4``'s); both pinned to the builders by the suite.
_PLAN_LEN = 5
TACTIC_FIELDS = 9

#: The layer's typed activations, every one a class of each pair.
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
#: The placeholder scale operands per indexed device (the op's constant rule, :func:`~.api_fp4._const`).
_NO_SF: Dict[str, torch.Tensor] = {}
_NO_PTS: Dict[str, torch.Tensor] = {}


def pair_name(pair: Tuple[QuantFormat, QuantFormat]) -> str:
    return f"{pair[0].name} x {pair[1].name}"


def unsupported_pair(quant: QuantConfig) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair the fp4 backend does not serve: it names the served pairs."""
    served = ", ".join(pair_name(p) for p in PAIRS)
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the fp4 MoE backend `{BACKEND_KEY}` (BatchInvariantFp4Config) does not serve the "
        f"quantization weight={quant.weight.name}, activation={quant.activation.name}; it serves {served} and no other "
        "pair. The batch-invariant mode never falls back to another kernel; call without batch_invariant=True to use the "
        "regular backends."
    )


def _no_scales(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The placeholder scale operands (never read): a 1-byte scale buffer and an empty f32 vector, cached per indexed device
    as the op's constants are (:func:`~.api_fp4._const`: a blocking host copy outside CUDA graph capture; under capture an
    allocation of that capture alone, never cached, so a capture's private pool serves no eager call). The empty vector also
    marks a pre-quantized NVFP4 pack without per-token scales (the per-tensor pack of the TRTLLM backends: unit scales,
    nothing to read or fill) and stands for the MX pairs' absent alphas and per-token scales."""
    from .api_fp4 import _const, _device_key

    key = _device_key(device)
    return (
        _const(_NO_SF, key, lambda d: torch.zeros(1, dtype=torch.uint8, device=d), device),
        _const(_NO_PTS, key, lambda d: torch.empty(0, dtype=torch.float32, device=d), device),
    )  # fmt: skip


def _tactic_fp4(c1, c2) -> tuple:
    """A pair of records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache): the bf16 runner's six
    per record plus the swapped-record and the gather-form flags and the record's k-block (the mixed pair's swapped records
    carry their stage width)."""
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
            cfg.k_block,
        )
    )


def _configs_from_tactic_fp4(t: tuple):
    """The two records of a tactic tuple; a tuple of another length (a tactic persisted under another tag) is refused by
    name, never split at the wrong field."""
    if len(t) != 2 * TACTIC_FIELDS:
        raise ValueError(
            f"the fp4 MoE backend's tactic is {2 * TACTIC_FIELDS} scalars ({TACTIC_FIELDS} per record), got {len(t)}"
        )

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
            k_block=int(v[8]),
        )

    return one(t[:TACTIC_FIELDS]), one(t[TACTIC_FIELDS : 2 * TACTIC_FIELDS])


class BatchInvariantFp4Runner(MoERunner):
    """The unified layer's runner for the batch-invariant fp4 MoE classes: the pair from the configuration's ``QuantConfig``,
    the routed scale constant and the part at build, the intermediate's class from the backend config."""

    backend_key = BACKEND_KEY
    TACTIC_TAG = "moe.fp4-v2"  # v2: the tactic carries the k-block
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
    # contract; the MX pairs have none and refuse the flag at support
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
        # the mixed pair's instruction class on this part (set at build; None for the fp4 pairs, whose classes follow the
        # weight shapes)
        self._in_fmt: Optional[str] = None
        # the routing term's state, set by _check_support (`_method` None: no from-logits class; a deferred refusal's
        # message, raised afresh at every from-logits call: no exception object, hence no traceback, is kept)
        self._method: Optional[S.RoutingMethod] = None
        self._grouped_other: bool = False
        self._logits_refusal: Optional[Tuple[Any, ...]] = None
        # the intermediate's class (`per_token` / `static`): the bound backend config's field
        cands = [c for c in config.backend if hasattr(c, "intermediate")]
        self._intermediate: str = cands[0].intermediate if cands else "per_token"
        # the activation's chain, a function of the configuration: derived once, at the support check
        self._act: Optional[Any] = None
        # the ranked records per shape key, taken at the pick generation `_picks_gen` (`runner.pick_generation`)
        self._picks: Dict[tuple, tuple] = {}
        self._picks_gen: int = -1

    @property
    def pair(self) -> Fp4Pair:
        """The configured quantization pair."""
        if self._pair is None:
            raise unsupported_pair(self.config.quant)
        return self._pair

    # ---- support ---------------------------------------------------------------------------------------------------
    def _activation(self):
        a = self._act
        if a is None:
            from .spec import activation_of

            a = self._act = activation_of(self.config.activation)
        return a

    def _memo(self) -> Dict[tuple, tuple]:
        """The runner's memoised rankings, emptied when the rankings were cleared since they were taken."""
        gen = pick_generation()
        if self._picks_gen != gen:
            self._picks.clear()
            self._picks_gen = gen
        return self._picks

    def _check_experts(self) -> None:
        """The class's rule on the expert range: the EP form is not part of this backend, so a shard is refused in the
        class's terms (before the layer's generic shard check). The pair is known here: ``_check_support`` refuses an
        unserved pair before this check, and the ``pair`` property refuses it again for a direct caller."""
        experts, routing = self.config.experts, self.config.routing
        if experts.local_expert_offset or (
            experts.local_num_experts not in (None, 0, routing.num_experts)
        ):
            raise unsupported(
                self.pair.cls,
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the class's EP form is not part of this backend)",
            )

    def _mixed_class(self) -> Optional[str]:
        """The mixed pair's instruction class on the runner's device (the dense mixed class's rule per architecture family);
        None for the fp4 pairs."""
        if not self.pair.mixed:
            return None
        from .api_mxfp4w8a import mx_class_of

        return mx_class_of(self.device)

    def _check_support(self) -> None:
        if self._pair is None:
            raise unsupported_pair(self.config.quant)
        # the class's shard refusal first, then the layer's generic rules, then the pair's
        self._check_experts()
        super()._check_support()
        cfg, pair, cls = self.config, self.pair, self.pair.cls
        if not pair.scaled:
            # the MX pairs carry no per-token scale and no static-intermediate class (both are the NVFP4 pair's)
            if cfg.quant.per_token_scale:
                raise unsupported(
                    cls,
                    "QuantConfig(per_token_scale=True)",
                    "no per-token scale (the MX pairs' scales are the block scales alone)",
                )
            if self._intermediate != "per_token":
                raise unsupported(
                    cls,
                    f"BatchInvariantFp4Config(intermediate={self._intermediate!r})",
                    "the intermediate quantized per 32-element block by the class itself (the static class is the NVFP4 pair's)",
                )
        if pair.mixed and cfg.quant.swizzled_scale_factors is True:
            # the MXFP8 pack's scales are the token-major rows: the flag names a layout the pair does not take
            raise unsupported(
                cls,
                "QuantConfig(swizzled_scale_factors=True)",
                "the token-major linear scale layout of the canonical MXFP8 pack (swizzled_scale_factors unset or False)",
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
        # the shape rule on the dimensions the config fixes (H enters at the first call, with the full rule), at the mixed
        # pair's instruction class of this part
        pair.check_shape(
            None,
            experts.intermediate_size,
            routing.num_experts,
            routing.top_k,
            a.gated,
            in_fmt=self._mixed_class(),
        )
        # a routing term the from-logits mode has no class for (Llama4 beyond top-1, a group shape outside the rule) is
        # refused at a from-logits call, not here: the pre-routed modes never consult it (the bf16 runner's rule)
        self._logits_refusal = None
        try:
            self._method, self._grouped_other = routing_term(routing, cls)
        except BatchInvariantUnsupportedError as e:
            self._method, self._grouped_other = None, False
            self._logits_refusal = tuple(e.args)
        if not cfg.finalize.do_finalize:
            raise unsupported(
                cls,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    def _build(self) -> None:
        from ...gemm.batch_invariant.api import part_of

        self._part = part_of(self.device)
        self._in_fmt = self._mixed_class()
        method = self._method
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
        ``pack_inputs``); a view of another pair stays a refusal, since no candidate of the layer serves it. The view names
        its pair and its class (``prepare_weights`` writes ``quant`` and ``intermediate`` for every pair); a view without
        either key is declined, never read as the NVFP4 pair or the per-token class (``pack_inputs`` names the missing
        key)."""
        if not super().accepts(act, weights):
            return False
        view = weights.get_view(self.backend_key)
        return "quant" in view and view.get("intermediate") == self._intermediate

    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        if x.ndim != 2:
            return False
        if self.pair.mixed:
            return (
                x.dtype == torch.float8_e4m3fn
                and x.shape[1] % 32 == 0
                and act.hidden_states_scale is not None
            )
        if x.dtype == torch.bfloat16:
            return x.shape[1] % 32 == 0
        return x.dtype == torch.uint8 and (2 * x.shape[1]) % 32 == 0

    @staticmethod
    def _hidden_of(x: torch.Tensor) -> int:
        return 2 * int(x.shape[1]) if x.dtype == torch.uint8 else int(x.shape[1])

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
        gather_sf_layout = _names.gather_sf_layout
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
        # the one layout rule; a tensor named linear with another shape is refused there
        layout = gather_sf_layout(sf, T, H, pair.block, named, cls)
        if layout == "linear":
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

    def _mxfp8_pack(
        self, act: MoEActivationPack
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The mixed pair's activation pack (the shared MXFP8 pack rule, :func:`~.api_mxfp8.check_activation_pack`): the e4m3
        rows and their token-major UE8M0 scales in the linear ``[T, H / 32]`` shape, taken as given (uint8, ``float8_e8m0fnu``
        or the e4m3-typed carrier ``TrtllmFp4Config.prepare_activations`` gives this pair), both 16-byte aligned; no per-token
        scale (the placeholder stands in the pack)."""
        check_activation_pack = _names.check_activation_pack
        cls = self.pair.cls
        if act.per_token_scale is not None:
            raise unsupported(
                cls,
                "a per-token scale",
                "no per-token scale (the pair's scales are the two operands' block scales alone)",
            )
        x, sf = check_activation_pack(
            act.hidden_states_q,
            act.hidden_states_scale,
            cls,
            sf_dtypes=(torch.uint8, torch.float8_e8m0fnu, torch.float8_e4m3fn),
        )
        return x, sf, _no_scales(x.device)[1]

    def _plan_form(self, T: int, H: int) -> dict:
        """The fused route + plan launch's form: the dual plan (the dense rows with the padded scale positions) for the fp4
        pairs' dense packing, and for the mixed pair the form of the packing the ranked records run on at this token count
        (the dual plan, or the padded plan of whole 128-row atoms per group)."""
        GROUP_ALIGN = _names.GROUP_ALIGN
        if self.pair.mixed and self._packing(T, H) == "padded":
            return dict(align=GROUP_ALIGN)
        return dict(dual=True, pad_align=GROUP_ALIGN)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        route, route_plan, MoEWeightsFp4 = (
            _names.route,
            _names.route_plan,
            _names.MoEWeightsFp4,
        )
        cfg, pair, cls = self.config, self.pair, self.pair.cls
        plan: List[torch.Tensor] = []
        w_round = False  # bound to the pack (the bf16 runner's rule): a later pack never changes an earlier one's rounding
        x = act.hidden_states_q
        if pair.mixed:
            x, sf, pts = self._mxfp8_pack(act)
        else:
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
        if "quant" not in view:
            raise unsupported(
                cls,
                "a weight view without its quantization pair (no `quant` key)",
                "the view of BatchInvariantFp4Config.prepare_weights, which names the pair for every pair",
            )
        if view["quant"] != pair.name:
            wq, aq = next(q for q, p in PAIRS.items() if p is pair)
            raise unsupported(
                cls,
                "weights prepared for another pair of the fp4 backend",
                f"weights from BatchInvariantFp4Config.prepare_weights(..., quant=QuantConfig(weight={wq.name}, "
                f"activation={aq.name}))",
            )
        if "intermediate" not in view:
            raise unsupported(
                cls,
                "a weight view without its intermediate class (no `intermediate` key)",
                "the view of BatchInvariantFp4Config.prepare_weights, which names the class for every pair",
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
        pair.check_shape(H, w.intermediate_size, E, k, w.gated, in_fmt=self._in_fmt)
        if pair.mixed:
            pass  # the pack's rows and scales as validated above
        elif x.dtype == torch.bfloat16:
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
                if self._logits_refusal is not None:
                    raise BatchInvariantUnsupportedError(*self._logits_refusal)
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
            if (
                0 < T <= _names.ROUTE_PLAN_MAX_TOKENS
                and T * k <= _names.ROUTE_PLAN_MAX_SLOTS
                and E <= _names.MAX_E
            ):
                # routing and the permutation plan in one launch (a placement: the same bits and integers), in the form of
                # the packing the pair runs on at this token count; the plan tensors follow the ids and weights as the
                # builder returns them
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
        the MX pairs (whose quantizers store the class's layout); a bf16 NVFP4 input's quantizer scatters in its own pass."""
        return x.dtype != torch.bfloat16 or not self.pair.scaled

    def _gather_form(self, x: torch.Tensor, sf: torch.Tensor) -> bool:
        """Whether GEMM1's narrow swapped record takes its gather form for this pack: the box form would scatter in a launch
        of its own (:meth:`_scatter_launch`) and the pack's scale layout admits the form (a linear layout's rows as whole
        4-byte groups: :func:`~.api_fp4.gather_form_ok`; the 128x4 buffer always). The records handed to the op are then the
        pack's own, so the op never has to fall back from a record the runner chose. The mixed pair's bodies read the token
        box: no gather form."""
        if self.pair.mixed or not self._scatter_launch(x):
            return False
        linear = x.dtype != torch.bfloat16 and sf.ndim == 2
        return _names.gather_form_ok(
            self._hidden_of(x), "linear" if linear else "128x4", self.pair.block
        )

    def _ranked(self, T: int, H: int, gather_form: bool = False) -> Tuple[list, list]:
        """The ranked records of both GEMMs (the tactic space's head); with ``gather_form`` GEMM1's list carries the
        gather-form twin the structural rule places (:func:`~.schedule_fp4.gather_form_rule`). The mixed pair's lists are
        its dispatcher's over both record spaces (:func:`~.schedule_mxfp4w8a.ranked_mx`), trimmed to the packing the first
        picks run on (on the padded packing the swapped records cannot run, so they leave the lists). Memoised per (token
        count, hidden size, gather form): a forward ranks nothing after the first at its shape."""
        memo = self._memo()
        key = (T, H, gather_form)
        r = memo.get(key)
        if r is not None:
            return r
        cfg, pair = self.config, self.pair
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        # the chains of the class (the weights' terms without the tensors)
        w = _names.MoEWeightsFp4(
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
        if pair.mixed:
            in_fmt = self._in_fmt or pair.fmts[0]
            # both record spaces are ranked for the fused-requantization spelling GEMM1 runs
            r1, r2 = _names.ranked_mx(
                T, k, H, I, E, w.chain1_quant, in_fmt, sm_count, family, _names.TOP_K
            )
            if _names.packing_for(r1[0], r2[0]) == "padded":
                r1 = [c for c in r1 if not c.swap_ab]
                r2 = [c for c in r2 if not c.swap_ab]
            memo[key] = (r1, r2)
            return r1, r2
        rows, n_act = _names.rows_per_expert(T, k, E), _names.active_experts(T, k, E)
        in1, in2 = pair.classes(H, I, family, a.gated)
        chain1, chain2 = w.chain1, w.chain2
        n1 = 2 * I if a.gated else I
        r1 = list(
            _names.rank_grouped_fp4(
                rows, n1, H, E, chain1, in1, sm_count, family, _names.TOP_K_FP4, n_active=n_act
            )
        )  # fmt: skip
        twins = [_names.gather_form_rule(c, gather_form) for c in r1]
        r1 += [t for t in twins if t.gather_a and t not in r1]
        r2 = list(
            _names.rank_grouped_fp4(
                rows, H, I, E, chain2, in2, sm_count, family, _names.TOP_K_FP4, n_active=n_act
            )
        )  # fmt: skip
        memo[key] = (r1, r2)
        return r1, r2

    def _packing(self, T: int, H: int) -> str:
        """The mixed pair's packing of the expanded rows at ``T`` tokens: the first picks' (`schedule_mxfp4w8a.packing_for`)."""
        r1, r2 = self._ranked(T, H)
        return _names.packing_for(r1[0], r2[0])

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        x = inputs[_X]
        r1, r2 = self._ranked(
            int(x.shape[0]), self._hidden_of(x), self._gather_form(x, inputs[_SF])
        )
        return [_tactic_fp4(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        """The autotuner cache key's extras: the pair, the intermediate class, the mixed pair's instruction class and the
        tactic's versioned tag last (a tuple persisted under another tag never reaches the decoder)."""
        return super()._cache_key_extras() + (
            self.pair.name,
            self._intermediate,
            self._in_fmt or "",
            self.TACTIC_TAG,
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
        MoEWeightsFp4 = _names.MoEWeightsFp4
        pair = self.pair
        x, sf, pts, ids, wts, w1, w1sf, w1a, w2, w2sf, w2a, out, gs2, inv2 = inputs[:14]
        plan = (
            tuple(inputs[14:]) or None
        )  # the plan's tensors, in the form the pack built them
        H = self._hidden_of(x)
        a = self._activation()
        if pair.mixed:
            packing_for = _names.packing_for
            T = int(x.shape[0])
            if tactic == -1 or tactic is None:
                r1, r2 = self._ranked(T, H)
                c1, c2 = r1[0], r2[0]
            else:
                c1, c2 = _configs_from_tactic_fp4(tuple(tactic))
            w = MoEWeightsFp4(
                pair, w1, w1sf, w2, w2sf, H, self.config.experts.intermediate_size, a.gated, a.act, a.params
            )  # fmt: skip
            # the packing is the records' own (a swapped record reads the dense packing, the grouped body reads either): the
            # layer keys a tuned winner by its token bucket, so a tactic may replay at a token count whose first picks pack
            # the other way; the plan packed in the first picks' form is kept where the records read it (the dual plan feeds
            # both bodies) and dropped otherwise (the op computes its own: the same integers, the same bits)
            path = packing_for(c1, c2)
            if plan is not None:
                if len(plan) == _PLAN_LEN:
                    path = "dense"
                elif path == "dense":
                    plan = None
            if T == 0:
                return (
                    out  # no tokens: nothing to route, the (0, H) output as it stands
                )
            if c1.gather_a:
                c1 = dataclasses.replace(
                    c1, gather_a=False
                )  # the mixed classes read the token box: no gather form
            # the op's launch on the validated pack (the pack proved every rule the public entry checks)
            _names.fused_moe_mxfp8_run(
                x, sf, ids, wts, w, out, c1, c2, self._in_fmt or pair.fmts[0], path, "fused", False,
                T, H, ids.shape[1], w.num_experts, self.config.experts.intermediate_size,
                plan=plan, w_round=getattr(inputs, "w_round", False),
            )  # fmt: skip
            return out
        if tactic == -1 or tactic is None:
            gather_form = self._gather_form(x, sf)
            r1, r2 = self._ranked(int(x.shape[0]), H, gather_form)
            c1, c2 = _names.gather_form_rule(r1[0], gather_form), r2[0]
        else:
            c1, c2 = _configs_from_tactic_fp4(tuple(tactic))
            if c1.gather_a and not self._gather_form(x, sf):
                # a tuned winner replayed on a pack whose scale layout the gather form cannot copy (the layer keys its
                # winners by token bucket, not by layout): the record's box form, a placement of the same bits
                c1 = dataclasses.replace(c1, gather_a=False)
        static = pair.scaled and self._intermediate == "static"
        w = MoEWeightsFp4(
            pair, w1, w1sf, w2, w2sf, H, self.config.experts.intermediate_size, a.gated, a.act, a.params,
            w1a if pair.scaled else None, w2a if pair.scaled else None,
            self._intermediate, gs2 if static else None, inv2 if static else None,
        )  # fmt: skip
        # the op's launch on the validated pack (the pack proved every rule the public entry checks): the bf16 rows as
        # given (the NVFP4 pair quantizes them into the grouped layout inside the op, the MX pair here, as the public entry
        # does), or the pre-quantized pack -- the empty per-token placeholder means unit scales (the per-tensor pack); the
        # scale tensor's rank names its layout (flat: the 128x4 buffer, 2-D: the linear rows, as pack_inputs shaped it);
        # both are the op's own placements (no fill, no re-tiling launch)
        T, dev = int(x.shape[0]), x.device
        gsi = _names.gs_inv(_names.TE_GS_INV, dev)
        if x.dtype == torch.bfloat16:
            x_rows, x4, x_sf, x_pts, sf_src, unit_pts = (
                x,
                None,
                None,
                None,
                "128x4",
                False,
            )
            if not pair.scaled:
                x4, x_sf, x_pts = pair.quantize(x_rows, gsi)
        else:
            sf_src = "linear" if sf.ndim == 2 else "128x4"
            unit_pts = not (pair.scaled and pts.numel() == T)
            x_rows, x4, x_sf = None, x, sf.reshape(-1)
            if not pair.scaled:
                x_pts = None
            elif unit_pts:
                x_pts = _names.gs_inv(
                    1.0, dev
                )  # a placeholder the unit-scale scatter never reads
            else:
                x_pts = pts
        in1, in2 = w.classes(self._part[1] if self._part is not None else None)
        _names.fused_moe_fp4_run(
            w, x_rows, x4, x_sf, x_pts, sf_src, unit_pts, ids, wts, out, c1, c2, in1, in2,
            T, H, ids.shape[1], w.num_experts, self.config.experts.intermediate_size, gsi,
            plan=plan, round_scales=getattr(inputs, "w_round", False),
        )  # fmt: skip
        return out
