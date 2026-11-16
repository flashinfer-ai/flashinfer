# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MoE backend of the unified MoE layer (``BatchInvariantBf16Config`` -> ``BatchInvariantMoeRunner``).

The runner serves the bf16 classes on SM100 / SM103 / SM107, one declared chain per typed activation of the layer:
``SwiGLU`` with the default parameters is ``moe.bf16.swiglu``; a ``SwiGLU`` with other ``alpha`` / ``beta`` / ``limit``,
``SwiGLUStep``, ``GeGLU``, ``GeGLUTanh``, ``SiTU`` and ``PowLU`` are the gated chains with their scalars as class parameters;
``ReLU2`` is ``moe.bf16.relu2`` and ``Identity``, ``GELU``, ``ReLU``, ``SiLU`` the other ungated chains (``spec.activation_of``).
It takes every routing input mode: ``FromLogits`` runs the class's own routing kernel
(the declared ``exp``, the strict selection order; the methods ``Default``, ``Renormalize``, ``RenormalizeNaive``,
``Sigmoid``, ``SigmoidRenorm``, ``TopK``, ``Llama4`` at ``top_k = 1``, ``TopKSigmoid`` and ``DeepSeekV3`` with or
without its expert groups; a configuration outside these terms is refused at the from-logits call, not at construction),
``PackedPrecomputed`` / ``UnpackedPrecomputed`` take the caller's expert ids and weights under any ``RoutingConfig`` (the
selection is the caller's); an expert-parallel shard (``ExpertConfig.local_expert_offset`` / ``local_num_experts``) computes the
slots of its local experts and drops the rest (the term ``+0`` in its tokens' folds): a placement of the class on the local
experts, the rank's partial of the dense output. The combine is the slot-order fp32 fold (never an atomic finalize:
``MoEFinalizeConfig.use_fused_finalize`` is ignored); with ``MoEFinalizeConfig(do_finalize=False)`` the combine is left to the caller and the forward returns the
layer's unfinalized triple ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]`` -- the per-slot rows, the
weights and the positions exactly as the class's combine would fold them (a placement of the op: the rows are reproducible
intermediates). The tactics are the ranked grouped-GEMM records of the dispatcher (any tactic gives the same bits), and an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

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
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingInputMode,
    RoutingMethodType,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from ..runners import MoERunner, _validate_logits_inputs, _validate_prerouted_inputs

CLASS = "moe.bf16"
#: The compute capabilities the classes are stated for (the dense GEMM class's parts).
MOE_BF16_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: Positions of the packed inputs: the activations, the expert ids, the routing weights, the two weights and the output.
_X, _IDS, _W, _W1, _W2, _OUT = range(6)


def _tactic(c1, c2) -> tuple:
    """A pair of records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache): the tile, the cluster,
    the CTA group, the raster and the swapped flag (the record's body: the grouped or the swapped one) per record."""
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
        )
    )


_TACTIC_FIELDS = 7


def _configs_from_tactic(t: tuple):
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
        )

    return one(t[:_TACTIC_FIELDS]), one(t[_TACTIC_FIELDS : 2 * _TACTIC_FIELDS])


def routing_term(routing: Any, cls: str) -> Tuple[Optional[Any], bool]:
    """The routing term of a ``RoutingConfig`` for a class runner of this module: the method of ``METHOD_OF_FLASHINFER`` (``None``
    for a method this module has no term for: the pre-routed modes never consult it, the logits mode refuses it at the call)
    with DeepSeekV3's expert groups attached (``with_groups``: the group-limited selection is a term of that method, its shape
    rule here) and the Llama4 top-1 rule; the flag: expert groups given with another method (no meaning in this module; the
    logits mode refuses them at the call). ``cls``: the class name of the errors. Every class runner derives its term here."""
    from .spec import METHOD_OF_FLASHINFER, group_shape_error

    if routing.method not in METHOD_OF_FLASHINFER:
        # the pre-routed modes never consult the method; the logits mode does (pack_inputs raises there too)
        return None, False
    else:
        method = METHOD_OF_FLASHINFER[routing.method]
        grouped_other = False
        grouped = (routing.n_group or 0) > 1 or (routing.topk_group or 0) > 1
        if grouped and routing.method is RoutingMethodType.DeepSeekV3:
            # the group-limited selection: a term of the class with the groups as parameters (its shape rule here;
            # the pre-routed modes never consult it)
            n_group, topk_group = (
                int(routing.n_group or 0),
                int(routing.topk_group or 0),
            )
            E = routing.num_experts
            rule = group_shape_error(E, routing.top_k, n_group, topk_group)
            if rule is not None:
                raise unsupported(
                    cls,
                    f"n_group {n_group}, topk_group {topk_group} over {E} experts for top_k {routing.top_k}",
                    rule,
                )
            method = method.with_groups(n_group, topk_group)
        elif grouped:
            grouped_other = True
        if routing.method is RoutingMethodType.Llama4 and routing.top_k != 1:
            raise unsupported(
                cls,
                f"Llama4 routing with top_k {routing.top_k}",
                "top_k = 1 (the method is the top-1 of the logits with its sigmoid as the weight; TopKSigmoid for other k)",
            )
    return method, grouped_other


class _PackedInputs(list):
    """The packed inputs with the mode's weight dtype attached (the dtype the unfinalized form returns the weights in; the
    autotuner's profiling lists carry no attribute and read as fp32, the class's own weight dtype)."""

    weights_dtype: torch.dtype = torch.float32
    w_round: bool = False  # f32 weights of the packed mode: rounded to bf16 by the combine kernel, not on the host


class BatchInvariantMoeRunner(MoERunner):
    """The unified layer's runner for the batch-invariant bf16 MoE classes."""

    backend_key = "batch_invariant_bf16"
    supported_routing_modes = (
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    supported_output_formats = (QuantFormat.BF16,)
    # every typed activation of the layer: a declared chain of the class each (``spec.activation_of``)
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
    # An expert-parallel shard computes its local experts' slots: the other slots are dropped (the term +0 in the fold),
    # a placement of the within-rank class on the local experts.
    supports_expert_parallelism = True

    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        # Exact shapes: the dispatcher's pick is a function of the token count, so a bucket's winner is not a valid
        # specialisation of another token count (every tactic computes the same bits; this is about the time).
        self.tuning_config = TuningConfig(use_cuda_graph=True)
        self._part: Optional[Tuple[int, str]] = None
        self._scale: Optional[torch.Tensor] = None

    # ---- support ---------------------------------------------------------------------------------------------------
    def _activation(self):
        """The chain of the layer's activation (``spec.ActSpec``: gating, chain name, binary32 parameters)."""
        from .spec import activation_of

        return activation_of(self.config.activation)

    def _local_experts(self) -> Tuple[int, int]:
        """``(offset, count)`` of this shard's experts (every expert without expert parallelism)."""
        experts = self.config.experts
        n = experts.local_num_experts
        return int(experts.local_expert_offset), int(
            self.config.routing.num_experts if n is None else n
        )

    def _check_support(self) -> None:
        super()._check_support()
        cfg = self.config
        check_device(self.device, CLASS, MOE_BF16_ARCHS)
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
        e_off, e_local = self._local_experts()
        if not (
            e_off >= 0 and e_local >= 1 and e_off + e_local <= cfg.routing.num_experts
        ):
            raise unsupported(
                CLASS,
                f"local experts [{e_off}, {e_off + e_local}) of {cfg.routing.num_experts}",
                "a local expert range inside [0, num_experts)",
            )
        if experts.intermediate_size % 8 or experts.intermediate_size <= 0:
            raise unsupported(
                CLASS,
                f"intermediate_size {experts.intermediate_size}",
                "a positive multiple of 8",
            )
        routing = cfg.routing
        from .spec import MAX_TOP_K

        if not 1 <= routing.top_k <= min(routing.num_experts, MAX_TOP_K):
            raise unsupported(
                CLASS,
                f"top_k {routing.top_k}",
                f"1 <= top_k <= min(num_experts, {MAX_TOP_K})",
            )
        from .blocks.scheduler import MAX_E

        if not 1 <= routing.num_experts <= MAX_E:
            raise unsupported(
                CLASS,
                f"num_experts {routing.num_experts}",
                f"1 <= num_experts <= {MAX_E}",
            )
        self._routing_term(routing, CLASS)

    def _routing_term(self, routing: Any, cls: str) -> None:
        """``_method`` / ``_grouped_other`` from :func:`routing_term`; a refusal (a group shape or a Llama4 top-k outside the
        term) is deferred to a from-logits call (``_logits_refusal``), since the pre-routed modes never consult the method:
        a layer that only ever receives pre-routed ids builds under any ``RoutingConfig``."""
        self._logits_refusal: Optional[BatchInvariantUnsupportedError] = None
        try:
            self._method, self._grouped_other = routing_term(routing, cls)
        except BatchInvariantUnsupportedError as e:
            self._method, self._grouped_other, self._logits_refusal = None, False, e

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
        return x.ndim == 2 and x.dtype == torch.bfloat16 and x.shape[1] % 8 == 0

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        from .api import MoEWeights, route, route_plan

        cfg = self.config
        plan: List[torch.Tensor] = []
        x = act.hidden_states_q
        if x.dtype != torch.bfloat16 or x.ndim != 2:
            raise unsupported(
                CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                "a bf16 [num_tokens, hidden_size]",
            )
        if act.hidden_states_scale is not None:
            raise unsupported(
                CLASS, "hidden_states_scale", "no activation scale (bf16 activations)"
            )
        x = x.contiguous()
        T = x.shape[0]
        k, E = cfg.routing.top_k, cfg.routing.num_experts
        e_off, e_local = self._local_experts()
        view = weights.get_view(self.backend_key)
        w = MoEWeights.from_view(view)
        if (
            x.shape[1] != w.hidden_size
            or w.num_experts != e_local
            or cfg.experts.intermediate_size != w.intermediate_size
        ):
            raise unsupported(
                CLASS,
                f"weights (E {w.num_experts}, I {w.intermediate_size}, H {w.hidden_size}) for hidden {x.shape[1]}, "
                f"local experts {e_local}, I {cfg.experts.intermediate_size}",
                "weights prepared for this configuration (the local experts of a shard)",
            )
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"a weight view of the {'gated' if w.gated else 'ungated'} chain {w.act!r} with parameters "
                f"{w.params!r} under {type(cfg.activation).__name__} (the {'gated' if a.gated else 'ungated'} chain "
                f"{a.act!r} with parameters {a.params!r})",
                "a weight view prepared with this layer's activation "
                "(prepare_weights(..., activation=config.activation))",
            )
        mode = act.routing_input_mode
        weights_dtype = torch.float32  # the class's weights from logits are fp32
        w_round = False
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
                # the refusal deferred at construction (the pre-routed modes never consulted it), else no term
                refusal = getattr(self, "_logits_refusal", None)
                if refusal is not None:
                    raise refusal
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
                and cfg.routing.num_experts <= MAX_E
            ):
                # routing and the permutation plan in one launch (a placement: the same bits and integers); a shard plans
                # its local experts by their local ids
                ids, wts, counts, dest, src = route_plan(
                    act.routing_logits,
                    k,
                    self._method,
                    bias,
                    scale,
                    local=None if (e_off, e_local) == (0, E) else (e_off, e_local),
                )
                plan = [counts, dest, src]
                if e_off:
                    ids = ids - e_off
            else:
                ids, wts = route(act.routing_logits, k, self._method, bias, scale)
                if e_off:
                    ids = ids - e_off
        else:
            _validate_prerouted_inputs(
                act,
                T,
                k,
                type(self).__name__,
                allowed_weights_dtypes=(torch.bfloat16, torch.float32),
                require_contiguous=True,
            )
            # a shard's local ids: the caller's global ids less the offset (an id outside the local range, like a
            # caller's dropped slot, is a dropped slot of the plan: never host-validated, the term +0 in the fold)
            ids = act.topk_ids.contiguous()
            if e_off:
                ids = ids - e_off
            wts = act.topk_weights
            if mode is RoutingInputMode.PackedPrecomputed:
                # the packed word carries the weight as bf16 (``(expert_id << 16) | bf16(weight)``): the rounded weights
                # are the class's inputs in this mode; f32 weights are rounded by the combine kernel itself (no host pass
                # over the slots), bf16 ones widen exactly
                weights_dtype = torch.bfloat16
                w_round = wts.dtype != torch.bfloat16
            else:
                weights_dtype = wts.dtype
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
        packed = _PackedInputs([x, ids, wts, w.w1i, w.w2, out] + plan)
        packed.weights_dtype = weights_dtype
        packed.w_round = w_round
        return packed

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int) -> Tuple[list, list]:
        """The dispatcher's ranked records of the two GEMMs at ``T`` tokens (``rank_grouped`` memoises its arguments, all
        scalars of the configuration and the part: a forward without a tactic pays the ranking once per token count)."""
        from .schedule import TOP_K, active_experts, rank_grouped, rows_per_expert

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        from .blocks.epilogue import GroupedChain

        E_local = self._local_experts()[1]
        H = self._hidden
        # the rows per expert follow the global routing; the groups are the shard's experts
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E_local)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        n1 = 2 * I if a.gated else I
        chain1 = GroupedChain("glu" if a.gated else "act", a.act, "bf16", a.params)
        return (
            rank_grouped(
                rows, n1, H, E_local, chain1, sm_count, family, TOP_K, n_active=n_act
            ),
            rank_grouped(
                rows, H, I, E_local, "plain", sm_count, family, TOP_K, n_active=n_act
            ),
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        self._hidden = int(inputs[_X].shape[1])
        r1, r2 = self._ranked(int(inputs[_X].shape[0]))
        return [_tactic(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            "moe.bf16-v2",
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
        from .api import MoEWeights, fused_moe

        x, ids, wts, w1i, w2, out = inputs[:6]
        plan = tuple(inputs[6:9]) if len(inputs) >= 9 else None
        self._hidden = int(x.shape[1])
        if tactic == -1 or tactic is None:
            r1, r2 = self._ranked(int(x.shape[0]))
            c1, c2 = r1[0], r2[0]
        else:
            c1, c2 = _configs_from_tactic(tuple(tactic))
        a = self._activation()
        w = MoEWeights(w1i, w2, a.gated, a.act, a.params)
        if not self.config.finalize.do_finalize:
            # the layer's unfinalized triple: GEMM2's per-slot rows grouped by expert in expert order and, inside a group,
            # in token-slot order (rows past the valid ones -- a dropped slot's share -- are not written), the weights as the
            # combine would fold them (exactly widened to fp32 by the fold: the mode's dtype carries the same values) and the
            # row of every (token, slot) in the flat layout, -1 for a dropped slot; the class's slot-order fold over the
            # triple is the finalized output bit for bit (TRTLLM-Gen's rows are atomically placed and not reproducible)
            _, im = fused_moe(
                x,
                ids,
                wts,
                w,
                out=out,
                config1=c1,
                config2=c2,
                plan=plan,
                return_intermediates=True,
                finalize=False,
            )
            weights_dtype = getattr(inputs, "weights_dtype", torch.float32)
            return [im["y"], wts.to(weights_dtype), im["dest"].reshape(-1)]
        fused_moe(
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
