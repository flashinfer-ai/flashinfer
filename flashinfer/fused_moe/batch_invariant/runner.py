# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MoE backend of the unified MoE layer (``BatchInvariantBf16Config`` -> ``BatchInvariantMoeRunner``).

The runner serves the bf16 classes ``moe.bf16.swiglu`` (``SwiGLU`` with the default parameters) and ``moe.bf16.relu2``
(``ReLU2``) on SM100 / SM103 / SM107. It takes every routing input mode: ``FromLogits`` runs the class's own routing kernel
(the declared ``exp``, the strict selection order), ``PackedPrecomputed`` / ``UnpackedPrecomputed`` take the caller's
expert ids and weights. The combine is the slot-order fp32 fold (never an atomic finalize: ``MoEFinalizeConfig.use_fused_finalize``
is ignored), the tactics are the ranked grouped-GEMM records of the dispatcher (any tactic gives the same bits), and an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

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

CLASS = "moe.bf16"
#: The compute capabilities the classes are stated for (the dense GEMM class's parts).
MOE_BF16_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: Positions of the packed inputs: the activations, the expert ids, the routing weights, the two weights and the output.
_X, _IDS, _W, _W1, _W2, _OUT = range(6)


def _tactic(c1, c2) -> tuple:
    """A pair of grouped records as a flat tuple of scalars (JSON round-trip safe for the autotuner's cache)."""
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
        )
    )


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
        )

    return one(t[:6]), one(t[6:12])


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
    # Expert-parallel shards (a local expert range with ids outside it) are the EP classes of a later PR.
    supports_expert_parallelism = False

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

    # ---- support ---------------------------------------------------------------------------------------------------
    def _activation(self):
        self._scale: Optional[torch.Tensor] = None
        """The chain of the layer's activation (``spec.ActSpec``: gating, chain name, binary32 parameters)."""
        from .spec import activation_of

        return activation_of(self.config.activation)

    def _check_support(self) -> None:
        super()._check_support()
        from .spec import METHOD_OF_FLASHINFER

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
        if experts.intermediate_size % 8 or experts.intermediate_size <= 0:
            raise unsupported(
                CLASS,
                f"intermediate_size {experts.intermediate_size}",
                "a positive multiple of 8",
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
        if routing.method not in METHOD_OF_FLASHINFER:
            # the pre-routed modes never consult the method; the logits mode does (pack_inputs raises there too)
            self._method = None
        else:
            self._method = METHOD_OF_FLASHINFER[routing.method]
        if not cfg.finalize.do_finalize:
            raise unsupported(
                CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )

    def _build(self) -> None:
        from ...gemm.batch_invariant.api import part_of

        self._part = part_of(self.device)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
    def _scale_tensor(self) -> torch.Tensor:
        """The routed scaling factor as a one-element fp32 tensor on the runner's device, built once with a blocking copy
        (the route kernel reads it as given: no per-call fill, no process-wide cache keyed by the value)."""
        if self._scale is None:
            v = self.config.routing.routed_scaling_factor
            self._scale = torch.tensor(
                [1.0 if v is None else float(v)], dtype=torch.float32
            ).to(self.device)
        return self._scale

        return x.ndim == 2 and x.dtype == torch.bfloat16 and x.shape[1] % 8 == 0

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._validate_pack_contract(act)
        self._require_built()
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
        view = weights.get_view(self.backend_key)
        w = MoEWeights.from_view(view)
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
        a = self._activation()
        if (w.gated, w.act, w.params) != (a.gated, a.act, a.params):
            raise unsupported(
                CLASS,
                f"{type(cfg.activation).__name__} with {'gated' if w.gated else 'ungated'} weights",
                "SwiGLU with fc1 [E, 2I, H], ReLU2 with fc1 [E, I, H]",
            )
        mode = act.routing_input_mode
        if mode is RoutingInputMode.FromLogits:
            _validate_logits_inputs(act, T, E, type(self).__name__)
            # group-limited selection is a routing term the class does not have; with pre-routed ids the group
            # structure never enters the computation, so only the logits mode refuses it
            if cfg.routing.n_group not in (
                None,
                0,
                1,
            ) or cfg.routing.topk_group not in (
                None,
                0,
                1,
            ):
                raise unsupported(
                    CLASS,
                    "group-limited routing (n_group > 1) from logits",
                    "n_group in (None, 0, 1) from logits, or pre-routed ids for grouped routing",
                )
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
            scale = self._scale_tensor() if self._method.scaled else None
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
                # the packed word carries the weight as bf16 (``(expert_id << 16) | bf16(weight)``): the rounded weights
                # are the class's inputs in this mode
                wts = wts.to(torch.bfloat16)
            wts = wts.to(torch.float32).contiguous()
        out = torch.empty(T, x.shape[1], dtype=torch.bfloat16, device=x.device)
        return [x, ids, wts, w.w1i, w.w2, out] + plan

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int) -> Tuple[list, list]:
        from .schedule import TOP_K, active_experts, rank_grouped, rows_per_expert

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = self._hidden
        rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
        gated = self._activation().gated
        sm_count, family = self._part if self._part is not None else (None, None)
        n1, chain1 = (2 * I, "swiglu") if gated else (I, "relu2")
        return (
            rank_grouped(
                rows, n1, H, E, chain1, sm_count, family, TOP_K, n_active=n_act
            ),
            rank_grouped(
                rows, H, I, E, "plain", sm_count, family, TOP_K, n_active=n_act
            ),
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        self._require_built()
        self._hidden = int(inputs[_X].shape[1])
        r1, r2 = self._ranked(int(inputs[_X].shape[0]))
        return [_tactic(c1, c2) for c1 in r1 for c2 in r2]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + ("moe.bf16-v1",)

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
        fused_moe(x, ids, wts, w, out=out, config1=c1, config2=c2, plan=plan)
        return out
