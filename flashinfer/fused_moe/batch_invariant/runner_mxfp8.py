# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(MXFP8, MXFP8)`` -> ``BatchInvariantFp8BlockRunner`` ->
``BatchInvariantMxfp8Runner``, the pair's runner).

The runner serves the classes ``moe.mxfp8.<act>`` (MXFP8 experts and activations, every typed activation of the layer)
on SM100 / SM103 / SM107, through the dense MXFP8 class of the part (the SM107 instruction folds 64 elements per step, the
SM100 / SM103 one 32: a dimension of the specification). It takes the layer's canonical MXFP8 activation pack (e4m3
``hidden_states_q [T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as ``TrtllmFp8BlockConfig`` /
``CutlassMxfp8Config`` / this config's ``prepare_activations`` produce it) and every routing input mode through the
shared flow of :class:`.runner.BatchInvariantMoeRunnerBase`. The combine is the slot-order fp32 fold (``use_fused_finalize``
is ignored, ``do_finalize=True`` only), the tactics are the ranked grouped-GEMM records of the dense MX heuristic over both
record spaces -- the padded body's and the swapped decode body's (the weights on the MMA's M side, the group's tokens a
narrow tile; `schedule_mxfp8`) -- any tactic gives the same bits, and the packing of the expanded rows follows the first
picks (the dense packing when either is a swapped record, else the padded one: a placement decided by the token count, so
the routing-and-plan launch takes the matching form). An unsupported configuration raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ...batch_invariant.support import unsupported
from ..api import MoEActivationPack
from .runner_fp8_block import PAIR_MXFP8, BatchInvariantFp8BlockRunner
from .spec_mxfp8 import BLOCK, CLASS

#: The compute capabilities the classes are stated for (the dense MXFP8 GEMM class's parts).
MOE_MXFP8_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))
#: Positions of the packed inputs: the quantized activations and their scales, the expert ids, the routing weights, the
#: two weights with their scales and the output.
_XQ, _XSF, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


class BatchInvariantMxfp8Runner(BatchInvariantFp8BlockRunner):
    """The fp8-block backend's runner for the MXFP8 x MXFP8 pair: the batch-invariant MXFP8 MoE classes."""

    supported_quant_variants = (PAIR_MXFP8,)

    CLASS = CLASS
    ARCHS = MOE_MXFP8_ARCHS
    X_DTYPE = torch.float8_e4m3fn
    ROW_ALIGN = BLOCK
    X_RULE = "the MXFP8 activation pack: an e4m3 [num_tokens, hidden_size] with its UE8M0 scales"
    ROW_RULE = f"a positive multiple of {BLOCK} (whole UE8M0 scale blocks)"
    SCALE_RULE = f"the token-major UE8M0 scales [num_tokens, hidden_size / {BLOCK}]"
    TACTIC_TAG = "moe.mxfp8-v2"  # v2: the tactic carries the swapped flag
    N_ACT_TENSORS = 2
    N_WEIGHT_TENSORS = 4

    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        self._in_fmt: Optional[str] = None

    def _build(self) -> None:
        super()._build()
        from .api_mxfp8 import mx_class_of

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

    def _act_tensors(self, act: MoEActivationPack) -> List[torch.Tensor]:
        """The pair's activation pack: the e4m3 rows and their token-major UE8M0 scales (uint8 or ``float8_e8m0fnu``, the
        linear layout), and the class's shape rule on the configuration at this hidden size."""
        from .api_mxfp8 import check_shape_mxfp8

        x = act.hidden_states_q
        if x.dtype != self.X_DTYPE or x.ndim != 2:
            raise unsupported(
                self.CLASS, f"hidden_states_q {x.dtype} of rank {x.ndim}", self.X_RULE
            )
        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(self.CLASS, "no hidden_states_scale", self.SCALE_RULE)
        x = x.contiguous()
        T, H = x.shape
        if sf.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise unsupported(
                self.CLASS,
                f"hidden_states_scale {sf.dtype}",
                "uint8 or UE8M0 scale bytes",
            )
        if H % BLOCK:
            raise unsupported(self.CLASS, f"hidden size {H}", self.ROW_RULE)
        sf = (
            sf.view(torch.uint8).reshape(T, -1 if T else H // BLOCK).contiguous()
        )  # the zero-token pack has no inferable column count
        if sf.shape[1] != H // BLOCK:
            raise unsupported(
                self.CLASS,
                f"hidden_states_scale of {sf.shape[1]} columns",
                f"{H // BLOCK} token-major UE8M0 scales per row (the linear layout; a swizzled buffer is not this pack)",
            )
        cfg = self.config
        check_shape_mxfp8(
            H, cfg.experts.intermediate_size, cfg.routing.num_experts, cfg.routing.top_k
        )
        return [x, sf]

    def _plan_in_kernel(self, T: int, H: int) -> bool:
        """The decode form's rule at this token count (`api_mxfp8.decode_form_default` at the first picks): GEMM1's gather
        record then builds the plan from the ids and stores the counts and the row map for GEMM2 and the combine (a tactic of
        another kind of record falls back to the plan computed by the op: the same integers)."""
        from .api_mxfp8 import decode_form_default

        k, E_local = self.config.routing.top_k, self._local_experts()[1]
        r1, r2 = self._ranked(T, H)
        return bool(
            r1
            and r2
            and decode_form_default(
                T, k, E_local, H, r1[0], r2[0], path=self._packing(T, H)
            )
        )

    def _plan_form(self, T: int, H: int) -> dict:
        """The fused route + plan launch in the form of the packing the ranked records run on at this token count: the
        dual plan (the dense rows with the padded scale positions) or the padded plan (whole 128-row atoms per group)."""
        from .api_mxfp8 import GROUP_ALIGN

        if self._packing(T, H) == "dense":
            return dict(dual=True, pad_align=GROUP_ALIGN)
        return dict(align=GROUP_ALIGN)

    # ---- the weight view -------------------------------------------------------------------------------------------
    def _weights_from_view(self, view):
        from .api_mxfp8 import MoEWeightsMXFP8

        return MoEWeightsMXFP8.from_view(view)

    def _weight_tensors(self, w) -> List[torch.Tensor]:
        return [w.w1, w.w1_sf, w.w2, w.w2_sf]

    def _weights_from_tensors(self, a, tensors: List[torch.Tensor]):
        from .api_mxfp8 import MoEWeightsMXFP8

        w1, w1_sf, w2, w2_sf = tensors
        return MoEWeightsMXFP8(w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params)

    def _chain1(self, a):
        """GEMM1's chain for the dispatcher: the bf16-output chain of the layer's activation at the part's instruction
        class (the fused requantization's extra term is the same for every record)."""
        from .blocks.epilogue_mx import MxChain

        return MxChain(
            "glu" if a.gated else "act",
            a.act,
            "bf16",
            a.params,
            self._in_fmt or "mxfp8",
        )

    def _chain2(self):
        from .schedule_mxfp8 import GEMM2_CHAIN

        return GEMM2_CHAIN.with_class(self._in_fmt or "mxfp8")

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int, H: Optional[int] = None) -> Tuple[list, list]:
        """The ranked records of both GEMMs over both record spaces at ``T`` tokens (`schedule_mxfp8.ranked_mx`, memoised),
        trimmed to the packing the first picks run on: on the padded packing the swapped records cannot run, so they leave
        the lists."""
        from .schedule import TOP_K
        from .schedule_mxfp8 import packing_for, ranked_mx

        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = H if H is not None else self._hidden
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in_fmt = self._in_fmt or "mxfp8"
        r1, r2 = ranked_mx(
            T, k, H, I, E, self._chain1(a), in_fmt, sm_count, family, TOP_K
        )
        if packing_for(r1[0], r2[0]) == "padded":
            r1 = [c for c in r1 if not c.swap_ab]
            r2 = [c for c in r2 if not c.swap_ab]
        return r1, r2

    def _packing(self, T: int, H: int) -> str:
        """The packing of the expanded rows at ``T`` tokens: the first picks' (`schedule_mxfp8.packing_for`)."""
        from .schedule_mxfp8 import packing_for

        r1, r2 = self._ranked(T, H)
        return packing_for(r1[0], r2[0])

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (self._in_fmt or "",)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def _launch(self, x, ids, wts, w, out, c1, c2, plan, inputs):
        from .api_mxfp8 import fused_moe_mxfp8

        xq, xsf = x
        # the packing follows the first picks at this token count (the plan, when given, was built in its form); a tactic
        # of the padded packing is a pair of dense records, one of the dense packing any pair
        fused_moe_mxfp8(
            (xq, xsf),
            ids,
            wts,
            w,
            out=out,
            config1=c1,
            config2=c2,
            in_fmt=self._in_fmt,
            plan=plan,
            path=self._packing(int(xq.shape[0]), int(xq.shape[1])),
            w_round=getattr(inputs, "w_round", False),
        )
        return out
