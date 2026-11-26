# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(MXFP8, MXFP8)`` -> ``BatchInvariantFp8BlockRunner`` ->
``BatchInvariantMxfp8Runner``, the pair's runner).

The runner serves the classes ``moe.mxfp8.<act>`` (MXFP8 experts and activations, every typed activation of the layer)
on SM100 / SM103 / SM107, through the dense MXFP8 class of the part (the SM107 instruction folds 64 elements per step, the
SM100 / SM103 one 32: a dimension of the specification). It takes the layer's canonical MXFP8 activation pack (e4m3
``hidden_states_q [T, H]`` with the token-major UE8M0 ``hidden_states_scale [T, H / 32]``, as ``TrtllmFp8BlockConfig`` /
``CutlassMxfp8Config`` / this config's ``prepare_activations`` produce it: the linear layout alone, taken as given and
never reshaped -- a swizzled buffer, ``QuantConfig(swizzled_scale_factors=True)`` and a per-token scale are refused) and
every routing input mode through the shared flow of :class:`.runner.BatchInvariantMoeRunnerBase`. The combine is the
slot-order fp32 fold (``use_fused_finalize`` is ignored, ``do_finalize=True`` only), the tactics are the ranked
grouped-GEMM records of the dense MX heuristic over both record spaces -- the padded body's and the swapped decode body's
(the weights on the MMA's M side, the group's tokens a narrow tile; `schedule_mxfp8`) -- any tactic gives the same bits,
and the packing of the expanded rows follows the records (the dense packing when either is a swapped record, else the
padded one: a placement; the routing-and-plan launch takes the first picks' form at the token count, and a tactic of the
other packing recomputes the plan). An unsupported configuration raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ...batch_invariant.support import unsupported
from ..api import MoEActivationPack, MoEWeightPack
from .api_mxfp8 import MOE_MXFP8_ARCHS, VIEW_PAIR
from ...batch_invariant.support import LazyNames
from .runner_fp8_block import PAIR_MXFP8, BatchInvariantFp8BlockRunner
from .spec_mxfp8 import BLOCK, CLASS


def _load_names() -> Dict[str, Any]:
    from .api_mxfp8 import (
        GROUP_ALIGN,
        MoEWeightsMXFP8,
        _fused_moe_mxfp8_run,
        _mx_forms,
        check_shape_mxfp8,
        decode_form_default,
    )
    from .blocks.epilogue_mx import MxChain
    from .schedule import TOP_K
    from .schedule_mxfp8 import GEMM2_CHAIN, packing_for, ranked_mx

    return dict(
        GROUP_ALIGN=GROUP_ALIGN,
        MoEWeightsMXFP8=MoEWeightsMXFP8,
        check_shape_mxfp8=check_shape_mxfp8,
        decode_form_default=decode_form_default,
        fused_moe_mxfp8_run=_fused_moe_mxfp8_run,
        mx_forms=_mx_forms,
        MxChain=MxChain,
        TOP_K=TOP_K,
        GEMM2_CHAIN=GEMM2_CHAIN,
        packing_for=packing_for,
        ranked_mx=ranked_mx,
    )


_names = LazyNames(_load_names)


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

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        # the pack's layout flag first (a rule of the configuration alone, before the device is consulted): the pack's
        # scales are the token-major rows, so the flag names a layout the class does not take
        if self.config.quant.swizzled_scale_factors is True:
            raise unsupported(
                self.CLASS,
                "QuantConfig(swizzled_scale_factors=True)",
                "the token-major linear scale layout of the canonical MXFP8 pack (swizzled_scale_factors unset or False)",
            )
        super()._check_support()

    def _build(self) -> None:
        super()._build()
        from .api_mxfp8 import mx_class_of

        self._in_fmt = mx_class_of(self.device)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def accepts(self, act: MoEActivationPack, weights: MoEWeightPack) -> bool:
        """The layer's candidate filter: the shared rule (a view under the backend's key, the input geometry) and the view
        naming this pair (``prepare_weights`` writes ``quant``); a view without the key or of another pair is declined,
        never read as this pair's (the runner driven directly refuses it in ``pack_inputs``, naming the key)."""
        if not super().accepts(act, weights):
            return False
        return weights.get_view(self.backend_key).get("quant") == VIEW_PAIR

    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return (
            x.ndim == 2
            and x.dtype == torch.float8_e4m3fn
            and x.shape[1] % BLOCK == 0
            and act.hidden_states_scale is not None
        )

    def _act_tensors(self, act: MoEActivationPack) -> List[torch.Tensor]:
        """The pair's activation pack: the e4m3 rows and their token-major UE8M0 scales (uint8 or ``float8_e8m0fnu``) in
        the linear ``[T, H / 32]`` shape, taken as given (a flattened or swizzled buffer is refused, never reshaped into
        rows; the empty pack takes any empty scale tensor), both 16-byte aligned; and the class's shape rule on the
        configuration at this hidden size."""
        x = act.hidden_states_q
        if x.dtype != self.X_DTYPE or x.ndim != 2:
            raise unsupported(
                self.CLASS, f"hidden_states_q {x.dtype} of rank {x.ndim}", self.X_RULE
            )
        sf = act.hidden_states_scale
        if sf is None:
            raise unsupported(self.CLASS, "no hidden_states_scale", self.SCALE_RULE)
        T, H = x.shape
        if sf.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise unsupported(
                self.CLASS,
                f"hidden_states_scale {sf.dtype}",
                "uint8 or UE8M0 scale bytes",
            )
        if H % BLOCK:
            raise unsupported(self.CLASS, f"hidden size {H}", self.ROW_RULE)
        sf = sf.view(torch.uint8)
        if T == 0 and sf.numel() == 0:
            sf = sf.reshape(
                0, H // BLOCK
            )  # the empty pack: no scale byte, whatever its shape
        if tuple(sf.shape) != (T, H // BLOCK):
            raise unsupported(
                self.CLASS,
                f"hidden_states_scale of shape {tuple(sf.shape)} for {T} rows of {H}",
                f"the token-major [{T}, {H // BLOCK}] UE8M0 scales (the linear layout of the canonical MXFP8 pack; a "
                "swizzled or flattened buffer is not reshaped into rows)",
            )
        x, sf = x.contiguous(), sf.contiguous()
        if x.data_ptr() % 16 or sf.data_ptr() % 16:
            raise unsupported(
                self.CLASS,
                "hidden_states_q or hidden_states_scale at a storage offset that is not a multiple of 16 bytes",
                "16-byte aligned rows and scale bytes (the kernels' vector loads and TMA boxes)",
            )
        cfg = self.config
        _names.check_shape_mxfp8(
            H, cfg.experts.intermediate_size, cfg.routing.num_experts, cfg.routing.top_k
        )
        return [x, sf]

    def _plan_in_kernel(self, T: int, H: int) -> bool:
        """The decode form's rule at this token count (`api_mxfp8.decode_form_default` at the first picks): GEMM1's gather
        record then builds the plan from the ids and stores the counts and the row map for GEMM2 and the combine (a tactic of
        another kind of record falls back to the plan computed by the op: the same integers)."""
        k, E_local = self.config.routing.top_k, self._local_experts()[1]
        r1, r2 = self._ranked(T, H)
        return bool(
            r1
            and r2
            and _names.decode_form_default(
                T, k, E_local, H, r1[0], r2[0], path=self._packing(T, H)
            )
        )

    def _plan_form(self, T: int, H: int) -> dict:
        """The fused route + plan launch in the form of the packing the ranked records run on at this token count: the
        dual plan (the dense rows with the padded scale positions) or the padded plan (whole 128-row atoms per group)."""
        if self._packing(T, H) == "dense":
            return dict(dual=True, pad_align=_names.GROUP_ALIGN)
        return dict(align=_names.GROUP_ALIGN)

    # ---- the weight view -------------------------------------------------------------------------------------------
    def _weights_from_view(self, view):
        """The view as the kernels consume it; a view without this pair's tag is refused by name (``from_view``)."""
        return _names.MoEWeightsMXFP8.from_view(view)

    def _weight_tensors(self, w) -> List[torch.Tensor]:
        return [w.w1, w.w1_sf, w.w2, w.w2_sf]

    def _weights_from_tensors(self, a, tensors: List[torch.Tensor]):
        w1, w1_sf, w2, w2_sf = tensors
        return _names.MoEWeightsMXFP8(w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params)

    def _chain1(self, a):
        """GEMM1's chain for the dispatcher: the bf16-output chain of the layer's activation at the part's instruction
        class (the fused requantization's extra term is the same for every record)."""
        return _names.MxChain(
            "glu" if a.gated else "act",
            a.act,
            "bf16",
            a.params,
            self._in_fmt or "mxfp8",
        )

    def _chain2(self):
        return _names.GEMM2_CHAIN.with_class(self._in_fmt or "mxfp8")

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int, H: Optional[int] = None) -> Tuple[list, list]:
        """The ranked records of both GEMMs over both record spaces at ``T`` tokens (`schedule_mxfp8.ranked_mx`), trimmed
        to the packing the first picks run on: on the padded packing the swapped records cannot run, so they leave the
        lists. Memoised per (token count, hidden size) with the runner's picks (the plan form, the decode rule and the
        pick all read it at every call)."""
        H = H if H is not None else self._hidden
        memo = self._memo()
        key = ("ranked", T, H)
        r = memo.get(key)
        if r is not None:
            return r
        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        in_fmt = self._in_fmt or "mxfp8"
        r1, r2 = _names.ranked_mx(
            T, k, H, I, E, self._chain1(a), in_fmt, sm_count, family, _names.TOP_K
        )
        if _names.packing_for(r1[0], r2[0]) == "padded":
            r1 = [c for c in r1 if not c.swap_ab]
            r2 = [c for c in r2 if not c.swap_ab]
        memo[key] = (r1, r2)
        return r1, r2

    def _packing(self, T: int, H: int) -> str:
        """The packing of the expanded rows at ``T`` tokens: the first picks' (`schedule_mxfp8.packing_for`)."""
        r1, r2 = self._ranked(T, H)
        return _names.packing_for(r1[0], r2[0])

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (self._in_fmt or "",)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def _launch(self, x, ids, wts, w, out, c1, c2, plan, inputs):
        """The op on the validated pack with the records of the pick or the tactic: the packing is the records' own (a
        swapped record reads the dense packing, the padded body reads either); the layer keys a tuned winner by its token
        bucket, so a tactic may replay at a token count whose first picks pack the other way -- the plan packed in the
        first picks' form is kept where the records read it (the dual plan feeds both bodies) and dropped otherwise (the op
        computes its own: the same integers, the same bits); GEMM1's form (the gather record, the in-kernel plan) follows
        the op's rules at these records."""
        xq, xsf = x
        T, H = xq.shape
        if T == 0:
            return out  # no tokens: nothing to route, the (0, H) output as it stands
        k, E, I = ids.shape[1], w.num_experts, w.intermediate_size
        path = _names.packing_for(c1, c2)
        if plan is not None:
            if len(plan) == 5:
                path = "dense"
            elif path == "dense":
                plan = None
        c1, in_kernel = _names.mx_forms(c1, c2, T, k, E, H, path, plan is not None)
        _names.fused_moe_mxfp8_run(
            xq, xsf, ids, wts, w, out, c1, c2, self._in_fmt or "mxfp8", path, "fused", in_kernel, T, H, k, E, I,
            plan=plan, w_round=getattr(inputs, "w_round", False),
        )  # fmt: skip
        return out
