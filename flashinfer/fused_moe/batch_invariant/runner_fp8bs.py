# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeekFp8 x DeepSeekFp8 pair of the batch-invariant fp8-block MoE backend of the unified MoE layer
(``BatchInvariantFp8BlockConfig`` under ``QuantConfig(DeepSeekFp8, DeepSeekFp8)`` -> ``BatchInvariantFp8BlockRunner`` ->
``BatchInvariantDeepSeekFp8Runner``, the pair's runner): the DeepSeek-style block-scale classes ``moe.e4m3bs.<act>`` (128-wide
blocks) and ``moe.e4m3bs32.<act>`` (32-wide) on SM100 / SM103 / SM107.

The surface is TRTLLM-Gen's block-fp8 one under the DeepSeek pair: e4m3 activations with their fp32 block scales
(``hidden_states_scale [H / block, T]``, the one layout ``TrtllmFp8BlockConfig.prepare_activations`` hands them in, taken as
given and never transposed; rows and scales 16-byte aligned) or the raw bf16 rows with no scale (the class quantizes them in
one launch with its permutation plan), the weights prepared from the experts per block (or FlashInfer's DeepSeek weight
view taken as given), every routing input mode and every typed activation of the layer through the shared flow of
:class:`.runner.BatchInvariantMoeRunnerBase` (the pack, the fused route + plan launch, the launch). The class is the prepared
weights' block size, read off the view at the pack; the view names its pair (``quant``) and a view without it is declined
by the layer's candidate filter. The tactics are the ranked block-scale record pairs in both requantization spellings (any
tactic gives the same bits on every pack form); the combine is the slot-order fp32 fold (``do_finalize=True`` only); an
unsupported configuration raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` instead of falling back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ...batch_invariant.support import unsupported
from ..api import MoEActivationPack, MoEWeightPack
from .api_fp8bs import MOE_FP8BS_ARCHS, VIEW_PAIR
from ...batch_invariant.support import LazyNames
from .runner import _TACTIC_FIELDS, _configs_from_tactic, _tactic
from .runner_fp8_block import PAIR_DEEPSEEK_FP8, BatchInvariantFp8BlockRunner
from .spec_fp8bs import BLOCK as QBLOCK
from .spec_fp8bs import CLASS, CLASS_OF


def _load_names() -> Dict[str, Any]:
    from .api_fp8bs import (
        MoEWeightsE4M3BS,
        _fused_moe_fp8bs_run,
        _gather_form,
        _quantize_rows,
        check_aligned,
        check_shape_fp8bs,
        scale_mn_view,
    )
    from .blocks.epilogue_fp8bs import GEMM2_CHAIN
    from .schedule import TOP_K, active_experts, rows_per_expert
    from .schedule_fp8bs import fused_quant_default, gemm1_chain, rank_grouped_fp8bs

    return dict(
        MoEWeightsE4M3BS=MoEWeightsE4M3BS,
        check_aligned=check_aligned,
        check_shape_fp8bs=check_shape_fp8bs,
        fused_moe_fp8bs_run=_fused_moe_fp8bs_run,
        gather_form=_gather_form,
        quantize_rows=_quantize_rows,
        scale_mn_view=scale_mn_view,
        GEMM2_CHAIN=GEMM2_CHAIN,
        TOP_K=TOP_K,
        active_experts=active_experts,
        rows_per_expert=rows_per_expert,
        fused_quant_default=fused_quant_default,
        gemm1_chain=gemm1_chain,
        rank_grouped_fp8bs=rank_grouped_fp8bs,
    )


_names = LazyNames(_load_names)

#: Positions of the packed inputs: the e4m3 activations, their block scales, the expert ids, the routing weights, the two
#: quantized weights with their block scales and the output.
_X, _SX, _IDS, _W, _W1, _W1SF, _W2, _W2SF, _OUT = range(9)


def _tactic_bs(c1, c2, fused: bool) -> tuple:
    """A pair of records (each with its swapped flag: the grouped or the swapped body) and the requantization spelling as a flat
    tuple of scalars (the autotuner's cache round-trips every knob the records carry; the swizzle and the grid are the class's
    constants)."""
    return _tactic(c1, c2) + (int(bool(fused)),)


def _configs_from_tactic_bs(t: tuple):
    """The two records and the requantization spelling of a tactic; a tuple of another length (another class's encoding in
    the autotuner's cache) is refused."""
    n = 2 * _TACTIC_FIELDS
    if len(t) != n + 1:
        raise ValueError(
            f"{CLASS}: a tactic of {len(t)} scalars; the pair's tactics carry {n + 1} (two records and the requantization "
            "spelling)"
        )
    c1, c2 = _configs_from_tactic(tuple(t[:n]))
    return c1, c2, bool(t[n])


class BatchInvariantDeepSeekFp8Runner(BatchInvariantFp8BlockRunner):
    """The fp8-block backend's runner for the DeepSeekFp8 x DeepSeekFp8 pair: the batch-invariant block-scale fp8 MoE
    classes."""

    supported_quant_variants = (PAIR_DEEPSEEK_FP8,)

    CLASS = CLASS
    ARCHS = MOE_FP8BS_ARCHS
    X_DTYPE = torch.float8_e4m3fn
    ROW_ALIGN = QBLOCK
    X_RULE = "an e4m3 [num_tokens, hidden_size] with its fp32 block scales (the DeepSeek pair)"
    ROW_RULE = f"a positive multiple of {QBLOCK} (whole smem stages of scale blocks of the intermediate and of both weights)"
    SCALE_RULE = (
        f"the fp32 block scales [hidden_size / {QBLOCK}, num_tokens] of the activations (/ 32 for the 32-wide class): the "
        "DeepSeek layout, taken as given"
    )
    TACTIC_TAG = (
        CLASS + "-v1"
    )  # the block's class joins the cache key at the pack (`_cache_key_extras`)
    N_ACT_TENSORS = 2
    N_WEIGHT_TENSORS = 4

    def __init__(self, config, device) -> None:
        super().__init__(config, device)
        self._block: int = QBLOCK
        self._fused: bool = False

    # ---- support ---------------------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        # the pack's layout flag first (a rule of the configuration alone, before the device is consulted): the fp32 block
        # scales have one layout, so the flag names a layout the class does not take
        if self.config.quant.swizzled_scale_factors is True:
            raise unsupported(
                self.CLASS,
                "QuantConfig(swizzled_scale_factors=True)",
                "the [hidden_size / block, num_tokens] fp32 block scales of the DeepSeek pair (swizzled_scale_factors "
                "unset or False)",
            )
        super()._check_support()

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def accepts(self, act: MoEActivationPack, weights: MoEWeightPack) -> bool:
        """The layer's candidate filter: the shared rule (a view under the backend's key, the input geometry) and the view
        naming this pair (``prepare_weights`` / ``from_trtllm_view`` write ``quant``); a view without the key or of another
        pair is declined, never read as this pair's (the runner driven directly refuses it in ``pack_inputs``, naming the
        key)."""
        if not super().accepts(act, weights):
            return False
        return weights.get_view(self.backend_key).get("quant") == VIEW_PAIR

    def _accepts_shape(self, act: MoEActivationPack) -> bool:
        x = act.hidden_states_q
        return (
            x.ndim == 2
            and x.shape[1] % QBLOCK == 0
            and (
                (x.dtype == torch.float8_e4m3fn and act.hidden_states_scale is not None)
                or (x.dtype == torch.bfloat16 and act.hidden_states_scale is None)
            )
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        self._validate_pack_contract(act)
        # the class is the prepared weights' block size: the activation scales are checked against it in `_act_tensors`
        self._block = self._weights_from_view(weights.get_view(self.backend_key)).block
        return super().pack_inputs(act, weights)

    def _act_tensors(self, act: MoEActivationPack) -> List[torch.Tensor]:
        """The pair's activation pack: the e4m3 rows and their fp32 block scales in the kernel's ``[H / block, T]`` layout
        (the DeepSeek layout, taken as given: a ``[T, H / block]`` or any other shape is refused, never transposed; a strided
        tensor of the layout is copied once, a placement; the empty pack takes any empty scale tensor), both 16-byte
        aligned, and the class's shape rule on the configuration at this hidden size; or the raw bf16 rows with no scale
        (the class quantizes them in one launch with its permutation plan: the pack contract's raw-bf16 encoding of a
        DeepSeek fp8 backend that quantizes internally), packed with an empty scale."""
        check_aligned, check_shape_fp8bs, scale_mn_view = (
            _names.check_aligned,
            _names.check_shape_fp8bs,
            _names.scale_mn_view,
        )
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype not in (self.X_DTYPE, torch.bfloat16):
            raise unsupported(
                self.CLASS,
                f"hidden_states_q {x.dtype} of rank {x.ndim}",
                self.X_RULE + ", or the bf16 rows with no scale",
            )
        T, H = x.shape
        if H % QBLOCK:
            raise unsupported(self.CLASS, f"hidden size {H}", self.ROW_RULE)
        cfg = self.config
        check_shape_fp8bs(
            H,
            cfg.experts.intermediate_size,
            cfg.routing.num_experts,
            cfg.routing.top_k,
            self._block,
        )
        x = x.contiguous()
        if x.dtype == torch.bfloat16:
            if act.hidden_states_scale is not None:
                raise unsupported(
                    self.CLASS,
                    "bf16 hidden_states_q with a hidden_states_scale",
                    "bf16 rows with no scale (the class quantizes them), or the e4m3 pair",
                )
            check_aligned(self._block, x, None)
            return [x, x.new_empty(0, dtype=torch.float32)]
        if act.hidden_states_scale is None:
            raise unsupported(self.CLASS, "no hidden_states_scale", self.SCALE_RULE)
        sx = (
            scale_mn_view(
                act.hidden_states_scale,
                H // self._block,
                T,
                "hidden_states_scale",
                self._block,
            )
            .t()
            .contiguous()
        )
        check_aligned(self._block, x, sx)
        return [x, sx]

    # ---- the weight view -------------------------------------------------------------------------------------------
    def _weights_from_view(self, view):
        return _names.MoEWeightsE4M3BS.from_view(view)

    def _weight_tensors(self, w) -> List[torch.Tensor]:
        return [w.w1, w.w1_sf, w.w2, w.w2_sf]

    def _weights_from_tensors(self, a, tensors: List[torch.Tensor]):
        w1, w1_sf, w2, w2_sf = tensors
        self._block = _names.MoEWeightsE4M3BS.block_of(w1, w1_sf)
        return _names.MoEWeightsE4M3BS(
            w1, w1_sf, w2, w2_sf, a.gated, a.act, a.params, self._block
        )

    # ---- tactics ---------------------------------------------------------------------------------------------------
    def _ranked(self, T: int, fused: Optional[bool] = None) -> Tuple[list, list]:
        """The ranked records of both GEMMs over both record spaces at ``T`` tokens for the requantization spelling ``fused``
        (the default spelling's at this token count when None), memoised (`schedule_fp8bs.rank_grouped_fp8bs`)."""
        cfg = self.config
        E, k, I = (
            cfg.routing.num_experts,
            cfg.routing.top_k,
            cfg.experts.intermediate_size,
        )
        H = self._hidden
        if fused is None:
            fused = self._fused_default(T)
        rows, n_act = _names.rows_per_expert(T, k, E), _names.active_experts(T, k, E)
        a = self._activation()
        sm_count, family = self._part if self._part is not None else (None, None)
        n1 = 2 * I if a.gated else I
        return (
            _names.rank_grouped_fp8bs(
                rows,
                n1,
                H,
                E,
                _names.gemm1_chain(a.act, a.params, a.gated, fused),
                sm_count,
                family,
                _names.TOP_K,
                n_active=n_act,
                block=self._block,
            ),
            _names.rank_grouped_fp8bs(
                rows,
                H,
                I,
                E,
                _names.GEMM2_CHAIN,
                sm_count,
                family,
                _names.TOP_K,
                n_active=n_act,
                block=self._block,
            ),
        )

    def _fused_default(self, T: int) -> bool:
        cfg = self.config
        return _names.fused_quant_default(
            T,
            cfg.routing.top_k,
            cfg.routing.num_experts,
            self._block,
            gated=self._activation().gated,
        )

    def _default(self, T: int):
        """``(c1, c2, fused)``: the dispatcher's records and spelling at ``T`` tokens, memoised per (token count, hidden
        size, block) with the runner's picks."""
        memo = self._memo()
        key = (T, self._hidden, self._block)
        p = memo.get(key)
        if p is None:
            fused = self._fused_default(T)
            r1, r2 = self._ranked(T, fused)
            p = memo[key] = (r1[0], r2[0], fused)
        return p

    def _pick(self, T: int) -> Tuple[Any, Any]:
        c1, c2, self._fused = self._default(T)
        return c1, c2

    def _configs_of_tactic(self, tactic: Any) -> Tuple[Any, Any]:
        c1, c2, self._fused = _configs_from_tactic_bs(tuple(tactic))
        return c1, c2

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile) -> List[Any]:
        """The ranked record pairs of both requantization spellings, the dispatcher's spelling first (every tactic computes
        the class)."""
        self._require_built()
        self._hidden = int(inputs[_X].shape[1])
        self._block = _names.MoEWeightsE4M3BS.block_of(inputs[_W1], inputs[_W1SF])
        T = int(inputs[_X].shape[0])
        default_fused = self._fused_default(T)
        out: List[Any] = []
        # both requantization spellings at the 128-wide class; the 32-wide class has the standalone pass only
        spellings = (
            (default_fused, not default_fused) if self._block == QBLOCK else (False,)
        )
        for fused in spellings:
            r1, r2 = self._ranked(T, fused)
            out += [_tactic_bs(c1, c2, fused) for c1 in r1 for c2 in r2]
        return out

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (CLASS_OF[self._block] + "-v1",)

    # ---- launch ----------------------------------------------------------------------------------------------------
    def _launch(self, x, ids, wts, w, out, c1, c2, plan, inputs):
        """The op on the validated pack: the bf16 rows quantized by the class first (with the plan in one launch when the
        plan is the op's), then the launch with the records and the requantization spelling of the pick or the tactic."""
        xq, sx = x
        T, H = xq.shape
        if T == 0:
            return out  # the empty output: no plan, no quantizer, no record
        if xq.dtype == torch.bfloat16:
            xq, sx, plan = _names.quantize_rows(xq, ids, w, plan)
        fused = self._fused
        _names.fused_moe_fp8bs_run(
            xq, sx, ids, wts, w, out, c1, c2, fused, _names.gather_form(c1, fused),
            T, H, ids.shape[1], w.num_experts, w.intermediate_size,
            plan=plan, w_round=getattr(inputs, "w_round", False),
        )  # fmt: skip
        return out


__all__ = ["BatchInvariantDeepSeekFp8Runner"]
