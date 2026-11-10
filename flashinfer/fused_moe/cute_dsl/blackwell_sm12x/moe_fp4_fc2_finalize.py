# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public entry for the fused fc2_down + finalize arm: tile choice, compile cache, and its gates."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.common import DSLUserCodeError

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from ....utils import ceil_div
from ._moe_utils.heuristic import select_fp4_fc2_finalize_tile
from ._moe_utils.moe_epilogue import (
    EPI_CONFIGS,
    EpiMethod,
)
from ._moe_utils.moe_kernel_builder import (
    Sm12xGemmConfig,
    dsl_targets_sm12x,
    get_moe_worklist_grid_size,
)
from ._moe_utils.sm12x_blockscaled_layout import (
    Sm120SfConfigNvfp4,
    compute_padded_offset,
)
from .kernel_moe_fp4_fc2_finalize import (
    SCATTER_EPIS,
    CuteDslSm120MoeNvfp4Fc2Finalize,
    CuteDslSm120MoeNvfp4Fc2FinalizeDecode,
    is_swapab,
    make_args,
    make_cfg,
    make_decode_cfg,
)

TILES = ((128, 128), (64, 128), (32, 128), (8, 128))
BK = 128
EPIS = SCATTER_EPIS
DEFAULT_EPI = EpiMethod.WG_SCATTER


def resolve_stage(tile, epi: EpiMethod = DEFAULT_EPI):
    if is_swapab(tile):
        return 4
    stage = Sm12xGemmConfig.max_ab_stage(
        functools.partial(make_cfg, epi=epi), tuple(tile)
    )
    return 3 if tile[0] == 128 else stage


def select_kernel(tile):
    if is_swapab(tile):
        return CuteDslSm120MoeNvfp4Fc2FinalizeDecode, make_decode_cfg
    return CuteDslSm120MoeNvfp4Fc2Finalize, make_cfg


class CuteDslSm120GroupedNvfp4Fc2FinalizeOp:
    TILES = TILES

    def __init__(
        self, n: int, k: int, tile, epi: EpiMethod = DEFAULT_EPI, enable_pdl=False
    ):
        if not self.can_implement(n=n, k=k, tile=tile, epi=epi):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} tile={tuple(tile)} "
                f"epi={epi}"
            )
        self.n, self.k, self.tile = n, k, tuple(tile)
        self.epi, self.enable_pdl = epi, enable_pdl
        self.ab_stage = resolve_stage(self.tile, epi)
        self.tactic = (self.tile[0], self.tile[1], epi)
        _, cfg_factory = select_kernel(self.tile)
        self.cfg = cfg_factory(self.tile, self.ab_stage, epi=epi, enable_pdl=enable_pdl)

    @classmethod
    def is_constructible(cls, tile, epi: EpiMethod = DEFAULT_EPI) -> bool:
        if not dsl_targets_sm12x():
            return False
        if epi not in EPIS or not EPI_CONFIGS[epi].supports_tile(tile):
            return False
        try:
            kernel_type, cfg_factory = select_kernel(tuple(tile))
            kernel_type(
                cfg_factory(tuple(tile), resolve_stage(tuple(tile), epi), epi=epi), 1
            )
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(
        cls, *, n: int, k: int, tile, epi: EpiMethod = DEFAULT_EPI
    ) -> bool:
        t = tuple(tile)
        if t[:2] not in cls.TILES or t[2] != BK or epi not in EPIS:
            return False
        k_alignment = (
            t[2]
            if t[0] == 8
            else Sm120SfConfigNvfp4.SF_VEC * Sm120SfConfigNvfp4.PACK_NSF
        )
        if n <= 0 or k <= 0 or n % t[1] != 0 or k % k_alignment != 0:
            return False
        return EPI_CONFIGS[epi].can_implement(
            t, n, cutlass.BFloat16.width
        ) and cls.is_constructible(t, epi)

    def build(self, grid_x: int):
        kernel_type, _ = select_kernel(self.tile)
        return kernel_type(self.cfg, grid_x)

    def get_grid_size(self, total_rows: int, num_experts: int, num_sms: int):
        if not is_swapab(self.tile):
            return num_sms
        return get_moe_worklist_grid_size(total_rows, num_experts, self.n, self.tile)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedNvfp4Fc2FinalizeOp.TILES
    for epi in EPIS
    if CuteDslSm120GroupedNvfp4Fc2FinalizeOp.is_constructible((bm, bn, BK), epi)
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, BK), epi


_COMPILED: dict = {}


def compiled_kernel(
    sample_args,
    *,
    op: CuteDslSm120GroupedNvfp4Fc2FinalizeOp,
    grid_x: int,
    sm_version: str,
):
    key = (op.tactic, op.enable_pdl, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    return select_fp4_fc2_finalize_tile(
        total_rows=total_rows, n=n, k=k, num_experts=num_experts, num_sms=num_sms
    )


def _prepare_worklist(op, inputs, out, offsets, worklist=None, workcount=None):
    if not is_swapab(op.tile):
        if worklist is not None or workcount is not None:
            raise ValueError("worklist is supported only by standard-grid FC2 decode")
        return None, None
    from . import moe_fp4_q0_route_triton as route

    return route.prepare_nvfp4_worklist(
        offsets,
        int(inputs[0].shape[0]),
        op.tile[0],
        worklist=worklist,
        workcount=workcount,
        tensors=(*inputs, out),
    )


def _validate_inputs(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    src_token,
    pair_scales,
    weight_global_scale,
    input_global_scale,
    num_tokens,
):
    tensors = (
        a_q,
        a_scale,
        b_q,
        b_scale,
        m_indptr,
        src_token,
        pair_scales,
        weight_global_scale,
        input_global_scale,
    )
    if any(t.device != a_q.device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if a_q.dtype != torch.uint8 or b_q.dtype != torch.uint8:
        raise TypeError("a_q and b_q must be uint8-packed FP4")
    if a_q.ndim != 2 or b_q.ndim != 3:
        raise ValueError("a_q must be 2D and b_q must be 3D")
    if a_scale.dtype != torch.uint8:
        raise TypeError("a_scale must be E4M3 bytes from fp4_quantize")
    if b_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("b_scale must be canonical E4M3 weight scales")
    if m_indptr.ndim != 1 or src_token.ndim != 1 or pair_scales.ndim != 1:
        raise ValueError("routing inputs must be 1D")
    m, packed_k = int(a_q.shape[0]), int(a_q.shape[1])
    e, n, b_packed_k = map(int, b_q.shape)
    if weight_global_scale.dtype != torch.float32 or tuple(
        weight_global_scale.shape
    ) != (e,):
        raise TypeError("weight_global_scale must be float32[num_experts]")
    if input_global_scale.dtype != torch.float32 or tuple(input_global_scale.shape) != (
        1,
    ):
        raise TypeError("input_global_scale must be float32[1]")
    if packed_k != b_packed_k:
        raise ValueError("a_q and b_q K extents must match")
    if int(m_indptr.shape[0]) != e + 1:
        raise ValueError("m_indptr length must equal num_experts + 1")
    if int(src_token.shape[0]) != m or int(pair_scales.shape[0]) != m:
        raise ValueError("routing input lengths must equal a_q rows")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    k = packed_k * 2
    if n % 128 or k % 64:
        raise ValueError(
            "canonical FC2 requires N divisible by 128 and K divisible by 64"
        )
    cols = ceil_div(k, 16)
    want_a_scale = (compute_padded_offset(m, e, 128), cols)
    want_b_scale = ((e, ceil_div(n, 128), ceil_div(cols, 4), 32, 16), (e, n, cols))
    if tuple(a_scale.shape) != want_a_scale:
        raise ValueError(f"a_scale shape must be {want_a_scale}")
    if tuple(b_scale.shape) not in want_b_scale:
        raise ValueError("b_scale must be a canonical 5D or 3D weight scale view")


def cute_dsl_sm12x_fc2_finalize_nvfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    src_token,
    pair_scales,
    weight_global_scale,
    input_global_scale,
    num_tokens: int,
    tile=None,
    epi: EpiMethod = DEFAULT_EPI,
    tune=None,
    enable_pdl: bool = False,
    *,
    out: torch.Tensor | None = None,
    worklist: torch.Tensor | None = None,
    workcount: torch.Tensor | None = None,
):
    assert not enable_pdl or out is not None, "PDL requires caller-owned out"
    _validate_inputs(
        a_q,
        a_scale,
        b_q,
        b_scale,
        m_indptr,
        src_token,
        pair_scales,
        weight_global_scale,
        input_global_scale,
        num_tokens,
    )
    with torch.cuda.device(a_q.device):
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        props = torch.cuda.get_device_properties(a_q.device)
        num_sms, sm_version = (
            props.multi_processor_count,
            f"sm_{props.major}{props.minor}",
        )
        inputs = (
            a_q,
            a_scale,
            b_q,
            b_scale,
            m_indptr,
            src_token,
            pair_scales,
            weight_global_scale,
            input_global_scale,
            num_tokens,
        )
        if tile is None and tune is not False:
            chosen = None
            if MOE_AUTOTUNE_ENABLED():
                with autotune():
                    _, tactic = AutoTuner.get().choose_one(
                        "cute_dsl_sm12x_fc2_finalize_nvfp4",
                        [_Fc2FinalizeRunner(enable_pdl)],
                        TuningConfig(),
                        list(inputs),
                    )
                if tactic != -1:
                    chosen = tactic
            if chosen is not None:
                tile, epi = split_tactic(chosen)
        if tile is None:
            tile = select_tile(
                total_rows=int(a_q.shape[0]),
                n=n,
                k=k,
                num_experts=int(b_q.shape[0]),
                num_sms=num_sms,
            )
        op = CuteDslSm120GroupedNvfp4Fc2FinalizeOp(n, k, tuple(tile), epi, enable_pdl)
        grid_x = op.get_grid_size(int(a_q.shape[0]), int(b_q.shape[0]), num_sms)
        if out is None:
            out = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=a_q.device)
        if (
            out.shape != (num_tokens, n)
            or out.dtype is not torch.bfloat16
            or out.device != a_q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out does not match the FC2 output contract")
        offsets = m_indptr.to(torch.int32)
        worklist, workcount = _prepare_worklist(
            op, inputs[:-1], out, offsets, worklist, workcount
        )
        args = make_args(
            a_q,
            a_scale,
            b_q,
            b_scale,
            out,
            src_token.to(torch.int32),
            pair_scales.to(torch.float32),
            offsets,
            weight_global_scale,
            input_global_scale,
            op.cfg.epi,
            op.cfg.TILE,
            worklist=worklist,
            workcount=workcount,
        )
        compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
        return out


class _Fc2FinalizeRunner(TunableRunner):
    def __init__(self, enable_pdl=False):
        self.out_dtype = torch.bfloat16
        self._out = None
        self.enable_pdl = enable_pdl

    def __hash__(self) -> int:
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile):
        return self.valid_tactics(inputs)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if self._out is None:
            self._out = self.alloc_out(inputs)
        if do_preparation:
            return self._out
        self.launch(inputs, self._out, None if tactic == -1 else tactic)
        return self._out

    def valid_tactics(self, inputs):
        (
            a_q,
            a_scale,
            b_q,
            b_scale,
            m_indptr,
            src_token,
            pair_scales,
            weight_global_scale,
            input_global_scale,
            num_tokens,
        ) = inputs
        n, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        return [
            t
            for t in TACTICS
            if CuteDslSm120GroupedNvfp4Fc2FinalizeOp.can_implement(
                n=n, k=k, tile=split_tactic(t)[0], epi=split_tactic(t)[1]
            )
        ]

    def alloc_out(self, inputs):
        (
            a_q,
            a_scale,
            b_q,
            b_scale,
            m_indptr,
            src_token,
            pair_scales,
            weight_global_scale,
            input_global_scale,
            num_tokens,
        ) = inputs
        n = int(b_q.shape[1])
        return torch.zeros(num_tokens, n, dtype=self.out_dtype, device=a_q.device)

    def launch(self, inputs, out, tactic):
        tile, epi = (None, DEFAULT_EPI) if tactic is None else split_tactic(tactic)
        cute_dsl_sm12x_fc2_finalize_nvfp4(
            *inputs,
            tile=tile,
            epi=epi,
            tune=False,
            enable_pdl=self.enable_pdl,
            out=out,
        )

    def get_cache_key_extras(self, inputs):
        (
            a_q,
            a_scale,
            b_q,
            b_scale,
            m_indptr,
            src_token,
            pair_scales,
            weight_global_scale,
            input_global_scale,
            num_tokens,
        ) = inputs
        return (
            int(a_q.shape[0]),
            int(b_q.shape[0]),
            int(b_q.shape[1]),
            int(b_q.shape[2]) * 2,
            int(num_tokens),
            self.enable_pdl,
            torch.cuda.get_device_capability(a_q.device),
        )


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
