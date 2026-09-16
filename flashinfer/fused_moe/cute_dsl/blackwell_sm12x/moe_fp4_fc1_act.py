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
"""NVFP4 token-packed FC1 activation entry and tactic selection."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.common import DSLUserCodeError

from ....autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from ....tllm_enums import DEFAULT_SITU_BETA as SITU_BETA
from ....tllm_enums import DEFAULT_SITU_LINEAR_BETA as SITU_LINEAR_BETA
from ....tllm_enums import ActivationType
from ....utils import ceil_div
from ._moe_utils.heuristic import select_fc1_act_tile
from ._moe_utils.moe_epilogue import EPI_CONFIGS, EpiMethod
from ._moe_utils.moe_kernel_builder import Sm12xGemmConfig, dsl_targets_sm12x
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset
from .kernel_moe_fp4_fc1_act import (
    CuteDslSm120MoeNvfp4Fc1Act,
    is_swapab,
    make_args,
    make_cfg,
)

PLAIN_TILE_K = 128


def epi_tactics(tile):
    candidates = (EpiMethod.DIRECT_STG,) if is_swapab(tile) else (EpiMethod.R2G_WG,)
    return tuple(epi for epi in candidates if EPI_CONFIGS[epi].supports_tile(tile))


def resolve_stage(tile, epi):
    return Sm12xGemmConfig.max_ab_stage(
        functools.partial(make_cfg, epi=epi, activation=ActivationType.Swiglu),
        tuple(tile),
    )


def resolve_stage_store(tile):
    best = None
    for epi in epi_tactics(tile):
        try:
            stage = resolve_stage(tile, epi)
        except (AssertionError, ValueError, DSLUserCodeError):
            continue
        if best is None or stage > best[0]:
            best = (stage, epi)
    if best is None:
        raise ValueError(f"no store method fits tile {tuple(tile)}")
    return best


class CuteDslSm120GroupedNvfp4Fc1ActOp:
    OUT_DTYPES = (torch.bfloat16,)
    TILES = ((128, 128), (64, 128), (32, 128), (8, 128))
    ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situ)

    def __init__(
        self,
        *,
        n: int,
        k: int,
        tile,
        out_dtype: torch.dtype,
        activation: ActivationType,
        epi=None,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    ):
        if not self.can_implement(
            n=n, k=k, tile=tile, out_dtype=out_dtype, activation=activation, epi=epi
        ):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} "
                f"tile={tuple(tile)} epi={epi} activation={activation} "
                f"out_dtype={out_dtype}"
            )
        self.n, self.k = n, k
        self.tile, self.out_dtype = tuple(tile), out_dtype
        self.activation = activation
        self.situ_beta, self.situ_linear_beta = (situ_beta, situ_linear_beta)
        if epi is None:
            self.ab_stage, self.epi = resolve_stage_store(self.tile)
        else:
            self.ab_stage, self.epi = resolve_stage(self.tile, epi), epi
        self.tactic = (self.tile[0], self.tile[1], self.epi)
        self.cfg = make_cfg(
            self.tile,
            self.ab_stage,
            epi=self.epi,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedNvfp4Fc1ActOp.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (bm, bn) in CuteDslSm120GroupedNvfp4Fc1ActOp.TILES and bk == PLAIN_TILE_K

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return n > 0 and k > 0 and n % tile[1] == 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(cls, tile, epi=None, activation=ActivationType.Swiglu) -> bool:
        if not dsl_targets_sm12x():
            return False
        if epi is not None and (
            epi not in epi_tactics(tile) or not EPI_CONFIGS[epi].supports_tile(tile)
        ):
            return False
        try:
            if epi is None:
                stage, epi = resolve_stage_store(tuple(tile))
            else:
                stage = resolve_stage(tuple(tile), epi)
            CuteDslSm120MoeNvfp4Fc1Act(
                make_cfg(
                    tuple(tile),
                    stage,
                    epi=epi,
                    activation=activation,
                ),
                1,
            )
        except (AssertionError, ValueError, DSLUserCodeError):
            return False
        return True

    @classmethod
    def can_implement(
        cls,
        *,
        n: int,
        k: int,
        tile,
        out_dtype,
        activation=ActivationType.Swiglu,
        epi=None,
    ) -> bool:
        candidates = epi_tactics(tile) if epi is None else (epi,)
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and activation in cls.ACTIVATIONS
            and cls.is_valid_alignment(n, k, tile)
            and any(
                candidate in epi_tactics(tile)
                and EPI_CONFIGS[candidate].can_implement(
                    tile, n, cutlass.BFloat16.width
                )
                and cls.is_constructible(tile, candidate, activation)
                for candidate in candidates
            )
        )

    def build(self, grid_x: int) -> CuteDslSm120MoeNvfp4Fc1Act:
        return CuteDslSm120MoeNvfp4Fc1Act(self.cfg, grid_x)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedNvfp4Fc1ActOp.TILES
    for epi in epi_tactics((bm, bn, PLAIN_TILE_K))
    if CuteDslSm120GroupedNvfp4Fc1ActOp.is_constructible((bm, bn, PLAIN_TILE_K), epi)
)


def split_tactic(tactic):
    bm, bn, epi = tactic
    return (bm, bn, PLAIN_TILE_K), epi


@functools.lru_cache(maxsize=None)
def _op(
    n: int,
    k: int,
    tile,
    out_dtype,
    activation,
    epi=None,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
):
    return CuteDslSm120GroupedNvfp4Fc1ActOp(
        n=n,
        k=k,
        tile=tile,
        out_dtype=out_dtype,
        activation=activation,
        epi=epi,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op, grid_x: int, sm_version: str):
    key = (
        op.tactic,
        op.activation,
        op.out_dtype,
        op.situ_beta,
        op.situ_linear_beta,
        grid_x,
        sm_version,
    )
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(*, total_rows: int, n: int, k: int, num_experts: int, num_sms: int):
    return select_fc1_act_tile(
        total_rows=total_rows,
        n=n,
        num_experts=num_experts,
        num_sms=num_sms,
        tiles=CuteDslSm120GroupedNvfp4Fc1ActOp.TILES,
        gran_k=PLAIN_TILE_K,
    )


def _validate_inputs(a_q, a_scale, b_q, b_scale, m_indptr, up_scale, gate_scale):
    tensors = (a_q, a_scale, b_q, b_scale, m_indptr, up_scale, gate_scale)
    if any(t.device != a_q.device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if a_q.dtype != torch.uint8 or b_q.dtype != torch.uint8:
        raise TypeError("a_q and b_q must be uint8-packed FP4")
    if a_q.ndim != 2 or b_q.ndim != 3:
        raise ValueError("a_q must be 2D and b_q must be 3D")
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("a_scale and b_scale must be E4M3 bytes from fp4_quantize")
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    e, n2 = int(b_q.shape[0]), int(b_q.shape[1])
    if n2 % 2 != 0:
        raise ValueError("b_q projection extent must be even")
    for name, scale in (("up_scale", up_scale), ("gate_scale", gate_scale)):
        if scale.dtype != torch.float32 or tuple(scale.shape) != (e,):
            raise TypeError(f"{name} must be float32[num_experts]")
    if int(m_indptr.shape[0]) != e + 1:
        raise ValueError("m_indptr length must equal num_experts + 1")
    if int(a_q.shape[1]) != int(b_q.shape[2]):
        raise ValueError("a_q and b_q K extents must match")
    m, packed_k = int(a_q.shape[0]), int(a_q.shape[1])
    k = packed_k * 2
    want_a_scale = (compute_padded_offset(m, e, 128), ceil_div(k, 16))
    want_b_scale = (e * n2, ceil_div(k, 16))
    if tuple(a_scale.shape) != want_a_scale:
        raise ValueError(f"a_scale shape must be {want_a_scale}")
    if tuple(b_scale.shape) != want_b_scale:
        raise ValueError(f"b_scale shape must be {want_b_scale}")


def cute_dsl_sm12x_fc1_act_nvfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    up_scale,
    gate_scale,
    out_dtype: torch.dtype = torch.bfloat16,
    tile=None,
    epi=None,
    tune=None,
    *,
    activation: ActivationType = ActivationType.Swiglu,
    situ_beta: float = SITU_BETA,
    situ_linear_beta: float = SITU_LINEAR_BETA,
) -> torch.Tensor:
    _validate_inputs(a_q, a_scale, b_q, b_scale, m_indptr, up_scale, gate_scale)
    n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x = props.multi_processor_count
    sm_version = f"sm_{props.major}{props.minor}"
    inputs = (a_q, a_scale, b_q, b_scale, m_indptr, up_scale, gate_scale)
    if tile is None and tune is not False:
        chosen = None
        if MOE_AUTOTUNE_ENABLED():
            with autotune():
                _, tactic = AutoTuner.get().choose_one(
                    "cute_dsl_sm12x_fc1_act_nvfp4",
                    [_Fc1ActRunner(out_dtype, activation, situ_beta, situ_linear_beta)],
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
            num_sms=grid_x,
        )
    op = _op(
        n,
        k,
        tuple(tile),
        out_dtype,
        activation,
        epi,
        situ_beta,
        situ_linear_beta,
    )
    out = torch.zeros(int(a_q.shape[0]), n, dtype=out_dtype, device=a_q.device)
    args = make_args(
        a_q,
        a_scale,
        b_q,
        b_scale,
        out,
        m_indptr,
        up_scale,
        gate_scale,
        op.cfg.epi,
        op.cfg.TILE,
    )
    compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
    return out


class _Fc1ActRunner(TunableRunner):
    def __init__(
        self,
        out_dtype,
        activation,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    ):
        self.out_dtype = out_dtype
        self._out = None
        self.activation = activation
        self.situ_beta, self.situ_linear_beta = (situ_beta, situ_linear_beta)

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
        _, _, b_q, _, _, _, _ = inputs
        n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
        return [
            tactic
            for tactic in TACTICS
            if CuteDslSm120GroupedNvfp4Fc1ActOp.can_implement(
                n=n,
                k=k,
                tile=split_tactic(tactic)[0],
                out_dtype=self.out_dtype,
                activation=self.activation,
                epi=split_tactic(tactic)[1],
            )
        ]

    def alloc_out(self, inputs):
        a_q, _, b_q, _, _, _, _ = inputs
        return torch.zeros(
            int(a_q.shape[0]),
            int(b_q.shape[1]) // 2,
            dtype=self.out_dtype,
            device=a_q.device,
        )

    def launch(self, inputs, out, tactic):
        a_q, a_scale, b_q, b_scale, m_indptr, up_scale, gate_scale = inputs
        n, k = int(b_q.shape[1]) // 2, int(b_q.shape[2]) * 2
        props = torch.cuda.get_device_properties(a_q.device)
        grid_x = props.multi_processor_count
        epi = None
        if tactic is None:
            tile = select_tile(
                total_rows=int(a_q.shape[0]),
                n=n,
                k=k,
                num_experts=int(b_q.shape[0]),
                num_sms=grid_x,
            )
        else:
            tile, epi = split_tactic(tactic)
        op = _op(
            n,
            k,
            tuple(tile),
            self.out_dtype,
            self.activation,
            epi,
            self.situ_beta,
            self.situ_linear_beta,
        )
        args = make_args(
            a_q,
            a_scale,
            b_q,
            b_scale,
            out,
            m_indptr,
            up_scale,
            gate_scale,
            op.cfg.epi,
            op.cfg.TILE,
        )
        sm_version = f"sm_{props.major}{props.minor}"
        compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)

    def get_cache_key_extras(self, inputs):
        a_q, _, b_q, _, _, _, _ = inputs
        return (
            int(a_q.shape[0]),
            int(b_q.shape[0]),
            int(b_q.shape[1]),
            int(b_q.shape[2]) * 2,
            str(self.out_dtype),
            self.activation,
            self.situ_beta,
            self.situ_linear_beta,
            torch.cuda.get_device_capability(),
        )


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
