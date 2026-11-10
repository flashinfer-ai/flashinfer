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
"""NVFP4 token-packed FC1 activation and Q1 entry."""

import functools
import math
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
from ....utils import round_up as align
from ._moe_utils.heuristic import select_fp4_fc1_act_q1_tile
from ._moe_utils.moe_epilogue import (
    EPI_CONFIGS,
    EpiMethod,
)
from ._moe_utils.moe_kernel_builder import (
    Sm12xGemmConfig,
    dsl_targets_sm12x,
    get_moe_worklist_grid_size,
)
from ._moe_utils.sm12x_blockscaled_layout import compute_padded_offset
from .kernel_moe_fp4_fc1_act_q1 import (
    CuteDslSm120MoeNvfp4Fc1ActQ1,
    CuteDslSm120MoeNvfp4Fc1ActQ1Decode,
    is_swapab,
    make_args,
    make_cfg,
    make_decode_cfg,
)

PLAIN_TILE_K = 128


def _uses_fused_gated_tile(tile, activation):
    return tuple(tile[:2]) == (8, 64) and activation is not ActivationType.Relu2


def select_kernel(tile, *, activation=ActivationType.Swiglu, k=None):
    if is_swapab(tile):
        if _uses_fused_gated_tile(tile, activation):
            return CuteDslSm120MoeNvfp4Fc1ActQ1Decode, functools.partial(
                make_decode_cfg, k=k
            )
        return CuteDslSm120MoeNvfp4Fc1ActQ1Decode, make_decode_cfg
    return CuteDslSm120MoeNvfp4Fc1ActQ1, make_cfg


def epi_tactics(tile):
    candidates = (
        (EpiMethod.WG_QUANT_SWAP,) if is_swapab(tile) else (EpiMethod.WG_S2R_QUANT,)
    )
    return tuple(epi for epi in candidates if EPI_CONFIGS[epi].supports_tile(tile))


def resolve_stage(tile, epi, activation=ActivationType.Swiglu, gather=False, *, k=None):
    _, cfg_factory = select_kernel(tile, activation=activation, k=k)
    if _uses_fused_gated_tile(tile, activation):
        stage = 4
        cfg_factory(tuple(tile), stage, epi=epi, activation=activation, gather=gather)
        return stage
    return Sm12xGemmConfig.max_ab_stage(
        functools.partial(cfg_factory, epi=epi, activation=activation, gather=gather),
        tuple(tile),
    )


def resolve_stage_store(
    tile, activation=ActivationType.Swiglu, gather=False, *, k=None
):
    best = None
    for epi in epi_tactics(tile):
        try:
            stage = resolve_stage(tile, epi, activation, gather, k=k)
        except (AssertionError, ValueError, DSLUserCodeError):
            continue
        if best is None or stage > best[0]:
            best = (stage, epi)
    if best is None:
        raise ValueError(f"no store method fits tile {tuple(tile)}")
    return best


def out_sf_shape(m: int, n: int, num_experts: int):
    return (
        compute_padded_offset(m, num_experts, 128),
        ceil_div(ceil_div(n, 16), 4) * 4,
    )


class CuteDslSm120GroupedNvfp4Fc1ActQ1Op:
    OUT_DTYPES = (torch.uint8,)
    TILES = (
        (128, 128),
        (64, 128),
        (32, 128),
        (8, 128),
        (128, 64),
        (64, 64),
        (32, 64),
        (8, 64),
    )
    ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situ, ActivationType.Relu2)

    def __init__(
        self,
        *,
        n: int,
        k: int,
        tile,
        out_dtype: torch.dtype,
        activation: ActivationType,
        epi=None,
        enable_pdl=False,
        swiglu_limit=None,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
        gather=False,
    ):
        if not self.can_implement(
            n=n,
            k=k,
            tile=tile,
            out_dtype=out_dtype,
            activation=activation,
            epi=epi,
            gather=gather,
        ):
            raise TypeError(
                f"{type(self).__name__}: unsupported n={n} k={k} "
                f"tile={tuple(tile)} epi={epi} activation={activation} "
                f"out_dtype={out_dtype}"
            )
        self.n, self.k = n, k
        self.tile, self.out_dtype = tuple(tile), out_dtype
        self.activation, self.enable_pdl = activation, enable_pdl
        self.gather = gather
        self.swiglu_limit = swiglu_limit
        self.situ_beta, self.situ_linear_beta = (situ_beta, situ_linear_beta)
        if epi is None:
            self.ab_stage, self.epi = resolve_stage_store(
                self.tile, activation, gather, k=k
            )
        else:
            self.ab_stage, self.epi = (
                resolve_stage(self.tile, epi, activation, gather, k=k),
                epi,
            )
        self.tactic = (self.tile[0], self.tile[1], self.epi)
        _, cfg_factory = select_kernel(self.tile, activation=activation, k=k)
        self.cfg = cfg_factory(
            self.tile,
            self.ab_stage,
            epi=self.epi,
            activation=activation,
            enable_pdl=enable_pdl,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            gather=gather,
        )

    @staticmethod
    def is_valid_dtypes(out_dtype) -> bool:
        return out_dtype in CuteDslSm120GroupedNvfp4Fc1ActQ1Op.OUT_DTYPES

    @staticmethod
    def is_valid_tile(tile) -> bool:
        bm, bn, bk = tile
        return (
            bm,
            bn,
        ) in CuteDslSm120GroupedNvfp4Fc1ActQ1Op.TILES and bk == PLAIN_TILE_K

    @staticmethod
    def is_valid_alignment(n: int, k: int, tile) -> bool:
        return n > 0 and k > 0 and n % tile[1] == 0 and k % tile[2] == 0

    @classmethod
    def is_constructible(
        cls, tile, epi=None, activation=ActivationType.Swiglu, gather=False, *, k=None
    ) -> bool:
        if not dsl_targets_sm12x():
            return False
        fused_gated = _uses_fused_gated_tile(tile, activation)
        if tuple(tile[:2]) == (8, 64) and activation is ActivationType.Relu2:
            return False
        if tile[1] == 64 and activation is not ActivationType.Relu2 and not fused_gated:
            return False
        if epi is not None and (
            epi not in epi_tactics(tile) or not EPI_CONFIGS[epi].supports_tile(tile)
        ):
            return False
        try:
            if epi is None:
                stage, epi = resolve_stage_store(tuple(tile), activation, gather, k=k)
            else:
                stage = resolve_stage(tuple(tile), epi, activation, gather, k=k)
            kernel_type, cfg_factory = select_kernel(
                tuple(tile), activation=activation, k=k
            )
            kernel_type(
                cfg_factory(
                    tuple(tile),
                    stage,
                    epi=epi,
                    activation=activation,
                    gather=gather,
                ),
                1,
                gather=gather,
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
        gather=False,
    ) -> bool:
        candidates = epi_tactics(tile) if epi is None else (epi,)
        return (
            cls.is_valid_dtypes(out_dtype)
            and cls.is_valid_tile(tile)
            and activation in cls.ACTIVATIONS
            and (
                tile[1] == 128
                or activation is ActivationType.Relu2
                or _uses_fused_gated_tile(tile, activation)
            )
            and (activation is ActivationType.Relu2 or n % 128 == 0)
            and cls.is_valid_alignment(n, k, tile)
            and any(
                candidate in epi_tactics(tile)
                and EPI_CONFIGS[candidate].can_implement(
                    tile, n, cutlass.Float4E2M1FN.width
                )
                and cls.is_constructible(tile, candidate, activation, gather, k=k)
                for candidate in candidates
            )
        )

    def build(self, grid_x: int):
        kernel_type, _ = select_kernel(self.tile)
        return kernel_type(self.cfg, grid_x, gather=self.gather)

    def get_grid_size(self, total_rows: int, num_experts: int, num_sms: int):
        if not is_swapab(self.tile):
            return num_sms
        return get_moe_worklist_grid_size(total_rows, num_experts, self.n, self.tile)


TACTICS = tuple(
    (bm, bn, epi)
    for bm, bn in CuteDslSm120GroupedNvfp4Fc1ActQ1Op.TILES
    for epi in epi_tactics((bm, bn, PLAIN_TILE_K))
    if CuteDslSm120GroupedNvfp4Fc1ActQ1Op.is_constructible(
        (bm, bn, PLAIN_TILE_K),
        epi,
        ActivationType.Relu2 if bn == 64 and bm != 8 else ActivationType.Swiglu,
        k=PLAIN_TILE_K,
    )
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
    enable_pdl=False,
    swiglu_limit=None,
    situ_beta=SITU_BETA,
    situ_linear_beta=SITU_LINEAR_BETA,
    gather=False,
):
    return CuteDslSm120GroupedNvfp4Fc1ActQ1Op(
        n=n,
        k=k,
        tile=tile,
        out_dtype=out_dtype,
        activation=activation,
        epi=epi,
        enable_pdl=enable_pdl,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gather=gather,
    )


_COMPILED: dict = {}


def compiled_kernel(sample_args, *, op, grid_x: int, sm_version: str):
    key = (
        op.tactic,
        op.activation,
        op.out_dtype,
        op.enable_pdl,
        op.gather,
        op.swiglu_limit,
        op.situ_beta,
        op.situ_linear_beta,
        grid_x,
        sm_version,
        sample_args[1].shape[0] == 1,
    )
    if _uses_fused_gated_tile(op.tile, op.activation):
        key += (op.k,)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(op.build(grid_x), *sample_args)
        _COMPILED[key] = hit
    return hit


def select_tile(
    *,
    total_rows: int,
    n: int,
    k: int,
    num_experts: int,
    num_sms: int,
    activation=ActivationType.Swiglu,
):
    tile = select_fp4_fc1_act_q1_tile(
        total_rows=total_rows, n=n, k=k, num_experts=num_experts, num_sms=num_sms
    )
    if tile[0] == 8 and activation is not ActivationType.Relu2:
        return (tile[0], 64, tile[2])
    return tile


def select_tactic(
    *,
    total_rows: int,
    n: int,
    k: int,
    num_experts: int,
    num_sms: int,
    activation=ActivationType.Swiglu,
):
    tile = select_tile(
        total_rows=total_rows,
        n=n,
        k=k,
        num_experts=num_experts,
        num_sms=num_sms,
        activation=activation,
    )
    return tile, tile[0] != 128


def _prepare_worklist(op, inputs, out, worklist=None, workcount=None):
    if not is_swapab(op.tile):
        if worklist is not None or workcount is not None:
            raise ValueError("worklist is supported only by standard-grid FC1 decode")
        return None, None
    from . import moe_fp4_q0_route_triton as route

    a_q, _, b_q, _, offsets = inputs[:5]
    total_rows = inputs[8].numel() if op.gather else int(a_q.shape[0])
    return route.prepare_nvfp4_worklist(
        offsets,
        total_rows,
        op.tile[0],
        worklist=worklist,
        workcount=workcount,
        tensors=(*inputs, *out),
        implicit=int(b_q.shape[0]) == 1 and not op.gather,
    )


def _validate_inputs(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    weight_global_scale,
    input_global_scale,
    global_scale,
    activation=ActivationType.Swiglu,
    token_map=None,
):
    tensors = (
        a_q,
        a_scale,
        b_q,
        b_scale,
        m_indptr,
        weight_global_scale,
        input_global_scale,
        global_scale,
    )
    if token_map is not None:
        tensors += (token_map,)
        if token_map.dtype != torch.int32 or token_map.ndim != 1:
            raise TypeError("token_map must be a 1D int32 tensor")
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
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    e, n2 = int(b_q.shape[0]), int(b_q.shape[1])
    if activation is not ActivationType.Relu2 and n2 % 2 != 0:
        raise ValueError("b_q projection extent must be even")
    if weight_global_scale.dtype != torch.float32 or tuple(
        weight_global_scale.shape
    ) != (e,):
        raise TypeError("weight_global_scale must be float32[num_experts]")
    for name, scale in (
        ("input_global_scale", input_global_scale),
        ("global_scale", global_scale),
    ):
        if scale.dtype != torch.float32 or tuple(scale.shape) != (1,):
            raise TypeError(f"{name} must be float32[1]")
    if int(m_indptr.shape[0]) != e + 1:
        raise ValueError("m_indptr length must equal num_experts + 1")
    if int(a_q.shape[1]) != int(b_q.shape[2]):
        raise ValueError("a_q and b_q K extents must match")
    m, packed_k = int(a_q.shape[0]), int(a_q.shape[1])
    k = packed_k * 2
    if k % 64:
        raise ValueError("canonical FP4 weight K must be divisible by 64")
    cols = ceil_div(k, 16)
    want_a_scale = (
        (m, cols) if token_map is not None else (compute_padded_offset(m, e, 128), cols)
    )
    want_b_scale = (
        (e, ceil_div(n2, 128), ceil_div(cols, 4), 32, 16),
        (e, align(n2, 128), cols),
    )
    if tuple(a_scale.shape) != want_a_scale:
        raise ValueError(f"a_scale shape must be {want_a_scale}")
    if tuple(b_scale.shape) not in want_b_scale:
        raise ValueError("b_scale must be a canonical 5D or 3D weight scale view")
    if token_map is not None:
        if n2 % (64 if activation is ActivationType.Relu2 else 256):
            raise ValueError(
                "gather FC1 requires output N divisible by 64 for ReLU2 or 128 for gated activations"
            )
        if any(t.data_ptr() % 16 for t in (a_q, a_scale)) or token_map.data_ptr() % 4:
            raise ValueError(
                "gather A/SFA must be 16-byte aligned and token_map 4-byte aligned"
            )


def _normalize_swiglu_limit(activation: ActivationType, swiglu_limit):
    if swiglu_limit is None:
        return None
    if activation is not ActivationType.Swiglu:
        raise ValueError("swiglu_limit is valid only for SwiGLU")
    swiglu_limit = float(swiglu_limit)
    if not math.isfinite(swiglu_limit) or swiglu_limit <= 0:
        raise ValueError("swiglu_limit must be finite and positive")
    return swiglu_limit


def cute_dsl_sm12x_fc1_act_q1_nvfp4(
    a_q,
    a_scale,
    b_q,
    b_scale,
    m_indptr,
    weight_global_scale,
    input_global_scale,
    global_scale,
    out_dtype: torch.dtype = torch.uint8,
    tile=None,
    epi=None,
    tune=None,
    enable_pdl: bool = False,
    *,
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float | None = None,
    situ_beta: float = SITU_BETA,
    situ_linear_beta: float = SITU_LINEAR_BETA,
    out_q: torch.Tensor | None = None,
    out_sf: torch.Tensor | None = None,
    token_map: torch.Tensor | None = None,
    worklist: torch.Tensor | None = None,
    workcount: torch.Tensor | None = None,
):
    _validate_inputs(
        a_q,
        a_scale,
        b_q,
        b_scale,
        m_indptr,
        weight_global_scale,
        input_global_scale,
        global_scale,
        activation,
        token_map,
    )
    with torch.cuda.device(a_q.device):
        swiglu_limit = _normalize_swiglu_limit(activation, swiglu_limit)
        n, k = (
            int(b_q.shape[1]) // (1 if activation is ActivationType.Relu2 else 2),
            int(b_q.shape[2]) * 2,
        )
        gather = token_map is not None
        total_rows = token_map.numel() if gather else int(a_q.shape[0])
        assert not enable_pdl or (out_q is not None and out_sf is not None), (
            "PDL requires caller-owned out_q and out_sf"
        )
        props = torch.cuda.get_device_properties(a_q.device)
        grid_x = props.multi_processor_count
        sm_version = f"sm_{props.major}{props.minor}"
        inputs = (
            a_q,
            a_scale,
            b_q,
            b_scale,
            m_indptr,
            weight_global_scale,
            input_global_scale,
            global_scale,
        )
        if gather:
            inputs += (token_map,)
        if tile is None and tune is not False:
            chosen = None
            if MOE_AUTOTUNE_ENABLED():
                with autotune():
                    _, tactic = AutoTuner.get().choose_one(
                        "cute_dsl_sm12x_fc1_act_q1_nvfp4",
                        [
                            _Fc1ActQ1Runner(
                                out_dtype,
                                activation,
                                enable_pdl,
                                swiglu_limit,
                                situ_beta,
                                situ_linear_beta,
                            )
                        ],
                        TuningConfig(),
                        list(inputs),
                    )
                if tactic != -1:
                    chosen = tactic
            if chosen is not None:
                tile, epi = split_tactic(chosen)
        if tile is None:
            tile = select_tile(
                total_rows=total_rows,
                n=n,
                k=k,
                num_experts=int(b_q.shape[0]),
                num_sms=grid_x,
                activation=activation,
            )
        op = _op(
            n,
            k,
            tuple(tile),
            out_dtype,
            activation,
            epi,
            enable_pdl,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            gather=gather,
        )
        sf_shape = out_sf_shape(total_rows, n, int(b_q.shape[0]))
        q = (
            torch.empty(total_rows, n // 2, dtype=torch.uint8, device=a_q.device)
            if out_q is None
            else out_q
        )
        sf = (
            torch.zeros(sf_shape, dtype=torch.uint8, device=a_q.device)
            if out_sf is None
            else out_sf
        )
        if (
            q.shape != (total_rows, n // 2)
            or q.dtype != torch.uint8
            or q.device != a_q.device
            or not q.is_contiguous()
        ):
            raise ValueError("out_q does not match the NVFP4 Q1 output contract")
        if (
            sf.shape != sf_shape
            or sf.dtype != torch.uint8
            or sf.device != a_q.device
            or not sf.is_contiguous()
        ):
            raise ValueError("out_sf does not match the NVFP4 Q1 scale contract")
        if q.data_ptr() % 16 != 0 or sf.data_ptr() % 16 != 0:
            raise ValueError("out_q and out_sf must be 16-byte aligned")
        worklist, workcount = _prepare_worklist(
            op, inputs, (q, sf), worklist, workcount
        )
        args = make_args(
            a_q,
            a_scale,
            b_q,
            b_scale,
            q,
            sf,
            m_indptr,
            weight_global_scale,
            input_global_scale,
            global_scale,
            op.cfg.epi,
            op.cfg.TILE,
            token_map=token_map,
            worklist=worklist,
            workcount=workcount,
        )
        grid_x = op.get_grid_size(total_rows, int(b_q.shape[0]), grid_x)
        compiled_kernel(args, op=op, grid_x=grid_x, sm_version=sm_version)(*args)
        return q, sf


def cute_dsl_sm12x_route_fc1_act_q1_nvfp4(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    tile=None,
    epi=None,
    activation: ActivationType = ActivationType.Swiglu,
    workspace=None,
    enable_pdl: bool = False,
    swiglu_limit: float | None = None,
    situ_beta: float = SITU_BETA,
    situ_linear_beta: float = SITU_LINEAR_BETA,
    out_q: torch.Tensor | None = None,
    out_sf: torch.Tensor | None = None,
):
    from ....quantization import fp4_quantize
    from . import moe_fp4_q0_route_triton as route

    if b_q.ndim != 3 or b_q.dtype is not torch.uint8:
        raise TypeError("b_q must be 3D uint8-packed FP4")
    experts = int(b_q.shape[0])
    route._validate(x, topk_ids, topk_weights, experts, input_global_scale)
    with torch.cuda.device(x.device):
        if topk_ids.numel() == 0:
            raise ValueError("topk_ids must contain at least one route")
        n2, k = int(b_q.shape[1]), int(b_q.shape[2]) * 2
        if int(x.shape[1]) != k:
            raise ValueError("x and b_q K extents must match")
        if activation is not ActivationType.Relu2 and n2 % 2:
            raise ValueError("b_q projection extent must be even")
        n = n2 // (1 if activation is ActivationType.Relu2 else 2)
        swiglu_limit = _normalize_swiglu_limit(activation, swiglu_limit)
        if tile is None:
            props = torch.cuda.get_device_properties(x.device)
            tile, gather = select_tactic(
                total_rows=topk_ids.numel(),
                n=n,
                k=k,
                num_experts=experts,
                num_sms=props.multi_processor_count,
                activation=activation,
            )
        else:
            tile = tuple(tile)
            gather = tile[0] != 128
        _op(
            n,
            k,
            tuple(tile),
            torch.uint8,
            activation,
            epi,
            enable_pdl,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            gather=gather,
        )
        decode = is_swapab(tile)
        if enable_pdl and (out_q is None or out_sf is None):
            raise ValueError("PDL requires caller-owned out_q and out_sf")
        if workspace is None:
            if decode:
                workspace = route.make_nvfp4_route_meta_workspace(
                    topk_ids, experts, tile_m=tile[0]
                )
            elif gather:
                workspace = route.make_nvfp4_route_meta_workspace(topk_ids, experts)
            else:
                workspace = route.make_nvfp4_q0_route_workspace(x, topk_ids, experts)
        if decode and not isinstance(workspace, route.Nvfp4DecodeRouteMetaWorkspace):
            raise TypeError(
                "standard-grid FC1 decode requires a worklist metadata workspace"
            )
        if gather:
            if not isinstance(workspace, route.Nvfp4RouteMetaWorkspace) or isinstance(
                workspace, route.Nvfp4Q0RouteWorkspace
            ):
                raise TypeError("gather requires a metadata-only workspace")
        elif not isinstance(workspace, route.Nvfp4Q0RouteWorkspace):
            raise TypeError("TMA requires a Q0 route workspace")
        sources = (
            x,
            topk_ids,
            topk_weights,
            b_q,
            b_scale,
            input_global_scale,
            weight_global_scale,
            global_scale,
            out_q,
            out_sf,
        )
        if any(
            route._overlaps(value, source)
            for value in vars(workspace).values()
            for source in sources
            if source is not None
        ):
            raise ValueError("workspace must not alias an input or Q1 output")
        if gather:
            x = x.contiguous()
            if decode:
                offsets, token_map, token_weights = route.nvfp4_route_meta_triton(
                    topk_ids, topk_weights, experts, workspace=workspace, tile_m=tile[0]
                )
            else:
                offsets, token_map, token_weights = route.nvfp4_route_meta_triton(
                    topk_ids, topk_weights, experts, workspace=workspace
                )
            a_q, a_scale = fp4_quantize(
                x,
                input_global_scale,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
                is_sf_8x4_layout=False,
                is_global_scale_inversed=False,
                enable_pdl=enable_pdl,
                backend="cuda",
                nvfp4_4over6=None,
            )
        else:
            offsets, token_map, token_weights, a_q, a_scale = (
                route.nvfp4_q0_route_triton(
                    x,
                    topk_ids,
                    topk_weights,
                    experts,
                    input_global_scale,
                    workspace=workspace,
                    enable_pdl=enable_pdl,
                )
            )
        q, sf = cute_dsl_sm12x_fc1_act_q1_nvfp4(
            a_q,
            a_scale,
            b_q,
            b_scale,
            offsets,
            weight_global_scale,
            input_global_scale,
            global_scale,
            tile=tile,
            epi=epi,
            activation=activation,
            tune=False,
            enable_pdl=enable_pdl,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            out_q=out_q,
            out_sf=out_sf,
            token_map=token_map if gather else None,
            worklist=workspace.worklist if decode else None,
            workcount=workspace.workcount if decode else None,
        )
        return offsets, token_map, token_weights, q, sf


class _Fc1ActQ1Runner(TunableRunner):
    def __init__(
        self,
        out_dtype,
        activation,
        enable_pdl=False,
        swiglu_limit=None,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    ):
        self.out_dtype = out_dtype
        self._out = None
        self.activation, self.enable_pdl = activation, enable_pdl
        self.swiglu_limit = swiglu_limit
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
        _, _, b_q, _, _, _, _, _ = inputs[:8]
        gather = len(inputs) == 9
        n, k = (
            int(b_q.shape[1]) // (1 if self.activation is ActivationType.Relu2 else 2),
            int(b_q.shape[2]) * 2,
        )
        return [
            tactic
            for tactic in TACTICS
            if CuteDslSm120GroupedNvfp4Fc1ActQ1Op.can_implement(
                n=n,
                k=k,
                tile=split_tactic(tactic)[0],
                out_dtype=self.out_dtype,
                activation=self.activation,
                epi=split_tactic(tactic)[1],
                gather=gather,
            )
        ]

    def alloc_out(self, inputs):
        a_q, _, b_q, _, _, _, _, _ = inputs[:8]
        total_rows = inputs[8].numel() if len(inputs) == 9 else int(a_q.shape[0])
        n, experts = (
            int(b_q.shape[1]) // (1 if self.activation is ActivationType.Relu2 else 2),
            int(b_q.shape[0]),
        )
        return (
            torch.empty(total_rows, n // 2, dtype=torch.uint8, device=a_q.device),
            torch.zeros(
                out_sf_shape(total_rows, n, experts),
                dtype=torch.uint8,
                device=a_q.device,
            ),
        )

    def launch(self, inputs, out, tactic):
        tile, epi = (None, None) if tactic is None else split_tactic(tactic)
        cute_dsl_sm12x_fc1_act_q1_nvfp4(
            *inputs[:8],
            out_dtype=self.out_dtype,
            tile=tile,
            epi=epi,
            tune=False,
            enable_pdl=self.enable_pdl,
            activation=self.activation,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            out_q=out[0],
            out_sf=out[1],
            token_map=inputs[8] if len(inputs) == 9 else None,
        )

    def get_cache_key_extras(self, inputs):
        a_q, _, b_q, _, _, _, _, _ = inputs[:8]
        gather = len(inputs) == 9
        return (
            int(a_q.shape[0]),
            int(b_q.shape[0]),
            int(b_q.shape[1]),
            int(b_q.shape[2]) * 2,
            gather,
            inputs[8].numel() if gather else int(a_q.shape[0]),
            str(self.out_dtype),
            self.activation,
            self.enable_pdl,
            self.swiglu_limit,
            self.situ_beta,
            self.situ_linear_beta,
            torch.cuda.get_device_capability(a_q.device),
        )


def MOE_AUTOTUNE_ENABLED() -> bool:
    return os.environ.get("MOE_AUTOTUNE", "0") not in ("0", "", "false", "False")
