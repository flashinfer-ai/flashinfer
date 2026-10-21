"""W4A16 launch and prepared-weight records."""

from __future__ import annotations

from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from typing import NamedTuple
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class W4A16GemmCompileResult:
    compiled: object
    tile_n: int
    tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"
    w13_layout: str = "w13"
    dense_route_fast_path: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    trellis_pair_kind: str | None = None
    trellis_rate_axis: str | None = None


@dataclass(frozen=True)
class W4A16ActivationCompileResult:
    compiled: object
    rows: int
    intermediate_size: int
    activation: str
    swiglu_limit: float | None
    swiglu_clamp_after_silu: bool
    swiglu_alpha: float
    swiglu_beta: float


@dataclass(frozen=True)
class W4A16TopKSumCompileResult:
    compiled: object
    m: int
    topk: int
    hidden_size: int
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    num_experts: int = 0
    route_num_experts: int = 0
    route_ids_dtype: torch.dtype = torch.int32
    use_expert_map: bool = False
    broadcast_svh: bool = False
    apply_topk_weights: bool = False


@dataclass(frozen=True)
class W4A16FusedMoeCompileResult:
    compiled: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    apply_router_weight_on_input: bool
    zero_fc2_output: bool
    element_dtype: str
    fast_math: bool
    swiglu_limit: float | None
    swiglu_clamp_after_silu: bool
    swiglu_alpha: float
    swiglu_beta: float
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    w13_layout: str = "w13"
    direct_topk_routes: bool = False
    use_expert_map: bool = False
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False
    prefill_fused_sum_fp32: bool = False
    collect_activation_amax: bool = False
    schedule_whole_tiles: bool = False
    intermediate_rotation: bool = False
    dual_a: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    fc1_trellis_pair_kind: str | None = None
    fc2_trellis_pair_kind: str | None = None
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    rotation_input_dtype: str = "fp16"
    cta_threads: int = -1
    shared_memory_bytes: int = -1
    broadcast_suh: bool = False
    small_m_direct_launches: tuple[_W4A16SmallMDirectLaunch, ...] = ()
    trellis_direct_lut: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
    kernel: W4A16GemmCompileResult
    c_tmp: torch.Tensor


class _W4A16SmallMDirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    activation: str
    fast_math: bool
    topk_ids_dtype: torch.dtype


@dataclass(frozen=True)
class W4A16SmallMDirectCompileResult:
    moe_block_size: int
    small_m_direct_launches: tuple[_W4A16SmallMDirectLaunch, ...]


class _W4A16FC2DirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk_ids_dtype: torch.dtype
