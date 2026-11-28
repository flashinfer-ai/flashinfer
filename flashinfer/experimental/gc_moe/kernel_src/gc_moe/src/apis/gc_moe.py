"""Checked host contract and launch configuration for the GC MoE worker graph."""

from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Optional

import torch

from ..jit_kernels.heuristics.gc_moe import get_block_m_for_gc_moe
from ..layout.gc_moe import GCBufferLayout

if TYPE_CHECKING:
    from ..gc import GCSymmBuffer


@dataclass(frozen=True)
class ValidatedCall:
    tokens: int
    block_m: int
    hidden: int
    intermediate: int
    local_experts: int
    shared_experts: int
    shared_weight_gran_k: int
    clamp: float
    alpha: float
    fast_math: bool


def _activation_scalars(activation_clamp: Optional[float],
                        activation_alpha: Optional[float]) -> tuple[float, float]:
    """Mirror pybind optional<float> conversion before the C++ host checks."""
    clamp = c_float(inf if activation_clamp is None else float(activation_clamp)).value
    alpha = c_float(1.0 if activation_alpha is None else float(activation_alpha)).value
    if not (clamp >= 0 and alpha > 0):
        raise ValueError('activation clamp must be nonnegative and alpha positive')
    return clamp, alpha


def _check_weight(w: torch.Tensor, expected: tuple[int, ...], dtype: torch.dtype,
                  name: str) -> None:
    if w.dtype != dtype or not w.is_cuda:
        raise ValueError(f"{name} needs dtype {dtype} and CUDA storage")
    if tuple(w.shape) != expected:
        raise ValueError(f'{name} needs K128 packed shape {expected}; call transform_weights_for_gc_moe first')
    if not w.is_contiguous():
        raise ValueError(f'{name} packed storage must be contiguous')
    if w.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned for TMA")
    if any(w.stride(axis) * w.element_size() % 16 for axis in range(w.ndim - 1)):
        raise ValueError(f"{name} TMA strides must be 16-byte aligned")


def _check_sf(sf: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if sf.dtype != torch.int32 or tuple(sf.shape) != expected or not sf.is_cuda:
        raise ValueError(f"{name} needs packed int32 scale shape {expected}")
    rows = expected[-2]
    if sf.stride(-2) != 1 or sf.stride(-1) < rows or sf.stride(-1) % 4:
        raise ValueError(f"{name} must have MN-major, 4-aligned TMA scale stride")
    if len(expected) == 3 and sf.stride(0) != sf.stride(-1) * expected[-1]:
        raise ValueError(f"{name} has invalid expert stride")
    if sf.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def validate_call(y: torch.Tensor, l1_weights: tuple[torch.Tensor, torch.Tensor],
                  l2_weights: tuple[torch.Tensor, torch.Tensor], sym_buffer: GCSymmBuffer,
                  shared_l1_weights: Optional[tuple[torch.Tensor, torch.Tensor]],
                  shared_l2_weights: Optional[tuple[torch.Tensor, torch.Tensor]],
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor],
                  recipe: tuple[int, int, int], activation: str,
                  activation_clamp: Optional[float], fast_math: bool,
                  shared_recipe: Optional[tuple[int, int, int]],
                  activation_alpha: Optional[float]) -> ValidatedCall:
    weights = (l1_weights, l2_weights, *(weight for weight in
               (shared_l1_weights, shared_l2_weights) if weight is not None))
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(t, torch.Tensor) for t in pair) for pair in weights):
        raise ValueError('gc_moe needs (packed payload, SF) pairs from transform_weights_for_gc_moe')
    if sym_buffer.buffer is None or sym_buffer.handle is None:
        raise ValueError('symmetric buffer has been destroyed')
    if recipe != (1, 1, 32) or activation != 'swiglu':
        raise ValueError('routed GC MoE requires recipe (1,1,32) and swiglu')
    if shared_recipe is None:
        shared_recipe = recipe
    if shared_recipe not in ((1, 1, 32), (1, 128, 128)):
        raise ValueError('shared recipe must be K32 or K128')
    clamp, alpha = _activation_scalars(activation_clamp, activation_alpha)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError('shared weight pairs must be supplied together')
    tokens = y.shape[0]
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    if (y.dtype != torch.bfloat16 or not y.is_cuda or not y.is_contiguous() or
            tuple(y.shape) != (tokens, h) or tokens > sym_buffer.num_max_tokens_per_rank):
        raise ValueError('y must be contiguous CUDA BF16 [T,H] within capacity')
    ranks = sym_buffer.group.size()
    if sym_buffer.num_experts % ranks:
        raise ValueError('expert count must divide rank count')
    local_e = sym_buffer.num_experts // ranks
    _check_weight(l1_weights[0], (local_e,h//128,2*i,64), torch.int8, 'routed W1')
    _check_weight(l2_weights[0], (local_e,i//128,h,64), torch.int8, 'routed W2')
    _check_sf(l1_weights[1], (local_e,2*i,h//128), 'routed W1 SF')
    _check_sf(l2_weights[1], (local_e,h,i//128), 'routed W2 SF')
    shared_experts = 0
    if shared_l1_weights is not None:
        if shared_l2_weights[0].ndim != 3:
            raise ValueError('shared W2 needs K128 packed shape; call transform_weights_for_gc_moe first')
        shared_i = shared_l2_weights[0].shape[0]*128
        if shared_i <= 0 or shared_i % i:
            raise ValueError('shared W2 K must contain whole shared experts')
        shared_experts = shared_i // i
        gran_k = shared_recipe[2]
        packed_gran_k = gran_k * 4
        _check_weight(shared_l1_weights[0], (h//128,2*shared_i,128), torch.float8_e4m3fn, 'shared W1')
        _check_weight(shared_l2_weights[0], (shared_i//128,h,128), torch.float8_e4m3fn, 'shared W2')
        _check_sf(shared_l1_weights[1], (2*shared_i,(h+packed_gran_k-1)//packed_gran_k), 'shared W1 SF')
        _check_sf(shared_l2_weights[1], (h,(shared_i+packed_gran_k-1)//packed_gran_k), 'shared W2 SF')
    if cumulative_local_expert_recv_stats is not None:
        stats = cumulative_local_expert_recv_stats
        if stats.dtype != torch.int32 or stats.numel() != local_e or \
                not stats.is_cuda or not stats.is_contiguous():
            raise ValueError('stats must be contiguous CUDA int32 [local_experts]')
    bm = get_block_m_for_gc_moe(ranks, sym_buffer.num_experts,
                                sym_buffer.num_max_tokens_per_rank, tokens,
                                sym_buffer.num_topk)
    call_layout = GCBufferLayout(
        ranks, sym_buffer.num_experts, sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk, h, i,
        sym_buffer.layout.ring_tokens, sym_buffer.layout.sf_ring_tokens,
        shared_experts)
    if call_layout.nbytes > sym_buffer.buffer.nbytes:
        raise ValueError("symmetric buffer is too small for call's shared experts")
    return ValidatedCall(tokens, bm, h, i, local_e, shared_experts,
                         shared_recipe[2], clamp, alpha, bool(fast_math))


def fp8_fp4_gc_moe(y: torch.Tensor,
                    l1_weights: tuple[torch.Tensor, torch.Tensor],
                    l2_weights: tuple[torch.Tensor, torch.Tensor],
                    sym_buffer: GCSymmBuffer,
                    shared_l1_weights: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                    shared_l2_weights: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    recipe: tuple[int, int, int] = (1, 1, 32),
                    activation: str = 'swiglu', activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    shared_recipe: Optional[tuple[int, int, int]] = None,
                    activation_alpha: Optional[float] = None) -> None:
    config = validate_call(y, l1_weights, l2_weights, sym_buffer,
                           shared_l1_weights, shared_l2_weights,
                           cumulative_local_expert_recv_stats, recipe, activation,
                           activation_clamp, fast_math, shared_recipe,
                           activation_alpha)
    call_buffer = sym_buffer._for_call(config.shared_experts)
    from ..jit_kernels.impls.sm100_fp8_fp4_gc_moe import launch_gc_moe
    launch_gc_moe(config, y, l1_weights, l2_weights, call_buffer,
                  shared_l1_weights, shared_l2_weights,
                  cumulative_local_expert_recv_stats)
