"""Public GC MoE allocation and weight-layout entrypoints."""

from __future__ import annotations

from typing import Optional, Tuple
import types

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..jit_kernels.heuristics.gc_moe import get_block_m_for_gc_moe, get_ring_tokens
from ..layout.gc_moe import GCBufferLayout, TOKEN_ALIGNMENT, align


class GCSymmBuffer:
    """GC-owned symmetric workspace; serialize calls sharing one allocation.

    Stage shared input scales in ``shared_l1_acts_sf`` at row
    ``(t // BM) * align(BM, 128) + (u // 128) * 128 + (u % 32) * 4
    + (u % 128) // 32``, where ``u = t % BM``. Query BM using this rank's
    actual token count. Shared activations always use K32 scales; ``x_sf``
    remains K-major while shared scale views are MN-major.

    ``base`` must be a live ``GCSymmBuffer`` with the same group and exact
    GC layout. Foreign workspaces (including DeepGEMM) are incompatible.
    Serialize calls across all wrappers sharing the allocation; retire captured
    graphs before destroying an allocation used by them.
    """

    _views = ('x', 'x_sf', 'topk_idx', 'topk_weights', 'shared_l1_acts',
              'shared_l1_acts_sf', 'shared_l2_acts', 'shared_l2_acts_sf',
              'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf')

    def __init__(self, group: dist.ProcessGroup, num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int, hidden: int,
                 intermediate_hidden: int, num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4', activation: str = 'swiglu', base=None):
        if mma_type != 'fp8xfp4' or activation != 'swiglu':
            raise ValueError('GC MoE requires fp8xfp4/swiglu')
        if base is not None and not isinstance(base, GCSymmBuffer):
            raise ValueError('base must be a GC-compatible GCSymmBuffer; foreign workspaces are unsupported')
        self.group = group
        self.rank_idx = dist.get_rank(group=group)
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank, TOKEN_ALIGNMENT)
        self.num_topk = num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type, self.activation = mma_type, activation
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        ring_tokens, sf_ring_tokens = get_ring_tokens(
            group.size(), num_experts, self.num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_sms)
        self.layout = GCBufferLayout(group.size(), num_experts,
                                     self.num_max_tokens_per_rank, num_topk,
                                     hidden, intermediate_hidden, ring_tokens,
                                     sf_ring_tokens, num_shared_experts)
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(self.layout.nbytes, dtype=torch.int8, device='cuda')
            self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                           if group.size() == 1 else symm_mem.rendezvous(self.buffer, group=group))
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('symmetric rendezvous returned invalid peer addresses')
            self.peer_addresses = peer_addresses
            self.buffer.zero_()
            group.barrier().wait()
            torch.cuda.synchronize()
        else:
            if base.buffer is None or base.handle is None or base.group is not group:
                raise ValueError('base must be a live allocation in the same group')
            for name in ('num_experts', 'num_max_tokens_per_rank', 'num_topk', 'hidden',
                         'intermediate_hidden', 'num_shared_experts', 'mma_type', 'activation'):
                if getattr(base, name) != getattr(self, name):
                    raise ValueError(f"incompatible base: {name}")
            if self.layout != base.layout:
                raise ValueError('incompatible base: GC workspace layout')
            if self.layout.nbytes > base.buffer.nbytes:
                raise ValueError('base allocation too small')
            self.buffer, self.handle = base.buffer, base.handle
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('base rendezvous returned invalid peer addresses')
            # A compatible base may own the allocation. The 592-byte host
            # SymBuffer value is assembled per call from these rendezvous UVA
            # addresses and retained by each child Graph argument vector.
            self.peer_addresses = peer_addresses
        for name in self._views:
            spec = self.layout.views[name]
            if spec.nbytes == 0:
                view = None
            elif name == 'shared_l1_acts':
                view = self.x
            else:
                dtype = getattr(torch, spec.dtype)
                elem_size = torch.empty((), dtype=dtype).element_size()
                raw = self.buffer.narrow(0, spec.offset, spec.nbytes).view(dtype)
                view = raw.as_strided(spec.shape, spec.stride) if spec.stride else raw.view(spec.shape)
                if spec.nbytes % elem_size:
                    raise AssertionError(f"unaligned view {name}")
            setattr(self, name, view)

    def _for_call(self, num_shared_experts: int):
        """Re-slice the raw allocation for the shared count of this call."""
        if self.buffer is None or self.handle is None:
            raise ValueError('symmetric buffer has been destroyed')
        if num_shared_experts == self.num_shared_experts:
            return self
        layout = GCBufferLayout(
            self.group.size(), self.num_experts,
            self.num_max_tokens_per_rank, self.num_topk,
            self.hidden, self.intermediate_hidden,
            self.layout.ring_tokens, self.layout.sf_ring_tokens,
            num_shared_experts)
        if layout.nbytes > self.buffer.nbytes:
            raise ValueError("symmetric buffer is too small for call's shared experts")
        return _CallGCSymmBuffer(self, layout)

    def destroy(self):
        if self.buffer is not None:
            from ..jit_kernels.impls.sm100_fp8_fp4_gc_moe import release_workspace
            release_workspace(self)
        for name in self._views:
            setattr(self, name, None)
        self.peer_addresses = self.handle = self.buffer = self.group = None


class _CallGCSymmBuffer:
    """Reference C++ call-time slice over the caller-owned symmetric bytes."""

    def __init__(self, base: GCSymmBuffer, layout: GCBufferLayout):
        self._base = base
        self.layout = layout
        self.num_shared_experts = layout.num_shared_experts
        for name in base._views:
            spec = layout.views[name]
            if spec.nbytes == 0:
                view = None
            elif name == 'shared_l1_acts':
                view = self.x
            else:
                dtype = getattr(torch, spec.dtype)
                elem_size = torch.empty((), dtype=dtype).element_size()
                raw = base.buffer.narrow(0, spec.offset, spec.nbytes).view(dtype)
                view = raw.as_strided(spec.shape, spec.stride) if spec.stride else raw.view(spec.shape)
                if spec.nbytes % elem_size:
                    raise AssertionError(f"unaligned call view {name}")
            setattr(self, name, view)

    def __getattr__(self, name):
        return getattr(self._base, name)


def get_symm_buffer_for_gc_moe(group: dist.ProcessGroup, num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 mma_type: str = 'fp8xfp4', activation: str = 'swiglu') -> GCSymmBuffer:
    return GCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                        hidden, intermediate_hidden, num_shared_experts, mma_type, activation)


def _gc_interleave_rows(t: torch.Tensor) -> torch.Tensor:
    if t.ndim not in (2, 3) or t.shape[-2] % 16:
        raise ValueError('W1 rows must be a multiple of 16')
    grouped = t if t.ndim == 3 else t.unsqueeze(0)
    groups, rows = grouped.shape[:2]
    half = rows // 2
    gate = grouped[:, :half].reshape(groups, half // 8, 8, *grouped.shape[2:])
    up = grouped[:, half:].reshape(groups, half // 8, 8, *grouped.shape[2:])
    result = torch.empty_like(grouped).copy_(torch.stack((gate, up), dim=2).reshape(grouped.shape))
    return result if t.ndim == 3 else result.squeeze(0)


def _gc_transpose_sf(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.int32 or sf.ndim not in (2, 3) or sf.shape[-2] % 128:
        raise ValueError('SF must be packed int32 with 128-aligned rows')
    grouped = sf if sf.ndim == 3 else sf.unsqueeze(0)
    groups, rows, packs = grouped.shape
    result = torch.empty_like(grouped).copy_(
        grouped.reshape(groups, -1, 4, 32, packs).transpose(2, 3).reshape(grouped.shape))
    return result if sf.ndim == 3 else result.squeeze(0)


def transform_weights_for_gc_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor], activation: str = 'swiglu',
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Convert original quantized weights to owned K128 (payload, SF) pairs.

    Shared K128 SF must already be expanded to one packed row per N, as for
    the original transform. Call outside capture with ordered source writes.
    Copies run asynchronously on the current CUDA stream; raw inputs may die
    on return.
    """
    from ..jit_kernels.impls.runtime_utils import repack_fp4_weights, repack_fp8_weights
    if activation != 'swiglu':
        raise ValueError('GC MoE supports swiglu')
    pairs = (l1_weights, l2_weights)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(t, torch.Tensor) for t in pair) for pair in pairs):
        raise ValueError('transform needs two original quantized (payload, SF) pairs')
    if any(not t.is_cuda or t.device.index != torch.cuda.current_device()
           for pair in pairs for t in pair):
        raise ValueError('transform inputs must use the current CUDA device')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('transform weights outside CUDA Graph capture')
    stream = torch.cuda.current_stream()
    for pair in pairs:
        for tensor in pair:
            tensor.record_stream(stream)
    source_storage = {pair[0].untyped_storage().data_ptr() for pair in pairs}
    repack = repack_fp4_weights if l1_weights[0].dtype == torch.int8 else repack_fp8_weights
    packed = [repack(_gc_interleave_rows(l1_weights[0])), repack(l2_weights[0])]
    # A degenerate K128 permutation may return a view of its input.
    packed = [t.clone() if t.untyped_storage().data_ptr() in source_storage else t
              for t in packed]
    sf1 = _gc_transpose_sf(_gc_interleave_rows(l1_weights[1]))
    sf2 = _gc_transpose_sf(l2_weights[1])
    return (packed[0], sf1), (packed[1], sf2)


def fp8_fp4_gc_moe(y: torch.Tensor,
                    l1_weights: Tuple[torch.Tensor, torch.Tensor],
                    l2_weights: Tuple[torch.Tensor, torch.Tensor],
                    sym_buffer: GCSymmBuffer,
                    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    recipe: Tuple[int, int, int] = (1, 1, 32),
                    activation: str = 'swiglu', activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    shared_recipe: Optional[Tuple[int, int, int]] = None,
                    activation_alpha: Optional[float] = None) -> None:
    if hasattr(sym_buffer, 'num_shared_experts'):
        assert (sym_buffer.num_shared_experts > 0) == (shared_l1_weights is not None)
    from ..apis.gc_moe import fp8_fp4_gc_moe as launch
    launch(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights,
           shared_l2_weights, cumulative_local_expert_recv_stats, recipe,
           activation, activation_clamp, fast_math, shared_recipe, activation_alpha)
