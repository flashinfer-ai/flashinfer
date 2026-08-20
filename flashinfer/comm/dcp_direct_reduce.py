"""Destination-owned DCP Output/LSE reduce over PyTorch symmetric memory.

Three Triton kernels: tiled publish, system-release signal, wait+merge.
Returned workspace views alias ``combined_*[slot]`` and remain valid
until the next ``run(..., slot=N)`` for that same slot.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from ..api_logging import flashinfer_api
from ..trace.templates.comm import dcp_direct_reduce_trace
from .torch_symmetric_memory import _enable_symm_mem_for_group

_MAX_FENCE_SPINS = 100_000_000
_SUPPORTED_WORLD_SIZES = (2, 4)
_SUPPORTED_HEAD_DIMS = (128, 256, 512)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _trap_if_nonzero(value):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_release_system(pointer, value, mask):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred enabled;
            setp.ne.u32 enabled, $3, 0;
            @enabled st.global.release.sys.u32 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r",
        args=[pointer, value, mask.to(tl.uint32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _triton_publish_kernel(
    partial_output,
    partial_lse,
    peer_output_ptrs,
    peer_lse_ptrs,
    local_epoch,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_token_stride,
    lse_head_stride,
    peer_output_parity_stride,
    peer_output_source_stride,
    peer_output_token_stride,
    peer_output_head_stride,
    peer_output_dim_stride,
    peer_lse_parity_stride,
    peer_lse_source_stride,
    peer_lse_token_stride,
    peer_lse_head_stride,
    my_rank: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_items: tl.constexpr,
    head_block_size: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    destination_rank = tl.program_id(1).to(tl.int64)
    item_block_idx = tl.program_id(2).to(tl.int64)
    epoch = tl.atomic_add(local_epoch, 0, sem="acquire", scope="gpu") + 1
    parity = epoch & 1

    output_ptr_table = peer_output_ptrs.to(tl.pointer_type(tl.uint64))
    peer_output = tl.load(output_ptr_table + destination_rank).to(
        tl.pointer_type(partial_output.dtype.element_ty)
    )
    item = item_block_idx * block_items + tl.arange(0, block_items)
    item_mask = item < local_heads * head_dim
    local_head_idx = item // head_dim
    dim = item % head_dim
    source_head_idx = destination_rank * local_heads + local_head_idx
    value = tl.load(
        partial_output
        + token_idx * output_token_stride
        + source_head_idx * output_head_stride
        + dim * output_dim_stride,
        mask=item_mask,
    )
    tl.store(
        peer_output
        + parity * peer_output_parity_stride
        + my_rank * peer_output_source_stride
        + token_idx * peer_output_token_stride
        + local_head_idx * peer_output_head_stride
        + dim * peer_output_dim_stride,
        value,
        mask=item_mask,
    )

    lse_ptr_table = peer_lse_ptrs.to(tl.pointer_type(tl.uint64))
    peer_lse = tl.load(lse_ptr_table + destination_rank).to(tl.pointer_type(tl.float32))
    lse_local_head_idx = tl.arange(0, head_block_size)
    lse_mask = (item_block_idx == 0) & (lse_local_head_idx < local_heads)
    lse_source_head_idx = destination_rank * local_heads + lse_local_head_idx
    tl.store(
        peer_lse
        + parity * peer_lse_parity_stride
        + my_rank * peer_lse_source_stride
        + token_idx * peer_lse_token_stride
        + lse_local_head_idx * peer_lse_head_stride,
        tl.load(
            partial_lse
            + token_idx * lse_token_stride
            + lse_source_head_idx * lse_head_stride,
            mask=lse_mask,
        ),
        mask=lse_mask,
    )
    # System fence so peer GPUs observe payload stores before the later
    # release signal. Do not advance local_epoch here: other CTAs still
    # sample it for parity.
    tl.inline_asm_elementwise(
        "membar.sys;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _triton_signal_kernel(
    local_epoch,
    peer_signal_ptrs,
    peer_signal_parity_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
):
    epoch = tl.atomic_add(local_epoch, 1, sem="acq_rel", scope="gpu") + 1
    destination_rank = tl.arange(0, block_size)
    destination_mask = destination_rank < world_size
    signal_ptr_table = peer_signal_ptrs.to(tl.pointer_type(tl.uint64))
    peer_signal = tl.load(
        signal_ptr_table + destination_rank, mask=destination_mask, other=0
    ).to(tl.pointer_type(tl.int32))
    parity = epoch & 1
    _store_release_system(
        peer_signal + parity * peer_signal_parity_stride + my_rank,
        epoch.to(tl.uint32),
        destination_mask,
    )


@triton.jit
def _triton_consumer_merge_kernel(
    received_output,
    received_lse,
    received_signal,
    local_epoch,
    combined_output,
    combined_lse,
    output_parity_stride,
    output_source_stride,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_parity_stride,
    lse_source_stride,
    lse_token_stride,
    lse_head_stride,
    signal_parity_stride,
    combined_token_stride,
    combined_head_stride,
    combined_dim_stride,
    combined_lse_token_stride,
    world_size: tl.constexpr,
    source_block_size: tl.constexpr,
    is_base_e: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    max_spins: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    epoch = tl.atomic_add(local_epoch, 0, sem="acquire", scope="gpu")
    parity = epoch & 1

    source_rank = tl.arange(0, source_block_size)
    source_mask = source_rank < world_size
    signal_offset = parity * signal_parity_stride + source_rank
    observed = tl.atomic_add(
        received_signal + signal_offset,
        0,
        mask=source_mask,
        sem="acquire",
        scope="sys",
    )
    expected = epoch.to(tl.int32)
    pending = tl.max(tl.where(source_mask & (observed != expected), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            received_signal + signal_offset,
            0,
            mask=source_mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(source_mask & (observed != expected), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)

    lse_offset = (
        parity * lse_parity_stride
        + source_rank * lse_source_stride
        + token_idx * lse_token_stride
        + local_head_idx * lse_head_stride
    )
    lse = tl.load(received_lse + lse_offset, mask=source_mask, other=-float("inf"))
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    weights = tl.exp(lse - lse_max) if is_base_e else tl.exp2(lse - lse_max)
    weight_sum = tl.sum(weights, axis=0)
    weights = tl.where(weight_sum == 0.0, 0.0, weights / weight_sum)
    final_lse = tl.where(
        weight_sum == 0.0,
        -float("inf"),
        (tl.log(weight_sum) if is_base_e else tl.log2(weight_sum)) + lse_max,
    )

    dim = tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    output_offset = (
        parity * output_parity_stride
        + source_rank[:, None] * output_source_stride
        + token_idx * output_token_stride
        + local_head_idx * output_head_stride
        + dim[None, :] * output_dim_stride
    )
    partial_output = tl.load(
        received_output + output_offset,
        mask=source_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    partial_output = tl.where(weights[:, None] == 0, 0.0, partial_output)
    output = tl.sum(partial_output.to(tl.float32) * weights[:, None], axis=0)
    tl.store(
        combined_output
        + token_idx * combined_token_stride
        + local_head_idx * combined_head_stride
        + dim * combined_dim_stride,
        output,
        mask=dim_mask,
    )
    tl.store(
        combined_lse + token_idx * combined_lse_token_stride + local_head_idx, final_lse
    )


class DCPDirectReduceWorkspace:
    """Destination-owned direct Output/LSE reduce workspace.

    Output ownership
    ----------------
    Workspace-backed mode (``out is None`` and ``lse_out is None``):
        Kernel 3 writes ``combined_output[slot, :T]`` and
        ``combined_lse[slot, :T]``. The returned tensors alias that
        workspace storage and stay valid until the next
        ``run(..., slot=N)`` for the same slot. A run on a different
        slot does not invalidate them.

    Caller-owned mode (both ``out`` and ``lse_out`` provided):
        Kernel 3 writes directly into the caller tensors. Their
        lifetime is controlled by the caller and is not affected by
        later workspace invocations.

    ``out`` and ``lse_out`` must both be provided or both be None.
    """

    supports_output_view: bool = True

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_tokens: int,
        total_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        num_slots: int = 1,
    ) -> None:
        if group is None:
            raise ValueError("group is required")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        world_size = group.size()
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(f"world_size must be in {_SUPPORTED_WORLD_SIZES}")
        if total_heads <= 0 or total_heads % world_size != 0:
            raise ValueError("total_heads must be positive and divisible by world_size")
        if head_dim not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim must be in {_SUPPORTED_HEAD_DIMS}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be in {_SUPPORTED_DTYPES}")

        self.group = group
        self.rank = group.rank()
        self.world_size = world_size
        self.max_tokens = int(max_tokens)
        self.total_heads = int(total_heads)
        self.local_heads = self.total_heads // self.world_size
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.num_slots = int(num_slots)
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._block_items = min(
            2048, triton.next_power_of_2(self.local_heads * self.head_dim)
        )
        self._block_h = triton.next_power_of_2(self.local_heads)
        self._allocations: list[tuple[torch.Tensor, object, list[torch.Tensor]]] = []

        _enable_symm_mem_for_group(group.group_name)

        recv_out_shape = (
            self.num_slots,
            2,
            self.world_size,
            self.max_tokens,
            self.local_heads,
            self.head_dim,
        )
        recv_lse_shape = (
            self.num_slots,
            2,
            self.world_size,
            self.max_tokens,
            self.local_heads,
        )
        recv_sig_shape = (self.num_slots, 2, self.world_size)

        self.received_output, _, peer_out_views = self._alloc_symmetric(
            recv_out_shape, dtype
        )
        self.received_lse, _, peer_lse_views = self._alloc_symmetric(
            recv_lse_shape, torch.float32
        )
        self.received_signal, _, peer_sig_views = self._alloc_symmetric(
            recv_sig_shape, torch.int32
        )
        self._peer_output_views = peer_out_views
        self._peer_lse_views = peer_lse_views
        self._peer_signal_views = peer_sig_views

        self.epoch = torch.zeros(self.num_slots, dtype=torch.int32, device=self.device)
        self.combined_output = torch.empty(
            (self.num_slots, self.max_tokens, self.local_heads, self.head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.combined_lse = torch.empty(
            (self.num_slots, self.max_tokens, self.local_heads),
            dtype=torch.float32,
            device=self.device,
        )

        self.peer_output_ptrs = torch.empty(
            (self.num_slots, self.world_size), dtype=torch.int64, device=self.device
        )
        self.peer_lse_ptrs = torch.empty_like(self.peer_output_ptrs)
        self.peer_signal_ptrs = torch.empty_like(self.peer_output_ptrs)

        def _ptr_table(views: list[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [
                    [
                        int(views[peer][slot].data_ptr())
                        for peer in range(self.world_size)
                    ]
                    for slot in range(self.num_slots)
                ],
                dtype=torch.int64,
            )

        self.peer_output_ptrs.copy_(_ptr_table(peer_out_views))
        self.peer_lse_ptrs.copy_(_ptr_table(peer_lse_views))
        self.peer_signal_ptrs.copy_(_ptr_table(peer_sig_views))

    def _alloc_symmetric(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, object, list[torch.Tensor]]:
        storage = symm_mem.empty(*shape, dtype=dtype, device=self.device)
        storage.zero_()
        torch.cuda.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None
        handle.barrier()
        peer_views = [
            handle.get_buffer(peer, tuple(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        self._allocations.append((storage, handle, peer_views))
        return storage, handle, peer_views

    def _validate_inputs(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        slot: int,
    ) -> None:
        if partial_output.ndim != 3:
            raise ValueError("partial_output must have shape [T, H_total, D]")
        if partial_lse.ndim != 2:
            raise ValueError("partial_lse must have shape [T, H_total]")
        if partial_output.shape[:2] != partial_lse.shape:
            raise ValueError(
                "partial_output and partial_lse batch/head shapes must match"
            )
        if partial_output.shape[2] != self.head_dim:
            raise ValueError("partial_output head_dim does not match workspace")
        if partial_output.shape[1] != self.total_heads:
            raise ValueError("partial_output total_heads does not match workspace")
        t = partial_output.shape[0]
        if t <= 0:
            raise ValueError("T must be > 0")
        if t > self.max_tokens:
            raise ValueError("T exceeds max_tokens")
        if partial_output.dtype != self.dtype:
            raise ValueError("partial_output dtype does not match workspace")
        if partial_lse.dtype != torch.float32:
            raise ValueError("partial_lse dtype must be float32")
        if partial_output.device != self.device:
            raise ValueError("partial_output device does not match workspace")
        if partial_lse.device != self.device:
            raise ValueError("partial_lse device does not match workspace")
        if partial_output.stride(2) != 1:
            raise ValueError("partial_output.stride(2) must be 1")
        if partial_lse.stride(1) != 1:
            raise ValueError("partial_lse.stride(1) must be 1")
        if slot < 0 or slot >= self.num_slots:
            raise ValueError("slot is out of range")

    def _validate_outputs(
        self,
        partial_output: torch.Tensor,
        out: Optional[torch.Tensor],
        lse_out: Optional[torch.Tensor],
    ) -> None:
        if (out is None) != (lse_out is None):
            raise ValueError(
                "out and lse_out must either both be provided or both be None"
            )
        if out is None:
            return
        t = partial_output.shape[0]
        expected_out_shape = (t, self.local_heads, self.head_dim)
        expected_lse_shape = (t, self.local_heads)
        if tuple(out.shape) != expected_out_shape:
            raise ValueError("out shape must be [T, H_local, D]")
        if tuple(lse_out.shape) != expected_lse_shape:
            raise ValueError("lse_out shape must be [T, H_local]")
        if out.dtype != self.dtype:
            raise ValueError("out dtype does not match workspace")
        if lse_out.dtype != torch.float32:
            raise ValueError("lse_out dtype must be float32")
        if out.device != self.device:
            raise ValueError("out device does not match workspace")
        if lse_out.device != self.device:
            raise ValueError("lse_out device does not match workspace")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if not lse_out.is_contiguous():
            raise ValueError("lse_out must be contiguous")

    @flashinfer_api(trace=dcp_direct_reduce_trace)
    def run(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        slot: int = 0,
        is_lse_base_on_e: bool = True,
        out: Optional[torch.Tensor] = None,
        lse_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce per-rank partial Output/LSE into the destination-owned shard.

        Every rank must call this with the same ``T``, ``slot``, and
        ``is_lse_base_on_e``. A mismatch is not detected on the host and
        leaves the merge kernel spinning until the spin trap fires.

        Parameters
        ----------
        partial_output
            ``[T, H_total, D]`` partial attention output on this rank.
        partial_lse
            ``[T, H_total]`` float32 partial LSE on this rank.
        slot
            Workspace slot in ``[0, num_slots)``.
        is_lse_base_on_e
            If True, LSE is natural-log. If False, LSE is log2.
        out
            Optional contiguous ``[T, H_local, D]`` destination. Must be
            provided together with ``lse_out``.
        lse_out
            Optional contiguous ``[T, H_local]`` float32 destination.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Combined output and LSE. Workspace-backed views alias
            ``combined_output[slot, :T]`` / ``combined_lse[slot, :T]``
            until the next ``run`` on the same slot.
        """
        self._validate_inputs(partial_output, partial_lse, slot)
        self._validate_outputs(partial_output, out, lse_out)

        t = partial_output.shape[0]
        if out is None:
            selected_output = self.combined_output[slot, :t]
            selected_lse = self.combined_lse[slot, :t]
        else:
            selected_output = out
            selected_lse = lse_out

        epoch_slot = self.epoch[slot : slot + 1]
        self._run_triton(
            partial_output,
            partial_lse,
            selected_output,
            selected_lse,
            slot,
            t,
            is_lse_base_on_e,
            epoch_slot,
        )
        return selected_output, selected_lse

    def _run_triton(
        self,
        partial_output,
        partial_lse,
        selected_output,
        selected_lse,
        slot,
        t,
        is_lse_base_on_e,
        epoch_slot,
    ) -> None:
        output_slot = self.received_output[slot]
        lse_slot = self.received_lse[slot]
        signal_slot = self.received_signal[slot]
        publish_blocks = triton.cdiv(
            self.local_heads * self.head_dim, self._block_items
        )
        _triton_publish_kernel[(t, self.world_size, publish_blocks)](
            partial_output,
            partial_lse,
            self.peer_output_ptrs[slot],
            self.peer_lse_ptrs[slot],
            epoch_slot,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            output_slot.stride(0),
            output_slot.stride(1),
            output_slot.stride(2),
            output_slot.stride(3),
            output_slot.stride(4),
            lse_slot.stride(0),
            lse_slot.stride(1),
            lse_slot.stride(2),
            lse_slot.stride(3),
            my_rank=self.rank,
            local_heads=self.local_heads,
            head_dim=self.head_dim,
            block_items=self._block_items,
            head_block_size=self._block_h,
            num_warps=8,
        )
        _triton_signal_kernel[(1,)](
            epoch_slot,
            self.peer_signal_ptrs[slot],
            signal_slot.stride(0),
            my_rank=self.rank,
            world_size=self.world_size,
            block_size=triton.next_power_of_2(self.world_size),
            num_warps=1,
        )
        _triton_consumer_merge_kernel[(t, self.local_heads)](
            output_slot,
            lse_slot,
            signal_slot,
            epoch_slot,
            selected_output,
            selected_lse,
            output_slot.stride(0),
            output_slot.stride(1),
            output_slot.stride(2),
            output_slot.stride(3),
            output_slot.stride(4),
            lse_slot.stride(0),
            lse_slot.stride(1),
            lse_slot.stride(2),
            lse_slot.stride(3),
            signal_slot.stride(0),
            selected_output.stride(0),
            selected_output.stride(1),
            selected_output.stride(2),
            selected_lse.stride(0),
            world_size=self.world_size,
            source_block_size=triton.next_power_of_2(self.world_size),
            is_base_e=int(is_lse_base_on_e),
            head_dim=self.head_dim,
            block_dim=triton.next_power_of_2(self.head_dim),
            max_spins=_MAX_FENCE_SPINS,
            num_warps=4,
        )
