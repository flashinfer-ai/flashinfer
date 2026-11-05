"""FP4 MegaMoE: SM90 (Hopper) NVFP4 fused/split and MXFP4 fused; SM120 MXFP4 fused.

The fused paths perform cross-rank token dispatch, two W4A8 GEMMs, clamped
SwiGLU and top-k reduction in one persistent kernel. NVFP4 split uses separate
L1 and L2 launches. All paths use FP8 activations and symmetric memory.
NVFP4 supports four or eight ranks; MXFP4 supports power-of-two rank counts up to 32
on SM90.

On SM120 (RTX PRO 6000 class, PCIe peer-to-peer, no NVLink) only MXFP4 is
available, for one host (up to eight ranks) and decode-size batches: the kernel
requantises the staged FP8 activations to MXFP8 (E4M3 with a UE8M0 scale per 32
channels) and runs both GEMMs on the block-scaled FP4 x FP8 tensor-core MMA.

Usage::

    # Once, at load time. Inputs are already in the format described in
    # flashinfer.megamoe.weights: Marlin byte order, scales within the decoder
    # range, and the excess moved into per-expert FP32 global scales g1 / g2.
    w1, w2 = transform_nvfp4_weights_for_mega_moe((w13, w13_sf), (w2_raw, w2_sf))
    buf = MegaMoESymmBuffer(group, num_experts=256, num_max_tokens_per_rank=8448,
                            num_topk=6, hidden=4096, intermediate_hidden=2048)
    # Every step: stage FP8 activations and routing, then run.
    buf.stage(x_fp8, x_sf, topk_idx, topk_weights)
    nvfp4_mega_moe(y, w1, w2, buf, l1_global_scales=g1, l2_global_scales=g2)
"""

import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from ..jit.megamoe import (
    gen_megamoe_sm120_mxfp4_module,
    gen_megamoe_sm90_layout_module,
    gen_megamoe_sm90_mxfp4_module,
    gen_megamoe_sm90_nvfp4_module,
    gen_megamoe_sm90_nvfp4_split_module,
    gen_megamoe_sm90_nvfp4_split_plan_module,
)
from .heuristics import (
    CANDIDATE_BLOCK_M,
    FusedPlan,
    align,
    get_num_max_pool_tokens,
    get_num_padded_sf_pool_tokens,
    select_fused_plan,
    select_mxfp4_fused_plan,
)
from .weights import (
    MXFP4_TILE_BYTES,
    transform_mxfp4_weights_for_mega_moe,
    transform_nvfp4_weights_for_mega_moe,
)

__all__ = [
    "MegaMoESymmBuffer",
    "nvfp4_mega_moe",
    "mxfp4_mega_moe",
    "transform_nvfp4_weights_for_mega_moe",
    "transform_mxfp4_weights_for_mega_moe",
    "select_fused_plan",
    "select_mxfp4_fused_plan",
    "FusedPlan",
    "SplitPlan",
    "select_split_plan",
    "FAMILY_THRESHOLD",
]


@functools.cache
def _get_jit_module(factory, *args):
    return factory(*args).build_and_load()


# SM120: the kernel's per-call counter block (sm120_mxfp4_mega_moe.cuh: kCtrL1 + kMaxEl ints; error word at 3)
_SM120_COUNTER_INTS = 8 + 256
_SM120_ERROR_INDEX = 3
_SM120_MAX_RANKS = 8
_SM120_MAX_TOPK = 8
_SM120_MAX_LOCAL_EXPERTS = 256
_SM120_MAX_HIDDEN = 7936  # the dispatch stages one MXFP8 row (hidden + hidden / 32 bytes) in 8 KB of shared memory
# The SM120 kernel multiplies 8 tokens per weight tile, a decode-size design. Up to 128 tokens per rank it
# beats the moe_ep SM120 MXFP4 backends on RTX 6000D (DeepSeek-V4-Flash, EP4 and EP8); at 256 it no
# longer does, so larger batches (prefill) belong on moe_ep.
_SM120_MAX_TOKENS_PER_RANK = 128


def _is_sm120() -> bool:
    return torch.cuda.get_device_capability()[0] == 12


def _sm120_layout(ranks, experts, capacity, topk, hidden, intermediate):
    """Byte offsets of the SM120 symmetric buffer, in the order the binding reads them:
    [total, x, x_sf, topk_idx, topk_weights, pool_x, pool_xs, pool_meta, recv_cnt,
    recv_flag, comb, comb_flag, route, route_flag, inter, inter_s, ctr]. Every region
    sits at the same offset on every rank; pool_* / recv_* / route* are written by the
    other ranks, comb by the expert ranks, the rest only locally."""
    epr = experts // ranks
    rows = (
        epr * ranks * capacity
    )  # an expert pool slot per (local expert, source rank, token)
    sizes = (
        capacity * hidden,
        capacity * (hidden // 128) * 4,
        capacity * topk * 8,
        capacity * topk * 4,
        rows * hidden,
        rows * (hidden // 128) * 4,
        rows * 4,
        ranks * epr * 4,
        ranks * 4,
        capacity * topk * hidden * 2,
        ranks * 4,
        ranks * (1 + capacity * topk) * 4,
        ranks * 4,
        rows * intermediate,
        rows * (intermediate // 128) * 4,
        _SM120_COUNTER_INTS * 4,
    )
    offsets, cursor = [], 0
    for size in sizes:
        offsets.append(cursor)
        cursor = align(cursor + size, 256)
    return [cursor, *offsets]


def _sm120_weight_strides(hidden, intermediate):
    """Bytes per expert of the SM120 L1 / L2 weights: packed FP4 then one UE8M0 byte per 32 values."""
    return (
        2 * intermediate * hidden // 2 + 2 * intermediate * hidden // 32,
        hidden * intermediate // 2 + hidden * intermediate // 32,
    )


def _jit_args(factory, plan, b):
    args = (plan, b.num_sms)
    # Keep the trailing fused rank argument compatible with existing JIT callers.
    if factory is not gen_megamoe_sm90_nvfp4_module:
        args += (b.group.size(),)
    args += (
        b.num_max_tokens_per_rank,
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.activation_clamp,
        b.fast_math,
    )
    if factory is gen_megamoe_sm90_nvfp4_module:
        args += (b.group.size(),)
    return args


def _run_kernel(factory, entry, plan, y, w1, w2, b, m, stats, gs1, gs2):
    module = _get_jit_module(factory, *_jit_args(factory, plan, b))
    getattr(module, entry)(
        y,
        b.buffer,
        b.handle.buffer_ptrs,
        b.group.rank(),
        w1,
        w2,
        b._offsets,
        m,
        stats,
        gs1,
        gs2,
    )
    return y


class MegaMoESymmBuffer:
    """Allocate symmetric input, expert-token and output-combine storage.

    All ranks must use identical geometry and invoke kernels in matching order,
    with the same token count on every rank (the plan is selected from it).
    Allocation and rendezvous are collective and must precede CUDA graph capture.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup
        Expert-parallel process group over NVLink-connected Hopper GPUs.
    num_experts : int
        Global expert count, divisible by the number of ranks.
    num_max_tokens_per_rank : int
        Maximum input token count on each rank.
    num_topk : int
        Number of routed expert slots per token.
    hidden : int
        Input and output width, a multiple of 256.
    intermediate_hidden : int
        Expert intermediate width, a multiple of 128.
    activation_clamp : float
        SwiGLU clamp value. Defaults to 10.0.
    fast_math : bool
        Use the family's fast-math policy. Defaults to True.

    Attributes
    ----------
    x, x_sf, topk_idx, topk_weights : torch.Tensor
        This rank's staging views (E4M3 ``[capacity, hidden]``, FP32
        ``[capacity, hidden // 128]``, INT64 ``[capacity, num_topk]`` and FP32
        ``[capacity, num_topk]``). :meth:`stage` copies into them; a producer
        that writes the first ``m`` rows directly calls
        :meth:`commit_staged` instead.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        activation_clamp: float = 10.0,
        fast_math: bool = True,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.sm120 = _is_sm120()
        if self.sm120:
            self._init_sm120(
                symm_mem,
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                activation_clamp,
                fast_math,
            )
            return
        # Buffer geometry is format-independent; NVFP4 checks its narrower
        # deployment constraints when selecting a fused/split plan.
        select_mxfp4_fused_plan(
            132,
            group.size(),
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            num_max_tokens_per_rank,
            1,
        )
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("SM90 MegaMoE requires a Hopper GPU")

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

        pool = get_num_max_pool_tokens(
            group.size(), num_max_tokens_per_rank, num_topk, num_experts // group.size()
        )
        sf_pool = max(
            get_num_padded_sf_pool_tokens(pool, bm) for bm in CANDIDATE_BLOCK_M
        )
        layout = list(
            _get_jit_module(
                gen_megamoe_sm90_layout_module
            ).megamoe_sm90_symm_buffer_layout(
                group.size(),
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                pool,
                sf_pool,
            )
        )
        self._offsets = layout
        num_bytes = layout[0]
        self.num_max_pool_tokens = layout[9]
        self.num_padded_sf_pool_tokens = layout[10]

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device="cuda")
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        group.barrier()
        torch.cuda.synchronize()

        self.x = self._view(
            layout[1], (num_max_tokens_per_rank, hidden), torch.float8_e4m3fn
        )
        self.x_sf = self._view(
            layout[2], (num_max_tokens_per_rank, hidden // 128), torch.float32
        )
        self.topk_idx = self._view(
            layout[3], (num_max_tokens_per_rank, num_topk), torch.int64
        )
        self.topk_weights = self._view(
            layout[4], (num_max_tokens_per_rank, num_topk), torch.float32
        )

    def _init_sm120(
        self,
        symm_mem,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
    ) -> None:
        ranks = group.size()
        if not 1 <= ranks <= _SM120_MAX_RANKS:
            raise ValueError(
                f"SM120 MegaMoE supports 1 to {_SM120_MAX_RANKS} ranks (one host)"
            )
        if num_experts <= 0 or num_experts % ranks:
            raise ValueError("num_experts must divide evenly across ranks")
        if num_experts // ranks > _SM120_MAX_LOCAL_EXPERTS:
            raise ValueError(
                f"SM120 MegaMoE supports at most {_SM120_MAX_LOCAL_EXPERTS} experts per rank"
            )
        if not 0 < num_topk <= min(_SM120_MAX_TOPK, num_experts):
            raise ValueError(
                f"num_topk must be in [1, min({_SM120_MAX_TOPK}, num_experts)] on SM120"
            )
        if hidden <= 0 or hidden % 256 or hidden > _SM120_MAX_HIDDEN:
            raise ValueError(
                f"hidden must be a multiple of 256 and at most {_SM120_MAX_HIDDEN} on SM120"
            )
        if intermediate_hidden <= 0 or intermediate_hidden % 256:
            raise ValueError("intermediate_hidden must be a multiple of 256 on SM120")
        if not 0 < num_max_tokens_per_rank <= _SM120_MAX_TOKENS_PER_RANK:
            raise ValueError(
                f"SM120 MegaMoE is a decode kernel: num_max_tokens_per_rank must be in "
                f"[1, {_SM120_MAX_TOKENS_PER_RANK}]; use flashinfer.moe_ep (SM120 MXFP4 x MXFP8 "
                f"cutedsl / cake) for larger batches"
            )
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        layout = _sm120_layout(
            ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        self._offsets = layout
        self.buffer = symm_mem.empty(layout[0], dtype=torch.int8, device="cuda")
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        group.barrier()
        torch.cuda.synchronize()
        self.x = self._view(
            layout[1], (num_max_tokens_per_rank, hidden), torch.float8_e4m3fn
        )
        self.x_sf = self._view(
            layout[2], (num_max_tokens_per_rank, hidden // 128), torch.float32
        )
        self.topk_idx = self._view(
            layout[3], (num_max_tokens_per_rank, num_topk), torch.int64
        )
        self.topk_weights = self._view(
            layout[4], (num_max_tokens_per_rank, num_topk), torch.float32
        )

    def error(self) -> int:
        """SM120: the error word of the last call, 0 when every spin wait completed. A wait that
        times out (a rank that never entered the call) records its code here and the kernel exits
        instead of hanging; each call clears it on entry. Synchronizes with the device."""
        if not self.sm120:
            raise ValueError("error() is available on the SM120 path only")
        ctr = self._view(self._offsets[-1], (_SM120_COUNTER_INTS,), torch.int32)
        return int(ctr[_SM120_ERROR_INDEX].item())

    def _view(self, byte_offset: int, shape: Tuple[int, ...], dtype: torch.dtype):
        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        flat = self.buffer[byte_offset : byte_offset + numel * itemsize]
        return flat.view(dtype).view(*shape)

    def stage(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Copy this rank's tokens and routing into the symmetric buffer.

        Parameters
        ----------
        x_fp8 : torch.Tensor
            E4M3 activations with shape ``[tokens, hidden]``.
        x_sf : torch.Tensor
            FP32 activation scales with shape ``[tokens, hidden // 128]``.
        topk_idx : torch.Tensor
            INT64 global expert indices with shape ``[tokens, num_topk]``.
            A value of -1 masks a route.
        topk_weights : torch.Tensor
            FP32 routing weights with shape ``[tokens, num_topk]``.

        All tensors must be on the buffer's CUDA device. The token count must
        be positive and fit the allocated capacity. Every rank must stage the
        same token count; pad short ranks with rows routed to -1.
        """
        m = x_fp8.shape[0]
        for tensor, shape, dtype in (
            (x_fp8, (m, self.hidden), torch.float8_e4m3fn),
            (x_sf, (m, self.hidden // 128), torch.float32),
            (topk_idx, (m, self.num_topk), torch.int64),
            (topk_weights, (m, self.num_topk), torch.float32),
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != self.buffer.device
            ):
                raise ValueError(
                    f"staged input must have shape {shape}, dtype {dtype}, device {self.buffer.device}"
                )
        if not 0 < m <= self.num_max_tokens_per_rank:
            raise ValueError("staged token count exceeds buffer capacity")
        self.x[:m].copy_(x_fp8)
        self.x_sf[:m].copy_(x_sf)
        self.topk_idx[:m].copy_(topk_idx)
        self.topk_weights[:m].copy_(topk_weights)
        self.num_staged_tokens = m

    def commit_staged(self, num_tokens: int) -> None:
        """Mark rows written directly into the staging views as this call's input.

        Use instead of :meth:`stage` when a producer (for example a fused
        quantize-and-route kernel) already wrote the first ``num_tokens`` rows of
        ``x``, ``x_sf``, ``topk_idx`` and ``topk_weights`` on the current stream.

        Parameters
        ----------
        num_tokens : int
            Number of staged rows; positive, at most the buffer capacity, and
            equal on every rank (pad short ranks with rows routed to -1).
        """
        if self.buffer is None:
            raise ValueError("symmetric buffer has been destroyed")
        if not 0 < num_tokens <= self.num_max_tokens_per_rank:
            raise ValueError("staged token count exceeds buffer capacity")
        self.num_staged_tokens = int(num_tokens)

    def prewarm(
        self, kernel_family: str = "auto", family_threshold: Optional[int] = None
    ) -> int:
        """Build every NVFP4 module :func:`nvfp4_mega_moe` can select up to capacity.

        Collective over the buffer's group, before CUDA graph capture: ranks
        compile disjoint shards (the JIT cache is file-locked), synchronize, then
        load every module, so the first call at any token count does not compile.

        Parameters
        ----------
        kernel_family : str
            ``"auto"``, ``"fused"`` or ``"split"``, as passed to
            :func:`nvfp4_mega_moe`.
        family_threshold : int, optional
            Auto-selection threshold; defaults to ``FAMILY_THRESHOLD``.

        Returns
        -------
        int
            Number of distinct modules built.
        """
        if self.sm120:  # one shape-generic MXFP4 module
            _get_jit_module(gen_megamoe_sm120_mxfp4_module, self.fast_math)
            torch.distributed.barrier(group=self.group)
            return 1
        if kernel_family not in ("auto", "fused", "split"):
            raise ValueError(
                f"kernel_family must be 'auto', 'fused' or 'split', got {kernel_family!r}"
            )
        threshold = FAMILY_THRESHOLD if family_threshold is None else family_threshold
        geometry = (
            self.num_sms,
            self.group.size(),
            self.num_experts,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            self.num_max_tokens_per_rank,
        )
        jobs: Dict[Tuple[Any, Tuple[Any, ...]], int] = {}
        for m in range(1, self.num_max_tokens_per_rank + 1):
            fused = kernel_family == "fused" or (
                kernel_family == "auto" and m <= threshold
            )
            factory: Callable[..., Any]
            plan: Union[FusedPlan, SplitPlan]
            if fused:
                factory = gen_megamoe_sm90_nvfp4_module
                plan = select_fused_plan(*geometry, m)
            else:
                factory = gen_megamoe_sm90_nvfp4_split_module
                plan = select_split_plan(*geometry, m)
            jobs.setdefault((factory, _jit_args(factory, plan, self)), m)
        keys = sorted(jobs, key=jobs.__getitem__)
        rank, ranks = self.group.rank(), self.group.size()
        for factory, args in keys[rank::ranks]:
            _get_jit_module(factory, *args)
        torch.distributed.barrier(group=self.group)
        for factory, args in keys:
            _get_jit_module(factory, *args)
        torch.distributed.barrier(group=self.group)
        return len(keys)

    def destroy(self) -> None:
        """Release this rank's views and symmetric-memory handle."""
        self.handle = None
        self.buffer = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None


def nvfp4_mega_moe(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    symm_buffer: MegaMoESymmBuffer,
    num_tokens: Optional[int] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
    force_push_dispatch: Optional[bool] = None,
    force_no_clean_barrier: Optional[bool] = None,
    kernel_family: str = "auto",
    family_threshold: int = 256,
) -> torch.Tensor:
    """Run NVFP4 MegaMoE using the fused or split family.

    Parameters
    ----------
    y : torch.Tensor
        Contiguous BF16 output with shape ``[at least num_tokens, hidden]``.
        The kernel writes the first ``num_tokens`` rows in place.
    l1_weights, l2_weights : torch.Tensor
        Packed NVFP4 weights from :func:`transform_nvfp4_weights_for_mega_moe`.
    symm_buffer : MegaMoESymmBuffer
        Buffer populated by :meth:`MegaMoESymmBuffer.stage` on every rank.
    num_tokens : int, optional
        Tokens on this rank. Defaults to ``y.shape[0]``. Must be equal on every
        rank: the kernel family, tile and dispatch mode are selected from it, and
        all ranks must launch the same kernel. Pad short ranks with rows routed
        to -1.
    cumulative_local_expert_recv_stats : torch.Tensor, optional
        INT32 vector with one entry per local expert. Receive counts are
        accumulated in place; initialize the vector before first use.
    l1_global_scales, l2_global_scales : torch.Tensor, optional
        FP32 vectors with one multiplier per local expert. Omitted scales use 1.
    force_push_dispatch, force_no_clean_barrier : bool, optional
        Override the fused counter-dispatch policy. None uses the default
        policy. Counter dispatch is limited to the validated small-M plans
        (``num_tokens <= 64`` and ``BLOCK_M <= 24``); outside them the
        overrides are ignored and the barrier path is used. These overrides
        do not apply to the split family.
    kernel_family : str
        ``"auto"``, ``"fused"`` or ``"split"``. Auto selects fused through
        ``family_threshold`` tokens per rank, then split.
    family_threshold : int
        Token threshold used by auto selection. Defaults to 256.

    Returns
    -------
    torch.Tensor
        The output tensor ``y``.

    Notes
    -----
    All tensors must be contiguous and on the buffer's CUDA device. NVFP4
    supports four or eight ranks, which must call matching families in matching order.
    """
    b = symm_buffer
    if getattr(b, "sm120", False):
        raise ValueError("SM120 MegaMoE supports MXFP4 weights only (mxfp4_mega_moe)")
    m = int(num_tokens if num_tokens is not None else y.shape[0])
    _validate_call(
        y,
        l1_weights,
        l2_weights,
        b,
        m,
        "nvfp4",
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
    )
    if b.group.size() not in (4, 8):
        raise ValueError("NVFP4 MegaMoE requires 4 or 8 ranks")

    if kernel_family == "auto":
        kernel_family = "fused" if m <= family_threshold else "split"
    if kernel_family not in ("fused", "split"):
        raise ValueError(
            f"kernel_family must be 'auto', 'fused' or 'split', got {kernel_family!r}"
        )
    if kernel_family == "split":
        return _run_split(
            y,
            l1_weights,
            l2_weights,
            b,
            m,
            cumulative_local_expert_recv_stats,
            l1_global_scales,
            l2_global_scales,
        )

    plan = select_fused_plan(
        b.num_sms,
        b.group.size(),
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.num_max_tokens_per_rank,
        m,
        force_push_dispatch=force_push_dispatch,
        force_no_clean_barrier=force_no_clean_barrier,
    )
    return _run_kernel(
        gen_megamoe_sm90_nvfp4_module,
        "megamoe_sm90_nvfp4_fused",
        plan,
        y,
        l1_weights,
        l2_weights,
        b,
        m,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
    )


def _validate_call(y, w1, w2, b, m, weight_format, stats, gs1, gs2):
    if b.buffer is None:
        raise ValueError("symmetric buffer has been destroyed")
    if not 0 < m <= b.num_max_tokens_per_rank or m > getattr(b, "num_staged_tokens", 0):
        raise ValueError("num_tokens must fit the staged input and buffer capacity")
    epr = b.num_experts // b.group.size()
    if getattr(b, "sm120", False):
        shapes = tuple(
            (epr, stride)
            for stride in _sm120_weight_strides(b.hidden, b.intermediate_hidden)
        )
    elif weight_format == "nvfp4":
        shapes = (
            (epr, b.intermediate_hidden * 2, b.hidden // 128 * 80),
            (epr, b.hidden, b.intermediate_hidden // 128 * 80),
        )
    else:
        shapes = (
            (
                epr,
                b.intermediate_hidden * 2 // 128 * (b.hidden // 128),
                MXFP4_TILE_BYTES,
            ),
            (epr, b.hidden // 128 * (b.intermediate_hidden // 128), MXFP4_TILE_BYTES),
        )
    if (
        y.ndim != 2
        or y.shape[0] < m
        or y.shape[1] != b.hidden
        or y.dtype != torch.bfloat16
    ):
        raise ValueError("y must be BF16 [at least num_tokens, hidden]")
    for tensor, shape, dtype in (
        (w1, shapes[0], torch.uint8),
        (w2, shapes[1], torch.uint8),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{weight_format} weights must have shape {shape} and dtype {dtype}"
            )
    for tensor, dtype in (
        (stats, torch.int32),
        (gs1, torch.float32),
        (gs2, torch.float32),
    ):
        if tensor is not None and (tensor.shape != (epr,) or tensor.dtype != dtype):
            raise ValueError(
                f"optional expert tensor must have shape {(epr,)} and dtype {dtype}"
            )
    for tensor in (y, w1, w2, stats, gs1, gs2):
        if tensor is not None and (
            tensor.device != b.buffer.device or not tensor.is_contiguous()
        ):
            raise ValueError(
                "all tensors must be contiguous and on the buffer's CUDA device"
            )


def mxfp4_mega_moe(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    symm_buffer: MegaMoESymmBuffer,
    num_tokens: Optional[int] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fused MXFP4 MegaMoE with chunk-major weights and FP8 activations.

    Parameters
    ----------
    y : torch.Tensor
        Contiguous BF16 output with shape ``[at least num_tokens, hidden]``.
        The kernel writes the first ``num_tokens`` rows in place.
    l1_weights, l2_weights : torch.Tensor
        Packed MXFP4 weights from :func:`transform_mxfp4_weights_for_mega_moe`.
    symm_buffer : MegaMoESymmBuffer
        Buffer populated by :meth:`MegaMoESymmBuffer.stage` on every rank.
    num_tokens : int, optional
        Tokens on this rank. Defaults to ``y.shape[0]``. Must be equal on every
        rank: the kernel family, tile and dispatch mode are selected from it, and
        all ranks must launch the same kernel. Pad short ranks with rows routed
        to -1.
    cumulative_local_expert_recv_stats : torch.Tensor, optional
        INT32 vector with one entry per local expert. Receive counts are
        accumulated in place; initialize the vector before first use.
    l1_global_scales, l2_global_scales : torch.Tensor, optional
        FP32 vectors with one multiplier per local expert. Omitted scales use 1.

    Returns
    -------
    torch.Tensor
        The output tensor ``y``.

    Notes
    -----
    All tensors must be contiguous and on the buffer's CUDA device. On SM90 the
    rank count must be a power of two from 1 to 32. Every rank must invoke the
    kernel in matching order; MXFP4 always uses the fused family.

    On SM120 one host's ranks (1 to 8) take part over PCIe peer-to-peer; the
    weights come from :func:`transform_mxfp4_weights_for_mega_moe` on the same
    device. The kernel targets decode-size batches: ``num_max_tokens_per_rank``
    is limited to 128 (beyond that the ``flashinfer.moe_ep`` SM120 backends are
    faster), and the expert pools hold that many rows per (local expert, source
    rank). Top-k ids outside ``[0, num_experts)`` and
    repeats of an earlier id of the same token are treated as masked (-1). A
    rank whose peers never enter the call records an error code
    (:meth:`MegaMoESymmBuffer.error`) instead of hanging.
    """
    b = symm_buffer
    m = int(num_tokens if num_tokens is not None else y.shape[0])
    _validate_call(
        y,
        l1_weights,
        l2_weights,
        b,
        m,
        "mxfp4",
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
    )
    if b.sm120:
        module = _get_jit_module(gen_megamoe_sm120_mxfp4_module, b.fast_math)
        module.megamoe_sm120_mxfp4_fused(
            y,
            b.buffer,
            b.handle.buffer_ptrs,
            b.group.rank(),
            l1_weights,
            l2_weights,
            b._offsets,
            [
                b.num_experts,
                b.num_topk,
                b.hidden,
                b.intermediate_hidden,
                b.num_max_tokens_per_rank,
            ],
            m,
            float(b.activation_clamp),
            cumulative_local_expert_recv_stats,
            l1_global_scales,
            l2_global_scales,
        )
        return y
    plan = select_mxfp4_fused_plan(
        b.num_sms,
        b.group.size(),
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.num_max_tokens_per_rank,
        m,
    )
    return _run_kernel(
        gen_megamoe_sm90_mxfp4_module,
        "megamoe_sm90_mxfp4_fused",
        plan,
        y,
        l1_weights,
        l2_weights,
        b,
        m,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
    )


# ---------------------------------------------------------------------------
# Split (BN128) family -- selected above the fused threshold
# ---------------------------------------------------------------------------

#: Default token threshold for selecting NVFP4 split instead of fused.
FAMILY_THRESHOLD = 256


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    l1_num_stages: int
    l1_smem_size: int
    l2_num_stages: int
    l2_smem_size: int
    num_experts_per_wave: int
    num_max_pool_tokens: int
    dispatch_dequant: bool
    l2_arrival_counter: bool
    num_padded_sf_pool_tokens: int


@functools.cache
def select_split_plan(
    num_sms,
    num_ranks,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    num_max_tokens_per_rank,
    num_tokens,
) -> SplitPlan:
    """Ask the plan module for the L1/L2 pipeline configuration.

    The sizing arithmetic mirrors the kernel's shared-memory layout exactly, so
    it is evaluated by the same C++ code the kernel is built from rather than
    being re-derived here.
    """
    sf_pool = max(
        get_num_padded_sf_pool_tokens(
            get_num_max_pool_tokens(
                num_ranks, num_max_tokens_per_rank, num_topk, num_experts // num_ranks
            ),
            bm,
        )
        for bm in CANDIDATE_BLOCK_M
    )
    mod = _get_jit_module(
        gen_megamoe_sm90_nvfp4_split_plan_module,
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    v = list(
        mod.megamoe_sm90_nvfp4_split_plan(
            num_sms,
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_tokens,
            num_topk,
            hidden,
            intermediate_hidden,
            sf_pool,
        )
    )
    return SplitPlan(
        l1_num_stages=v[0],
        l1_smem_size=v[1],
        l2_num_stages=v[2],
        l2_smem_size=v[3],
        num_experts_per_wave=v[4],
        num_max_pool_tokens=v[5],
        dispatch_dequant=bool(v[6]),
        l2_arrival_counter=bool(v[7]),
        num_padded_sf_pool_tokens=sf_pool,
    )


def _run_split(y, l1_weights, l2_weights, b, m, stats, gs1, gs2):
    plan = select_split_plan(
        b.num_sms,
        b.group.size(),
        b.num_experts,
        b.num_topk,
        b.hidden,
        b.intermediate_hidden,
        b.num_max_tokens_per_rank,
        m,
    )
    return _run_kernel(
        gen_megamoe_sm90_nvfp4_split_module,
        "megamoe_sm90_nvfp4_split",
        plan,
        y,
        l1_weights,
        l2_weights,
        b,
        m,
        stats,
        gs1,
        gs2,
    )
