"""SM90 (Hopper) FP4 MegaMoE: NVFP4 fused/split and MXFP4 fused.

The fused paths perform cross-rank token dispatch, two W4A8 GEMMs, clamped
SwiGLU and top-k reduction in one persistent kernel. NVFP4 split uses separate
L1 and L2 launches. All paths use FP8 activations and NVLink symmetric memory.
NVFP4 supports four or eight ranks; MXFP4 supports power-of-two rank counts up to 32.

Usage::

    buf = MegaMoESymmBuffer(group, num_experts=256, num_max_tokens_per_rank=8448,
                            num_topk=6, hidden=4096, intermediate_hidden=2048)
    buf.stage(x_fp8, x_sf, topk_idx, topk_weights)
    nvfp4_mega_moe(y, l1_weights, l2_weights, buf)
"""

import dataclasses
import functools
from typing import Optional, Tuple

import torch

from ..jit.megamoe import (
    gen_megamoe_sm90_layout_module,
    gen_megamoe_sm90_mxfp4_module,
    gen_megamoe_sm90_nvfp4_module,
    gen_megamoe_sm90_nvfp4_split_module,
    gen_megamoe_sm90_nvfp4_split_plan_module,
)
from .heuristics import (
    CANDIDATE_BLOCK_M,
    FusedPlan,
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


def _run_kernel(factory, entry, plan, y, w1, w2, b, m, stats, gs1, gs2):
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
    module = _get_jit_module(factory, *args)
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

    All ranks must use identical geometry and invoke kernels in matching order.
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
        be positive and fit the allocated capacity.
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
        Tokens on this rank. Defaults to ``y.shape[0]``.
    cumulative_local_expert_recv_stats : torch.Tensor, optional
        INT32 vector with one entry per local expert. Receive counts are
        accumulated in place; initialize the vector before first use.
    l1_global_scales, l2_global_scales : torch.Tensor, optional
        FP32 vectors with one multiplier per local expert. Omitted scales use 1.
    force_push_dispatch, force_no_clean_barrier : bool, optional
        Override the fused counter-dispatch policy. None uses the default
        policy. These overrides do not apply to the split family.
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
    if weight_format == "nvfp4":
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
        Tokens on this rank. Defaults to ``y.shape[0]``.
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
    All tensors must be contiguous and on the buffer's CUDA device. The rank
    count must be a power of two from 1 to 32. Every rank must invoke the kernel
    in matching order; MXFP4 always uses the fused family.
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
