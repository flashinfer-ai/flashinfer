"""FlashInfer-inspired forward trait selection for the primary paged backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.smem import make_tma_aligned_payload_storage

from .planner import PagedPlan
from ._controls import paged_control

_FP8_KV_DTYPE = torch.float8_e4m3fn

_BF16_EXTEND_LOW_SM_MAX_SMS = 64
_BF16_EXTEND_WIDE_TILE_MIN_NARROW_ITERS_PER_LOST_CTA = 704
_BF16_EXTEND_HIGH_SM_MIN_NARROW_ITERS = 3


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == _FP8_KV_DTYPE:
        return 1
    raise TypeError(f"unsupported dtype {dtype}")


def paged_get_num_warps_q(cta_tile_q: int) -> int:
    return 4 if cta_tile_q > 16 else 1


def paged_get_num_warps_kv(cta_tile_q: int) -> int:
    return 4 // paged_get_num_warps_q(cta_tile_q)


def paged_get_num_mma_q(cta_tile_q: int) -> int:
    return 2 if cta_tile_q > 64 else 1


@dataclass(frozen=True)
class PagedForwardTraits:
    cta_tile_q: int
    cta_tile_kv: int
    num_mma_q: int
    num_mma_kv: int
    num_mma_d_qk: int
    num_mma_d_vo: int
    num_warps_q: int
    num_warps_kv: int
    num_threads: int
    head_dim_qk: int
    head_dim_vo: int
    upcast_stride_q: int
    upcast_stride_k: int
    upcast_stride_v: int
    upcast_stride_o: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    o_dtype: torch.dtype
    q_smem_bytes: int
    shared_storage_bytes: int
    max_smem_per_sm: int
    num_ctas_per_sm: int
    max_smem_per_threadblock: int

    @property
    def uses_fp8_kv(self) -> bool:
        return self.kv_dtype == _FP8_KV_DTYPE

    @property
    def launch_smem_bytes(self) -> int:
        """Typed one-stage TMA allocation used by the primary decode kernel."""
        return int(
            make_tma_aligned_payload_storage(
                payload_bytes=self.shared_storage_bytes,
                num_stages=1,
            ).size_in_bytes()
        )


def _paged_is_invalid(
    *,
    num_mma_q: int,
    num_mma_kv: int,
    num_mma_d_vo: int,
    num_warps_q: int,
    kv_dtype: torch.dtype,
) -> bool:
    kv_is_fp8 = kv_dtype == _FP8_KV_DTYPE
    if num_mma_d_vo < 4:
        return True
    if num_mma_d_vo == 4 and num_mma_kv % 2 == 1:
        return True
    if num_mma_q * (8 * num_mma_d_vo + 8 * num_mma_kv) >= 256:
        return True
    if kv_is_fp8 and (num_mma_kv * 2) % num_warps_q != 0:
        return True
    return False


def select_paged_forward_traits(
    *,
    cta_tile_q: int,
    head_dim_qk: int,
    head_dim_vo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    o_dtype: torch.dtype | None = None,
    device: torch.device | int | None = None,
    exact_num_mma_kv: int | None = None,
    minimum_shared_storage_bytes: int = 0,
    compact_sync_rows: int = 0,
) -> PagedForwardTraits:
    if head_dim_qk % 16 != 0 or head_dim_vo % 16 != 0:
        raise ValueError("head_dim_qk and head_dim_vo must be multiples of 16")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported q dtype {q_dtype}")
    if kv_dtype not in (torch.float16, torch.bfloat16, _FP8_KV_DTYPE):
        raise TypeError(f"unsupported kv dtype {kv_dtype}")
    if kv_dtype == _FP8_KV_DTYPE and q_dtype != torch.bfloat16:
        raise TypeError("primary paged backend only supports bf16 queries with fp8 kv")
    o_dtype = q_dtype if o_dtype is None else o_dtype
    if o_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported output dtype {o_dtype}")
    minimum_shared_storage_bytes = int(minimum_shared_storage_bytes)
    if minimum_shared_storage_bytes < 0:
        raise ValueError("minimum_shared_storage_bytes must be nonnegative")
    compact_sync_rows = int(compact_sync_rows)
    if compact_sync_rows < 0 or compact_sync_rows > cta_tile_q:
        raise ValueError(
            "compact_sync_rows must be between zero and cta_tile_q, got "
            f"{compact_sync_rows}"
        )

    if kv_dtype == _FP8_KV_DTYPE and cta_tile_q == 48:
        device_props = torch.cuda.get_device_properties(
            torch.cuda.current_device() if device is None else device
        )
        max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)
        max_smem_per_block = min(
            max_smem_per_sm,
            int(
                getattr(device_props, "shared_memory_per_block_optin", max_smem_per_sm)
            ),
        )
        kv_bytes = _dtype_num_bytes(kv_dtype)
        upcast_stride_k = _align_up(head_dim_qk // (16 // kv_bytes), 8)
        upcast_stride_v = _align_up(head_dim_vo // (16 // kv_bytes), 8)
        shared_storage_bytes = max(49152, minimum_shared_storage_bytes)
        launch_smem_bytes = int(
            make_tma_aligned_payload_storage(
                payload_bytes=shared_storage_bytes,
                num_stages=1,
            ).size_in_bytes()
        )
        if launch_smem_bytes > max_smem_per_block:
            raise ValueError(
                f"paged forward typed shared storage requires {launch_smem_bytes} B, "
                f"but the device permits {max_smem_per_block} B per CTA"
            )
        num_ctas_per_sm = 2 if 2 * launch_smem_bytes <= max_smem_per_sm else 1
        return PagedForwardTraits(
            cta_tile_q=48,
            cta_tile_kv=32,
            num_mma_q=1,
            num_mma_kv=2,
            num_mma_d_qk=head_dim_qk // 16,
            num_mma_d_vo=head_dim_vo // 16,
            num_warps_q=3,
            num_warps_kv=1,
            num_threads=96,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            upcast_stride_q=head_dim_qk // 8,
            upcast_stride_k=upcast_stride_k,
            upcast_stride_v=upcast_stride_v,
            upcast_stride_o=head_dim_vo // (16 // _dtype_num_bytes(o_dtype)),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            q_smem_bytes=48 * head_dim_qk * _dtype_num_bytes(q_dtype),
            shared_storage_bytes=shared_storage_bytes,
            max_smem_per_sm=max_smem_per_sm,
            num_ctas_per_sm=num_ctas_per_sm,
            max_smem_per_threadblock=min(
                max_smem_per_block,
                max_smem_per_sm // num_ctas_per_sm,
            ),
        )

    num_mma_d_qk = head_dim_qk // 16
    num_mma_d_vo = head_dim_vo // 16
    num_warps_q = paged_get_num_warps_q(cta_tile_q)
    num_warps_kv = paged_get_num_warps_kv(cta_tile_q)
    num_mma_q = paged_get_num_mma_q(cta_tile_q)

    device_props = torch.cuda.get_device_properties(
        torch.cuda.current_device() if device is None else device
    )
    max_smem_per_sm = int(device_props.shared_memory_per_multiprocessor)
    max_smem_per_block = min(
        max_smem_per_sm,
        int(getattr(device_props, "shared_memory_per_block_optin", max_smem_per_sm)),
    )

    q_bytes = _dtype_num_bytes(q_dtype)
    kv_bytes = _dtype_num_bytes(kv_dtype)
    o_bytes = _dtype_num_bytes(o_dtype)
    upcast_stride_q = head_dim_qk // (16 // q_bytes)
    upcast_stride_k = head_dim_qk // (16 // kv_bytes)
    upcast_stride_v = head_dim_vo // (16 // kv_bytes)
    if kv_dtype == _FP8_KV_DTYPE:
        upcast_stride_k = _align_up(upcast_stride_k, 8)
        upcast_stride_v = _align_up(upcast_stride_v, 8)
    upcast_stride_o = head_dim_vo // (16 // o_bytes)
    q_smem_bytes = cta_tile_q * head_dim_qk * q_bytes
    kv_bytes_per_mma = (upcast_stride_k + upcast_stride_v) * 16 * 16 * num_warps_kv
    max_num_mma_kv_reg = 8 // num_mma_q
    sync_o_row_stride = head_dim_vo + (
        24
        if (
            kv_dtype != _FP8_KV_DTYPE and o_dtype == torch.bfloat16 and cta_tile_q == 16
        )
        else 0
    )
    if compact_sync_rows:
        if num_warps_kv <= 1:
            raise ValueError("compact sync storage requires multiple KV warps")
        # The exact single-row decode merge keeps warp zero's partial in
        # registers. Only the remaining KV warps spill the statically known
        # valid GQA rows.
        spilled_warps = num_warps_kv - 1
        cta_sync_o_bytes = (
            spilled_warps * compact_sync_rows * sync_o_row_stride * 4
        )
        cta_sync_md_bytes = spilled_warps * compact_sync_rows * 8
        # The compact epilogue keeps warp zero in registers and spills only
        # the other KV warps.  Its cooperative BF16 store must use a separate
        # staging plane: aliasing that plane onto the first spilled warp is
        # unsafe once H256 needs more than one cooperative store iteration.
        compact_store_stage_bytes = compact_sync_rows * head_dim_vo * 4
    else:
        cta_sync_o_bytes = (
            4
            if num_warps_kv == 1
            else num_warps_kv * cta_tile_q * sync_o_row_stride * 4
        )
        cta_sync_md_bytes = (
            8 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * 8
        )
        compact_store_stage_bytes = 0
    cta_sync_storage_bytes = cta_sync_o_bytes + cta_sync_md_bytes
    smem_o_bytes = cta_tile_q * head_dim_vo * o_bytes

    # Choose the MMA tile and residency together from exact CUTLASS typed
    # allocations. This preserves the occupancy-first policy without deriving
    # a per-CTA budget from assumed residency and feeding it back into the tile.
    # The general cta_q=16 backend is the exact-plane 64-row decode family.  An
    # exact graph specialization may request a wider tile explicitly; keeping
    # that choice out of the generic candidate sweep prevents unrelated decode
    # families from silently changing their register and shared-memory shape.
    if exact_num_mma_kv is not None:
        exact_num_mma_kv = int(exact_num_mma_kv)
        if exact_num_mma_kv <= 0 or exact_num_mma_kv > max_num_mma_kv_reg:
            raise ValueError(
                f"exact_num_mma_kv={exact_num_mma_kv} is outside the supported "
                f"range [1, {max_num_mma_kv_reg}]"
            )
    max_candidate_mma_kv = (
        exact_num_mma_kv
        if exact_num_mma_kv is not None
        else (1 if cta_tile_q == 16 else max_num_mma_kv_reg)
    )
    candidates: list[tuple[int, int, int, int, int]] = []
    for candidate_mma_kv in range(1, max_candidate_mma_kv + 1):
        if (
            exact_num_mma_kv is not None
            and candidate_mma_kv != exact_num_mma_kv
        ):
            continue
        if _paged_is_invalid(
            num_mma_q=num_mma_q,
            num_mma_kv=candidate_mma_kv,
            num_mma_d_vo=num_mma_d_vo,
            num_warps_q=num_warps_q,
            kv_dtype=kv_dtype,
        ):
            continue
        candidate_cta_tile_kv = candidate_mma_kv * num_warps_kv * 16
        candidate_qkv_bytes = q_smem_bytes + candidate_mma_kv * kv_bytes_per_mma
        candidate_payload_bytes = _align_up(
            max(
                candidate_qkv_bytes,
                cta_sync_storage_bytes + compact_store_stage_bytes,
                smem_o_bytes,
                minimum_shared_storage_bytes,
            ),
            16,
        )
        candidate_launch_bytes = int(
            make_tma_aligned_payload_storage(
                payload_bytes=candidate_payload_bytes,
                num_stages=1,
            ).size_in_bytes()
        )
        if candidate_launch_bytes > max_smem_per_block:
            continue
        candidate_residency = 2 if 2 * candidate_launch_bytes <= max_smem_per_sm else 1
        candidates.append(
            (
                candidate_residency,
                candidate_mma_kv,
                candidate_cta_tile_kv,
                candidate_payload_bytes,
                candidate_launch_bytes,
            )
        )

    if not candidates:
        raise ValueError(
            "no valid NUM_MMA_KV typed shared-memory allocation fits the device"
        )
    (
        num_ctas_per_sm,
        num_mma_kv,
        cta_tile_kv,
        shared_storage_bytes,
        launch_smem_bytes,
    ) = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    max_smem_per_threadblock = min(
        max_smem_per_block,
        max_smem_per_sm // num_ctas_per_sm,
    )
    if launch_smem_bytes > max_smem_per_threadblock:
        raise AssertionError(
            "selected paged typed shared-memory allocation exceeds its residency budget"
        )

    return PagedForwardTraits(
        cta_tile_q=cta_tile_q,
        cta_tile_kv=cta_tile_kv,
        num_mma_q=num_mma_q,
        num_mma_kv=num_mma_kv,
        num_mma_d_qk=num_mma_d_qk,
        num_mma_d_vo=num_mma_d_vo,
        num_warps_q=num_warps_q,
        num_warps_kv=num_warps_kv,
        num_threads=num_warps_q * num_warps_kv * 32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        upcast_stride_q=upcast_stride_q,
        upcast_stride_k=upcast_stride_k,
        upcast_stride_v=upcast_stride_v,
        upcast_stride_o=upcast_stride_o,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
        q_smem_bytes=q_smem_bytes,
        shared_storage_bytes=shared_storage_bytes,
        max_smem_per_sm=max_smem_per_sm,
        num_ctas_per_sm=num_ctas_per_sm,
        max_smem_per_threadblock=max_smem_per_threadblock,
    )


def select_paged_forward_traits_from_plan(
    plan: PagedPlan,
    *,
    o_dtype: torch.dtype | None = None,
) -> PagedForwardTraits:
    resolved_o_dtype = plan.dtype if o_dtype is None else o_dtype
    device_capability = tuple(torch.cuda.get_device_capability(plan.device))
    # Laguna's production decode graph uses one BF16 query row, 36 query heads,
    # four FP8 KV heads, and physical 128-token pages.  Select exactly two KV
    # MMAs per KV warp for that static family so each CTA consumes one physical
    # page per loop iteration.  This is plan/capacity metadata only: no live
    # sequence length participates in the choice.
    exact_num_mma_kv: int | None = None
    minimum_shared_storage_bytes = 0
    compact_sync_rows = 0
    if (
        plan.mode == "extend"
        and device_capability == (12, 0)
        and plan.enable_cuda_graph
        and not plan.msa_block_sparse
        and not plan.split_kv
        and plan.window_left == 511
        and plan.page_size == 128
        and plan.cta_tile_q == 128
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and plan.num_q_heads == 36
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 9
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == _FP8_KV_DTYPE
        and resolved_o_dtype == torch.bfloat16
    ):
        # Match the instruction-efficient verifier pipeline: one cooperative
        # FP8->BF16 widening pass per CTA, then reuse the widened N32 tile
        # across all four query warps.  N32 keeps Q + raw K/V + the reusable
        # widened K-or-V tile at two resident CTAs/SM on SM120.
        exact_num_mma_kv = 2
        minimum_shared_storage_bytes = (
            plan.cta_tile_q * plan.head_dim_qk * 2
            + 2 * 32 * plan.head_dim_qk
            + 32 * max(plan.head_dim_qk, plan.head_dim_vo) * 2
        )
    elif (
        plan.mode == "extend"
        and device_capability == (12, 0)
        and plan.enable_cuda_graph
        and not plan.msa_block_sparse
        and not plan.split_kv
        and plan.window_left < 0
        and plan.page_size == 128
        and plan.cta_tile_q == 64
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and plan.num_q_heads == 24
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 6
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == _FP8_KV_DTYPE
        and resolved_o_dtype == torch.bfloat16
    ):
        # Keep the exact Laguna prefill family on an N32 KV stage.  The
        # cooperative widening path removes per-fragment FP8 conversion state.
        exact_num_mma_kv = 2
        # Q, raw K/V (2x32x128 FP8), and one reusable 32x128 K-or-V
        # repack tile.  This is derived only from static production geometry.
        minimum_shared_storage_bytes = (
            plan.cta_tile_q * plan.head_dim_qk * 2
            + 2 * 32 * plan.head_dim_qk
            + 32 * max(plan.head_dim_qk, plan.head_dim_vo) * 2
        )
    elif (
        plan.mode == "decode"
        and paged_control("B12X_PAGED_GQA6_COMPACT_SYNC", "1") != "0"
        and device_capability[0] >= 12
        and plan.enable_cuda_graph
        and plan.split_kv
        and not plan.msa_block_sparse
        and plan.window_left < 0
        and plan.page_size in (64, 128)
        and plan.cta_tile_q == 16
        and plan.head_dim_qk in (128, 256)
        and plan.head_dim_vo == plan.head_dim_qk
        and plan.gqa_group_size == 6
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == _FP8_KV_DTYPE
        and resolved_o_dtype == torch.bfloat16
    ):
        # Single-row GQA6 decode needs only the six live query rows from the
        # three non-owner KV warps.  Derive the compact allocation from that
        # static contract for both H128 and H256 instead of keying residency
        # to a model name or exact head count.
        if (
            device_capability == (12, 0)
            and plan.page_size == 128
            and plan.head_dim_qk == 128
            and paged_control("B12X_PAGED_LAGUNA_DECODE_N128", "0") == "1"
        ):
            exact_num_mma_kv = 2
        compact_sync_rows = plan.gqa_group_size
    elif (
        plan.mode == "decode"
        and device_capability == (12, 0)
        and plan.enable_cuda_graph
        and plan.split_kv
        and not plan.msa_block_sparse
        and plan.page_size == 128
        and plan.cta_tile_q == 16
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and plan.num_q_heads == 36
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 9
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == _FP8_KV_DTYPE
        and resolved_o_dtype == torch.bfloat16
    ):
        exact_num_mma_kv = 2
    elif (
        plan.mode == "verify"
        and device_capability == (12, 0)
        and plan.enable_cuda_graph
        and plan.split_kv
        and not plan.msa_block_sparse
        and plan.window_left < 0
        and plan.page_size == 128
        and plan.cta_tile_q == 64
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and plan.num_q_heads == 24
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 6
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == _FP8_KV_DTYPE
        and resolved_o_dtype == torch.bfloat16
    ):
        # A 64-row KV stage halves the verifier's live probability fragment
        # relative to the original 128-row stage.  On SM120 this trades one
        # extra loop iteration per physical page for substantially lower
        # register and shared-memory pressure.
        exact_num_mma_kv = 4
    traits = select_paged_forward_traits(
        cta_tile_q=plan.cta_tile_q,
        head_dim_qk=plan.head_dim_qk,
        head_dim_vo=plan.head_dim_vo,
        q_dtype=plan.dtype,
        kv_dtype=plan.kv_dtype,
        o_dtype=resolved_o_dtype,
        device=plan.device,
        exact_num_mma_kv=exact_num_mma_kv,
        minimum_shared_storage_bytes=minimum_shared_storage_bytes,
        compact_sync_rows=compact_sync_rows,
    )
    force_wide_bf16_extend = (
        paged_control("B12X_PAGED_EXTEND_BF16_N32", "0") == "1"
    )
    force_narrow_bf16_extend = (
        paged_control("B12X_PAGED_EXTEND_BF16_N16", "0") == "1"
    )
    wide_bf16_extend_family = (
        exact_num_mma_kv is None
        and plan.mode == "extend"
        and plan.enable_cuda_graph
        and not plan.msa_block_sparse
        and not plan.split_kv
        and plan.cta_tile_q == 64
        and plan.head_dim_qk == 256
        and plan.head_dim_vo == 256
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == torch.bfloat16
        and resolved_o_dtype == torch.bfloat16
        and device_capability[0] >= 12
        and traits.num_mma_kv == 1
    )
    if not wide_bf16_extend_family or force_narrow_bf16_extend:
        return traits

    wide_traits = select_paged_forward_traits(
        cta_tile_q=plan.cta_tile_q,
        head_dim_qk=plan.head_dim_qk,
        head_dim_vo=plan.head_dim_vo,
        q_dtype=plan.dtype,
        kv_dtype=plan.kv_dtype,
        o_dtype=resolved_o_dtype,
        device=plan.device,
        exact_num_mma_kv=2,
        minimum_shared_storage_bytes=minimum_shared_storage_bytes,
        compact_sync_rows=compact_sync_rows,
    )
    batch_capacity = max(int(plan.page_table_shape[0]), 1)
    average_q_len = (int(plan.total_q) + batch_capacity - 1) // batch_capacity
    average_visible_kv = max(int(plan.kv_chunk_size) - average_q_len // 2, 1)
    if plan.window_left >= 0:
        average_visible_kv = min(average_visible_kv, int(plan.window_left) + 1)
    narrow_loop_iters = (
        average_visible_kv + int(traits.cta_tile_kv) - 1
    ) // int(traits.cta_tile_kv)
    lost_resident_ctas = max(
        int(traits.num_ctas_per_sm) - int(wide_traits.num_ctas_per_sm),
        1,
    )
    min_narrow_loop_iters = (
        _BF16_EXTEND_WIDE_TILE_MIN_NARROW_ITERS_PER_LOST_CTA
        * lost_resident_ctas
    )
    num_sms = int(
        torch.cuda.get_device_properties(plan.device).multi_processor_count
    )
    active_ctas = int(plan.num_qo_tiles) * int(plan.num_kv_heads)
    # N32 loses one resident CTA.  Retain N16 for the two-iteration region
    # unless the N32 grid occupies at most one quarter of the SMs.
    high_sm_wide_tile = num_sms > _BF16_EXTEND_LOW_SM_MAX_SMS and (
        narrow_loop_iters >= _BF16_EXTEND_HIGH_SM_MIN_NARROW_ITERS
        or active_ctas * 4 <= num_sms
    )
    return (
        wide_traits
        if force_wide_bf16_extend
        or high_sm_wide_tile
        or narrow_loop_iters >= min_narrow_loop_iters
        else traits
    )
