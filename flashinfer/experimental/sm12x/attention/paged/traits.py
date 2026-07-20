# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/paged/traits.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""FlashInfer-inspired forward trait selection for the primary paged backend."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from flashinfer.experimental.sm12x._lib.smem import make_tma_aligned_payload_storage

from .planner import PagedPlan

_FP8_KV_DTYPE = torch.float8_e4m3fn


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
        shared_storage_bytes = 49152
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
    cta_sync_o_bytes = (
        4 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * sync_o_row_stride * 4
    )
    cta_sync_md_bytes = 8 if num_warps_kv == 1 else num_warps_kv * cta_tile_q * 8
    cta_sync_storage_bytes = cta_sync_o_bytes + cta_sync_md_bytes
    smem_o_bytes = cta_tile_q * head_dim_vo * o_bytes

    # Choose the MMA tile and residency together from exact CUTLASS typed
    # allocations. This preserves the occupancy-first policy without deriving
    # a per-CTA budget from assumed residency and feeding it back into the tile.
    # The primary cta_q=16 backend is the exact-plane 64-row decode family; a
    # larger KV tile disables that ingress path, so it is not a valid candidate.
    max_candidate_mma_kv = 1 if cta_tile_q == 16 else max_num_mma_kv_reg
    candidates: list[tuple[int, int, int, int, int]] = []
    for candidate_mma_kv in range(1, max_candidate_mma_kv + 1):
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
            max(candidate_qkv_bytes, cta_sync_storage_bytes, smem_o_bytes),
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
    return select_paged_forward_traits(
        cta_tile_q=plan.cta_tile_q,
        head_dim_qk=plan.head_dim_qk,
        head_dim_vo=plan.head_dim_vo,
        q_dtype=plan.dtype,
        kv_dtype=plan.kv_dtype,
        o_dtype=o_dtype,
        device=plan.device,
    )
