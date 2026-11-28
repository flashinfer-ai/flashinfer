# Copyright (c) 2026 by FlashInfer team.
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

"""QToken-KvBlock-Sparse group sizes: legality, the auto policy and Q offsets.

The public entry points live in ``prims_ts.decode``; this module holds their
implementation and the group checks shared with decode planning.
"""

from collections.abc import Callable
from dataclasses import replace
import functools
from typing import TYPE_CHECKING, Optional, Union

import torch

from ...common.attention_args import SUPPORTED_PAGE_SIZES, validate_positive_int
from ...common.tensors import attention_dtype_key
from ..decode.validation import (
    GENERIC_HEAD_DIMS,
    SUPPORTED_INPUT_DTYPES,
    validate_head_dim,
    validate_head_geometry,
)
from . import selection as sparse_selection

if TYPE_CHECKING:
    from ..decode.kernel.config import FmhaDecodeConfig


def cpu_query_starts(
    query_start_loc_cpu: Optional[torch.Tensor], num_query_tokens: int
) -> list[int]:
    """Validate CPU request boundaries and return them as Python ints.

    The boundaries start at 0, never decrease and end at or below
    ``num_query_tokens``; rows past the last boundary are graph padding.
    """

    if (
        not isinstance(query_start_loc_cpu, torch.Tensor)
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.numel() < 2
        or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention requires CPU int32/int64 query_start_loc with at "
            "least two entries"
        )

    query_starts = [int(value) for value in query_start_loc_cpu.tolist()]
    if (
        query_starts[0] != 0
        or query_starts[-1] < 0
        or query_starts[-1] > num_query_tokens
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention query boundaries do not cover valid rows"
        )
    if any(
        end < begin for begin, end in zip(query_starts, query_starts[1:], strict=False)
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention query boundaries must be nondecreasing"
        )
    return query_starts


def make_qo_indptr(
    query_start_loc_cpu: torch.Tensor,
    num_query_tokens: int,
    *,
    group_size: int,
    device: Optional[Union[int, str, torch.device]],
) -> torch.Tensor:
    """Implementation of ``make_q_token_kv_block_sparse_qo_indptr``."""

    group_size = validate_group_value(group_size)
    num_query_tokens = validate_positive_int(num_query_tokens, "num_query_tokens")
    request_offsets = cpu_query_starts(query_start_loc_cpu, num_query_tokens)

    route_offsets = [0]
    for begin, end in zip(request_offsets, request_offsets[1:], strict=False):
        next_offset = begin
        while next_offset < end:
            next_offset = min(next_offset + group_size, end)
            route_offsets.append(next_offset)

    # Keep graph-padding rows out of the final real request's route even when
    # that request ends with a partial group.
    next_offset = request_offsets[-1]
    while next_offset < num_query_tokens:
        next_offset = min(next_offset + group_size, num_query_tokens)
        route_offsets.append(next_offset)

    return torch.tensor(
        validate_route_offsets_cpu(
            route_offsets,
            num_query_tokens=num_query_tokens,
            group_size=group_size,
        ),
        dtype=torch.int32,
        device=device,
    )


def nvfp4_keeps_launch_supported(launch: "sparse_selection.SparseLaunch") -> bool:
    """Whether a Keeps launch has the geometry the NVFP4 Keeps transform writes.

    The transform fills one-instance KV128 tiles with one D256 stage per tile
    (nvfp4_keeps_kv_profile sets the stage width).
    """
    return (
        launch.use_keeps_mma_ab
        and launch.num_insts_kv == 1
        and sparse_selection.TILE_SIZE_KV == 128
    )


def resolve_decode_config(
    make_config: Callable[..., "FmhaDecodeConfig"],
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    group_size: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    out_dtype_key: str,
    mask_type: str,
    packed_query: bool,
    window_left: int,
    max_active_clusters: int,
    use_pdl: bool,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> "FmhaDecodeConfig":
    """Resolve the private encoded-page QToken-KvBlock-Sparse-Attention profile without compiling it.

    NVFP4 K/V is transformed into Q's compute dtype before MMA. The transform
    targets TMEM on the SwapsMmaAb recipe, or SMEM K/V rings on the KeepsMmaAb
    recipe where ``prefers_nvfp4_keeps`` selects it (from Q's dtype and the
    group's size and rows). The cache layout (interleaved rows or separate
    scale tensors) does not affect the choice.
    """

    from ..decode.kernel.device.constants import MIN_LOOP_ITERS_PER_SPLIT
    from ..decode.kernel.config import nvfp4_keeps_kv_profile

    heads_q_per_kv = num_qo_heads // num_kv_heads
    validate_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )
    if group_size > 8 and not (
        packed_query and page_size <= sparse_selection.MAX_WIDE_GROUP_BLOCK_SIZE
    ):
        # G16/G32 memberships are read from page-major GMEM planes only (the
        # fixed-Q layout caches one plane in SMEM), four kernel pages per word.
        # ``page_size`` here is the kernel page: the K/V fragment
        # gcd(kv_block_size, storage page size) that holds one membership byte.
        raise ValueError(
            f"QToken-KvBlock-Sparse-Attention group_size={group_size} needs packed "
            "queries (qo_indptr) and K/V fragments, gcd(kv_block_size, page_size), "
            f"of at most {sparse_selection.MAX_WIDE_GROUP_BLOCK_SIZE} tokens; got "
            f"{'packed' if packed_query else 'fixed'} queries and {page_size}-token "
            "fragments"
        )
    # This route has no QK-BF16/PV-FP8 kernel, so Q, K, and V must match,
    # except NVFP4 K/V, which is transformed exactly to BF16 for BF16 Q.
    q_token_kv_block_sparse_dtype_supported = (
        q_dtype_key == k_dtype_key == v_dtype_key
        and (
            q_dtype_key == out_dtype_key
            and q_dtype_key in ("float16", "bfloat16")
            or (
                q_dtype_key == "float8_e4m3fn"
                and out_dtype_key in ("float16", "bfloat16")
            )
        )
    ) or (
        k_dtype_key == v_dtype_key == "float4_e2m1fn"
        and q_dtype_key in ("bfloat16", "float8_e4m3fn")
        and out_dtype_key == "bfloat16"
    )
    use_nvfp4_kv = k_dtype_key == "float4_e2m1fn"
    if use_nvfp4_kv and head_dim != 256:
        # The NVFP4 transform stages two 128-channel halves of a 144-byte
        # interleaved row; other widths have no staging or 16-byte row layout.
        raise ValueError(
            "PrimTS QToken-KvBlock-Sparse-Attention supports NVFP4 K/V only "
            f"with head_dim=256; got head_dim={head_dim}"
        )
    if not (
        page_size in SUPPORTED_PAGE_SIZES
        and head_dim in GENERIC_HEAD_DIMS
        and q_token_kv_block_sparse_dtype_supported
        and mask_type == "causal"
        and window_left < 0
    ):
        raise ValueError(
            "PrimTS QToken-KvBlock-Sparse-Attention requires a supported sparse-fragment size, head_dim in {64,128,192,256}, "
            "FP16/BF16 Q/K/V with matching output, FP8 Q/K/V with FP16/BF16 output "
            "or BF16/FP8 Q with NVFP4 K/V and BF16 output, "
            "and a causal non-windowed mask"
        )

    select_launch = functools.partial(
        sparse_selection.select_sparse_launch,
        group_size=group_size,
        heads_q_per_kv=heads_q_per_kv,
        head_dim=head_dim,
        num_routes=batch_size,
        num_kv_heads=num_kv_heads,
        route_kv_tokens=max_kv_len,
        multi_processor_count=max_active_clusters,
        split_kv=split_kv,
        nvfp4_kv=use_nvfp4_kv,
    )
    nvfp4_keeps = use_nvfp4_kv and sparse_selection.prefers_nvfp4_keeps(
        q_dtype_key, group_size, heads_q_per_kv
    )
    if nvfp4_keeps:
        # The Keeps target follows Q's dtype policy, preferring Keeps even
        # where Swaps would split KV (it splits with its own one-instance
        # fanout).
        launch = select_launch(q_dtype_key=q_dtype_key, prefer_keeps_mma_ab=True)
        # The transform writes one D256 FP8 or BF16 stage per KV128 tile of
        # the one-instance recipe; other launches take Swaps.
        nvfp4_keeps = nvfp4_keeps_launch_supported(launch)
    if not nvfp4_keeps:
        if (
            use_nvfp4_kv
            and group_size * heads_q_per_kv > sparse_selection.MAX_SWAPS_TILE_SIZE_Q
        ):
            raise ValueError(
                f"NVFP4 K/V with {group_size * heads_q_per_kv} Q rows per KV head "
                "needs the KeepsMmaAb transform target: the SwapsMmaAb recipes "
                f"hold at most {sparse_selection.MAX_SWAPS_TILE_SIZE_Q} Q rows"
            )
        # NVFP4 otherwise uses the TMEM-transformed Swaps recipe for either Q
        # dtype; the FP8-only Keeps fallback applies to raw FP8 caches.
        launch = select_launch(
            q_dtype_key="bfloat16" if use_nvfp4_kv else q_dtype_key,
            prefer_keeps_mma_ab=False,
        )
    if use_nvfp4_kv and not nvfp4_keeps:
        # The NVFP4 transform needs a complete warp group: each warp may only
        # address its own 32-lane TMEM quarter. One KV instance frees
        # Softmax1's group for it (interleaved rows move it to warps 16-23,
        # see nvfp4_interleaved_kv_profile).
        launch = replace(launch, num_insts_kv=1, head_dim_per_stage_kv=128)

    q_token_kv_block_sparse_profile = {
        "use_variable_seqlens_q": packed_query,
        "use_q_token_kv_block_sparse_route": True,
        "share_pattern_across_kv_heads": share_pattern_across_kv_heads,
        "use_pdl": use_pdl,
        "use_keeps_mma_ab": launch.use_keeps_mma_ab,
        "groups_tokens_heads_q": True,
        "tile_size_q": launch.tile_size_q,
        "tile_size_kv": sparse_selection.TILE_SIZE_KV,
        "head_dim_per_stage_kv": launch.head_dim_per_stage_kv,
        "num_insts_kv": launch.num_insts_kv,
        "use_persistent_scheduler": launch.use_persistent_scheduler,
        "correction_num_warps": 4,
        "mma_warp_idx": 12,
        # CLC uses the control slot beside MMA; page producers occupy the
        # remaining slots. TMA issuers keep their own complete warp groups.
        "scheduler_warp_idx": 13,
        "clc_load_warp_idx": 16,
        "page_offsets_warp_idx": 14,
        "page_offsets_num_warps": 2 if group_size == 1 else 1,
        "load_warp_idx": 16,
        # FP8's predicated TMA helper constructs coordinates in every load
        # lane, so Q1 needs only one producer warpgroup. BF16 distributes page
        # fragments across two producer warpgroups. Grouped routes keep
        # their qualified two-warpgroup schedule for both dtypes.
        "load_num_warps": (
            4 if group_size == 1 and q_dtype_key == "float8_e4m3fn" else 8
        ),
    }
    if launch.use_keeps_mma_ab:
        q_token_kv_block_sparse_profile["o_stages"] = launch.o_stages
    if nvfp4_keeps:
        q_token_kv_block_sparse_profile.update(
            nvfp4_keeps_kv_profile(
                tile_size_q=launch.tile_size_q,
                q_dtype_bytes=1 if q_dtype_key == "float8_e4m3fn" else 2,
            )
        )
    elif use_nvfp4_kv:
        # Softmax, transform and correction each own one warp group.
        # Four loader warps keep NVFP4 fragment issue ahead of the transform
        # while leaving registers to Softmax.
        q_token_kv_block_sparse_profile.update(
            transform_kv_warp_idx=4,
            transform_kv_num_warps=4,
            correction_warp_idx=8,
            load_num_warps=4,
        )
    if group_size > 8:
        # Swaps reads one membership byte plane; the policy takes Keeps here.
        assert launch.use_keeps_mma_ab
        q_token_kv_block_sparse_profile["q_token_membership_planes"] = (
            sparse_selection.membership_planes(group_size)
        )
    return make_config(
        q_token_kv_block_sparse_profile,
        split_kv_mode=(
            "gmem_reduction_with_separate_kernel"
            if launch.splits_kv > 1
            else "disabled"
        ),
        splits_kv=launch.splits_kv,
        max_splits_kv=launch.splits_kv,
        min_loop_iters_per_split=(1 if group_size == 1 else MIN_LOOP_ITERS_PER_SPLIT),
    )


def suggest_group_size(
    batch_size: int,
    seq_len_q: int,
    selected_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    multi_processor_count: int,
    *,
    head_dim: int,
    q_dtype: torch.dtype,
    split_kv: bool,
    kv_block_size: int,
    union_exponent: float,
    nvfp4_kv: bool,
    seq_len_kv: Optional[int],
    packed_query: bool,
    nvfp4_interleaved_kv: bool = False,
) -> int:
    """Implementation of ``suggest_q_token_kv_block_sparse_group_size``."""

    from .metadata import validate_sparse_block_size

    kv_block_size = validate_sparse_block_size(kv_block_size)
    batch_size = validate_positive_int(batch_size, "batch_size")
    seq_len_q = validate_positive_int(seq_len_q, "seq_len_q")
    selected_seq_len_kv = validate_positive_int(
        selected_seq_len_kv, "selected_seq_len_kv"
    )
    multi_processor_count = validate_positive_int(
        multi_processor_count, "multi_processor_count"
    )
    validate_head_geometry(num_qo_heads, num_kv_heads)
    heads_q_per_kv = num_qo_heads // num_kv_heads

    head_dim = validate_head_dim(head_dim)
    if q_dtype not in SUPPORTED_INPUT_DTYPES:
        raise ValueError("q_dtype must be float16, bfloat16, or float8_e4m3fn")
    if not isinstance(split_kv, bool):
        raise TypeError("split_kv must be a bool")
    if not isinstance(packed_query, bool):
        raise TypeError("packed_query must be a bool")
    if not isinstance(nvfp4_interleaved_kv, bool):
        raise TypeError("nvfp4_interleaved_kv must be a bool")
    kv_tokens = (
        0 if seq_len_kv is None else validate_positive_int(seq_len_kv, "seq_len_kv")
    )
    return sparse_selection.suggest_sparse_group(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        selected_kv_tokens=selected_seq_len_kv,
        heads_q_per_kv=heads_q_per_kv,
        num_kv_heads=num_kv_heads,
        multi_processor_count=multi_processor_count,
        head_dim=head_dim,
        q_dtype_key=attention_dtype_key(q_dtype),
        split_kv=split_kv,
        block_size=kv_block_size,
        union_exponent=float(union_exponent),
        nvfp4_kv=bool(nvfp4_kv),
        kv_tokens=kv_tokens,
        packed_query=packed_query,
        nvfp4_interleaved_kv=nvfp4_interleaved_kv,
    )


def validate_group_capacity(
    group_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate that one caller-selected sparse Q group fits TileQ128 and four membership planes."""

    from ..decode.kernel.device.constants import (
        Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS,
    )

    group_size = validate_group_value(group_size)
    validate_head_geometry(num_qo_heads, num_kv_heads)
    heads_q_per_kv = num_qo_heads // num_kv_heads
    # Up to four byte planes hold G9-G32 memberships (queries 8p..8p+7 in plane p).
    max_group_size = min(
        4 * Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS,
        sparse_selection.MAX_TILE_SIZE_Q // heads_q_per_kv,
    )
    if group_size > max_group_size:
        raise ValueError(
            f"QToken-KvBlock-Sparse-Attention group_size={group_size} exceeds the TileQ128/head capacity "
            f"of {max_group_size}"
        )

    return group_size


def validate_group_layout(
    group_size: int,
    query_start_loc_cpu: Optional[torch.Tensor],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate one caller-selected QToken-KvBlock-Sparse-Attention group against the live query rows."""

    group_size = validate_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )
    num_query_tokens = validate_positive_int(num_query_tokens, "num_query_tokens")
    if group_size == 1:
        return 1
    query_starts = cpu_query_starts(query_start_loc_cpu, num_query_tokens)
    if query_starts[-1] == 0:
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention requires at least one nonempty request"
        )

    return group_size


def validate_group_size(
    query_start_loc_cpu: Optional[torch.Tensor],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    *,
    group_size: int,
) -> int:
    """Implementation of ``validate_q_token_kv_block_sparse_group_size``."""

    return validate_group_layout(
        group_size,
        query_start_loc_cpu,
        num_query_tokens,
        num_qo_heads,
        num_kv_heads,
    )


def validate_group_value(group_size: int) -> int:
    """Validate one caller-selected QToken-KvBlock-Sparse-Attention grouping value."""

    group_size = validate_positive_int(group_size, "group_size")
    if group_size not in sparse_selection.SUPPORTED_GROUP_SIZES:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention group_size must be one of "
            f"{sparse_selection.SUPPORTED_GROUP_SIZES}, got {group_size}"
        )
    return group_size


def validate_page_memberships(
    q_token_kv_block_sparse_page_memberships: torch.Tensor,
    *,
    expected_device: torch.device,
    batch_size: int,
    max_num_pages: int,
    planes: int = 1,
) -> None:
    """Validate the private grouped-QToken-KvBlock-Sparse-Attention packed-membership table.

    G9-G32 rows hold ``planes`` byte planes (queries 8p..8p+7 in plane p).
    """

    if not isinstance(q_token_kv_block_sparse_page_memberships, torch.Tensor):
        raise TypeError(
            "q_token_kv_block_sparse_page_memberships must be a torch.Tensor"
        )
    if q_token_kv_block_sparse_page_memberships.ndim != 2:
        raise ValueError(
            f"q_token_kv_block_sparse_page_memberships must be rank 2, got rank {q_token_kv_block_sparse_page_memberships.ndim}"
        )
    if q_token_kv_block_sparse_page_memberships.dtype != torch.int32:
        raise TypeError(
            "q_token_kv_block_sparse_page_memberships must have dtype torch.int32"
        )
    if q_token_kv_block_sparse_page_memberships.device != expected_device:
        raise ValueError(
            f"q_token_kv_block_sparse_page_memberships must be on {expected_device}, got "
            f"{q_token_kv_block_sparse_page_memberships.device}"
        )
    if not q_token_kv_block_sparse_page_memberships.is_contiguous():
        raise ValueError("q_token_kv_block_sparse_page_memberships must be contiguous")
    expected_shape = (batch_size, planes * ((max_num_pages + 3) // 4))
    if tuple(q_token_kv_block_sparse_page_memberships.shape) != expected_shape:
        raise ValueError(
            "q_token_kv_block_sparse_page_memberships must have shape "
            f"{expected_shape}, got {tuple(q_token_kv_block_sparse_page_memberships.shape)}"
        )
    if q_token_kv_block_sparse_page_memberships.data_ptr() % 4 != 0:
        raise ValueError(
            "q_token_kv_block_sparse_page_memberships data pointer must be 4-byte aligned"
        )


def validate_route_offsets_cpu(
    route_offsets: list[int],
    *,
    num_query_tokens: int,
    group_size: int,
) -> tuple[int, ...]:
    """Validate CPU-generated QToken-KvBlock-Sparse-Attention routes before copying them to device storage."""

    offsets = tuple(route_offsets)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != num_query_tokens:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention route offsets must cover exactly all query tokens"
        )
    lengths = tuple(
        end - begin for begin, end in zip(offsets, offsets[1:], strict=False)
    )
    if any(length <= 0 or length > group_size for length in lengths):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention route offsets must describe nonempty routes no longer than group_size"
        )
    return offsets
