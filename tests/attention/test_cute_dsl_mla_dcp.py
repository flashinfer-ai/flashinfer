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

"""Correctness tests for static DCP support in monolithic CuTe DSL MLA."""

import math
from bisect import bisect_left
from unittest import mock

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported


_LATENT_DIM = 512
_ROPE_DIM = 64
_QK_DIM = _LATENT_DIM + _ROPE_DIM
_PAGE_SIZE = 64


def _skip_if_unsupported() -> None:
    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm110a_supported(device)):
        pytest.skip("Requires SM100-SM110 (tcgen05)")
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _local_causal_bound(
    global_bound_newest: int,
    q_len: int,
    q_idx: int,
    cp_world: int,
    cp_rank: int,
) -> int:
    return _ceil_div(
        global_bound_newest - cp_rank - (q_len - 1) + q_idx,
        cp_world,
    )


def _flat_dcp_score_is_valid(
    flat_q_row: int,
    num_heads: int,
    local_key: int,
    q_len: int,
    global_bound_newest: int,
    cp_world: int,
    cp_rank: int,
) -> bool:
    return flat_q_row >= num_heads * (
        cp_world * local_key + cp_rank - global_bound_newest + q_len
    )


def _local_length(global_length: int, cp_world: int, cp_rank: int) -> int:
    return max(_ceil_div(global_length - cp_rank, cp_world), 0)


def _make_inputs(
    *,
    global_length: int,
    q_len: int,
    num_heads: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one batch of quantized-once Q and global-coordinate KV."""
    return _make_batched_inputs(
        global_lengths=(global_length,),
        q_len=q_len,
        num_heads=num_heads,
        dtype=dtype,
    )


def _make_batched_inputs(
    *,
    global_lengths: tuple[int, ...],
    q_len: int,
    num_heads: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a heterogeneous batch with one padded global-coordinate KV pool."""
    device = torch.device("cuda")
    storage_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    batch_size = len(global_lengths)
    query = (
        torch.randn(
            batch_size,
            q_len,
            num_heads,
            _QK_DIM,
            device=device,
            dtype=storage_dtype,
        )
        * 0.1
    ).to(dtype)
    global_kv = (
        torch.randn(
            batch_size,
            max(global_lengths),
            _QK_DIM,
            device=device,
            dtype=storage_dtype,
        )
        * 0.1
    ).to(dtype)
    global_lens = torch.tensor(global_lengths, dtype=torch.int32, device=device)
    return query, global_kv, global_lens


def _pack_cyclic_rank_pages(
    global_kv: torch.Tensor,
    global_lens: torch.Tensor,
    cp_world: int,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack ``g % cp_world == cp_rank`` tokens into a contiguous paged cache."""
    batch_size, _, d_qk = global_kv.shape
    global_lens_host = global_lens.tolist()
    local_lens_host = [
        _local_length(global_len, cp_world, cp_rank) for global_len in global_lens_host
    ]
    pages_per_batch = [
        max(1, _ceil_div(local_len, _PAGE_SIZE)) for local_len in local_lens_host
    ]
    max_pages = max(pages_per_batch)
    total_pages = sum(pages_per_batch)

    local_cache = torch.zeros(
        total_pages,
        _PAGE_SIZE,
        d_qk,
        dtype=global_kv.dtype,
        device=global_kv.device,
    )
    block_tables = torch.zeros(
        batch_size,
        max_pages,
        dtype=torch.int32,
        device=global_kv.device,
    )

    next_page = 0
    for batch_idx, (global_len, local_len, num_pages) in enumerate(
        zip(
            global_lens_host,
            local_lens_host,
            pages_per_batch,
            strict=True,
        )
    ):
        page_ids = torch.arange(
            next_page,
            next_page + num_pages,
            dtype=torch.int32,
            device=global_kv.device,
        )
        block_tables[batch_idx, :num_pages] = page_ids
        if local_len:
            local_tokens = global_kv[batch_idx, cp_rank:global_len:cp_world]
            local_cache[next_page : next_page + num_pages].view(-1, d_qk)[
                :local_len
            ].copy_(local_tokens)
        next_page += num_pages

    local_lens = torch.tensor(
        local_lens_host, dtype=torch.int32, device=global_kv.device
    )
    return (
        local_cache,
        block_tables,
        local_lens,
        max(1, max(local_lens_host)),
    )


def _reference_attention(
    query: torch.Tensor,
    global_kv: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    cp_world: int = 1,
    cp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference one cyclic rank; world=1 is the full-context reference."""
    batch_size, q_len, num_heads, _ = query.shape
    out = torch.zeros(
        batch_size,
        q_len,
        num_heads,
        _LATENT_DIM,
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.full(
        (batch_size, q_len, num_heads),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    softmax_scale = 1.0 / math.sqrt(_LATENT_DIM)

    for batch_idx, global_len in enumerate(global_lens.tolist()):
        keys = global_kv[batch_idx, cp_rank:global_len:cp_world].float()
        global_positions = range(cp_rank, global_len, cp_world)
        for q_idx in range(q_len):
            global_bound = global_len - (q_len - 1) + q_idx
            # Count directly in global coordinates so this reference remains
            # independent of the ceiling-divided kernel formula without
            # synchronizing on a CUDA predicate.
            visible_count = bisect_left(global_positions, global_bound)
            if visible_count == 0:
                continue
            visible_keys = keys[:visible_count]
            scores = torch.einsum(
                "hd,kd->hk",
                query[batch_idx, q_idx].float(),
                visible_keys,
            )
            scores *= softmax_scale
            lse[batch_idx, q_idx] = torch.logsumexp(scores, dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            out[batch_idx, q_idx] = torch.einsum(
                "hk,kd->hd",
                probabilities,
                visible_keys[:, :_LATENT_DIM],
            )
    return out, lse


def _merge_rank_outputs_natural_log(
    rank_outputs: list[torch.Tensor],
    rank_lses: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable LSE-weighted merge for the kernel's natural-log public LSE."""
    outputs = torch.stack([out.float() for out in rank_outputs], dim=0)
    lses = torch.stack([lse.float() for lse in rank_lses], dim=0)
    max_lse = lses.max(dim=0).values
    has_keys = torch.isfinite(max_lse)
    safe_max = torch.where(has_keys, max_lse, torch.zeros_like(max_lse))
    weights = torch.where(
        torch.isfinite(lses),
        torch.exp(lses - safe_max.unsqueeze(0)),
        torch.zeros_like(lses),
    )
    weight_sum = weights.sum(dim=0)
    safe_sum = torch.where(has_keys, weight_sum, torch.ones_like(weight_sum))
    merged_out = (outputs * weights.unsqueeze(-1)).sum(dim=0) / safe_sum.unsqueeze(-1)
    merged_out = torch.where(
        has_keys.unsqueeze(-1), merged_out, torch.zeros_like(merged_out)
    )
    merged_lse = torch.where(
        has_keys,
        safe_max + torch.log(safe_sum),
        torch.full_like(max_lse, -torch.inf),
    )
    return merged_out, merged_lse


def _prepare_rank_call(
    query: torch.Tensor,
    global_kv: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    cp_world: int,
    cp_rank: int,
    is_var_seq: bool = False,
    enable_pdl: bool = False,
) -> tuple[dict, int]:
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
    )
    from flashinfer.cute_dsl.utils import get_num_sm

    kv_cache, block_tables, local_lens, max_local_len = _pack_cyclic_rank_pages(
        global_kv, global_lens, cp_world, cp_rank
    )
    batch_size, q_len, num_heads, _ = query.shape
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        batch_size,
        q_len,
        num_heads,
        _LATENT_DIM,
        get_num_sm(query.device),
        max_local_len,
    )
    workspace = torch.empty(
        max(workspace_size, 1), dtype=torch.int8, device=query.device
    )
    return (
        {
            "query": query,
            "kv_cache": kv_cache,
            "workspace_buffer": workspace,
            "kv_lora_rank": _LATENT_DIM,
            "qk_rope_head_dim": _ROPE_DIM,
            "block_tables": block_tables,
            "seq_lens": local_lens,
            "max_seq_len": max_local_len,
            "softmax_scale": 1.0 / math.sqrt(_LATENT_DIM),
            "is_var_seq": is_var_seq,
            "enable_pdl": enable_pdl,
        },
        split_kv,
    )


def _launch_rank(
    query: torch.Tensor,
    global_kv: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    cp_world: int,
    cp_rank: int,
    enable_dcp: bool,
    causal_lens: torch.Tensor | None = None,
    is_var_seq: bool = False,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        cute_dsl_mla_decode,
    )

    call_args, split_kv = _prepare_rank_call(
        query,
        global_kv,
        global_lens,
        cp_world=cp_world,
        cp_rank=cp_rank,
        is_var_seq=is_var_seq,
        enable_pdl=enable_pdl,
    )
    kwargs = {}
    if enable_dcp:
        kwargs = {
            "enable_dcp": True,
            "cp_world": cp_world,
            "cp_rank": cp_rank,
            "causal_seqlens_kv_global": (
                global_lens if causal_lens is None else causal_lens
            ),
        }
    out, lse = cute_dsl_mla_decode(
        return_lse=True,
        **call_args,
        **kwargs,
    )
    return out, lse, split_kv


def _assert_close_to_reference(
    out: torch.Tensor,
    lse: torch.Tensor,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float8_e4m3fn:
        out_atol, out_rtol = 0.1, 0.1
        lse_atol, lse_rtol = 0.2, 0.1
    else:
        out_atol = out_rtol = lse_atol = lse_rtol = 1e-2
    torch.testing.assert_close(out.float(), ref_out, atol=out_atol, rtol=out_rtol)
    torch.testing.assert_close(lse.float(), ref_lse, atol=lse_atol, rtol=lse_rtol)


def _assert_dcp_rank_merge_matches_reference(
    query: torch.Tensor,
    global_kv: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    cp_world: int,
    dtype: torch.dtype,
    is_var_seq: bool = False,
) -> list[int]:
    """Check every rank-local state and their final natural-log merge."""
    rank_outputs = []
    rank_lses = []
    split_kvs = []
    for cp_rank in range(cp_world):
        out, lse, split_kv = _launch_rank(
            query,
            global_kv,
            global_lens,
            cp_world=cp_world,
            cp_rank=cp_rank,
            enable_dcp=True,
            is_var_seq=is_var_seq,
        )
        ref_rank_out, ref_rank_lse = _reference_attention(
            query,
            global_kv,
            global_lens,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )
        _assert_close_to_reference(out, lse, ref_rank_out, ref_rank_lse, dtype)
        rank_outputs.append(out)
        rank_lses.append(lse)
        split_kvs.append(split_kv)

    merged_out, merged_lse = _merge_rank_outputs_natural_log(rank_outputs, rank_lses)
    ref_out, ref_lse = _reference_attention(query, global_kv, global_lens)
    _assert_close_to_reference(merged_out, merged_lse, ref_out, ref_lse, dtype)
    return split_kvs


def test_dcp_flat_mask_matches_ceiling_divided_bound():
    """The division-free flattened predicate must match global coordinates."""
    for num_heads in (6, 12, 24, 48, 64, 96, 128):
        for q_len in (1, 2, 4, 8):
            for cp_world in (1, 2, 4):
                for cp_rank in range(cp_world):
                    for global_len in (q_len, q_len + 1, 127, 128, 129):
                        local_len = _local_length(global_len, cp_world, cp_rank)
                        for q_idx in range(q_len):
                            for local_key in range(local_len):
                                expected = (
                                    local_key * cp_world + cp_rank
                                    < global_len - (q_len - 1) + q_idx
                                )
                                for head in (0, num_heads - 1):
                                    flat_q_row = q_idx * num_heads + head
                                    assert (
                                        _flat_dcp_score_is_valid(
                                            flat_q_row,
                                            num_heads,
                                            local_key,
                                            q_len,
                                            global_len,
                                            cp_world,
                                            cp_rank,
                                        )
                                        == expected
                                    )


def test_dcp_per_query_tile_dense_boundary_is_conservative():
    """A tile marked dense must be valid for every real row and local key."""
    k_tile = 128
    for num_heads in (6, 12, 24, 48, 64, 96, 128):
        for q_len in (2, 4, 8, 32):
            num_q_tiles = _ceil_div(q_len * num_heads, 128)
            for cp_world in (1, 2, 4):
                for cp_rank in range(cp_world):
                    global_len = max(q_len, 277)
                    local_len = _local_length(global_len, cp_world, cp_rank)
                    global_positions = range(cp_rank, global_len, cp_world)
                    for q_tile_idx in range(num_q_tiles):
                        first_flat_row = q_tile_idx * 128
                        first_q = first_flat_row // num_heads
                        earliest_bound = bisect_left(
                            global_positions,
                            global_len - (q_len - 1) + first_q,
                        )
                        effective_bound = min(max(earliest_bound, 0), local_len)
                        first_mask_k_tile = effective_bound // k_tile
                        for k_tile_idx in range(first_mask_k_tile):
                            local_begin = k_tile_idx * k_tile
                            local_end = min(local_begin + k_tile, local_len)
                            for flat_q_row in range(
                                first_flat_row,
                                min(
                                    first_flat_row + 128,
                                    q_len * num_heads,
                                ),
                            ):
                                for local_key in range(local_begin, local_end):
                                    assert _flat_dcp_score_is_valid(
                                        flat_q_row,
                                        num_heads,
                                        local_key,
                                        q_len,
                                        global_len,
                                        cp_world,
                                        cp_rank,
                                    )


def test_cute_dsl_mla_dcp_rejects_incomplete_static_contract():
    """Reject DCP calls that cannot produce mergeable rank-local states."""
    _skip_if_unsupported()
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        cute_dsl_mla_decode,
    )

    torch.manual_seed(40)
    query, global_kv, global_lens = _make_inputs(
        global_length=128,
        q_len=4,
        num_heads=96,
        dtype=torch.bfloat16,
    )
    call_args, _ = _prepare_rank_call(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=0,
    )

    ragged_call_args = {**call_args, "query": query[:, 0]}
    with pytest.raises(ValueError, match="query must have shape"):
        cute_dsl_mla_decode(**ragged_call_args)
    with pytest.raises(ValueError, match="requires return_lse=True"):
        cute_dsl_mla_decode(
            **call_args,
            enable_dcp=True,
            cp_world=2,
            cp_rank=0,
            causal_seqlens_kv_global=global_lens,
        )
    with pytest.raises(ValueError, match="causal_seqlens_kv_global is required"):
        cute_dsl_mla_decode(
            **call_args,
            enable_dcp=True,
            cp_world=2,
            cp_rank=0,
            return_lse=True,
        )
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        cute_dsl_mla_decode(
            **call_args,
            enable_dcp=True,
            cp_world=2,
            cp_rank=0,
            causal_seqlens_kv_global=[128],
            return_lse=True,
        )
    with pytest.raises(ValueError, match="0 <= cp_rank < cp_world"):
        cute_dsl_mla_decode(
            **call_args,
            enable_dcp=True,
            cp_world=2,
            cp_rank=2,
            causal_seqlens_kv_global=global_lens,
            return_lse=True,
        )
    with pytest.raises(ValueError, match="require enable_dcp=True"):
        cute_dsl_mla_decode(**call_args, cp_world=2)


def test_cute_dsl_mla_dcp_dispatch_is_strictly_monolithic():
    """DCP must never silently select a backend with non-DCP mask semantics."""
    from flashinfer.cute_dsl.attention.mla_dispatch import _resolve_impl
    from flashinfer.trace.templates.attention import (
        trtllm_batch_decode_mla_trace_dispatch,
    )

    assert (
        _resolve_impl(
            requested="auto",
            kwargs={"enable_dcp": True, "cp_world": 2, "cp_rank": 0},
        )
        == "monolithic"
    )
    with pytest.raises(ValueError, match="only supported by the monolithic"):
        _resolve_impl(
            requested="modular",
            kwargs={"enable_dcp": True, "cp_world": 2, "cp_rank": 0},
        )
    with pytest.raises(ValueError, match="cannot be combined with 'sinks'"):
        _resolve_impl(
            requested="auto",
            kwargs={
                "enable_dcp": True,
                "cp_world": 2,
                "cp_rank": 0,
                "sinks": torch.empty(1),
            },
        )
    with pytest.raises(NotImplementedError, match="does not yet represent"):
        trtllm_batch_decode_mla_trace_dispatch(enable_dcp=True)


def test_cute_dsl_mla_dcp_world1_matches_disabled():
    """World-one DCP must preserve the existing monolithic MLA result."""
    _skip_if_unsupported()
    torch.manual_seed(41)
    query, global_kv, global_lens = _make_inputs(
        global_length=128,
        q_len=4,
        num_heads=96,
        dtype=torch.bfloat16,
    )
    baseline_out, baseline_lse, baseline_split = _launch_rank(
        query,
        global_kv,
        global_lens,
        cp_world=1,
        cp_rank=0,
        enable_dcp=False,
    )
    dcp_out, dcp_lse, dcp_split = _launch_rank(
        query,
        global_kv,
        global_lens,
        cp_world=1,
        cp_rank=0,
        enable_dcp=True,
    )
    assert baseline_split == dcp_split == 1
    torch.testing.assert_close(dcp_out, baseline_out, atol=0, rtol=0)
    torch.testing.assert_close(dcp_lse, baseline_lse, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8"],
)
def test_cute_dsl_mla_dcp_q1_h128_rank_merge(dtype):
    """Cover the distinct Q1/H128 schedule for both kernel families."""
    _skip_if_unsupported()
    torch.manual_seed(49)
    query, global_kv, global_lens = _make_inputs(
        global_length=129,
        q_len=1,
        num_heads=128,
        dtype=dtype,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=dtype,
    )
    assert split_kvs == [1, 1]


@pytest.mark.parametrize("num_heads", [12, 24, 48])
def test_cute_dsl_mla_dcp_packed_head_counts(num_heads):
    """Exercise DCP across the supported packed query-tile geometries."""
    _skip_if_unsupported()
    torch.manual_seed(50 + num_heads)
    query, global_kv, global_lens = _make_inputs(
        global_length=130,
        q_len=8,
        num_heads=num_heads,
        dtype=torch.bfloat16,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=torch.bfloat16,
    )
    assert split_kvs == [1, 1]


def test_cute_dsl_mla_dcp_public_api_autotune_profiles_causal_tensor():
    """The public runner must batch-sweep DCP metadata and return caller B1."""
    _skip_if_unsupported()
    from flashinfer import autotune
    from flashinfer.autotuner import AutoTuner
    from flashinfer.mla._core import (
        trtllm_batch_decode_with_kv_cache_mla,
    )

    torch.manual_seed(47)
    query, global_kv, global_lens = _make_inputs(
        # Two local pages satisfy the common public dispatcher's aligned
        # block-table contract for page_size=64.
        global_length=256,
        q_len=4,
        num_heads=96,
        dtype=torch.bfloat16,
    )
    call_args, split_kv = _prepare_rank_call(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=0,
    )
    assert split_kv == 1

    public_args = {
        "query": query,
        "kv_cache": call_args["kv_cache"],
        "workspace_buffer": call_args["workspace_buffer"],
        "qk_nope_head_dim": 128,
        "kv_lora_rank": _LATENT_DIM,
        "qk_rope_head_dim": _ROPE_DIM,
        "block_tables": call_args["block_tables"],
        "seq_lens": call_args["seq_lens"],
        "max_seq_len": call_args["max_seq_len"],
        "bmm1_scale": call_args["softmax_scale"],
        "bmm2_scale": 1.0,
        "backend": "auto",
        "cute_dsl_impl": "monolithic",
        "is_var_seq": False,
        "enable_pdl": False,
        "return_lse": True,
        "enable_dcp": True,
        "cp_world": 2,
        "cp_rank": 0,
    }
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        trtllm_batch_decode_with_kv_cache_mla(
            **public_args,
            causal_seqlens_kv_global=[256],
        )

    AutoTuner.get().clear_cache()
    try:
        with (
            mock.patch(
                "flashinfer.mla._core._compute_mla_decode_buckets",
                return_value=(2,),
            ),
            autotune(tune_mode=True),
        ):
            out, lse = trtllm_batch_decode_with_kv_cache_mla(
                **public_args,
                causal_seqlens_kv_global=global_lens,
            )
    finally:
        AutoTuner.get().clear_cache()

    ref_out, ref_lse = _reference_attention(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=0,
    )
    _assert_close_to_reference(
        out,
        lse.view_as(ref_lse),
        ref_out,
        ref_lse,
        torch.bfloat16,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8"],
)
def test_cute_dsl_mla_dcp_rank_mask_and_merge(dtype):
    """Cover the rank-sensitive G128/Q4/H96 boundary and rank merge."""
    _skip_if_unsupported()
    torch.manual_seed(42)
    query, global_kv, global_lens = _make_inputs(
        global_length=128,
        q_len=4,
        num_heads=96,
        dtype=dtype,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=dtype,
    )
    assert split_kvs == [1, 1]

    # In particular, query zero on rank one has local bound 62, not 63.
    assert _local_causal_bound(128, 4, 0, cp_world=2, cp_rank=1) == 62


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float8_e4m3fn],
    ids=["fp16", "fp8"],
)
def test_cute_dsl_mla_dcp_world4_variable_batch_h6(dtype):
    """Cover W4, heterogeneous local tails, FP16, and packed H6 query rows."""
    _skip_if_unsupported()
    torch.manual_seed(46)
    query, global_kv, global_lens = _make_batched_inputs(
        global_lengths=(65, 130, 259),
        q_len=8,
        num_heads=6,
        dtype=dtype,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=4,
        dtype=dtype,
        is_var_seq=True,
    )
    assert split_kvs == [1] * 4


@pytest.mark.parametrize(
    "cp_world,global_length,dtype",
    [
        pytest.param(4, 4 * 1024 + 3, torch.bfloat16, id="w4-k4k-bf16"),
        pytest.param(
            4,
            4 * 1024 + 3,
            torch.float8_e4m3fn,
            id="w4-k4k-fp8",
        ),
        pytest.param(8, 8 * 1024 + 5, torch.bfloat16, id="w8-k8k-bf16"),
        pytest.param(
            8,
            8 * 1024 + 5,
            torch.float8_e4m3fn,
            id="w8-k8k-fp8",
        ),
        pytest.param(16, 16 * 1024 + 9, torch.bfloat16, id="w16-k16k-bf16"),
        pytest.param(
            16,
            16 * 1024 + 9,
            torch.float8_e4m3fn,
            id="w16-k16k-fp8",
        ),
    ],
)
def test_cute_dsl_mla_dcp_rank_scale_merge(cp_world, global_length, dtype):
    """Cover W4/W8/W16 four-query speculative decode and uneven split-KV."""
    _skip_if_unsupported()
    torch.manual_seed(51 + cp_world)
    query, global_kv, global_lens = _make_inputs(
        global_length=global_length,
        q_len=4,
        num_heads=96,
        dtype=dtype,
    )

    base_local_length, tail_ranks = divmod(global_length, cp_world)
    for cp_rank in range(cp_world):
        expected_local_length = base_local_length + (1 if cp_rank < tail_ranks else 0)
        assert _local_length(global_length, cp_world, cp_rank) == expected_local_length

    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=cp_world,
        dtype=dtype,
    )
    assert all(split_kv > 1 for split_kv in split_kvs)


def test_cute_dsl_mla_dcp_cuda_graph_reads_updated_causal_bound():
    """A captured launch must read causal bounds mutated in place on replay."""
    _skip_if_unsupported()
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        cute_dsl_mla_decode,
    )

    torch.manual_seed(45)
    query, global_kv, global_lens = _make_inputs(
        global_length=128,
        q_len=4,
        num_heads=96,
        dtype=torch.bfloat16,
    )
    call_args, split_kv = _prepare_rank_call(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
    )
    assert split_kv == 1
    out = torch.empty(
        1,
        4,
        96,
        _LATENT_DIM,
        dtype=torch.bfloat16,
        device=query.device,
    )
    lse = torch.empty(1, 4, 96, dtype=torch.float32, device=query.device)
    dcp_args = {
        "enable_dcp": True,
        "cp_world": 2,
        "cp_rank": 1,
        "causal_seqlens_kv_global": global_lens,
        "return_lse": True,
        "out": out,
        "lse": lse,
    }

    # Compile and initialize all persistent buffers before capture.
    returned_out, returned_lse = cute_dsl_mla_decode(**call_args, **dcp_args)
    assert returned_out is out
    assert returned_lse is lse
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_dsl_mla_decode(**call_args, **dcp_args)
    lse_at_128 = lse.clone()

    # Keep the physical local cache and seq_lens fixed at the G=128 shard.
    # The G=126 replay must mask its now-extra global keys 126 and 127.
    global_lens.fill_(126)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(lse, lse_at_128)
    ref_out, ref_lse = _reference_attention(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
    )
    _assert_close_to_reference(out, lse, ref_out, ref_lse, torch.bfloat16)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8"],
)
def test_cute_dsl_mla_dcp_empty_rank_row(dtype):
    """A rank with no visible key must contribute O=0 and LSE=-inf."""
    _skip_if_unsupported()
    torch.manual_seed(43)
    query, global_kv, global_lens = _make_inputs(
        global_length=4,
        q_len=4,
        num_heads=96,
        dtype=dtype,
    )
    out, lse, split_kv = _launch_rank(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
        enable_dcp=True,
    )
    assert split_kv == 1
    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0]))
    assert torch.isneginf(lse[:, 0]).all()
    ref_out, ref_lse = _reference_attention(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
    )
    _assert_close_to_reference(out, lse, ref_out, ref_lse, dtype)

    # Also cover a physically empty cyclic shard and the cross-rank merge for
    # early rows that have no visible key on any rank. The empty shard still
    # owns one padding page, but seq_lens=0, so it may not load or reduce K.
    query, global_kv, global_lens = _make_inputs(
        global_length=1,
        q_len=4,
        num_heads=96,
        dtype=dtype,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=dtype,
    )
    assert split_kvs == [1, 1]


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8"],
)
def test_cute_dsl_mla_dcp_split_kv_rank_merge(dtype):
    """DCP must preserve natural-log LSE through standalone split reduction."""
    _skip_if_unsupported()
    torch.manual_seed(44)
    query, global_kv, global_lens = _make_inputs(
        global_length=4096,
        q_len=4,
        num_heads=96,
        dtype=dtype,
    )
    split_kvs = _assert_dcp_rank_merge_matches_reference(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=dtype,
    )
    assert all(split_kv > 1 for split_kv in split_kvs)

    # Preserve the 2048-token physical shard and its split geometry, but make
    # every rank-1 key causally invisible. This exercises the all-empty split
    # reducer rather than the direct zero-tile output path above.
    empty_causal_lens = torch.tensor([1], dtype=torch.int32, device=query.device)
    empty_out, empty_lse, empty_split_kv = _launch_rank(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
        enable_dcp=True,
        causal_lens=empty_causal_lens,
    )
    assert empty_split_kv == split_kvs[1] > 1
    assert torch.equal(empty_out, torch.zeros_like(empty_out))
    assert torch.isneginf(empty_lse).all()
    ref_empty_out, ref_empty_lse = _reference_attention(
        query,
        global_kv,
        empty_causal_lens,
        cp_world=2,
        cp_rank=1,
    )
    _assert_close_to_reference(
        empty_out,
        empty_lse,
        ref_empty_out,
        ref_empty_lse,
        dtype,
    )


def test_cute_dsl_mla_dcp_pdl_all_empty_split_reduction():
    """PDL must order the all-empty producer path before split reduction."""
    _skip_if_unsupported()
    from flashinfer.utils import device_support_pdl

    if not device_support_pdl(torch.device("cuda")):
        pytest.skip("Programmatic dependent launch is not supported")

    torch.manual_seed(48)
    query, global_kv, global_lens = _make_inputs(
        global_length=4096,
        q_len=4,
        num_heads=96,
        dtype=torch.bfloat16,
    )
    empty_causal_lens = torch.tensor([1], dtype=torch.int32, device=query.device)
    out, lse, split_kv = _launch_rank(
        query,
        global_kv,
        global_lens,
        cp_world=2,
        cp_rank=1,
        enable_dcp=True,
        causal_lens=empty_causal_lens,
        enable_pdl=True,
    )
    assert split_kv > 1
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()
