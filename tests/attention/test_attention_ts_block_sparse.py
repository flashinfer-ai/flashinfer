# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-contract acceptance coverage for PrimTS block-sparse attention."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
import importlib
import inspect
import math
from types import SimpleNamespace
from typing import get_args
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

import flashinfer.attention.prims_ts as prims_ts
from flashinfer.attention.prims_ts import block_sparse as block_sparse_module
from flashinfer.attention.prims_ts._block_sparse import config as block_sparse_config
from flashinfer.attention.prims_ts._block_sparse import plan as block_sparse_plan
from flashinfer.attention.prims_ts._block_sparse.prepared import (
    _BlockSparseRouteLayout,
)


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS block-sparse attention requires SM100 or SM103",
)

_HEAD_DIM = 128
_Patterns = tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]


@dataclass(frozen=True)
class _Case:
    """One bounded public plan/run problem with an independent oracle."""

    name: str
    batch_size: int
    num_heads: int
    seq_len_q: int
    seq_len_kv: int
    q_block_size: int
    kv_block_size: int
    dtype: torch.dtype
    mask_type: str
    token_mask: str
    scheduler: str
    pattern: str = "sparse"
    include_empty_row: bool = False
    expected_parallel_loads: bool | None = None
    num_kv_heads: int | None = None
    expected_q_tile: int | None = None
    expected_kv_tile: int | None = None

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def heads_q_per_kv(self) -> int:
        return self.num_heads // self.effective_num_kv_heads


_CASES = (
    _Case(
        "q8_kv8_fp16_no_token_full_and_partial_clc",
        1,
        1,
        16,
        145,
        8,
        8,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="full_and_partial",
    ),
    _Case(
        "q8_kv8_fp16_noncausal_holey_static",
        1,
        1,
        17,
        151,
        8,
        8,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="mixed",
    ),
    _Case(
        "q8_kv8_bf16_noncausal_holey_clc",
        1,
        1,
        17,
        151,
        8,
        8,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q8_kv8_bf16_noncausal_holey_parallel_clc",
        1,
        1,
        17,
        2313,
        8,
        8,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
        pattern="wide_mixed",
        expected_parallel_loads=True,
    ),
    _Case(
        "q8_kv16_fp16_holey_empty_clc",
        1,
        1,
        23,
        207,
        8,
        16,
        torch.float16,
        "dense",
        "holey",
        "persistent",
        pattern="mixed",
        include_empty_row=True,
    ),
    _Case(
        "q8_kv16_fp16_no_token_short_clc",
        1,
        1,
        23,
        207,
        8,
        16,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q16_kv32_bf16_causal_holey_clc",
        1,
        1,
        47,
        289,
        16,
        32,
        torch.bfloat16,
        "causal",
        "holey",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q16_kv16_bf16_noncausal_holey_clc",
        1,
        1,
        47,
        289,
        16,
        16,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q16_kv16_fp16_noncausal_holey_static",
        1,
        1,
        49,
        287,
        16,
        16,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="mixed",
    ),
    _Case(
        "q16_kv16_fp16_no_token_tail_clc",
        1,
        1,
        47,
        289,
        16,
        16,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q32_kv32_fp16_no_token_tail_clc",
        1,
        1,
        65,
        233,
        32,
        32,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q32_kv16_bf16_noncausal_holey_parallel_clc",
        1,
        1,
        65,
        2304,
        32,
        16,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
        pattern="wide_mixed",
        expected_parallel_loads=True,
    ),
    _Case(
        "q32_kv64_fp16_noncausal_holey_tail_clc",
        1,
        1,
        65,
        321,
        32,
        64,
        torch.float16,
        "dense",
        "holey",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q32_kv8_fp16_unaligned_tail_static",
        1,
        1,
        65,
        233,
        32,
        8,
        torch.float16,
        "causal",
        "holey",
        "static",
        pattern="unaligned_tail",
        expected_parallel_loads=True,
    ),
    _Case(
        "q32_kv128_bf16_causal_tail_static",
        1,
        1,
        65,
        385,
        32,
        128,
        torch.bfloat16,
        "causal",
        "holey",
        "static",
    ),
    _Case(
        "q64_kv64_fp16_nonadjacent_tail_static",
        1,
        1,
        63,
        193,
        64,
        64,
        torch.float16,
        "dense",
        "full",
        "static",
    ),
    _Case(
        "q64_kv64_fp16_nonadjacent_tail_holey_static",
        1,
        1,
        63,
        193,
        64,
        64,
        torch.float16,
        "dense",
        "holey",
        "static",
    ),
    _Case(
        "q64_kv128_bf16_kv256_nomask_static",
        1,
        1,
        64,
        513,
        64,
        128,
        torch.bfloat16,
        "dense",
        "none",
        "static",
        pattern="mixed",
    ),
    _Case(
        "q64_kv256_fp16_kv256_nomask_clc",
        1,
        1,
        64,
        513,
        64,
        256,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q64_kv64_bf16_kv256_causal_holey_tail_clc",
        1,
        1,
        321,
        321,
        64,
        64,
        torch.bfloat16,
        "causal",
        "holey",
        "persistent",
        pattern="mixed",
    ),
    _Case(
        "q128_kv256_bf16_batched_heads_clc",
        2,
        2,
        129,
        641,
        128,
        256,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
    ),
    _Case(
        "q128_kv128_fp16_full_route_static",
        1,
        1,
        127,
        385,
        128,
        128,
        torch.float16,
        "dense",
        "full",
        "static",
    ),
    _Case(
        "q192_kv64_fp16_empty_row_static",
        1,
        1,
        385,
        257,
        192,
        64,
        torch.float16,
        "dense",
        "none",
        "static",
        include_empty_row=True,
    ),
    _Case(
        "q256_kv128_bf16_causal_holey_clc",
        1,
        1,
        257,
        513,
        256,
        128,
        torch.bfloat16,
        "causal",
        "holey",
        "persistent",
    ),
)


_GQA_CASES = (
    _Case(
        "gqa16_bf16_causal_token_routes_persistent_q16_kv128",
        1,
        16,
        8,
        1024,
        1,
        128,
        torch.bfloat16,
        "causal",
        "none",
        "persistent",
        num_kv_heads=1,
        expected_q_tile=16,
        expected_kv_tile=128,
    ),
    _Case(
        "gqa4_fp16_dense_holey_static_q32_kv128",
        1,
        8,
        17,
        151,
        8,
        8,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="mixed",
        expected_parallel_loads=True,
        num_kv_heads=2,
        expected_q_tile=32,
        expected_kv_tile=128,
    ),
    _Case(
        "gqa8_fp16_dense_static_q64_kv256",
        1,
        16,
        17,
        577,
        8,
        64,
        torch.float16,
        "dense",
        "none",
        "static",
        num_kv_heads=2,
        expected_q_tile=64,
        expected_kv_tile=256,
    ),
    _Case(
        "gqa8_bf16_causal_holey_empty_persistent_q64_kv256",
        1,
        16,
        17,
        577,
        8,
        64,
        torch.bfloat16,
        "causal",
        "holey",
        "persistent",
        pattern="mixed",
        include_empty_row=True,
        num_kv_heads=2,
        expected_q_tile=64,
        expected_kv_tile=256,
    ),
    _Case(
        "gqa32_bf16_dense_static_q128_kv128",
        1,
        64,
        9,
        257,
        8,
        64,
        torch.bfloat16,
        "dense",
        "none",
        "static",
        num_kv_heads=2,
        expected_q_tile=128,
        expected_kv_tile=128,
    ),
)


def _make_patterns(case: _Case) -> _Patterns:
    num_q_rows = math.ceil(case.seq_len_q / case.q_block_size)
    num_kv_blocks = math.ceil(case.seq_len_kv / case.kv_block_size)
    batches: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    for batch_idx in range(case.batch_size):
        heads: list[tuple[tuple[int, ...], ...]] = []
        for head_idx in range(case.effective_num_kv_heads):
            rows: list[tuple[int, ...]] = []
            for row_idx in range(num_q_rows):
                if (
                    case.include_empty_row
                    and batch_idx == case.batch_size - 1
                    and head_idx == case.effective_num_kv_heads - 1
                    and row_idx == num_q_rows - 1
                ):
                    rows.append(())
                    continue
                if case.pattern == "full_and_partial":
                    rows.append(
                        tuple(range(16)) if row_idx == 0 else (0, num_kv_blocks - 1)
                    )
                    continue
                if case.pattern == "mixed":
                    candidates = (0, 1, 3, 5, 6, 7, num_kv_blocks - 1)
                    rows.append(
                        tuple(
                            sorted(
                                {
                                    block_idx
                                    for block_idx in candidates
                                    if 0 <= block_idx < num_kv_blocks
                                }
                            )
                        )
                    )
                    continue
                if case.pattern == "wide_mixed":
                    # 144 B8 blocks form nine KV128 routes, exercising the
                    # two parallel load streams with an odd final route.
                    candidates = list(range(0, num_kv_blocks, 2))[:143]
                    candidates.append(num_kv_blocks - 1)
                    rows.append(tuple(sorted(candidates)))
                    continue
                if case.pattern == "unaligned_tail":
                    rows.append((*range(1, 9), *range(22, 30)))
                    continue
                selected = {0, num_kv_blocks - 1}
                if num_kv_blocks > 2:
                    selected.add(
                        (batch_idx + 2 * head_idx + row_idx + 1) % num_kv_blocks
                    )
                rows.append(tuple(sorted(selected)))
            heads.append(tuple(rows))
        batches.append(tuple(heads))
    return tuple(batches)


def _make_bsr(patterns: _Patterns) -> tuple[torch.Tensor, torch.Tensor]:
    flat_indices: list[int] = []
    pointer_batches: list[list[list[int]]] = []
    for batch in patterns:
        pointer_heads: list[list[int]] = []
        for head in batch:
            pointers = [len(flat_indices)]
            for row in head:
                flat_indices.extend(row)
                pointers.append(len(flat_indices))
            pointer_heads.append(pointers)
        pointer_batches.append(pointer_heads)
    return (
        torch.tensor(pointer_batches, device="cuda", dtype=torch.int32),
        torch.tensor(flat_indices, device="cuda", dtype=torch.int32),
    )


def _pack_token_mask(
    seq_len_kv: int,
    valid_by_batch: tuple[frozenset[int], ...],
) -> torch.Tensor:
    packed_by_batch: list[list[int]] = []
    for valid_tokens in valid_by_batch:
        words = [0] * math.ceil(seq_len_kv / 32)
        for token_idx in valid_tokens:
            words[token_idx // 32] |= 1 << (token_idx % 32)
        packed_by_batch.append(words)
    return torch.tensor(packed_by_batch, device="cuda", dtype=torch.uint32)


def _make_token_mask(
    case: _Case,
) -> tuple[torch.Tensor | None, tuple[frozenset[int], ...]]:
    all_tokens = frozenset(range(case.seq_len_kv))
    if case.token_mask == "none":
        return None, tuple(all_tokens for _ in range(case.batch_size))
    if case.token_mask == "full":
        valid_by_batch = tuple(all_tokens for _ in range(case.batch_size))
        return _pack_token_mask(case.seq_len_kv, valid_by_batch), valid_by_batch

    valid_by_batch = tuple(
        frozenset(
            token_idx
            for token_idx in range(case.seq_len_kv)
            if (token_idx + batch_idx) % 7 not in (0, 3)
        )
        for batch_idx in range(case.batch_size)
    )
    return _pack_token_mask(case.seq_len_kv, valid_by_batch), valid_by_batch


@torch.no_grad()
def _reference(
    case: _Case,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    patterns: _Patterns,
    valid_by_batch: tuple[frozenset[int], ...],
    sm_scale: float,
) -> torch.Tensor:
    """Evaluate semantic BSR visibility with FP32 PyTorch operations."""

    key_positions = torch.arange(case.seq_len_kv, device=q.device)
    key_blocks = torch.div(
        key_positions,
        case.kv_block_size,
        rounding_mode="floor",
    )
    query_positions = torch.arange(case.seq_len_q, device=q.device)
    causal_end = case.seq_len_kv - case.seq_len_q + query_positions + 1
    batch_outputs: list[torch.Tensor] = []
    for batch_idx in range(case.batch_size):
        valid_tokens = torch.tensor(
            [
                token_idx in valid_by_batch[batch_idx]
                for token_idx in range(case.seq_len_kv)
            ],
            device=q.device,
            dtype=torch.bool,
        )
        head_outputs: list[torch.Tensor] = []
        for q_head_idx in range(case.num_heads):
            kv_head_idx = q_head_idx // case.heads_q_per_kv
            structural_rows: list[torch.Tensor] = []
            for row_idx, selected_blocks in enumerate(patterns[batch_idx][kv_head_idx]):
                row_tokens = min(
                    case.q_block_size,
                    case.seq_len_q - row_idx * case.q_block_size,
                )
                selected = torch.tensor(
                    selected_blocks,
                    device=q.device,
                    dtype=key_blocks.dtype,
                )
                row_mask = torch.isin(key_blocks, selected)
                structural_rows.append(row_mask.expand(row_tokens, -1))
            allowed = torch.cat(structural_rows, dim=0) & valid_tokens[None, :]
            if case.mask_type == "causal":
                allowed = allowed & (key_positions[None, :] < causal_end[:, None])

            scores = (
                q[batch_idx, :, q_head_idx].float()
                @ k[batch_idx, :, kv_head_idx].float().transpose(0, 1)
            ) * sm_scale
            masked_scores = scores.masked_fill(~allowed, float("-inf"))
            active_rows = allowed.any(dim=1, keepdim=True)
            safe_scores = torch.where(
                active_rows,
                masked_scores,
                torch.zeros_like(masked_scores),
            )
            probabilities = torch.softmax(safe_scores, dim=-1)
            probabilities = torch.where(
                allowed,
                probabilities,
                torch.zeros_like(probabilities),
            )
            head_outputs.append(probabilities @ v[batch_idx, :, kv_head_idx].float())
        batch_outputs.append(torch.stack(head_outputs, dim=1))
    return torch.stack(batch_outputs).to(case.dtype)


@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "expected_by_q_head"),
    (
        pytest.param(2, 2, (2.0, 12.0), id="mha"),
        pytest.param(4, 2, (2.0, 2.0, 12.0, 12.0), id="gqa2"),
    ),
)
def test_block_sparse_reference_maps_q_heads_to_kv_heads(
    num_qo_heads: int,
    num_kv_heads: int,
    expected_by_q_head: tuple[float, ...],
) -> None:
    case = _Case(
        "reference_head_mapping",
        1,
        num_qo_heads,
        1,
        2,
        8,
        64,
        torch.float32,
        "dense",
        "none",
        "static",
        num_kv_heads=num_kv_heads,
    )
    patterns: _Patterns = ((tuple(((0,),) for _ in range(num_kv_heads))),)
    q = torch.zeros((1, 1, num_qo_heads, _HEAD_DIM))
    k = torch.zeros((1, 2, num_kv_heads, _HEAD_DIM))
    v = torch.empty_like(k)
    v[0, :, 0] = torch.tensor((1.0, 3.0))[:, None]
    v[0, :, 1] = torch.tensor((10.0, 14.0))[:, None]

    actual = _reference(
        case,
        q,
        k,
        v,
        patterns,
        (frozenset((0, 1)),),
        sm_scale=1.0,
    )

    expected = torch.tensor(expected_by_q_head)[:, None].expand(-1, _HEAD_DIM)
    torch.testing.assert_close(actual[0, 0], expected)


def _plan(
    wrapper: block_sparse_module.BlockSparseTSWrapper,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    batch_size: int = 1,
    seq_len_q: int = 64,
    seq_len_kv: int = 128,
    num_heads: int = 1,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    kv_valid_bits: torch.Tensor | None = None,
    max_blocks_per_row: int | None = None,
) -> None:
    if max_blocks_per_row is None:
        row_counts = block_indptr[..., 1:] - block_indptr[..., :-1]
        max_blocks_per_row = int(row_counts.max().item())
    wrapper.plan(
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_heads,
        num_heads,
        _HEAD_DIM,
        q_block_size,
        kv_block_size,
        device=block_indptr.device,
        max_blocks_per_row=max_blocks_per_row,
        use_kv_valid_bits=kv_valid_bits is not None,
    )


def test_public_exports() -> None:
    assert prims_ts.BlockSparseTSWrapper is block_sparse_module.BlockSparseTSWrapper
    assert (
        prims_ts.BlockSparsePagedTSWrapper
        is block_sparse_module.BlockSparsePagedTSWrapper
    )
    assert prims_ts.block_sparse_attention is block_sparse_module.block_sparse_attention
    assert (
        prims_ts.block_sparse_attention_with_paged_kv_cache
        is block_sparse_module.block_sparse_attention_with_paged_kv_cache
    )


def test_block_sparse_wrapper_routes_are_run_inputs() -> None:
    """A reusable plan owns capacity, while each run owns its sparse routes."""

    from dataclasses import fields

    from flashinfer.attention.prims_ts._block_sparse.plan import (
        _BlockSparsePlanState,
    )

    plan_parameters = inspect.signature(
        block_sparse_module.BlockSparseTSWrapper.plan
    ).parameters
    plan_routing_parameters = {
        "block_indptr",
        "block_indices",
        "kv_valid_bits",
        "dynamic_metadata",
    }
    assert plan_routing_parameters.isdisjoint(plan_parameters)
    for name in ("device", "max_blocks_per_row", "use_kv_valid_bits"):
        assert plan_parameters[name].default is inspect.Parameter.empty

    run_parameters = inspect.signature(
        block_sparse_module.BlockSparseTSWrapper.run
    ).parameters
    for name in ("block_indptr", "block_indices"):
        assert run_parameters[name].default is inspect.Parameter.empty
    assert "kv_valid_bits" in run_parameters

    state_fields = {field.name for field in fields(_BlockSparsePlanState)}
    assert {"block_indptr", "block_indices", "kv_valid_bits"}.isdisjoint(state_fields)
    assert {"num_qo_heads", "num_kv_heads"} <= state_fields
    assert "num_heads" not in state_fields


def test_block_sparse_contiguous_wrapper_trace_uses_bound_plan_state() -> None:
    """Trace a reusable contiguous run while keeping plan geometry optional."""

    from flashinfer.fi_trace import fi_trace

    wrapper = block_sparse_module.BlockSparseTSWrapper()
    q = torch.empty((2, 64, 8, 128), dtype=torch.float16)
    k = torch.empty((2, 256, 4, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "block_indptr": torch.empty((2, 4, 2), dtype=torch.int32),
        "block_indices": torch.empty((16,), dtype=torch.int32),
    }

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(**kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(wrapper.run, **kwargs)

    # A successful plan atomically publishes this state. Its contents are not
    # needed to select the single contiguous schema, so avoid a CUDA plan here.
    wrapper._plan_state = SimpleNamespace()
    defn = fi_trace(wrapper.run, **kwargs)
    assert defn["name"].startswith("prims_ts_block_sparse_wrapper")
    assert defn["inputs"]["q"]["shape"] == [
        "batch_size",
        "seq_len_q",
        "num_qo_heads",
        "head_dim",
    ]
    assert defn["inputs"]["k"]["shape"] == [
        "batch_size",
        "seq_len_kv",
        "num_kv_heads",
        "head_dim",
    ]
    for name in ("q_block_size", "kv_block_size", "mask_type"):
        assert defn["inputs"][name]["optional"] is True
    assert defn["inputs"]["block_indptr"].get("optional") is not True


def test_block_sparse_paged_wrapper_trace_uses_bound_plan_state() -> None:
    """Trace both public paged-cache forms with live run metadata."""

    from flashinfer.fi_trace import fi_trace

    wrapper = block_sparse_module.BlockSparsePagedTSWrapper()
    q = torch.empty((2, 4, 8, 128), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 4, 64, 128), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    common_kwargs = {
        "q": q,
        "paged_kv_indptr": torch.empty((3,), dtype=torch.int32),
        "paged_kv_indices": torch.empty((8,), dtype=torch.int32),
        "seq_lens_kv": torch.empty((2,), dtype=torch.int32),
        "block_indptr": torch.empty((2, 4, 2), dtype=torch.int32),
        "block_indices": torch.empty((16,), dtype=torch.int32),
    }

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(paged_kv_cache=(k_cache, v_cache), **common_kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(
            wrapper.run,
            paged_kv_cache=(k_cache, v_cache),
            **common_kwargs,
        )

    wrapper._plan_state = SimpleNamespace()
    cache_forms = (
        ((k_cache, v_cache), "tuple"),
        (torch.stack((k_cache, v_cache), dim=1), "combined"),
    )
    for paged_kv_cache, cache_form in cache_forms:
        defn = fi_trace(
            wrapper.run,
            paged_kv_cache=paged_kv_cache,
            **common_kwargs,
        )
        assert defn["name"].startswith(
            f"prims_ts_paged_block_sparse_wrapper_{cache_form}"
        )
        for name in (
            "q_block_size",
            "kv_block_size",
            "max_seq_len_kv",
            "mask_type",
        ):
            assert defn["inputs"][name]["optional"] is True
        for name in (
            "paged_kv_indptr",
            "paged_kv_indices",
            "seq_lens_kv",
            "block_indptr",
            "block_indices",
        ):
            assert defn["inputs"][name].get("optional") is not True


def test_public_paged_wrapper_uses_only_live_run_metadata() -> None:
    """Paged plans own capacity, while attention consumes caller live lengths."""

    from flashinfer.attention.prims_ts._block_sparse import (
        compiler as block_sparse_compiler,
    )
    from flashinfer.attention.prims_ts._block_sparse import (
        runtime as block_sparse_runtime,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode import (
        block_sparse_prepare,
    )

    plan_parameters = inspect.signature(
        block_sparse_module.BlockSparsePagedTSWrapper.plan
    ).parameters
    assert tuple(plan_parameters)[:10] == (
        "self",
        "batch_size",
        "seq_len_q",
        "max_seq_len_kv",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_block_size",
        "kv_block_size",
        "page_size",
    )
    assert {"paged_kv_indptr", "paged_kv_indices", "seq_lens_kv"}.isdisjoint(
        plan_parameters
    )
    for name in (
        "device",
        "max_blocks_per_row",
        "use_kv_valid_bits",
    ):
        assert plan_parameters[name].default is inspect.Parameter.empty

    run_parameters = inspect.signature(
        block_sparse_module.BlockSparsePagedTSWrapper.run
    ).parameters
    for name in (
        "paged_kv_indptr",
        "paged_kv_indices",
        "seq_lens_kv",
        "block_indptr",
        "block_indices",
    ):
        assert run_parameters[name].default is inspect.Parameter.empty

    one_shot_parameters = inspect.signature(
        block_sparse_module.block_sparse_attention_with_paged_kv_cache
    ).parameters
    assert "max_seq_len_kv" in one_shot_parameters
    assert "seq_len_kv" not in one_shot_parameters
    assert one_shot_parameters["max_seq_len_kv"].default is inspect.Parameter.empty
    assert one_shot_parameters["seq_lens_kv"].default is inspect.Parameter.empty

    state_fields = {
        field.name for field in fields(block_sparse_plan._BlockSparsePlanState)
    }
    assert {
        "paged_kv_indptr",
        "paged_kv_indices",
        "seq_lens_kv",
        "paged_kv",
        "validated_seq_lens_kv",
    }.isdisjoint(state_fields)
    assert "page_size" in state_fields
    assert "validated_seq_lens_kv" not in {
        field.name for field in fields(block_sparse_runtime._PagedKVLaunchPayload)
    }
    assert (
        "validated_seq_lens_kv"
        not in inspect.signature(
            block_sparse_prepare._PrepareBlockSparseRoutes.__call__
        ).parameters
    )
    assert "validated_seq_lens_kv" not in inspect.getsource(
        block_sparse_compiler._compile_block_sparse
    )


def test_block_sparse_plan_rejects_cuda_graph_capture_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    plan_stream = object()
    monkeypatch.setattr(
        block_sparse_module,
        "_resolve_cuda_device",
        lambda _device: (device, 0),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: plan_stream)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def unexpected_builder(*_args: object, **_kwargs: object) -> None:
        pytest.fail("capture must be rejected before plan-state allocation")

    monkeypatch.setattr(
        block_sparse_module,
        "_build_block_sparse_plan_state",
        unexpected_builder,
    )

    with pytest.raises(RuntimeError, match="planning.*CUDA Graph capture"):
        block_sparse_module.BlockSparseTSWrapper().plan(
            1,
            64,
            128,
            1,
            1,
            _HEAD_DIM,
            64,
            64,
            device="cuda:0",
            max_blocks_per_row=1,
            use_kv_valid_bits=False,
        )


@pytest.mark.parametrize("dimension", ("seq_len_q", "seq_len_kv"))
def test_block_sparse_plan_rejects_int32_sequence_overflow_before_device_work(
    monkeypatch: pytest.MonkeyPatch,
    dimension: str,
) -> None:
    def unexpected_device_work(_device: object) -> None:
        pytest.fail("Int32 overflow must fail before resolving the CUDA device")

    monkeypatch.setattr(
        block_sparse_module,
        "_resolve_cuda_device",
        unexpected_device_work,
    )
    arguments = {
        "batch_size": 1,
        "seq_len_q": 64,
        "seq_len_kv": 128,
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim": _HEAD_DIM,
        "q_block_size": 64,
        "kv_block_size": 64,
        "device": "cuda:0",
        "max_blocks_per_row": 0,
        "use_kv_valid_bits": False,
    }
    arguments[dimension] = 1 << 31

    with pytest.raises(OverflowError, match=rf"{dimension}.*signed int32"):
        block_sparse_module.BlockSparseTSWrapper().plan(**arguments)


def test_one_shot_rejects_invalid_static_profile_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_sparse_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())

    def unexpected_inspection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid static geometry must fail before BSR inspection")

    monkeypatch.setattr(
        block_sparse_module,
        "_inspect_block_sparse_bsr",
        unexpected_inspection,
    )
    q = torch.empty((1, 0, 1, _HEAD_DIM), dtype=torch.float16)
    k = torch.empty((1, 128, 1, _HEAD_DIM), dtype=torch.float16)
    block_indptr = torch.empty((1, 1, 1), dtype=torch.int32)
    block_indices = torch.empty(0, dtype=torch.int32)

    with pytest.raises(ValueError, match="seq_len_q must be positive"):
        block_sparse_module.block_sparse_attention(
            q,
            k,
            k,
            block_indptr,
            block_indices,
            64,
            64,
        )


def test_one_shot_inspection_publishes_only_the_semantic_row_bound() -> None:
    from flashinfer.attention.prims_ts._block_sparse.inspection import (
        _BlockSparseInspection,
        _inspect_block_sparse_bsr,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.block_sparse_inspect import (
        compile_block_sparse_inspection,
    )

    assert [field.name for field in fields(_BlockSparseInspection)] == [
        "max_row_block_count"
    ]
    assert (
        "kv_route_size" not in inspect.signature(_inspect_block_sparse_bsr).parameters
    )
    assert (
        "kv_route_size"
        not in inspect.signature(compile_block_sparse_inspection).parameters
    )


def test_block_sparse_bshd_tma_strides_use_int64_for_large_batches() -> None:
    """Descriptor strides must reach TMA's 16-byte ABI without Int32 overflow."""

    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _block_sparse_bshd_tma_strides,
    )

    q_strides, kv_strides = _block_sparse_bshd_tma_strides(
        q_seq=cutlass.Int32(8192),
        h_q=cutlass.Int32(16384),
        h_k=cutlass.Int32(16384),
        s_k=cutlass.Int32(8192),
        d=cutlass.Int32(128),
    )

    assert all(type(stride) is cutlass.Int64 for stride in (*q_strides, *kv_strides))
    assert tuple(map(int, q_strides)) == (16, 16, 262_144, 2_147_483_648)
    assert tuple(map(int, kv_strides)) == (262_144, 16, 2_147_483_648)


def test_block_sparse_selects_native_kv256_only_for_qualified_geometry() -> None:
    """Keep the native route selection narrow and independent of live BSR data."""

    select = block_sparse_config._select_block_sparse_kv_route_size
    assert (
        select(
            q_tile_size=64,
            kv_block_size=64,
        )
        == 256
    )
    assert (
        select(
            q_tile_size=64,
            kv_block_size=256,
        )
        == 256
    )
    assert (
        select(
            q_tile_size=128,
            kv_block_size=64,
        )
        == 128
    )


@pytest.mark.parametrize(
    ("q_block_size", "heads_q_per_kv", "kv_block_size", "expected_q_tile"),
    (
        pytest.param(1, 8, 128, 8, id="q1-gqa8-coarse-kv"),
        pytest.param(1, 16, 128, 16, id="q1-gqa16-coarse-kv"),
        pytest.param(1, 32, 128, 32, id="q1-gqa32-coarse-kv"),
        pytest.param(2, 4, 128, 8, id="q2-gqa4-coarse-kv"),
        pytest.param(4, 2, 128, 8, id="q4-gqa2-coarse-kv"),
        pytest.param(8, 4, 64, 32, id="q8-gqa4-coarse-kv"),
        pytest.param(8, 8, 64, 64, id="q8-gqa8-coarse-kv"),
        pytest.param(8, 8, 8, 32, id="q8-gqa8-fine-kv"),
        pytest.param(64, 1, 8, 32, id="q64-mha-fine-kv"),
        pytest.param(8, 32, 64, 128, id="q8-mqa32-coarse-kv"),
        pytest.param(8, 1, 64, 8, id="mha-q8"),
        pytest.param(16, 1, 64, 16, id="mha-q16"),
        pytest.param(32, 1, 64, 32, id="mha-q32"),
        pytest.param(64, 1, 64, 64, id="mha-q64"),
        pytest.param(96, 1, 64, 32, id="mha-q96"),
        pytest.param(128, 1, 64, 128, id="mha-q128"),
        pytest.param(192, 1, 64, 64, id="mha-q192"),
        pytest.param(256, 1, 64, 128, id="mha-q256"),
    ),
)
def test_block_sparse_q_tile_groups_effective_q_rows(
    q_block_size: int,
    heads_q_per_kv: int,
    kv_block_size: int,
    expected_q_tile: int,
) -> None:
    q_tile_size = block_sparse_config._select_block_sparse_q_tile_size(
        q_block_size=q_block_size,
        heads_q_per_kv=heads_q_per_kv,
        kv_block_size=kv_block_size,
    )

    assert q_tile_size == expected_q_tile
    assert q_tile_size % heads_q_per_kv == 0
    assert q_block_size % (q_tile_size // heads_q_per_kv) == 0


def _validate_static_block_sparse_heads(
    num_qo_heads: int,
    num_kv_heads: int,
) -> block_sparse_module._BlockSparseStaticProfile:
    return block_sparse_module._validate_block_sparse_static_profile(
        batch_size=1,
        seq_len_q=8,
        seq_len_kv=64,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_HEAD_DIM,
        q_block_size=8,
        kv_block_size=64,
        use_kv_valid_bits=False,
        mask_type="dense",
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
    )


@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "expected_q_tile"),
    (
        pytest.param(4, 1, 32, id="gqa4"),
        pytest.param(32, 1, 128, id="mqa32"),
    ),
)
def test_block_sparse_static_profile_accepts_supported_gqa_groups(
    num_qo_heads: int,
    num_kv_heads: int,
    expected_q_tile: int,
) -> None:
    profile = _validate_static_block_sparse_heads(num_qo_heads, num_kv_heads)

    assert profile.q_tile_size == expected_q_tile


def test_paged_block_sparse_static_profile_accepts_token_q_blocks() -> None:
    """A token-level route maps one GQA16 token to one physical Q16 tile."""

    profile = block_sparse_module._validate_block_sparse_static_profile(
        batch_size=1,
        seq_len_q=4,
        seq_len_kv=512,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=1,
        kv_block_size=128,
        use_kv_valid_bits=False,
        mask_type="dense",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        max_blocks_per_row=2,
        page_size=128,
    )

    assert profile.q_tile_size == 16
    assert profile.kv_route_size == 128


@pytest.mark.parametrize(
    ("q_block_size", "heads_q_per_kv"),
    (
        pytest.param(1, 4, id="q1-gqa4"),
        pytest.param(2, 2, id="q2-gqa2"),
        pytest.param(4, 1, id="q4-mha"),
    ),
)
def test_block_sparse_rejects_q_geometry_that_crosses_route_rows(
    q_block_size: int,
    heads_q_per_kv: int,
) -> None:
    with pytest.raises(ValueError, match="no row-pure Q tile"):
        block_sparse_config._select_block_sparse_q_tile_size(
            q_block_size=q_block_size,
            heads_q_per_kv=heads_q_per_kv,
            kv_block_size=128,
        )


@pytest.mark.parametrize(
    ("q_block_size", "error_type", "message"),
    (
        pytest.param(True, TypeError, "Python integer", id="bool"),
        pytest.param(1 << 31, OverflowError, "signed int32", id="int32-overflow"),
    ),
)
def test_block_sparse_rejects_unrepresentable_q_block_sizes(
    q_block_size: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        block_sparse_config._select_block_sparse_q_tile_size(
            q_block_size=q_block_size,
            heads_q_per_kv=8,
            kv_block_size=128,
        )


@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "message"),
    (
        pytest.param(6, 4, "divisible", id="nondivisible"),
        pytest.param(3, 1, "power of two", id="ratio3"),
        pytest.param(64, 1, "at most 32", id="ratio64"),
    ),
)
def test_block_sparse_static_profile_rejects_unsupported_gqa_groups(
    num_qo_heads: int,
    num_kv_heads: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_static_block_sparse_heads(num_qo_heads, num_kv_heads)


@pytest.mark.parametrize(
    (
        "q_block_size",
        "kv_block_size",
        "kv_route_size",
        "num_qo_heads",
        "num_kv_heads",
        "expected_q_tile",
        "persistent",
        "token_mask",
    ),
    (
        pytest.param(128, 128, 128, 8, 8, 128, False, False, id="mha-q128"),
        pytest.param(64, 64, 256, 8, 8, 64, True, True, id="mha-q64-kv256"),
        pytest.param(8, 64, 256, 8, 1, 64, False, False, id="gqa8-q64-kv256"),
        pytest.param(8, 64, 128, 32, 1, 128, False, False, id="mqa32-q128"),
    ),
)
def test_block_sparse_builds_standard_decode_schedule(
    q_block_size: int,
    kv_block_size: int,
    kv_route_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    expected_q_tile: int,
    persistent: bool,
    token_mask: bool,
) -> None:
    """Both physical KV topologies share the sparse Softmax resource ABI."""

    from cutlass.experimental.task_scheduling.enums import ScheduleStage

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _build_decode_gen_schedule,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources import (
        TmemSResource,
    )

    cfg = block_sparse_config._make_block_sparse_config(
        block_sparse_config._BlockSparseCompileKey(
            device_index=0,
            batch_size=1,
            seq_len_q=q_block_size,
            seq_len_kv=4096,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=_HEAD_DIM,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_route_size=kv_route_size,
            dtype_key="float16",
            mask_type="dense",
            use_kv_valid_bits=token_mask,
            use_persistent_scheduler=persistent,
            use_parallel_sparse_kv_loads=False,
        )
    )
    assert cfg.heads_q_per_kv == num_qo_heads // num_kv_heads
    assert cfg.tile_size_q == expected_q_tile
    assert cfg.q_tokens_per_cta == expected_q_tile // cfg.heads_q_per_kv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tasks, *_ = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=8,
            num_heads_kv=num_kv_heads,
        )

    tasks_by_name = {task.name: task for task in tasks}
    assert "LoadTask" in tasks_by_name
    assert "MmaTask" in tasks_by_name
    assert "Softmax0Task" in tasks_by_name
    assert "Softmax1Task" in tasks_by_name
    assert "CorrectionTask" in tasks_by_name
    assert all("Temporal" not in task.name for task in tasks)
    resource_names = {
        resource.name
        for task in tasks
        for resource in (*task.src_resources, *task.dst_resources)
    }
    if kv_route_size == 256:
        assert "smemKv" in resource_names
        assert {"smemK0", "smemK1", "smemV0", "smemV1"}.isdisjoint(resource_names)
    else:
        assert "smemKv" not in resource_names
        assert {"smemK0", "smemK1", "smemV0", "smemV1"} <= resource_names
    for task_name, resource_name in (
        ("Softmax0Task", "tmemS0"),
        ("Softmax1Task", "tmemS1"),
    ):
        softmax_task = tasks_by_name[task_name]
        tmem_s = next(
            resource
            for resource in softmax_task.src_resources
            if resource.name == resource_name
        )
        assert type(tmem_s) is TmemSResource
        consumer_labels = [
            entry[-1]
            for entry in softmax_task.loop_schedule_list
            if entry[0] is tmem_s and entry[1] == ScheduleStage.ConsumerWork
        ]
        assert consumer_labels == ["compute_block_sparse_softmax_loop"]


@pytest.mark.parametrize("storage", ("dense", "sparse"))
def test_decode_schedule_revalidates_mutable_paged_staging_config(
    storage: str,
) -> None:
    """The schedule boundary rejects invalid post-construction mutations."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        FmhaDecodeConfig,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _build_decode_gen_schedule,
    )

    if storage == "dense":
        cfg = FmhaDecodeConfig(
            use_paged_kv=True,
            num_tokens_per_page=64,
            tile_size_kv=128,
            page_offsets_num_warps=2,
        )
        message = "exactly one producer warp"
    else:
        cfg = block_sparse_config._make_block_sparse_config(
            block_sparse_config._BlockSparseCompileKey(
                device_index=0,
                batch_size=1,
                seq_len_q=8,
                seq_len_kv=128,
                num_qo_heads=8,
                num_kv_heads=1,
                head_dim=_HEAD_DIM,
                q_block_size=8,
                kv_block_size=64,
                kv_route_size=256,
                dtype_key="float16",
                mask_type="dense",
                use_kv_valid_bits=False,
                use_persistent_scheduler=False,
                use_parallel_sparse_kv_loads=False,
                page_size=64,
            )
        )
        cfg.num_tokens_per_page = 32
        message = "atom size must not exceed page size"

    with pytest.raises(ValueError, match=message):
        _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=1,
            num_heads_kv=1,
        )


def test_sparse_paged_staging_does_not_require_a_dense_page_offset_warp() -> None:
    cfg = block_sparse_config._make_block_sparse_config(
        block_sparse_config._BlockSparseCompileKey(
            device_index=0,
            batch_size=1,
            seq_len_q=8,
            seq_len_kv=128,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=_HEAD_DIM,
            q_block_size=8,
            kv_block_size=64,
            kv_route_size=256,
            dtype_key="float16",
            mask_type="dense",
            use_kv_valid_bits=False,
            use_persistent_scheduler=False,
            use_parallel_sparse_kv_loads=False,
            page_size=64,
        )
    )
    cfg.page_offsets_num_warps = 2

    cfg.validate_paged_kv_staging_config()


def test_q8_b8_parallel_load_tasks_partition_resources() -> None:
    """The two KV issuers own disjoint instruction-local resources."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _build_decode_gen_schedule,
    )

    cfg = block_sparse_config._make_block_sparse_config(
        block_sparse_config._BlockSparseCompileKey(
            device_index=0,
            batch_size=1,
            seq_len_q=128,
            seq_len_kv=2304,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim=_HEAD_DIM,
            q_block_size=8,
            kv_block_size=8,
            kv_route_size=128,
            dtype_key="bfloat16",
            mask_type="dense",
            use_kv_valid_bits=True,
            use_persistent_scheduler=True,
            use_parallel_sparse_kv_loads=True,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tasks, resource_dependency_graph, *_ = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=18,
            num_heads_kv=8,
        )

    tasks_by_name = {task.name: task for task in tasks}
    load0 = tasks_by_name["LoadTask0"]
    load1 = tasks_by_name["LoadTask1"]
    assert load0.warp_idx != load1.warp_idx
    dst0 = {resource.name for resource in load0.dst_resources}
    dst1 = {resource.name for resource in load1.dst_resources}
    assert dst0.isdisjoint(dst1)
    assert {"smemK0", "smemV0", "smemBlockSparseKvMetadata0"} <= dst0
    assert {"smemK1", "smemV1", "smemBlockSparseKvMetadata1"} <= dst1
    dependency_names = {
        resource.name: {dependency.name for dependency in dependencies}
        for resource, dependencies in resource_dependency_graph.items()
    }
    metadata_names = {
        "smemBlockSparseKvMetadata0",
        "smemBlockSparseKvMetadata1",
    }
    for inst_idx in (0, 1):
        matching_metadata = {f"smemBlockSparseKvMetadata{inst_idx}"}
        assert (
            dependency_names[f"smemK{inst_idx}"] & metadata_names == matching_metadata
        )
        assert (
            dependency_names[f"smemV{inst_idx}"] & metadata_names == matching_metadata
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("q_block_size", "dtype", "route_size"),
    (
        pytest.param(128, torch.bfloat16, 128, id="kv128-bf16"),
        pytest.param(64, torch.float16, 256, id="kv256-fp16"),
    ),
)
@torch.no_grad()
def test_block_sparse_keeps_rescale_threshold_gpu_edges(
    q_block_size: int,
    dtype: torch.dtype,
    route_size: int,
) -> None:
    """Exercise one deferred anchor, one re-anchor, and an all-masked replay."""

    # The two KV instructions own alternating routes. Instruction 0 therefore
    # observes route maxima 0 -> 7.5log2 -> 9log2: defer once, then re-anchor.
    route_scores = (
        0.0,
        0.0,
        7.5 * math.log(2.0),
        0.0,
        9.0 * math.log(2.0),
    )
    route_values = (1.0, 0.5, -2.0, -0.5, 4.0)
    block_size = 64
    seq_len_q = q_block_size
    seq_len_kv = len(route_scores) * route_size
    q = torch.zeros(
        (1, seq_len_q, 1, _HEAD_DIM),
        device="cuda",
        dtype=dtype,
    )
    q[..., 0] = 1.0
    k = torch.zeros(
        (1, seq_len_kv, 1, _HEAD_DIM),
        device="cuda",
        dtype=dtype,
    )
    v = torch.empty_like(k)
    for route_idx, (score, value) in enumerate(
        zip(route_scores, route_values, strict=True)
    ):
        route_slice = slice(
            route_idx * route_size,
            (route_idx + 1) * route_size,
        )
        k[:, route_slice, :, 0] = score
        v[:, route_slice] = value

    num_kv_blocks = seq_len_kv // block_size
    block_indptr = torch.tensor(
        [[[0, num_kv_blocks]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = torch.arange(
        num_kv_blocks,
        device="cuda",
        dtype=torch.int32,
    )
    kv_valid_bits = torch.full(
        (1, seq_len_kv // 32),
        0xFFFFFFFF,
        device="cuda",
        dtype=torch.uint32,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    wrapper.plan(
        1,
        seq_len_q,
        seq_len_kv,
        1,
        1,
        _HEAD_DIM,
        q_block_size,
        block_size,
        device=block_indptr.device,
        max_blocks_per_row=num_kv_blocks,
        use_kv_valid_bits=True,
        q_data_type=dtype,
    )
    state = wrapper._published_state()
    policy = dict(state.policy)
    assert policy["tile_size_q"] == q_block_size
    assert policy["tile_size_kv"] == route_size
    assert policy["use_kv_valid_bits"] is True

    actual = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=kv_valid_bits,
        sm_scale=1.0,
    )
    torch.cuda.synchronize()
    token_scores = k[0, :, 0, 0].float()
    expected_row = torch.softmax(token_scores, dim=0) @ v[0, :, 0].float()
    expected = expected_row.reshape(1, 1, 1, _HEAD_DIM).expand_as(actual)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    kv_valid_bits.zero_()
    all_masked = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=kv_valid_bits,
        sm_scale=1.0,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(all_masked).all()
    assert torch.count_nonzero(all_masked).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_q8_sparse_p_discards_dead_paired_instance() -> None:
    """The straight-line Q8 P helper must discard a dead paired instance."""

    torch.manual_seed(20260803)
    case = _Case(
        name="q8_sparse_p_dead_paired_instance",
        batch_size=1,
        num_heads=1,
        seq_len_q=8,
        seq_len_kv=256,
        q_block_size=8,
        kv_block_size=8,
        dtype=torch.float16,
        mask_type="dense",
        token_mask="holey",
        scheduler="static",
    )
    patterns: _Patterns = (((tuple(range(16)),),),)
    block_indptr, block_indices = _make_bsr(patterns)
    first_half = frozenset(range(64))
    valid_bits = _pack_token_mask(case.seq_len_kv, (first_half,))
    q = torch.randn((1, 8, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, case.seq_len_kv, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        seq_len_q=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        q_block_size=case.q_block_size,
        kv_block_size=case.kv_block_size,
        kv_valid_bits=valid_bits,
    )

    expected = _reference(case, q, k, v, patterns, (first_half,), sm_scale)
    actual = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    valid_bits.zero_()
    all_masked = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(all_masked).all()
    assert torch.count_nonzero(all_masked).item() == 0


def test_attention_core_sparse_abi_uses_only_prepared_metadata() -> None:
    """The attention launch must not retain a second raw-BSR parsing path."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        fmha_block_sparse_launch,
    )

    parameters = inspect.signature(fmha_block_sparse_launch).parameters
    assert "block_sparse_indptr_iter" not in parameters
    assert "block_sparse_indices_iter" not in parameters
    assert "kv_valid_bits_iter" not in parameters
    assert {
        "row_route_offsets_iter",
        "row_route_counts_iter",
        "route_metadata_iter",
    } <= parameters.keys()


@pytest.mark.parametrize(
    (
        "kv_route_size",
        "kv_block_size",
        "has_token_bits",
        "expected_geometry",
    ),
    (
        (128, 8, True, (8, 16, 4, 16, 17, 18, 24)),
        (128, 32, True, (32, 4, 4, 4, 5, 6, 12)),
        (128, 64, True, (64, 2, 4, 2, 3, 4, 8)),
        (256, 64, True, (64, 4, 8, 4, 5, 6, 16)),
        (256, 256, False, (64, 4, 8, 4, 5, None, 8)),
    ),
    ids=(
        "kv128-block8-mask",
        "kv128-block32-mask",
        "kv128-block64-mask",
        "kv256-block64-mask",
        "kv256-block256-no-mask",
    ),
)
def test_prepared_block_sparse_layout_geometry(
    kv_route_size: int,
    kv_block_size: int,
    has_token_bits: bool,
    expected_geometry: tuple[int, int, int, int, int, int | None, int],
) -> None:
    layout = _BlockSparseRouteLayout.create(
        kv_route_size=kv_route_size,
        kv_block_size=kv_block_size,
        has_token_bits=has_token_bits,
        route_metadata_capacity=3,
        num_rows=2,
    )

    assert (
        layout.atom_size,
        layout.logical_origins_per_route,
        layout.token_words_per_route,
        layout.atom_valid_mask_word_offset,
        layout.route_flags_word_offset,
        layout.token_words_word_offset,
        layout.route_metadata_stride_words,
    ) == expected_geometry
    assert layout.route_metadata_base_word_offset == 4
    assert layout.route_metadata_capacity == 3
    assert layout.workspace_size_words == 4 + 3 * layout.route_metadata_stride_words


def test_keeps_softmax_staging_rejects_more_than_four_route_origins() -> None:
    """Keeps consumers have registers for at most four staged origins."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_block_sparse_metadata import (
        _BlockSparseSoftmaxStagingLayout,
    )

    route_layout = SimpleNamespace(
        kv_route_size=256,
        atom_size=32,
        has_token_bits=False,
    )

    with pytest.raises(AssertionError, match="at most four route origins"):
        _BlockSparseSoftmaxStagingLayout.create(
            use_keeps_mma_ab=True,
            route_layout=route_layout,
            num_stages=2,
        )


def test_prepared_route_logical_origin_accessors_are_layout_nfc() -> None:
    """Naming logical origins must not add or move prepared-record words."""

    layout = _BlockSparseRouteLayout.create(
        kv_route_size=128,
        kv_block_size=64,
        has_token_bits=True,
        route_metadata_capacity=3,
        num_rows=2,
    )

    assert tuple(field.name for field in fields(layout)) == (
        "kv_route_size",
        "atom_size",
        "has_token_bits",
        "num_rows",
        "route_metadata_stride_words",
        "route_metadata_base_word_offset",
        "workspace_size_words",
        "page_size",
    )
    assert tuple(getattr(layout, field.name) for field in fields(layout)) == (
        128,
        64,
        True,
        2,
        8,
        4,
        28,
        None,
    )
    assert layout.logical_origins_per_route == 2
    # Logical origins remain the unchanged two-word record prefix.
    assert layout.atom_valid_mask_word_offset == 2
    assert layout.route_flags_word_offset == 3
    assert layout.token_words_word_offset == 4


def test_logical_origins_feed_masks_while_contiguous_loads_are_identity() -> None:
    """Mask coordinates stay logical; contiguous K/V loads preserve them."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import (
        block_sparse_prepare,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources import (
        smem_block_sparse_metadata,
        smem_resources,
        tmem_s,
    )

    assert (
        "logical_origin"
        in inspect.signature(block_sparse_prepare._load_atom_token_chunk).parameters
    )
    assert hasattr(block_sparse_prepare, "_resolve_route_logical_atom_origin")

    resource = smem_block_sparse_metadata.SmemBlockSparseKvMetadataResource
    assert "_prepared_route_logical_origin" in inspect.getsource(resource.resolve_route)
    assert (
        "logical_b_idx" in inspect.signature(resource.route_tma_coordinate).parameters
    )
    kv_load_sources = inspect.getsource(
        smem_resources.SmemKvTileResource
    ) + inspect.getsource(smem_resources.SmemKvResource)
    assert ".route_tma_coordinate(" in kv_load_sources
    assert ".route_origin(" not in kv_load_sources
    assert "route_load_origin" not in inspect.getsource(tmem_s.TmemSResource)


def test_sparse_task_cache_accessors_reuse_the_existing_two_slots() -> None:
    """Sparse route span naming must leave the shared ten-word ABI intact."""

    from cutlass import Int32

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources import (
        helpers_common,
    )

    assert len(get_args(helpers_common.TaskCache)) == 10
    assert (
        helpers_common._TASK_CACHE_SPARSE_ROUTE_BEGIN
        == helpers_common._TASK_CACHE_KV_REQUEST_BEGIN
        == 5
    )
    assert (
        helpers_common._TASK_CACHE_SPARSE_ROUTE_COUNT
        == helpers_common._TASK_CACHE_KV_PAGE_IDX_UB
        == 6
    )
    cache = tuple(Int32(i) for i in range(10))
    route_begin = helpers_common._sparse_task_cache_route_begin.__wrapped__(cache)
    route_count = helpers_common._sparse_task_cache_route_count.__wrapped__(cache)
    assert (int(route_begin), int(route_count)) == (5, 6)


def test_paged_route_seam_extends_only_the_shared_compile_key() -> None:
    """Storage axes live on the shared key without an intermediate policy type."""

    assert tuple(
        field.name for field in fields(block_sparse_config._BlockSparseCompileKey)
    ) == (
        "device_index",
        "batch_size",
        "seq_len_q",
        "seq_len_kv",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_block_size",
        "kv_block_size",
        "kv_route_size",
        "dtype_key",
        "mask_type",
        "use_kv_valid_bits",
        "use_persistent_scheduler",
        "use_parallel_sparse_kv_loads",
        "page_size",
    )
    assert not hasattr(block_sparse_module, "_BlockSparseExecutionPolicy")
    assert not hasattr(block_sparse_module, "_resolve_block_sparse_execution_policy")
    assert not hasattr(
        block_sparse_module,
        "_resolve_validated_block_sparse_execution_policy",
    )


@pytest.mark.parametrize(
    ("page_size", "expected_use_paged_kv"),
    (
        pytest.param(None, False, id="contiguous"),
        pytest.param(64, True, id="paged"),
    ),
)
def test_block_sparse_config_derives_storage_layout_from_page_size(
    page_size: int | None,
    expected_use_paged_kv: bool,
) -> None:
    cfg = block_sparse_config._make_block_sparse_config(
        block_sparse_config._BlockSparseCompileKey(
            device_index=0,
            batch_size=1,
            seq_len_q=64,
            seq_len_kv=256,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=_HEAD_DIM,
            q_block_size=64,
            kv_block_size=64,
            kv_route_size=256,
            dtype_key="float16",
            mask_type="dense",
            use_kv_valid_bits=True,
            use_persistent_scheduler=False,
            use_parallel_sparse_kv_loads=False,
            page_size=page_size,
        )
    )

    assert cfg.use_paged_kv is expected_use_paged_kv
    if page_size is not None:
        assert cfg.num_tokens_per_page == page_size


@pytest.mark.parametrize(
    ("kv_block_size", "expected_words"),
    ((8, 16), (16, 8), (32, 4), (64, 4)),
)
def test_kv_retained_route_storage_follows_origin_count(
    kv_block_size: int,
    expected_words: int,
) -> None:
    """K-to-V retention keeps only origins and coarse route validity."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_block_sparse_metadata import (
        _kv_retained_route_words,
    )

    layout = _BlockSparseRouteLayout.create(
        kv_route_size=128,
        kv_block_size=kv_block_size,
        has_token_bits=True,
        route_metadata_capacity=0,
        num_rows=1,
    )

    assert _kv_retained_route_words(layout) == expected_words


@pytest.mark.parametrize(
    ("kv_route_size", "route_capacity", "expected_offsets"),
    (
        (128, 2, [0, 2]),
        (256, 1, [0, 1]),
    ),
)
def test_route_storage_capacity_follows_route_width(
    kv_route_size: int,
    route_capacity: int,
    expected_offsets: list[int],
) -> None:
    """Static row packing must use the same route width as its metadata."""

    route_layout = _BlockSparseRouteLayout.create(
        kv_route_size=kv_route_size,
        kv_block_size=64,
        has_token_bits=False,
        route_metadata_capacity=route_capacity,
        num_rows=1,
    )
    row_route_offsets, _ = block_sparse_plan._allocate_route_storage(
        device=torch.device("cpu"),
        route_layout=route_layout,
        uniform_row_route_capacity=route_capacity,
    )

    assert row_route_offsets.tolist() == expected_offsets


def test_route_storage_rejects_int32_offset_overflow_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route ordinals must remain representable by the device Int32 ABI."""

    def unexpected_arange(*_args: object, **_kwargs: object) -> None:
        pytest.fail("route-offset overflow must fail before device allocation")

    monkeypatch.setattr(torch, "arange", unexpected_arange)
    route_layout = SimpleNamespace(
        num_rows=2,
        route_metadata_capacity=1 << 31,
        workspace_size_words=0,
    )

    with pytest.raises(OverflowError, match="route capacity.*signed int32"):
        block_sparse_plan._allocate_route_storage(
            device=torch.device("cpu"),
            route_layout=route_layout,
            uniform_row_route_capacity=1 << 30,
        )


def test_prepared_route_alignment_uses_route_width() -> None:
    """TensorMap eligibility is relative to one complete prepared route."""

    from flashinfer.attention.prims_ts._block_sparse.common import (
        _prepared_kv_routes_are_block_aligned,
    )

    assert _prepared_kv_routes_are_block_aligned(128, 128)
    assert not _prepared_kv_routes_are_block_aligned(128, 256)
    assert _prepared_kv_routes_are_block_aligned(256, 256)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    (
        (
            {"route_metadata_capacity": (1 << 31) - 1},
            OverflowError,
            "workspace_size_words",
        ),
        ({"num_rows": (1 << 31) - 1}, OverflowError, "row_route_offsets_length"),
        ({"route_metadata_capacity": True}, TypeError, "route_metadata_capacity"),
        ({"route_metadata_capacity": -1}, ValueError, "route_metadata_capacity"),
        ({"num_rows": 0}, ValueError, "num_rows"),
        ({"page_size": 32}, ValueError, "atom_size"),
        ({"page_size": 96}, ValueError, "page_size"),
    ),
    ids=(
        "route-metadata-address-overflow",
        "row-offset-length-overflow",
        "bool-route-metadata-capacity",
        "negative-route-metadata-capacity",
        "zero-num-rows",
        "paged-atom-larger-than-page",
        "unsupported-page-size",
    ),
)
def test_prepared_block_sparse_layout_rejects_invalid_extents(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "kv_route_size": 128,
        "kv_block_size": 64,
        "has_token_bits": True,
        "route_metadata_capacity": 1,
        "num_rows": 1,
    }
    arguments.update(overrides)

    with pytest.raises(error_type, match=message):
        _BlockSparseRouteLayout.create(**arguments)


def test_prepared_block_sparse_layout_rejects_unsupported_route_width() -> None:
    with pytest.raises(ValueError, match="kv_route_size must be 128 or 256"):
        _BlockSparseRouteLayout.create(
            kv_route_size=192,
            kv_block_size=64,
            has_token_bits=True,
            route_metadata_capacity=1,
            num_rows=1,
        )


def test_prepared_block_sparse_layout_allows_empty_route_metadata() -> None:
    layout = _BlockSparseRouteLayout.create(
        kv_route_size=128,
        kv_block_size=64,
        has_token_bits=False,
        route_metadata_capacity=0,
        num_rows=3,
    )

    assert layout.route_metadata_base_word_offset == 4
    assert layout.route_metadata_capacity == 0
    assert layout.workspace_size_words == layout.route_metadata_base_word_offset


def test_public_api_rejects_invalid_usage() -> None:
    metadata = torch.empty((1, 1, 2), dtype=torch.int32)
    indices = torch.empty((0,), dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"plan\(\).*before run"):
        q = torch.empty((1, 1, 1, _HEAD_DIM), dtype=torch.float16)
        block_sparse_module.BlockSparseTSWrapper().run(
            q,
            q,
            q,
            metadata,
            indices,
        )

    k = torch.empty((1, 128, 1, _HEAD_DIM), dtype=torch.float16)
    with pytest.raises(ValueError, match="q must be rank 4"):
        block_sparse_module.block_sparse_attention(
            torch.empty(0),
            k,
            k,
            metadata,
            indices,
            64,
            64,
        )


def _validate_cpu_routing(
    *,
    block_indptr: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    kv_valid_bits: torch.Tensor | None = None,
    use_kv_valid_bits: bool = False,
) -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        validate_block_sparse_metadata,
    )

    validate_block_sparse_metadata(
        torch.tensor([[[0, 0]]], dtype=torch.int32)
        if block_indptr is None
        else block_indptr,
        torch.empty(0, dtype=torch.int32) if block_indices is None else block_indices,
        kv_valid_bits,
        device=torch.device("cpu"),
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=1,
        num_kv_heads=1,
        q_block_size=1,
        use_kv_valid_bits=use_kv_valid_bits,
    )


def test_runtime_metadata_allows_unreferenced_spare_indices() -> None:
    """Per-row capacity is checked on device; spare flat storage is legal."""

    _validate_cpu_routing(
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
        block_indices=torch.arange(17, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    (
        pytest.param(
            {"block_indptr": torch.empty((1, 2), dtype=torch.int32)},
            ValueError,
            "block_indptr must be rank 3",
            id="rank",
        ),
        pytest.param(
            {"block_indices": torch.empty(0, dtype=torch.int64)},
            TypeError,
            "block_indices must have dtype torch.int32",
            id="dtype",
        ),
        pytest.param(
            {"block_indices": torch.empty(0, dtype=torch.int32, device="meta")},
            ValueError,
            "block_indices must be on planned device cpu",
            id="device",
        ),
        pytest.param(
            {"block_indptr": torch.empty((1, 1, 4), dtype=torch.int32)[..., ::2]},
            ValueError,
            "block_indptr must have compact rank-3 strides",
            id="stride",
        ),
        pytest.param(
            {
                "block_indices": torch.frombuffer(
                    bytearray(9),
                    dtype=torch.int32,
                    count=2,
                    offset=1,
                )
            },
            ValueError,
            "block_indices data pointer must be 4-byte aligned",
            id="alignment",
        ),
        pytest.param(
            {"use_kv_valid_bits": True},
            ValueError,
            "kv_valid_bits is required",
            id="missing-mask",
        ),
        pytest.param(
            {"kv_valid_bits": torch.empty((1, 1), dtype=torch.uint32)},
            ValueError,
            "kv_valid_bits must be None",
            id="unexpected-mask",
        ),
    ),
)
def test_runtime_metadata_rejects_invalid_abi(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _validate_cpu_routing(**overrides)


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    (
        (None, TypeError, "Python integer"),
        ("1", TypeError, "Python integer"),
        (True, TypeError, "Python integer"),
        (-1, ValueError, "non-negative"),
        (3, ValueError, "number of semantic KV blocks"),
    ),
)
def test_max_blocks_per_row_validation(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        block_sparse_config._validate_max_blocks_per_row(
            value,
            seq_len_kv=128,
            kv_block_size=64,
        )


def test_gqa_runtime_uses_distinct_q_and_kv_head_shapes() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _ContiguousKVStorage,
        validate_block_sparse_run,
    )

    state = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        seq_len_q=8,
        seq_len_kv=64,
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=8,
        use_kv_valid_bits=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        dummy_kv_valid_bits=torch.zeros((1, 2), dtype=torch.uint32),
        row_route_offsets=torch.zeros(2, dtype=torch.int32),
        route_workspace=torch.zeros(4, dtype=torch.int32),
        page_size=None,
    )
    q = torch.empty((1, 8, 8, _HEAD_DIM), dtype=torch.float16)
    k = torch.empty((1, 64, 1, _HEAD_DIM), dtype=torch.float16)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    run_args = validate_block_sparse_run(
        q,
        _ContiguousKVStorage(k=k, v=v),
        state=state,
        block_indptr=torch.zeros((1, 1, 2), dtype=torch.int32),
        block_indices=torch.empty(0, dtype=torch.int32),
        kv_valid_bits=None,
        sm_scale=None,
        out=out,
    )

    assert run_args.q is q
    assert run_args.k is k
    assert run_args.v is v
    assert run_args.out is out


def test_contiguous_runtime_records_every_launch_tensor_on_the_run_stream() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _BlockSparseRunArgs,
        record_block_sparse_run_args,
    )

    calls: list[tuple[str, object]] = []

    class RecordableTensor:
        def __init__(self, name: str) -> None:
            self.name = name

        def record_stream(self, stream: object) -> None:
            calls.append((self.name, stream))

    tensor_names = (
        "q",
        "k",
        "v",
        "out",
        "block_indptr",
        "block_indices",
        "kv_valid_bits",
    )
    tensors = {name: RecordableTensor(name) for name in tensor_names}
    run_args = _BlockSparseRunArgs(
        **tensors,
        kv_valid_bits_is_live=True,
        sm_scale=1.0,
        paged_kv=None,
    )
    run_stream = object()

    record_block_sparse_run_args(run_args, run_stream)

    assert calls == [(name, run_stream) for name in tensor_names]


def test_contiguous_launch_forwards_the_exact_compiled_adapter_abi() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _BlockSparseRunArgs,
        launch_block_sparse,
    )

    state = SimpleNamespace(
        page_size=None,
        row_route_offsets=object(),
        route_workspace=object(),
        max_blocks_per_row=3,
        compiled=lambda *args: calls.append(args),
    )
    values = {
        name: object()
        for name in (
            "q",
            "k",
            "v",
            "out",
            "block_indptr",
            "block_indices",
            "kv_valid_bits",
        )
    }
    calls: list[tuple[object, ...]] = []
    state.compiled = lambda *args: calls.append(args)
    run_args = _BlockSparseRunArgs(
        **values,
        kv_valid_bits_is_live=True,
        sm_scale=1.25,
        paged_kv=None,
    )

    result = launch_block_sparse(
        run_args,
        state=state,
    )

    assert result is run_args.out
    assert calls == [
        (
            run_args.q,
            run_args.k,
            run_args.v,
            run_args.out,
            run_args.block_indptr,
            run_args.block_indices,
            run_args.kv_valid_bits,
            state.row_route_offsets,
            state.route_workspace,
            3,
            1.25,
        )
    ]


def test_paged_launch_forwards_caller_live_lengths_to_attention() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _BlockSparseRunArgs,
        _PagedKVLaunchPayload,
        launch_block_sparse,
    )

    calls: list[tuple[object, ...]] = []
    state = SimpleNamespace(
        page_size=64,
        row_route_offsets=object(),
        route_workspace=object(),
        max_blocks_per_row=3,
        compiled=lambda *args: calls.append(args),
    )
    values = {
        name: object()
        for name in (
            "q",
            "k",
            "v",
            "out",
            "block_indptr",
            "block_indices",
            "kv_valid_bits",
        )
    }
    live_seq_lens_kv = object()
    paged_kv = _PagedKVLaunchPayload(
        paged_kv_indptr=object(),
        paged_kv_indices=object(),
        seq_lens_kv=live_seq_lens_kv,
        num_physical_kv_pages=17,
        k_page_stride=19,
        v_page_stride=23,
    )
    run_args = _BlockSparseRunArgs(
        **values,
        kv_valid_bits_is_live=True,
        sm_scale=1.25,
        paged_kv=paged_kv,
    )

    result = launch_block_sparse(run_args, state=state)

    assert result is run_args.out
    assert calls == [
        (
            run_args.q,
            run_args.k,
            run_args.v,
            run_args.out,
            run_args.block_indptr,
            run_args.block_indices,
            run_args.kv_valid_bits,
            paged_kv.paged_kv_indptr,
            paged_kv.paged_kv_indices,
            live_seq_lens_kv,
            state.row_route_offsets,
            state.route_workspace,
            3,
            17,
            19,
            23,
            1.25,
        )
    ]


@pytest.mark.parametrize(
    "aliased_name",
    ("block_indptr", "block_indices", "kv_valid_bits"),
)
def test_runtime_output_must_not_alias_sparse_metadata(aliased_name: str) -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _ContiguousKVStorage,
        validate_block_sparse_run,
    )

    shape = (1, 1, 1, _HEAD_DIM)
    q = torch.empty(shape, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    block_indptr_storage = torch.empty(_HEAD_DIM // 2, dtype=torch.int32)
    block_indices = torch.empty(_HEAD_DIM // 2, dtype=torch.int32)
    kv_valid_bits_storage = torch.empty(_HEAD_DIM // 2, dtype=torch.uint32)
    block_indptr = block_indptr_storage[:2].view(1, 1, 2)
    kv_valid_bits = kv_valid_bits_storage[:1].view(1, 1)
    aliased_tensor = {
        "block_indptr": block_indptr_storage,
        "block_indices": block_indices,
        "kv_valid_bits": kv_valid_bits_storage,
    }[aliased_name]
    out = torch.empty(0, dtype=torch.float16).set_(
        aliased_tensor.untyped_storage(),
        0,
        shape,
    )

    with pytest.raises(
        ValueError,
        match=rf"out must not overlap {aliased_name} storage",
    ):
        state = SimpleNamespace(
            device=torch.device("cpu"),
            batch_size=1,
            seq_len_q=1,
            seq_len_kv=1,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=_HEAD_DIM,
            q_block_size=1,
            use_kv_valid_bits=True,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            output_dtype=torch.float16,
            dummy_kv_valid_bits=None,
            row_route_offsets=torch.zeros(2, dtype=torch.int32),
            route_workspace=torch.zeros(4, dtype=torch.int32),
            page_size=None,
        )
        validate_block_sparse_run(
            q,
            _ContiguousKVStorage(k=k, v=v),
            state=state,
            block_indptr=block_indptr,
            block_indices=block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=None,
            out=out,
        )


def test_runtime_output_must_not_alias_plan_owned_route_workspace() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        _ContiguousKVStorage,
        validate_block_sparse_run,
    )

    shape = (1, 1, 1, _HEAD_DIM)
    q = torch.empty(shape, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    route_workspace = torch.empty(_HEAD_DIM // 2, dtype=torch.int32)
    out = torch.empty(0, dtype=torch.float16).set_(
        route_workspace.untyped_storage(),
        0,
        shape,
    )
    state = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        seq_len_q=1,
        seq_len_kv=1,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=1,
        use_kv_valid_bits=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
        dummy_kv_valid_bits=torch.zeros((1, 1), dtype=torch.uint32),
        row_route_offsets=torch.zeros(2, dtype=torch.int32),
        route_workspace=route_workspace,
        page_size=None,
    )

    with pytest.raises(
        ValueError,
        match=r"out must not overlap route_workspace storage",
    ):
        validate_block_sparse_run(
            q,
            _ContiguousKVStorage(k=k, v=v),
            state=state,
            block_indptr=torch.zeros((1, 1, 2), dtype=torch.int32),
            block_indices=torch.empty(0, dtype=torch.int32),
            kv_valid_bits=None,
            sm_scale=None,
            out=out,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param({"q_block_size": 0}, "positive", id="q-block-size"),
        pytest.param(
            {"num_qo_heads": 3},
            "power of two",
            id="unsupported-gqa-ratio",
        ),
        pytest.param(
            {"batch_size": 65_536},
            r"grid\.z.*batch_size=65536",
            id="batch-grid-limit",
        ),
        pytest.param(
            {"num_qo_heads": 65_536, "num_kv_heads": 65_536},
            r"grid\.y.*num_kv_heads=65536",
            id="head-grid-limit",
        ),
    ),
)
def test_plan_rejects_unsupported_profile(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "batch_size": 1,
        "seq_len_q": 64,
        "seq_len_kv": 128,
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim": _HEAD_DIM,
        "q_block_size": 64,
        "kv_block_size": 64,
        "device": "cuda",
        "max_blocks_per_row": 1,
        "use_kv_valid_bits": False,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        block_sparse_module.BlockSparseTSWrapper().plan(**arguments)


@pytest.mark.parametrize(
    (
        "q_tile_size",
        "kv_block_size",
        "mask_type",
        "max_row_route_capacity",
        "use_kv_valid_bits",
        "expected",
    ),
    (
        pytest.param(8, 8, "dense", 128, False, True, id="q8-b8-route128"),
        pytest.param(8, 8, "dense", 129, False, False, id="q8-b8-route129"),
        pytest.param(8, 8, "causal", 8, False, True, id="q8-causal-route8"),
        pytest.param(8, 8, "causal", 9, False, False, id="q8-b8-causal-route9"),
        pytest.param(8, 8, "dense", 12, True, True, id="q8-mask-route12"),
        pytest.param(8, 8, "dense", 13, True, False, id="q8-mask-route13"),
        pytest.param(8, 32, "dense", 8, False, True, id="q8-b32-route8"),
        pytest.param(8, 32, "dense", 9, False, False, id="q8-b32-route9"),
        pytest.param(16, 16, "dense", 256, True, True, id="q16-bypass"),
    ),
)
def test_block_sparse_clc_candidate_capacity_gate(
    q_tile_size: int,
    kv_block_size: int,
    mask_type: str,
    max_row_route_capacity: int,
    use_kv_valid_bits: bool,
    expected: bool,
) -> None:
    assert (
        block_sparse_config._should_consider_clc(
            q_tile_size=q_tile_size,
            kv_block_size=kv_block_size,
            mask_type=mask_type,
            max_row_route_capacity=max_row_route_capacity,
            use_kv_valid_bits=use_kv_valid_bits,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "kv_block_size",
        "use_kv_valid_bits",
        "max_row_route_capacity",
        "use_persistent_scheduler",
        "expected",
    ),
    (
        pytest.param(8, True, 4, True, False, id="b8-mask-short"),
        pytest.param(8, True, 5, True, True, id="b8-mask-three-pairs"),
        pytest.param(8, False, 4, True, True, id="b8-unmasked-short"),
        pytest.param(16, True, 6, True, False, id="b16-three-pairs"),
        pytest.param(16, True, 7, True, True, id="b16-four-pairs"),
        pytest.param(8, True, 4, False, True, id="b8-static"),
        pytest.param(32, True, 16, True, False, id="b32-combined"),
    ),
)
def test_block_sparse_parallel_kv_load_policy(
    kv_block_size: int,
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
    use_persistent_scheduler: bool,
    expected: bool,
) -> None:
    assert (
        block_sparse_config._select_parallel_sparse_kv_loads(
            kv_block_size=kv_block_size,
            use_kv_valid_bits=use_kv_valid_bits,
            max_row_route_capacity=max_row_route_capacity,
            use_persistent_scheduler=use_persistent_scheduler,
        )
        is expected
    )


def test_sparse_execution_policy_rejects_dense_decode_config() -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        FmhaDecodeConfig,
    )

    cfg = FmhaDecodeConfig(use_parallel_sparse_kv_loads=True)
    with pytest.raises(ValueError, match="requires block-sparse"):
        cfg.validate_block_sparse_profile(heads_q_per_kv=1)


def test_sparse_execution_policy_rejects_b32_parallel_loads() -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        FmhaDecodeConfig,
    )

    cfg = FmhaDecodeConfig(
        use_block_sparse=True,
        groups_tokens_heads_q=True,
        q_block_size=32,
        kv_block_size=32,
        tile_size_q=32,
        use_parallel_sparse_kv_loads=True,
    )
    with pytest.raises(ValueError, match="KV block size 8 or 16"):
        cfg.validate_block_sparse_profile(heads_q_per_kv=1)


def test_parallel_sparse_kv_load_capability_is_independent_of_q_tile_size() -> None:
    """A Swaps Q32 profile may reuse the same fine-KV issuer topology."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        FmhaDecodeConfig,
    )

    cfg = FmhaDecodeConfig(
        use_block_sparse=True,
        groups_tokens_heads_q=True,
        q_block_size=32,
        kv_block_size=16,
        tile_size_q=32,
        use_parallel_sparse_kv_loads=True,
    )
    cfg.validate_block_sparse_profile(heads_q_per_kv=1)


def test_block_sparse_clc_requires_about_two_sm_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial second wave stays static; CLC starts beyond two waves."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    config_module = importlib.import_module(
        "flashinfer.attention.prims_ts._block_sparse.config"
    )

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(
        fmha_decode_config.utils,
        "HardwareInfo",
        _FourSmHardware,
    )
    monkeypatch.setattr(
        config_module,
        "_make_block_sparse_config",
        lambda _key: None,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())

    common_arguments = dict(
        device_index=0,
        batch_size=1,
        seq_len_kv=4096,
        num_qo_heads=5,
        num_kv_heads=5,
        head_dim=_HEAD_DIM,
        q_block_size=64,
        kv_block_size=64,
        kv_route_size=256,
        dtype_key="bfloat16",
        mask_type="dense",
        use_kv_valid_bits=True,
        max_row_route_capacity=8,
    )
    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        launch_specs = tuple(
            dict(
                block_sparse_config._resolve_block_sparse_launch_spec(
                    **common_arguments,
                    seq_len_q=seq_len_q,
                ).policy
            )
            for seq_len_q in (64, 128)
        )
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()

    # Five CTAs are only 1.25 waves on this synthetic device; ten CTAs provide
    # enough work for CLC to steal after its request/response overhead.
    assert tuple(spec["use_persistent_scheduler"] for spec in launch_specs) == (
        False,
        True,
    )


def test_gqa_launch_spec_uses_q_token_cta_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    config_module = importlib.import_module(
        "flashinfer.attention.prims_ts._block_sparse.config"
    )
    selector_calls: list[dict[str, object]] = []

    def select_static(**kwargs: object) -> str:
        selector_calls.append(kwargs)
        return "static"

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        select_static,
    )
    monkeypatch.setattr(
        config_module,
        "_make_block_sparse_config",
        lambda _key: None,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())

    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        spec = block_sparse_config._resolve_block_sparse_launch_spec(
            device_index=0,
            batch_size=1,
            seq_len_q=17,
            seq_len_kv=4096,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=_HEAD_DIM,
            q_block_size=8,
            kv_block_size=64,
            kv_route_size=256,
            dtype_key="float16",
            mask_type="dense",
            use_kv_valid_bits=False,
            max_row_route_capacity=4,
        )
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()

    assert selector_calls == [
        {
            "batch_size": 1,
            "num_heads_kv": 1,
            "seq_len_kv": 1024,
            "num_q_tiles": 3,
            "tile_size_kv": 256,
            "persistent_min_waves": 2,
            "persistent_min_tiles_per_cta": 1,
        }
    ]
    assert spec.compile_key.num_qo_heads == 8
    assert spec.compile_key.num_kv_heads == 1


def test_clc_capacity_gates_control_launch_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route bounds must gate the selector used by the final policy."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    config_module = importlib.import_module(
        "flashinfer.attention.prims_ts._block_sparse.config"
    )
    selector_calls: list[dict[str, object]] = []

    def select_persistent(**kwargs: object) -> str:
        selector_calls.append(kwargs)
        return "persistent"

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        select_persistent,
    )
    monkeypatch.setattr(
        config_module,
        "_make_block_sparse_config",
        lambda _key: None,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())

    common_arguments = dict(
        device_index=0,
        batch_size=1,
        seq_len_q=128,
        seq_len_kv=4096,
        num_qo_heads=32,
        num_kv_heads=32,
        head_dim=_HEAD_DIM,
        q_block_size=8,
        kv_block_size=8,
        kv_route_size=128,
        dtype_key="bfloat16",
        mask_type="dense",
    )
    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        for capacity, expected_persistent in ((12, True), (13, False)):
            selector_calls.clear()
            spec = block_sparse_config._resolve_block_sparse_launch_spec(
                **common_arguments,
                use_kv_valid_bits=True,
                max_row_route_capacity=capacity,
            )
            assert dict(spec.policy)["use_persistent_scheduler"] is expected_persistent
            assert dict(spec.policy)["use_parallel_sparse_kv_loads"] is True
            assert len(selector_calls) == int(expected_persistent)
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()


def test_static_fallback_reselects_sparse_load_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected CLC profile must reselect its sparse load topology."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    config_module = importlib.import_module(
        "flashinfer.attention.prims_ts._block_sparse.config"
    )
    config_calls: list[object] = []

    def validate_config(key: object) -> None:
        config_calls.append(key)
        if key.use_persistent_scheduler:
            raise ValueError("reject persistent profile")

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "persistent",
    )
    monkeypatch.setattr(
        config_module,
        "_make_block_sparse_config",
        validate_config,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _index: nullcontext())

    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        spec = block_sparse_config._resolve_block_sparse_launch_spec(
            device_index=0,
            batch_size=1,
            seq_len_q=128,
            seq_len_kv=4096,
            num_qo_heads=32,
            num_kv_heads=32,
            head_dim=_HEAD_DIM,
            q_block_size=8,
            kv_block_size=8,
            kv_route_size=128,
            dtype_key="bfloat16",
            mask_type="dense",
            use_kv_valid_bits=True,
            max_row_route_capacity=4,
        )
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()

    assert [key.use_persistent_scheduler for key in config_calls] == [True, False]
    assert config_calls[0].use_parallel_sparse_kv_loads is False
    assert config_calls[1].use_parallel_sparse_kv_loads is True
    policy = dict(spec.policy)
    assert policy["use_persistent_scheduler"] is False
    assert policy["use_parallel_sparse_kv_loads"] is True


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("indptr", "indices", "message"),
    (
        pytest.param([[[0, 2]]], (1, 0), "strictly increasing", id="unsorted"),
        pytest.param([[[0, 2]]], (0, 2), "in-range KV block", id="out-of-range"),
        pytest.param(
            [[[0, 2]]],
            (0,),
            "bounded and monotone",
            id="invalid-indptr",
        ),
    ),
)
def test_one_shot_rejects_noncanonical_bsr(
    indptr: list[list[list[int]]],
    indices: tuple[int, ...],
    message: str,
) -> None:
    block_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    block_indices = torch.tensor(indices, device="cuda", dtype=torch.int32)
    q = torch.empty((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 128, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match=rf"canonical BSR.*{message}"):
        block_sparse_module.block_sparse_attention(
            q,
            k,
            k,
            block_indptr,
            block_indices,
            64,
            64,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_block_capacity_is_not_weakened_by_route_packing() -> None:
    """Runtime preparation enforces the semantic block bound before packing."""

    block_indptr = torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32)
    block_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    # A prepared route may pack multiple B64 semantic blocks. Reserve the
    # second index slot, then prove that replay still enforces max_blocks=1.
    block_indptr.copy_(torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32))
    block_indices[1] = -1
    valid_bits = torch.full(
        (1, 4),
        0xFFFFFFFF,
        device="cuda",
        dtype=torch.uint32,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
        max_blocks_per_row=1,
    )
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    block_indptr.copy_(torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32))

    overflow = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
    )
    torch.cuda.synchronize()

    state = wrapper._published_state()
    assert state.route_workspace[0].item() < 0
    assert torch.count_nonzero(overflow).item() == 0

    # Contract-invalid live IDs must still fail safely before address
    # arithmetic; in particular, the token-mask path must not read word -1.
    block_indices[0] = -1
    block_indptr.copy_(torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32))
    invalid_index = wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(invalid_index).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "indices",
    (
        pytest.param((0, 0), id="duplicate"),
        pytest.param((2, 1), id="unsorted"),
        pytest.param((0, 4, 2), id="mixed-valid-out-of-range"),
    ),
)
@torch.no_grad()
def test_runtime_routes_fail_closed_for_noncanonical_row(
    indices: tuple[int, ...],
) -> None:
    """Prepare rejects the whole row before retaining any partial route."""

    block_indptr = torch.tensor(
        [[[0, len(indices)]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = torch.tensor(indices, device="cuda", dtype=torch.int32)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        seq_len_kv=256,
        max_blocks_per_row=3,
    )
    q = torch.zeros((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.zeros((1, 256, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.ones_like(k)

    out = wrapper.run(q, k, v, block_indptr, block_indices)
    torch.cuda.synchronize()

    state = wrapper._published_state()
    assert state.route_workspace[0].item() < 0
    assert torch.count_nonzero(out).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_one_shot_all_empty_bsr_returns_finite_zero() -> None:
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_indptr = torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32)
    block_indices = torch.empty(0, device="cuda", dtype=torch.int32)

    out = block_sparse_module.block_sparse_attention(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        64,
        64,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_plan_owns_uniform_route_storage_for_skewed_rows() -> None:
    """Every row receives the declared maximum route capacity."""

    patterns: _Patterns = (
        (
            ((), (0,), (0, 2)),
            ((1, 2), (), (2,)),
        ),
    )
    block_indptr, block_indices = _make_bsr(patterns)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        seq_len_q=192,
        seq_len_kv=513,
        num_heads=2,
        q_block_size=64,
        kv_block_size=256,
    )

    state = wrapper._published_state()
    policy = dict(state.policy)
    route_layout = _BlockSparseRouteLayout.create(
        kv_route_size=policy["tile_size_kv"],
        kv_block_size=256,
        has_token_bits=False,
        route_metadata_capacity=12,
        num_rows=6,
    )
    assert state.row_route_offsets.tolist() == [0, 2, 4, 6, 8, 10, 12]
    assert state.route_workspace.numel() == route_layout.workspace_size_words
    assert policy["max_row_route_capacity"] == 2


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_public_block_sparse_correctness(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: case.scheduler,
    )
    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    torch.manual_seed(20260716)
    patterns = _make_patterns(case)
    block_indptr, block_indices = _make_bsr(patterns)
    valid_bits, valid_by_batch = _make_token_mask(case)
    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
        device="cuda",
        dtype=case.dtype,
    )
    k = torch.randn(
        (case.batch_size, case.seq_len_kv, case.num_heads, _HEAD_DIM),
        device="cuda",
        dtype=case.dtype,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    expected = _reference(
        case,
        q,
        k,
        v,
        patterns,
        valid_by_batch,
        sm_scale,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    one_shot_actual = None

    try:
        max_blocks_per_row = max(
            len(row) for batch in patterns for head in batch for row in head
        )
        wrapper.plan(
            case.batch_size,
            case.seq_len_q,
            case.seq_len_kv,
            case.num_heads,
            case.num_heads,
            _HEAD_DIM,
            case.q_block_size,
            case.kv_block_size,
            device=block_indptr.device,
            max_blocks_per_row=max_blocks_per_row,
            use_kv_valid_bits=valid_bits is not None,
            mask_type=case.mask_type,
            q_data_type=case.dtype,
        )
        policy = dict(wrapper._policy)
        assert policy["use_persistent_scheduler"] == (case.scheduler == "persistent")
        assert policy["use_kv_valid_bits"] == (valid_bits is not None)
        if case.q_block_size == 64 and case.kv_block_size % 64 == 0:
            assert policy["tile_size_kv"] == 256
        if case.expected_parallel_loads is not None:
            assert (
                policy["use_parallel_sparse_kv_loads"] is case.expected_parallel_loads
            )
        actual = wrapper.run(
            q,
            k,
            v,
            block_indptr,
            block_indices,
            kv_valid_bits=valid_bits,
            sm_scale=sm_scale,
        )
        if case.q_block_size == case.kv_block_size == 64:
            one_shot_actual = block_sparse_module.block_sparse_attention(
                q,
                k,
                v,
                block_indptr,
                block_indices,
                case.q_block_size,
                case.kv_block_size,
                kv_valid_bits=valid_bits,
                mask_type=case.mask_type,
                sm_scale=sm_scale,
            )
        torch.cuda.synchronize()
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()

    if case.include_empty_row:
        empty_row_begin = (
            math.ceil(case.seq_len_q / case.q_block_size) - 1
        ) * case.q_block_size
        assert torch.count_nonzero(actual[-1, empty_row_begin:, -1]).item() == 0
    tolerance = 2e-2 if case.dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    if one_shot_actual is not None:
        torch.testing.assert_close(
            one_shot_actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _GQA_CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_public_block_sparse_gqa_correctness(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: case.scheduler,
    )
    block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    torch.manual_seed(20260814)
    patterns = _make_patterns(case)
    block_indptr, block_indices = _make_bsr(patterns)
    valid_bits, valid_by_batch = _make_token_mask(case)
    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
        device="cuda",
        dtype=case.dtype,
    )
    k = torch.randn(
        (
            case.batch_size,
            case.seq_len_kv,
            case.effective_num_kv_heads,
            _HEAD_DIM,
        ),
        device="cuda",
        dtype=case.dtype,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    expected = _reference(
        case,
        q,
        k,
        v,
        patterns,
        valid_by_batch,
        sm_scale,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()

    try:
        max_blocks_per_row = max(
            len(row) for batch in patterns for head in batch for row in head
        )
        wrapper.plan(
            case.batch_size,
            case.seq_len_q,
            case.seq_len_kv,
            case.num_heads,
            case.effective_num_kv_heads,
            _HEAD_DIM,
            case.q_block_size,
            case.kv_block_size,
            device=block_indptr.device,
            max_blocks_per_row=max_blocks_per_row,
            use_kv_valid_bits=valid_bits is not None,
            mask_type=case.mask_type,
            q_data_type=case.dtype,
        )
        policy = dict(wrapper._policy)
        assert policy["tile_size_q"] == case.expected_q_tile
        assert policy["tile_size_kv"] == case.expected_kv_tile
        assert policy["use_persistent_scheduler"] == (case.scheduler == "persistent")
        if case.expected_parallel_loads is not None:
            assert (
                policy["use_parallel_sparse_kv_loads"] is case.expected_parallel_loads
            )
        actual = wrapper.run(
            q,
            k,
            v,
            block_indptr,
            block_indices,
            kv_valid_bits=valid_bits,
            sm_scale=sm_scale,
        )
        one_shot_actual = block_sparse_module.block_sparse_attention(
            q,
            k,
            v,
            block_indptr,
            block_indices,
            case.q_block_size,
            case.kv_block_size,
            kv_valid_bits=valid_bits,
            mask_type=case.mask_type,
            sm_scale=sm_scale,
        )
        torch.cuda.synchronize()
    finally:
        block_sparse_config._resolve_block_sparse_launch_spec.cache_clear()

    for output in (actual, one_shot_actual):
        assert torch.isfinite(output).all()
    if case.include_empty_row:
        empty_row_begin = (
            math.ceil(case.seq_len_q / case.q_block_size) - 1
        ) * case.q_block_size
        empty_head_begin = (case.effective_num_kv_heads - 1) * case.heads_q_per_kv
        for output in (actual, one_shot_actual):
            empty_output = output[-1, empty_row_begin:, empty_head_begin:]
            assert torch.isfinite(empty_output).all()
            assert torch.count_nonzero(empty_output).item() == 0

    tolerance = 2e-2 if case.dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        one_shot_actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_one_block_sparse_plan_runs_distinct_layer_routes() -> None:
    """One capacity plan accepts distinct eager and captured route storage."""

    torch.manual_seed(20260814)
    case = _Case(
        "distinct_layer_routes",
        1,
        1,
        64,
        256,
        64,
        64,
        torch.float16,
        "dense",
        "none",
        "static",
    )
    layer_a_patterns: _Patterns = ((((0,),),),)
    layer_b_patterns: _Patterns = ((((1, 3),),),)
    layer_a_indptr, layer_a_indices = _make_bsr(layer_a_patterns)
    layer_b_indptr, layer_b_indices = _make_bsr(layer_b_patterns)
    assert layer_a_indices.numel() != layer_b_indices.numel()

    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, 256, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    all_tokens = (frozenset(range(case.seq_len_kv)),)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        layer_a_indptr,
        layer_a_indices,
        seq_len_q=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        q_block_size=case.q_block_size,
        kv_block_size=case.kv_block_size,
        max_blocks_per_row=2,
    )

    actual_a = wrapper.run(q, k, v, layer_a_indptr, layer_a_indices, sm_scale=sm_scale)
    actual_b = wrapper.run(q, k, v, layer_b_indptr, layer_b_indices, sm_scale=sm_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_a,
        _reference(case, q, k, v, layer_a_patterns, all_tokens, sm_scale),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        actual_b,
        _reference(case, q, k, v, layer_b_patterns, all_tokens, sm_scale),
        rtol=1e-2,
        atol=1e-2,
    )

    captured_a_indptr = torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32)
    captured_a_indices = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    captured_b_indptr = torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32)
    captured_b_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    captured_a_out = torch.empty_like(q)
    captured_b_out = torch.empty_like(q)
    wrapper.run(
        q,
        k,
        v,
        captured_a_indptr,
        captured_a_indices,
        sm_scale=sm_scale,
        out=captured_a_out,
    )
    wrapper.run(
        q,
        k,
        v,
        captured_b_indptr,
        captured_b_indices,
        sm_scale=sm_scale,
        out=captured_b_out,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q,
            k,
            v,
            captured_a_indptr,
            captured_a_indices,
            sm_scale=sm_scale,
            out=captured_a_out,
        )
        wrapper.run(
            q,
            k,
            v,
            captured_b_indptr,
            captured_b_indices,
            sm_scale=sm_scale,
            out=captured_b_out,
        )

    replay_a_patterns: _Patterns = ((((1, 2),),),)
    replay_b_patterns: _Patterns = ((((3,),),),)
    captured_a_indptr.copy_(torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32))
    captured_a_indices.copy_(torch.tensor([1, 2], device="cuda", dtype=torch.int32))
    captured_b_indptr.copy_(torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32))
    captured_b_indices.copy_(torch.tensor([3, -1], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured_a_out,
        _reference(case, q, k, v, replay_a_patterns, all_tokens, sm_scale),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        captured_b_out,
        _reference(case, q, k, v, replay_b_patterns, all_tokens, sm_scale),
        rtol=1e-2,
        atol=1e-2,
    )


def _make_lifecycle_problem() -> tuple[
    block_sparse_module.BlockSparseTSWrapper,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(20260721)
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_indptr = torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32)
    block_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(wrapper, block_indptr, block_indices)
    return wrapper, q, k, v, block_indptr, block_indices


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_failed_replan_preserves_runnable_plan() -> None:
    wrapper, q, k, v, block_indptr, block_indices = _make_lifecycle_problem()
    old_state = wrapper._published_state()
    expected = wrapper.run(q, k, v, block_indptr, block_indices)

    with pytest.raises(ValueError, match="non-negative"):
        _plan(
            wrapper,
            block_indptr,
            block_indices,
            max_blocks_per_row=-1,
        )

    assert wrapper._published_state() is old_state
    actual = wrapper.run(q, k, v, block_indptr, block_indices)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_run_uses_callers_current_stream() -> None:
    wrapper, q, k, v, block_indptr, block_indices = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v, block_indptr, block_indices)
    default_stream = torch.cuda.current_stream(q.device)
    worker = torch.cuda.Stream(device=q.device)
    worker.wait_stream(default_stream)
    out = torch.empty_like(q)

    with torch.cuda.stream(worker):
        result = wrapper.run(q, k, v, block_indptr, block_indices, out=out)

    assert result is out
    worker.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_cuda_graph_keeps_captured_plan_after_replan() -> None:
    wrapper, q, k, v, old_indptr, old_indices = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v, old_indptr, old_indices).clone()
    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, old_indptr, old_indices, out=graph_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, old_indptr, old_indices, out=graph_out)
    captured_state_id = id(wrapper._plan_state)
    assert captured_state_id in wrapper._captured_plan_states

    _plan(
        wrapper,
        torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
    )
    assert id(wrapper._plan_state) != captured_state_id

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_runtime_routes_cuda_graph_replays_routes_and_token_mask() -> None:
    """A captured run observes in-place metadata updates without replanning."""

    torch.manual_seed(20260722)
    case = _Case(
        "runtime_routes_cuda_graph",
        1,
        1,
        64,
        448,
        64,
        192,
        torch.float16,
        "dense",
        "holey",
        "static",
    )
    initial_patterns: _Patterns = ((((1, 2),),),)
    replay_patterns: _Patterns = ((((0, 1),),),)
    initial_valid = frozenset(range(case.seq_len_kv))
    replay_valid = frozenset(
        token_idx for token_idx in range(case.seq_len_kv) if token_idx % 7 not in (0, 3)
    )

    block_indptr, block_indices = _make_bsr(initial_patterns)
    valid_bits = _pack_token_mask(case.seq_len_kv, (initial_valid,))
    replay_valid_bits = _pack_token_mask(case.seq_len_kv, (replay_valid,))
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, 448, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    initial_expected = _reference(
        case,
        q,
        k,
        v,
        initial_patterns,
        (initial_valid,),
        sm_scale,
    )
    replay_expected = _reference(
        case,
        q,
        k,
        v,
        replay_patterns,
        (replay_valid,),
        sm_scale,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        seq_len_q=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        q_block_size=case.q_block_size,
        kv_block_size=case.kv_block_size,
        kv_valid_bits=valid_bits,
    )

    graph_out = torch.empty_like(q)
    wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
        out=graph_out,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = wrapper.run(
            q,
            k,
            v,
            block_indptr,
            block_indices,
            kv_valid_bits=valid_bits,
            sm_scale=sm_scale,
            out=graph_out,
        )
    assert captured_result is graph_out
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)
    state = wrapper._published_state()
    assert state.route_workspace[0].item() == 1

    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    valid_bits.copy_(replay_valid_bits)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(graph_out, replay_expected, rtol=1e-2, atol=1e-2)
    assert state.route_workspace[0].item() == 2

    block_indices.copy_(torch.tensor([1, 2], device="cuda", dtype=torch.int32))
    valid_bits.copy_(_pack_token_mask(case.seq_len_kv, (initial_valid,)))
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)
    assert state.route_workspace[0].item() == 1


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_runtime_routes_repartition_rows_with_declared_capacity() -> None:
    """One plan accepts changing indptr rows inside a declared BSR envelope."""

    torch.manual_seed(20260801)
    case = _Case(
        "dynamic_indptr_capacity",
        1,
        1,
        128,
        512,
        64,
        128,
        torch.float16,
        "dense",
        "none",
        "static",
    )
    initial_patterns: _Patterns = ((((0,), (3,)),),)
    replay_patterns: _Patterns = ((((0, 1, 2), (1, 2, 3)),),)
    all_tokens = (frozenset(range(case.seq_len_kv)),)
    block_indptr, initial_indices = _make_bsr(initial_patterns)
    replay_indptr, replay_indices = _make_bsr(replay_patterns)
    # Reserve the replay's maximum index extent up front. Inspection reads only
    # the two entries referenced by the initial indptr and ignores spare slots.
    block_indices = torch.full_like(replay_indices, -1)
    block_indices[: initial_indices.numel()].copy_(initial_indices)

    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
        device="cuda",
        dtype=case.dtype,
    )
    k = torch.randn(
        (case.batch_size, case.seq_len_kv, case.num_heads, _HEAD_DIM),
        device="cuda",
        dtype=case.dtype,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan(
        wrapper,
        block_indptr,
        block_indices,
        seq_len_q=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        q_block_size=case.q_block_size,
        kv_block_size=case.kv_block_size,
        max_blocks_per_row=3,
    )

    state = wrapper._published_state()
    assert state.row_route_offsets.tolist() == [0, 2, 4]
    assert state.row_route_offsets[-1].item() == 4
    assert dict(state.policy)["max_row_route_capacity"] == 2

    initial = wrapper.run(q, k, v, block_indptr, block_indices, sm_scale=sm_scale)
    block_indptr.copy_(replay_indptr)
    block_indices.copy_(replay_indices)
    replay = wrapper.run(q, k, v, block_indptr, block_indices, sm_scale=sm_scale)
    torch.cuda.synchronize()

    initial_expected = _reference(
        case,
        q,
        k,
        v,
        initial_patterns,
        all_tokens,
        sm_scale,
    )
    replay_expected = _reference(
        case,
        q,
        k,
        v,
        replay_patterns,
        all_tokens,
        sm_scale,
    )
    torch.testing.assert_close(initial, initial_expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(replay, replay_expected, rtol=1e-2, atol=1e-2)

    # Exceeding the declared envelope is still a caller error, but prepare
    # fails closed instead of overwriting the following row's route slice.
    block_indices[:4].copy_(
        torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
    )
    block_indptr.copy_(torch.tensor([[[0, 4, 4]]], device="cuda", dtype=torch.int32))
    overflow = wrapper.run(q, k, v, block_indptr, block_indices, sm_scale=sm_scale)
    torch.cuda.synchronize()
    assert state.route_workspace[0].item() == -4
    assert torch.count_nonzero(overflow).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_paged_one_shot_q64_kv256_gqa_matches_reference() -> None:
    torch.manual_seed(20260818)
    case = _Case(
        "paged_one_shot_q64_kv256_gqa",
        1,
        8,
        64,
        128,
        64,
        64,
        torch.float16,
        "dense",
        "none",
        "static",
        num_kv_heads=2,
        expected_q_tile=64,
        expected_kv_tile=256,
    )
    page_size = 64
    live_seq_len_kv = 96
    paged_kv_indptr = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    block_indptr, block_indices = _make_bsr(
        (
            (
                ((0,),),
                ((1,),),
            ),
        )
    )
    q = (
        torch.randn(
            (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    k_pages = (
        torch.randn(
            (3, case.effective_num_kv_heads, page_size, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    v_pages = torch.randn_like(k_pages)
    paged_kv_cache = torch.stack((k_pages, v_pages), dim=1)
    logical_k = torch.cat(
        [
            paged_kv_cache[page_id, 0].transpose(0, 1)
            for page_id in paged_kv_indices.tolist()
        ],
        dim=0,
    ).unsqueeze(0)
    logical_v = torch.cat(
        [
            paged_kv_cache[page_id, 1].transpose(0, 1)
            for page_id in paged_kv_indices.tolist()
        ],
        dim=0,
    ).unsqueeze(0)
    sm_scale = _HEAD_DIM**-0.5
    expected = _reference(
        case,
        q,
        logical_k,
        logical_v,
        (
            (
                ((0,),),
                ((1,),),
            ),
        ),
        (frozenset(range(live_seq_len_kv)),),
        sm_scale,
    )

    actual = prims_ts.block_sparse_attention_with_paged_kv_cache(
        q,
        paged_kv_cache,
        paged_kv_indptr,
        paged_kv_indices,
        block_indptr,
        block_indices,
        case.q_block_size,
        case.kv_block_size,
        max_seq_len_kv=case.seq_len_kv,
        seq_lens_kv=torch.tensor([live_seq_len_kv], dtype=torch.int32, device=q.device),
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("q_block_size", "num_qo_heads", "expected_q_tile"),
    (
        pytest.param(1, 16, 16, id="q1-gqa16"),
        pytest.param(2, 4, 8, id="q2-gqa4"),
        pytest.param(4, 2, 8, id="q4-gqa2"),
    ),
)
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_paged_gqa_small_q_blocks_match_reference(
    q_block_size: int,
    num_qo_heads: int,
    expected_q_tile: int,
) -> None:
    """Small semantic Q blocks keep independent routes after page remapping."""

    torch.manual_seed(20260819 + q_block_size)
    case = _Case(
        f"paged_gqa_small_q_block_{q_block_size}",
        1,
        num_qo_heads,
        8,
        1024,
        q_block_size,
        128,
        torch.bfloat16,
        "dense",
        "none",
        "static",
        num_kv_heads=1,
        expected_q_tile=expected_q_tile,
        expected_kv_tile=128,
    )
    page_size = 128
    route_rows = (
        (0, 2, 7),
        (1, 3, 6),
        (2, 4, 5),
        (0, 5, 6),
        (1, 4, 7),
        (0, 3, 4),
        (2, 3, 7),
        (1, 5, 6),
    )
    num_q_rows = math.ceil(case.seq_len_q / case.q_block_size)
    patterns: _Patterns = ((tuple(route_rows[:num_q_rows]),),)
    assert all(route == tuple(sorted(route)) for route in patterns[0][0])
    block_indptr, block_indices = _make_bsr(patterns)

    paged_kv_indptr = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.tensor(
        [8, 1, 6, 3, 9, 0, 7, 2],
        device="cuda",
        dtype=torch.int32,
    )
    q = (
        torch.randn(
            (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    k_pages = (
        torch.randn(
            (10, case.effective_num_kv_heads, page_size, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    v_pages = torch.randn_like(k_pages)
    paged_kv_cache = (k_pages, v_pages)
    logical_k = torch.cat(
        [k_pages[page_id].transpose(0, 1) for page_id in paged_kv_indices.tolist()],
        dim=0,
    ).unsqueeze(0)
    logical_v = torch.cat(
        [v_pages[page_id].transpose(0, 1) for page_id in paged_kv_indices.tolist()],
        dim=0,
    ).unsqueeze(0)
    sm_scale = _HEAD_DIM**-0.5
    expected = _reference(
        case,
        q,
        logical_k,
        logical_v,
        patterns,
        (frozenset(range(case.seq_len_kv)),),
        sm_scale,
    )

    wrapper = prims_ts.BlockSparsePagedTSWrapper()
    seq_lens_kv = torch.full(
        (case.batch_size,),
        case.seq_len_kv,
        device="cuda",
        dtype=torch.int32,
    )
    wrapper.plan(
        case.batch_size,
        case.seq_len_q,
        case.seq_len_kv,
        case.num_heads,
        case.effective_num_kv_heads,
        _HEAD_DIM,
        case.q_block_size,
        case.kv_block_size,
        page_size,
        device=q.device,
        max_blocks_per_row=3,
        use_kv_valid_bits=False,
        mask_type="dense",
        q_data_type=case.dtype,
    )
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == case.expected_q_tile
    assert policy["tile_size_kv"] == case.expected_kv_tile
    assert policy["use_persistent_scheduler"] is False

    actual = wrapper.run(
        q,
        paged_kv_cache,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens_kv,
        block_indptr,
        block_indices,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    graph_out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            q,
            paged_kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            block_indptr,
            block_indices,
            sm_scale=sm_scale,
            out=graph_out,
        )
    assert captured is graph_out
    torch.cuda.synchronize()
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, rtol=2e-2, atol=2e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    (
        "kv_block_size",
        "page_size",
        "expected_q_tile",
        "expected_kv_tile",
        "dtype",
    ),
    (
        (16, 64, 32, 128, torch.float16),
        (64, 128, 64, 256, torch.bfloat16),
    ),
    ids=("q32-kv128-page64-fp16", "q64-kv256-page128-bf16"),
)
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_paged_gqa_graph_reloads_routes_and_pages(
    kv_block_size: int,
    page_size: int,
    expected_q_tile: int,
    expected_kv_tile: int,
    dtype: torch.dtype,
) -> None:
    """Public KV128/KV256 plans reload routes/pages and fail bad rows closed."""

    torch.manual_seed(20260814)
    case = _Case(
        f"paged_q{expected_q_tile}_kv{expected_kv_tile}_gqa_graph",
        1,
        16,
        8,
        2 * page_size,
        8,
        kv_block_size,
        dtype,
        "dense",
        "none",
        "static",
        num_kv_heads=2,
        expected_q_tile=expected_q_tile,
        expected_kv_tile=expected_kv_tile,
    )
    num_physical_pages = 4
    max_nnz = 3
    sm_scale = _HEAD_DIM**-0.5
    second_page_block = page_size // kv_block_size
    patterns_a = (
        (
            ((0,),),
            ((second_page_block,),),
        ),
    )
    patterns_b = (
        (
            ((0, second_page_block),),
            ((0,),),
        ),
    )
    patterns_invalid = (
        (
            ((second_page_block,),),
            ((0,),),
        ),
    )
    patterns_fail_closed = (
        (
            ((),),
            ((0,),),
        ),
    )

    def padded_bsr(patterns: object) -> tuple[torch.Tensor, torch.Tensor]:
        indptr, indices = _make_bsr(patterns)
        padded = torch.full((max_nnz,), -1, device="cuda", dtype=torch.int32)
        padded[: indices.numel()].copy_(indices)
        return indptr, padded

    indptr_a, indices_a = padded_bsr(patterns_a)
    indptr_b, indices_b = padded_bsr(patterns_b)
    indptr_invalid, indices_invalid = padded_bsr(patterns_invalid)
    page_ids_a = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    page_ids_b = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    page_ids_invalid = torch.tensor([2, 4], device="cuda", dtype=torch.int32)

    q = torch.randn((1, 8, 16, _HEAD_DIM), device="cuda", dtype=case.dtype) * 0.25
    k_pages = (
        torch.randn(
            (num_physical_pages, 2, page_size, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    markers = (
        torch.arange(
            1,
            num_physical_pages + 1,
            device="cuda",
            dtype=torch.float32,
        )[:, None]
        * 0.25
        + torch.arange(2, device="cuda", dtype=torch.float32)[None, :] * 0.03125
    ).to(case.dtype)
    v_pages = (
        markers[:, :, None, None]
        .expand(num_physical_pages, 2, page_size, _HEAD_DIM)
        .contiguous()
    )
    paged_kv_cache = torch.stack((k_pages, v_pages), dim=1)

    def logical_kv(
        physical_page_ids: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_k = torch.cat(
            [
                paged_kv_cache[page_id, 0].transpose(0, 1)
                for page_id in physical_page_ids
            ],
            dim=0,
        ).unsqueeze(0)
        logical_v = torch.cat(
            [
                paged_kv_cache[page_id, 1].transpose(0, 1)
                for page_id in physical_page_ids
            ],
            dim=0,
        ).unsqueeze(0)
        return logical_k, logical_v

    all_tokens = (frozenset(range(case.seq_len_kv)),)
    k_a, v_a = logical_kv((0, 1))
    k_b, v_b = logical_kv((2, 3))
    k_fail_closed, v_fail_closed = logical_kv((2, 0))
    expected_a = _reference(case, q, k_a, v_a, patterns_a, all_tokens, sm_scale)
    expected_b = _reference(case, q, k_b, v_b, patterns_b, all_tokens, sm_scale)
    expected_fail_closed = _reference(
        case,
        q,
        k_fail_closed,
        v_fail_closed,
        patterns_fail_closed,
        all_tokens,
        sm_scale,
    )
    assert not torch.allclose(
        expected_a.float(), expected_b.float(), rtol=1e-3, atol=1e-3
    )

    block_indptr = indptr_a.clone()
    block_indices = indices_a.clone()
    paged_kv_indices = page_ids_a.clone()
    paged_kv_indptr = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.full(
        (case.batch_size,),
        case.seq_len_kv,
        device="cuda",
        dtype=torch.int32,
    )
    wrapper = prims_ts.BlockSparsePagedTSWrapper()
    wrapper.plan(
        case.batch_size,
        case.seq_len_q,
        case.seq_len_kv,
        case.num_heads,
        case.effective_num_kv_heads,
        _HEAD_DIM,
        case.q_block_size,
        case.kv_block_size,
        page_size,
        device=q.device,
        max_blocks_per_row=2,
        use_kv_valid_bits=False,
        mask_type="dense",
        q_data_type=case.dtype,
    )
    planned_state = wrapper._published_state()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == expected_q_tile
    assert policy["tile_size_kv"] == expected_kv_tile
    assert policy["page_size"] == page_size

    eager = wrapper.run(
        q,
        paged_kv_cache,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens_kv,
        block_indptr,
        block_indices,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, expected_a, rtol=1e-2, atol=1e-2)

    graph_out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            q,
            paged_kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            block_indptr,
            block_indices,
            sm_scale=sm_scale,
            out=graph_out,
        )
    assert captured is graph_out
    torch.cuda.synchronize()

    def replay() -> None:
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

    replay()
    torch.testing.assert_close(graph_out, expected_a, rtol=1e-2, atol=1e-2)

    # The previous replay is complete before captured storage is updated.
    # Copies and the next replay share the current stream, preserving ordering.
    block_indptr.copy_(indptr_b)
    block_indices.copy_(indices_b)
    paged_kv_indices.copy_(page_ids_b)
    replay()
    assert torch.isfinite(graph_out).all()
    torch.testing.assert_close(graph_out, expected_b, rtol=1e-2, atol=1e-2)

    block_indptr.copy_(indptr_invalid)
    block_indices.copy_(indices_invalid)
    paged_kv_indices.copy_(page_ids_invalid)
    replay()
    assert torch.isfinite(graph_out).all()
    assert torch.count_nonzero(graph_out[:, :, :8]).item() == 0
    assert torch.count_nonzero(graph_out[:, :, 8:]).item() > 0
    torch.testing.assert_close(
        graph_out,
        expected_fail_closed,
        rtol=1e-2,
        atol=1e-2,
    )
    assert wrapper._published_state() is planned_state


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_paged_varlen_gqa_q64_kv256_graph_reloads_live_pages_bits_and_sparse_routes() -> (
    None
):
    """One graph reloads every live paged input while its plan stays fixed."""

    torch.manual_seed(20260814)
    case = _Case(
        "paged_q64_kv256_gqa_varlen_graph",
        2,
        16,
        8,
        128,
        8,
        64,
        torch.float16,
        "causal",
        "full",
        "static",
        num_kv_heads=2,
        expected_q_tile=64,
        expected_kv_tile=256,
    )
    page_size = 64
    num_physical_pages = 6
    max_nnz = 6
    sm_scale = _HEAD_DIM**-0.5
    seq_lens_a = torch.tensor([48, 128], device="cuda", dtype=torch.int32)
    seq_lens_b = torch.tensor([96, 64], device="cuda", dtype=torch.int32)
    paged_kv_indptr_a = torch.tensor([0, 1, 4], device="cuda", dtype=torch.int32)
    paged_kv_indptr_b = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    patterns_a = (
        (
            ((0,),),
            ((0,),),
        ),
        (
            ((0, 1),),
            ((1,),),
        ),
    )
    patterns_b = (
        (
            ((0, 1),),
            ((1,),),
        ),
        (
            ((0,),),
            ((0,),),
        ),
    )

    def padded_bsr(patterns: object) -> tuple[torch.Tensor, torch.Tensor]:
        indptr, indices = _make_bsr(patterns)
        padded = torch.full((max_nnz,), -1, device="cuda", dtype=torch.int32)
        padded[: indices.numel()].copy_(indices)
        return indptr, padded

    indptr_a, indices_a = padded_bsr(patterns_a)
    indptr_b, indices_b = padded_bsr(patterns_b)

    q = (
        torch.randn(
            (case.batch_size, case.seq_len_q, case.num_heads, _HEAD_DIM),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    k_pages = (
        torch.randn(
            (
                num_physical_pages,
                case.effective_num_kv_heads,
                page_size,
                _HEAD_DIM,
            ),
            device="cuda",
            dtype=case.dtype,
        )
        * 0.25
    )
    markers = (
        torch.arange(
            1,
            num_physical_pages + 1,
            device="cuda",
            dtype=torch.float32,
        )[:, None]
        * 0.25
        + torch.arange(
            case.effective_num_kv_heads,
            device="cuda",
            dtype=torch.float32,
        )[None, :]
        * 0.03125
    ).to(case.dtype)
    token_offsets = (
        torch.arange(page_size, device="cuda", dtype=torch.float32) / page_size
    )
    v_pages = (
        (markers.float()[:, :, None, None] + token_offsets[None, None, :, None])
        .expand(
            num_physical_pages,
            case.effective_num_kv_heads,
            page_size,
            _HEAD_DIM,
        )
        .to(case.dtype)
        .contiguous()
    )
    paged_kv_cache = (k_pages, v_pages)

    def logical_kv(
        physical_page_ids_by_batch: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_k_batches: list[torch.Tensor] = []
        logical_v_batches: list[torch.Tensor] = []
        for batch_page_ids in physical_page_ids_by_batch:
            logical_k_batches.append(
                torch.cat(
                    [k_pages[page_id].transpose(0, 1) for page_id in batch_page_ids],
                    dim=0,
                )
            )
            logical_v_batches.append(
                torch.cat(
                    [v_pages[page_id].transpose(0, 1) for page_id in batch_page_ids],
                    dim=0,
                )
            )
        return torch.stack(logical_k_batches), torch.stack(logical_v_batches)

    valid_by_batch_a = (
        frozenset(range(48)),
        frozenset(range(case.seq_len_kv)),
    )
    valid_by_batch_b = (
        frozenset(range(96)),
        frozenset(token for token in range(64) if token % 5 != 2),
    )
    kv_valid_bits_a = _pack_token_mask(case.seq_len_kv, valid_by_batch_a)
    kv_valid_bits_b = _pack_token_mask(case.seq_len_kv, valid_by_batch_b)
    page_ids_a = torch.tensor([0, 2, 3, 5], device="cuda", dtype=torch.int32)
    page_ids_b = torch.tensor([1, 4, 2, 5], device="cuda", dtype=torch.int32)

    wrapper = prims_ts.BlockSparsePagedTSWrapper()
    wrapper.plan(
        case.batch_size,
        case.seq_len_q,
        case.seq_len_kv,
        case.num_heads,
        case.effective_num_kv_heads,
        _HEAD_DIM,
        case.q_block_size,
        case.kv_block_size,
        page_size,
        device=q.device,
        max_blocks_per_row=2,
        use_kv_valid_bits=True,
        mask_type=case.mask_type,
        q_data_type=case.dtype,
    )
    planned_state = wrapper._published_state()
    assert planned_state.page_size == page_size
    assert ("tile_size_q", case.expected_q_tile) in wrapper._policy
    assert ("tile_size_kv", case.expected_kv_tile) in wrapper._policy

    k_a, v_a = logical_kv(((0, 1), (2, 3)))
    k_b, v_b = logical_kv(((1, 4), (2, 0)))

    def varlen_reference(
        k: torch.Tensor,
        v: torch.Tensor,
        patterns: object,
        valid_by_batch: tuple[frozenset[int], ...],
        seq_lens_kv: tuple[int, int],
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for batch_idx, live_kv in enumerate(seq_lens_kv):
            live_case = replace(case, batch_size=1, seq_len_kv=live_kv)
            outputs.append(
                _reference(
                    live_case,
                    q[batch_idx : batch_idx + 1],
                    k[batch_idx : batch_idx + 1, :live_kv],
                    v[batch_idx : batch_idx + 1, :live_kv],
                    (patterns[batch_idx],),
                    (
                        frozenset(
                            token
                            for token in valid_by_batch[batch_idx]
                            if token < live_kv
                        ),
                    ),
                    sm_scale,
                )
            )
        return torch.cat(outputs)

    expected_a = varlen_reference(
        k_a,
        v_a,
        patterns_a,
        valid_by_batch_a,
        (48, 128),
    )
    expected_b = varlen_reference(
        k_b,
        v_b,
        patterns_b,
        valid_by_batch_b,
        (96, 64),
    )
    wrong_static_a0 = _reference(
        replace(case, batch_size=1),
        q[:1],
        k_a[:1],
        v_a[:1],
        (patterns_a[0],),
        (valid_by_batch_a[0],),
        sm_scale,
    )
    assert not torch.allclose(
        expected_a[:1].float(),
        wrong_static_a0.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    assert not torch.allclose(
        expected_a.float(), expected_b.float(), rtol=1e-3, atol=1e-3
    )

    block_indptr = indptr_a.clone()
    block_indices = indices_a.clone()
    paged_kv_indices = page_ids_a.clone()
    paged_kv_indptr = paged_kv_indptr_a.clone()
    seq_lens_kv = seq_lens_a.clone()
    kv_valid_bits = kv_valid_bits_a.clone()

    eager = wrapper.run(
        q,
        paged_kv_cache,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens_kv,
        block_indptr,
        block_indices,
        kv_valid_bits=kv_valid_bits,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, expected_a, rtol=1e-2, atol=1e-2)

    graph_out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            q,
            paged_kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            block_indptr,
            block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=sm_scale,
            out=graph_out,
        )
    assert captured is graph_out
    torch.cuda.synchronize()

    def replay() -> None:
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

    replay()
    torch.testing.assert_close(graph_out, expected_a, rtol=1e-2, atol=1e-2)

    block_indptr.copy_(indptr_b)
    block_indices.copy_(indices_b)
    paged_kv_indptr.copy_(paged_kv_indptr_b)
    paged_kv_indices.copy_(page_ids_b)
    seq_lens_kv.copy_(seq_lens_b)
    kv_valid_bits.copy_(kv_valid_bits_b)
    replay()
    assert torch.isfinite(graph_out).all()
    torch.testing.assert_close(graph_out, expected_b, rtol=1e-2, atol=1e-2)

    def assert_batch0_fails_closed() -> None:
        replay()
        assert torch.isfinite(graph_out).all()
        assert torch.count_nonzero(graph_out[0]).item() == 0
        torch.testing.assert_close(
            graph_out[1:],
            expected_b[1:],
            rtol=1e-2,
            atol=1e-2,
        )

    invalid_page_ids = page_ids_b.clone()
    invalid_page_ids[0] = num_physical_pages
    paged_kv_indices.copy_(invalid_page_ids)
    replay()
    assert torch.isfinite(graph_out).all()
    assert torch.count_nonzero(graph_out[0, :, :8]).item() == 0
    torch.testing.assert_close(
        graph_out[0, :, 8:],
        expected_b[0, :, 8:],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(graph_out[1:], expected_b[1:], rtol=1e-2, atol=1e-2)

    paged_kv_indices.copy_(page_ids_b)
    paged_kv_indptr.copy_(torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32))
    paged_kv_indices.copy_(torch.tensor([1, 2, 5, 4], device="cuda", dtype=torch.int32))
    assert_batch0_fails_closed()

    paged_kv_indptr.copy_(torch.tensor([1, 3, 4], device="cuda", dtype=torch.int32))
    paged_kv_indices.copy_(page_ids_b)
    replay()
    assert torch.count_nonzero(graph_out).item() == 0

    paged_kv_indptr.copy_(torch.tensor([0, 3, 2], device="cuda", dtype=torch.int32))
    replay()
    assert torch.isfinite(graph_out).all()
    torch.testing.assert_close(graph_out[:1], expected_b[:1], rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(graph_out[1]).item() == 0

    paged_kv_indptr.copy_(torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32))
    replay()
    assert torch.isfinite(graph_out).all()
    torch.testing.assert_close(graph_out[:1], expected_b[:1], rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(graph_out[1]).item() == 0

    paged_kv_indptr.copy_(paged_kv_indptr_b)
    paged_kv_indices.copy_(page_ids_b)
    seq_lens_kv.copy_(seq_lens_b)
    replay()
    torch.testing.assert_close(graph_out, expected_b, rtol=1e-2, atol=1e-2)
    assert wrapper._published_state() is planned_state
