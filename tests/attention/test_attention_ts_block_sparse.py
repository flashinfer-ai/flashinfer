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

from dataclasses import dataclass
import math

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

import flashinfer.attention.prims_ts as prims_ts
from flashinfer.attention.prims_ts import block_sparse as block_sparse_module


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


_CASES = (
    _Case(
        "q8_kv16_fp16_holey_empty_static",
        1,
        1,
        23,
        207,
        8,
        16,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="mixed",
        include_empty_row=True,
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


def _make_patterns(case: _Case) -> _Patterns:
    num_q_rows = math.ceil(case.seq_len_q / case.q_block_size)
    num_kv_blocks = math.ceil(case.seq_len_kv / case.kv_block_size)
    batches: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    for batch_idx in range(case.batch_size):
        heads: list[tuple[tuple[int, ...], ...]] = []
        for head_idx in range(case.num_heads):
            rows: list[tuple[int, ...]] = []
            for row_idx in range(num_q_rows):
                if (
                    case.include_empty_row
                    and batch_idx == case.batch_size - 1
                    and head_idx == case.num_heads - 1
                    and row_idx == num_q_rows - 1
                ):
                    rows.append(())
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
        for head_idx in range(case.num_heads):
            structural_rows: list[torch.Tensor] = []
            for row_idx, selected_blocks in enumerate(patterns[batch_idx][head_idx]):
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
                q[batch_idx, :, head_idx].float()
                @ k[batch_idx, :, head_idx].float().transpose(0, 1)
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
            head_outputs.append(probabilities @ v[batch_idx, :, head_idx].float())
        batch_outputs.append(torch.stack(head_outputs, dim=1))
    return torch.stack(batch_outputs).to(case.dtype)


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
    dynamic_metadata: bool = False,
) -> None:
    wrapper.plan(
        block_indptr,
        block_indices,
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_heads,
        num_heads,
        _HEAD_DIM,
        q_block_size,
        kv_block_size,
        kv_valid_bits=kv_valid_bits,
        dynamic_metadata=dynamic_metadata,
    )


def test_public_exports() -> None:
    assert prims_ts.BlockSparseTSWrapper is block_sparse_module.BlockSparseTSWrapper
    assert prims_ts.block_sparse_attention is block_sparse_module.block_sparse_attention


def test_public_api_rejects_invalid_usage() -> None:
    with pytest.raises(RuntimeError, match=r"plan\(\).*before run"):
        q = torch.empty((1, 1, 1, _HEAD_DIM), dtype=torch.float16)
        block_sparse_module.BlockSparseTSWrapper().run(q, q, q)

    k = torch.empty((1, 128, 1, _HEAD_DIM), dtype=torch.float16)
    metadata = torch.empty((1, 1, 2), dtype=torch.int32)
    indices = torch.empty((0,), dtype=torch.int32)
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


def test_runtime_output_must_not_alias_sparse_metadata() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        prepare_block_sparse_runtime,
    )

    shape = (1, 1, 1, _HEAD_DIM)
    q = torch.empty(shape, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    block_indices = torch.empty(_HEAD_DIM // 2, dtype=torch.int32)
    out = torch.empty(0, dtype=torch.float16).set_(
        block_indices.untyped_storage(),
        0,
        shape,
    )

    with pytest.raises(ValueError, match="out must not overlap block_indices storage"):
        prepare_block_sparse_runtime(
            q,
            k,
            v,
            block_indptr=torch.tensor([[[0, 0]]], dtype=torch.int32),
            block_indices=block_indices,
            runtime_kv_valid_bits=torch.empty((1, 1), dtype=torch.uint32),
            device=torch.device("cpu"),
            batch_size=1,
            seq_len_q=1,
            seq_len_kv=1,
            num_heads=1,
            head_dim=_HEAD_DIM,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            output_dtype=torch.float16,
            sm_scale=None,
            out=out,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param({"q_block_size": 96}, "8, 16, 32", id="block-size"),
        pytest.param({"num_qo_heads": 2}, "Hq == Hkv", id="mha-only"),
    ),
)
def test_plan_rejects_unsupported_profile(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "block_indptr": None,
        "block_indices": None,
        "batch_size": 1,
        "seq_len_q": 64,
        "seq_len_kv": 128,
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim": _HEAD_DIM,
        "q_block_size": 64,
        "kv_block_size": 64,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        block_sparse_module.BlockSparseTSWrapper().plan(**arguments)


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
def test_plan_rejects_noncanonical_bsr(
    indptr: list[list[list[int]]],
    indices: tuple[int, ...],
    message: str,
) -> None:
    block_indptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    block_indices = torch.tensor(indices, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match=rf"canonical BSR.*{message}"):
        _plan(
            block_sparse_module.BlockSparseTSWrapper(),
            block_indptr,
            block_indices,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_public_block_sparse_correctness(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    # Exercise the raw-BSR Q8 static guard even when auto selection asks for CLC.
    selected_scheduler = "persistent" if case.q_block_size == 8 else case.scheduler
    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: selected_scheduler,
    )
    block_sparse_module._resolve_raw_block_sparse_launch_spec.cache_clear()
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
        wrapper.plan(
            block_indptr,
            block_indices,
            case.batch_size,
            case.seq_len_q,
            case.seq_len_kv,
            case.num_heads,
            case.num_heads,
            _HEAD_DIM,
            case.q_block_size,
            case.kv_block_size,
            kv_valid_bits=valid_bits,
            mask_type=case.mask_type,
            q_data_type=case.dtype,
        )
        policy = dict(wrapper._policy)
        assert policy["use_persistent_scheduler"] == (
            case.scheduler == "persistent"
        )
        assert policy["use_kv_valid_bits"] == (case.token_mask == "holey")
        actual = wrapper.run(q, k, v, sm_scale=sm_scale)
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
        block_sparse_module._resolve_raw_block_sparse_launch_spec.cache_clear()

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
    wrapper, q, k, v, block_indptr, _ = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v)
    invalid_indices = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="canonical BSR"):
        _plan(wrapper, block_indptr, invalid_indices)

    actual = wrapper.run(q, k, v)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_run_uses_callers_current_stream() -> None:
    wrapper, q, k, v, _, _ = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v)
    default_stream = torch.cuda.current_stream(q.device)
    worker = torch.cuda.Stream(device=q.device)
    worker.wait_stream(default_stream)
    out = torch.empty_like(q)

    with torch.cuda.stream(worker):
        result = wrapper.run(q, k, v, out=out)

    assert result is out
    worker.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_cuda_graph_keeps_captured_plan_after_replan() -> None:
    wrapper, q, k, v, old_indptr, old_indices = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v).clone()
    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, out=graph_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=graph_out)
    captured_state_id = id(wrapper._plan_state)
    assert captured_state_id in wrapper._captured_plan_states
    del old_indptr, old_indices

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
def test_dynamic_metadata_cuda_graph_replays_routes_and_token_mask() -> None:
    """A captured run observes in-place metadata updates without replanning."""

    torch.manual_seed(20260722)
    case = _Case(
        "dynamic_metadata_cuda_graph",
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
        dynamic_metadata=True,
    )

    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    assert captured_result is graph_out
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)

    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    valid_bits.copy_(replay_valid_bits)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(graph_out, replay_expected, rtol=1e-2, atol=1e-2)
