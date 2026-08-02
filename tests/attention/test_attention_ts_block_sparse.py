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
import inspect
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
from flashinfer.attention.prims_ts._block_sparse.prepared import (
    _PreparedBlockSparseLayout,
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
    max_blocks_per_row: int | None = None,
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
        max_blocks_per_row=max_blocks_per_row,
    )


def test_public_exports() -> None:
    assert prims_ts.BlockSparseTSWrapper is block_sparse_module.BlockSparseTSWrapper
    assert prims_ts.block_sparse_attention is block_sparse_module.block_sparse_attention


def test_block_sparse_keeps_rescale_threshold_preserves_anchor_invariant() -> None:
    """Deferred max updates must keep numerator and denominator in one frame."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources import (
        tmem_s,
    )

    threshold = getattr(tmem_s, "_BLOCK_SPARSE_RESCALE_THRESHOLD_LOG2", None)
    assert threshold == 8.0

    anchor = float("-inf")
    denominator = 0.0
    numerator = 0.0
    anchors: list[float] = []
    scores = (0.0, 8.0 * math.log(2.0), 9.0 * math.log(2.0))
    values = (1.0, -2.0, 4.0)
    for score, value in zip(scores, values, strict=True):
        candidate = max(anchor, score)
        if (
            math.isfinite(anchor)
            and (candidate - anchor) * math.log2(math.e) <= threshold
        ):
            new_anchor = anchor
        else:
            new_anchor = candidate
        old_scale = 0.0 if not math.isfinite(anchor) else math.exp(anchor - new_anchor)
        probability = math.exp(score - new_anchor)
        denominator = denominator * old_scale + probability
        numerator = numerator * old_scale + probability * value
        anchor = new_anchor
        anchors.append(anchor)

    assert anchors == [scores[0], scores[0], scores[2]]
    expected = torch.softmax(
        torch.tensor(scores, dtype=torch.float64),
        dim=0,
    ) @ torch.tensor(values, dtype=torch.float64)
    assert numerator / denominator == pytest.approx(expected.item(), abs=1e-12)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_block_sparse_keeps_rescale_threshold_gpu_edges() -> None:
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
    route_size = 2 * block_size
    seq_len_q = 64
    seq_len_kv = len(route_scores) * route_size
    q = torch.zeros(
        (1, seq_len_q, 1, _HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q[..., 0] = 1.0
    k = torch.zeros(
        (1, seq_len_kv, 1, _HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
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
        block_indptr,
        block_indices,
        1,
        seq_len_q,
        seq_len_kv,
        1,
        1,
        _HEAD_DIM,
        block_size,
        block_size,
        kv_valid_bits=kv_valid_bits,
        q_data_type=torch.bfloat16,
        dynamic_metadata=True,
    )
    state = wrapper._published_state()
    policy = dict(state.policy)
    assert policy["tile_size_q"] == block_size
    # The alternating-route construction below depends on the prepared KV128
    # route geometry, which is intentionally not part of the diagnostic policy.
    assert state.route_layout.kv_route_size == route_size
    assert policy["use_kv_valid_bits"] is True

    actual = wrapper.run(q, k, v, sm_scale=1.0)
    torch.cuda.synchronize()
    token_scores = k[0, :, 0, 0].float()
    expected_row = torch.softmax(token_scores, dim=0) @ v[0, :, 0].float()
    expected = expected_row.reshape(1, 1, 1, _HEAD_DIM).expand_as(actual)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    kv_valid_bits.zero_()
    all_masked = wrapper.run(q, k, v, sm_scale=1.0)
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
        (128, 64, True, (64, 2, 4, 2, 3, 4, 8)),
        (128, 8, True, (8, 16, 4, 16, 17, 18, 24)),
        (128, 16, True, (16, 8, 4, 8, 9, 10, 16)),
        (128, 32, True, (32, 4, 4, 4, 5, 6, 12)),
        (256, 64, True, (64, 4, 8, 4, 5, 6, 16)),
        (256, 128, False, (64, 4, 8, 4, 5, None, 8)),
        (256, 256, True, (64, 4, 8, 4, 5, 6, 16)),
    ),
    ids=(
        "kv128-block64-mask",
        "kv128-block8-mask",
        "kv128-block16-mask",
        "kv128-block32-mask",
        "kv256-block64-mask",
        "kv256-block128-no-mask",
        "kv256-block256-mask",
    ),
)
def test_prepared_block_sparse_layout_geometry(
    kv_route_size: int,
    kv_block_size: int,
    has_token_bits: bool,
    expected_geometry: tuple[int, int, int, int, int, int | None, int],
) -> None:
    layout = _PreparedBlockSparseLayout.create(
        kv_route_size=kv_route_size,
        kv_block_size=kv_block_size,
        has_token_bits=has_token_bits,
        route_metadata_capacity=3,
        num_rows=2,
    )

    assert (
        layout.atom_size,
        layout.origins_per_route,
        layout.token_words_per_route,
        layout.atom_valid_mask_word_offset,
        layout.route_flags_word_offset,
        layout.token_words_word_offset,
        layout.route_metadata_stride_words,
    ) == expected_geometry
    assert layout.route_metadata_base_word_offset == 4
    assert layout.route_metadata_capacity == 3
    assert layout.workspace_size_words == 4 + 3 * layout.route_metadata_stride_words


@pytest.mark.parametrize(
    ("kv_block_size", "expected_words"),
    ((8, 16), (16, 8), (32, 4), (64, 4), (128, 4), (256, 4)),
)
def test_kv_retained_route_storage_is_derived_from_prepared_layout(
    kv_block_size: int,
    expected_words: int,
) -> None:
    """K/V retention keeps only origins and coarse fragment validity."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_block_sparse_metadata import (
        _kv_retained_route_words,
    )

    layout = _PreparedBlockSparseLayout.create(
        kv_route_size=128,
        kv_block_size=kv_block_size,
        has_token_bits=True,
        route_metadata_capacity=0,
        num_rows=1,
    )

    assert _kv_retained_route_words(layout) == expected_words


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    (
        ({"kv_route_size": 96}, ValueError, "atom_size"),
        ({"kv_route_size": 24, "kv_block_size": 8}, ValueError, "32"),
        ({"kv_route_size": 512, "kv_block_size": 8}, ValueError, "single 32-bit mask"),
        (
            {"kv_route_size": 2048, "kv_block_size": 64},
            ValueError,
            "token_words_per_route",
        ),
        (
            {"route_metadata_capacity": (1 << 31) - 1},
            OverflowError,
            "workspace_size_words",
        ),
        ({"num_rows": (1 << 31) - 1}, OverflowError, "row_route_offsets_length"),
        ({"route_metadata_capacity": True}, TypeError, "route_metadata_capacity"),
        ({"route_metadata_capacity": -1}, ValueError, "route_metadata_capacity"),
        ({"num_rows": 0}, ValueError, "num_rows"),
    ),
    ids=(
        "tile-not-divisible-by-atom",
        "tile-not-divisible-by-word",
        "too-many-origins",
        "too-many-token-words",
        "route-metadata-address-overflow",
        "row-offset-length-overflow",
        "bool-route-metadata-capacity",
        "negative-route-metadata-capacity",
        "zero-num-rows",
    ),
)
def test_prepared_block_sparse_layout_rejects_invalid_geometry(
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
        _PreparedBlockSparseLayout.create(**arguments)


def test_prepared_block_sparse_layout_allows_empty_route_metadata() -> None:
    layout = _PreparedBlockSparseLayout.create(
        kv_route_size=128,
        kv_block_size=64,
        has_token_bits=False,
        route_metadata_capacity=0,
        num_rows=3,
    )

    assert layout.route_metadata_base_word_offset == 4
    assert layout.route_metadata_capacity == 0
    assert (
        layout.workspace_size_words == layout.route_metadata_base_word_offset
    )


def _signed_i32_bits(value: int) -> int:
    """Return the signed Int32 spelling of one stored metadata word."""

    value &= 0xFFFFFFFF
    return value if value < (1 << 31) else value - (1 << 32)


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


@pytest.mark.parametrize(
    ("value", "dynamic_metadata", "error_type", "message"),
    (
        (1, False, ValueError, "requires dynamic_metadata"),
        ("1", False, TypeError, "Python integer"),
        (True, True, TypeError, "Python integer"),
        (-1, True, ValueError, "non-negative"),
        (3, True, ValueError, "number of semantic KV blocks"),
    ),
)
def test_max_blocks_per_row_validation(
    value: object,
    dynamic_metadata: bool,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        block_sparse_module._validate_max_blocks_per_row(
            value,
            dynamic_metadata=dynamic_metadata,
            seq_len_kv=128,
            kv_block_size=64,
        )


def test_runtime_output_must_not_alias_sparse_metadata() -> None:
    from flashinfer.attention.prims_ts._block_sparse.runtime import (
        validate_block_sparse_run,
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
        validate_block_sparse_run(
            q,
            k,
            v,
            block_indptr=torch.tensor([[[0, 0]]], dtype=torch.int32),
            block_indices=block_indices,
            kv_valid_bits=torch.empty((1, 1), dtype=torch.uint32),
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
@torch.no_grad()
def test_block_capacity_is_not_weakened_by_route_packing() -> None:
    """The semantic block bound applies both at plan time and on replay."""

    block_indptr = torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32)
    block_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="smaller than an initial BSR row"):
        _plan(
            block_sparse_module.BlockSparseTSWrapper(),
            block_indptr,
            block_indices,
            dynamic_metadata=True,
            max_blocks_per_row=1,
        )

    # B64 packs two semantic blocks into one prepared KV128 route. Reserve the
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
        dynamic_metadata=True,
        max_blocks_per_row=1,
    )
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 1, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    block_indptr.copy_(torch.tensor([[[0, 2]]], device="cuda", dtype=torch.int32))

    overflow = wrapper.run(q, k, v)
    torch.cuda.synchronize()

    state = wrapper._published_state()
    assert state.route_workspace[0].item() < 0
    assert torch.count_nonzero(overflow).item() == 0

    # Contract-invalid live IDs must still fail safely before address
    # arithmetic; in particular, the token-mask path must not read word -1.
    block_indices[0] = -1
    block_indptr.copy_(torch.tensor([[[0, 1]]], device="cuda", dtype=torch.int32))
    invalid_index = wrapper.run(q, k, v)
    torch.cuda.synchronize()
    assert torch.count_nonzero(invalid_index).item() == 0


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_plan_owns_compact_route_storage_for_skewed_rows() -> None:
    """Plan capacities depend only on each immutable indptr row length."""

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
    assert state.route_layout.num_rows == 6
    assert state.route_layout.route_metadata_capacity == 12
    assert state.row_route_offsets.tolist() == [0, 0, 2, 6, 10, 10, 12]
    assert (
        state.route_workspace.numel()
        == state.route_layout.workspace_size_words
    )
    assert dict(state.policy)["max_row_route_capacity"] == 4


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_public_block_sparse_correctness(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    # Exercise the block-sparse Q8 static guard when auto selection asks for CLC.
    selected_scheduler = "persistent" if case.q_block_size == 8 else case.scheduler
    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: selected_scheduler,
    )
    block_sparse_module._resolve_block_sparse_launch_spec.cache_clear()
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
        assert policy["use_persistent_scheduler"] == (case.scheduler == "persistent")
        assert policy["use_kv_valid_bits"] == (valid_bits is not None)
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
        block_sparse_module._resolve_block_sparse_launch_spec.cache_clear()

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
    old_state = wrapper._published_state()
    expected = wrapper.run(q, k, v)
    invalid_indices = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="canonical BSR"):
        _plan(wrapper, block_indptr, invalid_indices)

    assert wrapper._published_state() is old_state
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
    state = wrapper._published_state()
    layout = state.route_layout
    assert state.route_workspace[0].item() == 2
    first_route_metadata_word_index = layout.route_metadata_base_word_offset
    assert state.route_workspace[
        first_route_metadata_word_index : first_route_metadata_word_index + 2
    ].tolist() == [192, 256]

    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    valid_bits.copy_(replay_valid_bits)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(graph_out, replay_expected, rtol=1e-2, atol=1e-2)
    assert state.route_workspace[0].item() == 3
    assert state.route_workspace[
        first_route_metadata_word_index : first_route_metadata_word_index + 2
    ].tolist() == [0, 64]
    assert layout.token_words_word_offset is not None
    assert state.route_workspace[
        first_route_metadata_word_index + layout.token_words_word_offset
    ].item() == _signed_i32_bits(int(replay_valid_bits[0, 0].item()))

    stale_route_metadata_word_index = (
        layout.route_metadata_base_word_offset
        + 2 * layout.route_metadata_stride_words
    )
    stale_route_metadata = state.route_workspace[
        stale_route_metadata_word_index : (
            stale_route_metadata_word_index + layout.route_metadata_stride_words
        )
    ].clone()
    block_indices.copy_(torch.tensor([1, 2], device="cuda", dtype=torch.int32))
    valid_bits.copy_(_pack_token_mask(case.seq_len_kv, (initial_valid,)))
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)
    assert state.route_workspace[0].item() == 2
    torch.testing.assert_close(
        state.route_workspace[
            stale_route_metadata_word_index : (
                stale_route_metadata_word_index + layout.route_metadata_stride_words
            )
        ],
        stale_route_metadata,
        rtol=0,
        atol=0,
    )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_dynamic_metadata_repartitions_rows_with_declared_capacity() -> None:
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

    with pytest.raises(ValueError, match="smaller than an initial BSR row"):
        _plan(
            block_sparse_module.BlockSparseTSWrapper(),
            block_indptr,
            block_indices,
            seq_len_q=case.seq_len_q,
            seq_len_kv=case.seq_len_kv,
            q_block_size=case.q_block_size,
            kv_block_size=case.kv_block_size,
            dynamic_metadata=True,
            max_blocks_per_row=0,
        )

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
        dynamic_metadata=True,
        max_blocks_per_row=3,
    )

    state = wrapper._published_state()
    assert state.row_route_offsets.tolist() == [0, 3, 6]
    assert state.route_layout.route_metadata_capacity == 6
    assert dict(state.policy)["max_row_route_capacity"] == 3

    initial = wrapper.run(q, k, v, sm_scale=sm_scale)
    block_indptr.copy_(replay_indptr)
    block_indices.copy_(replay_indices)
    replay = wrapper.run(q, k, v, sm_scale=sm_scale)
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
    block_indptr.copy_(
        torch.tensor([[[0, 4, 4]]], device="cuda", dtype=torch.int32)
    )
    overflow = wrapper.run(q, k, v, sm_scale=sm_scale)
    torch.cuda.synchronize()
    assert state.route_workspace[0].item() == -4
    assert torch.count_nonzero(overflow).item() == 0
