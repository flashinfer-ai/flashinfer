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

from dataclasses import dataclass, replace
import inspect
import math
from types import SimpleNamespace
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from cutlass import Float16

import flashinfer.attention.prims_ts as prims_ts
from flashinfer.attention.prims_ts import block_sparse as block_sparse_module
from flashinfer.attention.prims_ts._block_sparse.common import (
    _block_sparse_kv_routes_are_block_aligned,
)
from flashinfer.attention.prims_ts._block_sparse.inspection import (
    _BlockSparseInspection,
    _inspect_block_sparse_bsr,
)
from flashinfer.attention.prims_ts._block_sparse.plan import (
    _resolve_execution_geometry,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    FmhaDecodeConfig,
    make_decode_config,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
    build_decode_task_manager,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.helpers_block_sparse import (
    _block_sparse_row_retained_route_count_host,
    _resolve_block_sparse_aligned_route_origin_host,
    _resolve_block_sparse_route_origins_host,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_block_sparse_metadata import (
    SmemBlockSparseSoftmaxMetadataResource,
    _BlockSparseKvMetadataLayout,
    _BlockSparseSoftmaxMetadataLayout,
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
        "q64_kv64_fp16_nonadjacent_tail_static",
        1,
        1,
        63,
        193,
        64,
        64,
        torch.float16,
        "dense",
        "none",
        "static",
    ),
    _Case(
        "q64_kv128_bf16_causal_holey_clc",
        1,
        1,
        129,
        321,
        64,
        128,
        torch.bfloat16,
        "causal",
        "holey",
        "persistent",
    ),
    _Case(
        "q64_kv256_fp16_all_one_static",
        1,
        2,
        65,
        513,
        64,
        256,
        torch.float16,
        "dense",
        "all_one",
        "static",
    ),
    _Case(
        "q128_kv64_bf16_holey_clc",
        1,
        1,
        191,
        319,
        128,
        64,
        torch.bfloat16,
        "dense",
        "holey",
        "persistent",
    ),
    _Case(
        "q128_kv128_fp16_causal_static",
        1,
        1,
        190,
        322,
        128,
        128,
        torch.float16,
        "causal",
        "none",
        "static",
    ),
    _Case(
        "q128_kv256_bf16_batched_heads_holey_clc",
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
    _Case(
        "q256_kv256_fp16_full_density_raw_static",
        1,
        1,
        256,
        513,
        256,
        256,
        torch.float16,
        "dense",
        "all_one",
        "static",
        pattern="full",
    ),
    _Case(
        "q8_kv16_fp16_holey_empty_row_static",
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
        pattern="mixed_pairs",
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
        pattern="mixed_pairs",
    ),
    _Case(
        "q16_kv64_bf16_dense_clc",
        1,
        1,
        47,
        385,
        16,
        64,
        torch.bfloat16,
        "dense",
        "none",
        "persistent",
        pattern="mixed_pairs",
    ),
    _Case(
        "q32_kv8_fp16_holey_static",
        1,
        1,
        65,
        151,
        32,
        8,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="mixed_pairs",
    ),
    _Case(
        "q32_kv128_bf16_full_route_tail_static",
        1,
        1,
        65,
        385,
        32,
        128,
        torch.bfloat16,
        "dense",
        "none",
        "static",
        pattern="mixed_pairs",
    ),
    _Case(
        "q32_kv8_fp16_offset_contiguous_full_causal_holey_static",
        1,
        1,
        256,
        256,
        32,
        8,
        torch.float16,
        "causal",
        "holey",
        "static",
        pattern="offset_full_route",
    ),
    _Case(
        "q8_kv8_fp16_offset_partial_auto_static",
        1,
        1,
        16,
        257,
        8,
        8,
        torch.float16,
        "dense",
        "none",
        "persistent",
        pattern="offset_partial_route",
    ),
    _Case(
        "q16_kv16_bf16_partial_tail_causal_holey_static",
        1,
        1,
        31,
        73,
        16,
        16,
        torch.bfloat16,
        "causal",
        "holey",
        "static",
        pattern="offset_partial_route",
    ),
    _Case(
        "q32_kv32_fp16_contiguous_physical_tail_static",
        1,
        1,
        65,
        127,
        32,
        32,
        torch.float16,
        "dense",
        "none",
        "static",
        pattern="full",
    ),
    _Case(
        "q32_kv8_fp16_two_kv64_holey_static",
        1,
        1,
        65,
        512,
        32,
        8,
        torch.float16,
        "dense",
        "holey",
        "static",
        pattern="two_kv64",
    ),
    _Case(
        "q32_kv16_bf16_two_kv64_partial_causal_holey_static",
        1,
        1,
        33,
        512,
        32,
        16,
        torch.bfloat16,
        "causal",
        "holey",
        "static",
        pattern="two_kv64_partial",
    ),
    _Case(
        "q32_kv32_fp16_two_kv64_physical_tail_static",
        1,
        1,
        65,
        225,
        32,
        32,
        torch.float16,
        "dense",
        "none",
        "static",
        pattern="two_kv64_tail",
    ),
    _Case(
        "q32_kv8_fp16_two_kv64_unaligned_tail_causal_holey_static",
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
        pattern="two_kv64_unaligned_tail",
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
                if case.pattern == "full":
                    rows.append(tuple(range(num_kv_blocks)))
                    continue
                if case.pattern == "offset_full_route":
                    blocks_per_route = 128 // case.kv_block_size
                    rows.append(tuple(range(1, blocks_per_route + 1)))
                    continue
                if case.pattern == "offset_partial_route":
                    blocks_per_route = 128 // case.kv_block_size
                    partial_blocks = max(1, blocks_per_route - 4)
                    rows.append(
                        tuple(
                            range(
                                1,
                                min(num_kv_blocks, 1 + partial_blocks),
                            )
                        )
                    )
                    continue
                if case.pattern in ("two_kv64", "two_kv64_partial"):
                    blocks_per_kv64 = 64 // case.kv_block_size
                    second_blocks = blocks_per_kv64
                    if case.pattern == "two_kv64_partial":
                        second_blocks -= 1
                    rows.append(
                        (
                            *range(blocks_per_kv64),
                            *range(
                                2 * blocks_per_kv64,
                                2 * blocks_per_kv64 + second_blocks,
                            ),
                        )
                    )
                    continue
                if case.pattern == "two_kv64_tail":
                    blocks_per_kv64 = 64 // case.kv_block_size
                    rows.append(
                        (
                            *range(1, 1 + blocks_per_kv64),
                            *range(num_kv_blocks - blocks_per_kv64, num_kv_blocks),
                        )
                    )
                    continue
                if case.pattern == "two_kv64_unaligned_tail":
                    # Both KV64 segments start at a non-K32-aligned token.
                    # The second segment reaches Skv=233, so its last B8 block
                    # contributes one real token and exercises cross-word
                    # token-bit stitching together with the physical tail.
                    rows.append((*range(1, 9), *range(22, 30)))
                    continue
                if case.pattern == "mixed_pairs":
                    # Pair hits: (0, 1)/(6, 7); miss: (3, 5); last: partial tail.
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


def _pack_single_batch_token_mask(
    seq_len_kv: int,
    valid_tokens: frozenset[int],
) -> torch.Tensor:
    """Pack one batch item's valid-token indices into the public mask ABI."""

    words = [0] * math.ceil(seq_len_kv / 32)
    for token_idx in valid_tokens:
        words[token_idx // 32] |= 1 << (token_idx % 32)
    return torch.tensor([words], device="cuda", dtype=torch.uint32)


def _make_token_mask(
    case: _Case,
) -> tuple[torch.Tensor | None, tuple[frozenset[int], ...]]:
    all_tokens = frozenset(range(case.seq_len_kv))
    if case.token_mask == "none":
        return None, tuple(all_tokens for _ in range(case.batch_size))

    valid_by_batch: list[frozenset[int]] = []
    packed_by_batch: list[list[int]] = []
    for batch_idx in range(case.batch_size):
        if case.token_mask == "all_one":
            valid = all_tokens
        elif case.token_mask == "holey":
            valid = frozenset(
                token_idx
                for token_idx in range(case.seq_len_kv)
                if (token_idx + batch_idx) % 7 not in (0, 3)
            )
        else:
            raise AssertionError(f"unknown token-mask pattern {case.token_mask!r}")
        words = [0] * math.ceil(case.seq_len_kv / 32)
        for token_idx in valid:
            words[token_idx // 32] |= 1 << (token_idx % 32)
        valid_by_batch.append(valid)
        packed_by_batch.append(words)
    return (
        torch.tensor(packed_by_batch, device="cuda", dtype=torch.uint32),
        tuple(valid_by_batch),
    )


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


def _make_cuda_metadata(
    *,
    batch_size: int = 1,
    num_heads: int = 2,
    rows: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = batch_size * num_heads * rows
    offsets = torch.arange(row_count + 1, device="cuda", dtype=torch.int32)
    block_indptr = (
        offsets.unfold(0, rows + 1, rows)
        .reshape(batch_size, num_heads, rows + 1)
        .contiguous()
    )
    block_indices = torch.arange(row_count, device="cuda", dtype=torch.int32) % 2
    return block_indptr, block_indices


def _inspection(
    *,
    max_row_nnz: int = 1,
    max_retained_routes: int = 1,
    token_mask_has_holes: bool = False,
) -> _BlockSparseInspection:
    return _BlockSparseInspection(
        max_row_nnz=max_row_nnz,
        max_retained_routes=max_retained_routes,
        token_mask_has_holes=token_mask_has_holes,
    )


def _install_fake_plan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
    *,
    inspection: _BlockSparseInspection | None = None,
) -> None:
    inspection = _inspection() if inspection is None else inspection

    def inspect_bsr(*args: object, **kwargs: object) -> _BlockSparseInspection:
        calls = captured.setdefault("inspections", [])
        assert isinstance(calls, list)
        calls.append((args, kwargs))
        return inspection

    def resolve_spec(
        _device_index: int,
        _batch_size: int,
        _seq_len_q: int,
        _seq_len_kv: int,
        _num_heads: int,
        _head_dim: int,
        _q_block_size: int,
        _kv_block_size: int,
        q_tile_size: int,
        _q_dtype_key: str,
        _kv_dtype_key: str,
        _output_dtype_key: str,
        _mask_type: str,
        use_token_mask: bool,
        _max_retained_routes: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(use_persistent_scheduler=False),
            policy=(
                ("tile_size_q", q_tile_size),
                ("use_persistent_scheduler", False),
                ("execution_path", "raw_bsr_decode"),
                ("use_kv_valid_bits", use_token_mask),
            ),
            compile_key=("fake-block-sparse", use_token_mask),
        )

    def get_compiled(*_args: object) -> object:
        def compiled(
            q: torch.Tensor,
            _k: torch.Tensor,
            _v: torch.Tensor,
            out: torch.Tensor,
            block_indptr: torch.Tensor,
            block_indices: torch.Tensor,
            runtime_kv_valid_bits: torch.Tensor,
            sm_scale: float,
        ) -> None:
            launches = captured.setdefault("launches", [])
            assert isinstance(launches, list)
            launches.append(
                (block_indptr, block_indices, runtime_kv_valid_bits, sm_scale)
            )
            launch_streams = captured.setdefault("launch_streams", [])
            assert isinstance(launch_streams, list)
            launch_streams.append(torch.cuda.current_stream(q.device).cuda_stream)
            out.copy_(q)

        return compiled

    monkeypatch.setattr(block_sparse_module, "_inspect_block_sparse_bsr", inspect_bsr)
    monkeypatch.setattr(
        block_sparse_module,
        "_resolve_raw_block_sparse_launch_spec",
        resolve_spec,
    )
    monkeypatch.setattr(block_sparse_module, "_get_compiled_block_sparse", get_compiled)


def _plan_default(
    wrapper: block_sparse_module.BlockSparseTSWrapper,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    kv_valid_bits: torch.Tensor | None = None,
) -> None:
    wrapper.plan(
        block_indptr,
        block_indices,
        1,
        64,
        128,
        2,
        2,
        _HEAD_DIM,
        64,
        64,
        kv_valid_bits=kv_valid_bits,
    )


def _planned_fake_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[block_sparse_module.BlockSparseTSWrapper, dict[str, object]]:
    captured: dict[str, object] = {}
    _install_fake_plan_dependencies(monkeypatch, captured)
    block_indptr, block_indices = _make_cuda_metadata()
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan_default(wrapper, block_indptr, block_indices)
    return wrapper, captured


def _inspect(
    indptr: object,
    indices: object,
    *,
    seq_len_q: int = 64,
    seq_len_kv: int = 256,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    kv_valid_bits: torch.Tensor | None = None,
) -> _BlockSparseInspection:
    return _inspect_block_sparse_bsr(
        torch.tensor(indptr, device="cuda", dtype=torch.int32),
        torch.tensor(indices, device="cuda", dtype=torch.int32),
        batch_size=1,
        num_kv_heads=1,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        kv_valid_bits=kv_valid_bits,
    )


def _make_inspection_token_bits(
    seq_len_kv: int,
    *,
    holes: tuple[int, ...] = (),
) -> torch.Tensor:
    """Pack valid real-token bits while leaving physical-tail padding clear."""

    words = []
    for word_begin in range(0, seq_len_kv, 32):
        real_tokens = min(32, seq_len_kv - word_begin)
        words.append((1 << real_tokens) - 1)
    for token_idx in holes:
        words[token_idx // 32] &= ~(1 << (token_idx % 32))
    return torch.tensor([words], device="cuda", dtype=torch.uint32)


def _make_task_manager_config(
    *,
    q_tile_size: int,
    persistent: bool,
) -> FmhaDecodeConfig:
    return make_decode_config(
        headdim=_HEAD_DIM,
        args={
            "use_keeps_mma_ab": q_tile_size >= 64,
            "tile_size_q": q_tile_size,
            "tile_size_kv": 128,
            "groups_tokens_heads_q": True,
            "use_persistent_scheduler": persistent,
            "use_block_sparse": True,
            "q_block_size": q_tile_size,
            "kv_block_size": 128,
        },
        seq_len_q=256,
        seq_len_kv=512,
        batch_size=2,
        num_heads_q=2,
        num_heads_kv=2,
        qkv_dtype=Float16,
        o_dtype=Float16,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_lifecycle_problem() -> tuple[
    block_sparse_module.BlockSparseTSWrapper,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(20260721)
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_indptr = torch.tensor(
        [[[0, 2], [2, 4]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = torch.tensor(
        [0, 1, 0, 1],
        device="cuda",
        dtype=torch.int32,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    wrapper.plan(
        block_indptr,
        block_indices,
        1,
        64,
        128,
        2,
        2,
        _HEAD_DIM,
        64,
        64,
    )
    return wrapper, q, k, v


# Public API and plan/run validation.


def test_one_shot_is_strictly_plan_then_run(monkeypatch: pytest.MonkeyPatch) -> None:
    assert prims_ts.BlockSparseTSWrapper is block_sparse_module.BlockSparseTSWrapper
    assert prims_ts.block_sparse_attention is block_sparse_module.block_sparse_attention
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    class FakeWrapper:
        def plan(self, *args: object, **kwargs: object) -> None:
            calls.append(("plan", args, kwargs))

        def run(self, *args: object, **kwargs: object) -> object:
            calls.append(("run", args, kwargs))
            return sentinel

    monkeypatch.setattr(block_sparse_module, "BlockSparseTSWrapper", FakeWrapper)
    q = torch.empty((2, 64, 4, _HEAD_DIM), dtype=torch.float16)
    k = torch.empty((2, 192, 4, _HEAD_DIM), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indptr = torch.empty((2, 4, 2), dtype=torch.int32)
    block_indices = torch.empty((0,), dtype=torch.int32)
    out = torch.empty_like(q)

    result = block_sparse_module.block_sparse_attention(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        64,
        128,
        mask_type="causal",
        sm_scale=0.125,
        out=out,
    )

    assert result is sentinel
    assert [call[0] for call in calls] == ["plan", "run"]
    assert calls[0][1] == (
        block_indptr,
        block_indices,
        2,
        64,
        192,
        4,
        4,
        _HEAD_DIM,
        64,
        128,
    )
    assert calls[0][2] == {
        "kv_valid_bits": None,
        "mask_type": "causal",
        "q_data_type": torch.float16,
        "kv_data_type": torch.float16,
        "o_data_type": torch.float16,
    }
    assert calls[1][1] == (q, k, v)
    assert calls[1][2] == {"sm_scale": 0.125, "out": out}


def test_one_shot_validates_inputs_before_constructing_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    class UnexpectedWrapper:
        def __init__(self) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(
        block_sparse_module,
        "BlockSparseTSWrapper",
        UnexpectedWrapper,
    )
    q = torch.empty((1, 64, 1, _HEAD_DIM), dtype=torch.float16)
    k = torch.empty((1, 128, 1, _HEAD_DIM), dtype=torch.float16)
    metadata = torch.empty((1, 1, 2), dtype=torch.int32)
    indices = torch.empty((0,), dtype=torch.int32)

    with pytest.raises(ValueError, match="q must be rank 4"):
        block_sparse_module.block_sparse_attention(
            torch.empty(0), k, k, metadata, indices, 64, 64
        )
    with pytest.raises(TypeError, match="out must be a torch.Tensor"):
        block_sparse_module.block_sparse_attention(
            q, k, k, metadata, indices, 64, 64, out=object()
        )
    assert not constructed


def test_run_requires_a_successful_plan() -> None:
    q = torch.empty((1, 1, 1, _HEAD_DIM), dtype=torch.float16)
    with pytest.raises(RuntimeError, match=r"plan\(\).*before run"):
        block_sparse_module.BlockSparseTSWrapper().run(q, q, q)


def test_inspection_contract_excludes_runtime_route_specialization() -> None:
    """Planning facts must not freeze runtime route or mask morphology."""

    forbidden_compile_flags = {
        "use_token_word_full_guard",
        "use_q128_token_route_full_guard",
    }
    assert set(_BlockSparseInspection.__dataclass_fields__) == {
        "max_row_nnz",
        "max_retained_routes",
        "token_mask_has_holes",
    }
    assert forbidden_compile_flags.isdisjoint(
        block_sparse_module._RawBlockSparseLaunchTraits.__dataclass_fields__
    )
    assert forbidden_compile_flags.isdisjoint(FmhaDecodeConfig.__dataclass_fields__)
    assert "contiguous_kv_route_size" not in FmhaDecodeConfig.__dataclass_fields__
    for function in (
        block_sparse_module._make_block_sparse_config,
        block_sparse_module._resolve_raw_block_sparse_launch_spec,
        block_sparse_module._get_compiled_block_sparse,
    ):
        assert forbidden_compile_flags.isdisjoint(
            inspect.signature(function).parameters
        )


@pytest.mark.parametrize(
    ("q_block_size", "expected_q_tile"),
    (
        pytest.param(8, 8, id="q8"),
        pytest.param(16, 16, id="q16"),
        pytest.param(32, 32, id="q32"),
        pytest.param(64, 64, id="q64"),
        pytest.param(128, 128, id="q128"),
        pytest.param(192, 64, id="q192"),
        pytest.param(256, 128, id="q256"),
    ),
)
def test_execution_geometry_selects_canonical_q_tile(
    q_block_size: int,
    expected_q_tile: int,
) -> None:
    geometry = _resolve_execution_geometry(q_block_size, 8)

    assert geometry.q_tile_size == expected_q_tile
    assert geometry.kv_tile_size == 128


@pytest.mark.parametrize(
    (
        "q_block_size",
        "kv_block_size",
        "q_dtype_key",
        "mask_type",
        "persistent",
        "expected_keeps",
    ),
    (
        pytest.param(
            8,
            16,
            "float16",
            "dense",
            False,
            False,
            id="q8-kv16-fp16-static",
        ),
        pytest.param(
            64,
            64,
            "float16",
            "dense",
            False,
            True,
            id="q64-kv64-fp16-static",
        ),
        pytest.param(
            64,
            128,
            "bfloat16",
            "causal",
            True,
            True,
            id="q64-kv128-bf16-persistent",
        ),
        pytest.param(
            128,
            64,
            "float16",
            "causal",
            True,
            True,
            id="q128-kv64-fp16-persistent",
        ),
        pytest.param(
            128,
            128,
            "bfloat16",
            "dense",
            False,
            True,
            id="q128-kv128-bf16-static",
        ),
    ),
)
def test_block_sparse_config_selects_canonical_mma_orientation(
    q_block_size: int,
    kv_block_size: int,
    q_dtype_key: str,
    mask_type: str,
    persistent: bool,
    expected_keeps: bool,
) -> None:
    geometry = _resolve_execution_geometry(q_block_size, kv_block_size)

    config = block_sparse_module._make_block_sparse_config(
        batch_size=1,
        seq_len_q=2 * q_block_size + 1,
        seq_len_kv=257,
        num_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_tile_size=geometry.q_tile_size,
        q_dtype_key=q_dtype_key,
        output_dtype_key=q_dtype_key,
        mask_type=mask_type,
        use_kv_valid_bits=False,
        use_persistent_scheduler=persistent,
    )

    assert config.use_keeps_mma_ab is expected_keeps
    assert config.supports_grouped_keeps is expected_keeps
    assert config.tile_size_q == geometry.q_tile_size
    assert config.tile_size_kv == 128
    assert config.use_persistent_scheduler is persistent


@pytest.mark.parametrize(
    ("q_block_size", "q_dtype_key", "persistent"),
    (
        pytest.param(128, "float16", True, id="fp16-q128-persistent"),
        pytest.param(64, "bfloat16", False, id="bf16-q64-static"),
    ),
)
def test_block_sparse_keeps_recipes_do_not_enable_dense_profiles(
    q_block_size: int,
    q_dtype_key: str,
    persistent: bool,
) -> None:
    geometry = _resolve_execution_geometry(q_block_size, q_block_size)
    sparse_config = block_sparse_module._make_block_sparse_config(
        batch_size=1,
        seq_len_q=2 * q_block_size,
        seq_len_kv=256,
        num_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=q_block_size,
        kv_block_size=q_block_size,
        q_tile_size=geometry.q_tile_size,
        q_dtype_key=q_dtype_key,
        output_dtype_key=q_dtype_key,
        mask_type="dense",
        use_kv_valid_bits=False,
        use_persistent_scheduler=persistent,
    )
    dense_config = replace(
        sparse_config,
        use_block_sparse=False,
        q_block_size=0,
        kv_block_size=0,
    )

    assert sparse_config.supports_grouped_keeps is True
    assert dense_config.supports_grouped_keeps is False


@pytest.mark.parametrize("parameter_name", ("q_block_size", "kv_block_size"))
@pytest.mark.parametrize(
    ("invalid_value", "error_type", "message"),
    (
        pytest.param(
            True,
            TypeError,
            "8, 16, 32, or a positive multiple of 64",
            id="bool",
        ),
        pytest.param(
            64.0,
            TypeError,
            "8, 16, 32, or a positive multiple of 64",
            id="float",
        ),
        pytest.param(
            0,
            ValueError,
            "8, 16, 32, or a positive multiple of 64",
            id="zero",
        ),
        pytest.param(
            96,
            ValueError,
            "8, 16, 32, or a positive multiple of 64",
            id="unsupported-96",
        ),
        pytest.param(
            1 << 32,
            OverflowError,
            "fit in signed int32",
            id="int32-overflow",
        ),
    ),
)
def test_plan_rejects_unsupported_block_sizes(
    parameter_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "block_indptr": None,
        "block_indices": None,
        "batch_size": 1,
        "seq_len_q": 64,
        "seq_len_kv": 64,
        "num_qo_heads": 1,
        "num_kv_heads": 1,
        "head_dim": _HEAD_DIM,
        "q_block_size": 64,
        "kv_block_size": 64,
    }
    arguments[parameter_name] = invalid_value

    with pytest.raises(error_type, match=message):
        block_sparse_module.BlockSparseTSWrapper().plan(**arguments)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    (
        pytest.param({"head_dim": 64}, ValueError, "head_dim=128", id="head-dim"),
        pytest.param({"num_qo_heads": 2}, ValueError, "Hq == Hkv", id="mha-only"),
        pytest.param(
            {"seq_len_q": 129, "seq_len_kv": 128, "mask_type": "causal"},
            ValueError,
            "seq_len_q <= seq_len_kv",
            id="causal-length",
        ),
        pytest.param({"mask_type": "window"}, ValueError, "dense.*causal", id="mask"),
        pytest.param(
            {"kv_data_type": torch.bfloat16},
            ValueError,
            "matching",
            id="mixed-dtype",
        ),
        pytest.param(
            {
                "q_data_type": torch.float32,
                "kv_data_type": torch.float32,
                "o_data_type": torch.float32,
            },
            NotImplementedError,
            "supports",
            id="unsupported-matching-dtype",
        ),
        pytest.param(
            {"q_block_size": 64, "kv_block_size": 8},
            ValueError,
            "fine KV blocks require a SwapsMmaAb Q tile",
            id="fine-kv-block_sizes-keeps-profile",
        ),
        pytest.param(
            {"dynamic_metadata": 1},
            TypeError,
            "dynamic_metadata must be a bool",
            id="dynamic-metadata-type",
        ),
    ),
)
def test_plan_validates_supported_profile_before_metadata(
    overrides: dict[str, object],
    error_type: type[Exception],
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

    with pytest.raises(error_type, match=message):
        block_sparse_module.BlockSparseTSWrapper().plan(**arguments)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "invalid_kind",
    (
        "indptr-rank",
        "indptr-dtype",
        "indptr-shape",
        "indptr-stride",
        "indices-rank",
        "indices-dtype",
    ),
)
def test_plan_validates_canonical_bsr_tensor_contract(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    monkeypatch.setattr(
        block_sparse_module,
        "_inspect_block_sparse_bsr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inspection must not run for malformed tensors")
        ),
    )
    block_indptr, block_indices = _make_cuda_metadata()
    if invalid_kind == "indptr-rank":
        block_indptr = torch.zeros((1, 2), device="cuda", dtype=torch.int32)
    elif invalid_kind == "indptr-dtype":
        block_indptr = block_indptr.to(torch.int64)
    elif invalid_kind == "indptr-shape":
        block_indptr = torch.zeros((1, 2, 3), device="cuda", dtype=torch.int32)
    elif invalid_kind == "indptr-stride":
        block_indptr = torch.zeros((1, 2, 4), device="cuda", dtype=torch.int32)[
            ..., ::2
        ]
    elif invalid_kind == "indices-rank":
        block_indices = block_indices.reshape(1, -1)
    elif invalid_kind == "indices-dtype":
        block_indices = block_indices.to(torch.int64)
    else:
        raise AssertionError(f"unknown invalid kind {invalid_kind!r}")

    with pytest.raises((TypeError, ValueError)):
        _plan_default(
            block_sparse_module.BlockSparseTSWrapper(),
            block_indptr,
            block_indices,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "invalid_kind",
    ("dtype", "shape", "device", "stride"),
)
def test_plan_validates_optional_token_bits_contract(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    monkeypatch.setattr(
        block_sparse_module,
        "_inspect_block_sparse_bsr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inspection must not run for malformed token bits")
        ),
    )
    block_indptr, block_indices = _make_cuda_metadata()
    if invalid_kind == "dtype":
        bits = torch.zeros((1, 4), device="cuda", dtype=torch.int32)
    elif invalid_kind == "shape":
        bits = torch.zeros((1, 3), device="cuda", dtype=torch.uint32)
    elif invalid_kind == "device":
        bits = torch.zeros((1, 4), dtype=torch.uint32)
    elif invalid_kind == "stride":
        bits = torch.zeros((1, 8), device="cuda", dtype=torch.uint32)[:, ::2]
    else:
        raise AssertionError(f"unknown invalid kind {invalid_kind!r}")

    with pytest.raises((TypeError, ValueError)):
        _plan_default(
            block_sparse_module.BlockSparseTSWrapper(),
            block_indptr,
            block_indices,
            kv_valid_bits=bits,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "invalid_kind",
    ("q-shape", "k-shape", "v-dtype", "q-device", "q-stride"),
)
@torch.no_grad()
def test_run_validates_qkv_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    wrapper, _ = _planned_fake_wrapper(monkeypatch)
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    if invalid_kind == "q-shape":
        q = q[:, :-1]
    elif invalid_kind == "k-shape":
        k = k[:, :-1]
    elif invalid_kind == "v-dtype":
        v = v.to(torch.bfloat16)
    elif invalid_kind == "q-device":
        q = q.cpu()
    elif invalid_kind == "q-stride":
        q = torch.empty(
            (1, 64, 2, _HEAD_DIM * 2),
            device="cuda",
            dtype=torch.float16,
        )[..., ::2]
    else:
        raise AssertionError(f"unknown invalid kind {invalid_kind!r}")

    with pytest.raises(ValueError):
        wrapper.run(q, k, v)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_run_validates_scale_and_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _ = _planned_fake_wrapper(monkeypatch)
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    for invalid_scale in (0.0, float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            wrapper.run(q, k, v, sm_scale=invalid_scale)
    invalid_outputs = (
        q[:, :-1],
        torch.empty_like(q, dtype=torch.bfloat16),
        torch.empty(
            (1, 64, 2, _HEAD_DIM * 2),
            device="cuda",
            dtype=torch.float16,
        )[..., ::2],
        q,
    )
    for invalid_out in invalid_outputs:
        with pytest.raises(ValueError):
            wrapper.run(q, k, v, out=invalid_out)


# Published plan state and specialization.


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_plan_retains_raw_bsr_and_run_forwards_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_plan_dependencies(monkeypatch, captured)
    block_indptr, block_indices = _make_cuda_metadata()
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan_default(wrapper, block_indptr, block_indices)

    assert wrapper._block_indptr is block_indptr
    assert wrapper._block_indices is block_indices
    assert dict(wrapper._policy)["execution_path"] == "raw_bsr_decode"
    state = wrapper._plan_state
    assert state is not None
    assert state.kv_valid_bits is None
    assert state.runtime_kv_valid_bits.dtype == torch.uint32
    assert tuple(state.runtime_kv_valid_bits.shape) == (1, 4)

    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    result = wrapper.run(q, k, k, out=out)

    assert result is out
    torch.testing.assert_close(result, q)
    launch = captured["launches"]
    assert isinstance(launch, list)
    assert launch[-1][0] is block_indptr
    assert launch[-1][1] is block_indices
    assert launch[-1][2] is state.runtime_kv_valid_bits


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mask_kind", ("all-one", "holey"))
def test_plan_selects_dummy_or_runtime_token_mask(
    monkeypatch: pytest.MonkeyPatch,
    mask_kind: str,
) -> None:
    has_holes = mask_kind == "holey"
    captured: dict[str, object] = {}
    _install_fake_plan_dependencies(
        monkeypatch,
        captured,
        inspection=_inspection(token_mask_has_holes=has_holes),
    )
    block_indptr, block_indices = _make_cuda_metadata()
    valid_bits = torch.full(
        (1, 4),
        0xFFFFFFFF,
        device="cuda",
        dtype=torch.uint32,
    )
    if has_holes:
        valid_bits[0, 0] = 0xFFFFFFFE
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan_default(
        wrapper,
        block_indptr,
        block_indices,
        kv_valid_bits=valid_bits,
    )

    state = wrapper._plan_state
    assert state is not None
    assert dict(state.policy)["use_kv_valid_bits"] is has_holes
    if has_holes:
        assert state.kv_valid_bits is valid_bits
        assert state.runtime_kv_valid_bits is valid_bits
    else:
        assert state.kv_valid_bits is None
        assert state.runtime_kv_valid_bits is not valid_bits
        assert tuple(state.runtime_kv_valid_bits.shape) == tuple(valid_bits.shape)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_dynamic_plan_does_not_inspect_mutable_token_mask_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_plan_dependencies(monkeypatch, captured)
    block_indptr, block_indices = _make_cuda_metadata()
    valid_bits = torch.full(
        (1, 4),
        0xFFFFFFFF,
        device="cuda",
        dtype=torch.uint32,
    )
    wrapper = block_sparse_module.BlockSparseTSWrapper()

    wrapper.plan(
        block_indptr,
        block_indices,
        1,
        64,
        128,
        2,
        2,
        _HEAD_DIM,
        64,
        64,
        kv_valid_bits=valid_bits,
        dynamic_metadata=True,
    )

    inspections = captured["inspections"]
    assert isinstance(inspections, list)
    _, inspection_kwargs = inspections[-1]
    assert inspection_kwargs["kv_valid_bits"] is None
    state = wrapper._plan_state
    assert state is not None
    assert state.runtime_kv_valid_bits is valid_bits
    assert dict(state.policy)["use_kv_valid_bits"] is True


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_failed_replan_atomically_preserves_runnable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_plan_dependencies(monkeypatch, captured)
    block_indptr, block_indices = _make_cuda_metadata()
    wrapper = block_sparse_module.BlockSparseTSWrapper()
    _plan_default(wrapper, block_indptr, block_indices)
    previous_state = wrapper._plan_state

    def fail_inspection(*_args: object, **_kwargs: object) -> object:
        raise ValueError("canonical BSR inspection failed")

    monkeypatch.setattr(
        block_sparse_module,
        "_inspect_block_sparse_bsr",
        fail_inspection,
    )
    with pytest.raises(ValueError, match="canonical BSR inspection failed"):
        _plan_default(wrapper, block_indptr, block_indices)

    assert wrapper._plan_state is previous_state
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(wrapper.run(q, k, k), q)


# Inspector, route mapping, and task-manager behavior.


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("indptr", "indices", "reason"),
    (
        pytest.param([[[0, 2]]], (1, 0), "strictly increasing", id="unsorted"),
        pytest.param([[[0, 2]]], (0, 0), "strictly increasing", id="duplicate"),
        pytest.param(
            [[[0, 2]]],
            (0, 4),
            "in-range KV block",
            id="out-of-bounds-index",
        ),
        pytest.param(
            [[[-1, 0]]],
            (),
            "bounded and monotone",
            id="negative-row-begin",
        ),
        pytest.param(
            [[[1, 0]]],
            (0,),
            "bounded and monotone",
            id="row-begin-after-end",
        ),
        pytest.param(
            [[[0, 2]]],
            (0,),
            "bounded and monotone",
            id="row-end-past-indices",
        ),
    ),
)
def test_inspector_rejects_noncanonical_bsr(
    indptr: list[list[list[int]]],
    indices: tuple[int, ...],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=rf"canonical BSR.*{reason}"):
        _inspect(indptr, indices)
    torch.cuda.synchronize()


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    (
        "seq_len_kv",
        "q_block_size",
        "kv_block_size",
        "indices",
        "expected_nnz",
        "expected_routes",
    ),
    (
        pytest.param(193, 64, 64, (0, 2, 3), 3, 2, id="kv64"),
        pytest.param(385, 64, 128, (0, 3), 2, 2, id="kv128"),
        pytest.param(641, 64, 256, (0, 2), 2, 4, id="kv256"),
    ),
)
def test_inspector_reports_tail_trimmed_route_summary(
    seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
    indices: tuple[int, ...],
    expected_nnz: int,
    expected_routes: int,
) -> None:
    result = _inspect(
        [[[0, len(indices)]]],
        indices,
        seq_len_q=q_block_size,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )

    assert result.max_row_nnz == expected_nnz
    assert result.max_retained_routes == expected_routes


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    (
        "q_block_size",
        "kv_block_size",
        "seq_len_kv",
        "indices",
        "holes",
        "expected_has_holes",
        "expected_nnz",
        "expected_routes",
    ),
    (
        pytest.param(8, 8, 151, (0, 1, 3, 18), (24,), True, 4, 1, id="q8-kv8"),
        pytest.param(16, 16, 207, (0, 3, 5, 12), (85,), True, 4, 1, id="q16-kv16"),
        pytest.param(32, 32, 289, (0, 3, 7, 9), (225,), True, 4, 1, id="q32-kv32"),
        pytest.param(
            8,
            8,
            151,
            (18,),
            (0,),
            False,
            1,
            1,
            id="partial-tail-unselected-hole-ignored",
        ),
    ),
)
def test_inspector_reports_fine_swaps_selected_token_holes(
    q_block_size: int,
    kv_block_size: int,
    seq_len_kv: int,
    indices: tuple[int, ...],
    holes: tuple[int, ...],
    expected_has_holes: bool,
    expected_nnz: int,
    expected_routes: int,
) -> None:
    result = _inspect(
        [[[0, len(indices)]]],
        indices,
        seq_len_q=q_block_size,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        kv_valid_bits=_make_inspection_token_bits(seq_len_kv, holes=holes),
    )

    assert result.token_mask_has_holes is expected_has_holes
    assert result.max_row_nnz == expected_nnz
    assert result.max_retained_routes == expected_routes


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("words", "expected_has_holes"),
    (
        pytest.param(
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            False,
            id="all-selected-tokens-valid",
        ),
        pytest.param(
            (0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF),
            True,
            id="one-selected-token-hole",
        ),
    ),
)
def test_inspector_reports_reachable_token_holes_for_keeps(
    words: tuple[int, ...],
    expected_has_holes: bool,
) -> None:
    valid_bits = torch.tensor([words], device="cuda", dtype=torch.uint32)
    result = _inspect(
        [[[0, 2]]],
        (0, 1),
        seq_len_q=128,
        seq_len_kv=128,
        q_block_size=128,
        kv_valid_bits=valid_bits,
    )

    assert result == _BlockSparseInspection(
        max_row_nnz=2,
        max_retained_routes=1,
        token_mask_has_holes=expected_has_holes,
    )


@pytest.mark.parametrize(
    (
        "kv_block_size",
        "seq_len_kv",
        "selected",
        "expected_valid_offsets",
        "expected_route_count",
    ),
    (
        pytest.param(
            8,
            151,
            (0, 1, 3, 5, 6, 7, 18),
            (0, 8, 24, 40, 48, 56, 144),
            1,
            id="kv8-mixed-pairs-tail",
        ),
        pytest.param(
            16,
            207,
            (0, 1, 3, 5, 6, 7, 12),
            (0, 16, 48, 80, 96, 112, 192),
            1,
            id="kv16-mixed-pairs-tail",
        ),
        pytest.param(
            32,
            289,
            (0, 1, 3, 5, 6, 7, 9),
            (0, 32, 96, 160, 192, 224, 288),
            2,
            id="kv32-mixed-pairs-tail",
        ),
        pytest.param(
            64,
            193,
            (0, 2, 3),
            (0, 128, 192),
            2,
            id="kv64-nonadjacent-tail",
        ),
        pytest.param(
            128,
            385,
            (0, 3),
            (0, 64, 384),
            2,
            id="kv128-tail",
        ),
        pytest.param(
            256,
            641,
            (0, 2),
            (0, 64, 128, 192, 512, 576, 640),
            4,
            id="kv256-expansion-tail",
        ),
    ),
)
def test_host_route_mapping_for_supported_kv_blocks(
    kv_block_size: int,
    seq_len_kv: int,
    selected: tuple[int, ...],
    expected_valid_offsets: tuple[int, ...],
    expected_route_count: int,
) -> None:
    block_indices = (99, *selected, 98)
    row_begin = 1
    route_count = _block_sparse_row_retained_route_count_host(
        block_indices,
        row_begin,
        len(selected),
        kv_block_size,
        seq_len_kv,
    )
    routes = tuple(
        _resolve_block_sparse_route_origins_host(
            block_indices,
            row_begin,
            len(selected),
            route_idx,
            kv_block_size,
            seq_len_kv,
        )
        for route_idx in range(route_count)
    )
    valid_offsets = tuple(
        fragment.physical_token_offset
        for route in routes
        for fragment in route
        if fragment.valid
    )
    invalid_fragments = tuple(
        fragment for route in routes for fragment in route if not fragment.valid
    )
    atom_size = min(kv_block_size, 64)

    assert len(routes) == expected_route_count
    assert all(len(route) == 128 // atom_size for route in routes)
    assert valid_offsets == expected_valid_offsets
    assert all(fragment.physical_token_offset == 0 for fragment in invalid_fragments)


@pytest.mark.parametrize(
    ("kv_block_size", "seq_len_kv", "selected", "route_idx", "expected"),
    (
        pytest.param(128, 385, (0, 3), 0, (0, True), id="b128-first"),
        pytest.param(128, 385, (0, 3), 1, (384, True), id="b128-one-token-tail"),
        pytest.param(128, 385, (0, 3), 2, (0, False), id="b128-dummy"),
        pytest.param(256, 641, (0, 2), 1, (128, True), id="b256-subroute"),
        pytest.param(256, 641, (0, 2), 3, (640, True), id="b256-one-token-tail"),
        pytest.param(256, 641, (0, 2), 4, (0, False), id="b256-dummy"),
        pytest.param(384, 640, (1,), 1, (512, True), id="b384-subroute"),
        pytest.param(384, 640, (1,), 2, (0, False), id="b384-physical-tail"),
    ),
)
def test_aligned_coarse_route_resolves_one_kv128_origin(
    kv_block_size: int,
    seq_len_kv: int,
    selected: tuple[int, ...],
    route_idx: int,
    expected: tuple[int, bool],
) -> None:
    block_indices = (99, *selected, 98)

    actual = _resolve_block_sparse_aligned_route_origin_host(
        block_indices,
        row_begin=1,
        row_nnz=len(selected),
        route_idx=route_idx,
        kv_block_size=kv_block_size,
        seq_len_kv=seq_len_kv,
    )

    assert actual == expected


def test_empty_host_row_still_validates_semantic_kv_block_size() -> None:
    with pytest.raises(
        ValueError,
        match="kv_block_size must be 8, 16, 32, or a positive multiple of 64",
    ):
        _block_sparse_row_retained_route_count_host(
            (),
            row_begin=0,
            row_nnz=0,
            kv_block_size=96,
            seq_len_kv=128,
        )


@pytest.mark.parametrize(
    (
        "kv_block_size",
        "expected_atom_size",
        "expected_origin_words",
        "expected_total_words",
        "expected_route_flags_offset",
    ),
    (
        pytest.param(8, 8, 16, 16, None, id="kv8"),
        pytest.param(16, 16, 8, 8, None, id="kv16"),
        pytest.param(32, 32, 4, 4, None, id="kv32"),
        pytest.param(64, 64, 2, 4, 2, id="kv64"),
        pytest.param(128, 64, 2, 4, 2, id="kv128"),
        pytest.param(192, 64, 2, 4, 2, id="kv192"),
        pytest.param(256, 64, 2, 4, 2, id="kv256"),
    ),
)
def test_kv_route_metadata_layout_matches_execution_atoms(
    kv_block_size: int,
    expected_atom_size: int,
    expected_origin_words: int,
    expected_total_words: int,
    expected_route_flags_offset: int | None,
) -> None:
    layout = _BlockSparseKvMetadataLayout.create(kv_block_size=kv_block_size)

    assert layout.atom_size == expected_atom_size
    assert layout.num_origin_words == expected_origin_words
    assert layout.total_words == expected_total_words
    assert layout.route_flags_word_offset == expected_route_flags_offset


@pytest.mark.parametrize(
    ("kv_block_size", "expected"),
    (
        pytest.param(8, False, id="kv8"),
        pytest.param(32, False, id="kv32"),
        pytest.param(64, False, id="kv64"),
        pytest.param(128, True, id="kv128"),
        pytest.param(192, False, id="kv192"),
        pytest.param(256, True, id="kv256"),
        pytest.param(384, True, id="kv384"),
    ),
)
def test_kv128_routes_stay_inside_aligned_semantic_blocks(
    kv_block_size: int,
    expected: bool,
) -> None:
    assert _block_sparse_kv_routes_are_block_aligned(kv_block_size) is expected


@pytest.mark.parametrize(
    (
        "use_keeps_mma_ab",
        "kv_block_size",
        "has_token_bits",
        "expected_origin_words",
        "expected_origins_per_warp",
        "expected_token_offset",
        "expected_stage_stride",
    ),
    (
        pytest.param(True, 64, False, 2, 2, None, 4, id="keeps-no-mask"),
        pytest.param(True, 256, True, 2, 2, 4, 8, id="keeps-mask"),
        pytest.param(False, 8, False, 16, 4, None, 16, id="swaps-kv8"),
        pytest.param(False, 16, True, 8, 2, 8, 12, id="swaps-kv16-mask"),
        pytest.param(False, 32, True, 4, 1, 4, 8, id="swaps-kv32-mask"),
    ),
)
def test_softmax_metadata_layout_has_profile_specific_view(
    use_keeps_mma_ab: bool,
    kv_block_size: int,
    has_token_bits: bool,
    expected_origin_words: int,
    expected_origins_per_warp: int,
    expected_token_offset: int | None,
    expected_stage_stride: int,
) -> None:
    assert len(SmemBlockSparseSoftmaxMetadataResource._task_local_specs) == 7
    layout = _BlockSparseSoftmaxMetadataLayout.create(
        use_keeps_mma_ab=use_keeps_mma_ab,
        kv_block_size=kv_block_size,
        has_token_bits=has_token_bits,
        num_stages=2,
    )

    assert layout.num_origin_words == expected_origin_words
    assert layout.origins_per_warp == expected_origins_per_warp
    assert layout.token_words_word_offset == expected_token_offset
    assert layout.stage_stride_words == expected_stage_stride
    assert layout.total_words == 2 * expected_stage_stride


@pytest.mark.parametrize(
    "q_tile_size",
    (8, 16, 32, 64, 128),
    ids=("q8", "q16", "q32", "q64", "q128"),
)
@pytest.mark.parametrize("persistent", (False, True), ids=("static", "clc"))
def test_q_tile_scheduler_task_managers_validate(
    q_tile_size: int,
    persistent: bool,
) -> None:
    config = _make_task_manager_config(
        q_tile_size=q_tile_size,
        persistent=persistent,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager = build_decode_task_manager(
            config,
            seq_len_kv=512,
            batch_size=2,
            num_heads_kv=2,
            verbose=False,
        )

    assert task_manager is not None


@pytest.mark.parametrize("q_tile_size", (32, 64), ids=("swaps", "keeps"))
def test_block_sparse_schedule_builds_with_deeper_kv_prefetch(
    q_tile_size: int,
) -> None:
    """Build the SWAP and Keeps task graphs with >1 stage per K/V instruction."""

    config = replace(
        _make_task_manager_config(q_tile_size=q_tile_size, persistent=False),
        kv_stages=6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager = build_decode_task_manager(
            config,
            seq_len_kv=512,
            batch_size=2,
            num_heads_kv=2,
            verbose=False,
        )

    assert task_manager is not None


# Bounded public numerical and lifecycle coverage.


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_public_block_sparse_correctness_matrix(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
) -> None:
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda *, tile_size_q, persistent_min_tile_size_q, **_kwargs: (
            "static" if tile_size_q < persistent_min_tile_size_q else case.scheduler
        ),
    )
    block_sparse_module._clear_block_sparse_launch_profile_cache()
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
        actual = wrapper.run(q, k, v, sm_scale=sm_scale)
        torch.cuda.synchronize()
    finally:
        block_sparse_module._clear_block_sparse_launch_profile_cache()

    policy = dict(wrapper._policy)
    assert policy["execution_path"] == "raw_bsr_decode"
    geometry = _resolve_execution_geometry(case.q_block_size, case.kv_block_size)
    assert policy["tile_size_q"] == geometry.q_tile_size
    expected_persistent = case.scheduler == "persistent" and geometry.q_tile_size != 8
    assert policy["use_persistent_scheduler"] is expected_persistent
    has_runtime_holes = case.token_mask == "holey"
    assert policy["use_kv_valid_bits"] is has_runtime_holes
    assert "contiguous_kv_route_size" not in policy
    state = wrapper._plan_state
    assert state is not None
    if has_runtime_holes:
        assert state.runtime_kv_valid_bits is valid_bits
    else:
        assert state.kv_valid_bits is None
        assert state.runtime_kv_valid_bits is not valid_bits
    if case.pattern == "full":
        expected_blocks = tuple(range(math.ceil(case.seq_len_kv / case.kv_block_size)))
        assert all(
            row == expected_blocks
            for batch in patterns
            for head in batch
            for row in head
        )
    if case.include_empty_row:
        empty_row_begin = (
            math.ceil(case.seq_len_q / case.q_block_size) - 1
        ) * case.q_block_size
        assert torch.count_nonzero(actual[-1, empty_row_begin:, -1]).item() == 0
    tolerance = 2e-2 if case.dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_head_ranges_use_actual_indptr() -> None:
    """Each KV head must resolve its BSR row from the supplied indptr."""

    torch.manual_seed(20260723)
    block_indptr = torch.tensor(
        [[[2, 4], [0, 2]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = torch.tensor(
        [0, 1, 2, 3],
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 256, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    expected_by_head = []
    for head_idx, (token_begin, token_end) in enumerate(((128, 256), (0, 128))):
        q_head = q[0, :, head_idx].float()
        k_head = k[0, token_begin:token_end, head_idx].float()
        v_head = v[0, token_begin:token_end, head_idx].float()
        probabilities = torch.softmax(q_head @ k_head.T * sm_scale, dim=-1)
        expected_by_head.append(probabilities @ v_head)
    expected = torch.stack(expected_by_head, dim=1).unsqueeze(0).to(q.dtype)
    wrapper = block_sparse_module.BlockSparseTSWrapper()

    try:
        wrapper.plan(
            block_indptr,
            block_indices,
            1,
            64,
            256,
            2,
            2,
            _HEAD_DIM,
            64,
            64,
            q_data_type=q.dtype,
        )
        actual = wrapper.run(q, k, v, sm_scale=sm_scale)
        torch.cuda.synchronize()
    finally:
        block_sparse_module._clear_block_sparse_launch_profile_cache()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("selected_blocks", "token_mask"),
    (
        pytest.param((0, 1, 4, 5), "none", id="even-routes"),
        pytest.param((0, 4), "holey", id="odd-route-token-mask"),
    ),
)
@torch.no_grad()
def test_public_q64_kv64_even_odd_and_token_mask(
    selected_blocks: tuple[int, ...],
    token_mask: str,
) -> None:
    """General KV64 routing handles complete, padded, and masked route groups."""

    torch.manual_seed(20260723)
    case = _Case(
        "general_q64_kv64",
        1,
        1,
        64,
        384,
        64,
        64,
        torch.float16,
        "dense",
        token_mask,
        "static",
    )
    patterns: _Patterns = (((selected_blocks,),),)
    block_indptr, block_indices = _make_bsr(patterns)
    valid_bits, valid_by_batch = _make_token_mask(case)
    q = torch.randn((1, 64, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, 384, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
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
        wrapper.plan(
            block_indptr,
            block_indices,
            1,
            64,
            384,
            1,
            1,
            _HEAD_DIM,
            64,
            64,
            kv_valid_bits=valid_bits,
            q_data_type=case.dtype,
        )
        actual = wrapper.run(q, k, v, sm_scale=sm_scale)
        torch.cuda.synchronize()
    finally:
        block_sparse_module._clear_block_sparse_launch_profile_cache()

    policy = dict(wrapper._policy)
    assert policy["use_kv_valid_bits"] is (token_mask == "holey")
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_run_uses_the_callers_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, captured = _planned_fake_wrapper(monkeypatch)
    q = torch.randn((1, 64, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 2, _HEAD_DIM), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    default_stream = torch.cuda.current_stream(q.device)
    worker = torch.cuda.Stream(device=q.device)
    worker.wait_stream(default_stream)
    out = torch.empty_like(q)

    with torch.cuda.stream(worker):
        result = wrapper.run(q, k, v, out=out)

    assert result is out
    launch_streams = captured["launch_streams"]
    assert isinstance(launch_streams, list)
    assert launch_streams[-1] == worker.cuda_stream
    assert launch_streams[-1] != default_stream.cuda_stream
    worker.synchronize()
    torch.testing.assert_close(out, q, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_public_run_captures_and_replays_a_pinned_plan_revision() -> None:
    wrapper, q, k, v = _make_lifecycle_problem()
    expected = wrapper.run(q, k, v).clone()
    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        captured_result = wrapper.run(q, k, v, out=graph_out)

    assert captured_result is graph_out
    empty_indptr = torch.zeros((1, 2, 2), device="cuda", dtype=torch.int32)
    empty_indices = torch.empty((0,), device="cuda", dtype=torch.int32)
    wrapper.plan(
        empty_indptr,
        empty_indices,
        1,
        64,
        128,
        2,
        2,
        _HEAD_DIM,
        64,
        64,
    )
    new_plan_output = wrapper.run(q, k, v)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        new_plan_output,
        torch.zeros_like(new_plan_output),
        rtol=0,
        atol=0,
    )
    assert dict(wrapper._policy)["max_execution_tiles"] == 0
    assert not torch.equal(new_plan_output, expected)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_dynamic_metadata_cuda_graph_replays_updated_routes_and_token_mask() -> None:
    torch.manual_seed(20260721)
    case = _Case(
        "dynamic_metadata_cuda_graph",
        1,
        1,
        16,
        225,
        16,
        16,
        torch.float16,
        "dense",
        "holey",
        "static",
    )
    initial_patterns: _Patterns = (((((0, 1, 2),)),),)
    replay_patterns: _Patterns = (((((0, 8, 12),)),),)
    initial_valid = frozenset(range(case.seq_len_kv))
    replay_valid = frozenset(
        token_idx
        for token_idx in range(case.seq_len_kv)
        if token_idx < 128 or token_idx % 7 not in (0, 3)
    )

    block_indptr, block_indices = _make_bsr(initial_patterns)
    valid_bits = _pack_single_batch_token_mask(case.seq_len_kv, initial_valid)
    replay_valid_bits = _pack_single_batch_token_mask(case.seq_len_kv, replay_valid)
    inspection = _inspect_block_sparse_bsr(
        block_indptr,
        block_indices,
        batch_size=case.batch_size,
        num_kv_heads=case.num_heads,
        seq_len_q=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        q_block_size=case.q_block_size,
        kv_block_size=case.kv_block_size,
        kv_valid_bits=valid_bits,
    )
    assert inspection.token_mask_has_holes is False

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
        q_data_type=case.dtype,
        dynamic_metadata=True,
    )
    state = wrapper._plan_state
    assert state is not None
    assert state.revision == 0
    assert dict(state.policy)["use_kv_valid_bits"] is True
    assert state.block_indices is block_indices
    assert state.runtime_kv_valid_bits is valid_bits
    indices_pointer = block_indices.data_ptr()
    mask_pointer = valid_bits.data_ptr()

    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    assert captured_result is graph_out
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)

    replay_indices = torch.tensor([0, 8, 12], device="cuda", dtype=torch.int32)
    block_indices.copy_(replay_indices)
    valid_bits.copy_(replay_valid_bits)
    assert block_indices.data_ptr() == indices_pointer
    assert valid_bits.data_ptr() == mask_pointer
    assert block_indices.tolist() == [0, 8, 12]
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(graph_out, replay_expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("q_block_size", (64, 128), ids=("q64", "q128"))
@torch.no_grad()
def test_dynamic_keeps_route_full_guard_replays_token_mask(
    q_block_size: int,
) -> None:
    """Exercise runtime full/partial routes without weakening causal/tail masks."""

    torch.manual_seed(20260723 + q_block_size)
    case = _Case(
        "dynamic_keeps_route_full_guard",
        1,
        1,
        q_block_size,
        289,
        q_block_size,
        128,
        torch.float16,
        "causal",
        "holey",
        "static",
        pattern="full",
    )
    patterns: _Patterns = (((((0, 1, 2),)),),)
    initial_valid = frozenset(
        token_idx for token_idx in range(case.seq_len_kv) if token_idx != 160
    )
    replay_valid = frozenset(
        token_idx for token_idx in range(case.seq_len_kv) if token_idx != 32
    )

    block_indptr, block_indices = _make_bsr(patterns)
    valid_bits = _pack_single_batch_token_mask(case.seq_len_kv, initial_valid)
    replay_valid_bits = _pack_single_batch_token_mask(case.seq_len_kv, replay_valid)
    q = torch.randn((1, case.seq_len_q, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, case.seq_len_kv, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    initial_expected = _reference(case, q, k, v, patterns, (initial_valid,), sm_scale)
    replay_expected = _reference(case, q, k, v, patterns, (replay_valid,), sm_scale)

    wrapper = block_sparse_module.BlockSparseTSWrapper()
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
        dynamic_metadata=True,
    )
    state = wrapper._plan_state
    assert state is not None
    assert state.revision == 0
    assert state.runtime_kv_valid_bits is valid_bits
    policy = dict(state.policy)
    assert policy["use_kv_valid_bits"] is True

    graph_out = torch.empty_like(q)
    wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = wrapper.run(q, k, v, sm_scale=sm_scale, out=graph_out)
    assert captured_result is graph_out
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, initial_expected, rtol=1e-2, atol=1e-2)

    mask_pointer = valid_bits.data_ptr()
    valid_bits.copy_(replay_valid_bits)
    assert valid_bits.data_ptr() == mask_pointer
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(graph_out, replay_expected, rtol=1e-2, atol=1e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_dynamic_metadata_sizes_execution_tiles_for_untrimmed_replay() -> None:
    # A kv_block_size=192 block expands into three KV64 atoms, and the last
    # block of seq_len_kv=448 keeps only one atom after physical-tail
    # trimming. A dynamic plan must size max_execution_tiles for the
    # untrimmed indptr bound, because a later in-place index update may
    # avoid the tail and legitimately need more execution routes than the
    # plan-time data measured.
    torch.manual_seed(20260722)
    case = _Case(
        "dynamic_metadata_untrimmed_replay",
        1,
        1,
        64,
        448,
        64,
        192,
        torch.float16,
        "dense",
        "none",
        "static",
    )
    initial_patterns: _Patterns = ((((1, 2),),),)
    replay_patterns: _Patterns = ((((0, 1),),),)
    all_valid = frozenset(range(case.seq_len_kv))

    block_indptr, block_indices = _make_bsr(initial_patterns)
    q = torch.randn((1, case.seq_len_q, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    k = torch.randn((1, case.seq_len_kv, 1, _HEAD_DIM), device="cuda", dtype=case.dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    initial_expected = _reference(
        case, q, k, v, initial_patterns, (all_valid,), sm_scale
    )
    replay_expected = _reference(case, q, k, v, replay_patterns, (all_valid,), sm_scale)

    wrapper = block_sparse_module.BlockSparseTSWrapper()
    wrapper.plan(
        block_indptr,
        block_indices,
        1,
        case.seq_len_q,
        case.seq_len_kv,
        1,
        1,
        _HEAD_DIM,
        case.q_block_size,
        case.kv_block_size,
        q_data_type=case.dtype,
        dynamic_metadata=True,
    )
    # Untrimmed bound: ceil(max_row_nnz * kv_block_size / 128) = 3 routes,
    # while the plan-time data measures only 2 after tail trimming.
    assert dict(wrapper._policy)["max_execution_tiles"] == 3

    initial_actual = wrapper.run(q, k, v, sm_scale=sm_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(initial_actual, initial_expected, rtol=1e-2, atol=1e-2)

    block_indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    replay_actual = wrapper.run(q, k, v, sm_scale=sm_scale)
    torch.cuda.synchronize()
    assert not torch.equal(replay_expected, initial_expected)
    torch.testing.assert_close(replay_actual, replay_expected, rtol=1e-2, atol=1e-2)
