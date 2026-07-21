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
from flashinfer.attention.prims_ts._block_sparse.inspection import (
    _BlockSparseInspection,
    _inspect_block_sparse_bsr,
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
    _resolve_block_sparse_route_origins_host,
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
        "q256_kv64_fp16_empty_row_static",
        1,
        1,
        385,
        257,
        256,
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


def _inspection(*, token_mask_has_holes: bool = False) -> _BlockSparseInspection:
    return _BlockSparseInspection(
        max_row_nnz=1,
        max_retained_routes=1,
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
        *,
        use_token_word_full_guard: bool = False,
        use_q128_token_route_full_guard: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(use_persistent_scheduler=False),
            policy=(
                ("tile_size_q", q_tile_size),
                ("use_persistent_scheduler", False),
                ("execution_path", "raw_bsr_decode"),
                ("use_kv_valid_bits", use_token_mask),
                (
                    "use_token_word_full_guard",
                    use_token_word_full_guard,
                ),
                (
                    "use_q128_token_route_full_guard",
                    use_q128_token_route_full_guard,
                ),
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


def _make_task_manager_config(
    *,
    q_tile_size: int,
    persistent: bool,
) -> FmhaDecodeConfig:
    return make_decode_config(
        headdim=_HEAD_DIM,
        args={
            "use_keeps_mma_ab": True,
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


@pytest.mark.parametrize("parameter_name", ("q_block_size", "kv_block_size"))
@pytest.mark.parametrize(
    ("invalid_value", "error_type"),
    (
        pytest.param(True, TypeError, id="bool"),
        pytest.param(64.0, TypeError, id="float"),
        pytest.param(0, ValueError, id="zero"),
        pytest.param(96, ValueError, id="not-multiple-of-64"),
        pytest.param(1 << 32, OverflowError, id="int32-overflow"),
    ),
)
def test_plan_requires_positive_64_multiple_block_sizes(
    parameter_name: str,
    invalid_value: object,
    error_type: type[Exception],
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

    with pytest.raises(error_type, match="positive multiple of 64"):
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
    ("indices", "reason"),
    (
        pytest.param((1, 0), "strictly increasing", id="unsorted"),
        pytest.param((0, 0), "strictly increasing", id="duplicate"),
        pytest.param((0, 4), "in-range KV block", id="out-of-bounds"),
    ),
)
def test_inspector_rejects_noncanonical_bsr(
    indices: tuple[int, ...],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=rf"canonical BSR.*{reason}"):
        _inspect([[[0, 2]]], indices)
    torch.cuda.synchronize()


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    (
        "seq_len_kv",
        "kv_block_size",
        "indices",
        "expected_nnz",
        "expected_routes",
    ),
    (
        pytest.param(193, 64, (0, 2, 3), 3, 2, id="kv64"),
        pytest.param(385, 128, (0, 3), 2, 2, id="kv128"),
        pytest.param(641, 256, (0, 2), 2, 4, id="kv256"),
    ),
)
def test_inspector_reports_tail_trimmed_route_summary(
    seq_len_kv: int,
    kv_block_size: int,
    indices: tuple[int, ...],
    expected_nnz: int,
    expected_routes: int,
) -> None:
    result = _inspect(
        [[[0, len(indices)]]],
        indices,
        seq_len_kv=seq_len_kv,
        kv_block_size=kv_block_size,
    )

    assert result.max_row_nnz == expected_nnz
    assert result.max_retained_routes == expected_routes


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("words", "expected_full_slots"),
    (
        pytest.param(
            (0xAAAAAAAA, 0x55555555, 0x33333333, 0xCCCCCCCC),
            0,
            id="random-like",
        ),
        pytest.param(
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000),
            3,
            id="full-word-suffix",
        ),
    ),
)
def test_inspector_reports_observable_token_morphology(
    words: tuple[int, ...],
    expected_full_slots: int,
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

    assert result.token_mask_has_holes is True
    assert result.runtime_token_guard_skip_count == expected_full_slots
    assert result.runtime_token_guard_check_count == 8
    assert result.runtime_token_mask_full_route_count == 0


@pytest.mark.parametrize(
    ("kv_block_size", "seq_len_kv", "selected", "expected"),
    (
        pytest.param(
            64,
            193,
            (0, 2, 3),
            (((0, True), (128, True)), ((192, True), (0, False))),
            id="kv64-nonadjacent-tail",
        ),
        pytest.param(
            128,
            385,
            (0, 3),
            (((0, True), (64, True)), ((384, True), (0, False))),
            id="kv128-tail",
        ),
        pytest.param(
            256,
            641,
            (0, 2),
            (
                ((0, True), (64, True)),
                ((128, True), (192, True)),
                ((512, True), (576, True)),
                ((640, True), (0, False)),
            ),
            id="kv256-expansion-tail",
        ),
    ),
)
def test_host_route_mapping_for_supported_kv_blocks(
    kv_block_size: int,
    seq_len_kv: int,
    selected: tuple[int, ...],
    expected: tuple[tuple[tuple[int, bool], tuple[int, bool]], ...],
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
    actual = tuple(
        tuple((fragment.physical_token_offset, fragment.valid) for fragment in route)
        for route in routes
    )

    assert actual == expected


@pytest.mark.parametrize("q_tile_size", (64, 128), ids=("q64", "q128"))
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
        lambda **_kwargs: case.scheduler,
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
    assert policy["tile_size_q"] == (128 if case.q_block_size % 128 == 0 else 64)
    assert policy["use_persistent_scheduler"] is (case.scheduler == "persistent")
    has_runtime_holes = case.token_mask == "holey"
    assert policy["use_kv_valid_bits"] is has_runtime_holes
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
    sparse_indptr = torch.tensor(
        [[[0, 1], [1, 2]]],
        device="cuda",
        dtype=torch.int32,
    )
    sparse_indices = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
    wrapper.plan(
        sparse_indptr,
        sparse_indices,
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
    selected_k = k[:, 64:128].transpose(1, 2).float()
    selected_v = v[:, 64:128].transpose(1, 2).float()
    scores = q.transpose(1, 2).float() @ selected_k.transpose(-1, -2)
    probabilities = torch.softmax(scores / math.sqrt(_HEAD_DIM), dim=-1)
    new_plan_expected = (probabilities @ selected_v).transpose(1, 2).to(q.dtype)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        new_plan_output,
        new_plan_expected,
        rtol=1e-2,
        atol=1e-2,
    )
    assert not torch.equal(new_plan_output, expected)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)
