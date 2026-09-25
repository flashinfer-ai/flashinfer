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

"""Sage attention (8-bit Q/K/V with dequantization scales) for PrimTS attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import (
    BlockSparseTSWrapper,
    SageAttentionConfig,
    SageAttentionParams,
    block_sparse_attention,
)
from flashinfer.attention.prims_ts._block_sparse import config as sparse_config

from tests.attention.test_attention_ts_block_sparse import (
    _HEAD_DIM,
    _REQUIRES_PRIMTS_GPU,
    _make_bsr,
    _make_exact_block_bits,
    _pack_token_mask,
)

_FP8 = torch.float8_e4m3fn
# The kernel scales probabilities to the E4M3 maximum before quantizing them.
_P_SCALE = 448.0
# INT8 Q/K are drawn with this standard deviation and their scales divided by
# it, so dequantized INT8 and E4M3 inputs share one distribution.
_INT8_STD = 40.0
# Reciprocal power-of-two factors on the Q and K scales leave every logit
# unchanged, so the output must not depend on either scale's magnitude.
_SCALE_SHIFT = 2.0**24


@pytest.fixture(autouse=True)
def _ieee_float32_references(monkeypatch: pytest.MonkeyPatch):
    """Keep FP32 references independent of the environment's TF32 preference."""
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous_precision)


@dataclass(frozen=True)
class _SageCase:
    """One Sage problem: geometry, route mode, recipe and launch path."""

    name: str
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    q_block_size: int
    kv_block_size: int
    # 256 selects the Q64/KV256 profile, 128 the Q128/KV128 one.
    kv_tile: int
    qk_dtype: torch.dtype = _FP8
    out_dtype: torch.dtype = torch.bfloat16
    sage: SageAttentionConfig = SageAttentionConfig()
    mask_type: str = "dense"
    persistent: bool = False
    # "dense" plans use_block_sparse=False; "exact" and "proxy" route blocks.
    routes: str = "dense"
    sparse_format: str = "bsr"
    use_token_mask: bool = False
    one_shot: bool = False
    # Batch 1 takes Q scales near zero: masked INT8 lanes must still carry no
    # mass when ``sfQ * sfK`` is tiny.
    tiny_q_scales: bool = False

    @property
    def num_kv_blocks(self) -> int:
        return -(-self.seq_len_kv // self.kv_block_size)


_KV256 = dict(
    batch_size=2,
    seq_len_q=128,
    seq_len_kv=1000,
    num_qo_heads=2,
    num_kv_heads=2,
    q_block_size=64,
    kv_block_size=64,
    kv_tile=256,
)
# Eight Q heads per KV head fill the Q128 tile with 16 tokens.
_Q128 = dict(
    batch_size=2,
    seq_len_q=48,
    seq_len_kv=500,
    num_qo_heads=16,
    num_kv_heads=2,
    q_block_size=16,
    kv_block_size=64,
    kv_tile=128,
)

_DENSE_CASES = (
    _SageCase(
        "kv256_fp8_k1_one_shot",
        **_KV256,
        sage=SageAttentionConfig(k_block_size=1),
        one_shot=True,
    ),
    _SageCase(
        "kv256_int8_k4_q16_mean_causal_persistent",
        **_KV256,
        qk_dtype=torch.int8,
        out_dtype=torch.float16,
        sage=SageAttentionConfig(q_block_size=16, k_block_size=4, v_mean=True),
        mask_type="causal",
        persistent=True,
    ),
    # The one-shot recipe defaults to (1, 16) with a V mean read off the scales.
    _SageCase(
        "q128_fp8_k16_mean_causal_one_shot",
        **_Q128,
        sage=SageAttentionConfig(v_mean=True),
        mask_type="causal",
        one_shot=True,
    ),
    _SageCase(
        "q128_int8_k1_q4_persistent",
        **_Q128,
        qk_dtype=torch.int8,
        out_dtype=torch.float16,
        sage=SageAttentionConfig(q_block_size=4, k_block_size=1),
        persistent=True,
    ),
)
_SPARSE_CASES = (
    _SageCase(
        "q128_exact_fp8_k16_mean_token_mask",
        **_Q128,
        sage=SageAttentionConfig(v_mean=True),
        routes="exact",
        use_token_mask=True,
    ),
    _SageCase(
        "kv256_exact_int8_bk128_k4_q16_bitmask_token_mask_persistent",
        **{**_KV256, "kv_block_size": 128},
        qk_dtype=torch.int8,
        out_dtype=torch.float16,
        sage=SageAttentionConfig(q_block_size=16, k_block_size=4),
        routes="exact",
        sparse_format="bitmask",
        use_token_mask=True,
        persistent=True,
        tiny_q_scales=True,
    ),
    # Exact routes read 16-token K scales, summaries one-token scales.
    _SageCase(
        "kv256_proxy_fp8_k16_s1_mean_bitmask",
        **_KV256,
        sage=SageAttentionConfig(v_mean=True, k_summary_block_size=1),
        routes="proxy",
        sparse_format="bitmask",
    ),
    # 33 summaries span three 16-summary scale blocks per batch.
    _SageCase(
        "q128_proxy_int8_k4_s16_persistent",
        **{**_Q128, "seq_len_kv": 2100},
        qk_dtype=torch.int8,
        out_dtype=torch.float16,
        sage=SageAttentionConfig(k_block_size=4, k_summary_block_size=16),
        routes="proxy",
        persistent=True,
    ),
)


def _scale_slots(batch_size: int, seq_len: int, block_size: int) -> torch.Tensor:
    """Return the trtllm-gen flat scale slot ``b * S // blk + b + t // blk`` ``[B, S]``."""

    batch = torch.arange(batch_size, device="cuda")[:, None]
    token = torch.arange(seq_len, device="cuda")
    return batch * seq_len // block_size + batch + token // block_size


def _random_scales(heads, batch_size, seq_len, block_size, low, high) -> torch.Tensor:
    """Uniform ``[low, high)`` scales of ``ceil(B * S / blk) + B - 1`` slots per head."""

    numel = -(-batch_size * seq_len // block_size) + batch_size - 1
    return torch.empty((heads, numel), device="cuda").uniform_(low, high)


def _random_qk(shape, dtype: torch.dtype) -> torch.Tensor:
    values = torch.randn(shape, device="cuda")
    if dtype == torch.int8:
        return (values * _INT8_STD).round().clamp(-127, 127).to(torch.int8)
    return values.to(dtype)


def _random_inputs(case: _SageCase):
    """Random 8-bit Q/K, E4M3 V, proxy summaries and positive scales."""

    unit = 1.0 / _INT8_STD if case.qk_dtype == torch.int8 else 1.0
    batch, hq, hkv = case.batch_size, case.num_qo_heads, case.num_kv_heads
    q = _random_qk((batch, case.seq_len_q, hq, _HEAD_DIM), case.qk_dtype)
    k = _random_qk((batch, case.seq_len_kv, hkv, _HEAD_DIM), case.qk_dtype)
    v = torch.randn(k.shape, device="cuda").to(_FP8)
    q_scale = _random_scales(
        hq, batch, case.seq_len_q, case.sage.q_block_size, 0.1 * unit, 0.4 * unit
    )
    if case.tiny_q_scales:
        second_batch = _scale_slots(batch, case.seq_len_q, case.sage.q_block_size)[1]
        q_scale[:, second_batch] *= 1e-8
    k_scale = _random_scales(
        hkv, batch, case.seq_len_kv, case.sage.k_block_size, 0.5 * unit, 2.0 * unit
    )
    params = SageAttentionParams(
        q_scale=q_scale / _SCALE_SHIFT,
        k_scale=k_scale * _SCALE_SHIFT,
        v_scale=torch.empty((hkv, _HEAD_DIM), device="cuda").uniform_(0.25, 1.0),
        v_mean=(
            torch.randn((hkv, _HEAD_DIM), device="cuda") if case.sage.v_mean else None
        ),
    )
    summaries = None
    if case.routes == "proxy":
        shape = (batch, case.num_kv_blocks, hkv, _HEAD_DIM)
        summaries = (
            _random_qk(shape, case.qk_dtype),
            torch.randn(shape, device="cuda").to(_FP8),
        )
        k_summary_scale = _random_scales(
            hkv,
            batch,
            case.num_kv_blocks,
            case.sage.summary_k_block_size,
            0.5 * unit,
            2.0 * unit,
        )
        params = replace(params, k_summary_scale=k_summary_scale * _SCALE_SHIFT)
    return q, k, v, params, summaries


def _random_patterns(case: _SageCase, generator: torch.Generator):
    """Exact blocks per (batch, KV head, Q row).

    Odd rows keep the ragged final block; the first row of an exact case has
    no route, so its output stays zero despite a V mean.
    """

    def row(batch_idx: int, head_idx: int, row_idx: int) -> tuple[int, ...]:
        if case.routes == "exact" and batch_idx == head_idx == row_idx == 0:
            return ()
        count = 1 + int(torch.randint(0, case.num_kv_blocks, (1,), generator=generator))
        blocks = torch.randperm(case.num_kv_blocks, generator=generator)[:count]
        selected = set(blocks.tolist())
        if row_idx % 2:
            selected.add(case.num_kv_blocks - 1)
        return tuple(sorted(selected))

    num_rows = -(-case.seq_len_q // case.q_block_size)
    return tuple(
        tuple(
            tuple(row(batch_idx, head_idx, row_idx) for row_idx in range(num_rows))
            for head_idx in range(case.num_kv_heads)
        )
        for batch_idx in range(case.batch_size)
    )


def _widest_row(patterns) -> int:
    return max(len(row) for batch in patterns for head in batch for row in head)


def _token_mask(case: _SageCase) -> torch.Tensor:
    """Boolean ``[B, Skv]`` validity that drops two of every seven tokens."""

    shifted = torch.arange(case.seq_len_kv, device="cuda") + torch.arange(
        case.batch_size, device="cuda"
    ).unsqueeze(1)
    return (shifted % 7 != 0) & (shifted % 7 != 3)


def _routing(case: _SageCase, patterns, summaries, valid) -> dict[str, torch.Tensor]:
    """Return the ``run`` routing arguments of one sparse case."""

    if case.routes == "dense":
        return {}
    if case.sparse_format == "bsr":
        block_indptr, block_indices = _make_bsr(patterns)
        routing = {"block_indptr": block_indptr, "block_indices": block_indices}
    else:
        routing = {
            "exact_block_bits": _make_exact_block_bits(patterns, case.num_kv_blocks)
        }
    if summaries is not None:
        routing.update(k_summary=summaries[0], v_summary=summaries[1])
    if valid is not None:
        routing["kv_valid_bits"] = _pack_token_mask(
            case.seq_len_kv,
            tuple(frozenset(row.nonzero().flatten().tolist()) for row in valid),
        )
    return routing


def _plan(case: _SageCase, *, max_blocks_per_row: int | None = None):
    """Plan ``case`` on a fresh wrapper and check the published tile policy."""

    wrapper = BlockSparseTSWrapper()
    route_kwargs = {}
    if case.routes != "dense":
        route_kwargs = dict(
            max_blocks_per_row=max_blocks_per_row,
            use_kv_valid_bits=case.use_token_mask,
            sparse_format=case.sparse_format,
            use_proxy_routes=case.routes == "proxy",
        )
    select_scheduler = sparse_config._select_block_sparse_scheduler
    sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            if case.persistent:
                # The test shapes are too small for the planner to pick the
                # persistent scheduler; keep its Q tile and force the choice.
                monkeypatch.setattr(
                    sparse_config,
                    "_select_block_sparse_scheduler",
                    lambda **kwargs: (select_scheduler(**kwargs)[0], True),
                )
            wrapper.plan(
                case.batch_size,
                case.seq_len_q,
                case.seq_len_kv,
                case.num_qo_heads,
                case.num_kv_heads,
                _HEAD_DIM,
                case.q_block_size,
                case.kv_block_size,
                device=torch.device("cuda", 0),
                use_block_sparse=case.routes != "dense",
                mask_type=case.mask_type,
                q_data_type=case.qk_dtype,
                # E4M3 Q/K leave V to follow K; INT8 Q/K name their E4M3 V.
                v_data_type=_FP8 if case.qk_dtype == torch.int8 else None,
                o_data_type=case.out_dtype,
                sage_config=case.sage,
                **route_kwargs,
            )
    finally:
        sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == (64 if case.kv_tile == 256 else 128)
    assert policy["tile_size_kv"] == case.kv_tile
    assert policy["scheduler"] == ("persistent" if case.persistent else "static")
    return wrapper


def _run(case: _SageCase, q, k, v, params, *, sm_scale, patterns, summaries, valid):
    """Run ``case`` through its public entry point, synchronized."""

    if case.one_shot:
        default_recipe = SageAttentionConfig(v_mean=case.sage.v_mean)
        actual = block_sparse_attention(
            q,
            k,
            v,
            None,
            None,
            case.q_block_size,
            case.kv_block_size,
            use_block_sparse=False,
            mask_type=case.mask_type,
            sm_scale=sm_scale,
            sage=params,
            sage_config=None if case.sage == default_recipe else case.sage,
        )
        torch.cuda.synchronize()
        return actual
    wrapper = _plan(
        case, max_blocks_per_row=None if patterns is None else _widest_row(patterns)
    )
    routing = _routing(case, patterns, summaries, valid)
    actual = wrapper.run(q, k, v, sm_scale=sm_scale, sage=params, **routing)
    unchecked = wrapper.run(
        q, k, v, sm_scale=sm_scale, sage=params, validate=False, **routing
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, unchecked)
    return actual


def _dequantize(x: torch.Tensor, scales: torch.Tensor, block_size: int):
    """Return FP32 ``[B, S, H, D]`` values of 8-bit data under flat-layout scales."""

    slots = _scale_slots(x.shape[0], x.shape[1], block_size)
    return x.float() * scales[:, slots].permute(1, 2, 0).unsqueeze(-1)


def _row_folds(case: _SageCase, exact_blocks):
    """Yield the ``(stream, source, indices)`` column folds of one Q row in kernel order.

    A route (a dense tile) packs ``kv_tile`` tokens of the exact blocks or
    ``kv_tile`` summaries, in K64 atoms; exact routes come first. K/V instance
    ``route % 2`` owns a route and, on KV256, spatial half ``atom % 2`` splits
    it into two streams, each folding its half of the route at once.
    """

    kv_block, atoms_per_route = case.kv_block_size, case.kv_tile // 64
    token_atoms = [
        range(begin, min(begin + 64, case.seq_len_kv))
        for block in exact_blocks
        for begin in range(block * kv_block, (block + 1) * kv_block, 64)
    ]
    routes = [
        ("token", token_atoms[begin : begin + atoms_per_route])
        for begin in range(0, len(token_atoms), atoms_per_route)
    ]
    if case.routes == "proxy":
        exact = set(exact_blocks)
        routes += [
            (
                "summary",
                [
                    [
                        index
                        for index in range(begin, min(begin + 64, case.num_kv_blocks))
                        if index not in exact
                    ]
                    for begin in range(first, first + case.kv_tile, 64)
                ],
            )
            for first in range(0, case.num_kv_blocks, case.kv_tile)
        ]
    halves = 2 if case.kv_tile == 256 else 1
    for route_idx, (source, atoms) in enumerate(routes):
        for half in range(halves):
            indices = [index for atom in atoms[half::halves] for index in atom]
            if indices:
                stream = route_idx % 2 * halves + half
                yield stream, source, torch.tensor(indices, device="cuda")


def _fold(state, logits: torch.Tensor, values: torch.Tensor):
    """Fold one set of columns into an online-softmax stream ``(max, sum, acc)``.

    The row sum takes FP32 probabilities; the PV operand quantizes them to
    E4M3 at ``448`` times their ratio to the stream's running maximum.
    """

    if state is None:
        total = torch.zeros(logits.shape[:-1], device="cuda")
        acc = torch.zeros((*logits.shape[:-1], values.shape[-1]), device="cuda")
        state = (total - float("inf"), total, acc)
    running_max, total, acc = state
    new_max = torch.maximum(running_max, logits.amax(dim=-1))
    anchor = torch.where(new_max.isfinite(), new_max, 0.0)
    probabilities = torch.exp(logits - anchor.unsqueeze(-1)) * _P_SCALE
    correction = torch.exp(running_max - anchor)
    return (
        new_max,
        total * correction + probabilities.sum(dim=-1),
        acc * correction.unsqueeze(-1) + probabilities.to(_FP8).float() @ values,
    )


@torch.no_grad()
def _reference(
    case: _SageCase, q, k, v, params, *, sm_scale, patterns, summaries, valid
):
    """FP32 attention on the dequantized inputs with the kernel's stream model.

    Every online-softmax stream quantizes its probabilities against its own
    running maximum; a proxy summary of ``mass`` tokens weighs its
    probability by that mass. Rows without visible mass stay zero.
    """

    q_real = _dequantize(q, params.q_scale, case.sage.q_block_size)
    k_real = _dequantize(k, params.k_scale, case.sage.k_block_size)
    if summaries is not None:
        k_summary = _dequantize(
            summaries[0], params.k_summary_scale, case.sage.summary_k_block_size
        )
        block_begins = torch.arange(case.num_kv_blocks, device="cuda")
        masses = (case.seq_len_kv - block_begins * case.kv_block_size).clamp(
            max=case.kv_block_size
        )
    group = case.num_qo_heads // case.num_kv_heads
    row_size = case.seq_len_q if patterns is None else case.q_block_size
    # The last key each query sees under the causal mask.
    last_keys = torch.arange(case.seq_len_q, device="cuda") + (
        case.seq_len_kv - case.seq_len_q
    )
    output = torch.zeros(q.shape, device="cuda")
    for batch_idx in range(case.batch_size):
        for head_idx in range(case.num_kv_heads):
            heads = slice(head_idx * group, (head_idx + 1) * group)
            for row_idx, begin in enumerate(range(0, case.seq_len_q, row_size)):
                rows = slice(begin, begin + row_size)
                row_last_keys = last_keys[rows, None]
                exact_blocks = (
                    range(case.num_kv_blocks)
                    if patterns is None
                    else patterns[batch_idx][head_idx][row_idx]
                )
                streams = {}
                for stream, source, index in _row_folds(case, exact_blocks):
                    visible = torch.ones(
                        (len(row_last_keys), len(index)),
                        dtype=torch.bool,
                        device="cuda",
                    )
                    if source == "token":
                        keys = k_real[batch_idx, index, head_idx]
                        values = v[batch_idx, index, head_idx]
                        bias = 0.0
                        if valid is not None:
                            visible &= valid[batch_idx, index]
                        if case.mask_type == "causal":
                            visible &= row_last_keys >= index
                    else:
                        keys = k_summary[batch_idx, index, head_idx]
                        values = summaries[1][batch_idx, index, head_idx]
                        bias = masses[index].float().log()
                    logits = torch.einsum(
                        "tgd,cd->tgc", q_real[batch_idx, rows, heads], keys
                    )
                    logits = (logits * sm_scale + bias).masked_fill(
                        ~visible.unsqueeze(1), float("-inf")
                    )
                    streams[stream] = _fold(streams.get(stream), logits, values.float())
                if not streams:
                    continue
                maxima, totals, accs = (
                    torch.stack(part) for part in zip(*streams.values(), strict=True)
                )
                final_max = maxima.amax(dim=0)
                weights = torch.exp(
                    maxima - torch.where(final_max.isfinite(), final_max, 0.0)
                )
                total = (weights * totals).sum(dim=0).unsqueeze(-1)
                result = (weights.unsqueeze(-1) * accs).sum(dim=0) / total
                result = result * params.v_scale[head_idx]
                if params.v_mean is not None:
                    result = result + params.v_mean[head_idx]
                output[batch_idx, rows, heads] = torch.where(total > 0, result, 0.0)
    return output


def _check_case(case: _SageCase, q, k, v, params, summaries) -> None:
    """Run ``case`` on the given inputs and compare it with the reference."""

    patterns = None
    if case.routes != "dense":
        patterns = _random_patterns(case, torch.Generator().manual_seed(20260908))
    valid = _token_mask(case) if case.use_token_mask else None
    arguments = dict(
        sm_scale=_HEAD_DIM**-0.5, patterns=patterns, summaries=summaries, valid=valid
    )
    expected = _reference(case, q, k, v, params, **arguments)
    actual = _run(case, q, k, v, params, **arguments)
    assert actual.dtype == case.out_dtype
    assert torch.isfinite(actual).all()
    # Sparse rows attend to few tokens, so the kernel's emulated exponentials
    # move single E4M3 probability steps visibly into their output.
    if case.routes == "dense":
        torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)
    else:
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "case", _DENSE_CASES + _SPARSE_CASES, ids=lambda case: case.name
)
@torch.no_grad()
def test_sage_matches_dequantized_reference(case: _SageCase) -> None:
    """Dense and block-sparse Sage launches match FP32 attention on dequantized inputs."""

    torch.manual_seed(20260908)
    _check_case(case, *_random_inputs(case))


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_dense_int8_extreme_scores_stay_in_the_bias_binade() -> None:
    """The extreme INT8 dot products ``+2**21`` and ``-128 * 127 * 128`` stay exact.

    INT32 scores become FP32 after accumulating onto ``1.5 * 2**23``, which
    holds only while every ``bias + score`` stays inside ``[2**23, 2**24)``.
    """

    case = _DENSE_CASES[1]
    assert case.qk_dtype == torch.int8
    torch.manual_seed(20260909)
    q, k, v, params, summaries = _random_inputs(case)
    batch_idx, q_token, k_max_token, k_min_token = 0, 5, 300, 700
    q[batch_idx, q_token] = -128
    k[batch_idx, k_max_token] = -128
    k[batch_idx, k_min_token] = 127
    # These scales put both extreme logits near +-2, so the two keys carry
    # visible weight in the softmax row.
    q_slots = _scale_slots(case.batch_size, case.seq_len_q, case.sage.q_block_size)
    k_slots = _scale_slots(case.batch_size, case.seq_len_kv, case.sage.k_block_size)
    params.q_scale[:, q_slots[batch_idx, q_token]] = 1e-3 / _SCALE_SHIFT
    params.k_scale[:, k_slots[batch_idx, [k_max_token, k_min_token]]] = (
        1e-2 * _SCALE_SHIFT
    )
    _check_case(case, q, k, v, params, summaries)


@_REQUIRES_PRIMTS_GPU
def test_int8_qk_plans_only_on_sm100(monkeypatch: pytest.MonkeyPatch) -> None:
    """INT8 Q/K need the INT8 tcgen05 MMA, which later architectures drop."""

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (10, 3)
    )
    with pytest.raises(NotImplementedError, match="SM100a"):
        _plan(_DENSE_CASES[1])


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        pytest.param(
            {"sage_config": None}, NotImplementedError, "(?i)float16", id="no-recipe"
        ),
        pytest.param(
            {"kv_data_type": torch.int8}, ValueError, "one dtype", id="mixed-qk"
        ),
        pytest.param(
            {"q_data_type": torch.bfloat16}, ValueError, "Int8 Q and K", id="16-bit-qk"
        ),
        pytest.param(
            {"v_data_type": torch.bfloat16}, ValueError, "E4M3FN V", id="16-bit-v"
        ),
        pytest.param(
            {"sage_config": SageAttentionConfig(k_block_size=8)},
            ValueError,
            "k_block_size",
            id="k-block-8",
        ),
        pytest.param(
            {"q_block_size": 8, "kv_block_size": 8},
            ValueError,
            "Keeps profile",
            id="swap-profile",
        ),
    ),
)
def test_sage_plan_rejects_unsupported_recipes(
    overrides: dict[str, object], error: type[Exception], match: str
) -> None:
    """A Sage plan takes 8-bit Q/K of one dtype, E4M3 V, a supported recipe and a Keeps profile."""

    arguments = dict(
        batch_size=1,
        seq_len_q=64,
        seq_len_kv=256,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=64,
        kv_block_size=64,
        device=torch.device("cuda", 0),
        use_block_sparse=False,
        q_data_type=_FP8,
        sage_config=SageAttentionConfig(),
    )
    with pytest.raises(error, match=match):
        BlockSparseTSWrapper().plan(**{**arguments, **overrides})


def _sage_decode_config(
    tile_size_q: int = 64,
    qk_dtype_name: str = "Float8E4M3FN",
    k_block_size: int = 16,
    args: dict[str, object] | None = None,
    **overrides: object,
):
    """Build a dense contiguous Sage decode configuration at the kernel layer."""

    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    qk_dtype = getattr(cutlass, qk_dtype_name)
    heads_q_per_kv = 1 if tile_size_q == 64 else 8
    return make_decode_config(
        headdim=_HEAD_DIM,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": tile_size_q,
            "tile_size_kv": 256 if tile_size_q == 64 else 128,
            "groups_tokens_heads_q": True,
            "sage_k_block_size": k_block_size,
            "sage_k_summary_block_size": k_block_size,
            "sage_q_block_size": 1,
            **(args or {}),
        },
        seq_len_q=64 if tile_size_q == 64 else 16,
        seq_len_kv=1000,
        batch_size=2,
        num_heads_q=8,
        num_heads_kv=8 // heads_q_per_kv,
        q_dtype=qk_dtype,
        k_dtype=qk_dtype,
        v_dtype=cutlass.Float8E4M3FN,
        o_dtype=cutlass.BFloat16,
        mask_type="dense",
        auto_tuner=False,
        **overrides,
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        pytest.param(
            {
                "args": {"use_split_kv": True, "splits_kv": 2, "max_splits_kv": 2},
                "split_kv_mode": "gmem_reduction",
                "splits_kv": 2,
            },
            "direct-output grid",
            id="split-kv",
        ),
        pytest.param(
            {"qkv_layout": "pagedKv", "num_tokens_per_page": 128},
            "contiguous K/V",
            id="paged-kv",
        ),
    ),
)
def test_sage_decode_config_rejects_unsupported_launches(
    overrides: dict[str, object], match: str
) -> None:
    """Only the direct output store of contiguous K/V applies the V scales."""

    with pytest.raises(ValueError, match=match):
        _sage_decode_config(**overrides)


def _mixed_proxy_config():
    """A block-sparse proxy plan with 16-token K scales and 4-token summary scales."""

    spec = sparse_config._resolve_block_sparse_launch_spec(
        device_index=0,
        batch_size=1,
        seq_len_q=128,
        seq_len_kv=4096,
        num_qo_heads=8,
        num_kv_heads=8,
        share_pattern_across_kv_heads=False,
        head_dim=_HEAD_DIM,
        q_block_size=64,
        kv_block_size=64,
        kv_route_size=256,
        page_size=None,
        dtype_key="float8_e4m3fn",
        mask_type="dense",
        use_kv_valid_bits=False,
        max_row_route_capacity=16,
        sparse_format="bsr",
        use_proxy_routes=True,
        use_block_sparse=True,
        out_dtype_key="bfloat16",
        v_dtype_key="float8_e4m3fn",
        sage=SageAttentionConfig(k_block_size=16, k_summary_block_size=4),
    )
    return sparse_config._make_block_sparse_config(spec.compile_key)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    "make_config",
    (
        pytest.param(lambda: _sage_decode_config(), id="dense-fp8-k16"),
        pytest.param(
            lambda: _sage_decode_config(128, "Int8", k_block_size=4),
            id="dense-int8-q128-k4",
        ),
        pytest.param(lambda: _mixed_proxy_config(), id="block-sparse-mixed-proxy"),
    ),
)
def test_sage_schedule_passes_strict_validation(make_config) -> None:
    """The scale resources appear in the resource dependency graph."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        build_decode_task_manager,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            make_config(),
            seq_len_kv=1024,
            batch_size=1,
            num_heads_kv=1,
            verbose=False,
            skip_validation=False,
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("change", "match"),
    (
        pytest.param(lambda params: None, "required by a Sage plan", id="missing"),
        pytest.param(
            lambda params: replace(params, k_summary_scale=None),
            "k_summary_scale",
            id="no-summary-scale",
        ),
        pytest.param(
            lambda params: replace(params, v_mean=None), "v_mean", id="no-v-mean"
        ),
        pytest.param(
            lambda params: replace(params, k_scale=params.k_scale[:, 1:].contiguous()),
            "k_scale",
            id="k-scale-shape",
        ),
        pytest.param(
            lambda params: replace(params, q_scale=params.q_scale.half()),
            "float32",
            id="fp16-scales",
        ),
    ),
)
def test_sage_run_rejects_mismatched_scales(change, match: str) -> None:
    """Every run supplies the scale tensors of the planned recipe and route mode."""

    case = _SPARSE_CASES[2]
    assert case.routes == "proxy" and case.sage.v_mean
    q, k, v, params, summaries = _random_inputs(case)
    # The reference test's patterns keep the compiled launch shared with it.
    patterns = _random_patterns(case, torch.Generator().manual_seed(20260908))
    wrapper = _plan(case, max_blocks_per_row=_widest_row(patterns))
    with pytest.raises(ValueError, match=match):
        wrapper.run(
            q, k, v, sage=change(params), **_routing(case, patterns, summaries, None)
        )


def test_one_shot_sage_config_requires_the_scale_tensors() -> None:
    """A one-shot recipe without scale tensors has nothing to run."""

    q = torch.empty((1, 64, 1, _HEAD_DIM), dtype=_FP8)
    with pytest.raises(ValueError, match="sage_config requires"):
        block_sparse_attention(
            q,
            q,
            q,
            None,
            None,
            64,
            64,
            use_block_sparse=False,
            sage_config=SageAttentionConfig(),
        )
