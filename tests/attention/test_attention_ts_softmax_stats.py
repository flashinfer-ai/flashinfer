# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The shared max/denominator export contract across PrimTS attention paths."""

from dataclasses import dataclass
import inspect
import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    BatchMLADecodePagedTSWrapper,
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
    batch_decode_with_paged_kv_cache,
    batch_mla_decode_with_paged_kv_cache,
    batch_prefill_with_paged_kv_cache,
)


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS attention requires SM100 or SM103",
)
_FP8 = torch.float8_e4m3fn
# Captured DSL launches keep raw pointers into these allocations.
_GRAPH_OWNERS: list[tuple[object, ...]] = []


@pytest.mark.parametrize(
    "wrapper_type",
    (
        BatchPrefillTSWrapper,
        BatchPrefillPagedTSWrapper,
        BatchDecodePagedTSWrapper,
        BatchMLADecodePagedTSWrapper,
    ),
)
def test_attention_ts_softmax_stats_public_contract(wrapper_type):
    plan = inspect.signature(wrapper_type.plan).parameters
    run = inspect.signature(wrapper_type.run).parameters
    assert "enable_softmax_stats" not in plan
    assert plan["store_softmax_stats"].default is False
    assert plan["store_softmax_stats"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run["softmax_stats"].default is None
    assert run["softmax_stats"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "one_shot",
    (
        batch_prefill_with_paged_kv_cache,
        batch_decode_with_paged_kv_cache,
        batch_mla_decode_with_paged_kv_cache,
    ),
)
def test_attention_ts_softmax_stats_one_shot_contract(one_shot):
    parameters = inspect.signature(one_shot).parameters
    assert parameters["store_softmax_stats"].default is False
    assert parameters["softmax_stats"].default is None


@pytest.mark.parametrize("invalid", (None, 1, "True"))
@pytest.mark.parametrize(
    "wrapper_type",
    (
        BatchPrefillTSWrapper,
        BatchPrefillPagedTSWrapper,
        BatchDecodePagedTSWrapper,
        BatchMLADecodePagedTSWrapper,
    ),
)
def test_attention_ts_softmax_stats_plan_requires_bool(wrapper_type, invalid):
    common = dict(
        device="cuda",
        batch_size=1,
        max_seq_len_q=1,
        max_kv_len=128,
        store_softmax_stats=invalid,
    )
    if wrapper_type in (BatchPrefillTSWrapper, BatchPrefillPagedTSWrapper):
        common.update(
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
        )
        if wrapper_type is BatchPrefillTSWrapper:
            common["packed"] = False
    elif wrapper_type is BatchDecodePagedTSWrapper:
        common.update(
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            q_data_type=torch.bfloat16,
            validate=False,
        )
    else:
        common.update(
            num_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=32,
            packed_query=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            validate=False,
        )
    with pytest.raises(TypeError, match="store_softmax_stats.*bool"):
        wrapper_type().plan(**common)


@pytest.mark.parametrize("split_kv", (1, 4))
def test_attention_ts_mla_stats_workspace_is_opt_in(split_kv):
    import cutlass
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import (
        config as one_cta,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta import (
        config as two_cta,
    )

    cfg = one_cta.make_throughput_latency_mla_config(
        batch_size=2,
        num_heads_q=8,
        seq_len_q=1,
        seq_len_kv=8193,
        logical_num_heads_q=8,
        logical_seq_len_q=1,
        tile_size_q=8,
        explicit_split_kv=split_kv,
        explicit_persistent=False,
        max_active_clusters=148,
        reduction_mode="gmem_separate",
    )
    kwargs = dict(cfg=cfg, partial_o_dtype=cutlass.BFloat16, lse_dtype=cutlass.Float32)
    original = one_cta.compute_workspace_size(**kwargs)
    assert (
        one_cta.compute_workspace_size(**kwargs, store_softmax_stats=False) == original
    )
    extra_rows = (
        0
        if split_kv == 1
        else cfg.batch_size * cfg.seq_len_q * cfg.num_heads_q * split_kv
    )
    assert (
        one_cta.compute_workspace_size(**kwargs, store_softmax_stats=True)
        == original + extra_rows * 8
    )
    kwargs = dict(
        tile_size_q=128,
        num_q_tiles=2,
        latent_dim=512,
        batch_size=2,
        split_kv=split_kv,
        partial_o_dtype=cutlass.BFloat16,
        lse_dtype=cutlass.Float32,
    )
    original = two_cta.compute_workspace_size(**kwargs)
    assert (
        two_cta.compute_workspace_size(**kwargs, store_softmax_stats=False) == original
    )
    extra_rows = 0 if split_kv == 1 else 128 * 2 * 2 * split_kv
    assert (
        two_cta.compute_workspace_size(**kwargs, store_softmax_stats=True)
        == original + extra_rows * 8
    )


@pytest.mark.parametrize("kind", ("decode", "mla"))
@pytest.mark.parametrize("use_wrapper", (False, True))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("store_softmax_stats", (False, True))
def test_attention_ts_softmax_stats_trace_outputs(
    kind, use_wrapper, packed, store_softmax_stats
):
    from flashinfer.fi_trace import fi_trace

    query_shape = (
        (5, 8, 576 if kind == "mla" else 128)
        if packed
        else (2, 2, 8, 576 if kind == "mla" else 128)
    )
    query = torch.empty(query_shape, dtype=torch.bfloat16)
    table = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)
    lengths = torch.tensor([32, 64], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32) if packed else None
    max_q = 3 if packed else 2
    kwargs = dict(block_tables=table, qo_indptr=offsets)
    if store_softmax_stats:
        kwargs["softmax_stats"] = torch.empty((*query_shape[:-1], 2))
    state = dict(
        mask_type="causal", max_kv_len=64, store_softmax_stats=store_softmax_stats
    )
    if kind == "decode":
        key = torch.empty((8, 2, 32, 128), dtype=torch.bfloat16)
        kwargs.update(q=query, paged_kv_cache=(key, torch.empty_like(key)))
        if use_wrapper:
            wrapper = BatchDecodePagedTSWrapper()
            wrapper._plan_state = SimpleNamespace(
                **state,
                page_size=32,
                storage_page_size=32,
                use_packed_q=packed,
                seq_len_q=max_q,
                output_dtype=torch.bfloat16,
                window_left=-1,
                kv_prefix_mode="dynamic",
                kv_lengths_mode="dynamic",
                planned_seq_lens_host=None,
                planned_seq_lens_device=None,
                config=SimpleNamespace(store_softmax_stats=store_softmax_stats),
            )
            target = wrapper.run
            kwargs["seq_lens"] = lengths
        else:
            target = batch_decode_with_paged_kv_cache
            kwargs.update(seq_lens_kv=lengths, seq_len_q=max_q, max_seq_len_q=max_q)
    else:
        kwargs.update(
            query=query,
            kv_cache=torch.empty((8, 32, 576), dtype=torch.bfloat16),
            seq_lens=lengths,
        )
        if use_wrapper:
            wrapper = BatchMLADecodePagedTSWrapper()
            wrapper._plan_state = SimpleNamespace(
                **state,
                packed_query=packed,
                max_seq_len_q=max_q,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
            target = wrapper.run
        else:
            target = batch_mla_decode_with_paged_kv_cache
            kwargs["max_seq_len_q"] = max_q
    if not use_wrapper:
        kwargs.update(mask_type="causal", max_kv_len=64)
        if store_softmax_stats:
            kwargs["store_softmax_stats"] = True
    definition = fi_trace(target, **kwargs)
    assert ("softmax_stats" in definition["outputs"]) == store_softmax_stats
    if store_softmax_stats:
        stats = definition["outputs"]["softmax_stats"]
        assert stats["dtype"] == "float32"
        assert stats["param"] == "softmax_stats"
        assert stats["shape"][:-1] == definition["outputs"]["output"]["shape"][:-1]
        assert definition["axes"][stats["shape"][-1]]["value"] == 2
        assert "softmax_stats" not in definition["inputs"]


def _reference(q, k, v, scale, output_scale, *, mask=None, sink=None):
    """Independent FP32 oracle, deliberately without P448 or kernel scratch."""
    heads = q.shape[-2]
    k = k.repeat_interleave(heads // k.shape[-2], dim=-2)
    v = v.repeat_interleave(heads // v.shape[-2], dim=-2)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
    if mask is not None:
        scores.masked_fill_(~mask[:, None, :], -torch.inf)
    maximum = scores.amax(-1)
    weights = (scores - maximum[..., None]).exp()
    denominator = weights.sum(-1)
    # A sink contributes once to the denominator, but is not a token-logit max.
    if sink is not None:
        denominator += (sink[None, :] - maximum).exp()
    result = torch.einsum("qhk,khd->qhd", weights, v.float())
    result *= output_scale / denominator[..., None]
    return result, torch.stack((maximum, denominator), dim=-1)


@dataclass
class _PagedCase:
    kind: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    qo_indptr: torch.Tensor | None
    max_q: int
    max_k: int
    mask_type: str
    window_left: int
    sm_scale: float
    output_scale: float

    @property
    def page_size(self):
        return self.k.shape[-2]

    @property
    def output_dtype(self):
        if self.kind == "decode" and self.q.dtype in (torch.float16, _FP8):
            return torch.float16
        return torch.bfloat16

    def expected(self):
        outputs, stats = [], []
        offsets = None if self.qo_indptr is None else self.qo_indptr.tolist()
        for batch, length in enumerate(self.seq_lens.tolist()):
            query = (
                self.q[batch]
                if offsets is None
                else self.q[offsets[batch] : offsets[batch + 1]]
            )
            if query.ndim == 2:
                query = query.unsqueeze(0)
            pages = self.block_tables[
                batch, : math.ceil(length / self.page_size)
            ].long()
            key = self.k.float()[pages].permute(0, 2, 1, 3).flatten(0, 1)[:length]
            value = self.v.float()[pages].permute(0, 2, 1, 3).flatten(0, 1)[:length]
            mask = None
            if self.mask_type == "causal":
                positions = torch.arange(length, device=self.q.device)
                right = (
                    length
                    - query.shape[0]
                    + torch.arange(query.shape[0], device=self.q.device)
                )
                mask = positions[None, :] <= right[:, None]
                if self.window_left >= 0:
                    mask &= positions[None, :] >= right[:, None] - self.window_left
            output, stat = _reference(
                query, key, value, self.sm_scale, self.output_scale, mask=mask
            )
            outputs.append(output)
            stats.append(stat)
        combine = torch.stack if offsets is None else torch.cat
        return (
            combine(outputs).reshape(*self.q.shape[:-1], self.v.shape[-1]),
            combine(stats).reshape(*self.q.shape[:-1], 2),
        )

    def plan(self, store_softmax_stats=True):
        common = dict(
            device=self.q.device,
            batch_size=self.block_tables.shape[0],
            max_seq_len_q=self.max_q,
            max_kv_len=self.max_k,
            page_size=self.page_size,
            mask_type=self.mask_type,
            store_softmax_stats=store_softmax_stats,
        )
        if self.kind == "prefill":
            wrapper = BatchPrefillPagedTSWrapper()
            wrapper.plan(
                **common,
                num_qo_heads=self.q.shape[-2],
                num_kv_heads=self.k.shape[1],
                head_dim=self.q.shape[-1],
                q_dtype=self.q.dtype,
                k_dtype=self.k.dtype,
                v_dtype=self.v.dtype,
                out_dtype=self.output_dtype,
                window_left=self.window_left,
                sm_scale=self.sm_scale,
                output_scale=self.output_scale,
            )
        elif self.kind == "decode":
            wrapper = BatchDecodePagedTSWrapper()
            wrapper.plan(
                **common,
                num_qo_heads=self.q.shape[-2],
                num_kv_heads=self.k.shape[1],
                head_dim=self.q.shape[-1],
                packed_query=self.qo_indptr is not None,
                q_data_type=self.q.dtype,
                k_data_type=self.k.dtype,
                v_data_type=self.v.dtype,
                o_data_type=self.output_dtype,
                window_left=self.window_left,
            )
        else:
            wrapper = BatchMLADecodePagedTSWrapper()
            wrapper.plan(
                **common,
                num_heads=self.q.shape[-2],
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                packed_query=self.qo_indptr is not None,
                q_data_type=self.q.dtype,
                kv_data_type=self.k.dtype,
                o_data_type=self.output_dtype,
            )
        return wrapper

    def run(self, wrapper, stats, *, out=None, validate=True):
        common = dict(out=out, softmax_stats=stats, validate=validate)
        if self.kind == "prefill":
            return wrapper.run(
                self.q,
                self.k,
                self.v,
                self.qo_indptr,
                self.block_tables,
                self.seq_lens,
                **common,
            )
        common.update(
            qo_indptr=self.qo_indptr,
            bmm1_scale=self.sm_scale,
            bmm2_scale=self.output_scale,
        )
        if self.kind == "decode":
            return wrapper.run(
                self.q, (self.k, self.v), self.seq_lens, self.block_tables, **common
            )
        return wrapper.run(self.q, self.k, self.block_tables, self.seq_lens, **common)

    def one_shot(self, stats):
        common = dict(
            mask_type=self.mask_type,
            out_dtype=self.output_dtype,
            store_softmax_stats=True,
            softmax_stats=stats,
        )
        if self.kind == "prefill":
            return batch_prefill_with_paged_kv_cache(
                self.q,
                self.k,
                self.v,
                self.qo_indptr,
                self.block_tables,
                self.seq_lens,
                page_size=self.page_size,
                window_left=self.window_left,
                sm_scale=self.sm_scale,
                output_scale=self.output_scale,
                **common,
            )
        common.update(
            qo_indptr=self.qo_indptr,
            max_seq_len_q=self.max_q,
            max_kv_len=self.max_k,
            bmm1_scale=self.sm_scale,
            bmm2_scale=self.output_scale,
        )
        if self.kind == "decode":
            return batch_decode_with_paged_kv_cache(
                self.q,
                (self.k, self.v),
                self.block_tables,
                self.seq_lens,
                seq_len_q=self.max_q,
                window_left=self.window_left,
                **common,
            )
        return batch_mla_decode_with_paged_kv_cache(
            self.q, self.k, self.block_tables, self.seq_lens, **common
        )


def _make_paged_case(
    kind,
    *,
    dtype=torch.bfloat16,
    packed=True,
    heads=None,
    head_dim=128,
    mask_type="causal",
    window_left=-1,
    kv_len=513,
    mixed_v=False,
    max_q=None,
):
    torch.manual_seed(85192)
    mla = kind == "mla"
    heads = (16 if mla else 8) if heads is None else heads
    kv_heads = 1 if mla else 2
    head_dim = 576 if mla else head_dim
    max_q = (129 if kind == "prefill" else 2) if max_q is None else max_q
    q_lengths = (max_q, max_q - 1) if packed else (max_q, max_q)
    page_size = 32
    columns = math.ceil(kv_len / page_size)
    page_count = columns * 2 + 3
    q = (0.2 * torch.randn(sum(q_lengths), heads, head_dim, device="cuda")).to(dtype)
    k = (
        0.2 * torch.randn(page_count, kv_heads, page_size, head_dim, device="cuda")
    ).to(dtype)
    v = (
        k[..., :512]
        if mla
        else (0.2 * torch.randn_like(k.float())).to(_FP8 if mixed_v else dtype)
    )
    block_tables = torch.randperm(page_count, device="cuda", dtype=torch.int32)[
        : columns * 2
    ].reshape(2, columns)
    seq_lens = torch.tensor([kv_len, kv_len - 17], device="cuda", dtype=torch.int32)
    qo_indptr = None
    if packed:
        qo_indptr = torch.tensor(
            [0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32
        )
    else:
        q = q.reshape(2, max_q, heads, head_dim)
        if kind == "decode" and max_q == 1:
            q = q.squeeze(1)
    return _PagedCase(
        kind,
        q,
        k,
        v,
        block_tables,
        seq_lens,
        qo_indptr,
        max_q,
        kv_len,
        mask_type,
        window_left,
        0.75 / math.sqrt(192 if mla else head_dim),
        0.625,
    )


def _assert_output(output, expected_output, case):
    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=1e-2)
    # Match the continuous MLA-prefill test's normalized-L2 accuracy budget.
    # This also bounds small outputs for which an absolute tolerance is weak;
    # FP8 V implies FP8 P, including mixed BF16-Q/K attention.
    relative_l2 = torch.linalg.vector_norm(
        output.float() - expected_output
    ) / torch.linalg.vector_norm(expected_output).clamp_min(1e-6)
    assert relative_l2.item() < (0.05 if case.v.dtype == _FP8 else 0.01)


def _assert_result(output, stats, case):
    expected_output, expected_stats = case.expected()
    torch.testing.assert_close(stats, expected_stats, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        pytest.param("prefill", {}, id="prefill-bf16-packed-causal"),
        pytest.param(
            "prefill",
            dict(dtype=_FP8, head_dim=256, mask_type="dense"),
            id="prefill-fp8-d256-dense",
        ),
        pytest.param("prefill", dict(mixed_v=True), id="prefill-mixed-qkbf16-vfp8"),
        pytest.param(
            "prefill",
            dict(head_dim=256, window_left=63),
            id="prefill-d256-head-paired-window",
        ),
        pytest.param(
            "decode",
            dict(dtype=torch.float16, packed=False, mask_type="dense"),
            id="decode-fp16-fixed-dense",
        ),
        pytest.param(
            "decode",
            dict(dtype=_FP8, heads=64, kv_len=4097),
            id="decode-fp8-packed-keeps-split",
        ),
        pytest.param(
            "decode", dict(head_dim=256, window_left=63), id="decode-d256-packed-window"
        ),
        pytest.param(
            "decode", dict(mixed_v=True, packed=False), id="decode-mixed-qkbf16-vfp8"
        ),
        pytest.param("mla", dict(dtype=_FP8, packed=False), id="mla-fp8-1cta-fixed"),
        pytest.param(
            "mla", dict(heads=128, mask_type="dense"), id="mla-bf16-2cta-packed-dense"
        ),
        pytest.param(
            "mla", dict(heads=12, kv_len=4097), id="mla-bf16-partial-flat-row-split"
        ),
    ],
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_paged_softmax_stats(kind, kwargs):
    case = _make_paged_case(kind, **kwargs)
    baseline = case.run(case.plan(False), None)
    wrapper = case.plan()
    stats = torch.full((*case.q.shape[:-1], 2), torch.nan, device="cuda")
    for validate in (False, True):
        with pytest.raises(ValueError, match="softmax_stats"):
            case.run(wrapper, None, validate=validate)
    output = case.run(wrapper, stats)
    _assert_result(output, stats, case)
    expected_output, _ = case.expected()
    # Stats-on centers scores before scaling to avoid cancellation. This can
    # change low-precision P rounding, so output equivalence is not guaranteed.
    # Assert attention accuracy independently for both paths instead.
    _assert_output(output, expected_output, case)
    _assert_output(baseline, expected_output, case)
    one_shot_stats = torch.full_like(stats, torch.nan)
    one_shot_output = case.one_shot(one_shot_stats)
    _assert_result(one_shot_output, one_shot_stats, case)


@pytest.mark.parametrize("kind", ("prefill", "decode", "mla"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_softmax_stats_runtime_buffer_validation(kind):
    case = _make_paged_case(kind)
    enabled, disabled = case.plan(), case.plan(False)
    shape = (*case.q.shape[:-1], 2)
    valid = torch.empty(shape, device="cuda")
    for validate in (False, True):
        with pytest.raises(ValueError, match="softmax_stats"):
            case.run(disabled, valid, validate=validate)
    for invalid in (
        valid.to(torch.bfloat16),
        torch.empty((*shape[:-1], 3), device="cuda"),
        torch.empty((*shape[:-1], 4), device="cuda")[..., ::2],
        torch.empty(shape, device="cpu"),
    ):
        with pytest.raises((TypeError, ValueError), match="softmax_stats"):
            case.run(enabled, invalid)
    shifted_storage = torch.empty(math.prod(shape) + 1, device="cuda")
    shifted = shifted_storage[1:].view(shape)
    assert shifted.is_contiguous() and shifted.data_ptr() % 16 == 4
    output = case.run(enabled, shifted)
    _assert_result(output, shifted, case)


@pytest.mark.parametrize("kind", ("prefill", "decode", "mla"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_softmax_stats_graph_reloads_metadata(kind):
    case = _make_paged_case(kind)
    wrapper = case.plan()
    stats = torch.full((*case.q.shape[:-1], 2), torch.nan, device="cuda")
    output = case.run(wrapper, stats)
    before = stats.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(wrapper, stats, out=output, validate=False)
    # Keep capacities/addresses fixed; change each request's query interval,
    # logical KV length, and physical page mapping after capture.
    case.qo_indptr[1] -= 1
    case.seq_lens.copy_(
        torch.tensor(
            [case.max_k - 19, case.max_k - 3], dtype=torch.int32, device="cuda"
        )
    )
    case.block_tables.copy_((case.block_tables + 1) % case.k.shape[0])
    stats.fill_(torch.nan)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    _assert_result(output, stats, case)
    assert not torch.equal(stats, before)
    # Another caller-owned statistics buffer can be bound without replanning.
    rebound_stats = torch.full_like(stats, torch.nan)
    rebound_output = case.run(wrapper, rebound_stats)
    torch.testing.assert_close(rebound_stats, stats, rtol=0, atol=0)
    torch.testing.assert_close(rebound_output, output, rtol=0, atol=0)
    _GRAPH_OWNERS.append((graph, wrapper, case, output, stats))


@pytest.mark.parametrize("head_dim", (128, 256))
@pytest.mark.parametrize("paged", (False, True), ids=("contiguous", "paged"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_prefill_large_uniform_logits_stats(head_dim, paged):
    """The maximum's probability is one even when scaled logits lose low bits."""
    q_shape = (1, 8, head_dim) if paged else (1, 1, 8, head_dim)
    k_shape = (4, 2, 32, head_dim) if paged else (1, 128, 2, head_dim)
    q = torch.zeros(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(k_shape, device="cuda", dtype=torch.bfloat16)
    q[..., 0], q[..., 1] = 4096, 1
    k[..., 0], k[..., 1] = 4096, 2
    v = torch.full_like(k, 0.25)
    stats = torch.empty((*q.shape[:-1], 2), device=q.device, dtype=torch.float32)
    plan_kwargs = dict(
        device=q.device,
        batch_size=1,
        max_seq_len_q=1,
        max_kv_len=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        mask_type="dense",
        sm_scale=1.0,
        store_softmax_stats=True,
    )
    if paged:
        wrapper = BatchPrefillPagedTSWrapper()
        wrapper.plan(**plan_kwargs, page_size=32)
        qo_indptr = torch.tensor([0, 1], device=q.device, dtype=torch.int32)
        block_tables = torch.arange(4, device=q.device, dtype=torch.int32)[None, :]
        seq_lens = torch.tensor([128], device=q.device, dtype=torch.int32)
        output = wrapper.run(
            q, k, v, qo_indptr, block_tables, seq_lens, softmax_stats=stats
        )
    else:
        wrapper = BatchPrefillTSWrapper()
        wrapper.plan(**plan_kwargs, packed=False)
        output = wrapper.run(q, k, v, softmax_stats=stats)
    # Every raw dot product is exactly representable as FP32: 2**24 + 2.
    # Forming FMA(score, scale, round(-max*scale)) instead would make every
    # maximum's probability about 1.847 rather than one.
    torch.testing.assert_close(
        stats[..., 0], torch.full_like(stats[..., 0], 2**24 + 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        stats[..., 1], torch.full_like(stats[..., 1], 128), rtol=1e-6, atol=1e-4
    )
    torch.testing.assert_close(output, torch.full_like(output, 0.25), rtol=0, atol=0)


@pytest.mark.parametrize(
    "head_dim,mask_type",
    ((128, "variable_window"), (256, "variable_window"), (256, "causal")),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_contiguous_window_softmax_stats(head_dim, mask_type):
    """Both head-paired owners export their own query/head statistics."""
    torch.manual_seed(76229)
    q = (0.2 * torch.randn(1, 257, 8, head_dim, device="cuda")).half()
    k = (0.2 * torch.randn(1, 513, 2, head_dim, device="cuda")).half()
    # A zero anchor after fully masked tiles must not replace a negative max.
    q.abs_()
    k.copy_(-k.abs())
    v = (0.2 * torch.randn_like(k)).half()
    stats = torch.full((*q.shape[:-1], 2), torch.nan, device="cuda")
    scale = 0.75 / math.sqrt(head_dim)
    window_left = 63 if mask_type == "causal" else -1
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=1,
        max_seq_len_q=257,
        max_kv_len=513,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        packed=False,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=scale,
        store_softmax_stats=True,
    )
    q_positions = torch.arange(257, device="cuda")
    k_positions = torch.arange(513, device="cuda")
    kwargs = {}
    if mask_type == "variable_window":
        starts = (q_positions % 3 * 128)[None, :].to(torch.int32)
        # The current variable-window domain uses the last Q row's end as
        # the CTA upper bound, so keep ends nondecreasing while exercising
        # different (nonmonotonic) starts within each Q tile.
        ends = (q_positions + 256)[None, :].to(torch.int32)
        kwargs.update(
            variable_window_token_starts=starts, variable_window_token_ends=ends
        )
        mask = (k_positions[None, :] >= starts.T) & (k_positions[None, :] <= ends.T)
    else:
        ends = 513 - 257 + q_positions
        mask = (k_positions[None, :] <= ends[:, None]) & (
            k_positions[None, :] >= ends[:, None] - window_left
        )
    output = wrapper.run(q, k, v, softmax_stats=stats, **kwargs)
    expected_output, expected_stats = _reference(
        q[0], k[0], v[0], scale, 1.0, mask=mask
    )
    torch.testing.assert_close(stats[0], expected_stats, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(output[0].float(), expected_output, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("kind", ("prefill", "decode", "mla"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_paged_softmax_stats_merge_kv_partitions(kind):
    case = _make_paged_case(kind, mask_type="dense", kv_len=512)
    # Equal page-aligned partitions share Q and use independently planned KV.
    case.seq_lens.fill_(512)
    original_tables = case.block_tables
    pieces = []
    for begin, end in ((0, 8), (8, 16)):
        case.block_tables = original_tables[:, begin:end].contiguous()
        case.seq_lens.fill_(256)
        case.max_k = 256
        stats = torch.empty((*case.q.shape[:-1], 2), device="cuda")
        output = case.run(case.plan(), stats)
        _assert_result(output, stats, case)
        pieces.append((output.float(), stats))
    (first, first_stats), (second, second_stats) = pieces
    maximum = torch.maximum(first_stats[..., 0], second_stats[..., 0])
    first_weight = (first_stats[..., 0] - maximum).exp() * first_stats[..., 1]
    second_weight = (second_stats[..., 0] - maximum).exp() * second_stats[..., 1]
    denominator = first_weight + second_weight
    merged = (
        first * first_weight[..., None] + second * second_weight[..., None]
    ) / denominator[..., None]
    case.block_tables = original_tables
    case.seq_lens.fill_(512)
    case.max_k = 512
    _assert_result(merged, torch.stack((maximum, denominator), -1), case)


def _force_decode_topology(monkeypatch, case, family, reduction, *, tile_kv=128):
    import cutlass
    from flashinfer.attention.prims_ts import decode
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    # The public API intentionally does not expose kernel policy knobs. Force
    # a qualified internal configuration so changes in automatic selection do
    # not silently remove coverage of one final-output/statistics owner.
    modes = {
        "direct": "disabled",
        "fused": "gmem_reduction",
        "cluster": "cluster_smem_reduction",
        "serial": "gmem_reduction_with_separate_kernel",
        "parallel": "gmem_reduction_with_separate_kernel",
    }
    if reduction == "serial":
        for name in (
            "use_parallel_separate_reduction",
            "use_parallel_separate_reduction_pdl",
        ):
            monkeypatch.setattr(
                fmha_decode_config.FmhaDecodeConfig, name, property(lambda _self: False)
            )
    keeps = family == "keeps"
    dtype = cutlass.Float8E4M3FN if case.q.dtype == _FP8 else cutlass.BFloat16
    config = fmha_decode_config.make_decode_config(
        headdim=case.q.shape[-1],
        args={
            "use_keeps_mma_ab": keeps,
            "groups_tokens_heads_q": not keeps or tile_kv == 256,
            "tile_size_q": 64 if keeps else 16,
            "tile_size_kv": tile_kv,
            "use_variable_seqlens_q": case.qo_indptr is not None,
            "use_persistent_scheduler": False,
            "store_softmax_stats": True,
        },
        seq_len_q=case.max_q,
        seq_len_kv=case.max_k,
        batch_size=case.block_tables.shape[0],
        num_heads_q=case.q.shape[-2],
        num_heads_kv=case.k.shape[1],
        q_dtype=dtype,
        k_dtype=dtype,
        v_dtype=dtype,
        o_dtype=cutlass.BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=case.page_size,
        split_kv_mode=modes[reduction],
        splits_kv=1 if reduction == "direct" else 4,
        max_splits_kv=1 if reduction == "direct" else 4,
        mask_type=case.mask_type,
        auto_tuner=False,
    )
    assert config.use_keeps_mma_ab == keeps
    assert config.use_split_kv == (reduction != "direct")
    assert config.use_cluster_smem_reduction == (reduction == "cluster")
    assert config.use_separate_reduction_kernel == (reduction in ("serial", "parallel"))
    if config.use_separate_reduction_kernel:
        assert config.use_parallel_separate_reduction == (reduction == "parallel")
    spec = decode._decode_launch_spec_from_config(
        config,
        batch_size=case.block_tables.shape[0],
        num_qo_heads=case.q.shape[-2],
        num_kv_heads=case.k.shape[1],
        head_dim=case.q.shape[-1],
        seq_len_q=case.max_q,
        max_active_clusters=fmha_decode_config.get_max_active_clusters_for_cluster_size(
            1
        ),
    )
    monkeypatch.setattr(
        decode, "_resolve_decode_launch_spec", lambda *_args, **_kwargs: spec
    )
    return decode._get_compiled_decode


@pytest.mark.parametrize(
    "family,reduction",
    [
        (family, reduction)
        for family in ("keeps", "swaps")
        for reduction in ("direct", "fused", "serial", "parallel")
    ]
    + [("swaps", "cluster")],
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_softmax_stats_forced_reduction(
    monkeypatch, family, reduction
):
    case = _make_paged_case(
        "decode",
        heads=128 if family == "keeps" else 16,
        packed=False,
        kv_len=4097,
        max_q=1,
        mask_type="dense",
    )
    compile_cache = _force_decode_topology(monkeypatch, case, family, reduction)
    compile_cache.cache_clear()
    try:
        wrapper = case.plan()
        stats = torch.full((*case.q.shape[:-1], 2), torch.nan, device="cuda")
        output = case.run(wrapper, stats)
        _assert_result(output, stats, case)
    finally:
        # The serial test temporarily overrides derived configuration properties.
        compile_cache.cache_clear()


@pytest.mark.parametrize("reduction", ("direct", "fused", "parallel"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_softmax_stats_kv256_actual_maximum(monkeypatch, reduction):
    case = _make_paged_case(
        "decode", heads=64, packed=False, max_q=2, kv_len=4097, mask_type="dense"
    )
    # Successive tiles have close, increasing maxima. Deferring a softmax
    # rescale is valid for O but must not turn the exported max into an anchor.
    case.q.fill_(0.125)
    for page in range(case.k.shape[0]):
        case.k[page].fill_(0.03125 + page / 32768)
    compile_cache = _force_decode_topology(
        monkeypatch, case, "keeps", reduction, tile_kv=256
    )
    compile_cache.cache_clear()
    try:
        wrapper = case.plan()
        assert not wrapper._plan_state.config.defers_softmax_anchor_updates
        stats = torch.full((*case.q.shape[:-1], 2), torch.nan, device="cuda")
        output = case.run(wrapper, stats)
        _assert_result(output, stats, case)
    finally:
        compile_cache.cache_clear()


def _force_mla_topology(monkeypatch, family, reduction):
    from flashinfer.attention.prims_ts import mla_decode
    from flashinfer.attention.prims_ts.kernels.mla_decode import kernel_policy
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import (
        kernel as one_cta,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta import (
        kernel as two_cta,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta import (
        config as two_cta_config,
    )

    kernel_name = "throughput_latency_1cta" if family == "1cta" else "throughput_2cta"
    splits = (
        1
        if reduction == "direct"
        else (64 if family == "2cta" and reduction == "parallel" else 4)
    )
    monkeypatch.setattr(
        kernel_policy,
        "resolve_mla_kernel_policy",
        lambda *_args, **_kwargs: (kernel_name, "explicit"),
    )
    if family == "1cta":
        constructor = one_cta.ThroughputLatencyMlaDecodeTs

        def make_kernel(**kwargs):
            kwargs.update(
                profile=None,
                explicit_split_kv=splits,
                explicit_persistent=False,
                reduction_mode="cluster" if reduction == "cluster" else "gmem_separate",
            )
            kernel = constructor(**kwargs)
            if reduction == "serial":
                kernel.use_parallel_reduction = False
                kernel.parallel_reduction_topology = None
            if reduction == "parallel":
                assert kernel.use_parallel_reduction
            return kernel

        monkeypatch.setattr(one_cta, "ThroughputLatencyMlaDecodeTs", make_kernel)
    else:
        constructor = two_cta.MlaDecodeTs
        monkeypatch.setattr(
            two_cta_config, "compute_split_kv", lambda **_kwargs: splits
        )

        def make_kernel(**kwargs):
            kwargs["is_persistent"] = False
            kernel = constructor(**kwargs)
            if reduction == "serial":
                kernel.use_parallel_reduction = False
                kernel.parallel_reduction_topology = None
            if reduction == "parallel":
                assert kernel.use_parallel_reduction
            return kernel

        monkeypatch.setattr(two_cta, "MlaDecodeTs", make_kernel)
    return (
        mla_decode._resolve_mla_decode_launch_spec,
        mla_decode._get_compiled_mla_decode,
    )


@pytest.mark.parametrize(
    "family,reduction",
    [("1cta", mode) for mode in ("direct", "cluster", "serial", "parallel")]
    + [("2cta", mode) for mode in ("direct", "serial", "parallel")],
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_mla_softmax_stats_forced_reduction(
    monkeypatch, family, reduction
):
    case = _make_paged_case(
        "mla",
        heads=8 if family == "1cta" else 128,
        packed=False,
        max_q=1,
        kv_len=8193,
        dtype=_FP8 if reduction in ("direct", "serial") else torch.bfloat16,
    )
    caches = _force_mla_topology(monkeypatch, family, reduction)
    for cache in caches:
        cache.cache_clear()
    try:
        wrapper = case.plan()
        policy = dict(wrapper._plan_state.policy)
        assert policy["kernel"] == (
            "throughput_latency_1cta" if family == "1cta" else "throughput_2cta"
        )
        assert (policy["split_kv"] == 1) == (reduction == "direct")
        assert policy["use_cluster_reduction"] == (reduction == "cluster")
        stats = torch.full((*case.q.shape[:-1], 2), torch.nan, device="cuda")
        output = case.run(wrapper, stats)
        _assert_result(output, stats, case)
    finally:
        for cache in caches:
            cache.cache_clear()
