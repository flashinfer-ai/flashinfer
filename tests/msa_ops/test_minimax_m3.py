# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""MiniMax-M3 contract tests; the original indexer is intentionally not invoked."""

import pytest
import torch

from flashinfer.msa_ops import (
    MiniMaxM3SparseDecodeWorkspace,
    minimax_m3_sparse_attn_decode,
)
from flashinfer.utils import get_compute_capability


def make_case(
    batch=2, qlen=4, hkv=1, context=8192, shared=True, seed=4567, ragged=True
):
    """Unsorted per-token rows, poisoned unused slots and permuted physical pages."""
    torch.manual_seed(seed)
    pages = (context + 127) // 128
    # Share only a logical prefix between requests, never alias within a request.
    table = torch.randperm(batch * pages, dtype=torch.int32).reshape(batch, pages)
    if shared and batch > 1:
        table[1:, : pages // 2] = table[0, : pages // 2]
    lengths = torch.tensor(
        [
            (
                max(qlen, context - (i * 137) % max(1, context - qlen + 1))
                if ragged
                else max(qlen, context)
            )
            for i in range(batch)
        ],
        dtype=torch.int32,
    )
    indices = torch.full((hkv, batch * qlen, 16), 0x7FFFFFFF, dtype=torch.int32)
    for h in range(hkv):
        for t in range(batch * qlen):
            length = int(lengths[t // qlen]) - qlen + t % qlen + 1
            count = (length + 127) // 128
            # Alternate including/excluding the partial last page. Neither
            # sorted selections nor an always-present tail may be assumed.
            candidates = torch.randperm(count)
            if count > 16:
                candidates = candidates[candidates != count - 1]
                if t % 2 == 0:
                    candidates = torch.cat((torch.tensor([count - 1]), candidates))
            selected = candidates[:16]
            indices[h, t, : selected.numel()] = selected
    q = torch.randn(batch * qlen, hkv * 16, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        batch * pages, hkv, 128, 256, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    return dict(
        q=q,
        kv_cache=kv,
        topk_idx=indices.cuda(),
        block_table=table.cuda(),
        seq_lens=lengths.cuda(),
        k_scale=torch.tensor([0.7], device="cuda"),
        v_scale=torch.tensor([1.3], device="cuda"),
        out=torch.empty_like(q),
        workspace=MiniMaxM3SparseDecodeWorkspace(batch, hkv * 16, hkv, qlen),
    )


def torch_reference(case):
    """Independent test oracle. Gathering/host reads are outside the tested API."""
    q, kv = case["q"], case["kv_cache"]
    w = case["workspace"]
    table, indices, lengths = (
        case[n].cpu() for n in ("block_table", "topk_idx", "seq_lens")
    )
    result = torch.zeros_like(q)
    for t in range(w.total_q):
        n = max(
            0,
            int(lengths[t // w.decode_query_len])
            - w.decode_query_len
            + t % w.decode_query_len
            + 1,
        )
        for h in range(w.num_kv_heads):
            count = min(16, (n + 127) // 128)
            if count == 0:
                continue
            selected = indices[h, t, :count].long()
            physical = table[t // w.decode_query_len, selected].long().cuda()
            # PyTorch's FP8 indexing is not implemented on all supported builds.
            cache = kv.view(torch.uint8)[physical, h].view(torch.float8_e4m3fn).float()
            positions = selected[:, None] * 128 + torch.arange(128)[None, :]
            valid = (positions < n).flatten().cuda()
            cache = cache.reshape(-1, 256)[valid]
            k = (cache[:, :128] * case["k_scale"]).to(q.dtype).float()
            v = (cache[:, 128:] * case["v_scale"]).to(q.dtype).float()
            qq = q[t, h * 16 : (h + 1) * 16].float()
            scale = case.get("sm_scale", 128**-0.5)
            result[t, h * 16 : (h + 1) * 16] = ((qq @ k.T * scale).softmax(-1) @ v).to(
                q.dtype
            )
    return result


def assert_numerics(actual, expected):
    assert torch.isfinite(actual).all()
    # Match the existing MSA BF16 tolerance (_msa_attention_check). The native
    # route folds scalar K/V scales into attention; the Triton reference rounds
    # scaled K/V to BF16 first. Cancellation can expose that extra rounding in
    # a near-zero coordinate even when both agree with FP32 dequantized attention.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    # Enforce the relative RMS bound separately for EVERY query/head, so a
    # small-magnitude incorrect row cannot hide behind the absolute tolerance
    # or be diluted by a large batch of correct rows.
    error_rms = (actual.float() - expected.float()).square().mean(dim=-1).sqrt()
    ref_rms = expected.float().square().mean(dim=-1).sqrt()
    assert torch.all(error_rms <= 0.015 * ref_rms + 1e-5)


def test_numeric_check_rejects_small_incorrect_rows():
    expected = torch.full((2, 16, 128), 0.001, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError):
        assert_numerics(torch.zeros_like(expected), expected)


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in {(10, 0), (10, 3)}:
        pytest.skip("requires SM100/SM103")


@pytest.mark.parametrize("hkv", [1, 4])
@pytest.mark.parametrize("qlen", [2, 3, 4, 5, 6, 7, 8])
def test_independent_rows(hkv, qlen):
    case = make_case(qlen=qlen, hkv=hkv, context=8193)
    assert not torch.equal(case["topk_idx"][:, 0], case["topk_idx"][:, 1])
    assert_numerics(minimax_m3_sparse_attn_decode(**case), torch_reference(case))


@pytest.mark.parametrize(
    "context", [8, 127, 128, 129, 255, 2047, 2048, 2049, 100003, 200003, 262144]
)
@pytest.mark.parametrize("hkv", [1, 4])
def test_lengths_and_shared_pages(context, hkv):
    case = make_case(context=context, hkv=hkv)
    result = minimax_m3_sparse_attn_decode(**case)
    assert result is case["out"]
    assert_numerics(result, torch_reference(case))
    # Independently inspect the metadata; poisoned unused slots must be ignored.
    w = case["workspace"]
    for t in range(w.total_q):
        n = (
            int(case["seq_lens"][t // w.decode_query_len])
            - w.decode_query_len
            + t % w.decode_query_len
            + 1
        )
        count = min(16, (n + 127) // 128)
        for h in range(hkv):
            sel = case["topk_idx"][h, t, :count].sort().values.long()
            expected_pages = case["block_table"][t // w.decode_query_len, sel]
            torch.testing.assert_close(
                w.sparse_pages[h, t, :count], expected_pages, rtol=0, atol=0
            )
            length = torch.clamp(n - sel * 128, min=0, max=128).sum()
            assert int(w.sparse_lens[h, t]) == int(length)


@pytest.mark.parametrize("hkv", [1, 4])
def test_graph_replay_updates_every_metadata_input(hkv):
    case = make_case(batch=4, hkv=hkv, context=262144)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            minimax_m3_sparse_attn_decode(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        minimax_m3_sparse_attn_decode(**case)
    pointers = {k: v.data_ptr() for k, v in case.items() if isinstance(v, torch.Tensor)}
    original_q = case["q"].clone()
    original_kv = case["kv_cache"].view(torch.uint8).clone()
    previous = None
    for seed in [11, 23, 37, 51]:
        replacement = make_case(batch=4, hkv=hkv, context=262144, seed=seed)
        # Change tables, lengths and independently selected rows in place.
        for name in ("block_table", "seq_lens", "topk_idx"):
            if name == "seq_lens":
                continue
            case[name].copy_(replacement[name])
        # Cross the <16-page and partial-page boundaries without recapturing.
        # Inactive top-k storage is poisoned again, not carried from warmup.
        lengths = {
            11: [4, 17, 129, 2047],
            23: [100003, 200003, 262144, 128],
            37: [5, 200001, 262143, 2031],
            51: [8193, 262144, 129, 8],
        }[seed]
        case["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        case["topk_idx"].fill_(0x7FFFFFFF)
        for t in range(case["workspace"].total_q):
            n = int(case["seq_lens"][t // 4]) - 4 + t % 4 + 1
            pages = (n + 127) // 128
            for h in range(hkv):
                count = min(16, pages)
                case["topk_idx"][h, t, :count].copy_(
                    torch.randperm(pages, device="cuda")[:count]
                )
        case["k_scale"].fill_(0.3 + seed / 100)
        case["v_scale"].fill_(1.1 + seed / 100)
        graph.replay()
        torch.cuda.synchronize()
        assert_numerics(case["out"], torch_reference(case))
        if previous is not None:
            assert not torch.equal(case["out"], previous)
        previous = case["out"].clone()
        assert pointers == {
            k: v.data_ptr() for k, v in case.items() if isinstance(v, torch.Tensor)
        }
    torch.testing.assert_close(case["q"], original_q, rtol=0, atol=0)
    torch.testing.assert_close(
        case["kv_cache"].view(torch.uint8), original_kv, rtol=0, atol=0
    )


@pytest.mark.parametrize("strided", [False, True])
def test_no_steady_state_tensor_allocations(monkeypatch, strided):
    case = make_case(hkv=4 if strided else 1)
    if strided:
        total = case["workspace"].total_q
        storage = torch.empty(4, total * 2 + 7, 16, dtype=torch.int32, device="cuda")
        storage[:, :total].copy_(case["topk_idx"])
        case["topk_idx"] = storage[:, :total]
    minimax_m3_sparse_attn_decode(**case)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]

    def forbidden(*args, **kwargs):
        raise AssertionError("device allocation or tensor D2H in steady state")

    with monkeypatch.context() as m:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like"):
            m.setattr(torch, name, forbidden)
        for name in ("item", "cpu", "tolist"):
            m.setattr(torch.Tensor, name, forbidden)
        for _ in range(4):
            minimax_m3_sparse_attn_decode(**case)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before


def test_changing_one_row_does_not_change_other_tokens():
    case = make_case(hkv=4)
    original = minimax_m3_sparse_attn_decode(**case).clone()
    case["topk_idx"][2, 3] = torch.arange(16, device="cuda", dtype=torch.int32)
    result = minimax_m3_sparse_attn_decode(**case).clone()
    delta = (result != original).any(-1)
    expected_mask = torch.zeros_like(delta)
    expected_mask[3, 32:48] = True
    assert not delta[~expected_mask].any()
    assert delta[expected_mask].any()
    assert_numerics(result, torch_reference(case))


@pytest.mark.parametrize(
    "name", ["q", "kv_cache", "topk_idx", "seq_lens", "k_scale", "v_scale", "out"]
)
def test_invalid_tensor_contract_rejected(name):
    case = make_case(context=129)
    case[name] = case[name].to(torch.float64)
    with pytest.raises(ValueError, match=name):
        minimax_m3_sparse_attn_decode(**case)


def test_per_token_scale_rejected_explicitly():
    case = make_case(context=129)
    case["k_scale"] = torch.ones(1, 256, device="cuda")
    with pytest.raises(ValueError, match="scalar"):
        minimax_m3_sparse_attn_decode(**case)


def test_zero_dimensional_scales():
    case = make_case(context=129)
    case["k_scale"] = case["k_scale"].reshape(())
    case["v_scale"] = case["v_scale"].reshape(())
    assert_numerics(minimax_m3_sparse_attn_decode(**case), torch_reference(case))


@pytest.mark.parametrize("scale_device", ["cpu", "cuda"])
@pytest.mark.parametrize("capturing", [False, True])
def test_tensor_sm_scale_rejected_without_sync(monkeypatch, scale_device, capturing):
    import flashinfer.msa_ops.minimax_m3 as m3

    case = make_case(context=129)
    case["sm_scale"] = torch.tensor(0.1, device=scale_device)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Rejected calls must not convert data or load/launch kernels"
        )

    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    monkeypatch.setattr(m3, "_get_metadata_module", forbidden)

    def reject():
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(TypeError, match="sm_scale must be a host float"):
                minimax_m3_sparse_attn_decode(**case)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    if capturing:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reject()
            case["out"].zero_()  # The rejected call must not invalidate capture.
        graph.replay()
        assert torch.count_nonzero(case["out"]) == 0
    else:
        reject()


@pytest.mark.parametrize("sm_scale", [0.0625, 0.125])
def test_host_sm_scale_still_works(sm_scale):
    case = make_case(context=129)
    case["sm_scale"] = sm_scale
    assert_numerics(minimax_m3_sparse_attn_decode(**case), torch_reference(case))


def test_noncontiguous_cache_rejected_without_copy():
    case = make_case(hkv=4, context=129)
    case["kv_cache"] = case["kv_cache"].transpose(0, 1)
    with pytest.raises(ValueError, match="kv_cache"):
        minimax_m3_sparse_attn_decode(**case)


@pytest.mark.parametrize("hkv", [1, 4])
@pytest.mark.parametrize("layout", ["indexer", "stepped"])
def test_strided_metadata_graph(hkv, layout):
    case = make_case(batch=4, hkv=hkv, context=4099)
    total = case["workspace"].total_q
    width = 32 if layout == "stepped" else 16
    storage = torch.full(
        (hkv, total * 2 + 7, width), 0x12345678, dtype=torch.int32, device="cuda"
    )
    indices = (
        storage[:, 3 : 3 + total, ::2] if layout == "stepped" else storage[:, :total, :]
    )
    indices.copy_(case["topk_idx"])
    case["topk_idx"] = indices
    pages = case["block_table"].shape[1]
    table_storage = torch.full((8, pages + 11), -99, dtype=torch.int32, device="cuda")
    table = table_storage[::2, 5 : 5 + pages]
    table.copy_(case["block_table"])
    case["block_table"] = table
    if layout == "stepped":
        length_storage = torch.zeros(8, dtype=torch.int32, device="cuda")
        length_storage[::2].copy_(case["seq_lens"])
        case["seq_lens"] = length_storage[::2]
    minimax_m3_sparse_attn_decode(**case)
    assert_numerics(case["out"], torch_reference(case))
    original_storage = storage.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        minimax_m3_sparse_attn_decode(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        minimax_m3_sparse_attn_decode(**case)
    case["block_table"].copy_(case["block_table"].roll(1, 1))
    case["k_scale"].fill_(0.9)
    graph.replay()
    torch.cuda.synchronize()
    assert_numerics(case["out"], torch_reference(case))
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)


def test_graph_padding_rows_can_become_active():
    case = make_case(batch=4, context=8193, hkv=4)
    minimax_m3_sparse_attn_decode(**case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        minimax_m3_sparse_attn_decode(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        minimax_m3_sparse_attn_decode(**case)
    for lengths in (
        [0, 8, 0, 129],
        [8193, 0, 17, 0],
        [0, 0, 0, 0],
        [17, 129, 4097, 8193],
    ):
        case["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        case["topk_idx"].fill_(0x7FFFFFFF)
        for t in range(16):
            n = max(0, lengths[t // 4] - 4 + t % 4 + 1)
            count = min(16, (n + 127) // 128)
            if count:
                for h in range(4):
                    case["topk_idx"][h, t, :count].copy_(
                        torch.randperm((n + 127) // 128, device="cuda")[:count]
                    )
        graph.replay()
        torch.cuda.synchronize()
        active = (case["seq_lens"] > 0).repeat_interleave(4)
        if any(lengths):
            assert_numerics(case["out"][active], torch_reference(case)[active])
