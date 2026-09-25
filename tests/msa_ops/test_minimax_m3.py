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
    MSASparseAttentionWorkspace,
    msa_sparse_decode_attention,
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
        k=kv[..., :128],
        v=kv[..., 128:],
        seqlen_q=qlen,
        q2k_indices=indices.cuda(),
        page_table=table.cuda(),
        seqused_k=lengths.cuda(),
        k_scale=torch.tensor([0.7], device="cuda"),
        v_scale=torch.tensor([1.3], device="cuda"),
        out=torch.empty_like(q),
        workspace=MSASparseAttentionWorkspace(q.device),
    )


def packed_kv_cache(case):
    """Recover the original packed allocation without copying K/V."""
    k = case["k"]
    return k.as_strided((*k.shape[:-1], 256), k.stride())


def torch_reference(case):
    """Independent test oracle. Gathering/host reads are outside the tested API."""
    q, kv = case["q"], packed_kv_cache(case)
    table, indices, lengths = (
        case[n].cpu() for n in ("page_table", "q2k_indices", "seqused_k")
    )
    result = torch.zeros_like(q)
    for t in range(case["q"].shape[0]):
        n = max(
            0,
            int(lengths[t // case["seqlen_q"]])
            - case["seqlen_q"]
            + t % case["seqlen_q"]
            + 1,
        )
        for h in range(case["k"].shape[1]):
            count = min(16, (n + 127) // 128)
            if count == 0:
                continue
            selected = indices[h, t, :count].long()
            physical = table[t // case["seqlen_q"], selected].long().cuda()
            # PyTorch's FP8 indexing is not implemented on all supported builds.
            cache = kv.view(torch.uint8)[physical, h].view(torch.float8_e4m3fn).float()
            positions = selected[:, None] * 128 + torch.arange(128)[None, :]
            valid = (positions < n).flatten().cuda()
            cache = cache.reshape(-1, 256)[valid]
            k = (cache[:, :128] * case["k_scale"]).to(q.dtype).float()
            v = (cache[:, 128:] * case["v_scale"]).to(q.dtype).float()
            qq = q[t, h * 16 : (h + 1) * 16].float()
            scale = case.get("softmax_scale", 128**-0.5)
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


@pytest.mark.parametrize(
    "scale", [0, -0.125, float("inf"), float("-inf"), float("nan"), torch.tensor(0.1)]
)
def test_packed_fp8_host_scale_rejected_before_cuda_work(monkeypatch, scale):
    import flashinfer.msa_ops.sparse_decode as decode
    import flashinfer.jit.blackwell_msa as jit_msa

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "invalid scale must not be converted from a Tensor or launch work"
        )

    monkeypatch.setattr(decode, "is_blackwell_msa_device", lambda device: True)
    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    monkeypatch.setattr(jit_msa, "load_msa_decode_metadata_module", forbidden)
    packed = torch.empty(1, 1, 128, 256, dtype=torch.float8_e4m3fn)
    error = TypeError if isinstance(scale, torch.Tensor) else ValueError
    with pytest.raises(error, match="softmax_scale"):
        decode.msa_sparse_decode_attention(
            torch.empty(4, 16, 128, dtype=torch.bfloat16),
            packed[..., :128],
            packed[..., 128:],
            torch.empty(1, 4, 16, dtype=torch.int32),
            softmax_scale=scale,
        )


def test_packed_fp8_dispatch_reuses_existing_msa_interface(monkeypatch):
    import flashinfer.msa_ops.sparse_decode as decode
    from flashinfer.msa_ops import _blackwell_sm100 as backend

    packed = torch.empty(1, 1, 128, 256, dtype=torch.float8_e4m3fn)
    q = torch.empty(4, 16, 128, dtype=torch.bfloat16)
    indices = torch.empty(1, 4, 16, dtype=torch.int32)
    workspace = object()

    def route(actual_q, k, v, actual_indices, **kwargs):
        assert actual_q is q and actual_indices is indices
        assert k.stride(-2) == v.stride(-2) == 256
        assert kwargs["seqlen_q"] == 4
        assert kwargs["workspace"] is workspace
        return q

    monkeypatch.setattr(decode, "is_blackwell_msa_device", lambda device: True)
    monkeypatch.setattr(backend, "_run_packed_fp8_decode", route)
    assert (
        decode.msa_sparse_decode_attention(
            q,
            packed[..., :128],
            packed[..., 128:],
            indices,
            seqlen_q=4,
            workspace=workspace,
        )
        is q
    )


@pytest.fixture(autouse=True)
def require_blackwell(request):
    # These descriptor/dispatch tests also run without a CUDA device.
    if request.function in (
        test_packed_fp8_host_scale_rejected_before_cuda_work,
        test_packed_fp8_dispatch_reuses_existing_msa_interface,
    ):
        return
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in {(10, 0), (10, 3)}:
        pytest.skip("requires SM100/SM103")


@pytest.mark.parametrize("hkv", [1, 4])
@pytest.mark.parametrize("qlen", [2, 3, 4, 5, 6, 7, 8])
def test_independent_rows(hkv, qlen):
    case = make_case(qlen=qlen, hkv=hkv, context=8193)
    assert not torch.equal(case["q2k_indices"][:, 0], case["q2k_indices"][:, 1])
    assert_numerics(msa_sparse_decode_attention(**case), torch_reference(case))


@pytest.mark.parametrize(
    "context", [8, 127, 128, 129, 255, 2047, 2048, 2049, 100003, 200003, 262144]
)
@pytest.mark.parametrize("hkv", [1, 4])
def test_lengths_and_shared_pages(context, hkv):
    case = make_case(context=context, hkv=hkv)
    result = msa_sparse_decode_attention(**case)
    assert result is case["out"]
    assert_numerics(result, torch_reference(case))
    # Independently inspect the metadata; poisoned unused slots must be ignored.
    w = case["workspace"]
    for t in range(case["q"].shape[0]):
        n = (
            int(case["seqused_k"][t // case["seqlen_q"]])
            - case["seqlen_q"]
            + t % case["seqlen_q"]
            + 1
        )
        count = min(16, (n + 127) // 128)
        for h in range(hkv):
            sel = case["q2k_indices"][h, t, :count].sort().values.long()
            expected_pages = case["page_table"][t // case["seqlen_q"], sel]
            torch.testing.assert_close(
                w._buffers["packed_fp8_pages"][h, t, :count],
                expected_pages,
                rtol=0,
                atol=0,
            )
            length = torch.clamp(n - sel * 128, min=0, max=128).sum()
            assert int(w._buffers["packed_fp8_lengths"][h, t]) == int(length)


@pytest.mark.parametrize("hkv", [1, 4])
def test_graph_replay_updates_every_metadata_input(hkv):
    case = make_case(batch=4, hkv=hkv, context=262144)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            msa_sparse_decode_attention(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        msa_sparse_decode_attention(**case)
    pointers = {k: v.data_ptr() for k, v in case.items() if isinstance(v, torch.Tensor)}
    original_q = case["q"].clone()
    original_kv = packed_kv_cache(case).view(torch.uint8).clone()
    previous = None
    for seed in [11, 23, 37, 51]:
        replacement = make_case(batch=4, hkv=hkv, context=262144, seed=seed)
        # Change tables, lengths and independently selected rows in place.
        for name in ("page_table", "seqused_k", "q2k_indices"):
            if name == "seqused_k":
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
        case["seqused_k"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        case["q2k_indices"].fill_(0x7FFFFFFF)
        for t in range(case["q"].shape[0]):
            n = int(case["seqused_k"][t // 4]) - 4 + t % 4 + 1
            pages = (n + 127) // 128
            for h in range(hkv):
                count = min(16, pages)
                case["q2k_indices"][h, t, :count].copy_(
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
        packed_kv_cache(case).view(torch.uint8), original_kv, rtol=0, atol=0
    )


@pytest.mark.parametrize("strided", [False, True])
def test_no_steady_state_tensor_allocations(monkeypatch, strided):
    case = make_case(hkv=4 if strided else 1)
    if strided:
        total = case["q"].shape[0]
        storage = torch.empty(4, total * 2 + 7, 16, dtype=torch.int32, device="cuda")
        storage[:, :total].copy_(case["q2k_indices"])
        case["q2k_indices"] = storage[:, :total]
    msa_sparse_decode_attention(**case)
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
            msa_sparse_decode_attention(**case)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before


def test_changing_one_row_does_not_change_other_tokens():
    case = make_case(hkv=4)
    original = msa_sparse_decode_attention(**case).clone()
    case["q2k_indices"][2, 3] = torch.arange(16, device="cuda", dtype=torch.int32)
    result = msa_sparse_decode_attention(**case).clone()
    delta = (result != original).any(-1)
    expected_mask = torch.zeros_like(delta)
    expected_mask[3, 32:48] = True
    assert not delta[~expected_mask].any()
    assert delta[expected_mask].any()
    assert_numerics(result, torch_reference(case))


@pytest.mark.parametrize(
    "name", ["q", "k", "v", "q2k_indices", "seqused_k", "k_scale", "v_scale", "out"]
)
def test_invalid_tensor_contract_rejected(name):
    case = make_case(context=129)
    case[name] = case[name].to(torch.float64)
    with pytest.raises(ValueError, match=name):
        msa_sparse_decode_attention(**case)


def test_per_token_scale_rejected_explicitly():
    case = make_case(context=129)
    case["k_scale"] = torch.ones(1, 256, device="cuda")
    with pytest.raises(ValueError, match="scalar"):
        msa_sparse_decode_attention(**case)


def test_zero_dimensional_scales():
    case = make_case(context=129)
    case["k_scale"] = case["k_scale"].reshape(())
    case["v_scale"] = case["v_scale"].reshape(())
    assert_numerics(msa_sparse_decode_attention(**case), torch_reference(case))


@pytest.mark.parametrize("scale_device", ["cpu", "cuda"])
@pytest.mark.parametrize("capturing", [False, True])
def test_tensor_softmax_scale_rejected_without_sync(
    monkeypatch, scale_device, capturing
):
    import flashinfer.jit.blackwell_msa as jit_msa

    case = make_case(context=129)
    case["softmax_scale"] = torch.tensor(0.1, device=scale_device)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Rejected calls must not convert data or load/launch kernels"
        )

    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    monkeypatch.setattr(jit_msa, "load_msa_decode_metadata_module", forbidden)

    def reject():
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(TypeError, match="softmax_scale must be a host float"):
                msa_sparse_decode_attention(**case)
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


@pytest.mark.parametrize("softmax_scale", [0.0625, 0.125])
def test_host_softmax_scale_still_works(softmax_scale):
    case = make_case(context=129)
    case["softmax_scale"] = softmax_scale
    assert_numerics(msa_sparse_decode_attention(**case), torch_reference(case))


def test_unrelated_packed_views_rejected_without_copy():
    case = make_case(hkv=4, context=129)
    case["v"] = case["v"].clone()
    with pytest.raises(ValueError, match="packed"):
        msa_sparse_decode_attention(**case)


@pytest.mark.parametrize("hkv", [1, 4])
@pytest.mark.parametrize("layout", ["indexer", "stepped"])
def test_strided_metadata_graph(hkv, layout):
    case = make_case(batch=4, hkv=hkv, context=4099)
    total = case["q"].shape[0]
    width = 32 if layout == "stepped" else 16
    storage = torch.full(
        (hkv, total * 2 + 7, width), 0x12345678, dtype=torch.int32, device="cuda"
    )
    indices = (
        storage[:, 3 : 3 + total, ::2] if layout == "stepped" else storage[:, :total, :]
    )
    indices.copy_(case["q2k_indices"])
    case["q2k_indices"] = indices
    pages = case["page_table"].shape[1]
    table_storage = torch.full((8, pages + 11), -99, dtype=torch.int32, device="cuda")
    table = table_storage[::2, 5 : 5 + pages]
    table.copy_(case["page_table"])
    case["page_table"] = table
    if layout == "stepped":
        length_storage = torch.zeros(8, dtype=torch.int32, device="cuda")
        length_storage[::2].copy_(case["seqused_k"])
        case["seqused_k"] = length_storage[::2]
    original_storage = storage.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        msa_sparse_decode_attention(**case)
    torch.cuda.current_stream().wait_stream(stream)
    assert_numerics(case["out"], torch_reference(case))
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        msa_sparse_decode_attention(**case)
    case["page_table"].copy_(case["page_table"].roll(1, 1))
    case["k_scale"].fill_(0.9)
    graph.replay()
    torch.cuda.synchronize()
    assert_numerics(case["out"], torch_reference(case))
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)


def test_graph_padding_rows_can_become_active():
    case = make_case(batch=4, context=8193, hkv=4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        msa_sparse_decode_attention(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        msa_sparse_decode_attention(**case)
    for lengths in (
        [0, 8, 0, 129],
        [8193, 0, 17, 0],
        [0, 0, 0, 0],
        [17, 129, 4097, 8193],
    ):
        case["seqused_k"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        case["q2k_indices"].fill_(0x7FFFFFFF)
        for t in range(16):
            n = max(0, lengths[t // 4] - 4 + t % 4 + 1)
            count = min(16, (n + 127) // 128)
            if count:
                for h in range(4):
                    case["q2k_indices"][h, t, :count].copy_(
                        torch.randperm((n + 127) // 128, device="cuda")[:count]
                    )
        graph.replay()
        torch.cuda.synchronize()
        active = (case["seqused_k"] > 0).repeat_interleave(4)
        if any(lengths):
            assert_numerics(case["out"][active], torch_reference(case)[active])
