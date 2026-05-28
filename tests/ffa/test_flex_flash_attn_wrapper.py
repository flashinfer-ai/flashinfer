import math
import os
import types

import pytest
import torch

from flashinfer.ffa_kernels.flex_flash_attn import FlexFlashAttentionWrapper
import flashinfer.ffa_kernels.flex_flash_attn as flex_flash_attn


def _ranges():
    q_ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    attn_type_map = torch.tensor([0], dtype=torch.int32)
    return q_ranges, k_ranges, attn_type_map


def _full_attention_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v.float())
    return out, lse


def _skip_without_magi_attention_cuda():
    pytest.importorskip("magi_attention")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MagiAttention FFA")


def _assert_wrapper_matches_direct_magi_attention(
    *,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    q_ranges_list,
    k_ranges_list,
    attn_type_map_list,
    plan_kwargs=None,
):
    from magi_attention.api import flex_flash_attn_func

    plan_kwargs = plan_kwargs or {}
    device = torch.device("cuda")
    head_dim = 128

    q = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
    attn_type_map = torch.tensor(attn_type_map_list, dtype=torch.int32, device=device)

    direct_out, direct_meta = flex_flash_attn_func(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        **plan_kwargs,
    )

    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map, **plan_kwargs)
    wrapper_out, wrapper_lse = wrapper.run(q, k, v, return_lse=True)

    torch.testing.assert_close(wrapper_out, direct_out, rtol=0, atol=0)
    torch.testing.assert_close(wrapper_lse, direct_meta.lse, rtol=0, atol=0)


def test_plan_caches_magi_attention_range_metadata(monkeypatch):
    captured = {}

    def fake_ffa(**kwargs):
        captured.update(kwargs)
        return torch.ones_like(kwargs["q"]), types.SimpleNamespace(lse=None)

    monkeypatch.setattr(flex_flash_attn, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale=0.5,
        softcap=1.0,
        sm_margin=2,
        sparse_load=True,
    )

    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)
    wrapper.run(q, k, v)

    assert captured["q_ranges"] is q_ranges
    assert captured["k_ranges"] is k_ranges
    assert captured["attn_type_map"] is attn_type_map
    assert captured["softmax_scale"] == 0.5
    assert captured["softcap"] == 1.0
    assert captured["sm_margin"] == 2
    assert captured["sparse_load"] is True


def test_plan_does_not_load_magi_attention_dependency(monkeypatch):
    def fail_load():
        raise AssertionError("plan() should not import MagiAttention")

    monkeypatch.setattr(flex_flash_attn, "_load_flex_flash_attn_func", fail_load)

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)


def test_run_uses_cached_magi_attention_function(monkeypatch):
    load_calls = 0
    run_calls = 0

    def fake_ffa(**kwargs):
        nonlocal run_calls
        run_calls += 1
        return torch.full_like(kwargs["q"], run_calls), types.SimpleNamespace(lse=None)

    def load_once():
        nonlocal load_calls
        load_calls += 1
        return fake_ffa

    monkeypatch.setattr(flex_flash_attn, "_load_flex_flash_attn_func", load_once)

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)

    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    first = wrapper.run(q, k, v)
    second = wrapper.run(q, k, v)

    assert load_calls == 1
    assert run_calls == 2
    assert torch.all(first == 1)
    assert torch.all(second == 2)


def test_run_return_lse_and_copy_to_user_buffers(monkeypatch):
    def fake_ffa(**kwargs):
        out = torch.full_like(kwargs["q"], 3)
        lse = torch.full(
            kwargs["q"].shape[:2], 7, dtype=torch.float32, device=kwargs["q"].device
        )
        return out, types.SimpleNamespace(lse=lse)

    monkeypatch.setattr(flex_flash_attn, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)

    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32)

    actual_out, actual_lse = wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)

    assert actual_out is out
    assert actual_lse is lse
    assert torch.all(out == 3)
    assert torch.all(lse == 7)


def test_run_return_lse_alias(monkeypatch):
    def fake_ffa(**kwargs):
        out = torch.full_like(kwargs["q"], 5)
        lse = torch.full(
            kwargs["q"].shape[:2], 11, dtype=torch.float32, device=kwargs["q"].device
        )
        return out, types.SimpleNamespace(lse=lse)

    monkeypatch.setattr(flex_flash_attn, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)

    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    actual_out, actual_lse = wrapper.run_return_lse(q, k, v)

    assert torch.all(actual_out == 5)
    assert torch.all(actual_lse == 11)


def test_run_without_plan_raises():
    wrapper = FlexFlashAttentionWrapper()
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    with pytest.raises(RuntimeError, match="plan"):
        wrapper.run(q, k, v)


def test_plan_rejects_invalid_range_shapes():
    wrapper = FlexFlashAttentionWrapper()
    q_ranges = torch.tensor([0, 4], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4]], dtype=torch.int32)

    with pytest.raises(ValueError, match="q_ranges"):
        wrapper.plan(q_ranges, k_ranges)


def test_plan_rejects_mismatched_range_lengths():
    wrapper = FlexFlashAttentionWrapper()
    q_ranges = torch.tensor([[0, 2], [2, 4]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4]], dtype=torch.int32)

    with pytest.raises(ValueError, match="same length"):
        wrapper.plan(q_ranges, k_ranges)


def test_plan_rejects_mismatched_attn_type_map_length():
    wrapper = FlexFlashAttentionWrapper()
    q_ranges, k_ranges, _ = _ranges()
    attn_type_map = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="same length"):
        wrapper.plan(q_ranges, k_ranges, attn_type_map)


def test_run_reports_missing_magi_attention_dependency(monkeypatch):
    def missing_magi_attention():
        raise ImportError("MagiAttention is required")

    monkeypatch.setattr(
        flex_flash_attn, "_load_flex_flash_attn_func", missing_magi_attention
    )

    q_ranges, k_ranges, attn_type_map = _ranges()
    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)

    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    with pytest.raises(ImportError, match="MagiAttention"):
        wrapper.run(q, k, v)


def test_run_matches_direct_magi_attention_full_forward():
    _skip_without_magi_attention_cuda()

    from magi_attention.api import flex_flash_attn_func

    torch.manual_seed(42)
    device = torch.device("cuda")
    seq_len = 64
    num_heads = 2
    head_dim = 128

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    direct_out, direct_meta = flex_flash_attn_func(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
    )

    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)
    wrapper_out, wrapper_lse = wrapper.run(q, k, v, return_lse=True)

    torch.testing.assert_close(wrapper_out, direct_out, rtol=0, atol=0)
    torch.testing.assert_close(wrapper_lse, direct_meta.lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "attn_type",
    [
        pytest.param(0, id="full"),
        pytest.param(1, id="causal"),
        pytest.param(2, id="inverse_causal"),
        pytest.param(3, id="bidirectional_causal"),
    ],
)
def test_run_matches_direct_magi_attention_mask_types(attn_type):
    _skip_without_magi_attention_cuda()
    torch.manual_seed(42 + attn_type)

    _assert_wrapper_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[attn_type],
    )


def test_run_matches_direct_magi_attention_gqa_forward():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(47)

    _assert_wrapper_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[0],
    )


def test_run_matches_direct_magi_attention_multi_range_mixed_masks():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(48)

    _assert_wrapper_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 32], [32, 64]],
        k_ranges_list=[[0, 32], [32, 64]],
        attn_type_map_list=[0, 1],
    )


def test_run_matches_direct_magi_attention_complex_multisegment_gqa_masks():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(50)

    _assert_wrapper_matches_direct_magi_attention(
        seq_len=96,
        num_qo_heads=4,
        num_kv_heads=2,
        q_ranges_list=[
            [0, 16],
            [16, 32],
            [32, 48],
            [48, 64],
            [64, 80],
            [80, 96],
        ],
        k_ranges_list=[
            [0, 40],
            [0, 32],
            [32, 96],
            [16, 80],
            [64, 96],
            [0, 96],
        ],
        attn_type_map_list=[0, 1, 2, 3, 0, 1],
    )


def test_run_matches_direct_magi_attention_auto_range_merge_extended():
    if os.getenv("FLASHINFER_TEST_MAGI_FFA_EXTENDED") != "1":
        pytest.skip("set FLASHINFER_TEST_MAGI_FFA_EXTENDED=1 to run extended FFA cases")
    _skip_without_magi_attention_cuda()
    torch.manual_seed(49)

    _assert_wrapper_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 32], [0, 32]],
        k_ranges_list=[[0, 32], [32, 64]],
        attn_type_map_list=[0, 0],
        plan_kwargs={"auto_range_merge": True, "max_seqlen_q": 64},
    )


def test_run_matches_torch_reference_full_forward():
    _skip_without_magi_attention_cuda()

    torch.manual_seed(42)
    device = torch.device("cuda")
    seq_len = 32
    num_heads = 2
    head_dim = 128

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    wrapper = FlexFlashAttentionWrapper()
    wrapper.plan(q_ranges, k_ranges, attn_type_map)
    wrapper_out, wrapper_lse = wrapper.run(q, k, v, return_lse=True)
    ref_out, ref_lse = _full_attention_reference(q, k, v)

    torch.testing.assert_close(wrapper_out.float(), ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(wrapper_lse, ref_lse, rtol=2e-2, atol=2e-2)
