"""Tests for the MagiAttention Flex Flash Attention adapter (flashinfer.magi_ffa).

The dependency-free unit tests (MagiAttention monkeypatched) run anywhere. The
``test_matches_direct_magi_attention_*`` cases exercise the REAL kernel and are
skipped unless MagiAttention is installed and CUDA is available, since MagiAttention
is an optional dependency that FlashInfer does not install.

Running the FlashInfer-validated real path locally (Hopper/H20):

    python -m pip install magi_attention==1.1.0.post10
    # MagiAttention's install can downgrade nvidia-cutlass-dsl below FlashInfer's
    # requirement (>=4.5.0) and break `import flashinfer`; restore it afterwards:
    python -m pip install "nvidia-cutlass-dsl>=4.5.0"
    FLASHINFER_TEST_MAGI_FFA_EXTENDED=1 python -m pytest tests/ffa -v

MagiAttention upstream also supports Blackwell/Ampere through FFA_FA4, but those
configurations have not been validated by FlashInfer under the dependency override
above.

In CI, the real path is covered by the opt-in workflow
``.github/workflows/magi-ffa-optin-test.yml`` (manual, non-blocking).
"""

import math
import os
import types

import pytest
import torch

# ``ffa`` is the implementation module (for monkeypatching the lazy loader);
# ``flex_flash_attn`` is the public function re-exported by the package.
import flashinfer.magi_ffa._flex_flash_attn as ffa
from flashinfer.magi_ffa import flex_flash_attn


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
    # Gate only on the optional dependency + CUDA — NOT on a specific arch.
    # FlashInfer validates Hopper/H20, while MagiAttention owns runtime architecture
    # support (including its upstream FFA_FA4 paths).
    pytest.importorskip("magi_attention")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MagiAttention FFA")


# --------------------------------------------------------------------------------
# Dependency-free unit tests (MagiAttention monkeypatched) — these run in default CI.
# --------------------------------------------------------------------------------


def test_forwards_normalized_args_and_returns_out(monkeypatch):
    captured = {}

    def fake_ffa(**kwargs):
        captured.update(kwargs)
        return torch.full_like(kwargs["q"], 3), types.SimpleNamespace(lse=None)

    monkeypatch.setattr(ffa, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, attn_type_map = _ranges()
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    out = flex_flash_attn(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=0.5,
        softcap=1.0,
        sm_margin=2,
        sparse_load=True,
    )

    # native (out, meta) is hidden; only the normalized out is returned
    assert torch.all(out == 3)
    assert captured["q_ranges"] is q_ranges
    assert captured["k_ranges"] is k_ranges
    assert captured["attn_type_map"] is attn_type_map
    # advanced options are forwarded verbatim to MagiAttention
    assert captured["softmax_scale"] == 0.5
    assert captured["softcap"] == 1.0
    assert captured["sm_margin"] == 2
    assert captured["sparse_load"] is True


def test_return_lse(monkeypatch):
    def fake_ffa(**kwargs):
        out = torch.full_like(kwargs["q"], 5)
        lse = torch.full(
            kwargs["q"].shape[:2], 11.0, dtype=torch.float32, device=kwargs["q"].device
        )
        return out, types.SimpleNamespace(lse=lse)

    monkeypatch.setattr(ffa, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, _ = _ranges()
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    k = torch.zeros((4, 1, 8), dtype=torch.float16)
    v = torch.zeros((4, 1, 8), dtype=torch.float16)

    out, lse = flex_flash_attn(
        q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, return_lse=True
    )
    assert torch.all(out == 5)
    assert torch.all(lse == 11)
    assert lse.shape == (4, 2)


def test_hnd_layout_is_normalized_in_and_out(monkeypatch):
    captured = {}

    def fake_ffa(**kwargs):
        captured.update(kwargs)
        # FFA always sees/returns token-major (NHD)
        out = torch.arange(
            kwargs["q"].numel(), dtype=torch.float32, device=kwargs["q"].device
        ).reshape(kwargs["q"].shape)
        lse = torch.zeros(
            kwargs["q"].shape[:2], dtype=torch.float32, device=kwargs["q"].device
        )
        return out, types.SimpleNamespace(lse=lse)

    monkeypatch.setattr(ffa, "_load_flex_flash_attn_func", lambda: fake_ffa)

    q_ranges, k_ranges, _ = _ranges()
    # HND inputs: (num_heads, num_tokens, head_dim)
    q = torch.zeros((2, 4, 8), dtype=torch.float16)
    k = torch.zeros((1, 4, 8), dtype=torch.float16)
    v = torch.zeros((1, 4, 8), dtype=torch.float16)

    out, lse = flex_flash_attn(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        return_lse=True,
        tensor_layout="HND",
    )

    # MagiAttention was called with token-major (NHD) tensors
    assert captured["q"].shape == (4, 2, 8)
    assert captured["k"].shape == (4, 1, 8)
    # output normalized back to HND
    assert out.shape == (2, 4, 8)
    assert lse.shape == (2, 4)


def test_trace_dispatches_by_tensor_layout():
    q_ranges, k_ranges, attn_type_map = _ranges()
    q = torch.zeros((7, 4, 8), dtype=torch.bfloat16)
    k = torch.zeros((9, 2, 8), dtype=torch.bfloat16)
    v = torch.zeros_like(k)

    nhd = flex_flash_attn.fi_trace(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
    )
    hnd = flex_flash_attn.fi_trace(
        q=q.transpose(0, 1).contiguous(),
        k=k.transpose(0, 1).contiguous(),
        v=v.transpose(0, 1).contiguous(),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        tensor_layout="HND",
    )

    assert nhd["name"] == "magi_ffa_flex_nhd_h4_kv2_d8"
    assert hnd["name"] == "magi_ffa_flex_hnd_h4_kv2_d8"
    assert nhd["inputs"]["q"]["shape"] == [
        "num_tokens_q",
        "num_qo_heads",
        "head_dim",
    ]
    assert hnd["inputs"]["q"]["shape"] == [
        "num_qo_heads",
        "num_tokens_q",
        "head_dim",
    ]


def test_rejects_invalid_tensor_layout():
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    q_ranges, k_ranges, _ = _ranges()
    with pytest.raises(ValueError, match="tensor_layout"):
        flex_flash_attn(
            q, q, q, q_ranges=q_ranges, k_ranges=k_ranges, tensor_layout="BHSD"
        )


def test_rejects_return_max_logits():
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    q_ranges, k_ranges, _ = _ranges()
    with pytest.raises(ValueError, match="return_max_logits"):
        flex_flash_attn(
            q, q, q, q_ranges=q_ranges, k_ranges=k_ranges, return_max_logits=True
        )


def test_rejects_invalid_range_shapes():
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    q_ranges = torch.tensor([0, 4], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    with pytest.raises(ValueError, match="q_ranges"):
        flex_flash_attn(q, q, q, q_ranges=q_ranges, k_ranges=k_ranges)


def test_rejects_mismatched_range_lengths():
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    q_ranges = torch.tensor([[0, 2], [2, 4]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    with pytest.raises(ValueError, match="same length"):
        flex_flash_attn(q, q, q, q_ranges=q_ranges, k_ranges=k_ranges)


def test_rejects_mismatched_attn_type_map_length():
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    q_ranges, k_ranges, _ = _ranges()
    attn_type_map = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="same length"):
        flex_flash_attn(
            q, q, q, q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map
        )


def test_reports_missing_magi_attention_dependency(monkeypatch):
    def missing():
        raise ImportError("MagiAttention is required")

    monkeypatch.setattr(ffa, "_load_flex_flash_attn_func", missing)

    q_ranges, k_ranges, _ = _ranges()
    q = torch.zeros((4, 2, 8), dtype=torch.float16)
    with pytest.raises(ImportError, match="MagiAttention"):
        flex_flash_attn(q, q, q, q_ranges=q_ranges, k_ranges=k_ranges)


# --------------------------------------------------------------------------------
# Real-path correctness tests vs MagiAttention (skipped unless MagiAttention + CUDA).
# --------------------------------------------------------------------------------


def _assert_matches_direct_magi_attention(
    *,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    q_ranges_list,
    k_ranges_list,
    attn_type_map_list,
    extra_kwargs=None,
    deterministic=False,
):
    from magi_attention.api import flex_flash_attn_func

    extra_kwargs = dict(extra_kwargs or {})
    if deterministic:
        extra_kwargs["deterministic"] = True
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
        **extra_kwargs,
    )

    out, lse = flex_flash_attn(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        return_lse=True,
        **extra_kwargs,
    )

    if deterministic:
        # With deterministic=True on both calls, the adapter must be a pure
        # pass-through: results are required to be bit-identical.
        torch.testing.assert_close(out, direct_out, rtol=0, atol=0)
        torch.testing.assert_close(lse, direct_meta.lse, rtol=0, atol=0)
    else:
        # FFA's default path uses atomic reductions, so two independent kernel
        # launches are not guaranteed bit-identical; compare with the repo's
        # standard bf16 tolerances (see flashinfer.trace.template.default_tolerances).
        torch.testing.assert_close(out, direct_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, direct_meta.lse, rtol=1e-2, atol=1e-2)


def test_matches_direct_magi_attention_full_forward():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(42)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[0],
    )


def test_matches_direct_magi_attention_deterministic_bitwise():
    """With deterministic=True on both sides the adapter must be bit-exact.

    This is the strict proof that the adapter is a pure pass-through; the
    default-path tests above use tolerances because FFA's atomic reductions
    make independent launches non-bit-identical.
    """
    _skip_without_magi_attention_cuda()
    torch.manual_seed(52)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[1],
        deterministic=True,
    )


@pytest.mark.parametrize(
    "attn_type",
    [
        pytest.param(0, id="full"),
        pytest.param(1, id="causal"),
        pytest.param(2, id="inverse_causal"),
        pytest.param(3, id="bidirectional_causal"),
    ],
)
def test_matches_direct_magi_attention_mask_types(attn_type):
    _skip_without_magi_attention_cuda()
    torch.manual_seed(42 + attn_type)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[attn_type],
    )


def test_matches_direct_magi_attention_gqa_forward():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(47)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        q_ranges_list=[[0, 64]],
        k_ranges_list=[[0, 64]],
        attn_type_map_list=[0],
    )


def test_matches_direct_magi_attention_multi_range_mixed_masks():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(48)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 32], [32, 64]],
        k_ranges_list=[[0, 32], [32, 64]],
        attn_type_map_list=[0, 1],
    )


def test_matches_direct_magi_attention_complex_multisegment_gqa_masks():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(50)
    _assert_matches_direct_magi_attention(
        seq_len=96,
        num_qo_heads=4,
        num_kv_heads=2,
        q_ranges_list=[[0, 16], [16, 32], [32, 48], [48, 64], [64, 80], [80, 96]],
        k_ranges_list=[[0, 40], [0, 32], [32, 96], [16, 80], [64, 96], [0, 96]],
        attn_type_map_list=[0, 1, 2, 3, 0, 1],
    )


def test_matches_direct_magi_attention_auto_range_merge_extended():
    if os.getenv("FLASHINFER_TEST_MAGI_FFA_EXTENDED") != "1":
        pytest.skip("set FLASHINFER_TEST_MAGI_FFA_EXTENDED=1 to run extended FFA cases")
    _skip_without_magi_attention_cuda()
    torch.manual_seed(49)
    _assert_matches_direct_magi_attention(
        seq_len=64,
        num_qo_heads=2,
        num_kv_heads=2,
        q_ranges_list=[[0, 32], [0, 32]],
        k_ranges_list=[[0, 32], [32, 64]],
        attn_type_map_list=[0, 0],
        extra_kwargs={"auto_range_merge": True, "max_seqlen_q": 64},
    )


def test_hnd_layout_matches_nhd_on_real_kernel():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(51)
    device = torch.device("cuda")
    seq_len, num_heads, head_dim = 64, 2, 128
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    nhd_out = flex_flash_attn(
        q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map
    )
    hnd_out = flex_flash_attn(
        q.transpose(0, 1).contiguous(),
        k.transpose(0, 1).contiguous(),
        v.transpose(0, 1).contiguous(),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        tensor_layout="HND",
    )
    # Two independent kernel launches (atomic reductions): compare with bf16
    # tolerances rather than torch's default atol=1e-5.
    torch.testing.assert_close(
        hnd_out.transpose(0, 1).contiguous(), nhd_out, rtol=1e-2, atol=1e-2
    )


def test_matches_torch_reference_full_forward():
    _skip_without_magi_attention_cuda()
    torch.manual_seed(42)
    device = torch.device("cuda")
    seq_len, num_heads, head_dim = 32, 2, 128
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    out, lse = flex_flash_attn(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        return_lse=True,
    )
    ref_out, ref_lse = _full_attention_reference(q, k, v)
    torch.testing.assert_close(out.float(), ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-2, atol=2e-2)
