import math

import pytest
import torch

import flashinfer
from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float


def head_dim_512_supported() -> bool:
    # 16-bit FA2 head_dim > 256 uses the Ampere+ large-head path.
    return torch.cuda.get_device_capability()[0] >= 8


def skip_if_head_dim_unsupported(head_dim: int):
    if head_dim > 256 and not head_dim_512_supported():
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")


def build_causal_mask(qo_len, kv_len):
    i = torch.arange(qo_len).unsqueeze(1).to("cuda:0")
    j = torch.arange(kv_len).unsqueeze(0).to("cuda:0")
    offset = kv_len - qo_len

    mask = (j - offset > i).to(torch.bool)
    return mask


def _repeat_kv(t: torch.Tensor, num_groups: int) -> torch.Tensor:
    return t.repeat_interleave(num_groups, dim=1)


def single_prefill_with_kv_cache_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
):
    Lq, Hq, D = q.shape
    Lk, Hkv, _ = k.shape
    assert (Lk, Hkv, D) == v.shape
    assert Hq % Hkv == 0

    groups = Hq // Hkv
    k_states = _repeat_kv(k, groups)
    v_states = _repeat_kv(v, groups)

    q_t = q.permute(1, 0, 2)  # (Hq, Lq, D)
    k_t = k_states.permute(1, 2, 0)  # (Hq, D, Lk)
    v_t = v_states.permute(1, 0, 2)  # (Hq, Lk, D)

    scale = 1.0 / math.sqrt(D)
    attn_scores = torch.bmm(q_t, k_t) * scale  # (Hq, Lq, Lk)

    if causal:
        # apply causal mask
        causal_mask = build_causal_mask(Lq, Lk)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = torch.bmm(attn_weights, v_t)  # (Hq, Lq, D)
    attn_output = attn_output.permute(1, 0, 2).contiguous()  # (Lq, Hq, D)

    return attn_output


@pytest.mark.parametrize("kv_len", [501, 2042, 3771, 4932])
@pytest.mark.parametrize("qo_len", [37, 127, 577, 1024])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [4, 7])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
def test_sinqle_prefill_with_paged_kv_cache(
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    pos_encoding_mode,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = torch.randn(
        qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    k = torch.randn(
        kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode=pos_encoding_mode, backend="fa2"
    )

    o_ref = single_prefill_with_kv_cache_ref(q, k, v, causal=causal)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


def _run_single_prefill_with_kv_cache_nvfp4(
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
    pos_encoding_mode="NONE",
):
    """Test single_prefill_with_kv_cache with NVFP4 KV cache (contiguous layout).

    KV layout (NHD):
      k/v:    [kv_len, num_kv_heads, head_dim//2]   uint8 (packed FP4x2)
      k/v_sf: [kv_len, num_kv_heads, head_dim//16]  uint8 (FP8 scale factors)

    Reference uses dequantized KV with the standard fp16 kernel.
    """
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    torch.manual_seed(42)

    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)

    kv_shape = (kv_len, num_kv_heads, head_dim // 2)
    k_packed, k_sf, k_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")

    k_dq = nvfp4_to_float(k_packed, k_sf, k_global_scale).to(q_dtype)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_global_scale).to(q_dtype)

    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q,
        k_packed,
        v_packed,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=0.0,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=(k_sf, v_sf),
        backend="fa2",
    )

    o_ref = flashinfer.prefill.single_prefill_with_kv_cache(
        q,
        k_dq,
        v_dq,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=0.0,
    )

    # NVFP4 is 4-bit; use relaxed tolerance
    torch.testing.assert_close(o, o_ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_with_kv_cache_nvfp4(
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
):
    _run_single_prefill_with_kv_cache_nvfp4(
        kv_len,
        qo_len,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        causal,
        q_dtype,
    )


def test_single_prefill_with_kv_cache_nvfp4_large_head():
    skip_if_head_dim_unsupported(512)
    _run_single_prefill_with_kv_cache_nvfp4(
        kv_len=256,
        qo_len=64,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
    )


def test_single_prefill_with_kv_cache_nvfp4_rope_large_head():
    skip_if_head_dim_unsupported(512)
    _run_single_prefill_with_kv_cache_nvfp4(
        kv_len=128,
        qo_len=64,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=512,
        causal=False,
        q_dtype=torch.float16,
        pos_encoding_mode="ROPE_LLAMA",
    )
