import pytest
import torch

pytestmark = pytest.mark.cuda

def _import_flashinfer():
    try:
        import flashinfer as fi
        return fi
    except Exception as e:
        pytest.skip(f"flashinfer import failed in test env: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_single_prefill_short_seq_is_finite():
    fi = _import_flashinfer()

    torch.manual_seed(0)
    device = "cuda"

    # KV cache (typical shapes from README/docs)
    kv_len = 3
    num_kv_heads = 4
    head_dim = 64
    k = torch.randn(kv_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    # Short query length: qo_len = 1 (the edge case we care about)
    qo_len = 1
    num_qo_heads = 4
    q = torch.randn(qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16)

    # Use the public API documented by FlashInfer
    # https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html
    out = fi.single_prefill_with_kv_cache(q, k, v, causal=True)

    # Some APIs may return (out, lse); accept both
    if isinstance(out, (tuple, list)):
        out = out[0]

    assert torch.isfinite(out).all(), "Non-finite values from single_prefill_with_kv_cache on qo_len=1"
    # shape sanity: [qo_len, num_qo_heads, head_dim] for prefill
    assert out.shape[0] == qo_len and out.shape[1] == num_qo_heads
