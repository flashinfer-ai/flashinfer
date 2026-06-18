import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@pytest.mark.cuda
def test_trtllm_ragged_kv_large_stride_overflow():
    """
    Test that ragged KV with large numel (>2^31) doesn't cause TMA descriptor error.

    Constructs a scenario where key.numel() = 131072 * 128 * 192 > 2^31, which
    triggers int32 overflow in kStrideBatch. Before the fix, this caused negative
    stride and TMA descriptor error. After the fix, negative strideBatch is clamped
    to 0 for ragged layouts.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("trtllm_ragged_attention_deepseek is not available in this build")

    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if compute_capability[0] != 10:
        pytest.skip(
            f"TRTLLM-gen ragged attention requires SM100 and SM103 GPUs, got sm{compute_capability[0]}{compute_capability[1]}"
        )

    torch.manual_seed(42)

    # Configuration that triggers numel > 2^31
    batch_size = 16
    max_kv_len = 8192
    num_kv_heads = 128
    head_dim_qk = 192
    head_dim_vo = 128

    # Construct ragged Q
    seq_lens_q = torch.randint(
        low=50, high=150, size=(batch_size,), device=device, dtype=torch.int32
    )
    cum_seq_lens_q = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    total_q = int(cum_seq_lens_q[-1].item())
    max_q_len = int(seq_lens_q.max().item())

    q = torch.randn(
        total_q,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )

    # Construct ragged KV: total_kv = 16 * 8192 = 131072
    # key.numel() = 131072 * 128 * 192 = 3,221,225,472 (0xC0000000) > 2^31
    seq_lens_kv = torch.full(
        (batch_size,), max_kv_len, device=device, dtype=torch.int32
    )
    cum_seq_lens_kv = torch.arange(
        0,
        (batch_size + 1) * max_kv_len,
        max_kv_len,
        device=device,
        dtype=torch.int32,
    )
    total_kv = int(cum_seq_lens_kv[-1].item())

    k = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    scale = float(1.0 / (head_dim_qk**0.5))

    # Should not raise "buildNdTmaDescriptor: invalid argument" error
    output = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens_kv,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        enable_pdl=False,
        is_causal=True,
        return_lse=False,
    )

    # Basic shape check
    assert output.shape[0] == total_q
    assert output.shape[1] == num_kv_heads
    assert output.shape[2] == head_dim_vo
