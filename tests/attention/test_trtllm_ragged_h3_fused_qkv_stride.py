import math

import pytest
import torch

import flashinfer


NUM_HEADS = 7
HEAD_DIM = 128
FUSED_HEAD_DIM = 3 * HEAD_DIM
WORKSPACE_BYTES = 256 * 1024 * 1024


def _require_b200_trtllm_gen() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("the exact H3 fused-QKV stride smoke requires SM100")
    if torch.cuda.get_device_name(device) != "NVIDIA B200":
        pytest.skip("the exact H3 fused-QKV stride smoke requires NVIDIA B200")
    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("TRTLLM-gen ragged attention is unavailable in this build")


def _make_fused_qkv(
    live_qkv: torch.Tensor,
    total_tokens: int,
    padding_poison: float,
) -> torch.Tensor:
    fused = torch.full(
        (total_tokens, NUM_HEADS, FUSED_HEAD_DIM),
        padding_poison,
        dtype=torch.bfloat16,
        device=live_qkv.device,
    )
    fused[: live_qkv.size(0)].copy_(live_qkv)
    return fused


def _launch(
    fused_qkv: torch.Tensor,
    used_tokens: int,
    workspace: torch.Tensor,
    seq_lens: torch.Tensor,
    indptr: torch.Tensor,
) -> torch.Tensor:
    query = fused_qkv[:used_tokens, :, :HEAD_DIM]
    key = fused_qkv[:used_tokens, :, HEAD_DIM : 2 * HEAD_DIM]
    value = fused_qkv[:used_tokens, :, 2 * HEAD_DIM :]
    expected_stride = (NUM_HEADS * FUSED_HEAD_DIM, FUSED_HEAD_DIM, 1)
    assert query.stride() == expected_stride
    assert key.stride() == expected_stride
    assert value.stride() == expected_stride

    fused_before = fused_qkv.view(torch.int16).clone()
    output = torch.full(
        (fused_qkv.size(0), NUM_HEADS, HEAD_DIM),
        -321.0,
        dtype=torch.bfloat16,
        device=fused_qkv.device,
    )
    output_padding_before = output[used_tokens:].clone()
    live_output = output[:used_tokens]
    result = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=query,
        key=key,
        value=value,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=used_tokens,
        max_kv_len=used_tokens,
        bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
        bmm2_scale=1.0,
        o_sf_scale=-1.0,
        batch_size=1,
        window_left=-1,
        cum_seq_lens_q=indptr,
        cum_seq_lens_kv=indptr,
        enable_pdl=False,
        is_causal=False,
        return_lse=False,
        out=live_output,
        backend="trtllm-gen",
    )
    torch.cuda.synchronize()

    assert isinstance(result, torch.Tensor)
    assert result.data_ptr() == live_output.data_ptr()
    assert torch.equal(fused_qkv.view(torch.int16), fused_before)
    assert torch.equal(output[used_tokens:], output_padding_before)
    return live_output.clone()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "used_tokens,total_tokens",
    [(80, 128), (896, 1024), (1008, 1024)],
    ids=("tail48", "aligned128", "tail16"),
)
def test_trtllm_ragged_h3_fused_qkv_split_view_stride(
    used_tokens: int,
    total_tokens: int,
) -> None:
    """The public runner must consume H3 split views without materialization."""
    _require_b200_trtllm_gen()
    torch.manual_seed(20260810 + used_tokens)
    device = torch.device("cuda")
    live_qkv = (
        0.2
        * torch.randn(
            (used_tokens, NUM_HEADS, FUSED_HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    ).to(torch.bfloat16)
    fused_positive = _make_fused_qkv(live_qkv, total_tokens, 127.0)
    fused_negative = _make_fused_qkv(live_qkv, total_tokens, -127.0)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    seq_lens = torch.tensor([used_tokens], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, used_tokens], dtype=torch.int32, device=device)

    actual_positive = _launch(
        fused_positive,
        used_tokens,
        workspace,
        seq_lens,
        indptr,
    )
    actual_negative = _launch(
        fused_negative,
        used_tokens,
        workspace,
        seq_lens,
        indptr,
    )

    query = live_qkv[:, :, :HEAD_DIM].float().transpose(0, 1)
    key = live_qkv[:, :, HEAD_DIM : 2 * HEAD_DIM].float().transpose(0, 1)
    value = live_qkv[:, :, 2 * HEAD_DIM :].float().transpose(0, 1)
    probabilities = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(HEAD_DIM),
        dim=-1,
    )
    reference = torch.matmul(probabilities, value).transpose(0, 1)

    torch.testing.assert_close(
        actual_positive.float(), reference, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_negative.float(), reference, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_positive, actual_negative, atol=0.0, rtol=0.0
    )
