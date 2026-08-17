import pytest
import torch

from flashinfer.mla._cake_dsv4 import _route


@pytest.mark.parametrize(
    "dtype,num_heads,max_q_len,ragged,sparse_topk,expected",
    [
        (torch.bfloat16, 8, 5, True, 192, "bf16_h8_h32"),
        (torch.bfloat16, 64, 5, True, 128, "bf16_swa128_single_cta"),
        (torch.bfloat16, 64, 5, True, 640, "bf16_h64_compressed"),
        (torch.bfloat16, 64, 5, False, 640, "bf16_h64_fixed_q"),
        (torch.bfloat16, 64, 257, True, 640, "bf16_h64_prefill"),
        (torch.bfloat16, 128, 5, True, 128, "bf16_h128_swa128"),
        (torch.bfloat16, 128, 5, True, 1152, "bf16_h128_topk4x"),
        (torch.bfloat16, 128, 5, True, 388, "bf16_h128_topk128x"),
        (torch.bfloat16, 128, 257, True, 1152, "bf16_h128_prefill"),
        (torch.float8_e4m3fn, 8, 5, True, 192, "fp8_lowhead_decode"),
        (torch.float8_e4m3fn, 64, 257, True, 640, "fp8_lowhead_prefill"),
        (torch.float8_e4m3fn, 128, 5, True, 1152, "fp8_h128"),
    ],
)
def test_cake_dsv4_semantic_routes(
    dtype, num_heads, max_q_len, ragged, sparse_topk, expected
):
    assert (
        _route(
            dtype=dtype,
            num_heads=num_heads,
            max_q_len=max_q_len,
            ragged=ragged,
            sparse_topk=sparse_topk,
        )
        == expected
    )
