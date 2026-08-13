import pytest
import torch

from flashinfer.trace.templates.attention import mla_paged_decode_trace


@pytest.mark.parametrize("use_group_scales", [False, True])
def test_mla_paged_decode_fp8_reference_dequantization(use_group_scales):
    head_dim_ckv = 512
    q_nope = torch.zeros(1, 1, head_dim_ckv, dtype=torch.bfloat16)
    q_pe = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    q_pe[..., 0] = 1
    ckv_cache = torch.ones(1, 2, head_dim_ckv, dtype=torch.float8_e4m3fn)
    ckv_cache[:, 1] = 2
    kpe_cache = torch.zeros(1, 2, 64, dtype=torch.float8_e4m3fn)
    kpe_cache[:, 0, 0] = 1
    kpe_cache[:, 1, 0] = 2
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.tensor([0], dtype=torch.int32)
    ckv_scale = 7.0
    kpe_scale = 2.0
    ckv_scale_arr = None
    dequantized_ckv = ckv_cache.float() * ckv_scale

    if use_group_scales:
        ckv_scale_arr = torch.tensor(
            [
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0],
                ]
            ]
        )
        dequantized_ckv = (
            ckv_cache.float().reshape(1, 2, 4, 128) * ckv_scale_arr.unsqueeze(-1)
        ).reshape_as(dequantized_ckv)

    weights = torch.softmax(torch.tensor([kpe_scale, 2 * kpe_scale]), dim=0)
    expected = torch.matmul(weights, dequantized_ckv[0]).reshape(1, 1, head_dim_ckv)
    expected = expected.to(torch.bfloat16)

    output, _ = mla_paged_decode_trace.reference(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
        1.0,
        ckv_scale,
        kpe_scale,
        ckv_scale_arr,
        False,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
