"""Token-strided Sage quantization must preserve packed-input numerics."""

import pytest
import torch

import flashinfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_inputs(layout, seq_len, heads=14, dim=128, dtype=torch.bfloat16):
    if layout == "plane":
        packed = torch.randn(seq_len, 3, heads, dim, device="cuda", dtype=dtype)
        return tuple(packed[:, i] for i in range(3))
    if layout == "interleaved":
        packed = torch.randn(seq_len, heads, 3 * dim, device="cuda", dtype=dtype)
        return packed.split(dim, dim=-1)
    if layout == "unaligned_stride":
        return tuple(
            torch.randn(
                seq_len * (heads * dim + 1), device="cuda", dtype=dtype
            ).as_strided((seq_len, heads, dim), (heads * dim + 1, dim, 1))
            for _ in range(3)
        )
    if layout == "unaligned_base":
        return tuple(
            torch.randn(seq_len * 3 * heads * dim + 1, device="cuda", dtype=dtype)[
                1:
            ].as_strided((seq_len, heads, dim), (3 * heads * dim, dim, 1))
            for _ in range(3)
        )
    if layout == "zero_stride":
        return tuple(
            torch.randn(1, heads, dim, device="cuda", dtype=dtype).expand(
                seq_len, -1, -1
            )
            for _ in range(3)
        )
    step = 2 if layout == "channel_stride" else 1
    return tuple(
        torch.randn(seq_len, heads, dim * step, device="cuda", dtype=dtype)[..., ::step]
        for _ in range(3)
    )


def require_sm100():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family regression coverage")


def check_bitwise(qkv, **kwargs):
    expected = flashinfer.trtllm_sage_attention_quantize(*qkv, **kwargs)
    actual = flashinfer.trtllm_sage_attention_quantize(
        *qkv, **kwargs, allow_strided_input=True
    )
    for i, (old, new) in enumerate(zip(expected, actual, strict=True)):
        assert new.is_contiguous()
        assert old.shape == new.shape
        bounds = kwargs.get("cum_seq_lens_q")
        if bounds is not None and i in (3, 4):
            block = kwargs["q_block_size"] if i == 3 else kwargs["k_block_size"]
            active = []
            host = bounds.cpu().tolist()
            for b, (start, end) in enumerate(zip(host, host[1:], strict=False)):
                active.extend(
                    range(
                        start // block + b,
                        start // block + b + (end - start + block - 1) // block,
                    )
                )
            old, new = old[:, active], new[:, active]
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))


@pytest.mark.parametrize(
    "layout",
    [
        "plane",
        "interleaved",
        "channel_stride",
        "unaligned_stride",
        "unaligned_base",
        "zero_stride",
        "contiguous",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.int8])
def test_strided_matches_default(layout, dtype, quant_dtype):
    require_sm100()
    torch.manual_seed(42)
    check_bitwise(
        make_inputs(layout, 652, dtype=dtype),
        q_block_size=1,
        k_block_size=16,
        qk_quant_dtype=quant_dtype,
    )


@pytest.mark.parametrize("layout", ["plane", "interleaved"])
def test_h3_real_shape(layout):
    require_sm100()
    check_bitwise(
        make_inputs(layout, 39756),
        q_block_size=1,
        k_block_size=16,
        qk_quant_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "dim,q_block,k_block", [(64, 1, 4), (128, 4, 16), (256, 16, 1)]
)
def test_ragged_bucket_offsets(dim, q_block, k_block):
    require_sm100()
    bounds = torch.tensor([0, 17, 99, 652], device="cuda", dtype=torch.int32)
    check_bitwise(
        make_inputs("plane", 652, heads=3, dim=dim),
        q_block_size=q_block,
        k_block_size=k_block,
        qk_quant_dtype=torch.float8_e4m3fn,
        cum_seq_lens_q=bounds,
        cum_seq_lens_kv=bounds,
    )


def test_smoothing_materializes_key_and_preserves_dequantized_values():
    require_sm100()
    qkv = make_inputs("plane", 652, heads=3, dim=64)
    kwargs = dict(
        q_block_size=1,
        k_block_size=16,
        qk_quant_dtype=torch.float8_e4m3fn,
        smooth_k=True,
    )
    old = flashinfer.trtllm_sage_attention_quantize(*qkv, **kwargs)
    new = flashinfer.trtllm_sage_attention_quantize(
        *qkv, **kwargs, allow_strided_input=True
    )
    for i in range(3):

        def dequantized(result):
            values, scales = result[i].float(), result[i + 3]
            if i == 0:
                return values * scales.T.unsqueeze(-1)
            if i == 1:
                scale = scales.repeat_interleave(16, -1)[
                    :, : values.shape[0]
                ].T.unsqueeze(-1)
                return values * scale + result[6].unsqueeze(0)
            return values * scales.unsqueeze(0)

        torch.testing.assert_close(
            dequantized(old), dequantized(new), atol=0.02, rtol=0.02
        )
