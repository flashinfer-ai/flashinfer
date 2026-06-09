import pytest
import torch

from flashinfer.gpt_oss_ops import reshape_and_cache_fp8
from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
    is_sm12x_supported,
)


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (
        is_sm100a_supported(device)
        or is_sm110a_supported(device)
        or is_sm12x_supported(device)
    ):
        pytest.skip("reshape_and_cache_fp8 requires SM100 or newer")


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("cache_dtype_name", ["uint8", "float8_e4m3fn"])
@pytest.mark.parametrize("scale_ndim", [0, 1])
def test_reshape_and_cache_fp8(num_heads, cache_dtype_name, scale_ndim):
    _require_sm100()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    if cache_dtype_name == "float8_e4m3fn":
        cache_dtype = fp8_dtype
    else:
        cache_dtype = torch.uint8

    torch.manual_seed(0)
    num_tokens = 9
    num_blocks = 3
    block_size = 16
    head_dim = 64
    key = torch.randn(
        (num_tokens, num_heads, head_dim), device="cuda:0", dtype=torch.bfloat16
    )
    value_storage = torch.randn(
        (num_tokens, 10, num_heads, head_dim),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    value = value_storage[:, 0]
    slot_mapping = torch.tensor(
        [0, 1, 2, 15, -1, 16, 31, 33, 100],
        device="cuda:0",
        dtype=torch.int64,
    )
    if scale_ndim == 0:
        k_scale = torch.tensor(1.5, device="cuda:0", dtype=torch.float32)
        v_scale = torch.tensor(0.8, device="cuda:0", dtype=torch.float32)
    else:
        k_scale = torch.tensor([1.5], device="cuda:0", dtype=torch.float32)
        v_scale = torch.tensor([0.8], device="cuda:0", dtype=torch.float32)

    cache_shape = (num_blocks, block_size, num_heads, head_dim)
    cache_stride = (
        block_size * num_heads * head_dim,
        head_dim,
        block_size * head_dim,
        1,
    )

    def make_cache():
        cache = torch.empty_strided(
            cache_shape, cache_stride, device="cuda:0", dtype=cache_dtype
        )
        cache.zero_()
        return cache

    k_ref = make_cache()
    v_ref = make_cache()
    k_out = make_cache()
    v_out = make_cache()

    for token, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0 or slot >= num_blocks * block_size:
            continue
        block = slot // block_size
        offset = slot % block_size
        for head in range(num_heads):
            k_quant = (
                (key[token, head].float() / k_scale).to(fp8_dtype).view(torch.uint8)
            )
            v_quant = (
                (value[token, head].float() / v_scale).to(fp8_dtype).view(torch.uint8)
            )
            k_ref[block, offset, head].view(torch.uint8).copy_(k_quant)
            v_ref[block, offset, head].view(torch.uint8).copy_(v_quant)

    reshape_and_cache_fp8(key, value, k_out, v_out, slot_mapping, k_scale, v_scale)
    torch.cuda.synchronize()

    assert torch.equal(
        k_out.contiguous().view(torch.uint8), k_ref.contiguous().view(torch.uint8)
    )
    assert torch.equal(
        v_out.contiguous().view(torch.uint8), v_ref.contiguous().view(torch.uint8)
    )


def test_reshape_and_cache_fp8_fi_trace():
    num_tokens = 4
    num_heads = 2
    num_blocks = 2
    block_size = 16
    head_dim = 64
    key = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    key_cache = torch.empty(
        (num_blocks, block_size, num_heads, head_dim), dtype=torch.uint8
    )
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.empty((num_tokens,), dtype=torch.int64)
    k_scale = torch.tensor([1.0], dtype=torch.float32)
    v_scale = torch.tensor([1.0], dtype=torch.float32)

    definition = reshape_and_cache_fp8.fi_trace(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert definition["op_type"] == "gpt_oss_reshape_cache_fp8"
    assert definition["axes"]["num_heads"]["value"] == num_heads
    assert definition["axes"]["head_dim"]["value"] == head_dim
    assert definition["axes"]["block_size"]["value"] == block_size
    assert definition["outputs"]["key_cache"]["param"] == "key_cache"
    assert definition["outputs"]["value_cache"]["param"] == "value_cache"
