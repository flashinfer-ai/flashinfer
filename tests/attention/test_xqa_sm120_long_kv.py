import pytest
import torch

from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count


@pytest.mark.parametrize("packed", [True, False], ids=["packed-hnd", "separate-hnd"])
def test_xqa_sm120_long_kv(packed):
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("requires SM120")
    if torch.cuda.get_device_properties(device).multi_processor_count <= 80:
        pytest.skip("requires a partition of 80 SMs")

    gen = torch.Generator(device=device).manual_seed(816)
    max_len = 32768
    lengths = [1, 17, 129, max_len] * 8
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    # Reuse physical pages so a long logical sequence only needs a small cache.
    kv = torch.randn(
        (7, 2, 8, 16, 64), device=device, dtype=torch.bfloat16, generator=gen
    ).to(torch.float8_e4m3fn)
    k, v = kv.unbind(1)
    cache = kv if packed else (k.contiguous(), v.contiguous())
    table = torch.randint(
        7, (32, max_len // 16), device=device, dtype=torch.int32, generator=gen
    )
    q = torch.randn((32, 64, 64), device=device, dtype=torch.bfloat16, generator=gen)
    workspace = torch.zeros(16 << 20, device=device, dtype=torch.uint8)
    streams, _ = split_device_green_ctx_by_sm_count(device, [80])
    stream = streams[0]
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        output = trtllm_batch_decode_with_kv_cache(
            q,
            cache,
            workspace,
            table,
            lens,
            max_len,
            bmm1_scale=0.125,
            backend="xqa",
            kv_layout="HND",
            enable_pdl=False,
        )
    torch.cuda.current_stream(device).wait_stream(stream)

    for row, length in enumerate(lengths[:4]):
        pages = table[row, : (length + 15) // 16].long()
        keys = k.float()[pages].transpose(0, 1).reshape(8, -1, 64)[:, :length]
        values = v.float()[pages].transpose(0, 1).reshape(8, -1, 64)[:, :length]
        scores = q[row].float().reshape(8, 8, 64) @ keys.transpose(1, 2)
        expected = ((scores * 0.125).softmax(-1) @ values).reshape(64, 64)
        torch.testing.assert_close(output[row].float(), expected, rtol=0.03, atol=0.015)
