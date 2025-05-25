import torch
from triton.testing import do_bench

from flashinfer.triton import pad_ragged_tensor_to_multiple_of


def bench_pad_ragged_tensor_to_multiple_of(batch_size, qkv_len, d, multiple_of):
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    indptr = torch.arange(0, (batch_size + 1) * qkv_len, qkv_len, device=device)
    ragged_tensor = torch.randn((indptr[-1], d), device=device)

    ms = do_bench(
        lambda: pad_ragged_tensor_to_multiple_of(ragged_tensor, indptr, multiple_of)
    )
    mem_bandwidth_gb_s = (
        2 * ragged_tensor.numel() * ragged_tensor.element_size() / ms * 1e-6
    )

    print(
        f"batch_size={batch_size}, qkv_len={qkv_len}, d={d}, multiple_of={multiple_of}, ms={ms}, mem_bandwidth={mem_bandwidth_gb_s} GB/s"
    )


if __name__ == "__main__":
    for batch_size in [11, 47, 101]:
        for qkv_len in [500, 1017, 8011]:
            for d in [2048, 4096, 16384]:
                for multiple_of in [128]:
                    bench_pad_ragged_tensor_to_multiple_of(
                        batch_size, qkv_len, d, multiple_of
                    )
