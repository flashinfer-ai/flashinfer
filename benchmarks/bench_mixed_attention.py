import numpy as np
import torch
from triton.testing import do_bench

import flashinfer


def run_bench(
    kv_lens,
    qo_lens,
    page_block_size=1,
    num_kv_heads=4,
    num_qo_heads=28,
    head_dim=128,
    device=0,
    causal=False,
):
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )
    kv_data = torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    ms_old = do_bench(lambda: wrapper_old.run(q, kv_data))

    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        seq_lens.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    ms = do_bench(lambda: wrapper.run(q, kv_data))
    o = wrapper.run(q, kv_data)
    print(o)

    print(f"Elapsed time (old scheduler): {ms_old:.2f} ms")
    print(f"Elapsed time (mixed scheduler): {ms:.2f} ms")

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    print(f"Loading memory size (MB): {total_bytes / (1024**2):.2f} MB")

    bandwidth_old_gb_s = total_bytes / (ms_old * 1e-3) / (1024**3)
    bandwidth_mixed_gb_s = total_bytes / (ms * 1e-3) / (1024**3)

    print(f"Memory bandwidth (old scheduler): {bandwidth_old_gb_s:.2f} GB/s")
    print(f"Memory bandwidth (mixed scheduler): {bandwidth_mixed_gb_s:.2f} GB/s\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)

    seq_len_configs = [
        # [(600, 1)] * 122 + [(10000, 17)] * 8,
        [(64, 16)] * 768,
        # [(400, 1)] * 242 + [(8192, 17)] * 16,
        # [(8192, 1)] * 256,
    ]

    # construct random length testcases
    # for _ in range(1):
    #     bsz = 256
    #     stride = 16
    #     sparsity = 0.05

    #     full_kv_len = np.random.randint(1000, 8192, size=bsz)
    #     seq_len = []
    #     for i in range(bsz):
    #         if i % stride == 0:
    #             kv_len = full_kv_len[i]
    #             qo_len = stride + 1
    #         else:
    #             kv_len = int(full_kv_len[i] * sparsity)
    #             qo_len = 1

    #         seq_len.append((kv_len, qo_len))
    #     seq_len_configs.append(seq_len)

    # for _ in range(1):
    #     bsz = 128
    #     stride = 16
    #     sparsity = 0.05

    #     full_kv_len = np.random.randint(2000, 16000, size=bsz)
    #     seq_len = []
    #     for i in range(bsz):
    #         if i % stride == 0:
    #             kv_len = full_kv_len[i]
    #             qo_len = stride + 1
    #         else:
    #             kv_len = int(full_kv_len[i] * sparsity)
    #             qo_len = 1

    #         seq_len.append((kv_len, qo_len))
    #     seq_len_configs.append(seq_len)

    page_block_size = 1
    num_kv_heads = 4
    num_qo_heads = 32
    head_dim = 128

    for idx, seq_len_pairs in enumerate(seq_len_configs):
        kv_lens = [p[0] for p in seq_len_pairs]
        qo_lens = [p[1] for p in seq_len_pairs]

        print(f"===== Benchmark {idx+1}: (kv_len, qo_len) set =====")
        run_bench(
            kv_lens=kv_lens,
            qo_lens=qo_lens,
            page_block_size=page_block_size,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            device=0,
            causal=True,
        )
