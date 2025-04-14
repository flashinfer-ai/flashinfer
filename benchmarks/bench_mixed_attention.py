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
    causal=True,
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

    print(f"Elapsed time: {ms:.2f} ms")

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    print(f"Loading memory size (MB): {total_bytes / (1024**2):.2f} MB")

    bandwidth_util_percent = 100 * total_bytes / ((ms * 3352 * (1024**3)) / 1000)
    print(f"Memory bandwidth utilization: {bandwidth_util_percent:.4f} %\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)

    seq_len_configs = [
        [(600, 1)] * 122 + [(10000, 17)] * 8,
        [(10000, 1)] * 128,
        [(400, 1)] * 242 + [(8192, 17)] * 16,
        [(8192, 1)] * 256,
    ]

    # construct random length testcases
    for _ in range(1):
        bsz = 256
        stride = 16
        sparsity = 0.05

        full_kv_len = np.random.randint(1000, 8192, size=bsz)
        seq_len = []
        for i in range(bsz):
            if i % stride == 0:
                kv_len = full_kv_len[i]
                qo_len = stride + 1
            else:
                kv_len = int(full_kv_len[i] * sparsity)
                qo_len = 1

            seq_len.append((kv_len, qo_len))
        seq_len_configs.append(seq_len)

    for _ in range(1):
        bsz = 128
        stride = 16
        sparsity = 0.05

        full_kv_len = np.random.randint(2000, 16000, size=bsz)
        seq_len = []
        for i in range(bsz):
            if i % stride == 0:
                kv_len = full_kv_len[i]
                qo_len = stride + 1
            else:
                kv_len = int(full_kv_len[i] * sparsity)
                qo_len = 1

            seq_len.append((kv_len, qo_len))
        seq_len_configs.append(seq_len)

    page_block_size = 1
    num_kv_heads = 4
    num_qo_heads = 28
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
