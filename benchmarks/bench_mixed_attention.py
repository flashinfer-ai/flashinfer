import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def run_bench(
    p_qo_lens,
    p_kv_lens,
    d_qo_lens,
    d_kv_lens,
    # page_block_size=1,
    num_kv_heads=4,
    num_qo_heads=28,
    head_dim=128,
    device=0,
    causal=True,
):
    # POD Attention only supports page size = 1 due to use of single prefill kernel
    page_block_size = 1
    seq_lens = torch.tensor(d_kv_lens + p_kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(d_qo_lens + p_qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
    d_seq_lens_blocks = (
        torch.tensor(d_kv_lens, dtype=torch.int32) / page_block_size
    ).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    d_q_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(d_qo_lens), 0)], dim=0
    ).int()
    d_kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(d_seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(q_indptr[-1].item(), num_qo_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )
    kv_data = torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim).to(
        device, dtype=torch.bfloat16
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv_layout = "NHD"

    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout=kv_layout,
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
    o = wrapper_old.run(q, kv_data)
    measurements = bench_gpu_time(
        lambda: wrapper_old.run(q, kv_data),
        dry_runs=10,
        num_iters=100,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms_old = np.median(measurements)

    if len(p_kv_lens) == 1:
        q_d = q[: d_q_indptr[-1]]
        kv_d = kv_data[: d_kv_indptr[-1]].unbind(1)
        q_p = q[d_q_indptr[-1] :]
        k_p, v_p = kv_data[d_kv_indptr[-1] :].unbind(1)
        k_p, v_p = k_p.squeeze(1), v_p.squeeze(1)
        kv_indices_d = torch.arange(
            0, d_kv_indptr[-1], device=device, dtype=torch.int32
        )

        last_page_len_d = (d_seq_lens_blocks - 1) % page_block_size + 1
        wrapper_pod = flashinfer.PODWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout=kv_layout,
        )
        wrapper_pod.plan(
            d_kv_indptr.to(device),
            kv_indices_d.to(device),
            last_page_len=last_page_len_d,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_block_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        o_p, o_d = wrapper_pod.run(
            q_p,
            k_p,
            v_p,
            q_d,
            kv_data,
            causal_p=causal,
        )
        o_pod = torch.cat([o_d, o_p], dim=0)
        # Verify output matches
        torch.testing.assert_close(
            o, o_pod, rtol=1e-3, atol=1e-3, msg="POD-Attention output mismatch!"
        )
        measurements = bench_gpu_time(
            lambda: wrapper_pod.run(
                q_p,
                k_p,
                v_p,
                q_d,
                kv_d,
                causal_p=causal,
                causal_d=causal,
            )
        )
        ms_pod = np.median(measurements)
    print(f"Elapsed time (Batched Prefill): {ms_old:.2f} ms")
    if len(p_kv_lens) == 1:
        print(f"Elapsed time (POD Attention): {ms_pod:.2f} ms")
    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    print(f"Loading memory size (MB): {total_bytes / (1024**2):.2f} MB")

    bandwidth_old_gb_s = total_bytes / (ms_old * 1e-3) / (1024**3)

    print(f"Memory bandwidth (Batched Prefill): {bandwidth_old_gb_s:.2f} GB/s")
    if len(p_kv_lens) == 1:
        bandwidth_pod_gb_s = total_bytes / (ms_pod * 1e-3) / (1024**3)
        print(f"Memory bandwidth (POD Attention): {bandwidth_pod_gb_s:.2f} GB/s")


if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)

    # Irregular sequence lengths for prefill and decode
    d_q_len_configs = [[1] * 122, [1] * 128, [1] * 242, [1] * 256]
    d_kv_len_configs = [[600] * 122, [10000] * 128, [400] * 242, [8192] * 256]
    p_q_configs = [[17] * 1, [10000], [17] * 1, []]
    p_kv_configs = [[10000] * 1, [10000], [8192] * 1, []]

    # construct random length testcases
    for _ in range(1):
        bsz = 256
        stride = 16
        sparsity = 0.05

        full_kv_len = np.random.randint(1000, 8192, size=bsz)
        p_q_lens = []
        p_kv_lens = []
        d_q_lens = []
        d_kv_lens = []
        for i in range(bsz):
            if i % stride == 0:
                kv_len = full_kv_len[i]
                qo_len = stride + 1
                p_q_lens.append(qo_len)
                p_kv_lens.append(kv_len)
            else:
                kv_len = int(full_kv_len[i] * sparsity)
                qo_len = 1
                d_q_lens.append(qo_len)
                d_kv_lens.append(kv_len)

        p_q_configs.append(p_q_lens)
        p_kv_configs.append(p_kv_lens)
        d_q_len_configs.append(d_q_lens)
        d_kv_len_configs.append(d_kv_lens)

    for _ in range(1):
        bsz = 128
        stride = 16
        sparsity = 0.05

        full_kv_len = np.random.randint(2000, 16000, size=bsz)
        p_q_lens = []
        p_kv_lens = []
        d_q_len = []
        d_kv_len = []

        for i in range(bsz):
            if i % stride == 0:
                kv_len = full_kv_len[i]
                qo_len = stride + 1
                p_q_lens.append(qo_len)
                p_kv_lens.append(kv_len)
            else:
                kv_len = int(full_kv_len[i] * sparsity)
                qo_len = 1
                d_q_lens.append(qo_len)
                d_kv_lens.append(kv_len)

        p_q_configs.append(p_q_lens)
        p_kv_configs.append(p_kv_lens)
        d_q_len_configs.append(d_q_lens)
        d_kv_len_configs.append(d_kv_lens)

    page_block_size = 1
    num_kv_heads = 4
    num_qo_heads = 28
    head_dim = 128

    for idx, (p_q_lens, p_kv_lens, d_q_len, d_kv_len) in enumerate(
        zip(p_q_configs, p_kv_configs, d_q_len_configs, d_kv_len_configs)
    ):

        print(f"===== Benchmark {idx+1}: (kv_len, qo_len) set =====")
        run_bench(
            p_q_lens,
            p_kv_lens,
            d_q_len,
            d_kv_len,
            # page_block_size=page_block_size,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            device=0,
            causal=True,
        )
