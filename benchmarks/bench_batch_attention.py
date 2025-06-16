from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from triton.testing import do_bench

import flashinfer


def run_bench(
    kv_lens: Sequence[int],
    qo_lens: Sequence[int],
    *,
    page_block_size: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    device: int = 0,
    causal: bool = True,
) -> Tuple[float, float, float, float, float]:
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_data = torch.randn(
        num_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # old
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout="NHD",
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks, dtype=torch.int32, device=device),
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

    # new
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    wrapper.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks, dtype=torch.int32, device=device),
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
    ms_new = do_bench(lambda: wrapper.run(q, kv_data))

    total_bytes = (
        q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    )
    mem_MB = total_bytes / 1024**2
    bw_old = total_bytes / (ms_old * 1e-3) / 1024**3
    bw_new = total_bytes / (ms_new * 1e-3) / 1024**3

    return ms_old, ms_new, mem_MB, bw_old, bw_new


def synthesize_seq_len_configs() -> List[List[Tuple[int, int]]]:
    cfgs: List[List[Tuple[int, int]]] = [
        [(8192, 1)] * 128,  # decode-only
        [(4096, 128)] * 4,  # prefill-only
        [(600, 1)] * 122 + [(10_000, 17)] * 8,  # hybird
        [(8192, 1)] * 127 * 2 + [(2048, 512)] * 1,  # hybrid (chunked-prefill)
    ]

    def _rand_case(bsz: int, lo: int, hi: int) -> List[Tuple[int, int]]:
        stride, sparsity = 16, 0.05
        full = np.random.randint(lo, hi, size=bsz)
        out = []
        for i, kv_len in enumerate(full):
            if i % stride == 0:
                out.append((kv_len, stride + 1))
            else:
                out.append((int(kv_len * sparsity), 1))
        return out

    cfgs.append(_rand_case(256, 1000, 8192))
    cfgs.append(_rand_case(128, 2000, 16_000))
    return cfgs


def main() -> None:
    np.random.seed(42)
    torch.random.manual_seed(42)

    seq_len_cfgs = synthesize_seq_len_configs()

    sweep = {
        "page_block_size": (1, 8, 16),
        "head_dim": (64, 128),
        "num_kv_heads": (4,),
        "num_qo_heads": (28,),
    }

    records = []

    for cfg_id, pairs in enumerate(seq_len_cfgs, start=1):
        kv_lens = [p[0] for p in pairs]
        qo_lens = [p[1] for p in pairs]
        for pbs, hd, n_kv, n_qo in itertools.product(
            sweep["page_block_size"],
            sweep["head_dim"],
            sweep["num_kv_heads"],
            sweep["num_qo_heads"],
        ):

            ms_old, ms_new, mem_MB, bw_old, bw_new = run_bench(
                kv_lens,
                qo_lens,
                page_block_size=pbs,
                num_kv_heads=n_kv,
                num_qo_heads=n_qo,
                head_dim=hd,
                device=0,
                causal=True,
            )
            records.extend(
                [
                    {
                        "scheduler": "BatchPrefillWithPagedKVCacheWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "time_ms": ms_old,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_old,
                    },
                    {
                        "scheduler": "BatchAttentionWrapper",
                        "seq_cfg_id": cfg_id,
                        "page_size": pbs,
                        "head_dim": hd,
                        "num_kv_heads": n_kv,
                        "num_qo_heads": n_qo,
                        "time_ms": ms_new,
                        "memory_MB": mem_MB,
                        "bandwidth_GB_s": bw_new,
                    },
                ]
            )

    df = pd.DataFrame(
        records,
        columns=[
            "scheduler",
            "seq_cfg_id",
            "page_size",
            "head_dim",
            "num_kv_heads",
            "num_qo_heads",
            "time_ms",
            "memory_MB",
            "bandwidth_GB_s",
        ],
    )
    print(df.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()
