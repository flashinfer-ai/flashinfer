"""
Benchmark for sparse MLA (trtllm-gen backend) across a grid of:
  batch_size  : 1, 32, 128, 512
  seqlen_kv   : 1024, 2048, 4096, 8192, 32768
  num_heads_q : 16, 32, 64, 128
  dtype       : bf16 (query+kv+out), e4m3 (query+kv, bf16 out)

DeepSeek-V3 sparse MLA config:
  kv_lora_rank = 512, qk_rope_head_dim = 64, qk_nope_head_dim = 512
  sparse_mla_top_k = min(2048, seqlen_kv)
  page_size = 32
"""

import csv
import math
import random
import sys
from datetime import datetime

import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

# ---------------------------------------------------------------------------
# DeepSeek-V3 MLA dims
# ---------------------------------------------------------------------------
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = KV_LORA_RANK  # = 512
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # = 576
PAGE_SIZE = 32
SPARSE_TOP_K_MAX = 2048

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 32, 128, 512]
SEQLEN_KVS = [1024, 2048, 4096, 8192, 32768]
NUM_HEADS_Q_LIST = [16, 32, 64, 128]
DTYPES = [
    ("bf16", torch.bfloat16, torch.bfloat16),  # (tag, q_dtype, kv_dtype)
    ("e4m3", torch.float8_e4m3fn, torch.float8_e4m3fn),
]

NUM_ITERS = 30
DRY_RUN_ITERS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_sparse_indices(
    batch_size, q_len, seq_lens, topk, page_size, block_tables, device
):
    """Returns indices_in_kvcache: [batch_size, q_len, topk] pointing into the flat KV pool."""
    block_tables_cpu = block_tables.cpu()
    seq_lens_cpu = seq_lens.cpu()

    indices_in_kvcache = torch.empty(
        batch_size, q_len, topk, dtype=torch.int32, device="cpu"
    )

    for i in range(batch_size):
        cur_seq_len = int(seq_lens_cpu[i].item())
        actual_topk = min(topk, cur_seq_len)
        for j in range(q_len):
            cur_abs = torch.arange(0, actual_topk, device="cpu")
            cur_blocked = block_tables_cpu[i, cur_abs // page_size] * page_size + (
                cur_abs % page_size
            )
            if actual_topk < topk:
                pad = torch.full(
                    (topk - actual_topk,), -1, dtype=torch.int32, device="cpu"
                )
                cur_blocked = torch.cat([cur_blocked, pad])
            indices_in_kvcache[i, j, :] = cur_blocked

    return indices_in_kvcache.to(device)


def setup_inputs(batch_size, seqlen_kv, num_heads_q, q_dtype, kv_dtype, device):
    """Create all tensors needed for a sparse MLA decode call."""
    topk = min(SPARSE_TOP_K_MAX, seqlen_kv)
    q_len = 1  # decode phase

    # Query: [B, q_len, H, QK_HEAD_DIM]
    query = torch.randn(batch_size, q_len, num_heads_q, QK_HEAD_DIM, device=device)
    query.clamp_(-1.0, 1.0)
    query = query.to(q_dtype)

    # KV cache pool
    seq_lens = torch.full((batch_size,), seqlen_kv, dtype=torch.int32, device=device)
    blocks_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_blocks = int(blocks_per_seq.sum().item())

    all_block_ids = torch.randperm(total_blocks, device=device)
    max_blocks = int(blocks_per_seq.max().item())
    block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
    bid = 0
    for i in range(batch_size):
        nb = int(blocks_per_seq[i].item())
        block_tables[i, :nb] = all_block_ids[bid : bid + nb]
        bid += nb

    kv_cache = torch.randn(total_blocks, PAGE_SIZE, QK_HEAD_DIM, device=device)
    kv_cache.clamp_(-1.0, 1.0)
    kv_cache = kv_cache.to(kv_dtype)

    # Sparse indices: [B, q_len, topk]
    indices_in_kvcache = generate_sparse_indices(
        batch_size, q_len, seq_lens, topk, PAGE_SIZE, block_tables, device
    )

    # Workspace (zero-initialised, as required)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device=device)

    bmm1_scale = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    bmm2_scale = 1.0

    return dict(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),  # [blocks, 1, page_size, head_dim]
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=indices_in_kvcache,
        seq_lens=seq_lens,
        max_seq_len=seqlen_kv,
        sparse_mla_top_k=topk,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        backend="trtllm-gen",
    )


def run_one(batch_size, seqlen_kv, num_heads_q, dtype_tag, q_dtype, kv_dtype, device):
    topk = min(SPARSE_TOP_K_MAX, seqlen_kv)
    kwargs = setup_inputs(batch_size, seqlen_kv, num_heads_q, q_dtype, kv_dtype, device)

    # Warmup + measure
    measurements = bench_gpu_time(
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla,
        dry_run_iters=DRY_RUN_ITERS,
        repeat_iters=NUM_ITERS,
        enable_cupti=True,
        use_cuda_graph=True,
        input_kwargs=kwargs,
    )
    median_ms = float(torch.tensor(measurements).median().item())
    std_ms = float(torch.tensor(measurements).float().std().item())

    # Memory-bandwidth estimate: kv bytes accessed
    def elem_bytes(dtype):
        return torch.empty(1, dtype=dtype).element_size()

    kv_bytes = batch_size * topk * QK_HEAD_DIM * elem_bytes(kv_dtype)
    q_bytes = batch_size * num_heads_q * QK_HEAD_DIM * elem_bytes(q_dtype)
    o_bytes = batch_size * num_heads_q * KV_LORA_RANK * 2  # bf16 output always
    total_bytes = kv_bytes + q_bytes + o_bytes
    bw_tbs = total_bytes / median_ms / 1e9

    print(
        f"bs={batch_size:4d}  seqkv={seqlen_kv:6d}  H={num_heads_q:3d}  "
        f"dtype={dtype_tag}  topk={topk:5d}  "
        f"median={median_ms:.3f}ms  std={std_ms:.3f}ms  bw={bw_tbs:.2f}TB/s"
    )
    return dict(
        batch_size=batch_size,
        seqlen_kv=seqlen_kv,
        num_heads_q=num_heads_q,
        dtype=dtype_tag,
        sparse_top_k=topk,
        median_ms=median_ms,
        std_ms=std_ms,
        bw_tbs=bw_tbs,
    )


def main():
    device = torch.device("cuda:0")
    cc = get_compute_capability(device)
    if cc[0] != 10:
        print(
            f"ERROR: trtllm-gen sparse MLA requires SM100/SM103, got SM{cc[0]}{cc[1]}"
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"bench_sparse_mla_{timestamp}.csv"

    fieldnames = [
        "batch_size",
        "seqlen_kv",
        "num_heads_q",
        "dtype",
        "sparse_top_k",
        "median_ms",
        "std_ms",
        "bw_tbs",
    ]

    results = []
    total = len(BATCH_SIZES) * len(SEQLEN_KVS) * len(NUM_HEADS_Q_LIST) * len(DTYPES)
    done = 0

    print(f"Running {total} configurations. Results -> {csv_path}\n")
    print(
        f"{'bs':>5} {'seqkv':>7} {'H':>4} {'dtype':>5} {'topk':>6} "
        f"{'median_ms':>10} {'std_ms':>8} {'bw_TB/s':>9}"
    )
    print("-" * 65)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dtype_tag, q_dtype, kv_dtype in DTYPES:
            for num_heads_q in NUM_HEADS_Q_LIST:
                for seqlen_kv in SEQLEN_KVS:
                    for batch_size in BATCH_SIZES:
                        try:
                            row = run_one(
                                batch_size,
                                seqlen_kv,
                                num_heads_q,
                                dtype_tag,
                                q_dtype,
                                kv_dtype,
                                device,
                            )
                            results.append(row)
                            writer.writerow(row)
                            f.flush()
                        except Exception as e:
                            print(
                                f"  SKIP bs={batch_size} seqkv={seqlen_kv} "
                                f"H={num_heads_q} dtype={dtype_tag}: {e}"
                            )
                        done += 1

    print(f"\nDone. {len(results)}/{total} succeeded. CSV saved to {csv_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
