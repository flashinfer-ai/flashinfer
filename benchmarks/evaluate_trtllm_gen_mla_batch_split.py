import argparse
import os

import numpy as np
import torch


LUT_ENV = "FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT"


def dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp8_e4m3", "fp8", "e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def make_benchmark_case(
    batch_size,
    q_len_per_request,
    seq_len,
    page_size,
    dtype,
    num_q_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    workspace_mib,
    fixed_seq_len,
):
    torch.manual_seed(42)
    device = "cuda:0"
    head_dim_qk = kv_lora_rank + qk_rope_head_dim

    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        head_dim_qk,
        device=device,
    ).to(dtype)

    if fixed_seq_len:
        seq_lens = [seq_len] * batch_size
    else:
        seq_lens = [torch.randint(1, seq_len, (1,)).item() for _ in range(batch_size)]
        seq_lens[-1] = seq_len
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = int(blocks_per_seq.max().item())
    total_blocks_needed = int(blocks_per_seq.sum().item())
    block_ids = torch.randperm(total_blocks_needed, device=device)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )

    block_offset = 0
    for batch_idx in range(batch_size):
        num_blocks = int(blocks_per_seq[batch_idx].item())
        block_tables[batch_idx, :num_blocks] = block_ids[
            block_offset : block_offset + num_blocks
        ]
        block_offset += num_blocks

    kv_cache = torch.randn(
        total_blocks_needed,
        1,
        page_size,
        head_dim_qk,
        device=device,
    ).to(dtype)
    workspace_buffer = torch.zeros(
        workspace_mib * 1024 * 1024, dtype=torch.int8, device=device
    )
    out = torch.empty(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    return (
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens_tensor,
        max_seq_len,
        out,
    )


def run_decode(
    query,
    kv_cache,
    workspace_buffer,
    block_tables,
    seq_lens,
    max_seq_len,
    out,
    flashinfer_module,
    backend,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    sm_scale,
):
    flashinfer_module.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        backend=backend,
    )


def calculate_metrics(case, args, execution_time_ms):
    query, _, _, _, seq_lens, _, _ = case
    elem_size = query.element_size()
    actual_kv_tokens = int(seq_lens.sum().item())
    query_bytes = query.numel() * elem_size
    kv_bytes = (
        actual_kv_tokens * (args.kv_lora_rank + args.qk_rope_head_dim) * elem_size
    )
    output_bytes = (
        args.batch_size
        * args.q_len_per_request
        * args.num_qo_heads
        * args.kv_lora_rank
        * elem_size
    )
    total_bytes = query_bytes + kv_bytes + output_bytes
    flops = (
        2
        * args.num_qo_heads
        * (2 * args.kv_lora_rank + args.qk_rope_head_dim)
        * actual_kv_tokens
        * args.q_len_per_request
    )
    return total_bytes / execution_time_ms / 1e6, flops / execution_time_ms / 1e9


def evaluate_trtllm_mla(args):
    import flashinfer as flashinfer_module
    from flashinfer.testing.utils import bench_gpu_time

    dtype = dtype_from_str(args.dtype)
    head_dim_qk = args.kv_lora_rank + args.qk_rope_head_dim
    sm_scale = 1.0 / (head_dim_qk**0.5)
    case = make_benchmark_case(
        args.batch_size,
        args.q_len_per_request,
        args.seq_len,
        args.page_size,
        dtype,
        args.num_qo_heads,
        args.kv_lora_rank,
        args.qk_rope_head_dim,
        args.workspace_mib,
        args.fixed_seq_len,
    )
    input_args = (
        *case,
        flashinfer_module,
        args.backend,
        args.qk_nope_head_dim,
        args.kv_lora_rank,
        args.qk_rope_head_dim,
        sm_scale,
    )
    measurements = bench_gpu_time(
        run_decode,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=False,
        use_cuda_graph=True,
        input_args=input_args,
        cold_l2_cache=True,
    )
    execution_time_ms = float(np.median(measurements))
    bandwidth_gbps, tflops = calculate_metrics(case, args, execution_time_ms)

    print(
        f"backend={args.backend}, batch_size={args.batch_size}, "
        f"q_len_per_request={args.q_len_per_request}, seq_len={args.seq_len}, "
        f"fixed_seq_len={args.fixed_seq_len}, num_q_heads={args.num_qo_heads}, "
        f"qk_nope_head_dim={args.qk_nope_head_dim}, "
        f"qk_rope_head_dim={args.qk_rope_head_dim}, "
        f"kv_lora_rank={args.kv_lora_rank}, page_size={args.page_size}, "
        f"dtype={args.dtype}, workspace_mib={args.workspace_mib}, "
        f"{LUT_ENV}={os.getenv(LUT_ENV, '')}"
    )
    print(f"execution time: {execution_time_ms:.4f} ms")
    print(f"memory bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"FLOPs: {tflops:.2f} TFLOPs/s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TRTLLM-GEN MLA decode with an optional batch-split LUT"
    )
    parser.add_argument("--backend", type=str, default="trtllm-gen")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--q_len_per_request", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=20480)
    parser.add_argument("--page_size", type=int, default=32, choices=[32, 64])
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp8_e4m3",
        choices=["bf16", "bfloat16", "fp8_e4m3", "fp8", "e4m3"],
    )
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--dry_run_iters", type=int, default=5)
    parser.add_argument("--fixed_seq_len", action="store_true")
    parser.add_argument("--num_qo_heads", type=int, default=64)
    parser.add_argument("--head_dim_ckv", type=int, default=512)
    parser.add_argument("--head_dim_kpe", type=int, default=64)
    parser.add_argument("--qk_nope_head_dim", type=int, default=128)
    parser.add_argument("--workspace_mib", type=int, default=256)
    args = parser.parse_args()
    args.kv_lora_rank = args.head_dim_ckv
    args.qk_rope_head_dim = args.head_dim_kpe
    return args


if __name__ == "__main__":
    evaluate_trtllm_mla(parse_args())
