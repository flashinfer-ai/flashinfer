"""Reproduce cross-row NaNs in TRTLLM FP8 sparse MLA on a GB300.

No checkpoint or saved tensors are needed. The first 512 rows have finite queries,
all cache values are 0.25, so every output in these rows should be 0.25. Later
independent rows have NaN queries. Their outputs are intentionally ignored.

Example: python test_trtllm_gen_sparse_mla_nan.py --iterations 100000
Controls: add --mode zero (finite queries), or --mode split (two 32-head calls).
The failure is intermittent; rerun or increase --iterations if it does not occur.
"""

import json
import time

import pytest

import flashinfer
import flashinfer.decode
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="TRTLLM FP8 sparse MLA requires Blackwell",
)
def test_trtllm_gen_sparse_mla_nan_isolation():
    run_repro()


def run_repro(iterations=100000, mode="original"):
    torch.manual_seed(0)
    device = "cuda"
    batch_size = 1024
    top_k = 2048
    query = torch.full(
        (batch_size, 1, 64, 576), 0.5, device=device, dtype=torch.float8_e4m3fn
    )
    if mode != "zero":
        query.view(torch.uint8)[512:] = 127  # FP8 NaN encoding.
    kv_cache = torch.full((32768, 1, 64, 576), 0.25, device=device, dtype=query.dtype)
    rows = torch.arange(batch_size, device=device, dtype=torch.int32)
    columns = torch.arange(top_k, device=device, dtype=torch.int32)
    lengths = torch.randint(
        320, 370, (256,), device=device, dtype=torch.int32
    ).repeat_interleave(4)
    lengths += rows % 4
    lengths[512:] = 2 + rows[512:] % 4
    groups = rows // 4
    indices = torch.where(
        columns[None, :] < 256,
        64 + groups[:, None] * 256 + columns[None, :],
        torch.where(
            columns[None, :] < 320,
            (64 + 128 * 256) + groups[:, None] * 64 + columns[None, :] - 256,
            (64 + 128 * (256 + 64)) + groups[:, None] * 64 + columns[None, :] - 320,
        ),
    )
    indices[512:] = columns[None, :]
    block_tables = (
        torch.where(columns[None, :] < lengths[:, None], indices, -1)
        .unsqueeze(1)
        .contiguous()
    )
    workspace = torch.zeros(384 * 1024 * 1024, device=device, dtype=torch.uint8)
    counter = torch.zeros(2 * 1024 * 1024, device=device, dtype=torch.uint8)
    failures = torch.zeros(3, device=device, dtype=torch.int64)
    first_rows = torch.full((3,), batch_size, device=device, dtype=torch.int64)
    batch_sizes = (1024, 768, 32)

    def step(n, index):
        def attention(q):
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                block_tables=block_tables[:n],
                seq_lens=lengths[:n],
                multi_ctas_kv_counter_buffer=counter,
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                max_seq_len=1048576,
                sparse_mla_top_k=top_k,
                bmm1_scale=0.0625,
                backend="trtllm-gen",
            )

        output = (
            torch.cat(
                [attention(query[:n, :, :32]), attention(query[:n, :, 32:])], dim=2
            )
            if mode == "split"
            else attention(query[:n])
        )
        bad = (~output.isfinite()).flatten(1).any(1) & (rows[:n] < 512)
        failures[index].add_(bad.any().to(torch.int64))
        first_rows[index].copy_(
            torch.minimum(
                first_rows[index], torch.where(bad, rows[:n], batch_size).min()
            )
        )
        return output

    graphs = []
    outputs = []
    for index, n in enumerate(batch_sizes):
        eager_output = step(n, index)
        finite_output = eager_output[: min(n, 512)].float()
        torch.testing.assert_close(
            finite_output, torch.full_like(finite_output, 0.25), atol=0.005, rtol=0
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = step(n, index)
        graphs.append(graph)
        outputs.append(output)
    failures.zero_()
    first_rows.fill_(batch_size)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "flashinfer": flashinfer.__version__,
                "mode": mode,
            }
        ),
        flush=True,
    )
    start = time.monotonic()
    for completed in range(1000, iterations + 1, 1000):
        # All launches are ordered on one stream; scratch reuse is sequential.
        for _ in range(1000):
            graphs[1].replay()
            graphs[2].replay()
            graphs[0].replay()
        counts = failures.tolist()
        if any(counts) or completed % 10000 == 0:
            print(
                json.dumps(
                    {
                        "cycles": completed,
                        "batch_sizes": batch_sizes,
                        "failures": counts,
                        "first_bad_rows": first_rows.tolist(),
                        "elapsed_s": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        if any(counts):
            break
    if any(failures.tolist()):
        raise AssertionError(
            "NaN in an output row whose query and selected KV are all finite"
        )
    print("No NaNs observed in finite rows.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument(
        "--mode", choices=["original", "zero", "split"], default="original"
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.iterations % 1000:
        parser.error("--iterations must be a positive multiple of 1000")
    run_repro(args.iterations, args.mode)
