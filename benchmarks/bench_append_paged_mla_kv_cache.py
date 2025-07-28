import argparse
import dataclasses
from typing import Tuple, cast

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    num_layers: int
    ckv_dim: int = 512
    kpe_dim: int = 64


MODELS = {
    "deepseek_r1": ModelConfig(num_layers=61),
    "deepseek_v2_lite": ModelConfig(num_layers=27),
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--page-len", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    seqlens_ = [
        [1] * args.batch_size,
        [args.seqlen - args.batch_size + 1] + [1] * (args.batch_size - 1),
        [args.seqlen],
        [args.seqlen // args.batch_size] * args.batch_size,
    ]
    seqlen_strlen = max(len(str(seqlens)) for seqlens in seqlens_)
    page_len = int(args.page_len)
    dtype = getattr(torch, args.dtype)
    assert isinstance(dtype, torch.dtype)
    device = torch.device("cuda:0")
    total_pages = int(25600 / page_len)

    torch.cuda.profiler.start()

    for model_name, model in MODELS.items():
        ckv_page_shape = (page_len, model.ckv_dim)
        kpe_page_shape = (page_len, model.kpe_dim)
        ckv_layer_buf = torch.empty(
            (total_pages,) + ckv_page_shape, dtype=dtype, device=device
        )
        kpe_layer_buf = torch.empty(
            (total_pages,) + kpe_page_shape, dtype=dtype, device=device
        )
        for seqlens in seqlens_:
            ckv = torch.rand(
                (sum(seqlens), model.ckv_dim),
                dtype=dtype,
                device=device,
            )
            kpe = torch.rand(
                (sum(seqlens), model.kpe_dim),
                dtype=dtype,
                device=device,
            )
            x_indptr = torch.tensor([0] + seqlens, device=device, dtype=torch.int32)
            x_indptr = torch.cumsum(x_indptr, 0, dtype=torch.int32)
            kv_indices_host = []
            kv_indptr_host = [0]
            next_page_id = 0
            for seqlen in seqlens:
                npages = (seqlen + page_len - 1) // page_len
                kv_indices_host.extend(range(next_page_id, next_page_id + npages))
                next_page_id += npages
                kv_indptr_host.append(len(kv_indices_host))
            kv_indices = torch.tensor(kv_indices_host, device=device, dtype=torch.int32)
            kv_indptr = torch.tensor(kv_indptr_host, device=device, dtype=torch.int32)
            kv_last_page_len = torch.tensor(
                [(seqlen - 1) % page_len + 1 for seqlen in seqlens],
                device=device,
                dtype=torch.int32,
            )

            @torch.cuda.nvtx.range(f"convert model={model_name}, seqlens={seqlens}")
            def fn_convert() -> Tuple[torch.Tensor, torch.Tensor]:
                return flashinfer.get_batch_indices_positions(
                    x_indptr,
                    flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_len),
                    ckv.shape[0],
                )

            batch_indices, positions = fn_convert()
            convert_latencies = bench_gpu_time(
                fn_convert,
                dry_runs=25,
                num_iters=100,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
            )
            convert_latency_ms = np.median(convert_latencies)

            @torch.cuda.nvtx.range(f"append model={model_name}, seqlens={seqlens}")
            def fn() -> None:
                flashinfer.append_paged_mla_kv_cache(
                    ckv,
                    kpe,
                    batch_indices,
                    positions,
                    ckv_layer_buf,
                    kpe_layer_buf,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                )

            latencies = bench_gpu_time(
                fn,
                dry_runs=25,
                num_iters=100,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
            )
            latency_ms = np.median(latencies)
            all_layers_latency_ms = convert_latency_ms + latency_ms * model.num_layers
            throughput = (
                (ckv.numel() + kpe.numel())
                * ckv.element_size()
                * sum(1 for _ in ["read", "write"])
                / (latency_ms * 1e-3)
            )
            print(
                f"model: {model_name:8}",
                f"seqlens: {seqlens!r:{seqlen_strlen}}",
                f"convert: {convert_latency_ms*1e3:2.0f}us",
                f"1layer: {latency_ms*1e3:2.0f}us",
                f"{model.num_layers}layers: {all_layers_latency_ms*1e3:3.0f}us",
                f"throughput: {throughput*1e-9:8.3f}GB/s",
            )
        print("---")

    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
