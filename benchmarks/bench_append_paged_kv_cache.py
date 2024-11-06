import argparse
import dataclasses
from typing import Tuple, cast

import torch
from triton.testing import do_bench

import flashinfer


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    num_kv_heads: int
    num_layers: int
    head_dim: int


def _make_70b(tp: int) -> ModelConfig:
    return ModelConfig(
        num_kv_heads=8 // tp,
        num_layers=80,
        head_dim=128,
    )


MODELS = {
    "l1b": ModelConfig(
        num_kv_heads=8,
        num_layers=16,
        head_dim=64,
    ),
    "l3b": ModelConfig(
        num_kv_heads=8,
        num_layers=28,
        head_dim=128,
    ),
    "l8b": ModelConfig(
        num_kv_heads=8,
        num_layers=32,
        head_dim=128,
    ),
    "l70b-tp8": _make_70b(8),
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
    total_pages = int(256000 / page_len)

    torch.cuda.profiler.start()

    for model_name, model in MODELS.items():
        page_shape = (2, page_len, model.num_kv_heads, model.head_dim)
        layer_buf = torch.empty((total_pages,) + page_shape, dtype=dtype, device=device)
        for seqlens in seqlens_:
            k = torch.rand(
                (sum(seqlens), model.num_kv_heads, model.head_dim),
                dtype=dtype,
                device=device,
            )
            v = torch.rand(
                (sum(seqlens), model.num_kv_heads, model.head_dim),
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
                    k.shape[0],
                )

            batch_indices, positions = fn_convert()
            convert_latency_ms = cast(float, do_bench(fn_convert))

            @torch.cuda.nvtx.range(f"append model={model_name}, seqlens={seqlens}")
            def fn() -> None:
                flashinfer.append_paged_kv_cache(
                    k,
                    v,
                    batch_indices,
                    positions,
                    layer_buf,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    "NHD",
                )

            latency_ms = cast(float, do_bench(fn))
            all_layers_latency_ms = convert_latency_ms + latency_ms * model.num_layers
            throughput = (
                k.numel()
                * k.element_size()
                * sum(1 for _ in ["k", "v"])
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
