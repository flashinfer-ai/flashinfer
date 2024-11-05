import argparse
import dataclasses
from typing import cast

import flashinfer
import torch
from triton.testing import do_bench


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_qo_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int


def _make_70b(tp: int) -> ModelConfig:
    return ModelConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_qo_heads=64,
        num_kv_heads=8 // tp,
        num_layers=80,
        head_dim=128,
    )


MODELS = {
    "l1b": ModelConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_qo_heads=32,
        num_kv_heads=8,
        num_layers=16,
        head_dim=64,
    ),
    "l3b": ModelConfig(
        hidden_size=3072,
        intermediate_size=8192,
        num_qo_heads=24,
        num_kv_heads=8,
        num_layers=28,
        head_dim=128,
    ),
    "l8b": ModelConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_qo_heads=32,
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
        [args.seqlen] + [1] * args.batch_size,
        [args.seqlen],
        [args.seqlen // args.batch_size] * args.batch_size,
    ]
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

            @torch.cuda.nvtx.range(f"model={model_name}, seqlens={seqlens}")
            def fn():
                flashinfer.append_paged_kv_cache(
                    k,
                    v,
                    x_indptr,
                    layer_buf,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    "NHD",
                )

            latency_ms = cast(float, do_bench(fn))
            all_layers_latency_ms = latency_ms * model.num_layers
            throughput = (
                k.numel()
                * k.element_size()
                * sum(1 for _ in ["k", "v"])
                * sum(1 for _ in ["read", "write"])
                / (latency_ms * 1e-3)
            )
            print(
                f"model: {model_name:8}",
                f"seqlens: {str(seqlens):50}",
                f"single_layer: {latency_ms:5.3f}ms",
                f"all_layers: {all_layers_latency_ms:7.3f}ms",
                f"throughput: {throughput*1e-9:8.3f}GB/s",
            )
        print("---")

    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
