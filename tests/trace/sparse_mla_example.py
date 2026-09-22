"""Small executable example for the native sparse MLA trace definition."""

import json
from pathlib import Path

import torch
from flashinfer import fi_trace
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata


def generate_sparse_mla_example(
    device="cuda", save_dir=None, assume_valid_prefix=False
):
    q = torch.randn(2, 1, 16, 512, dtype=torch.bfloat16, device=device)
    swa = torch.randn(128, 1, 512, dtype=torch.bfloat16, device=device)
    compressed = torch.randn(64, 1, 512, dtype=torch.bfloat16, device=device)
    si = (
        torch.arange(128, device=device, dtype=torch.int32)
        .view(1, 1, 128)
        .expand(2, 1, 128)
        .contiguous()
    )
    ci = (
        torch.arange(64, device=device, dtype=torch.int32)
        .view(1, 1, 64)
        .expand(2, 1, 64)
        .contiguous()
    )
    sinks = torch.zeros(16, dtype=torch.float32, device=device)
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        2,
        16,
        max_topk=128,
        max_extra_topk=64,
        has_sinks=True,
        return_lse=True,
        assume_valid_prefix=assume_valid_prefix,
    )
    metadata = prepare_sparse_mla_metadata(
        wrapper, q, swa, si, extra_kv_cache=compressed, extra_indices=ci, sinks=sinks
    )
    kwargs = dict(
        query=q, kv_cache=swa, extra_kv_cache=compressed, metadata=metadata, sinks=sinks
    )
    wrapper.run(**kwargs)
    definition = fi_trace(wrapper.run, **kwargs)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / (definition["name"] + ".json")).write_text(
            json.dumps(definition, indent=2) + "\n"
        )
    return definition


def generate_prepared_sparse_mla_example(device="cuda", save_dir=None):
    """One source, with metadata preparation outside the attention call."""
    q = torch.randn(2, 1, 16, 512, dtype=torch.bfloat16, device=device)
    kv = torch.randn(64, 1, 512, dtype=q.dtype, device=device)
    indices = torch.arange(64, device=device, dtype=torch.int32).repeat(2, 1)
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device, 2, 16, max_topk=64, return_lse=True, assume_valid_prefix=True
    )
    metadata = prepare_sparse_mla_metadata(wrapper, q, kv, indices)
    kwargs = dict(query=q, kv_cache=kv, metadata=metadata)
    wrapper.run(**kwargs)
    definition = fi_trace(wrapper.run, **kwargs)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / (definition["name"] + ".json")).write_text(
            json.dumps(definition, indent=2) + "\n"
        )
    return definition


if __name__ == "__main__":
    generate_sparse_mla_example(save_dir=Path(__file__).parent / "fi_trace_out")
    generate_sparse_mla_example(
        save_dir=Path(__file__).parent / "fi_trace_out", assume_valid_prefix=True
    )
    generate_prepared_sparse_mla_example(
        save_dir=Path(__file__).parent / "fi_trace_out"
    )
