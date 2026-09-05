"""Shared MSA attention test inputs and fp32 reference implementation.

Extracted from tests/msa_ops/test_cake_msa_sm100.py so that additional MSA
backend test suites (for example the VibeCUDA backend tests) can reuse the
exact same input generator and reference semantics without importing another
test module.
"""

import math
from typing import Any

import pytest
import torch

BLOCK_SIZE = 128
HEAD_DIM = 128
FP8 = "float8_e4m3fn"


def require_supported_msa_gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires an SM100 or SM103 CUDA device")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    minimum_cuda = {(10, 0): (12, 8), (10, 3): (12, 9)}.get(capability)
    if minimum_cuda is None:
        pytest.skip("requires compute capability 10.0 or 10.3")
    cuda_version = torch.version.cuda
    if cuda_version is None:
        pytest.skip("requires a CUDA-enabled PyTorch build")
    version = tuple(int(part) for part in cuda_version.split(".")[:2])
    if version < minimum_cuda:
        pytest.skip(
            f"compute capability {capability[0]}.{capability[1]} requires "
            f"CUDA {minimum_cuda[0]}.{minimum_cuda[1]} or newer"
        )
    return device


def attention_case(
    operation: str,
    *,
    q_dtype: str,
    kv_dtype: str,
    kv_layout: str,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    causal: bool,
    seed: int,
    q_lens: list[int] | None = None,
    kv_lens: list[int] | None = None,
    batch_size: int | None = None,
    seqlen_q: int | None = None,
    seqlen_kv: int | None = None,
    q_offset: list[int] | None = None,
    return_softmax_lse: bool = True,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    selection_mode: str = "random_valid",
    force_fused: bool | None = None,
    cuda_graph: bool = False,
    use_workspace: bool = True,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "q_dtype": q_dtype,
        "kv_dtype": kv_dtype,
        "kv_layout": kv_layout,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "topk": topk,
        "causal": causal,
        "seed": seed,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "q_offset": q_offset,
        "return_softmax_lse": return_softmax_lse,
        "return_temperature_lse": return_temperature_lse,
        "lse_temperature_scale": lse_temperature_scale,
        "selection_mode": selection_mode,
        "force_fused": force_fused,
        "cuda_graph": cuda_graph,
        "use_workspace": use_workspace,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        FP8: torch.float8_e4m3fn,
    }[name]


def indptr(lengths: list[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return result


def _make_q2k(
    *,
    q_lens: list[int],
    kv_lens: list[int],
    kv_heads: int,
    topk: int,
    causal: bool,
    q_offset: torch.Tensor | None,
    selection_mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.full((kv_heads, sum(q_lens), topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_start = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        offset = kv_len - q_len if q_offset is None else int(q_offset[batch].item())
        all_blocks = math.ceil(kv_len / BLOCK_SIZE)
        for local_q in range(q_len):
            for kv_head in range(kv_heads):
                if selection_mode == "future_only":
                    candidates = torch.tensor([1], dtype=torch.int64)
                else:
                    visible_tokens = offset + local_q + 1 if causal else kv_len
                    visible_blocks = max(
                        0,
                        min(all_blocks, math.ceil(visible_tokens / BLOCK_SIZE)),
                    )
                    candidates = torch.randperm(visible_blocks, generator=generator)
                count = min(topk, candidates.numel())
                output[kv_head, q_start + local_q, :count] = (
                    candidates[:count].sort().values.to(torch.int32)
                )
        q_start += q_len
    return output.to(device).contiguous()


def _make_paged_cache(
    logical_k: torch.Tensor,
    logical_v: torch.Tensor,
    kv_lens: list[int],
    kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_counts = [math.ceil(length / BLOCK_SIZE) for length in kv_lens]
    total_pages = sum(page_counts)
    k = torch.zeros(
        (total_pages, kv_heads, BLOCK_SIZE, HEAD_DIM),
        dtype=logical_k.dtype,
        device=device,
    )
    v = torch.zeros_like(k)
    page_table = torch.full(
        (len(kv_lens), max(page_counts)), -1, dtype=torch.int32, device=device
    )
    physical_ids = list(reversed(range(total_pages)))
    token_start = 0
    logical_page = 0
    for batch, kv_len in enumerate(kv_lens):
        for block in range(page_counts[batch]):
            physical_page = physical_ids[logical_page]
            page_table[batch, block] = physical_page
            token_count = min(BLOCK_SIZE, kv_len - block * BLOCK_SIZE)
            source = slice(
                token_start + block * BLOCK_SIZE,
                token_start + block * BLOCK_SIZE + token_count,
            )
            k[physical_page, :, :token_count] = logical_k[source].transpose(0, 1)
            v[physical_page, :, :token_count] = logical_v[source].transpose(0, 1)
            logical_page += 1
        token_start += kv_len
    return k.contiguous(), v.contiguous(), page_table.contiguous()


def make_attention_inputs(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if case["operation"] == "sparse_prefill":
        q_lens = case["q_lens"]
    else:
        q_lens = [case["seqlen_q"]] * case["batch_size"]
    if case["kv_lens"] is not None:
        kv_lens = case["kv_lens"]
    else:
        kv_lens = [case["seqlen_kv"]] * len(q_lens)

    generator = torch.Generator(device=device).manual_seed(case["seed"])
    q = (
        torch.randn(
            (sum(q_lens), case["num_q_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(dtype_from_name(case["q_dtype"]))
    logical_k = (
        torch.randn(
            (sum(kv_lens), case["num_kv_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(dtype_from_name(case["kv_dtype"]))
    logical_v = (
        torch.randn(
            (sum(kv_lens), case["num_kv_heads"], HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(dtype_from_name(case["kv_dtype"]))
    cu_q = indptr(q_lens, device)
    cu_k = indptr(kv_lens, device)
    q_offset = (
        None
        if case["q_offset"] is None
        else torch.tensor(case["q_offset"], dtype=torch.int32, device=device)
    )
    q2k = _make_q2k(
        q_lens=q_lens,
        kv_lens=kv_lens,
        kv_heads=case["num_kv_heads"],
        topk=case["topk"],
        causal=case["causal"],
        q_offset=q_offset,
        selection_mode=case["selection_mode"],
        seed=case["seed"] + 101,
        device=device,
    )

    page_table = None
    seqused_k = None
    if case["kv_layout"] == "paged":
        k, v, page_table = _make_paged_cache(
            logical_k,
            logical_v,
            kv_lens,
            case["num_kv_heads"],
            device,
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    else:
        k, v = logical_k.contiguous(), logical_v.contiguous()

    return {
        **case,
        "q": q.contiguous(),
        "k": k,
        "v": v,
        "logical_k": logical_k.float(),
        "logical_v": logical_v.float(),
        "q2k_indices": q2k,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "q_offset": q_offset,
    }


def reference_attention(inputs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    q = inputs["q"]
    q_lens = inputs["q_lens"]
    kv_lens = inputs["kv_lens"]
    batch_size = len(q_lens)
    max_q = max(q_lens)
    max_k = max(kv_lens)
    q_heads = inputs["num_q_heads"]
    kv_heads = inputs["num_kv_heads"]
    group_size = q_heads // kv_heads
    device = q.device

    q_padded = torch.zeros(
        (batch_size, max_q, q_heads, HEAD_DIM), dtype=torch.float32, device=device
    )
    k_padded = torch.zeros(
        (batch_size, max_k, kv_heads, HEAD_DIM), dtype=torch.float32, device=device
    )
    v_padded = torch.zeros_like(k_padded)
    selections = torch.full(
        (batch_size, max_q, kv_heads, inputs["topk"]),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q_start = 0
    kv_start = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_padded[batch, :q_len] = q[q_start : q_start + q_len].float()
        k_padded[batch, :kv_len] = inputs["logical_k"][kv_start : kv_start + kv_len]
        v_padded[batch, :kv_len] = inputs["logical_v"][kv_start : kv_start + kv_len]
        selections[batch, :q_len] = inputs["q2k_indices"][
            :, q_start : q_start + q_len
        ].permute(1, 0, 2)
        q_start += q_len
        kv_start += kv_len

    token_ids = torch.arange(max_k, device=device)
    block_ids = token_ids // BLOCK_SIZE
    allowed = (block_ids.view(1, 1, 1, max_k, 1) == selections.unsqueeze(-2)).any(-1)
    valid_q = torch.arange(max_q, device=device).view(1, max_q, 1, 1) < torch.tensor(
        q_lens, device=device
    ).view(batch_size, 1, 1, 1)
    valid_k = token_ids.view(1, 1, 1, max_k) < torch.tensor(
        kv_lens, device=device
    ).view(batch_size, 1, 1, 1)
    allowed &= valid_q & valid_k
    if inputs["causal"]:
        if inputs["q_offset"] is None:
            offsets = torch.tensor(
                [kv_len - q_len for q_len, kv_len in zip(q_lens, kv_lens, strict=True)],
                device=device,
            )
        else:
            offsets = inputs["q_offset"].to(torch.long)
        q_positions = offsets.view(batch_size, 1) + torch.arange(
            max_q, device=device
        ).view(1, max_q)
        allowed &= token_ids.view(1, 1, 1, max_k) <= q_positions.view(
            batch_size, max_q, 1, 1
        )

    out = torch.zeros_like(q_padded)
    lse = torch.full(
        (batch_size, max_q, q_heads),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    temperature_lse = torch.full_like(lse, float("-inf"))
    scale = HEAD_DIM**-0.5
    for kv_head in range(kv_heads):
        head_start = kv_head * group_size
        head_end = head_start + group_size
        logits = (
            torch.einsum(
                "bqgd,bkd->bqgk",
                q_padded[:, :, head_start:head_end],
                k_padded[:, :, kv_head],
            )
            * scale
        )
        mask = allowed[:, :, kv_head].unsqueeze(2)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        has_tokens = mask.any(-1)
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = torch.where(
            has_tokens.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
        )
        out[:, :, head_start:head_end] = torch.einsum(
            "bqgk,bkd->bqgd", probabilities, v_padded[:, :, kv_head]
        )
        lse[:, :, head_start:head_end] = torch.logsumexp(masked_logits, dim=-1)
        scaled_logits = (logits * inputs["lse_temperature_scale"]).masked_fill(
            ~mask, float("-inf")
        )
        temperature_lse[:, :, head_start:head_end] = torch.logsumexp(
            scaled_logits, dim=-1
        )

    out_rows = [out[batch, :q_len] for batch, q_len in enumerate(q_lens)]
    lse_rows = [lse[batch, :q_len] for batch, q_len in enumerate(q_lens)]
    temperature_rows = [
        temperature_lse[batch, :q_len] for batch, q_len in enumerate(q_lens)
    ]
    return (
        torch.cat(out_rows).to(q.dtype),
        torch.cat(lse_rows),
        torch.cat(temperature_rows),
    )
