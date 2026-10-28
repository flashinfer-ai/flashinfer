# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from flashinfer.trace.template import Const, Scalar, Tensor, TraceTemplate, Var
from flashinfer.trace.templates.attn_scores import (
    _make_paged_block_table,
    _pack_fused_kv_fp8,
)


def _selective_topk_row_scores(
    q, kv_fused, weights, qo_indptr, kv_indptr, block_table, math_mode="relu_strict"
):
    """Return request-local causal score rows for the selective TopK reference."""
    num_blocks, block_size, _one, row_bytes = kv_fused.shape
    head_dim = row_bytes - 4
    flat = kv_fused.reshape(num_blocks, -1)
    kv_fp8 = (
        flat[:, : block_size * head_dim]
        .reshape(num_blocks, block_size, head_dim)
        .view(torch.float8_e4m3fn)
    )
    scales = (
        flat[:, block_size * head_dim :]
        .contiguous()
        .view(torch.float32)
        .reshape(num_blocks, block_size)
    )

    rows = []
    q_prefix = qo_indptr.tolist()
    kv_prefix = kv_indptr.tolist()
    for request in range(len(q_prefix) - 1):
        q_begin, q_end = q_prefix[request : request + 2]
        kv_begin, kv_end = kv_prefix[request : request + 2]
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        page_count = (kv_len + block_size - 1) // block_size
        page_ids = block_table[request, :page_count].long()
        request_kv = kv_fp8[page_ids].reshape(-1, head_dim)[:kv_len].float()
        request_scales = scales[page_ids].reshape(-1)[:kv_len]
        for local_q, row in enumerate(range(q_begin, q_end)):
            row_end = kv_len - q_len + local_q + 1
            per_head = torch.matmul(q[row].float(), request_kv[:row_end].T)
            if math_mode == "relu_fast":
                activated = per_head + per_head.abs()
                scale = request_scales[:row_end] * 0.5
            elif math_mode == "relu_strict":
                activated = torch.relu(per_head)
                scale = request_scales[:row_end]
            else:
                raise ValueError("unknown FP8 math_mode")
            score = (activated * weights[row].float().unsqueeze(1)).sum(dim=0)
            rows.append(score * scale)
    return rows


@torch.no_grad()
def _fp8_paged_mqa_selective_topk_reference(
    q,
    kv_fused,
    weights,
    block_table,
    qo_indptr=None,
    kv_indptr=None,
    top_k=512,
    math_mode="relu_strict",
    **_unused,
):
    """Compute exact request-local causal TopK indices with a PyTorch oracle."""
    if qo_indptr is None or kv_indptr is None:
        raise ValueError(
            "selective TopK reference requires plan-owned qo_indptr and kv_indptr"
        )
    score_rows = _selective_topk_row_scores(
        q, kv_fused, weights, qo_indptr, kv_indptr, block_table, math_mode
    )
    selected = torch.full((q.shape[0], top_k), -1, dtype=torch.int32, device=q.device)
    for row, scores in enumerate(score_rows):
        count = min(top_k, scores.numel())
        selected[row, :count] = torch.topk(
            scores, count, largest=True, sorted=False
        ).indices.to(torch.int32)
    return selected


def _fp8_paged_mqa_selective_topk_check(
    reference_outputs,
    actual_outputs,
    *,
    q=None,
    kv_fused=None,
    weights=None,
    block_table=None,
    qo_indptr=None,
    kv_indptr=None,
    top_k=512,
    math_mode="relu_strict",
    rtol=2e-2,
    atol=2e-2,
    **_unused,
):
    """Validate bounds, uniqueness, padding, and tie-aware TopK membership."""
    actual = actual_outputs[0] if isinstance(actual_outputs, list) else actual_outputs
    reference = (
        reference_outputs[0]
        if isinstance(reference_outputs, list)
        else reference_outputs
    )
    if actual.shape != reference.shape:
        return False
    if any(
        value is None
        for value in (q, kv_fused, weights, block_table, qo_indptr, kv_indptr)
    ):
        return torch.equal(actual, reference)

    score_rows = _selective_topk_row_scores(
        q, kv_fused, weights, qo_indptr, kv_indptr, block_table, math_mode
    )
    for row, scores in enumerate(score_rows):
        valid_count = min(top_k, scores.numel())
        indices = actual[row, :valid_count].long()
        if (indices < 0).any() or (indices >= scores.numel()).any():
            return False
        if torch.unique(indices).numel() != valid_count:
            return False
        if scores.numel() < top_k:
            if not torch.equal(
                torch.sort(indices).values,
                torch.arange(scores.numel(), device=indices.device),
            ):
                return False
            if not bool((actual[row, valid_count:] == -1).all()):
                return False
            continue
        kth = torch.topk(scores, top_k).values[-1]
        tolerance = atol + rtol * kth.abs()
        if (scores[indices] < kth - tolerance).any():
            return False
    return True


def _fp8_paged_mqa_selective_topk_init(
    *,
    total_q: int = 16,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 64,
    max_kv_len: int = 8192,
    top_k: int = 512,
    num_pages: int = 0,
    max_pages_per_request: int = 0,
    batch_plus_one: int = 0,
    num_kv_heads: int = 1,
    block_row_bytes: int = 0,
    device: str = "cuda",
    seed: int = 0,
):
    """Build a full FP8 score-to-TopK wrapper invocation."""
    del num_pages, max_pages_per_request, batch_plus_one, block_row_bytes
    if num_kv_heads != 1:
        raise ValueError("selective FP8 MQA trace requires num_kv_heads=1")
    if total_q < batch_size or max_kv_len < (total_q + batch_size - 1) // batch_size:
        raise ValueError("trace shape requires at least one causal KV row per query")

    torch.manual_seed(seed)
    base = total_q // batch_size
    remainder = total_q % batch_size
    q_lengths = [base + (request < remainder) for request in range(batch_size)]
    q_prefix = [0]
    for length in q_lengths:
        q_prefix.append(q_prefix[-1] + length)
    qo_indptr = torch.tensor(q_prefix, dtype=torch.int32, device=device)
    kv_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * max_kv_len
    )
    context_lens = torch.full(
        (batch_size,), max_kv_len, dtype=torch.int32, device=device
    )
    block_table, physical_pages = _make_paged_block_table(
        context_lens, page_size, device
    )
    q = (torch.randn(total_q, num_heads, head_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(physical_pages, page_size, head_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(
        physical_pages, page_size, dtype=torch.float32, device=device
    ).add_(0.5)
    kv_fused = _pack_fused_kv_fp8(kv, scales, page_size, head_dim)
    weights = torch.randn(total_q, num_heads, dtype=torch.float32, device=device)
    return {
        "constructor": {
            "strategy": "full",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "top_k": top_k,
            "max_context_length": max_kv_len,
            "max_chunk_size": total_q,
            "max_batch_size": batch_size,
            "kv_layout": kv_fused,
        },
        "plan": {
            "qo_indptr": qo_indptr,
            "kv_indptr": kv_indptr,
            "block_table": block_table,
            "num_query_rows": total_q,
        },
        "run": {
            "q": q,
            "kv_fused": kv_fused,
            "weights": weights,
        },
    }


_fp8_paged_mqa_selective_topk_reference._trace_reference_dependencies = (
    _selective_topk_row_scores,
)
_fp8_paged_mqa_selective_topk_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _make_paged_block_table,
    _pack_fused_kv_fp8,
)


class _SelectiveTopKTraceTemplate(TraceTemplate):
    """Inject plan-owned metadata when tracing a live full-strategy wrapper."""

    def build_fi_trace_fn(self, fi_api):
        base_fi_trace = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            wrapper = kwargs.get("self")
            if wrapper is not None:
                state = wrapper._full_state
                batches = wrapper._full_batches
                kwargs.setdefault("qo_indptr", state.qo_indptr[: batches + 1])
                kwargs.setdefault("kv_indptr", state.kv_indptr[: batches + 1])
                kwargs.setdefault("block_table", state.block_table[:batches])
                kwargs.setdefault("top_k", wrapper._full_top_k)
                kwargs["math_mode"] = wrapper.math_mode
                if getattr(wrapper, "compression_ratio", 1) != 1:
                    kwargs["compression_ratio"] = wrapper.compression_ratio
            else:
                # Stable trace consistency tests construct the template without
                # a live wrapper; the public dispatcher still requires one.
                kwargs.setdefault("top_k", 512)
                kwargs.setdefault("math_mode", "relu_strict")
            return base_fi_trace(save_dir=save_dir, name=name, **kwargs)

        return fi_trace


fp8_paged_mqa_topk_trace = _SelectiveTopKTraceTemplate(
    op_type="paged_mqa_topk",
    name_prefix="fp8_paged_mqa_topk",
    description=(
        "Exact causal full FP8 paged MQA scores plus GVR-v2 TopK512, TopK1024 or TopK2048."
    ),
    axes={
        "total_q": Var(description="Total query rows across the ragged batch."),
        "batch_size": Var(description="Number of requests."),
        "batch_plus_one": Var(description="Length of each ragged prefix tensor."),
        "num_heads": Const(abbrev="H", description="Scoring heads."),
        "head_dim": Const(abbrev="D", description="Head dimension."),
        "num_pages": Var(description="Physical paged-KV pool size."),
        "page_size": Const(abbrev="ps", description="Tokens per physical KV page."),
        "num_kv_heads": Const(abbrev="", description="KV heads (MQA => 1)."),
        "block_row_bytes": Const(
            abbrev="", description="Fused KV row bytes (head_dim + 4)."
        ),
        "max_pages_per_request": Var(description="Padded page-table width."),
        "max_kv_len": Var(description="Maximum logical KV length."),
        "top_k": Const(abbrev="k", description="Selected indices per query row."),
    },
    inputs={
        "q": Tensor(
            ["total_q", "num_heads", "head_dim"],
            dtype="float8_e4m3fn",
            description="Contiguous FP8 E4M3 queries.",
        ),
        "kv_fused": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "block_row_bytes"],
            dtype="uint8",
            description="Fused paged FP8 KV and per-token FP32 scales.",
        ),
        "weights": Tensor(
            ["total_q", "num_heads"],
            dtype="float32",
            description="Per-query head weights.",
        ),
        "block_table": Tensor(
            ["batch_size", "max_pages_per_request"],
            dtype="int32",
            optional=True,
            description="Logical-to-physical KV page mapping.",
        ),
        "qo_indptr": Tensor(
            ["batch_plus_one"],
            dtype="int32",
            optional=True,
            description="Plan-owned ragged query prefix sums.",
        ),
        "kv_indptr": Tensor(
            ["batch_plus_one"],
            dtype="int32",
            optional=True,
            description="Plan-owned ragged KV prefix sums.",
        ),
        "top_k": Scalar("int32", optional=True, description="Plan-owned TopK width."),
        "math_mode": Scalar(
            "string", optional=True, description="Constructor-owned FP8 ReLU policy."
        ),
        "max_kv_len": Scalar(
            "int32", optional=True, description="Largest plan-owned KV length."
        ),
    },
    outputs={
        "indices": Tensor(
            ["total_q", "top_k"],
            dtype="int32",
            description="Request-local exact causal TopK indices.",
        )
    },
    constraints=[
        "num_kv_heads == 1",
        "block_row_bytes == head_dim + 4",
        "batch_plus_one == batch_size + 1",
        "top_k in (512, 1024, 2048)",
    ],
    tags=["status:verified", "arch:sm100", "stage:score-to-topk"],
    reference=_fp8_paged_mqa_selective_topk_reference,
    check=_fp8_paged_mqa_selective_topk_check,
    init=_fp8_paged_mqa_selective_topk_init,
)


def _compressed_topk_row_scores(
    q,
    kv_fused,
    weights,
    cu_q,
    cu_kv,
    block_table,
    math_mode="relu_strict",
    compression_ratio=4,
):
    """Independent original-token bounds over already-compressed physical keys."""
    if compression_ratio != 4:
        raise ValueError("compressed TopK trace requires ratio4")
    blocks, page_size, _one, row_bytes = kv_fused.shape
    dim = row_bytes - 4
    flat = kv_fused.flatten(1)
    keys = (
        flat[:, : page_size * dim]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(blocks, page_size, dim)
    )
    scales = (
        flat[:, page_size * dim :]
        .contiguous()
        .view(torch.float32)
        .reshape(blocks, page_size)
    )
    q_prefix, token_prefix = cu_q.tolist(), cu_kv.tolist()
    scores = []
    for request in range(len(q_prefix) - 1):
        qb, qe = q_prefix[request : request + 2]
        tokens = token_prefix[request + 1] - token_prefix[request]
        physical = tokens // compression_ratio
        if physical:
            ids = block_table[request, : (physical + page_size - 1) // page_size].long()
            request_keys = keys[ids].reshape(-1, dim)[:physical].float()
            request_scales = scales[ids].reshape(-1)[:physical]
        for local, row in enumerate(range(qb, qe)):
            end = (tokens - (qe - qb) + local + 1) // compression_ratio
            if end == 0:
                scores.append(torch.empty(0, dtype=torch.float32, device=q.device))
                continue
            per_head = q[row].float() @ request_keys[:end].T
            if math_mode == "relu_strict":
                activated, scale = torch.relu(per_head), request_scales[:end]
            elif math_mode == "relu_fast":
                activated, scale = per_head + per_head.abs(), request_scales[:end] * 0.5
            else:
                raise ValueError("unknown FP8 math_mode")
            scores.append((activated * weights[row, :, None]).sum(0) * scale)
    return scores


@torch.no_grad()
def _compressed_topk_reference(
    q,
    kv_fused,
    weights,
    block_table,
    cu_q=None,
    cu_kv=None,
    top_k=512,
    math_mode="relu_strict",
    compression_ratio=4,
    **_unused,
):
    rows = _compressed_topk_row_scores(
        q, kv_fused, weights, cu_q, cu_kv, block_table, math_mode, compression_ratio
    )
    output = torch.full((q.shape[0], top_k), -1, dtype=torch.int32, device=q.device)
    for row, scores in enumerate(rows):
        count = min(top_k, scores.numel())
        output[row, :count] = scores.topk(count, sorted=False).indices.to(torch.int32)
    return output


def _compressed_topk_check(
    reference_outputs,
    actual_outputs,
    *,
    q,
    kv_fused,
    weights,
    block_table,
    cu_q,
    cu_kv,
    top_k=512,
    math_mode="relu_strict",
    compression_ratio=4,
    atol=5e-5,
    rtol=1e-5,
    **_unused,
):
    actual = actual_outputs[0] if isinstance(actual_outputs, list) else actual_outputs
    if actual.shape != (q.shape[0], top_k) or actual.dtype != torch.int32:
        return False
    rows = _compressed_topk_row_scores(
        q, kv_fused, weights, cu_q, cu_kv, block_table, math_mode, compression_ratio
    )
    for row, scores in enumerate(rows):
        count = min(top_k, scores.numel())
        indices = actual[row, :count].long()
        if (
            (indices < 0).any()
            or (indices >= scores.numel()).any()
            or torch.unique(indices).numel() != count
            or not (actual[row, count:] == -1).all()
        ):
            return False
        if count:
            target = scores.topk(count).values
            if not torch.allclose(
                scores[indices].sort(descending=True).values,
                target,
                atol=atol,
                rtol=rtol,
            ):
                return False
    return True


_compressed_topk_reference._trace_reference_dependencies = (
    _compressed_topk_row_scores,
)


def _capacity_topk_step_is_valid(
    q,
    qo_indptr,
    kv_indptr,
    kv_fused,
    block_table,
    compression_ratio,
    max_context_length=None,
    max_kv_len=None,
    max_chunk_size=None,
    max_batch_size=None,
):
    """Validate device-value semantics before the reference indexes any pages."""
    qo, kv = qo_indptr.tolist(), kv_indptr.tolist()
    batches = len(qo) - 1
    if (
        batches < 0
        or len(kv) != len(qo)
        or q.shape[0] > (max_chunk_size if max_chunk_size is not None else q.shape[0])
        or batches
        > (max_batch_size if max_batch_size is not None else block_table.shape[0])
        or batches > block_table.shape[0]
    ):
        return False
    if qo[0] != 0 or kv[0] != 0:
        return False
    context = max_context_length if max_context_length is not None else max_kv_len
    lengths = []
    for request in range(batches):
        queries = qo[request + 1] - qo[request]
        tokens = kv[request + 1] - kv[request]
        if (
            qo[request] < 0
            or kv[request] < 0
            or queries < 0
            or tokens < queries
            or qo[request + 1] > q.shape[0]
            or (context is not None and tokens > context)
        ):
            return False
        pages = (tokens // compression_ratio + kv_fused.shape[1] - 1) // kv_fused.shape[
            1
        ]
        if pages > block_table.shape[1]:
            return False
        lengths.append(pages)
    for request, pages in enumerate(lengths):
        live = block_table[request, :pages]
        if bool(((live < 0) | (live >= kv_fused.shape[0])).any()):
            return False
    return True


def _capacity_topk_rows(
    q,
    weights,
    qo_indptr,
    kv_indptr,
    kv_fused,
    block_table,
    math_mode="relu_strict",
    compression_ratio=1,
    **capacities,
):
    if compression_ratio not in (1, 4):
        raise ValueError("compression_ratio must be 1 or 4")
    if not _capacity_topk_step_is_valid(
        q,
        qo_indptr,
        kv_indptr,
        kv_fused,
        block_table,
        compression_ratio,
        **capacities,
    ):
        return []
    if compression_ratio == 4:
        rows = _compressed_topk_row_scores(
            q, kv_fused, weights, qo_indptr, kv_indptr, block_table, math_mode, 4
        )
    else:
        rows = _selective_topk_row_scores(
            q, kv_fused, weights, qo_indptr, kv_indptr, block_table, math_mode
        )
    # The full strategy shares the raw score helpers but keeps its own policy.
    return [torch.where(torch.isnan(row), float("-inf"), row) for row in rows]


def _capacity_topk_reference(
    q,
    weights,
    qo_indptr,
    kv_indptr,
    kv_fused,
    block_table,
    top_k=512,
    math_mode="relu_strict",
    compression_ratio=1,
    max_context_length=None,
    max_kv_len=None,
    max_chunk_size=None,
    max_batch_size=None,
    **_unused,
):
    rows = _capacity_topk_rows(
        q,
        weights,
        qo_indptr,
        kv_indptr,
        kv_fused,
        block_table,
        math_mode,
        compression_ratio,
        max_context_length=max_context_length,
        max_kv_len=max_kv_len,
        max_chunk_size=max_chunk_size,
        max_batch_size=max_batch_size,
    )
    output = torch.full((q.shape[0], top_k), -1, dtype=torch.int32, device=q.device)
    for row, scores in enumerate(rows):
        count = min(top_k, scores.numel())
        output[row, :count] = scores.topk(count, sorted=False).indices.to(torch.int32)
    return output


def _capacity_topk_check(
    reference_outputs,
    actual_outputs,
    *,
    q,
    weights,
    qo_indptr,
    kv_indptr,
    kv_fused,
    block_table,
    top_k=512,
    math_mode="relu_strict",
    compression_ratio=1,
    max_context_length=None,
    max_kv_len=None,
    max_chunk_size=None,
    max_batch_size=None,
    atol=5e-5,
    rtol=1e-5,
    **_unused,
):
    actual = actual_outputs[0] if isinstance(actual_outputs, list) else actual_outputs
    if actual.shape != (q.shape[0], top_k) or actual.dtype != torch.int32:
        return False
    rows = _capacity_topk_rows(
        q,
        weights,
        qo_indptr,
        kv_indptr,
        kv_fused,
        block_table,
        math_mode,
        compression_ratio,
        max_context_length=max_context_length,
        max_kv_len=max_kv_len,
        max_chunk_size=max_chunk_size,
        max_batch_size=max_batch_size,
    )
    for row, scores in enumerate(rows):
        count = min(top_k, scores.numel())
        indices = actual[row, :count].long()
        if (
            (indices < 0).any()
            or (indices >= scores.numel()).any()
            or indices.unique().numel() != count
            or not (actual[row, count:] == -1).all()
        ):
            return False
        if count and not torch.allclose(
            scores[indices].sort().values,
            scores.topk(count).values.sort().values,
            atol=atol,
            rtol=rtol,
        ):
            return False
    return bool((actual[len(rows) :] == -1).all())


def _capacity_topk_init(
    *,
    total_q=16,
    batch_size=1,
    num_heads=8,
    head_dim=128,
    page_size=64,
    max_kv_len=8192,
    top_k=512,
    compression_ratio=1,
    num_pages=0,
    max_pages_per_request=0,
    batch_plus_one=0,
    max_chunk_size=0,
    max_batch_size=0,
    device="cuda",
    seed=0,
    **_unused,
):
    del num_pages, max_pages_per_request, batch_plus_one
    max_chunk_size = max_chunk_size or total_q
    max_batch_size = max_batch_size or batch_size
    if max_chunk_size < total_q or max_batch_size < batch_size:
        raise ValueError("capacity must cover the supplied step")
    if compression_ratio not in (1, 4):
        raise ValueError("compression_ratio must be 1 or 4")
    if batch_size <= 0 or total_q <= 0:
        raise ValueError("total_q and batch_size must be positive")
    if max_kv_len < (total_q + batch_size - 1) // batch_size:
        raise ValueError("original KV capacity must cover every request's query rows")
    initial = _fp8_paged_mqa_selective_topk_init(
        total_q=total_q,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_kv_len=max(
            max_kv_len // compression_ratio, (total_q + batch_size - 1) // batch_size
        ),
        top_k=top_k,
        device=device,
        seed=seed,
    )
    initial = {**initial["plan"], **initial["run"]}
    table = initial["block_table"]
    if max_batch_size > batch_size:
        table = torch.cat(
            (
                table,
                torch.zeros(
                    (max_batch_size - batch_size, table.shape[1]),
                    dtype=table.dtype,
                    device=table.device,
                ),
            )
        )
    return {
        "constructor": dict(
            strategy="selective",
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            top_k=top_k,
            compression_ratio=compression_ratio,
            max_context_length=max_kv_len,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            kv_layout=initial["kv_fused"],
        ),
        "plan": dict(
            qo_indptr=initial["qo_indptr"],
            kv_indptr=torch.arange(batch_size + 1, device=device, dtype=torch.int32)
            * max_kv_len,
            block_table=table,
            num_query_rows=total_q,
        ),
        "run": dict(
            q=initial["q"],
            weights=initial["weights"],
            kv_fused=initial["kv_fused"],
        ),
    }


_capacity_topk_reference._trace_reference_dependencies = (  # type: ignore[attr-defined]
    _selective_topk_row_scores,
    _compressed_topk_row_scores,
    _capacity_topk_step_is_valid,
    _capacity_topk_rows,
)
_capacity_topk_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _make_paged_block_table,
    _pack_fused_kv_fp8,
    _fp8_paged_mqa_selective_topk_init,
)


def _capacity_compressed_topk_init(
    *,
    total_q=16,
    batch_size=1,
    num_heads=8,
    head_dim=128,
    page_size=64,
    max_kv_len=8192,
    top_k=512,
    compression_ratio=4,
    num_pages=0,
    max_pages_per_request=0,
    batch_plus_one=0,
    max_chunk_size=0,
    max_batch_size=0,
    device="cuda",
    seed=0,
    **_unused,
):
    if compression_ratio != 4:
        raise ValueError("compressed trace requires compression_ratio=4")
    return _capacity_topk_init(
        total_q=total_q,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_kv_len=max_kv_len,
        top_k=top_k,
        compression_ratio=4,
        num_pages=num_pages,
        max_pages_per_request=max_pages_per_request,
        batch_plus_one=batch_plus_one,
        max_chunk_size=max_chunk_size,
        max_batch_size=max_batch_size,
        device=device,
        seed=seed,
    )


_capacity_compressed_topk_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    *_capacity_topk_init._trace_init_dependencies,  # type: ignore[attr-defined]
    _capacity_topk_init,
)


class _CapacityTopKTraceTemplate(TraceTemplate):
    """Combine the layer pool with the last prepared step's metadata."""

    def build_fi_trace_fn(self, fi_api):
        base = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            wrapper = kwargs.get("self")
            if wrapper is not None:
                plan = wrapper._capacity_plan
                state = plan._state
                batches = plan._supplied_batch
                kwargs["block_table"] = state.block_table
                kwargs["qo_indptr"] = state.qo_indptr[: batches + 1]
                kwargs["kv_indptr"] = state.kv_indptr[: batches + 1]
                kwargs["top_k"], kwargs["math_mode"] = plan.top_k, plan.math_mode
                kwargs["compression_ratio"] = wrapper.compression_ratio
                kwargs["max_kv_len"] = state.capacity.max_context_length
                kwargs["max_chunk_size"] = state.capacity.max_chunk_size
                kwargs["max_batch_size"] = state.capacity.max_batch_size
            if kwargs.get("qo_indptr") is not None:
                kwargs["batch_size"] = kwargs["qo_indptr"].numel() - 1
            return base(save_dir=save_dir, name=name, **kwargs)

        return fi_trace


fp8_paged_mqa_selective_topk_trace = _CapacityTopKTraceTemplate(
    op_type="paged_mqa_topk",
    name_prefix="fp8_paged_mqa_selective_topk",
    description="Capacity-planned exact selective FP8 TopK with device step prefixes, NaN scores ranked as -inf and -1 inactive suffixes.",
    axes={
        **fp8_paged_mqa_topk_trace.axes,
        "max_chunk_size": Var(description="Planned total Q-row capacity."),
        "max_batch_size": Var(description="Planned request capacity."),
        "compression_ratio": Const(
            abbrev="", description="Original tokens per physical key.", value=1
        ),
    },
    inputs={
        "q": fp8_paged_mqa_topk_trace.inputs["q"],
        "weights": fp8_paged_mqa_topk_trace.inputs["weights"],
        "qo_indptr": Tensor(
            ["batch_plus_one"],
            dtype="int32",
            optional=True,
            description="Query prefixes bound by plan.",
        ),
        "kv_indptr": Tensor(
            ["batch_plus_one"],
            dtype="int32",
            optional=True,
            description="Original-token KV prefixes bound by plan.",
        ),
        "kv_fused": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "block_row_bytes"],
            dtype="uint8",
            description="Layer pool matching the planned KV layout.",
        ),
        "block_table": Tensor(
            ["max_batch_size", "max_pages_per_request"],
            dtype="int32",
            optional=True,
            description="Physical page map bound by plan.",
        ),
        "top_k": Scalar(
            "int32", optional=True, description="Constructor-owned TopK width."
        ),
        "math_mode": fp8_paged_mqa_topk_trace.inputs["math_mode"],
        "max_kv_len": Scalar(
            "int32", optional=True, description="Original-token context capacity."
        ),
        "max_chunk_size": Scalar(
            "int32", optional=True, description="Total Q-row capacity."
        ),
        "max_batch_size": Scalar(
            "int32", optional=True, description="Request capacity."
        ),
    },
    outputs={
        "indices": Tensor(
            ["total_q", "top_k"],
            dtype="int32",
            description="Request-local exact causal TopK indices; inactive rows and short-row suffixes are -1.",
        )
    },
    constraints=[
        "num_kv_heads == 1",
        "block_row_bytes == head_dim + 4",
        "batch_plus_one == batch_size + 1",
        "top_k >= 1",
        "top_k <= 2048",
        "total_q <= max_chunk_size",
        "batch_size <= max_batch_size",
        "max_pages_per_request * page_size >= max_kv_len // compression_ratio",
    ],
    tags=fp8_paged_mqa_topk_trace.tags,
    reference=_capacity_topk_reference,
    check=_capacity_topk_check,
    init=_capacity_topk_init,
)

# Compressed selective steps share the device-prefix contract, not full's plan prefixes.
fp8_paged_mqa_compressed_topk_trace = _CapacityTopKTraceTemplate(
    op_type="paged_mqa_topk",
    name_prefix="fp8_paged_mqa_compressed_topk",
    description="Capacity-planned selective FP8 TopK over compressed KV and original-token step prefixes, with NaN scores ranked as -inf.",
    axes={
        **fp8_paged_mqa_selective_topk_trace.axes,
        "compression_ratio": Const(
            abbrev="cr", description="Original tokens per physical key.", value=4
        ),
    },
    inputs={
        **fp8_paged_mqa_selective_topk_trace.inputs,
        "compression_ratio": Scalar(
            "int32", optional=True, description="Constructor-owned ratio4."
        ),
    },
    outputs=fp8_paged_mqa_selective_topk_trace.outputs,
    constraints=[
        *fp8_paged_mqa_selective_topk_trace.constraints,
        "compression_ratio == 4",
    ],
    tags=fp8_paged_mqa_selective_topk_trace.tags,
    reference=_capacity_topk_reference,
    check=_capacity_topk_check,
    init=_capacity_compressed_topk_init,
)


def fp8_paged_mqa_topk_trace_dispatch(**kwargs):
    """Require a planned live wrapper for plan-owned trace metadata."""
    wrapper = kwargs.get("self")
    if wrapper is None:
        raise ValueError(
            "Tracing FP8PagedMQATopKWrapper.run requires the live "
            "wrapper's plan state. Use flashinfer.fi_trace(wrapper.run, ...)."
        )
    if getattr(wrapper, "strategy", None) == "selective":
        if not wrapper._capacity_plan._step_prepared:
            raise RuntimeError("plan() must be called before tracing run()")
        return (
            fp8_paged_mqa_compressed_topk_trace
            if wrapper.compression_ratio == 4
            else fp8_paged_mqa_selective_topk_trace
        )
    if not getattr(wrapper, "_full_step_is_prepared", False):
        raise RuntimeError("plan() must be called before tracing run()")
    return fp8_paged_mqa_topk_trace


fp8_paged_mqa_topk_trace_dispatch.templates = [  # type: ignore[attr-defined]
    fp8_paged_mqa_topk_trace,
    fp8_paged_mqa_selective_topk_trace,
    fp8_paged_mqa_compressed_topk_trace,
]
