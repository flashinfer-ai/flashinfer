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

"""TraceTemplates for paged MQA logits (FP8 / FP4 attention-score kernels).

These describe the schema of :func:`flashinfer.fp8_paged_mqa_logits` and
:func:`flashinfer.fp4_paged_mqa_logits` for the fi_trace / flashinfer-bench
system.  Both compute, for each batch element b, speculative slot t, and KV
position pos::

    logits[b*next_n+t, pos] = Σ_h w[b*next_n+t,h] · relu(Q[b,t,h,:]@K[pos,:]ᵀ)

where K is paged via ``block_tables``.  ReLU is applied per head, *before*
weighting and reduction -- not to the sum, so the output is not clamped to be
non-negative.  FP8 multiplies the result by the per-token KV scale
(``· scale[pos]``); FP4 folds its per-(token, K-group) UE8M0 scales into
dequantizing Q and K, so it has no trailing scale factor.

Both kernels run on datacentre Blackwell (SM100/SM103) and Rubin (SM107).
"""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ──────────────────────────────────────────────────────────────────────────────
# FP8 helpers (self-contained so the embedded init/reference source is runnable)
# ──────────────────────────────────────────────────────────────────────────────


def _ceil_to_ue8m0_fp(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _make_paged_block_table(seq_lens, block_size, device):
    """Random paged block table + total block count for a batch of seq_lens.

    Natural width -- ceil(max(seq_lens) / block_size), one entry per KV block a
    request occupies, i.e. the table a paged-KV serving stack already keeps.  The
    kernels predicate every block-table read on the row's own block count, so no
    padding columns are needed; a shorter row's trailing entries (zeros here) are
    never read."""
    n_blk = (seq_lens + block_size - 1) // block_size
    total = int(n_blk.sum().item()) + seq_lens.shape[0] * 2
    max_blk = int(n_blk.max().item())
    block_tables = torch.zeros(
        (seq_lens.shape[0], max_blk), dtype=torch.int32, device=device
    )
    pool = torch.randperm(total, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk.tolist()):
        block_tables[i, :nb] = pool[off : off + nb]
        off += nb
    return block_tables, total


def _pack_fused_kv_fp8(kv_fp8, kv_scales, block_size, head_dim):
    """Pack fp8 KV + per-token fp32 scales into the fused layout used by the kernel:
    per physical block = [all KV bytes (block_size*D)] [all scale bytes (block_size*4)]."""
    num_blocks = kv_fp8.shape[0]
    scale_offset = block_size * head_dim
    fused = torch.zeros(
        num_blocks,
        block_size * (head_dim + 4),
        dtype=torch.uint8,
        device=kv_fp8.device,
    )
    for blk in range(num_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def _validate_fp8_logits_trace_precision(
    output_dtype=torch.float32,
    epi_dtype=torch.float32,
    acc_dtype=torch.float32,
):
    """Keep unsupported arithmetic precision out of the FP32 trace recipe."""
    if epi_dtype is not torch.float32 or acc_dtype is not torch.float32:
        raise ValueError("FP8 logits trace supports only FP32 epi_dtype and acc_dtype")
    if output_dtype not in (torch.float32, torch.float16):
        raise ValueError("FP8 logits trace supports only FP32 or FP16 output_dtype")


@torch.no_grad()
def _fp8_paged_mqa_logits_reference(
    q,
    kv_fused,
    weights,
    block_tables,
    seq_lens,
    max_seq_len,
    output_dtype=torch.float32,
    math_mode="relu_strict",
    *,
    epi_dtype=torch.float32,
    acc_dtype=torch.float32,
):
    """Pure-torch reference operating on the packed FP8 API inputs.

    Unpacks ``kv_fused`` (flat [all KV][all scales] per block) back into fp8 KV
    and fp32 per-token scales. Only FP32 epilogue/accumulation is modeled;
    output conversion happens at the final store, not in intermediate math."""
    _validate_fp8_logits_trace_precision(output_dtype, epi_dtype, acc_dtype)
    num_blocks, block_size, _one, row_bytes = kv_fused.shape
    head_dim = row_bytes - 4
    B, next_n, H, _D = q.shape
    device = q.device

    flat = kv_fused.reshape(num_blocks, -1)  # [num_blocks, block_size*(D+4)]
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

    q_f32 = q.float()
    logits = torch.full(
        (B * next_n, max_seq_len), float("-inf"), device=device, dtype=output_dtype
    )
    for b in range(B):
        seq_len = int(seq_lens[b].item())
        q_pos = torch.arange(seq_len - next_n, seq_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].float()
        for blk in range((seq_len + block_size - 1) // block_size):
            phys = int(block_tables[b, blk].item())
            k = kv_fp8[phys].float()
            sc = scales[phys].float()
            kpos = torch.arange(blk * block_size, (blk + 1) * block_size, device=device)
            mask = (kpos[None, :] < seq_len) & (kpos[None, :] <= q_pos[:, None])
            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k.T)
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            if math_mode == "relu_fast":
                activated = qk + qk.abs()
                scale = sc * 0.5
            elif math_mode == "relu_strict":
                activated = torch.relu(qk)
                scale = sc
            else:
                raise ValueError("unknown FP8 math_mode")
            weighted = (w.T[:, :, None] * activated).sum(dim=0) * scale[None, :]
            # max_seq_len is a free axis and need not be block-aligned, so
            # the last physical block can extend past the output width. Clip
            # both the destination and the right-hand side to it.
            s = blk * block_size
            e = min(s + block_size, max_seq_len)
            if s >= max_seq_len:
                break
            width = e - s
            logits[b * next_n : (b + 1) * next_n, s:e] = torch.where(
                mask[:, :width],
                weighted[:, :width],
                torch.tensor(float("-inf"), device=device, dtype=output_dtype),
            )
    return logits


_fp8_paged_mqa_logits_reference._trace_reference_dependencies = (
    _validate_fp8_logits_trace_precision,
)


def _paged_mqa_logits_masked_check(
    reference_outputs,
    actual_outputs,
    *,
    seq_lens=None,
    next_n=None,
    rtol=2e-2,
    atol=2e-2,
    **_unused,
):
    """Compare kernel logits vs reference over the causal, within-length region.

    Positions beyond each row's causal limit (and the padded trailing columns the
    kernel may write) are excluded.  A non-finite *reference* is tolerated --
    that is the fp8/fp4 accumulation-order argument -- but a non-finite *actual*
    is a kernel failure and fails the check.  Excluding both sides symmetrically
    made an all-NaN kernel output pass: every NaN position was dropped from
    ``valid``, ``valid.any()`` went False, and the checker returned True having
    compared nothing.  Uses an element-wise tolerance plus a relative-L2 floor."""
    ref = (
        reference_outputs[0]
        if isinstance(reference_outputs, (list, tuple))
        else reference_outputs
    )
    act = (
        actual_outputs[0]
        if isinstance(actual_outputs, (list, tuple))
        else actual_outputs
    )
    if ref.shape != act.shape:
        return False
    device = ref.device
    rows, max_len = ref.shape
    if seq_lens is None or next_n is None:
        neginf_mask = torch.zeros((rows, max_len), dtype=torch.bool, device=device)
    else:
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(rows, -1)
        offsets = torch.arange(rows, device=device)
        limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
        neginf_mask = ~(positions <= limits)

    r = ref.float().masked_fill(neginf_mask, 0)
    a = act.float().masked_fill(neginf_mask, 0)
    # A non-finite value the KERNEL produced inside the causal region is a
    # failure, never something to filter out. Checked before `valid` is built so
    # it cannot be masked away by its own presence.
    in_region = ~neginf_mask
    if in_region.any() and not torch.isfinite(a[in_region]).all():
        return False

    # A non-finite REFERENCE is tolerated: fp8/fp4 accumulation order can differ
    # at extremes, and that is a property of the comparison target, not of the
    # kernel under test.
    valid = in_region & torch.isfinite(r)
    if not valid.any():
        # Nothing comparable. Only a vacuously-empty causal region is a pass;
        # a region that exists but whose reference is entirely non-finite gives
        # no evidence either way, so do not claim success.
        return not in_region.any()

    rv, av = r[valid], a[valid]
    if not torch.allclose(av, rv, rtol=rtol, atol=atol):
        # Fallback: relative-L2 error ||a-r|| / ||r||. Tolerates a few boundary
        # mismatches and fp8/fp4 quantization noise (both well under 5%), but —
        # unlike a loose cosine floor — REJECTS a systematic k*ref scale error
        # (rel-L2 == |k-1|), so a mis-scaled / bad-weight-cast kernel cannot pass.
        rnorm = rv.double().norm()
        if rnorm > 0:
            if float((av.double() - rv.double()).norm() / rnorm) > 0.05:
                return False
        else:
            # An all-zero reference makes the relative error undefined; the old
            # code substituted 0.0, which no threshold can exceed, so ANY actual
            # passed. Fall back to an absolute comparison against zero.
            if not torch.allclose(av, torch.zeros_like(av), rtol=0, atol=atol):
                return False
    return True


def _fp8_paged_mqa_logits_init(
    *,
    batch_size: int,
    next_n: int = 1,
    num_heads: int = 64,
    head_dim: int = 128,
    block_size: int = 64,
    max_seq_len: int = 4096,
    output_dtype: torch.dtype = torch.float32,
    math_mode: str = "relu_strict",
    device: str = "cuda",
    seed: int = 0,
):
    """Build valid inputs for ``flashinfer.fp8_paged_mqa_logits`` (all rows at max_seq_len)."""
    _validate_fp8_logits_trace_precision(output_dtype)
    torch.manual_seed(seed)
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    block_tables, num_blocks = _make_paged_block_table(seq_lens, block_size, device)

    q = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(num_blocks, block_size, head_dim, device=device)
    kv_amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _pack_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)
    return {
        "q": q,
        "kv_fused": kv_fused,
        "weights": weights,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": max_seq_len,
        "output_dtype": output_dtype,
        "math_mode": math_mode,
    }


_fp8_paged_mqa_logits_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _ceil_to_ue8m0_fp,
    _make_paged_block_table,
    _pack_fused_kv_fp8,
    _validate_fp8_logits_trace_precision,
)


class _FP8LogitsTraceTemplate(TraceTemplate):
    """Reject scorer semantics the exported ordinary FP32 recipe cannot model."""

    def build_fi_trace_fn(self, fi_api):
        base_fi_trace = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            if (
                kwargs.get("query_metadata") is not None
                or kwargs.get("output_rows") is not None
                or getattr(kwargs.get("kernel_config"), "query_tile", 1) > 1
            ):
                raise ValueError(
                    "fp8_paged_mqa_logits trace export does not support grouped "
                    "query metadata or output rows"
                )
            _validate_fp8_logits_trace_precision(
                kwargs.get("output_dtype", torch.float32),
                kwargs.get("epi_dtype", torch.float32),
                kwargs.get("acc_dtype", torch.float32),
            )
            return base_fi_trace(save_dir=save_dir, name=name, **kwargs)

        return fi_trace


fp8_paged_mqa_logits_trace = _FP8LogitsTraceTemplate(
    op_type="paged_mqa_logits",
    name_prefix="fp8_paged_mqa_logits",
    description=(
        "FP8 paged MQA logits: weighted dot-product attention scores against a paged "
        "KV cache, with per-token FP32 KV scales. Datacentre Blackwell "
        "(SM100/SM103) and Rubin (SM107)."
    ),
    axes={
        "batch_size": Var(description="Number of decode requests."),
        "next_n": Const(
            abbrev="nn", description="Speculative-decode slots per request."
        ),
        "num_heads": Const(abbrev="H", description="Number of query heads."),
        "head_dim": Const(
            abbrev="D", description="Head dimension (FP8 elements per row)."
        ),
        "num_kv_heads": Const(abbrev="", description="KV heads (MQA => 1)."),
        "block_size": Const(abbrev="bs", description="Tokens per physical KV block."),
        "block_row_bytes": Const(
            abbrev="", description="Fused KV row bytes (head_dim + 4)."
        ),
        "max_seq_len": Var(description="Maximum KV sequence length."),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "next_n", "num_heads", "head_dim"],
            dtype="float8_e4m3fn",
            description="Query, FP8 e4m3.",
        ),
        "kv_fused": Tensor(
            ["num_blocks", "block_size", "num_kv_heads", "block_row_bytes"],
            dtype="uint8",
            description="Fused paged KV: [KV fp8 bytes][per-token fp32 scale bytes] per block.",
        ),
        "weights": Tensor(
            ["num_rows", "num_heads"],
            dtype="float32",
            description="Per-head mixing weights.",
        ),
        "block_tables": Tensor(
            ["batch_size", "max_blocks_per_seq"],
            dtype="int32",
            description="Paged block indices.",
        ),
        "seq_lens": Tensor(
            ["batch_size"], dtype="int32", description="Per-request KV length."
        ),
        "max_seq_len": Scalar("int32", description="Maximum KV sequence length."),
        "output_dtype": Scalar(
            "dtype", optional=True, description="Final storage dtype; FP32 by default."
        ),
        "math_mode": Scalar(
            "string",
            optional=True,
            description="FP32 epilogue policy: relu_strict (default) or relu_fast.",
        ),
    },
    outputs={
        "logits": Tensor(
            ["num_rows", "max_seq_len"],
            dtype="float32",
            dtype_from="out",
            dtype_from_scalar="output_dtype",
            description="Attention-score logits.",
        ),
    },
    constraints=[
        "num_rows == batch_size * next_n",
        "block_row_bytes == head_dim + 4",
        "num_kv_heads == 1",
    ],
    tags=["status:verified", "arch:sm100", "arch:sm103", "arch:sm107"],
    reference=_fp8_paged_mqa_logits_reference,
    check=_paged_mqa_logits_masked_check,
    init=_fp8_paged_mqa_logits_init,
)


# ------------------------------------------------------------------------------
# Selective FP8 paged MQA TopK wrapper
# ------------------------------------------------------------------------------


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
        "plan": {
            "q": q,
            "kv_fused": kv_fused,
            "weights": weights,
            "qo_indptr": qo_indptr,
            "kv_indptr": kv_indptr,
            "block_table": block_table,
            "top_k": top_k,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "run": {
            "q": q,
            "kv_fused": kv_fused,
            "weights": weights,
            "block_table": block_table,
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
    """Inject plan-owned metadata when tracing a live selective wrapper."""

    def build_fi_trace_fn(self, fi_api):
        base_fi_trace = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            wrapper = kwargs.get("self")
            if wrapper is not None:
                kwargs.setdefault("qo_indptr", getattr(wrapper, "_cu_q", None))
                kwargs.setdefault("kv_indptr", getattr(wrapper, "_cu_kv", None))
                kwargs.setdefault("top_k", getattr(wrapper, "_top_k", 512))
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
    )["plan"]
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
    if getattr(wrapper, "_full_graph_identity", None) is None:
        raise RuntimeError("plan() must be called before tracing run()")
    return fp8_paged_mqa_topk_trace


fp8_paged_mqa_topk_trace_dispatch.templates = [  # type: ignore[attr-defined]
    fp8_paged_mqa_topk_trace,
    fp8_paged_mqa_selective_topk_trace,
    fp8_paged_mqa_compressed_topk_trace,
]


# ──────────────────────────────────────────────────────────────────────────────
# FP4 (MXFP4) helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ceil_to_ue8m0_int(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def _unpack_ue8m0_from_int(packed: torch.Tensor) -> torch.Tensor:
    return (packed.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    return (code | (sign.to(torch.uint8) << 3)).view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float
    )
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def _per_token_cast_to_fp4(x: torch.Tensor, gran_k: int = 32):
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = _ceil_to_ue8m0_int(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), _pack_ue8m0_to_int(sf)


def _cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 32):
    m, n2 = packed.shape
    n = n2 * 2
    sf = _unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x = _dequantize_from_fp4_e2m1(unpacked)
    return x * sf[:, torch.arange(n, device=packed.device) // gran_k]


def _pack_kv_cache_fp4(kv_cache):
    """bf16 KV cache [num_blocks, block_size, 1, D] -> fused uint8
    [num_blocks, block_size, 1, (D/2)+4] (data bytes then SF bytes), plus the
    dequantized bf16 reference."""
    num_blocks, block_size, num_heads, head_dim = kv_cache.shape
    x_scaled, sf = _per_token_cast_to_fp4(kv_cache.view(-1, head_dim), gran_k=32)
    x_back = _cast_back_from_fp4(x_scaled, sf, gran_k=32).view(
        num_blocks, block_size, 1, head_dim
    )
    x_fp4 = torch.empty(
        (num_blocks, block_size * ((head_dim // 2) + 4)),
        device=kv_cache.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : block_size * head_dim // 2] = x_scaled.view(
        num_blocks, block_size * head_dim // 2
    ).view(torch.uint8)
    x_fp4[:, block_size * head_dim // 2 :] = sf.view(num_blocks, block_size).view(
        torch.uint8
    )
    return x_fp4.view(
        num_blocks, block_size, num_heads, (head_dim // 2) + 4
    ), x_back.to(kv_cache.dtype)


@torch.no_grad()
def _fp4_paged_mqa_logits_reference(
    q,
    q_sf,
    kv_fused,
    weights,
    block_tables,
    seq_lens,
    max_seq_len,
    output_dtype=torch.bfloat16,
):
    """Pure-torch reference for the packed FP4 API inputs.

    Dequantizes q (fp4 codes + UE8M0 scales) and kv_fused (fp4 codes + flat UE8M0
    scales) back to float, then reproduces the kernel math.

    ``output_dtype`` defaults to bfloat16 to match both fp4_paged_mqa_logits and
    this template's output schema; the math is done in float regardless and only
    the final cast follows it.  A caller wanting float32 logits passes it
    explicitly on both sides."""
    num_blocks, block_size, _one, row_bytes = kv_fused.shape
    half_D = row_bytes - 4
    head_dim = half_D * 2
    B, next_n, H, _hd = q.shape
    device = q.device

    # Dequantize KV: flat [KV codes (block_size*half_D bytes)][SF int32 (block_size*4 bytes)] per block.
    flat = kv_fused.reshape(num_blocks, -1)
    kv_codes = flat[:, : block_size * half_D].reshape(num_blocks * block_size, half_D)
    kv_sf = (
        flat[:, block_size * half_D :]
        .contiguous()
        .view(torch.int32)
        .reshape(num_blocks * block_size, 1)
    )
    kv_deq = (
        _cast_back_from_fp4(kv_codes, kv_sf, gran_k=32)
        .view(num_blocks, block_size, 1, head_dim)
        .float()
    )

    # Dequantize Q.
    q_codes = q.reshape(B * next_n * H, half_D)
    q_sf = q_sf.reshape(B * next_n * H, 1)
    q_deq = (
        _cast_back_from_fp4(q_codes, q_sf, gran_k=32)
        .view(B, next_n, H, head_dim)
        .float()
    )

    logits = torch.full(
        (B * next_n, max_seq_len), float("-inf"), device=device, dtype=torch.float32
    )
    for b in range(B):
        seq_len = int(seq_lens[b].item())
        q_pos = torch.arange(seq_len - next_n, seq_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].transpose(0, 1).contiguous()
        n_blk = (seq_len + block_size - 1) // block_size
        blocks = block_tables[b, :n_blk]
        kx = kv_deq[blocks].permute(2, 3, 0, 1).reshape(1, head_dim, -1)
        qx = q_deq[b].transpose(0, 1)  # [H, next_n, D]
        s = torch.matmul(qx, kx).to(torch.float32)  # [H, next_n, total_len]
        total_len = n_blk * block_size
        kpos = torch.arange(0, total_len, device=device)
        mask = (kpos[None, :] < seq_len) & (kpos[None, :] <= q_pos[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * w[..., None]
        s = s.sum(dim=0)
        # max_seq_len is a free axis and need not be block-aligned, so
        # total_len (a whole number of blocks) can exceed the output width.
        end = min(total_len, max_seq_len)
        logits[b * next_n : (b + 1) * next_n, :end] = torch.where(
            (kpos[None, :] <= q_pos[:, None])[:, :end], s[:, :end], float("-inf")
        )
    return logits.to(output_dtype)


_fp4_paged_mqa_logits_reference._trace_reference_dependencies = (
    _ceil_to_ue8m0_int,
    _pack_ue8m0_to_int,
    _unpack_ue8m0_from_int,
    _quantize_to_fp4_e2m1,
    _dequantize_from_fp4_e2m1,
    _per_token_cast_to_fp4,
    _cast_back_from_fp4,
)


def _fp4_paged_mqa_logits_init(
    *,
    batch_size: int,
    next_n: int = 1,
    num_heads: int = 64,
    head_dim: int = 128,
    block_size: int = 64,
    max_seq_len: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build valid inputs for ``flashinfer.fp4_paged_mqa_logits`` (all rows at max_seq_len)."""
    torch.manual_seed(seed)
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    block_tables, num_blocks = _make_paged_block_table(seq_lens, block_size, device)

    q_bf = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_bf.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(
        batch_size, next_n, num_heads, head_dim // 2
    )
    q_sf = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    kv_fused, _ = _pack_kv_cache_fp4(kv_cache)
    return {
        "q": q_fp4,
        "q_sf": q_sf,
        "kv_fused": kv_fused,
        "weights": weights,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": max_seq_len,
    }


_fp4_paged_mqa_logits_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _ceil_to_ue8m0_int,
    _pack_ue8m0_to_int,
    _unpack_ue8m0_from_int,
    _quantize_to_fp4_e2m1,
    _dequantize_from_fp4_e2m1,
    _per_token_cast_to_fp4,
    _cast_back_from_fp4,
    _make_paged_block_table,
    _pack_kv_cache_fp4,
)


fp4_paged_mqa_logits_trace = TraceTemplate(
    op_type="paged_mqa_logits",
    name_prefix="fp4_paged_mqa_logits",
    description=(
        "FP4 (MXFP4) paged MQA logits: weighted dot-product attention scores against a "
        "paged KV cache, with block-scaled FP4 Q/K and per-(token, K-group) UE8M0 scale "
        "factors. Datacentre Blackwell (SM100/SM103) and Rubin (SM107)."
    ),
    axes={
        "batch_size": Var(description="Number of decode requests."),
        "next_n": Const(
            abbrev="nn", description="Speculative-decode slots per request."
        ),
        "num_heads": Const(abbrev="H", description="Number of query heads."),
        "packed_head_dim": Const(
            abbrev="Dp", description="Packed head bytes (head_dim // 2)."
        ),
        "num_kv_heads": Const(abbrev="", description="KV heads (MQA => 1)."),
        "block_size": Const(abbrev="bs", description="Tokens per physical KV block."),
        "block_row_bytes": Const(
            abbrev="", description="Fused KV row bytes (head_dim//2 + 4)."
        ),
        "max_seq_len": Var(description="Maximum KV sequence length."),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "next_n", "num_heads", "packed_head_dim"],
            dtype="uint8",
            description="Query, FP4 e2m1 (two per byte).",
        ),
        "q_sf": Tensor(
            ["batch_size", "next_n", "num_heads"],
            dtype="int32",
            description="Query UE8M0 scale factors (4 packed per (token, head) int32).",
        ),
        "kv_fused": Tensor(
            ["num_blocks", "block_size", "num_kv_heads", "block_row_bytes"],
            dtype="uint8",
            description="Fused paged KV: [FP4 codes][UE8M0 SF int32] per block.",
        ),
        "weights": Tensor(
            ["num_rows", "num_heads"],
            dtype="float32",
            description="Per-head mixing weights.",
        ),
        "block_tables": Tensor(
            ["batch_size", "max_blocks_per_seq"],
            dtype="int32",
            description="Paged block indices.",
        ),
        "seq_lens": Tensor(
            ["batch_size"], dtype="int32", description="Per-request KV length."
        ),
        "max_seq_len": Scalar("int32", description="Maximum KV sequence length."),
    },
    outputs={
        "logits": Tensor(
            ["num_rows", "max_seq_len"],
            dtype="bfloat16",
            description="Attention-score logits.",
        ),
    },
    constraints=[
        "num_rows == batch_size * next_n",
        "block_row_bytes == packed_head_dim + 4",
        "num_kv_heads == 1",
    ],
    tags=["status:verified", "arch:sm100", "arch:sm103", "arch:sm107"],
    reference=_fp4_paged_mqa_logits_reference,
    check=_paged_mqa_logits_masked_check,
    init=_fp4_paged_mqa_logits_init,
)


single_head_paged_mqa_logits_trace = TraceTemplate(
    op_type="single_head_paged_mqa_logits",
    description=(
        "Single-head paged MQA logits: FP32 dot products of query rows with their "
        "request's paged BF16 keys (no head weights or ReLU). SM100/SM103."
    ),
    axes={
        "rows": Var(description="Query rows, 1 to 8 per request group."),
        "groups": Var(description="Groups of query rows, one request each."),
        "requests": Var(description="Block-table rows."),
        "pages": Var(description="Key cache pages."),
        "blocks": Var(description="64-key blocks per request."),
        "max_context": Var(description="Score columns: the context capacity."),
        "page_size": Const(description="Keys per page."),
        "dim": Const(description="Key and query dimension."),
    },
    inputs={
        "query": Tensor(["rows", "dim"]),
        "key_cache": Tensor(["pages", "page_size", "dim"]),
        "block_table": Tensor(["requests", "blocks"]),
        "context_lengths": Tensor(["groups"]),
        "requests": Tensor(["groups"]),
        "scores": Tensor(["rows", "max_context"]),
    },
    outputs={
        "scores": Tensor(["rows", "max_context"], dtype="float32", param="scores")
    },
    tags=["arch:sm100", "arch:sm103"],
)
