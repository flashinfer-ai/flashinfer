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

where K is paged via ``block_table``.  ReLU is applied per head, *before*
weighting and reduction -- not to the sum, so the output is not clamped to be
non-negative.  FP8 multiplies the result by the per-token KV scale
(``· scale[pos]``); FP4 folds its per-(token, K-group) UE8M0 scales into
dequantizing Q and K, so it has no trailing scale factor.

Both kernels are Blackwell (SM100/SM103) only.
"""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ──────────────────────────────────────────────────────────────────────────────
# FP8 helpers (self-contained so the embedded init/reference source is runnable)
# ──────────────────────────────────────────────────────────────────────────────


def _ceil_to_ue8m0_fp(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _make_paged_block_table(context_lens, block_size, device):
    """Random paged block table + total block count for a batch of context lens.

    Sized for the kernel's access pattern: it reads ceil(ctx/128) compute tiles *
    (128 // block_size) physical blocks per row, which exceeds ceil(ctx/block_size)
    when ctx is not a multiple of 128. The extra columns default to physical index
    0 (a valid pool block); those positions are beyond ctx (masked), so this avoids
    an out-of-bounds block_table / KV read."""
    n_blk = (context_lens + block_size - 1) // block_size
    kern_blk = ((context_lens + 127) // 128) * (128 // block_size)
    total = int(n_blk.sum().item()) + context_lens.shape[0] * 2
    max_blk = int(kern_blk.max().item())
    block_table = torch.zeros(
        (context_lens.shape[0], max_blk), dtype=torch.int32, device=device
    )
    pool = torch.randperm(total, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk.tolist()):
        block_table[i, :nb] = pool[off : off + nb]
        off += nb
    return block_table, total


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


@torch.no_grad()
def _fp8_paged_mqa_logits_reference(
    q,
    kv_fused,
    weights,
    context_lens,
    block_table,
    max_context_len,
    output_dtype=torch.float32,
):
    """Pure-torch reference operating on the packed FP8 API inputs.

    Unpacks ``kv_fused`` (flat [all KV][all scales] per block) back into fp8 KV
    and fp32 per-token scales, then reproduces the kernel math."""
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
        (B * next_n, max_context_len), float("-inf"), device=device, dtype=output_dtype
    )
    for b in range(B):
        ctx = int(context_lens[b].item())
        q_pos = torch.arange(ctx - next_n, ctx, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].to(output_dtype)
        for blk in range((ctx + block_size - 1) // block_size):
            phys = int(block_table[b, blk].item())
            k = kv_fp8[phys].float()
            sc = scales[phys].to(output_dtype)
            kpos = torch.arange(blk * block_size, (blk + 1) * block_size, device=device)
            mask = (kpos[None, :] < ctx) & (kpos[None, :] <= q_pos[:, None])
            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k.T)
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk).to(output_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0) * sc[None, :]
            # max_context_len is a free axis and need not be page-aligned, so
            # the last physical page can extend past the output width. Clip
            # both the destination and the right-hand side to it.
            s = blk * block_size
            e = min(s + block_size, max_context_len)
            if s >= max_context_len:
                break
            width = e - s
            logits[b * next_n : (b + 1) * next_n, s:e] = torch.where(
                mask[:, :width],
                weighted[:, :width],
                torch.tensor(float("-inf"), device=device, dtype=output_dtype),
            )
    return logits


_fp8_paged_mqa_logits_reference._trace_reference_dependencies = ()


def _paged_mqa_logits_masked_check(
    reference_outputs,
    actual_outputs,
    *,
    context_lens=None,
    next_n=None,
    rtol=2e-2,
    atol=2e-2,
    **_unused,
):
    """Compare kernel logits vs reference over the causal, in-context region.

    Positions beyond each row's causal limit (and the SPLIT_KV-padded padding the
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
    if context_lens is None or next_n is None:
        neginf_mask = torch.zeros((rows, max_len), dtype=torch.bool, device=device)
    else:
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(rows, -1)
        offsets = torch.arange(rows, device=device)
        limits = (
            context_lens[offsets // next_n] - next_n + offsets % next_n
        ).unsqueeze(1)
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
    max_context_len: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build valid inputs for ``flashinfer.fp8_paged_mqa_logits`` (fixed context)."""
    torch.manual_seed(seed)
    context_lens = torch.full(
        (batch_size,), max_context_len, dtype=torch.int32, device=device
    )
    block_table, num_blocks = _make_paged_block_table(context_lens, block_size, device)

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
        "context_lens": context_lens,
        "block_table": block_table,
        "max_context_len": max_context_len,
    }


_fp8_paged_mqa_logits_init._trace_init_dependencies = (  # type: ignore[attr-defined]
    _ceil_to_ue8m0_fp,
    _make_paged_block_table,
    _pack_fused_kv_fp8,
)


fp8_paged_mqa_logits_trace = TraceTemplate(
    op_type="paged_mqa_logits",
    name_prefix="fp8_paged_mqa_logits",
    description=(
        "FP8 paged MQA logits: weighted dot-product attention scores against a paged "
        "KV cache, with per-token FP32 KV scales. Blackwell (SM100/SM103) only."
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
        "block_size": Const(abbrev="bs", description="Tokens per physical KV page."),
        "block_row_bytes": Const(
            abbrev="", description="Fused KV row bytes (head_dim + 4)."
        ),
        "max_context_len": Var(description="Maximum KV sequence length."),
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
        "context_lens": Tensor(
            ["batch_size"], dtype="int32", description="Per-request KV length."
        ),
        "block_table": Tensor(
            ["batch_size", "max_blocks_per_seq"],
            dtype="int32",
            description="Paged block indices.",
        ),
        "max_context_len": Scalar("int32", description="Maximum KV sequence length."),
    },
    outputs={
        "logits": Tensor(
            ["num_rows", "max_context_len"],
            dtype="float32",
            description="Attention-score logits.",
        ),
    },
    constraints=[
        "num_rows == batch_size * next_n",
        "block_row_bytes == head_dim + 4",
        "num_kv_heads == 1",
    ],
    tags=["status:verified", "arch:sm100"],
    reference=_fp8_paged_mqa_logits_reference,
    check=_paged_mqa_logits_masked_check,
    init=_fp8_paged_mqa_logits_init,
)


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
    sf_q,
    kv_fused,
    weights,
    context_lens,
    block_table,
    max_context_len,
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
    q_sf = sf_q.reshape(B * next_n * H, 1)
    q_deq = (
        _cast_back_from_fp4(q_codes, q_sf, gran_k=32)
        .view(B, next_n, H, head_dim)
        .float()
    )

    logits = torch.full(
        (B * next_n, max_context_len), float("-inf"), device=device, dtype=torch.float32
    )
    for b in range(B):
        ctx = int(context_lens[b].item())
        q_pos = torch.arange(ctx - next_n, ctx, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].transpose(0, 1).contiguous()
        n_blk = (ctx + block_size - 1) // block_size
        blocks = block_table[b, :n_blk]
        kx = kv_deq[blocks].permute(2, 3, 0, 1).reshape(1, head_dim, -1)
        qx = q_deq[b].transpose(0, 1)  # [H, next_n, D]
        s = torch.matmul(qx, kx).to(torch.float32)  # [H, next_n, total_len]
        total_len = n_blk * block_size
        kpos = torch.arange(0, total_len, device=device)
        mask = (kpos[None, :] < ctx) & (kpos[None, :] <= q_pos[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * w[..., None]
        s = s.sum(dim=0)
        # max_context_len is a free axis and need not be page-aligned, so
        # total_len (a whole number of pages) can exceed the output width.
        end = min(total_len, max_context_len)
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
    max_context_len: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build valid inputs for ``flashinfer.fp4_paged_mqa_logits`` (fixed context)."""
    torch.manual_seed(seed)
    context_lens = torch.full(
        (batch_size,), max_context_len, dtype=torch.int32, device=device
    )
    block_table, num_blocks = _make_paged_block_table(context_lens, block_size, device)

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
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    kv_fused, _ = _pack_kv_cache_fp4(kv_cache)
    return {
        "q": q_fp4,
        "sf_q": sf_q,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "max_context_len": max_context_len,
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
        "factors. Blackwell (SM100/SM103) only."
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
        "block_size": Const(abbrev="bs", description="Tokens per physical KV page."),
        "block_row_bytes": Const(
            abbrev="", description="Fused KV row bytes (head_dim//2 + 4)."
        ),
        "max_context_len": Var(description="Maximum KV sequence length."),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "next_n", "num_heads", "packed_head_dim"],
            dtype="uint8",
            description="Query, FP4 e2m1 (two per byte).",
        ),
        "sf_q": Tensor(
            ["batch_size", "next_n", "num_heads"],
            dtype="int32",
            description="Query UE8M0 scale factors (4 packed per token).",
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
        "context_lens": Tensor(
            ["batch_size"], dtype="int32", description="Per-request KV length."
        ),
        "block_table": Tensor(
            ["batch_size", "max_blocks_per_seq"],
            dtype="int32",
            description="Paged block indices.",
        ),
        "max_context_len": Scalar("int32", description="Maximum KV sequence length."),
    },
    outputs={
        "logits": Tensor(
            ["num_rows", "max_context_len"],
            dtype="bfloat16",
            description="Attention-score logits.",
        ),
    },
    constraints=[
        "num_rows == batch_size * next_n",
        "block_row_bytes == packed_head_dim + 4",
        "num_kv_heads == 1",
    ],
    tags=["status:verified", "arch:sm100"],
    reference=_fp4_paged_mqa_logits_reference,
    check=_paged_mqa_logits_masked_check,
    init=_fp4_paged_mqa_logits_init,
)
