# Copyright (c) 2026 by FlashInfer team.
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

"""VC-Attention (bf16 or E4M3 Q/K with E4M3 V tile residuals) for the PrimTS context kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0a0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0a0",
)

from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper,
    VCAttentionConfig,
    batch_prefill,
    vc_attention as vca,
)
from flashinfer.attention.prims_ts.vc_attention.quantization import (
    VC_KMEANS_ITERS,
    VCAttentionOperands,
    VCAttentionPreprocessor,
    hadamard_matrix,
    vc_quantize,
    vc_quantize_fp8,
    vc_quantize_repair,
    vc_token_permutation,
)

from tests.attention.test_attention_ts_context import _REQUIRES_CONTEXT_GPU

# The SM107 plan rejects VC-Attention, so tests that need a VC plan skip there.
_VC_PLAN_GPU = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7),
    reason="VC-Attention is off on SM107",
)

_FP8 = torch.float8_e4m3fn
_HEAD_DIM = 128


@pytest.fixture(autouse=True)
def _exact_fp32_matmul():
    """The torch reference needs true fp32 matmuls; NGC containers export
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which would turn them into TF32."""
    prev_prec = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(prev_prec)
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32


# ---------------------------------------------------------------------------
# Torch references of the two preparations (the library ships only the CuTe
# DSL kernels); the fused-vs-reference tests compare against these.
# ---------------------------------------------------------------------------
def _block_amax_scale(x: torch.Tensor, block: int) -> torch.Tensor:
    """Per-(batch, head, token block) E4M3 scale of ``[B, S, H, D]``: amax / 448."""
    b, s, h, d = x.shape
    xp = _pad_tokens(x.float().abs(), block)
    nb = xp.shape[1] // block
    amax = xp.view(b, nb, block, h, d).amax(dim=(2, 4))  # [B, nb, H]
    return (amax / vca.E4M3_MAX).clamp_min(1e-12).permute(0, 2, 1).contiguous()


def _quantize_blocks(x: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    """Divide ``[B, S, H, D]`` by its ``[B, H, nb]`` block scale and round to E4M3."""
    b, s, h, d = x.shape
    per_token = vca._per_token_scales(scale, s, block)  # [B, S, H]
    return (x.float() / per_token.unsqueeze(-1)).to(torch.float8_e4m3fn)


def _pad_tokens(x: torch.Tensor, block: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1) of ``[B, S, H, D]`` up to a block multiple."""
    pad = vca._blocks(x.shape[1], block) * block - x.shape[1]
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))


@torch.no_grad()
def _reference_quantize(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    demean: bool = True,
) -> VCAttentionOperands:
    """Turn ``[B, S, H, D]`` K/V into VC-Attention-QK16 kernel operands.

    Keys are permuted with the values and kept in their dtype. Values are
    permuted, split into 128-token tile means and residuals (``demean=False``
    keeps the means at zero), and the residuals are quantized to E4M3 with one
    scale per (batch, head, channel), returned as ``v_scale`` for the run's
    ``vc`` operands.
    """
    if k.dim() != 4 or k.shape != v.shape:
        raise ValueError("k and v must be [B, S, H, D] with matching shapes")
    b, s_k, h, d = k.shape
    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)[0]
    gather_idx = (
        perm.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, d)
    )  # [B, S, H, D]
    k_p = torch.gather(k, 1, gather_idx)
    v_p = torch.gather(v.float(), 1, gather_idx)

    num_kv_tiles = vca._blocks(s_k, vca.VC_K_BLOCK_SIZE)
    v_pad = _pad_tokens(v_p, vca.VC_K_BLOCK_SIZE).view(
        b, num_kv_tiles, vca.VC_K_BLOCK_SIZE, h, d
    )
    valid = torch.zeros(
        num_kv_tiles, vca.VC_K_BLOCK_SIZE, device=v.device, dtype=torch.float32
    )
    valid.view(-1)[:s_k] = 1.0
    counts = valid.sum(dim=1).clamp_min(1.0)  # [T]
    mean = (v_pad * valid.view(1, num_kv_tiles, vca.VC_K_BLOCK_SIZE, 1, 1)).sum(
        dim=2
    ) / counts.view(1, num_kv_tiles, 1, 1)  # [B, T, H, D]
    if not demean:
        mean = torch.zeros_like(mean)
    residual = (v_pad - mean.unsqueeze(2)) * valid.view(
        1, num_kv_tiles, vca.VC_K_BLOCK_SIZE, 1, 1
    )
    residual = residual.view(b, num_kv_tiles * vca.VC_K_BLOCK_SIZE, h, d)[:, :s_k]
    # One E4M3 scale per (batch, head, channel) over all tokens (paper).
    v_scale = (
        (residual.abs().amax(dim=1) / vca.E4M3_MAX).clamp_min(1e-12).float()
    )  # [B, H, D]
    v8 = (residual / v_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    mean = mean.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    return VCAttentionOperands(
        k=k_p.contiguous(),
        v=v8.contiguous(),
        v_scale=v_scale,
        mean=mean,
        mu=_pack_tile_means(mean, v_scale),
        perm=perm,
        demean=demean,
    )


@torch.no_grad()
def _reference_quantize_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    demean: bool = True,
) -> VCAttentionOperands:
    """Turn ``[B, S, H, D]`` Q/K/V into VC-Attention-QK8 kernel operands.

    Keys are centred by their per-(batch, head) channel mean over tokens
    (softmax-invariant), permuted with the values, rotated with Q by the
    normalised Hadamard matrix and quantized to E4M3 with one scale per
    128-token tile; queries take one E4M3 scale per 128-token block.
    Values are prepared as in ``_reference_quantize``.
    """
    if q.dim() != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("q, k, v must be [B, S, H, D] with matching batch and heads")
    b, s_k, h, d = k.shape
    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)[0]
    gather_idx = perm.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, d)
    k_smooth = k.float() - k.float().mean(dim=1, keepdim=True)
    k_p = torch.gather(k_smooth, 1, gather_idx)
    v_p = torch.gather(v.float(), 1, gather_idx)
    hm = hadamard_matrix(d, q.device)
    q_f = q.float() @ hm
    k_p = k_p @ hm
    q_scale = _block_amax_scale(q_f, vca.VC_K_BLOCK_SIZE)
    k_scale = _block_amax_scale(k_p, vca.VC_K_BLOCK_SIZE)
    q8 = _quantize_blocks(q_f, q_scale, vca.VC_K_BLOCK_SIZE)
    k8 = _quantize_blocks(k_p, k_scale, vca.VC_K_BLOCK_SIZE)
    num_kv_tiles = vca._blocks(s_k, vca.VC_K_BLOCK_SIZE)
    v_pad = _pad_tokens(v_p, vca.VC_K_BLOCK_SIZE).view(
        b, num_kv_tiles, vca.VC_K_BLOCK_SIZE, h, d
    )
    valid = torch.zeros(
        num_kv_tiles, vca.VC_K_BLOCK_SIZE, device=v.device, dtype=torch.float32
    )
    valid.view(-1)[:s_k] = 1.0
    counts = valid.sum(dim=1).clamp_min(1.0)
    mean = (v_pad * valid.view(1, num_kv_tiles, vca.VC_K_BLOCK_SIZE, 1, 1)).sum(
        dim=2
    ) / counts.view(1, num_kv_tiles, 1, 1)  # [B, T, H, D]
    if not demean:
        mean = torch.zeros_like(mean)
    residual = (v_pad - mean.unsqueeze(2)) * valid.view(
        1, num_kv_tiles, vca.VC_K_BLOCK_SIZE, 1, 1
    )
    residual = residual.view(b, num_kv_tiles * vca.VC_K_BLOCK_SIZE, h, d)[:, :s_k]
    v_scale = (residual.abs().amax(dim=1) / vca.E4M3_MAX).clamp_min(1e-12).float()
    v8 = (residual / v_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    mean = mean.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    return VCAttentionOperands(
        k=k8.contiguous(),
        v=v8.contiguous(),
        v_scale=v_scale,
        mean=mean,
        mu=_pack_tile_means(mean, v_scale),
        perm=perm,
        demean=demean,
        q=q8.contiguous(),
        q_scale=q_scale.permute(1, 0, 2).contiguous(),
        k_scale=k_scale.permute(1, 0, 2).contiguous(),
    )


@dataclass(frozen=True)
class _VCCase:
    """One VC problem: geometry and launch path."""

    name: str
    batch_size: int
    seq_len: int
    num_heads: int
    one_shot: bool = False
    # Q tokens per sequence when they differ from the K/V tokens. None means equal.
    seq_len_q: int | None = None
    # bf16 selects VC-Attention-QK16, E4M3 VC-Attention-QK8.
    qk_dtype: torch.dtype = torch.bfloat16
    # V repair budget as a fraction of tokens. 0 means tile means.
    repair_budget: float = 0.0


_CASES = (
    _VCCase("single-tile", 1, 512, 1),
    _VCCase("partial-tile", 1, 2000, 2),
    # Odd Q-tile count (cluster padding), partial last K/V tile, two batches.
    _VCCase("two-batches", 2, 2120, 2),
    _VCCase("one-shot", 1, 2000, 2, one_shot=True),
    _VCCase("qk8", 2, 2048, 2, qk_dtype=_FP8),
    _VCCase("qk8-partial-tile", 1, 2000, 2, qk_dtype=_FP8),
    # V repair, one repair tile before the masked partial tail, two tiles with no tail.
    _VCCase("repair-partial-tile", 1, 2000, 2, repair_budget=0.02),
    _VCCase("repair-two-batches", 2, 4096, 2, repair_budget=0.05),
    _VCCase("qk8-repair-partial-tile", 1, 2000, 2, qk_dtype=_FP8, repair_budget=0.02),
    _VCCase("qk8-repair-two-batches", 2, 4096, 2, qk_dtype=_FP8, repair_budget=0.05),
    # Q and K/V lengths differ, so the Q and K scale tables have different block counts.
    _VCCase("cross-length", 2, 2000, 2, seq_len_q=1000),
    _VCCase("qk8-cross-length", 2, 2000, 2, seq_len_q=1000, qk_dtype=_FP8),
)


def _structured_values(shape, device):
    """Values with a strong per-token component so tile means matter."""
    b, s, h, d = shape
    return torch.randn(shape, device=device) * 0.3 + torch.randn(
        b, s, h, 1, device=device
    ) * torch.randn(b, 1, h, d, device=device)


def _random_inputs(batch_size, seq_len, num_heads, *, seed=0, seq_len_q=None):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(
        batch_size, seq_len_q or seq_len, num_heads, _HEAD_DIM, device=device
    ).to(torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, _HEAD_DIM, device=device).to(
        torch.bfloat16
    )
    v = _structured_values(k.shape, device).to(torch.bfloat16)
    return q, k, v


_MEAN_CONFIG = VCAttentionConfig()


def _pack_tile_means(mean: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
    """Torch reference of the packed bf16 mean operands the preparation writes.

    Each group of 16 tiles gives two operands in unswizzled tcgen05 K-major
    core-matrix order. K slots 2i and 2i+1 of operand o hold mean / v_scale of
    tile 8o+i and the tiles past the sequence are zero.
    """
    b, h, t, d = mean.shape
    vs = v_scale.reshape(b, h, 1, d)
    m = (mean.float() / vs).to(torch.bfloat16)
    g = -(-t // vca.VC_MEAN_GROUP_TILES)
    tiles_per_operand = vca.VC_MEAN_MMA_K // 2
    rows = torch.zeros(
        (b, h, g * vca.VC_MEAN_GROUP_TILES, d), dtype=torch.bfloat16, device=mean.device
    )
    rows[:, :, :t] = m
    # Operand o of a group holds its tiles 8o to 8o+7 in K slots 2i and 2i+1.
    rows = rows.view(
        b, h, g, vca.VC_MEAN_OPERANDS, tiles_per_operand, d // 8, 8
    ).permute(0, 1, 2, 3, 5, 6, 4)
    tile = torch.stack((rows, rows), dim=-1).reshape(
        b, h, g, vca.VC_MEAN_OPERANDS, d // 8, 8, vca.VC_MEAN_MMA_K
    )
    # Core-matrix order of 8-row group, K core matrix, row and k.
    tile = tile.view(
        b, h, g, vca.VC_MEAN_OPERANDS, d // 8, 8, 2, vca.VC_MEAN_MMA_K // 2
    ).permute(0, 1, 2, 3, 4, 6, 5, 7)
    return tile.reshape(b, h, g, *vca.vc_mean_group_shape(d)).contiguous()


def _plan(
    wrapper,
    *,
    batch_size,
    seq_len,
    num_heads,
    sm_scale=None,
    qk_dtype=torch.bfloat16,
    vc_config=_MEAN_CONFIG,
    seq_len_q=None,
    **overrides,
):
    arguments = dict(
        device=torch.device("cuda"),
        batch_size=batch_size,
        max_seq_len_q=seq_len_q or seq_len,
        max_kv_len=seq_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=_HEAD_DIM,
        q_dtype=qk_dtype,
        k_dtype=qk_dtype,
        v_dtype=_FP8,
        out_dtype=torch.bfloat16,
        sm_scale=sm_scale,
        vc_config=vc_config,
    )
    wrapper.plan(**{**arguments, **overrides})


def _quantize(case: _VCCase, q, k, v):
    if case.repair_budget:
        return vc_quantize_repair(
            k, v, budget=case.repair_budget, q=q if case.qk_dtype == _FP8 else None
        )
    if case.qk_dtype == _FP8:
        return vc_quantize_fp8(q, k, v)
    return vc_quantize(k, v)


def _run_case(case: _VCCase):
    q, k, v = _random_inputs(
        case.batch_size, case.seq_len, case.num_heads, seq_len_q=case.seq_len_q
    )
    ops = _quantize(case, q, k, v)
    q_run = q if ops.q is None else ops.q
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    if case.one_shot:
        out = batch_prefill(
            q_run,
            ops.k,
            ops.v,
            sm_scale=sm_scale,
            out_dtype=torch.bfloat16,
            vc=ops.params,
            vc_config=VCAttentionConfig(
                demean=ops.demean,
                repair_tiles=ops.repair_tiles,
            ),
        )
    else:
        wrapper = BatchPrefillTSWrapper()
        _plan(
            wrapper,
            batch_size=case.batch_size,
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            sm_scale=sm_scale,
            qk_dtype=case.qk_dtype,
            seq_len_q=case.seq_len_q,
            vc_config=VCAttentionConfig(
                demean=ops.demean,
                repair_tiles=ops.repair_tiles,
            ),
        )
        out = wrapper.run(q_run, ops.k, ops.v, vc=ops.params)
    reference = vca.vc_reference(q, ops, sm_scale=sm_scale)
    if case.repair_budget and case.qk_dtype != _FP8:
        # Under QK8 the repair tiles carry their own K scale, so their key
        # copies are not byte identical and this match does not apply.
        torch.testing.assert_close(
            reference, _dense_repair_reference(q, ops, sm_scale), rtol=1e-4, atol=1e-4
        )
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)
    return out.float(), reference, exact


def _dense_repair_reference(q, ops, sm_scale):
    """fp32 attention over the original tokens with the repair residuals added
    back to the values of the selected tokens, which the repair rows identify
    by their key copies. Independent of the row layout the kernel sees."""
    s_k, repair = ops.seq_len_kv, ops.repair_tiles * vca.VC_K_BLOCK_SIZE
    whole = s_k // vca.VC_K_BLOCK_SIZE * vca.VC_K_BLOCK_SIZE
    rows = torch.cat(
        [torch.arange(whole), torch.arange(whole + repair, s_k + repair)]
    ).to(q.device)
    k = ops.k[:, rows].float()
    v = ops.v[:, rows].float() * ops.v_scale.unsqueeze(1)
    rep_k = ops.k[:, whole : whole + repair].float()
    rep_v = ops.v[:, whole : whole + repair].float() * ops.v_scale.unsqueeze(1)
    for b in range(k.shape[0]):
        for h in range(k.shape[2]):
            match = (rep_k[b, :, h].unsqueeze(1) == k[b, :, h].unsqueeze(0)).all(-1)
            v[b, :, h] += match.float().T @ rep_v[b, :, h]
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) * sm_scale
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)


def _relative_error(out: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(out - reference) / torch.linalg.vector_norm(reference)
    )


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_vc_attention_matches_dequantized_reference(case: _VCCase):
    out, reference, exact = _run_case(case)
    assert torch.isfinite(out).all()
    # ExpCast-FP8 codes P with Mitchell's log2 approximation (<= 7.5% per
    # element, bounded total variation), plus bf16 output rounding.
    assert _relative_error(out, reference) < 6e-2
    torch.testing.assert_close(out, reference, rtol=1e-1, atol=1e-1)
    # The recipe itself stays close to unquantized attention.
    assert _relative_error(out, exact) < 1e-1


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
def test_vc_mean_step():
    """A constant tile mean shifts every output by exactly that constant. A plan
    with demean off equals the means plan fed zero tile means and differs from a
    run whose means are non-zero."""
    q, k, v = _random_inputs(1, 1024, 2)
    ops = vc_quantize(k, v)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2)
    outs = {}
    for name, value in (("zero", 0.0), ("one", 1.0)):
        mean = torch.full_like(ops.mean, value)
        mu = _pack_tile_means(mean, ops.v_scale)
        outs[name] = wrapper.run(q, ops.k, ops.v, vc=replace(ops.params, tile_means=mu))
    torch.testing.assert_close(
        outs["one"].float() - outs["zero"].float(),
        torch.ones_like(outs["zero"], dtype=torch.float32),
        rtol=0,
        atol=1.6e-2,
    )
    plain = vc_quantize(k, v, perm=ops.perm, demean=False)
    assert plain.params.tile_means is None
    wrapper_off = BatchPrefillTSWrapper()
    _plan(
        wrapper_off,
        batch_size=1,
        seq_len=1024,
        num_heads=2,
        vc_config=VCAttentionConfig(
            demean=plain.demean,
            repair_tiles=plain.repair_tiles,
        ),
    )
    out_off = wrapper_off.run(q, plain.k, plain.v, vc=plain.params)
    zero_mu = _pack_tile_means(torch.zeros_like(ops.mean), plain.v_scale)
    out_on = wrapper.run(
        q, plain.k, plain.v, vc=replace(plain.params, tile_means=zero_mu)
    )
    torch.testing.assert_close(out_on, out_off, rtol=0, atol=0)
    assert not torch.equal(wrapper.run(q, ops.k, ops.v, vc=ops.params), outs["zero"])


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        pytest.param({"v_dtype": torch.bfloat16}, ValueError, "E4M3 V", id="bf16-v"),
        pytest.param(
            {"q_dtype": _FP8, "k_dtype": _FP8, "v_dtype": torch.bfloat16},
            NotImplementedError,
            "same dtype",
            id="qk8-bf16-v",
        ),
        pytest.param(
            {"mask_type": "causal"}, ValueError, "dense contiguous", id="causal"
        ),
        pytest.param({"packed": True}, ValueError, "dense contiguous", id="packed"),
        pytest.param(
            {"mask_type": "causal", "window_left": 256, "num_qo_heads": 2},
            ValueError,
            "dense contiguous",
            id="head-paired",
        ),
        pytest.param({"num_qo_heads": 2}, ValueError, "equal Q and K/V head", id="gqa"),
        pytest.param(
            {"vc_config": {"repair_tiles": 0}},
            TypeError,
            "VCAttentionConfig instance",
            id="dict-config",
        ),
    ),
)
def test_vc_plan_rejects_unsupported_recipes(overrides, error, match):
    """A VC plan takes the dense contiguous D128 MHA kernel with bf16 or E4M3 Q/K
    and E4M3 V."""
    with pytest.raises(error, match=match):
        _plan(
            BatchPrefillTSWrapper(),
            batch_size=1,
            seq_len=256,
            num_heads=1,
            **{k: v() if callable(v) else v for k, v in overrides.items()},
        )


def test_vc_kernel_config_rejects_invalid_profiles():
    """The kernel config owns the VC recipe rules."""
    from cutlass import BFloat16, Float32, Float8E4M3FN

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    arguments = dict(
        in_qk_dtype=BFloat16,
        in_pv_dtype=Float8E4M3FN,
        qk_acc_dtype=Float32,
        pv_acc_dtype=Float32,
        d=_HEAD_DIM,
        d_v=_HEAD_DIM,
        is_persistent=False,
        is_causal=False,
        is_clc_dynamic=False,
        two_cta_umma=True,
    )
    FmhaTs(**arguments, vc_attention="means")
    FmhaTs(**arguments, vc_attention="outliers")
    FmhaTs(**{**arguments, "in_qk_dtype": Float8E4M3FN}, vc_attention="means")
    with pytest.raises(ValueError, match="unknown vc_attention mode"):
        FmhaTs(**arguments, vc_attention="repair")


@_REQUIRES_CONTEXT_GPU
@_VC_PLAN_GPU
def test_vc_run_rejects_mismatched_operands():
    q, k, v = _random_inputs(1, 1024, 2)
    ops = vc_quantize(k, v)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2)
    good = ops.params
    with pytest.raises(ValueError, match="v_scale must have shape"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, v_scale=good.v_scale[:, :, :-1]))
    with pytest.raises(ValueError, match="v_scale must have dtype"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, v_scale=good.v_scale.half()))
    with pytest.raises(ValueError, match="tile_means must be contiguous"):
        wrapper.run(
            q,
            ops.k,
            ops.v,
            vc=replace(
                good, tile_means=good.tile_means.repeat(1, 1, 1, 1, 2)[..., ::2]
            ),
        )
    with pytest.raises(TypeError, match="VCAttentionParams instance"):
        wrapper.run(q, ops.k, ops.v, vc={"v_scale": good.v_scale})
    with pytest.raises(ValueError, match="required by a VC plan"):
        wrapper.run(q, ops.k, ops.v)
    plain = BatchPrefillTSWrapper()
    _plan(plain, batch_size=1, seq_len=1024, num_heads=2, vc_config=None)
    with pytest.raises(ValueError, match="rejected by a plan without"):
        plain.run(q, ops.k, ops.v, vc=good)
    ops8 = vc_quantize_fp8(q, k, v)
    with pytest.raises(ValueError, match="belong to VC-Attention-QK8"):
        wrapper.run(q, ops.k, ops.v, vc=replace(good, k_scale=ops8.k_scale))
    qk8 = BatchPrefillTSWrapper()
    _plan(qk8, batch_size=1, seq_len=1024, num_heads=2, qk_dtype=_FP8)
    good8 = ops8.params
    with pytest.raises(ValueError, match="k_scale must have shape"):
        qk8.run(
            ops8.q, ops8.k, ops8.v, vc=replace(good8, k_scale=good8.k_scale[:, :-1])
        )
    with pytest.raises(ValueError, match="q_scale must have dtype"):
        qk8.run(ops8.q, ops8.k, ops8.v, vc=replace(good8, q_scale=good8.q_scale.half()))
    with pytest.raises(ValueError, match="require vc.q_scale"):
        qk8.run(ops8.q, ops8.k, ops8.v, vc=good, validate=False)


@_REQUIRES_CONTEXT_GPU
def test_one_shot_vc_config_requires_the_operands():
    """A one-shot recipe without operands has nothing to run."""
    q = torch.empty((1, 256, 1, _HEAD_DIM), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="vc_config requires"):
        batch_prefill(q, q, q.to(_FP8), vc_config=VCAttentionConfig())
    q, k, v = _random_inputs(1, 1024, 2)
    repair = vc_quantize_repair(k, v, budget=0.02)
    with pytest.raises(NotImplementedError, match="does not run V repair"):
        batch_prefill(q, repair.k, repair.v, vc=repair.params)
    with pytest.raises(NotImplementedError, match="does not run V repair"):
        batch_prefill(
            q,
            repair.k,
            repair.v,
            vc=repair.params,
            vc_config=VCAttentionConfig(
                demean=repair.demean,
                repair_tiles=repair.repair_tiles,
            ),
        )


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize("batch,seq_len,heads", [(1, 2000, 2), (2, 4096, 4)])
def test_vc_quantize_fused_matches_reference(batch, seq_len, heads):
    _, k, v = _random_inputs(batch, seq_len, heads)
    perm = vc_token_permutation(v)[0]
    ref = _reference_quantize(k, v, perm=perm)
    fused = vc_quantize(k, v, perm=perm)
    assert torch.equal(fused.k, ref.k)
    torch.testing.assert_close(fused.v_scale, ref.v_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.mean, ref.mean, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused.mu, ref.mu)
    # E4M3 rounding ties may differ by one code between the two paths.
    mismatch = (fused.v.float() != ref.v.float()).float().mean().item()
    assert mismatch < 1e-3


@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize("batch,seq_len,heads", [(1, 2000, 2), (2, 4096, 4)])
def test_vc_quantize_fp8_fused_matches_reference(batch, seq_len, heads):
    q, k, v = _random_inputs(batch, seq_len, heads)
    perm = vc_token_permutation(v)[0]
    ref = _reference_quantize_fp8(q, k, v, perm=perm)
    fused = vc_quantize_fp8(q, k, v, perm=perm)
    torch.testing.assert_close(fused.q_scale, ref.q_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.k_scale, ref.k_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.v_scale, ref.v_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(fused.mean, ref.mean, rtol=1e-5, atol=1e-6)
    assert torch.equal(fused.mu, ref.mu)
    for name in ("q", "k", "v"):
        # E4M3 rounding ties may differ by one code between the two paths.
        mismatch = (
            (getattr(fused, name).float() != getattr(ref, name).float()).float().mean()
        )
        assert mismatch.item() < 1e-3, name


@_REQUIRES_CONTEXT_GPU
@pytest.mark.arch_blackwell
def test_vc_preprocessor_step_schedule():
    """First 25% of steps: grouping + demeaning (refreshed every 4 steps); afterwards demeaning
    is off and the permutation the window left behind is kept (paper Section 3.4)."""
    q, k, v = _random_inputs(1, 1024, 2)
    sm_scale = 1.0 / math.sqrt(_HEAD_DIM)
    wrapper = BatchPrefillTSWrapper()
    _plan(wrapper, batch_size=1, seq_len=1024, num_heads=2, sm_scale=sm_scale)
    prep = VCAttentionPreprocessor()
    exact = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)
    step0 = prep.prepare(k, v, denoise_step=(0, 40))
    assert step0.demean
    out = wrapper.run(q, step0.k, step0.v, vc=step0.params)
    assert (
        _relative_error(out.float(), vca.vc_reference(q, step0, sm_scale=sm_scale))
        < 6e-2
    )
    step1 = prep.prepare(k, v, denoise_step=(1, 40))
    assert torch.equal(step1.perm, step0.perm)  # no refresh at step 1
    step4 = prep.prepare(k, v.flip(1), denoise_step=(4, 40))
    assert not torch.equal(step4.perm, step0.perm)  # refreshed at step 4
    late = prep.prepare(k, v, denoise_step=(20, 40))  # V-Smooth off
    assert not late.demean
    assert torch.equal(late.perm, step4.perm)
    wrapper_late = BatchPrefillTSWrapper()
    _plan(
        wrapper_late,
        batch_size=1,
        seq_len=1024,
        num_heads=2,
        sm_scale=sm_scale,
        vc_config=VCAttentionConfig(
            demean=late.demean,
            repair_tiles=late.repair_tiles,
        ),
    )
    out_late = wrapper_late.run(q, late.k, late.v, vc=late.params)
    assert _relative_error(out_late.float(), exact) < 1e-1
    # The same schedule drives the QK8 preparation, which also returns E4M3 Q.
    qk8 = VCAttentionPreprocessor().prepare(k, v, q=q, denoise_step=(0, 40))
    assert qk8.q.dtype == _FP8 and qk8.q_scale is not None and qk8.demean
    assert torch.equal(qk8.v_scale, vc_quantize_fp8(q, k, v, perm=qk8.perm).v_scale)
