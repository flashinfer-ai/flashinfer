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

"""VC-Attention-QK16 host-side preprocessing for the prims_ts context kernel.

VC-Attention-QK16 adds two changes on the value side of attention: a token
permutation that groups similar value rows into the same 128-token K/V tile,
and per-tile value smoothing. Each V tile is stored as E4M3 residuals around
its tile mean; the kernel restores the mean inside the online softmax as
``O += rowsum(P_tile) * mean_tile`` (one bf16 K=16 UMMA step per tile), so the
residual quantization error no longer carries the value DC component. This
variant keeps Q and K in bf16. Only P and V are E4M3.

This module provides the reference preprocessing:

* ``vc_token_permutation`` -- online k-means over value rows, one
  permutation per (batch, head), applied to K and V (attention is invariant to
  a common key/value permutation);
* ``vc_quantize`` -- K permutation, V tile-mean subtraction, per-channel
  E4M3 residual quantization, and the packed kernel operands;
* ``VCAttentionPreprocessor`` -- the paper's V-Smooth schedule over the
  denoising steps on top of ``vc_quantize_fused``;
* ``pack_vc_tile_means`` -- the bf16 mean operand layout the kernel's
  unswizzled K-major descriptor expects;
* ``vc_reference`` -- the fp32 attention over the dequantized operands.

The kernel takes the means and the per-channel V scale through
``BatchPrefillTSWrapper.run`` (``vc=VCAttentionParams(...)``), which folds the
scale into ``output_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# The K/V tile whose value mean the kernel restores, its only supported size.
VC_K_BLOCK_SIZE = 128
# K of the bf16 UMMA mean step, one kind::f16 K step.
VC_MEAN_MMA_K = 16
# Mean UMMA steps issued back to back per group of tiles; each tile takes two of a
# step's 16 K slots (bf16 hi/lo row sum).
VC_MEAN_OPERANDS = 2
VC_MEAN_GROUP_TILES = VC_MEAN_OPERANDS * VC_MEAN_MMA_K // 2
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# V-Smooth schedule. The paper groups and demeans on the first quarter of the
# denoising steps and keeps the permutation afterwards. The refresh cadence
# inside that window and the k-means size are not fixed by the paper. 64
# clusters sort the values into 128-token tiles of similar rows while keeping
# the online k-means a small fraction of the attention call, and 3 iterations
# are enough because every refresh warm-starts from the previous centroids.
VC_SMOOTH_STEP_FRACTION = 0.25
VC_PERM_REFRESH_EVERY = 4
VC_KMEANS_CLUSTERS = 64
VC_KMEANS_ITERS = 3


def vc_mean_group_shape(head_dim: int) -> tuple[int, int]:
    """Shape of the packed mean operands of one tile group: the ``VC_MEAN_OPERANDS``
    operands of ``vc_mean_operand_shape`` stacked along the row axis."""
    rows, cols = vc_mean_operand_shape(head_dim)
    return (VC_MEAN_OPERANDS * rows, cols)


def vc_mean_operand_shape(head_dim: int) -> tuple[int, int]:
    """Shape of one packed bf16 ``[head_dim x VC_MEAN_MMA_K]`` mean operand, in
    rows of 256 elements (the kernel's 512-byte TMA box row)."""
    return (head_dim * VC_MEAN_MMA_K // 256, 256)


@dataclass(frozen=True)
class VCAttentionConfig:
    """Compile-time VC-Attention-QK16 recipe of one plan.

    ``k_block_size`` tokens share one restored value mean. The kernel supports
    its 128-token K/V tile only.
    """

    k_block_size: int = VC_K_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.k_block_size != VC_K_BLOCK_SIZE:
            raise ValueError(
                f"k_block_size must be {VC_K_BLOCK_SIZE}, got {self.k_block_size}"
            )


@dataclass(frozen=True)
class VCAttentionParams:
    """Per-run VC-Attention-QK16 operands beyond ``q``, the permuted ``k`` and the
    E4M3 ``v`` residuals.

    ``v_scale`` is the ``[B, Hkv, D]`` fp32 per-channel E4M3 residual scale.
    ``tile_means`` is the packed bf16 mean operand ``[B, Hkv, num_kv_tiles, 8,
    256]`` from ``pack_vc_tile_means`` (means already divided by
    ``v_scale``). The scale must be positive and finite. The kernel does not
    check it. ``demean`` says whether the run restores the means. It is
    ``False`` after the V-Smooth window, when they are zero and the kernel
    skips the mean steps, so it changes per run.
    """

    v_scale: torch.Tensor
    tile_means: torch.Tensor
    demean: bool = True


def vc_scale_shapes(
    config: VCAttentionConfig,
    *,
    batch_size: int,
    seq_len_kv: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of every ``VCAttentionParams`` tensor a plan consumes, by field name."""
    num_kv_tiles = _blocks(seq_len_kv, config.k_block_size)
    return {
        "v_scale": (batch_size, num_kv_heads, head_dim),
        "tile_means": (
            batch_size,
            num_kv_heads,
            _blocks(num_kv_tiles, VC_MEAN_GROUP_TILES),
            *vc_mean_group_shape(head_dim),
        ),
    }


_VC_PARAM_DTYPES = {
    "v_scale": torch.float32,
    "tile_means": torch.bfloat16,
}


def validate_vc_params(
    params: VCAttentionParams,
    expected_shapes: dict[str, tuple[int, ...]],
    *,
    device: torch.device,
) -> None:
    """Validate the operands of one run against the plan's expected shapes.

    ``expected_shapes`` is the plan's ``vc_scale_shapes`` result.
    """
    if not isinstance(params, VCAttentionParams):
        raise TypeError("vc must be a VCAttentionParams instance")
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(params, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != _VC_PARAM_DTYPES[name]:
            raise ValueError(
                f"{name} must have dtype {_VC_PARAM_DTYPES[name]}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} data pointer must be 16-byte aligned")
        if tensor.device != device:
            raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


@dataclass(frozen=True)
class VCAttentionOperands:
    """Everything one VC-Attention-QK16 forward needs beyond ``q``, in kernel layout.

    ``k`` is the bf16 input in permuted token order; ``v`` is E4M3
    ``[B, S, H, D]`` in the same order holding the tile residuals
    ``(V[perm] - mean) / v_scale``.
    """

    k: torch.Tensor
    v: torch.Tensor
    v_scale: (
        torch.Tensor
    )  # [B, H, D] fp32 per-channel E4M3 residual scale (feeds output_scale)
    mean: torch.Tensor  # [B, H, num_kv_tiles, D] fp32 tile means (permuted order)
    mu: (
        torch.Tensor
    )  # [B, H, ceil(num_kv_tiles / 16), 16, 256] bf16 packed kernel operands
    perm: torch.Tensor  # [B, H, S_k] int64 token permutation
    demean: bool = True  # whether the means were subtracted (else zero)

    @property
    def params(self) -> "VCAttentionParams":
        """The per-run operands ``BatchPrefillTSWrapper.run`` takes as ``vc``."""
        return VCAttentionParams(
            v_scale=self.v_scale, tile_means=self.mu, demean=self.demean
        )


def _blocks(length: int, block: int) -> int:
    return (length + block - 1) // block


def _pad_tokens(x: torch.Tensor, block: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1) of ``[B, S, H, D]`` up to a block multiple."""
    pad = _blocks(x.shape[1], block) * block - x.shape[1]
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))


@torch.no_grad()
def vc_token_permutation_with_centroids(
    v: torch.Tensor,
    *,
    num_clusters: int | None = None,
    iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    init_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means per (batch, head) over value rows, returning ``(perm, centroids)``.

    ``init_centroids`` (``[B*H, k, D]`` from a previous call) warm-starts the
    clustering, as the paper does when the grouping is refreshed a few
    denoising steps later.
    """
    b, s, h, d = v.shape
    if num_clusters is None:
        num_clusters = VC_KMEANS_CLUSTERS
    n_groups = b * h
    # Work on a few heads at a time so the fp32 copy of V and the distance
    # matrix each stay under 0.75 GB.
    per_group = 4 * s * max(d, num_clusters)
    chunk_h = max(1, min(h, int(0.75e9 // per_group)))
    perm_out = torch.empty((b, h, s), dtype=torch.int64, device=v.device)
    cent_out = torch.empty(
        (n_groups, num_clusters, d), dtype=torch.float32, device=v.device
    )
    warm = init_centroids is not None and tuple(init_centroids.shape) == (
        n_groups,
        num_clusters,
        d,
    )
    for bi in range(b):
        for h0 in range(0, h, chunk_h):
            h1 = min(h, h0 + chunk_h)
            gsz = h1 - h0
            g0 = bi * h + h0
            x = v[bi, :, h0:h1, :].permute(1, 0, 2).float()  # [G, S, D]
            if warm:
                centroids = init_centroids[g0 : g0 + gsz].to(x.dtype)
            else:
                idx = torch.stack(
                    [
                        torch.randperm(s, device=v.device, generator=generator)[
                            :num_clusters
                        ]
                        for _ in range(gsz)
                    ]
                )  # [G, k]
                centroids = torch.gather(
                    x, 1, idx.unsqueeze(-1).expand(-1, -1, d)
                )  # [G, k, D]
            x_sq = (x * x).sum(-1, keepdim=True)  # [G, S, 1]
            labels = None
            for _ in range(iters):
                c_sq = (centroids * centroids).sum(-1).unsqueeze(1)  # [G, 1, k]
                dist = x_sq + c_sq - 2.0 * torch.bmm(x, centroids.transpose(1, 2))
                labels = dist.argmin(dim=-1)  # [G, S]
                del dist
                counts = torch.zeros(
                    gsz, num_clusters, device=v.device, dtype=torch.float32
                )
                counts.scatter_add_(
                    1, labels, torch.ones_like(labels, dtype=torch.float32)
                )
                sums = torch.zeros_like(centroids)
                sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), x)
                nonempty = counts > 0
                centroids = torch.where(
                    nonempty.unsqueeze(-1),
                    sums / counts.clamp_min(1.0).unsqueeze(-1),
                    centroids,
                )
            assert labels is not None
            perm_out[bi, h0:h1] = torch.argsort(labels, dim=-1, stable=True)
            cent_out[g0 : g0 + gsz] = centroids
            del x, x_sq
    return perm_out, cent_out


@torch.no_grad()
def vc_token_permutation(
    v: torch.Tensor,
    *,
    num_clusters: int | None = None,
    iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Group similar value rows by k-means per (batch, head), then argsort labels.

    Returns an int64 permutation ``[B, H, S]`` such that ``V[b, perm[b, h], h]``
    lists the tokens cluster by cluster, so 128-token K/V tiles hold similar
    values and their tile means remove most of the value energy.
    """
    return vc_token_permutation_with_centroids(
        v, num_clusters=num_clusters, iters=iters, generator=generator
    )[0]


def pack_vc_tile_means(
    mean: torch.Tensor, v_scale: torch.Tensor | float
) -> torch.Tensor:
    """Pack ``[B, H, T, D]`` tile means into the kernel's bf16 UMMA operands,
    ``VC_MEAN_OPERANDS`` per group of ``VC_MEAN_GROUP_TILES`` consecutive tiles.

    ``v_scale`` is the per-channel residual scale ``[B, H, D]`` (or a scalar);
    the means are stored divided by it.

    Operand ``o`` of a group covers its tiles ``8o`` to ``8o+7``. The kernel
    multiplies a ``[128 x 16]`` row-sum operand (``k=2i`` bf16 high part,
    ``k=2i+1`` bf16 remainder of each row's sum over tile ``8o+i``) by the
    ``[D x 16]`` K-major operand, so rows ``k=2i`` and ``k=2i+1`` both carry
    ``mean / v_scale`` of that tile; tiles past the end of the sequence are zero.
    Each operand is laid out as the unswizzled tcgen05 K-major core-matrix order:
    8-row groups of 256 bytes, two 128-byte core matrices (k 0-7, k 8-15) of 8
    rows x 16 bytes.
    """
    b, h, t, d = mean.shape
    if d % 8 != 0:
        raise ValueError("head_dim must be a multiple of 8")
    vs = (
        v_scale.reshape(b, h, 1, d)
        if isinstance(v_scale, torch.Tensor) and v_scale.dim() == 3
        else v_scale
    )
    m = (mean.float() / vs).to(torch.bfloat16)
    g = _blocks(t, VC_MEAN_GROUP_TILES)
    tiles_per_operand = VC_MEAN_MMA_K // 2
    rows = torch.zeros(
        (b, h, g * VC_MEAN_GROUP_TILES, d), dtype=torch.bfloat16, device=mean.device
    )
    rows[:, :, :t] = m
    # Operand o of a group holds its tiles 8o..8o+7; K slots 2i and 2i+1 hold tile i.
    rows = rows.view(b, h, g, VC_MEAN_OPERANDS, tiles_per_operand, d // 8, 8).permute(
        0, 1, 2, 3, 5, 6, 4
    )
    tile = torch.stack((rows, rows), dim=-1).reshape(
        b, h, g, VC_MEAN_OPERANDS, d // 8, 8, VC_MEAN_MMA_K
    )
    # [8-row group, K core matrix, row, k] in core-matrix order.
    tile = tile.view(
        b, h, g, VC_MEAN_OPERANDS, d // 8, 8, 2, VC_MEAN_MMA_K // 2
    ).permute(0, 1, 2, 3, 4, 6, 5, 7)
    return tile.reshape(b, h, g, *vc_mean_group_shape(d)).contiguous()


@torch.no_grad()
def vc_quantize(
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
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)
    gather_idx = (
        perm.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, d)
    )  # [B, S, H, D]
    k_p = torch.gather(k, 1, gather_idx)
    v_p = torch.gather(v.float(), 1, gather_idx)

    num_kv_tiles = _blocks(s_k, VC_K_BLOCK_SIZE)
    v_pad = _pad_tokens(v_p, VC_K_BLOCK_SIZE).view(
        b, num_kv_tiles, VC_K_BLOCK_SIZE, h, d
    )
    valid = torch.zeros(
        num_kv_tiles, VC_K_BLOCK_SIZE, device=v.device, dtype=torch.float32
    )
    valid.view(-1)[:s_k] = 1.0
    counts = valid.sum(dim=1).clamp_min(1.0)  # [T]
    mean = (v_pad * valid.view(1, num_kv_tiles, VC_K_BLOCK_SIZE, 1, 1)).sum(
        dim=2
    ) / counts.view(1, num_kv_tiles, 1, 1)  # [B, T, H, D]
    if not demean:
        mean = torch.zeros_like(mean)
    residual = (v_pad - mean.unsqueeze(2)) * valid.view(
        1, num_kv_tiles, VC_K_BLOCK_SIZE, 1, 1
    )
    residual = residual.view(b, num_kv_tiles * VC_K_BLOCK_SIZE, h, d)[:, :s_k]
    # One E4M3 scale per (batch, head, channel) over all tokens (paper).
    v_scale = (
        (residual.abs().amax(dim=1) / E4M3_MAX).clamp_min(1e-12).float()
    )  # [B, H, D]
    v8 = (residual / v_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    mean = mean.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    return VCAttentionOperands(
        k=k_p.contiguous(),
        v=v8.contiguous(),
        v_scale=v_scale,
        mean=mean,
        mu=pack_vc_tile_means(mean, v_scale),
        perm=perm,
        demean=demean,
    )


# ---------------------------------------------------------------------------
# Fused preprocessing (CuTe DSL kernels in kernels/vc_prepare.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def vc_quantize_fused(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    perm: torch.Tensor | None = None,
    kmeans_iters: int = VC_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    demean: bool = True,
) -> VCAttentionOperands:
    """Same contract as ``vc_quantize``, computed by two CuTe DSL kernels
    (``flashinfer.attention.prims_ts.kernels.vc_prepare``).

    Pass 1 gathers each permuted 128-token K/V tile once, writes the permuted
    K rows, and writes the V tile mean and residual amax. A reduction over the
    tile amaxes gives the per-channel V scale. Pass 2 writes the E4M3 residuals
    and the packed bf16 mean operand. Requires head_dim 128 and bf16/fp16
    inputs; other cases fall back to ``vc_quantize``.
    """
    if k.dim() != 4 or k.shape != v.shape:
        raise ValueError("k and v must be [B, S, H, D] with matching shapes")
    if (
        k.shape[-1] != 128
        or k.dtype not in (torch.bfloat16, torch.float16)
        or not k.is_cuda
    ):
        return vc_quantize(
            k,
            v,
            perm=perm,
            kmeans_iters=kmeans_iters,
            generator=generator,
            demean=demean,
        )
    from .kernels.vc_prepare import vc_prepare

    if perm is None:
        perm = vc_token_permutation(v, iters=kmeans_iters, generator=generator)
    k_p, v8, v_scale, mean, mu = vc_prepare(
        k.contiguous(),
        v.contiguous(),
        perm,
        demean=demean,
    )
    return VCAttentionOperands(
        k=k_p,
        v=v8,
        v_scale=v_scale,
        mean=mean,
        mu=mu,
        perm=perm,
        demean=demean,
    )


class VCAttentionPreprocessor:
    """Caller-owned preparation of bf16 K/V with the paper's V-Smooth schedule.

    Value grouping and demeaning run on the first ``smooth_step_fraction`` of
    the denoising steps. Inside that window the token permutation is
    recomputed every ``perm_refresh_every`` steps (k-means warm-started from
    the previous centroids) and kept afterwards, when the plain per-channel
    E4M3 V runs with zero tile means. ``kmeans_clusters`` (``None`` = 64) and
    ``kmeans_iters`` parametrize the online k-means. Without a denoise step
    the grouping is computed once per K/V geometry and kept.
    """

    def __init__(
        self,
        *,
        smooth_step_fraction: float = VC_SMOOTH_STEP_FRACTION,
        perm_refresh_every: int = VC_PERM_REFRESH_EVERY,
        kmeans_clusters: int | None = None,
        kmeans_iters: int = VC_KMEANS_ITERS,
    ) -> None:
        if not (0.0 <= smooth_step_fraction <= 1.0):
            raise ValueError("smooth_step_fraction must be in [0, 1]")
        if perm_refresh_every < 1:
            raise ValueError("perm_refresh_every must be at least 1")
        if kmeans_clusters is not None and kmeans_clusters < 1:
            raise ValueError("kmeans_clusters must be at least 1")
        if kmeans_iters < 1:
            raise ValueError("kmeans_iters must be at least 1")
        self.smooth_step_fraction = smooth_step_fraction
        self.perm_refresh_every = perm_refresh_every
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_iters = kmeans_iters
        # Per (batch, seq_len_k, num_kv_heads): perm, centroids, step of the last refresh.
        self._groupings: dict[tuple[int, int, int], dict[str, object]] = {}

    @torch.no_grad()
    @torch.compiler.disable
    def prepare(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        denoise_step: tuple[int, int] | None = None,
    ) -> VCAttentionOperands:
        """Return the operands of one run for ``[B, S, H, D]`` bf16 K/V.

        ``denoise_step`` is the sampler's ``(step_index, num_steps)``. Kept
        out of torch.compile so the step index and the k-means refresh do not
        recompile the caller per step.
        """
        key = (k.shape[0], k.shape[1], k.shape[2])
        entry = self._groupings.get(key)
        smooth = True
        refresh = entry is None
        if denoise_step is not None:
            step_idx, num_steps = denoise_step
            smooth = step_idx < math.ceil(self.smooth_step_fraction * num_steps)
            # Step 0 starts a new generation, so it always regroups.
            refresh = smooth and (
                entry is None
                or step_idx == 0
                or (
                    step_idx % self.perm_refresh_every == 0
                    and entry["step"] != step_idx
                )
            )
        if smooth:
            if refresh:
                perm, centroids = vc_token_permutation_with_centroids(
                    v,
                    num_clusters=self.kmeans_clusters,
                    iters=self.kmeans_iters,
                    init_centroids=None if entry is None else entry["centroids"],
                )
                entry = {
                    "perm": perm,
                    "centroids": centroids,
                    "step": None if denoise_step is None else denoise_step[0],
                }
                self._groupings[key] = entry
            perm = entry["perm"]
        elif entry is not None:
            # V-Smooth off (paper Section 3.4). The plain kernel runs on the last
            # permutation with zero tile means.
            perm = entry["perm"]
        else:
            perm = torch.arange(k.shape[1], device=k.device, dtype=torch.int64)
            perm = perm.view(1, 1, -1).expand(k.shape[0], k.shape[2], -1).contiguous()
        return vc_quantize_fused(k, v, perm=perm, demean=smooth)


@torch.no_grad()
def vc_reference(
    q: torch.Tensor, ops: VCAttentionOperands, *, sm_scale: float | None = None
) -> torch.Tensor:  # noqa: D401
    """fp32 attention of ``q`` over the dequantized VC operands (what the kernel
    computes, up to E4M3 P rounding), as ``[B, S_q, H, D]`` fp32."""
    b, s_k, h, d = ops.k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    q = q.float()
    k = ops.k.float()
    # The kernel restores bf16(mean / v_scale) * v_scale, per channel.
    vs = ops.v_scale.reshape(b, ops.k.shape[2], 1, d)  # [B, H, 1, D]
    mean = (ops.mean / vs).to(torch.bfloat16).float() * vs
    mean_tokens = mean.permute(0, 2, 1, 3).repeat_interleave(VC_K_BLOCK_SIZE, dim=1)[
        :, :s_k
    ]
    v = ops.v.float() * ops.v_scale.unsqueeze(1) + mean_tokens
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


__all__ = [
    "E4M3_MAX",
    "VC_K_BLOCK_SIZE",
    "VC_KMEANS_CLUSTERS",
    "VC_KMEANS_ITERS",
    "VC_PERM_REFRESH_EVERY",
    "VC_SMOOTH_STEP_FRACTION",
    "VC_MEAN_MMA_K",
    "VCAttentionConfig",
    "VCAttentionOperands",
    "VCAttentionParams",
    "VCAttentionPreprocessor",
    "pack_vc_tile_means",
    "validate_vc_params",
    "vc_quantize",
    "vc_mean_group_shape",
    "vc_mean_operand_shape",
    "VC_MEAN_GROUP_TILES",
    "VC_MEAN_OPERANDS",
    "vc_quantize_fused",
    "vc_reference",
    "vc_scale_shapes",
    "vc_token_permutation",
    "vc_token_permutation_with_centroids",
]
