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

"""TraceTemplates for RoPE (Rotary Position Embedding) operations."""

import math
from typing import Dict, Optional, Tuple, Union

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_AxisT = Union[Var, Const]
_InputT = Union[Tensor, Scalar]


# ── Reference helpers ────────────────────────────────────────────────────────


@torch.no_grad()
def _rope_freqs(
    rotary_dim: int,
    rope_theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Base RoPE inverse-frequency vector (length rotary_dim // 2)."""
    i = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / torch.pow(
        torch.tensor(rope_theta, dtype=torch.float32, device=device), i / rotary_dim
    )


@torch.no_grad()
def _llama31_freqs(
    rotary_dim: int,
    rope_theta: float,
    rope_scale: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
    device: torch.device,
) -> torch.Tensor:
    """Llama 3.1 piecewise NTK-aware frequency scaling."""
    freqs = _rope_freqs(rotary_dim, rope_theta, device)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / freqs
    # Default: scale by 1/rope_scale (low-frequency regime).
    new_freqs = freqs / rope_scale
    # Smooth interpolation for mid-range.
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    mid = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    new_freqs = torch.where(
        mid,
        (1.0 - smooth) * freqs / rope_scale + smooth * freqs,
        new_freqs,
    )
    # High frequency (short wavelength): keep original.
    new_freqs = torch.where(wavelen < high_freq_wavelen, freqs, new_freqs)
    return new_freqs


@torch.no_grad()
def _rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    """Apply RoPE rotation to the last ``rotary_dim`` channels of x.

    cos/sin have shape ``[..., rotary_dim//2]`` broadcastable to x's leading
    dims. If ``interleave`` the rotation is on even/odd pairs, otherwise on
    the half-split halves (first-half / second-half).
    """
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    if interleave:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x2 * cos + x1 * sin
        interleaved = torch.stack([rotated_1, rotated_2], dim=-1)
        rotated = interleaved.reshape(*x_rot.shape)
    else:
        half = rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x2 * cos + x1 * sin
        rotated = torch.cat([rotated_1, rotated_2], dim=-1)
    if x_pass.numel() == 0:
        return rotated.to(x.dtype)
    return torch.cat([rotated.to(x.dtype), x_pass], dim=-1)


@torch.no_grad()
def _positions_from_indptr(
    indptr: torch.Tensor, offsets: torch.Tensor, nnz: int
) -> torch.Tensor:
    """Expand (indptr, offsets) into a per-token position tensor of length nnz."""
    positions = torch.zeros(nnz, dtype=torch.float32, device=indptr.device)
    batch_size = offsets.shape[0]
    for b in range(batch_size):
        start = int(indptr[b].item())
        end = int(indptr[b + 1].item())
        off = int(offsets[b].item())
        n = end - start
        if n > 0:
            positions[start:end] = off + torch.arange(
                n, dtype=torch.float32, device=indptr.device
            )
    return positions


@torch.no_grad()
def _apply_rope_core(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    interleave: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared core: given per-token positions and freqs, rotate q and k."""
    # cos/sin: [nnz, rotary_dim//2]
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(1)  # [nnz, 1, rotary_dim//2]
    sin = torch.sin(angles).unsqueeze(1)
    q_rope = _rotate(q.to(torch.float32), cos, sin, interleave)
    k_rope = _rotate(k.to(torch.float32), cos, sin, interleave)
    return q_rope, k_rope


# ── Per-template references ──────────────────────────────────────────────────


@torch.no_grad()
def _apply_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs = _rope_freqs(rotary_dim, rope_theta, q.device) / rope_scale
    positions = _positions_from_indptr(indptr, offsets, q.shape[0])
    return _apply_rope_core(q, k, positions, freqs, interleave)


@torch.no_grad()
def _apply_rope_pos_ids_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs = _rope_freqs(rotary_dim, rope_theta, q.device) / rope_scale
    return _apply_rope_core(q, k, pos_ids.to(torch.float32), freqs, interleave)


@torch.no_grad()
def _apply_llama31_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs = _llama31_freqs(
        rotary_dim,
        rope_theta,
        rope_scale,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
        q.device,
    )
    positions = _positions_from_indptr(indptr, offsets, q.shape[0])
    return _apply_rope_core(q, k, positions, freqs, interleave)


@torch.no_grad()
def _apply_llama31_rope_pos_ids_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    freqs = _llama31_freqs(
        rotary_dim,
        rope_theta,
        rope_scale,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
        q.device,
    )
    return _apply_rope_core(q, k, pos_ids.to(torch.float32), freqs, interleave)


@torch.no_grad()
def _apply_rope_with_cos_sin_cache_reference(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with a precomputed cos/sin cache.

    cos_sin_cache is ``[max_seq_len, rotary_dim]`` where the first half is
    cos and the second half is sin. is_neox=True → half-split rotation;
    is_neox=False → interleaved rotation.
    """
    rotary_dim = cos_sin_cache.shape[-1]
    cos_cache = cos_sin_cache[:, : rotary_dim // 2]
    sin_cache = cos_sin_cache[:, rotary_dim // 2 :]
    cos = cos_cache[positions.to(torch.long)].unsqueeze(1)  # [nnz, 1, rotary_dim//2]
    sin = sin_cache[positions.to(torch.long)].unsqueeze(1)
    # Reshape flattened (nnz, H*D) → (nnz, H, D) for rotation.
    q_view = query.view(query.shape[0], -1, head_size)
    k_view = key.view(key.shape[0], -1, head_size)
    q_rope = _rotate(q_view.to(torch.float32), cos, sin, interleave=not is_neox)
    k_rope = _rotate(k_view.to(torch.float32), cos, sin, interleave=not is_neox)
    return (
        q_rope.reshape(query.shape).to(query.dtype),
        k_rope.reshape(key.shape).to(key.dtype),
    )


# ── Shared axes ───────────────────────────────────────────────────────────────

_RAGGED_AXES: Dict[str, _AxisT] = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "batch_size": Var(description="Number of sequences in the batch."),
    "num_q_heads": Const(abbrev="h"),
    "num_k_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
}

_POSIDS_AXES: Dict[str, _AxisT] = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads": Const(abbrev="h"),
    "num_k_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
}

_COSSIN_AXES: Dict[str, _AxisT] = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads_x_head_size": Const(
        description="num_q_heads * head_size (flattened query dimension).", abbrev=""
    ),
    "num_k_heads_x_head_size": Const(
        description="num_k_heads * head_size (flattened key dimension).", abbrev=""
    ),
    "head_size": Const(abbrev="d"),
    "max_seq_len": Var(description="cos_sin_cache length (max supported position)."),
    "rotary_dim": Const(
        description="Rotary dimension (cos+sin concatenated along last axis).",
        abbrev="",
    ),
}

# ── Base ragged RoPE (indptr + offsets) ──────────────────────────────────────

_RAGGED_INPUTS: Dict[str, _InputT] = {
    "q": Tensor(["nnz", "num_q_heads", "head_dim"]),
    "k": Tensor(["nnz", "num_k_heads", "head_dim"]),
    "indptr": Tensor(
        ["batch_size_plus_1"],
        dtype="int32",
        description="Ragged batch indptr, shape (batch_size + 1).",
    ),
    "offsets": Tensor(
        ["batch_size"],
        dtype="int32",
        description="Per-sequence starting position offset.",
    ),
    "rotary_dim": Scalar(
        "int32",
        optional=True,
        description="If None, uses head_dim. Rotate only the first `rotary_dim` dims.",
    ),
    "interleave": Scalar(
        "int32",
        optional=True,
        description="Bool: interleaved (True) vs half-split (False) rotation.",
    ),
    "rope_scale": Scalar("float32", optional=True, description="Scale factor."),
    "rope_theta": Scalar("float32", optional=True, description="Theta value."),
}

apply_rope_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope",
    description="Standard RoPE on ragged q/k using indptr + per-seq offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_RAGGED_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
    reference=_apply_rope_reference,
)

apply_rope_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_inplace",
    description="In-place standard RoPE; q and k are mutated.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_RAGGED_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
    reference=_apply_rope_reference,
)

# ── pos_ids RoPE ──────────────────────────────────────────────────────────────

_POSIDS_INPUTS: Dict[str, _InputT] = {
    "q": Tensor(["nnz", "num_q_heads", "head_dim"]),
    "k": Tensor(["nnz", "num_k_heads", "head_dim"]),
    "pos_ids": Tensor(["nnz"], dtype="int32", description="Per-token position index."),
    "rotary_dim": Scalar("int32", optional=True),
    "interleave": Scalar("int32", optional=True),
    "rope_scale": Scalar("float32", optional=True),
    "rope_theta": Scalar("float32", optional=True),
}

apply_rope_pos_ids_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_pos_ids",
    description="Standard RoPE using explicit per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_POSIDS_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    tags=["status:verified"],
    reference=_apply_rope_pos_ids_reference,
)

apply_rope_pos_ids_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_pos_ids_inplace",
    description="In-place RoPE using explicit per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_POSIDS_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    tags=["status:verified"],
    reference=_apply_rope_pos_ids_reference,
)

# ── Llama 3.1 RoPE ────────────────────────────────────────────────────────────

_LLAMA31_EXTRA: Dict[str, _InputT] = {
    "low_freq_factor": Scalar(
        "float32", optional=True, description="Llama 3.1 low-frequency scaling factor."
    ),
    "high_freq_factor": Scalar(
        "float32", optional=True, description="Llama 3.1 high-frequency scaling factor."
    ),
    "old_context_len": Scalar(
        "int32", optional=True, description="Original pretraining context length."
    ),
}

_LLAMA31_RAGGED_INPUTS: Dict[str, _InputT] = {**_RAGGED_INPUTS, **_LLAMA31_EXTRA}
_LLAMA31_POSIDS_INPUTS: Dict[str, _InputT] = {**_POSIDS_INPUTS, **_LLAMA31_EXTRA}

apply_llama31_rope_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope",
    description="Llama 3.1 RoPE on ragged q/k with indptr + offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_LLAMA31_RAGGED_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified", "model:llama"],
    reference=_apply_llama31_rope_reference,
)

apply_llama31_rope_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_inplace",
    description="In-place Llama 3.1 RoPE with indptr + offsets.",
    axes={**_RAGGED_AXES, "batch_size_plus_1": Var(description="batch_size + 1.")},
    inputs=_LLAMA31_RAGGED_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified", "model:llama"],
    reference=_apply_llama31_rope_reference,
)

apply_llama31_rope_pos_ids_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_pos_ids",
    description="Llama 3.1 RoPE using per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_LLAMA31_POSIDS_INPUTS,
    outputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "head_dim"], dtype_from="q"),
        "k_rope": Tensor(["nnz", "num_k_heads", "head_dim"], dtype_from="k"),
    },
    tags=["status:verified", "model:llama"],
    reference=_apply_llama31_rope_pos_ids_reference,
)

apply_llama31_rope_pos_ids_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="llama31_rope_pos_ids_inplace",
    description="In-place Llama 3.1 RoPE using per-token position ids.",
    axes=_POSIDS_AXES,
    inputs=_LLAMA31_POSIDS_INPUTS,
    outputs={
        "q": Tensor(
            ["nnz", "num_q_heads", "head_dim"],
            dtype_from="q",
            description="Updated q (in-place).",
        ),
        "k": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            dtype_from="k",
            description="Updated k (in-place).",
        ),
    },
    tags=["status:verified", "model:llama"],
    reference=_apply_llama31_rope_pos_ids_reference,
)

# ── cos/sin cache variant (SGL/vLLM-compatible) ───────────────────────────────

_COSSIN_INPUTS: Dict[str, _InputT] = {
    "positions": Tensor(
        ["nnz"], dtype="int32", description="Per-token position index."
    ),
    "query": Tensor(
        ["nnz", "num_q_heads_x_head_size"],
        description="Flattened query tensor (nnz, num_q_heads * head_size).",
    ),
    "key": Tensor(
        ["nnz", "num_k_heads_x_head_size"],
        description="Flattened key tensor (nnz, num_k_heads * head_size).",
    ),
    "head_size": Scalar("int32", description="Head dimension."),
    "cos_sin_cache": Tensor(
        ["max_seq_len", "rotary_dim"],
        dtype="float32",
        description="Precomputed cos+sin cache; cos first half, sin second half.",
    ),
    "is_neox": Scalar(
        "int32", optional=True, description="Bool: Neox (True) vs interleaved (False)."
    ),
}

apply_rope_with_cos_sin_cache_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_cos_sin_cache",
    description="RoPE with precomputed cos/sin cache (SGL/vLLM-compatible).",
    axes=_COSSIN_AXES,
    inputs=_COSSIN_INPUTS,
    outputs={
        "query_out": Tensor(["nnz", "num_q_heads_x_head_size"], dtype_from="query"),
        "key_out": Tensor(["nnz", "num_k_heads_x_head_size"], dtype_from="key"),
    },
    tags=["status:verified"],
    reference=_apply_rope_with_cos_sin_cache_reference,
)

apply_rope_with_cos_sin_cache_inplace_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_cos_sin_cache_inplace",
    description="In-place RoPE with precomputed cos/sin cache.",
    axes=_COSSIN_AXES,
    inputs=_COSSIN_INPUTS,
    outputs={
        "query": Tensor(
            ["nnz", "num_q_heads_x_head_size"],
            dtype_from="query",
            description="Updated query (in-place).",
        ),
        "key": Tensor(
            ["nnz", "num_k_heads_x_head_size"],
            dtype_from="key",
            description="Updated key (in-place).",
        ),
    },
    tags=["status:verified"],
    reference=_apply_rope_with_cos_sin_cache_reference,
)


# ── RoPE + FP8 quantize (split-rotary + non-rotary) ──────────────────────────


@torch.no_grad()
def _rope_quantize_fp8_reference(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope,
    k_nope,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype=None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference RoPE + FP8 quantize.

    Applies RoPE (cos/sin cache) to the rotary halves only, then quantizes
    all four tensors (``q_rope``, ``k_rope``, ``q_nope``, ``k_nope``) to
    FP8 (``float8_e4m3fn`` by default) after multiplying by the per-tensor
    quantization scale. Matches ``flashinfer.rope_quantize_fp8`` and its
    MLA wrapper ``mla_rope_quantize_fp8``.
    """
    quantize_dtype = quantize_dtype or torch.float8_e4m3fn
    rotary_dim = cos_sin_cache.shape[-1]
    cos_cache = cos_sin_cache[:, : rotary_dim // 2]
    sin_cache = cos_sin_cache[:, rotary_dim // 2 :]
    idx = pos_ids.to(torch.long)
    cos = cos_cache[idx].unsqueeze(1)
    sin = sin_cache[idx].unsqueeze(1)

    q_rope_rot = _rotate(q_rope.to(torch.float32), cos, sin, interleave=not is_neox).to(
        q_rope.dtype
    )
    # k_rope may be 2D (MLA: [nnz, rope_dim]) or 3D (GQA/MHA: [nnz, H, rope_dim]).
    k_rope_3d = k_rope.unsqueeze(1) if k_rope.dim() == 2 else k_rope
    k_rope_rot_3d = _rotate(
        k_rope_3d.to(torch.float32), cos, sin, interleave=not is_neox
    ).to(k_rope.dtype)
    k_rope_rot = k_rope_rot_3d.squeeze(1) if k_rope.dim() == 2 else k_rope_rot_3d

    # nope branches are optional; if None, materialize an empty tensor.
    nnz = q_rope.shape[0]
    num_q_heads = q_rope.shape[1]
    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_q_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        shape = (nnz, 0) if k_rope.dim() == 2 else (nnz, k_rope.shape[1], 0)
        k_nope = torch.empty(shape, dtype=k_rope.dtype, device=k_rope.device)

    def _q(t, scale):
        return (
            (t.to(torch.float32) * float(scale)).clamp(-448.0, 448.0).to(quantize_dtype)
        )

    return (
        _q(q_rope_rot, quant_scale_q),
        _q(k_rope_rot, quant_scale_kv),
        _q(q_nope, quant_scale_q),
        _q(k_nope, quant_scale_kv),
    )


_ROPE_QUANT_AXES: Dict[str, _AxisT] = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads": Const(abbrev="h"),
    "num_k_heads": Const(
        abbrev="kv", description="Number of K/V heads. 1 for MLA (rank-compressed)."
    ),
    "rope_dim": Const(description="Rotary dimension.", abbrev="rope"),
    "no_rope_dim": Var(
        description="Non-rotary dimension (can be 0 if no nope branch).",
    ),
    "max_seq_len": Var(description="cos_sin_cache length."),
    "rotary_dim": Const(abbrev=""),
}

_ROPE_QUANT_INPUTS: Dict[str, _InputT] = {
    "q_rope": Tensor(
        ["nnz", "num_q_heads", "rope_dim"], description="Query rotary part (fp16/bf16)."
    ),
    "k_rope": Tensor(
        ["nnz", "num_k_heads", "rope_dim"],
        description="Key rotary part. For MLA (num_k_heads=1) the kernel accepts a 2D [nnz, rope_dim] tensor.",
    ),
    "q_nope": Tensor(
        ["nnz", "num_q_heads", "no_rope_dim"],
        optional=True,
        description="Query non-rotary part; None allowed.",
    ),
    "k_nope": Tensor(
        ["nnz", "num_k_heads", "no_rope_dim"],
        optional=True,
        description="Key non-rotary part; None allowed. MLA uses a 2D [nnz, no_rope_dim] tensor.",
    ),
    "cos_sin_cache": Tensor(
        ["max_seq_len", "rotary_dim"],
        dtype="float32",
        description="Cos concatenated with sin along the last axis.",
    ),
    "pos_ids": Tensor(["nnz"], dtype="int32"),
    "is_neox": Scalar(
        "int32",
        optional=True,
        description="Bool: Neox half-split (True) vs interleaved (False).",
    ),
    "quant_scale_q": Scalar("float32", optional=True),
    "quant_scale_kv": Scalar("float32", optional=True),
}


rope_quantize_fp8_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_quantize_fp8",
    description=(
        "Fused RoPE + per-tensor FP8 quantize. Applies rotary embedding to "
        "the rotary half of Q/K and emits FP8 (e4m3 by default) Q/K for "
        "both rotary and non-rotary branches. Shared by GQA/MHA and MLA; "
        "MLA passes a 2D k_rope/k_nope (num_k_heads=1 compressed)."
    ),
    axes=_ROPE_QUANT_AXES,
    inputs=_ROPE_QUANT_INPUTS,
    outputs={
        "q_rope_out": Tensor(["nnz", "num_q_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "k_rope_out": Tensor(["nnz", "num_k_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "q_nope_out": Tensor(
            ["nnz", "num_q_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
        "k_nope_out": Tensor(
            ["nnz", "num_k_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
    },
    tags=["status:verified", "fused", "quantize:fp8"],
    reference=_rope_quantize_fp8_reference,
)


mla_rope_quantize_fp8_trace = TraceTemplate(
    op_type="rope",
    name_prefix="mla_rope_quantize_fp8",
    description=(
        "DeepSeek-MLA variant of rope_quantize_fp8. Identical math — the "
        "MLA wrapper just passes num_k_heads=1 (rank-compressed key/nope "
        "latents)."
    ),
    axes=_ROPE_QUANT_AXES,
    inputs=_ROPE_QUANT_INPUTS,
    outputs={
        "q_rope_out": Tensor(["nnz", "num_q_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "k_rope_out": Tensor(["nnz", "num_k_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "q_nope_out": Tensor(
            ["nnz", "num_q_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
        "k_nope_out": Tensor(
            ["nnz", "num_k_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
    },
    tags=["status:verified", "fused", "quantize:fp8", "mla"],
    reference=_rope_quantize_fp8_reference,
)


# ── RoPE + FP8 quantize + append paged KV cache (fused) ──────────────────────


@torch.no_grad()
def _rope_quantize_fp8_append_paged_kv_cache_reference(
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    v,
    cos_sin_cache,
    pos_ids,
    paged_kv_cache,
    kv_indices,
    kv_indptr,
    batch_indices,
    positions,
    is_neox: bool = True,
    quantize_dtype=None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
    kv_layout: str = "NHD",
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for rope_quantize_fp8_append_paged_kv_cache.

    Three steps:
      1. Apply RoPE to q_rope / k_rope (rotary halves only).
      2. Quantize to FP8 (per-tensor scales).
      3. Append the resulting K (and V for GQA/MHA) into paged_kv_cache.

    Returns quantized (q_rope_out, q_nope_out) for use in attention.

    ``paged_kv_cache`` is (k_cache, v_cache) for GQA/MHA, or
    (ckv_cache, kpe_cache) for MLA. This reference only models the
    append side for the GQA/MHA case — the MLA stack is covered by
    ``append_paged_mla_kv_cache_trace`` on the storage side.
    """
    quantize_dtype = quantize_dtype or torch.float8_e4m3fn
    # Step 1+2: RoPE then FP8 quantize.
    q_rope_q, k_rope_q, q_nope_q, k_nope_q = _rope_quantize_fp8_reference(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        is_neox=is_neox,
        quantize_dtype=quantize_dtype,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
    )
    # Step 3: append into paged cache (GQA/MHA) — materialize the quantized
    # K (as [K_nope ‖ K_rope]) and V into (k_cache, v_cache).
    is_mla = k_rope.dim() == 2
    if not is_mla and v is not None:
        v_q = (
            (v.to(torch.float32) * float(quant_scale_kv))
            .clamp(-448.0, 448.0)
            .to(quantize_dtype)
        )
        # Reassemble K from k_nope_q + k_rope_q along head_dim.
        k_full = torch.cat([k_nope_q, k_rope_q], dim=-1)
        k_cache, v_cache = paged_kv_cache
        nnz = batch_indices.shape[0]
        for i in range(nnz):
            b = int(batch_indices[i].item())
            pos = int(positions[i].item())
            page_offset = pos // page_size
            in_page_offset = pos % page_size
            idx_base = int(kv_indptr[b].item())
            page_id = int(kv_indices[idx_base + page_offset].item())
            if kv_layout == "NHD":
                k_cache[page_id, in_page_offset] = k_full[i]
                v_cache[page_id, in_page_offset] = v_q[i]
            else:  # HND
                k_cache[page_id, :, in_page_offset] = k_full[i]
                v_cache[page_id, :, in_page_offset] = v_q[i]
    return q_rope_q, q_nope_q


rope_quantize_fp8_append_paged_kv_cache_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_quantize_fp8_append_paged_kv_cache",
    description=(
        "Fused RoPE + FP8 quantize + append-K/V-to-paged-KV-cache. Returns "
        "quantized Q (for attention) and mutates the provided paged KV "
        "cache with quantized K and V. Shared by MLA, GQA and MHA; layout "
        "distinction is made by the shape of k_rope (2-D for MLA, 3-D "
        "otherwise) and the optional v tensor."
    ),
    axes={
        "nnz": Var(description="Total number of tokens across the batch."),
        "num_q_heads": Const(abbrev="h"),
        "num_k_heads": Const(abbrev="kv"),
        "rope_dim": Const(abbrev="rope"),
        "no_rope_dim": Var(),
        "head_dim": Var(description="Full KV head_dim (nope + rope); unset for MLA."),
        "max_seq_len": Var(),
        "rotary_dim": Const(abbrev=""),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "batch_size": Var(),
        "batch_size_plus_1": Var(),
        "num_kv_indices": Var(),
    },
    inputs={
        "q_rope": Tensor(["nnz", "num_q_heads", "rope_dim"]),
        "k_rope": Tensor(["nnz", "num_k_heads", "rope_dim"]),
        "q_nope": Tensor(["nnz", "num_q_heads", "no_rope_dim"], optional=True),
        "k_nope": Tensor(["nnz", "num_k_heads", "no_rope_dim"], optional=True),
        "v": Tensor(
            ["nnz", "num_k_heads", "head_dim"],
            optional=True,
            description="GQA/MHA value tensor (None for MLA).",
        ),
        "cos_sin_cache": Tensor(["max_seq_len", "rotary_dim"], dtype="float32"),
        "pos_ids": Tensor(["nnz"], dtype="int32"),
        "paged_kv_cache": Tensor(
            ["num_pages", "page_size", "num_k_heads", "head_dim"],
            description="Paged KV cache tuple — (k_cache, v_cache) for GQA/MHA, (ckv_cache, kpe_cache) for MLA.",
        ),
        "kv_indices": Tensor(["num_kv_indices"], dtype="int32"),
        "kv_indptr": Tensor(["batch_size_plus_1"], dtype="int32"),
        "batch_indices": Tensor(["nnz"], dtype="int32"),
        "positions": Tensor(["nnz"], dtype="int32"),
        "is_neox": Scalar("int32", optional=True),
        "quant_scale_q": Scalar("float32", optional=True),
        "quant_scale_kv": Scalar("float32", optional=True),
    },
    outputs={
        "q_rope_out": Tensor(["nnz", "num_q_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "q_nope_out": Tensor(
            ["nnz", "num_q_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
    },
    tags=["status:verified", "fused", "quantize:fp8"],
    reference=_rope_quantize_fp8_append_paged_kv_cache_reference,
)
