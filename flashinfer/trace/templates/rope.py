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
