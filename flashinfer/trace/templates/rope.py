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
from ._init_helpers import make_pos_ids, make_rope_cos_sin_cache

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


def _rope_ragged_init(
    *,
    nnz: int,
    batch_size: int = 4,
    batch_size_plus_1: int = 0,  # derived
    num_q_heads: int = 32,
    num_k_heads: int = 8,
    head_dim: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ragged RoPE variants (indptr + offsets).

    ``nnz`` is the total token count; ``batch_size`` controls how many
    sequences they're split across. ``batch_size_plus_1`` is derived.
    Sourced from ``tests/attention/test_rope.py`` and the example call
    in ``tests/trace/example.py``.
    """
    del batch_size_plus_1  # derived
    torch.manual_seed(seed)
    batch_size = max(1, batch_size)
    # Distribute nnz evenly across batch_size sequences.
    base = nnz // batch_size
    rem = nnz % batch_size
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cur = 0
    for i in range(batch_size):
        cur += base + (1 if i < rem else 0)
        indptr[i + 1] = cur
    q = torch.randn(nnz, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(nnz, num_k_heads, head_dim, dtype=torch.bfloat16, device=device)
    offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return {"q": q, "k": k, "indptr": indptr, "offsets": offsets}


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
    init=_rope_ragged_init,
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
    init=_rope_ragged_init,
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


def _rope_pos_ids_init(
    *,
    nnz: int,
    num_q_heads: int = 32,
    num_k_heads: int = 8,
    head_dim: int = 128,
    max_seq_len: int = 8192,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for pos_ids RoPE variants (no indptr; per-token positions)."""
    torch.manual_seed(seed)
    q = torch.randn(nnz, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(nnz, num_k_heads, head_dim, dtype=torch.bfloat16, device=device)
    pos_ids = make_pos_ids(nnz, max_seq_len, device=device)
    return {"q": q, "k": k, "pos_ids": pos_ids}


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
    init=_rope_pos_ids_init,
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
    init=_rope_pos_ids_init,
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
    init=_rope_ragged_init,
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
    init=_rope_ragged_init,
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
    init=_rope_pos_ids_init,
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
    init=_rope_pos_ids_init,
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


def _rope_cos_sin_cache_init(
    *,
    nnz: int,
    num_q_heads_x_head_size: int = 4096,  # 32 * 128
    num_k_heads_x_head_size: int = 1024,  # 8 * 128
    head_size: int = 128,
    max_seq_len: int = 8192,
    rotary_dim: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``apply_rope_with_cos_sin_cache``.

    Sourced from ``tests/trace/example.py`` cos/sin section. ``query``
    and ``key`` are flattened (``num_heads * head_size``) per the
    SGL/vLLM convention.
    """
    torch.manual_seed(seed)
    query = torch.randn(
        nnz, num_q_heads_x_head_size, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(nnz, num_k_heads_x_head_size, dtype=torch.bfloat16, device=device)
    cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, rotary_dim, device=device)
    positions = make_pos_ids(nnz, max_seq_len, device=device)
    return {
        "positions": positions,
        "query": query,
        "key": key,
        "head_size": int(head_size),
        "cos_sin_cache": cos_sin_cache,
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
    init=_rope_cos_sin_cache_init,
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
    init=_rope_cos_sin_cache_init,
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
        abbrev="kv", description="Number of K/V heads for the GQA/MHA path."
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
        description="Key rotary part.",
    ),
    "q_nope": Tensor(
        ["nnz", "num_q_heads", "no_rope_dim"],
        optional=True,
        description="Query non-rotary part; None allowed.",
    ),
    "k_nope": Tensor(
        ["nnz", "num_k_heads", "no_rope_dim"],
        optional=True,
        description="Key non-rotary part; None allowed.",
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


def _rope_quantize_fp8_init(
    *,
    nnz: int,
    num_q_heads: int = 8,
    num_k_heads: int = 2,
    rope_dim: int = 64,
    no_rope_dim: int = 64,
    max_seq_len: int = 4096,
    rotary_dim: int = 64,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``rope_quantize_fp8``.

    Mirrors ``tests/attention/test_rope.py``: GQA K tensors are 3D and the
    unit-test path uses interleaved RoPE (``is_neox=False``).
    """
    torch.manual_seed(seed)
    q_rope = torch.randn(
        nnz, num_q_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    k_rope = torch.randn(
        nnz, num_k_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    q_nope = torch.randn(
        nnz, num_q_heads, no_rope_dim, dtype=torch.bfloat16, device=device
    )
    k_nope = torch.randn(
        nnz, num_k_heads, no_rope_dim, dtype=torch.bfloat16, device=device
    )
    cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, rotary_dim, device=device)
    pos_ids = make_pos_ids(nnz, max_seq_len, device=device)
    return {
        "q_rope": q_rope,
        "k_rope": k_rope,
        "q_nope": q_nope,
        "k_nope": k_nope,
        "cos_sin_cache": cos_sin_cache,
        "pos_ids": pos_ids,
        "is_neox": False,
    }


def _mla_rope_quantize_fp8_init(
    *,
    nnz: int,
    num_q_heads: int = 128,
    rope_dim: int = 64,
    no_rope_dim: int = 512,
    max_seq_len: int = 4096,
    rotary_dim: int = 64,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``mla_rope_quantize_fp8``.

    This mirrors ``tests/attention/test_rope.py``: Q tensors are 3D, while
    rank-compressed MLA K tensors are 2D ``[nnz, dim]``.
    """
    torch.manual_seed(seed)
    q_rope = torch.randn(
        nnz, num_q_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    k_rope = torch.randn(nnz, rope_dim, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(
        nnz, num_q_heads, no_rope_dim, dtype=torch.bfloat16, device=device
    )
    k_nope = torch.randn(nnz, no_rope_dim, dtype=torch.bfloat16, device=device)
    cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, rotary_dim, device=device)
    pos_ids = make_pos_ids(nnz, max_seq_len, device=device).to(torch.int64)
    return {
        "q_rope": q_rope,
        "k_rope": k_rope,
        "q_nope": q_nope,
        "k_nope": k_nope,
        "cos_sin_cache": cos_sin_cache,
        "pos_ids": pos_ids,
        "is_neox": False,
    }


rope_quantize_fp8_trace = TraceTemplate(
    op_type="rope",
    name_prefix="rope_quantize_fp8",
    description=(
        "Fused RoPE + per-tensor FP8 quantize. Applies rotary embedding to "
        "the rotary half of Q/K and emits FP8 (e4m3 by default) Q/K for "
        "both rotary and non-rotary branches for the GQA/MHA layout."
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
    init=_rope_quantize_fp8_init,
)


# MLA-specific axes/inputs: num_k_heads is collapsed (always 1), so k_rope and
# k_nope are passed as 2D tensors. Keeping these separate from the GQA dicts
# above so the dumped JSON reflects the actual rank-2 K shape.
_MLA_ROPE_QUANT_AXES: Dict[str, _AxisT] = {
    "nnz": Var(description="Total number of tokens across the batch."),
    "num_q_heads": Const(abbrev="h"),
    "rope_dim": Const(description="Rotary dimension.", abbrev="rope"),
    "no_rope_dim": Var(
        description="Non-rotary dimension (can be 0 if no nope branch).",
    ),
    "max_seq_len": Var(description="cos_sin_cache length."),
    "rotary_dim": Const(abbrev=""),
}

_MLA_ROPE_QUANT_INPUTS: Dict[str, _InputT] = {
    "q_rope": Tensor(
        ["nnz", "num_q_heads", "rope_dim"],
        description="Query rotary part (fp16/bf16).",
    ),
    "k_rope": Tensor(
        ["nnz", "rope_dim"],
        description="Key rotary part. MLA passes a 2D [nnz, rope_dim] tensor "
        "(num_k_heads=1 rank-compressed).",
    ),
    "q_nope": Tensor(
        ["nnz", "num_q_heads", "no_rope_dim"],
        optional=True,
        description="Query non-rotary part; None allowed.",
    ),
    "k_nope": Tensor(
        ["nnz", "no_rope_dim"],
        optional=True,
        description="Key non-rotary part. MLA passes a 2D [nnz, no_rope_dim] tensor.",
    ),
    "cos_sin_cache": Tensor(
        ["max_seq_len", "rotary_dim"],
        dtype="float32",
        description="Cos concatenated with sin along the last axis.",
    ),
    "pos_ids": Tensor(["nnz"], dtype="int64"),
    "is_neox": Scalar(
        "int32",
        optional=True,
        description="Bool: Neox half-split (True) vs interleaved (False).",
    ),
    "quant_scale_q": Scalar("float32", optional=True),
    "quant_scale_kv": Scalar("float32", optional=True),
}


mla_rope_quantize_fp8_trace = TraceTemplate(
    op_type="rope",
    name_prefix="mla_rope_quantize_fp8",
    description=(
        "DeepSeek-MLA variant of rope_quantize_fp8. Identical math — the "
        "MLA wrapper passes rank-2 K tensors (num_k_heads=1 collapsed)."
    ),
    axes=_MLA_ROPE_QUANT_AXES,
    inputs=_MLA_ROPE_QUANT_INPUTS,
    outputs={
        "q_rope_out": Tensor(["nnz", "num_q_heads", "rope_dim"], dtype="float8_e4m3fn"),
        "k_rope_out": Tensor(["nnz", "rope_dim"], dtype="float8_e4m3fn"),
        "q_nope_out": Tensor(
            ["nnz", "num_q_heads", "no_rope_dim"], dtype="float8_e4m3fn"
        ),
        "k_nope_out": Tensor(["nnz", "no_rope_dim"], dtype="float8_e4m3fn"),
    },
    tags=["status:verified", "fused", "quantize:fp8", "mla"],
    reference=_rope_quantize_fp8_reference,
    init=_mla_rope_quantize_fp8_init,
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


def _rope_quantize_fp8_append_paged_kv_cache_init(
    *,
    nnz: int,
    num_q_heads: int = 8,
    num_k_heads: int = 2,
    rope_dim: int = 64,
    no_rope_dim: int = 64,
    head_dim: int = 0,  # derived: rope_dim + no_rope_dim
    max_seq_len: int = 4096,
    rotary_dim: int = 64,
    num_pages: int = 4,
    page_size: int = 16,
    batch_size: int = 2,
    batch_size_plus_1: int = 0,  # derived
    num_kv_indices: int = 0,  # derived
    num_pages_per_seq: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``rope_quantize_fp8_append_paged_kv_cache``.

    GQA path: returns a (k_cache, v_cache) tuple under
    ``paged_kv_cache``. Page capacity is grown so it fits the full
    ``nnz`` tokens; ``(batch_indices, positions)`` follow the same
    contiguous per-sequence layout as ``flashinfer.get_batch_indices_positions``.
    """
    del batch_size_plus_1, num_kv_indices, num_pages_per_seq  # derived
    torch.manual_seed(seed)
    if nnz < 0:
        raise ValueError(f"nnz must be non-negative, got {nnz}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    expected_head_dim = rope_dim + no_rope_dim
    if head_dim < 0:
        raise ValueError(f"head_dim must be non-negative, got {head_dim}")
    if head_dim not in (0, expected_head_dim):
        raise ValueError(
            "head_dim must be 0 or equal to rope_dim + no_rope_dim "
            f"({expected_head_dim}), got {head_dim}"
        )
    # Auto-grow num_pages so capacity >= nnz (so the returned tensors
    # keep the requested ``nnz`` along axis 0).
    min_pages = batch_size * max(1, (nnz + page_size - 1) // page_size // batch_size)
    num_pages = max(num_pages, min_pages, batch_size)
    pages_per_seq = max(1, num_pages // max(1, batch_size))
    while pages_per_seq * page_size * batch_size < nnz:
        pages_per_seq += 1
    num_pages = max(num_pages, pages_per_seq * batch_size)
    capacity_per_seq = pages_per_seq * page_size

    full_dim = expected_head_dim if head_dim == 0 else head_dim
    q_rope = torch.randn(
        nnz, num_q_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    k_rope = torch.randn(
        nnz, num_k_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    q_nope = torch.randn(
        nnz, num_q_heads, no_rope_dim, dtype=torch.bfloat16, device=device
    )
    k_nope = torch.randn(
        nnz, num_k_heads, no_rope_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(nnz, num_k_heads, full_dim, dtype=torch.bfloat16, device=device)
    cos_sin_cache = make_rope_cos_sin_cache(max_seq_len, rotary_dim, device=device)
    pos_ids = make_pos_ids(nnz, max_seq_len, device=device)
    k_cache = torch.zeros(
        num_pages,
        page_size,
        num_k_heads,
        full_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * pages_per_seq
    )
    # Distribute nnz across batch_size, clamping per-seq counts to
    # capacity so positions stay valid.
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.rand((batch_size,), generator=g)
    raw = raw / raw.sum() * nnz
    seq_lens_cpu = raw.round().to(torch.int64)
    diff = int(nnz - seq_lens_cpu.sum().item())
    if diff != 0:
        seq_lens_cpu[0] = max(0, seq_lens_cpu[0].item() + diff)
    seq_lens_cpu = torch.minimum(
        seq_lens_cpu, torch.full_like(seq_lens_cpu, capacity_per_seq)
    )
    overflow = nnz - int(seq_lens_cpu.sum().item())
    for i in range(batch_size):
        if overflow <= 0:
            break
        room = capacity_per_seq - int(seq_lens_cpu[i].item())
        if room > 0:
            take = min(room, overflow)
            seq_lens_cpu[i] += take
            overflow -= take
    seq_lens = seq_lens_cpu.to(torch.int32).to(device)
    append_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    append_indptr[1:] = torch.cumsum(seq_lens, dim=0).to(torch.int32)
    bidx_parts = []
    pos_parts = []
    for b, length in enumerate(seq_lens_cpu.tolist()):
        if length <= 0:
            continue
        bidx_parts.append(torch.full((length,), b, dtype=torch.int32, device=device))
        pos_parts.append(torch.arange(length, dtype=torch.int32, device=device))
    if bidx_parts:
        bidx = torch.cat(bidx_parts)
        positions = torch.cat(pos_parts)
    else:
        bidx = torch.empty((0,), dtype=torch.int32, device=device)
        positions = torch.empty((0,), dtype=torch.int32, device=device)
    assert int(append_indptr[-1].item()) == nnz, (
        "internal: capacity grow failed to fit nnz"
    )
    return {
        "q_rope": q_rope,
        "k_rope": k_rope,
        "q_nope": q_nope,
        "k_nope": k_nope,
        "v": v,
        "cos_sin_cache": cos_sin_cache,
        "pos_ids": pos_ids,
        "paged_kv_cache": (k_cache, v_cache),
        "kv_indices": kv_indices,
        "kv_indptr": kv_indptr,
        "batch_indices": bidx,
        "positions": positions,
        "is_neox": False,
        "page_size": int(page_size),
        "kv_layout": "NHD",
    }


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
        "k_cache": Tensor(
            ["num_pages", "page_size", "num_k_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=0,
            description="Paged K cache from the paged_kv_cache tuple.",
        ),
        "v_cache": Tensor(
            ["num_pages", "page_size", "num_k_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=1,
            description="Paged V cache from the paged_kv_cache tuple.",
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
    init=_rope_quantize_fp8_append_paged_kv_cache_init,
)


# ── HY3 Q/K RMSNorm + RoPE + paged KV store (B200 decode fast path) ─────────


def _qk_rmsnorm_rope_store_hy3_fp8_decode_init(
    *,
    batch_size: int,
    batch_plus_one: int = 0,
    pages_per_request: int = 1,
    num_pages: int = 0,
    max_position: int = 4096,
    packed_width: int = 0,
    num_q_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 64,
    scale_count: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build the validated uniform one-row decode shape for B200."""
    del batch_plus_one  # derived from batch_size
    if (num_q_heads, num_kv_heads, head_dim, scale_count) != (64, 8, 128, 1):
        raise ValueError("HY3 B200 trace requires 64Q/8KV/D128 and one scale")
    expected_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    if packed_width not in (0, expected_width):
        raise ValueError(f"packed_width must be 0 or {expected_width}")
    if pages_per_request <= 0 or page_size <= 0:
        raise ValueError("pages_per_request and page_size must be positive")
    num_pages = max(num_pages, batch_size * pages_per_request)
    torch.manual_seed(seed)
    packed_qkv = torch.randn(
        batch_size, expected_width, dtype=torch.bfloat16, device=device
    )
    cos_sin_cache = make_rope_cos_sin_cache(max_position, head_dim, device=device)
    sequence_lengths = (
        torch.arange(batch_size, dtype=torch.int32, device=device) % page_size + 1
    )
    q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(
        batch_size * pages_per_request, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_request)
    cache_shape = (num_pages, page_size, num_kv_heads, head_dim)
    key_cache = torch.zeros(cache_shape, dtype=torch.float8_e4m3fn, device=device)
    value_cache = torch.zeros_like(key_cache)
    return {
        "packed_qkv": packed_qkv,
        "cos_sin_cache": cos_sin_cache,
        "sequence_lengths": sequence_lengths,
        "q_indptr": q_indptr,
        "block_table": block_table,
        "paged_kv_cache": (key_cache, value_cache),
        "is_prefill": False,
        "q_norm_weight": torch.linspace(0.75, 1.25, head_dim, device=device),
        "k_norm_weight": torch.linspace(1.25, 0.75, head_dim, device=device),
        "norm_policy": 2,
        "quant_policy": 1,
        "k_scale": torch.tensor([0.5], dtype=torch.float32, device=device),
        "v_scale": torch.tensor([0.25], dtype=torch.float32, device=device),
        "out_q": torch.empty(
            batch_size,
            num_q_heads,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "out_q_scale": torch.empty(
            batch_size, num_q_heads, dtype=torch.float32, device=device
        ),
        "split_k_flag": torch.empty(
            batch_size, num_kv_heads, dtype=torch.int32, device=device
        ),
        "uniform_one_token_decode": True,
    }


qk_rmsnorm_rope_store_hy3_fp8_decode_trace = TraceTemplate(
    op_type="rope",
    name_prefix="qk_rmsnorm_rope_append_paged_kv_cache_hy3_fp8_decode",
    description=(
        "B200 uniform one-row decode specialization that fuses Q/K RMSNorm, "
        "NeoX RoPE, dynamic FP8 Q quantization, and NHD paged K/V storage. "
        "The paged cache tuple is updated in place. Prefill, BF16, static-FP8, "
        "and redirected K/V outputs are intentionally outside this trace variant."
    ),
    axes={
        "batch_size": Var(),
        "batch_plus_one": Var(),
        "pages_per_request": Var(),
        "num_pages": Var(),
        "max_position": Var(),
        "packed_width": Const(abbrev=""),
        "num_q_heads": Const(value=64, abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "page_size": Const(abbrev="ps"),
        "scale_count": Const(value=1, abbrev=""),
    },
    inputs={
        "packed_qkv": Tensor(["batch_size", "packed_width"], dtype="bfloat16"),
        "cos_sin_cache": Tensor(["max_position", "head_dim"], dtype="float32"),
        "sequence_lengths": Tensor(["batch_size"], dtype="int32"),
        "q_indptr": Tensor(["batch_plus_one"], dtype="int32"),
        "block_table": Tensor(["batch_size", "pages_per_request"], dtype="int32"),
        "k_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=0,
            dtype="float8_e4m3fn",
            description="NHD key cache updated in place.",
        ),
        "v_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=1,
            dtype="float8_e4m3fn",
            description="NHD value cache updated in place.",
        ),
        "is_prefill": Scalar("int32"),
        "q_norm_weight": Tensor(["head_dim"], dtype="float32"),
        "k_norm_weight": Tensor(["head_dim"], dtype="float32"),
        "norm_policy": Scalar("int32"),
        "quant_policy": Scalar("int32"),
        "k_scale": Tensor(["scale_count"], dtype="float32"),
        "v_scale": Tensor(["scale_count"], dtype="float32"),
        "max_sequence_length": Scalar("int32", optional=True),
        "fp8_upper_bound": Scalar("float32", optional=True),
        "out_q": Tensor(
            ["batch_size", "num_q_heads", "head_dim"],
            dtype="float8_e4m3fn",
            optional=True,
        ),
        "out_q_scale": Tensor(
            ["batch_size", "num_q_heads"], dtype="float32", optional=True
        ),
        "split_k_flag": Tensor(
            ["batch_size", "num_kv_heads"], dtype="int32", optional=True
        ),
        "uniform_one_token_decode": Scalar("int32"),
    },
    outputs={
        "out_q": Tensor(
            ["batch_size", "num_q_heads", "head_dim"],
            dtype="float8_e4m3fn",
            param="out_q",
        ),
        "out_q_scale": Tensor(
            ["batch_size", "num_q_heads"],
            dtype="float32",
            param="out_q_scale",
        ),
        "split_k_flag": Tensor(
            ["batch_size", "num_kv_heads"],
            dtype="int32",
            param="split_k_flag",
        ),
    },
    constraints=[
        "batch_plus_one == batch_size + 1",
        "packed_width == (64 + 2 * num_kv_heads) * head_dim",
        "head_dim == 128",
        "num_kv_heads == 8",
        "num_pages >= batch_size * pages_per_request",
    ],
    tags=["status:experimental", "fused", "quantize:fp8", "arch:sm100"],
    init=_qk_rmsnorm_rope_store_hy3_fp8_decode_init,
)


def qk_rmsnorm_rope_store_hy3_trace_dispatch(save_dir=None, name=None, **kwargs):
    """Trace only the validated SM100 uniform dynamic-FP8 decode shape."""
    del save_dir, name
    if "packed_qkv" not in kwargs and "batch_size" in kwargs:
        return qk_rmsnorm_rope_store_hy3_fp8_decode_trace
    packed_qkv = kwargs.get("packed_qkv")
    paged_kv_cache = kwargs.get("paged_kv_cache")
    if (
        not isinstance(packed_qkv, torch.Tensor)
        or packed_qkv.ndim != 2
        or not isinstance(paged_kv_cache, (tuple, list))
        or len(paged_kv_cache) != 2
    ):
        return None
    key_cache, value_cache = paged_kv_cache
    if (
        not isinstance(key_cache, torch.Tensor)
        or not isinstance(value_cache, torch.Tensor)
        or key_cache.ndim != 4
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != torch.float8_e4m3fn
        or value_cache.dtype != torch.float8_e4m3fn
    ):
        return None
    num_kv_heads, head_dim = key_cache.shape[2:]
    num_q_heads = packed_qkv.shape[1] // head_dim - 2 * num_kv_heads
    quant_policy = kwargs.get("quant_policy")
    if (
        bool(kwargs.get("is_prefill"))
        or int(kwargs.get("norm_policy", 0)) != 2
        or quant_policy not in (None, 1)
        or not bool(kwargs.get("uniform_one_token_decode"))
        or (num_q_heads, num_kv_heads, head_dim) != (64, 8, 128)
        or kwargs.get("q_norm_weight") is None
        or kwargs.get("k_norm_weight") is None
        or kwargs.get("k_scale") is None
        or kwargs.get("v_scale") is None
        or kwargs.get("q_scale_inverse") is not None
        or kwargs.get("out_k") is not None
        or kwargs.get("out_v") is not None
    ):
        return None
    return qk_rmsnorm_rope_store_hy3_fp8_decode_trace


qk_rmsnorm_rope_store_hy3_trace_dispatch.templates = (  # type: ignore[attr-defined]
    qk_rmsnorm_rope_store_hy3_fp8_decode_trace,
)
