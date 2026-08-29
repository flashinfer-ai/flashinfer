"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

VibeCUDA MSA backend (compute capability 10.0/10.3). Hand-written SM100 CUDA
kernels: a warp-specialized UMMA/TMEM prefill for GQA group 16, a
block-bucketed UMMA/TMEM split-KV prefill for GQA group 4 paged KV, and a
general per-token / packed-pair HMMA fallback. Selected with
``backend="vibecuda"`` on the MSA entry points. The route and workspace
formulas below mirror the C++ dispatcher in csrc/msa_vibecuda exactly — keep
them in sync (tests cover every route).
"""

from __future__ import annotations

import threading
import weakref
from typing import Optional

import torch

from ..utils import get_compute_capability

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}

# ---- route mirrors of csrc/msa_vibecuda (source of truth: the .cu files) ----
_G16_GROUP = 16
_G16_Q_TILE = 16
_G16_MIN_TOPK = 12
_G16_MAX_TOPK = 64

_G4_GROUP = 4
_G4_Q_TILE = 32
_G4_MAX_TOPK = 8


def _g16_eligible(
    group: int, seqlen_q: int, topk: int, paged: bool, kv_fp8: bool
) -> bool:
    if group != _G16_GROUP or paged or kv_fp8:
        return False
    if seqlen_q < _G16_Q_TILE:
        return False
    return _G16_MIN_TOPK <= topk <= _G16_MAX_TOPK and topk % 4 == 0


def _g4_eligible(
    group: int,
    paged: bool,
    kv_fp8: bool,
    topk: int,
    nbatch: int,
    max_pages: int,
    num_kv_heads: int,
    total_q: int,
) -> bool:
    if group != _G4_GROUP or not paged or kv_fp8:
        return False
    if topk < 1 or topk > _G4_MAX_TOPK:
        return False
    nbuckets = num_kv_heads * nbatch * max_pages
    if nbuckets < 1 or nbuckets > 32768:
        return False
    if max_pages > 65536:
        return False
    slots = total_q * num_kv_heads * _G4_GROUP * topk
    return slots <= (1 << 22)


def _g4_workspace(
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    nbatch: int,
    max_pages: int,
) -> tuple[int, int]:
    """(ints, floats) scratch element counts; mirrors umma_g4_forward."""

    hn = num_kv_heads * total_q
    rows = total_q * num_q_heads
    slots = rows * topk
    nbuckets = num_kv_heads * nbatch * max_pages
    rows_bound = hn * topk + nbuckets * _G4_Q_TILE
    tiles_bound = hn * topk // _G4_Q_TILE + nbuckets
    need_i = (
        nbuckets * 2
        + hn
        + (nbuckets + 1) * 2
        + nbuckets * 5
        + tiles_bound
        + hn * topk
        + rows_bound
        + 1  # tile_total
        + 4  # route-barrier cnt/flag pairs (host-zeroed per call)
    )
    need_f = slots * 64 + slots * 2
    return need_i, need_f


def is_vibecuda_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a supported VibeCUDA MSA target."""

    normalized_device = torch.device(device)
    return (
        normalized_device.type == "cuda"
        and get_compute_capability(normalized_device) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


def require_vibecuda_device(device: torch.device | str) -> None:
    """Raise loudly when the explicit VibeCUDA backend is not supported."""

    if not is_vibecuda_device(device):
        normalized_device = torch.device(device)
        capability = (
            get_compute_capability(normalized_device)
            if normalized_device.type == "cuda"
            else None
        )
        raise RuntimeError(
            "the vibecuda MSA backend requires compute capability 10.0 or 10.3; "
            f"got {capability} on device {normalized_device}"
        )


def _select_target(device: torch.device):
    capability = get_compute_capability(device)
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "the vibecuda MSA backend supports compute capability 10.0 or 10.3, "
        f"got {capability[0]}.{capability[1]}"
    )


def _get_module(target: str):
    from ..jit.msa_vibecuda import get_msa_vibecuda_module

    return get_msa_vibecuda_module(target)


def _stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


# Shape-keyed constants (device index + sizes fully determine the contents,
# never input values): dummy scratch placeholders and the uniform cu_seqlens_q
# built by the decode entry. Caching them keeps per-call host overhead on par
# with the bundled backends on tiny workloads.
_shape_const: dict[tuple, torch.Tensor] = {}


def _shape_constant(key: tuple, make) -> torch.Tensor:
    cached = _shape_const.get(key)
    if cached is None:
        cached = make()
        if len(_shape_const) < 128:
            _shape_const[key] = cached
    return cached


def _dummies(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    di = _shape_constant(
        ("di", device.index), lambda: torch.empty(1, dtype=torch.int32, device=device)
    )
    df = _shape_constant(
        ("df", device.index), lambda: torch.empty(1, dtype=torch.float32, device=device)
    )
    return di, df


def _uniform_cu_q(batch_size: int, seqlen_q: int, device: torch.device) -> torch.Tensor:
    return _shape_constant(
        ("cuq", batch_size, seqlen_q, device.index),
        lambda: torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            seqlen_q,
            dtype=torch.int32,
            device=device,
        ),
    )


# Cached host-side read of the per-batch query lengths, keyed on the cu_q
# tensor identity and version exactly like the bundled CAKE route caches.
_uniform_q_len_cache: dict[int, tuple] = {}
_uniform_q_len_cache_lock = threading.Lock()


def _resolve_uniform_seqlen_q(cu_q: torch.Tensor, batch_size: int) -> int:
    """Right-aligned MSA semantics use one scalar seqlen_q per call; verify
    uniformity of the per-batch query lengths (one sync per cu_q mutation)."""

    def resolve() -> int:
        lengths = (cu_q[1:] - cu_q[:-1]).cpu().tolist()
        first = lengths[0]
        if any(length != first for length in lengths):
            raise NotImplementedError(
                "the vibecuda MSA backend requires uniform per-batch query "
                "lengths for one call; got ragged cu_seqlens_q"
            )
        return int(first)

    tensor_id = id(cu_q)
    try:
        version = cu_q._version
    except RuntimeError:
        return resolve()
    with _uniform_q_len_cache_lock:
        cached = _uniform_q_len_cache.get(tensor_id)
        if cached is not None and cached[0]() is cu_q and cached[1] == version:
            return cached[2]
    seqlen_q = resolve()
    with _uniform_q_len_cache_lock:
        if len(_uniform_q_len_cache) >= 64:
            dead_keys = [
                key for key, value in _uniform_q_len_cache.items() if value[0]() is None
            ]
            for key in dead_keys:
                _uniform_q_len_cache.pop(key, None)
            if len(_uniform_q_len_cache) >= 64:
                _uniform_q_len_cache.clear()
        _uniform_q_len_cache[tensor_id] = (weakref.ref(cu_q), version, seqlen_q)
    return seqlen_q


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    causal: bool,
) -> tuple[int, int, int, int, bool]:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16/fp16, got {q.dtype}")
    if q.ndim != 3 or q.shape[2] != 128:
        raise ValueError("q must have shape (total_q, num_q_heads, 128)")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    total_q, num_q_heads, _ = (int(x) for x in q.shape)
    if total_q <= 0 or num_q_heads <= 0:
        raise ValueError("q must contain at least one query and one head")

    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("k and v must be CUDA tensors")
    paged = page_table is not None
    expected_ndim = 4 if paged else 3
    if k.ndim != expected_ndim or v.ndim != expected_ndim:
        raise ValueError(
            "k/v must be flat (total_k, num_kv_heads, 128) or paged "
            "(num_pages, num_kv_heads, 128, 128) matching page_table"
        )
    if not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("k/v must be contiguous")
    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("k and v must share shape and dtype")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_fp8:
        if q.dtype != torch.bfloat16:
            raise NotImplementedError(
                "fp8 K/V requires bf16 Q on the vibecuda MSA backend"
            )
    elif k.dtype != q.dtype:
        raise ValueError("dense k/v dtype must match q")
    if k.device != q.device or v.device != q.device:
        raise ValueError("k/v must be on the same device as q")

    num_kv_heads = int(k.shape[1])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    if not 0 < group_size <= 16:
        raise ValueError("the GQA group size must be in [1, 16]")

    if (
        not isinstance(q2k_indices, torch.Tensor)
        or q2k_indices.device != q.device
        or q2k_indices.dtype != torch.int32
        or q2k_indices.ndim != 3
        or tuple(q2k_indices.shape[:2]) != (num_kv_heads, total_q)
        or not q2k_indices.is_contiguous()
    ):
        raise ValueError(
            "q2k_indices must be contiguous CUDA int32 with shape "
            "(num_kv_heads, total_q, topk)"
        )

    if paged:
        if seqused_k is None:
            raise ValueError("paged KV requires seqused_k")
        if k.shape[2] != 128 or k.shape[3] != 128:
            raise ValueError(
                "paged k/v must have shape (num_pages, num_kv_heads, 128, 128)"
            )
        if (
            not isinstance(page_table, torch.Tensor)
            or page_table.device != q.device
            or page_table.dtype != torch.int32
            or page_table.ndim != 2
            or not page_table.is_contiguous()
        ):
            raise ValueError(
                "page_table must be contiguous CUDA int32 with shape "
                "(batch_size, max_pages)"
            )
    elif not causal and cu_seqlens_k is None:
        # non-causal flat still needs cu_seqlens_k for batch membership
        pass
    return total_q, num_q_heads, num_kv_heads, group_size, kv_fp8


def _reject_unsupported_options(
    *,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    workspace,
) -> None:
    if return_softmax_lse or return_temperature_lse:
        raise NotImplementedError(
            "the vibecuda MSA backend does not emit softmax LSE outputs"
        )
    if any(
        value is not None
        for value in (k_scale, v_scale, k_global_scale, v_global_scale)
    ):
        raise NotImplementedError(
            "K/V scale arguments are not supported by the vibecuda MSA backend"
        )
    if workspace is not None:
        raise NotImplementedError(
            "CUDA graph capture workspaces are not supported by the "
            "vibecuda MSA backend"
        )


def _run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    seqlen_q: int,
    causal: bool,
) -> torch.Tensor:
    device = q.device
    total_q, num_q_heads, num_kv_heads, group_size, kv_fp8 = _validate_inputs(
        q, k, v, q2k_indices, page_table, seqused_k, cu_k, causal
    )
    if cu_q.dtype != torch.int32 or not cu_q.is_contiguous():
        cu_q = cu_q.to(device=device, dtype=torch.int32).contiguous()
    if cu_k.dtype != torch.int32 or not cu_k.is_contiguous():
        cu_k = cu_k.to(device, dtype=torch.int32).contiguous()
    batch_size = int(cu_q.numel()) - 1
    topk = int(q2k_indices.shape[2])
    paged = page_table is not None
    kv_kind = 2 if kv_fp8 else (0 if q.dtype == torch.bfloat16 else 1)

    out = torch.empty_like(q)
    target = _select_target(device)
    module = _get_module(target)

    k_arg = k.view(torch.uint8) if kv_fp8 else k
    v_arg = v.view(torch.uint8) if kv_fp8 else v

    dummy_i32, dummy_f32 = _dummies(device)
    ws_int, ws_float = dummy_i32, dummy_f32
    need_i = need_f = 0

    g16_ok = _g16_eligible(group_size, seqlen_q, topk, paged, kv_fp8)
    if not g16_ok and _g4_eligible(
        group_size,
        paged,
        kv_fp8,
        topk,
        batch_size,
        int(page_table.shape[1]) if paged else 0,
        num_kv_heads,
        total_q,
    ):
        need_i, need_f = _g4_workspace(
            total_q,
            num_q_heads,
            num_kv_heads,
            topk,
            batch_size,
            int(page_table.shape[1]),
        )
        ws_int = torch.empty(need_i, dtype=torch.int32, device=device)
        ws_float = torch.empty(need_f, dtype=torch.float32, device=device)

    module.run(
        q,
        k_arg,
        v_arg,
        out,
        q2k_indices,
        cu_q,
        cu_k,
        page_table if paged else dummy_i32,
        seqused_k if paged else dummy_i32,
        ws_int,
        ws_float,
        kv_kind,
        int(seqlen_q),
        int(causal),
        need_i,
        need_f,
        _stream_ptr(device),
    )
    return out


def vibecuda_msa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    workspace=None,
):
    """Sparse prefill on the VibeCUDA backend (compute capability 10.0/10.3)."""

    del lse_temperature_scale
    _reject_unsupported_options(
        return_softmax_lse=return_softmax_lse,
        return_temperature_lse=return_temperature_lse,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        workspace=workspace,
    )
    if q_offset is not None:
        raise NotImplementedError(
            "the vibecuda MSA backend always right-aligns queries to the KV "
            "sequence (q_offset=None semantics)"
        )
    if softmax_scale is not None and abs(float(softmax_scale) - 128**-0.5) > 1e-12:
        raise NotImplementedError(
            "the vibecuda MSA backend uses the fixed head_dim**-0.5 softmax scale"
        )
    cu_q = cu_seqlens_q
    if cu_q.ndim != 1 or cu_q.numel() < 2:
        raise ValueError("cu_seqlens_q must contain at least two entries")
    batch_size = int(cu_q.numel()) - 1
    if page_table is None and cu_seqlens_k is None:
        raise ValueError("flat K/V requires cu_seqlens_k")
    if page_table is not None and seqused_k is None:
        raise ValueError("paged K/V requires seqused_k")
    if cu_seqlens_k is None:
        # derive from seqused_k (paged path kernels only need the lengths).
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        cu_k[1:] = seqused_k.to(torch.int32).cumsum(0, dtype=torch.int32)
    else:
        cu_k = cu_seqlens_k
    seqlen_q = _resolve_uniform_seqlen_q(cu_q, batch_size)
    return _run(
        q=q,
        k=k,
        v=v,
        q2k_indices=q2k_indices,
        cu_q=cu_q,
        cu_k=cu_k,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=seqlen_q,
        causal=causal,
    )


def vibecuda_msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    partial_dtype: Optional[torch.dtype] = None,
    force_fused: Optional[bool] = None,
    workspace=None,
):
    """Sparse decode on the VibeCUDA backend (compute capability 10.0/10.3).

    The backend always runs its internally routed schedule, which already
    covers fused and split regimes per shape; ``force_fused`` is advisory and
    does not change the numerics. ``partial_dtype`` is accepted for interface
    parity (the kernels use their own fixed partial precision).
    """

    del partial_dtype
    _reject_unsupported_options(
        return_softmax_lse=return_softmax_lse,
        return_temperature_lse=False,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        workspace=workspace,
    )
    if q_offset is not None:
        raise NotImplementedError(
            "the vibecuda MSA backend always right-aligns queries to the KV "
            "sequence (q_offset=None semantics)"
        )
    if softmax_scale is not None and abs(float(softmax_scale) - 128**-0.5) > 1e-12:
        raise NotImplementedError(
            "the vibecuda MSA backend uses the fixed head_dim**-0.5 softmax scale"
        )
    if force_fused not in (None, True, False):
        raise ValueError("force_fused must be True, False, or None")
    total_q = int(q.shape[0])
    if seqlen_q <= 0 or total_q % seqlen_q:
        raise ValueError("q rows must equal batch_size * positive seqlen_q")
    batch_size = total_q // seqlen_q
    k_ndim = 4 if page_table is not None else 3
    if k.ndim != k_ndim:
        raise ValueError(
            "k/v must be flat (total_k, num_kv_heads, 128) or paged "
            "(num_pages, num_kv_heads, 128, 128) matching page_table"
        )
    if page_table is None and cu_seqlens_k is None:
        raise ValueError("flat K/V requires cu_seqlens_k")
    if page_table is not None and seqused_k is None:
        raise ValueError("paged K/V requires seqused_k")
    if cu_seqlens_k is None:
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        cu_k[1:] = seqused_k.to(torch.int32).cumsum(0, dtype=torch.int32)
    else:
        cu_k = cu_seqlens_k
    cu_q = _uniform_cu_q(batch_size, int(seqlen_q), q.device)
    return _run(
        q=q,
        k=k,
        v=v,
        q2k_indices=q2k_indices,
        cu_q=cu_q,
        cu_k=cu_k,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=int(seqlen_q),
        causal=causal,
    )


__all__ = [
    "is_vibecuda_device",
    "require_vibecuda_device",
    "vibecuda_msa_sparse_attention",
    "vibecuda_msa_sparse_decode_attention",
]
