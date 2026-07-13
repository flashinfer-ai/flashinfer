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
"""

import functools
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import magi_ffa_flex_trace

_SUPPORTED_LAYOUTS = ("NHD", "HND")

# FlashInfer's own floor for nvidia-cutlass-dsl (see requirements.txt).
# MagiAttention's installer is known to downgrade this package (e.g. to
# 4.3.5), which breaks `import flashinfer` (cute.nvgpu.OperandMajorMode
# needs >=4.5.0). Documented, validated override: magi_attention==1.1.0.post10
# with nvidia-cutlass-dsl>=4.5.0 reinstalled AFTER MagiAttention.
_MIN_CUTLASS_DSL_VERSION = (4, 5, 0)


def _check_cutlass_dsl_not_downgraded() -> None:
    try:
        installed = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return  # flashinfer's own cute-dsl paths surface this when needed
    installed_tuple = tuple(
        int(part) for part in installed.split(".")[:3] if part.isdigit()
    )
    if installed_tuple < _MIN_CUTLASS_DSL_VERSION:
        minimum = ".".join(str(v) for v in _MIN_CUTLASS_DSL_VERSION)
        raise RuntimeError(
            f"nvidia-cutlass-dsl=={installed} is below FlashInfer's requirement "
            f">={minimum}; it was likely downgraded by the MagiAttention install. "
            f'Restore it with: pip install "nvidia-cutlass-dsl>={minimum}" '
            "(run AFTER installing MagiAttention). Validated combination: "
            f"magi_attention==1.1.0.post10 + nvidia-cutlass-dsl>={minimum}."
        )


@functools.cache
def _load_flex_flash_attn_func():
    """Lazily import MagiAttention's native ``flex_flash_attn_func``.

    MagiAttention (SandAI, Apache-2.0) is an optional, separately-installed
    dependency. The import is deferred so that importing FlashInfer never
    requires MagiAttention to be present.
    """
    try:
        from magi_attention.api import flex_flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "MagiAttention is required to use flashinfer.magi_ffa.flex_flash_attn. "
            "It is an optional dependency (SandAI MagiAttention, Apache-2.0) and is "
            "not installed automatically; install it in the active Python environment "
            "before calling flex_flash_attn(). Note: MagiAttention's install may "
            "downgrade nvidia-cutlass-dsl; afterwards run pip install "
            '"nvidia-cutlass-dsl>=4.5.0" to keep flashinfer importable.'
        ) from exc
    _check_cutlass_dsl_not_downgraded()
    return flex_flash_attn_func


def _check_range_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have shape (num_ranges, 2)")


def _check_attn_type_map(
    attn_type_map: Optional[torch.Tensor], num_ranges: int
) -> None:
    if attn_type_map is None:
        return
    if not isinstance(attn_type_map, torch.Tensor):
        raise TypeError("attn_type_map must be a torch.Tensor")
    if attn_type_map.dtype != torch.int32:
        raise TypeError("attn_type_map must have dtype torch.int32")
    if attn_type_map.ndim != 1:
        raise ValueError("attn_type_map must have shape (num_ranges,)")
    if attn_type_map.numel() != num_ranges:
        raise ValueError("attn_type_map must have the same length as q_ranges")


def _swap_token_head_dims(t: torch.Tensor) -> torch.Tensor:
    # HND <-> NHD: swaps the leading (num_heads, num_tokens) dims in either
    # direction. Works for the 3D q/k/v/out tensors and the 2D lse.
    return t.transpose(0, 1).contiguous()


@flashinfer_api(trace=magi_ffa_flex_trace)
def flex_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    tensor_layout: str = "NHD",
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Single-GPU Flex Flash Attention via MagiAttention (experimental, optional).

    This is a thin **adapter** around MagiAttention's ``flex_flash_attn_func``: it
    presents a FlashInfer-normalized surface (layout in, output out) while keeping
    MagiAttention's native call internal. MagiAttention is an **optional,
    separately-installed dependency** (SandAI MagiAttention, Apache-2.0); calling
    this without it installed raises an informative ``ImportError``.

    .. note::
        This path is experimental. Because MagiAttention is not a FlashInfer
        dependency, its real kernel path is exercised only when MagiAttention is
        installed (e.g. locally or in an opt-in CI job); FlashInfer's default CI
        runs only the dependency-free unit tests.

    FFA uses a ragged, range-based mask model that does not map onto FlashInfer's
    paged / ``indptr`` abstractions, so it is exposed as a dedicated entry point
    rather than a ``backend=`` of the paged wrappers. Arch support (Hopper FFA,
    Blackwell/Ampere via FFA_FA4) is owned by MagiAttention; this adapter adds no
    architecture gate.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Ragged query/key/value. With ``tensor_layout="NHD"`` (default) the shapes are
        ``q: (num_tokens_q, num_heads_q, head_dim)`` and
        ``k, v: (num_tokens_kv, num_heads_kv, head_dim)``. With ``"HND"`` the first
        two dims are swapped; the adapter transposes to token-major internally.
    q_ranges, k_ranges : torch.Tensor
        ``(num_ranges, 2)`` int32 range tensors encoding the attention mask
        (MagiAttention's representation). Required.
    attn_type_map : torch.Tensor, optional
        ``(num_ranges,)`` int32 per-range mask type (0=full, 1=causal,
        2=inverse-causal, 3=bidirectional-causal). ``None`` => full attention.
    return_lse : bool
        If True, also return the log-sum-exp tensor.
    tensor_layout : {"NHD", "HND"}
        Layout of ``q/k/v`` (and of the returned ``out``/``lse``).
    **kwargs
        Forwarded verbatim to MagiAttention's ``flex_flash_attn_func`` (e.g.
        ``softmax_scale``, ``softcap``, ``sink``, ``sink_layout``, ``deterministic``,
        ``sm_margin``, ``auto_range_merge``, ``sparse_load``, ``ref_block_size``,
        ``max_seqlen_q``). These are MagiAttention-owned options; refer to its docs.
        ``return_max_logits`` is rejected because this adapter returns only
        ``out``/``lse`` and would otherwise silently drop the extra metadata.

    Returns
    -------
    out : torch.Tensor
        Attention output in the requested ``tensor_layout``.
    (out, lse) : Tuple[torch.Tensor, torch.Tensor]
        When ``return_lse=True``; ``lse`` is float32 in the requested layout.
    """
    if tensor_layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"tensor_layout must be one of {_SUPPORTED_LAYOUTS}")
    if kwargs.get("return_max_logits"):
        raise ValueError(
            "return_max_logits is not supported by flashinfer.magi_ffa.flex_flash_attn; "
            "it returns out / (out, lse) only."
        )
    if q_ranges is None or k_ranges is None:
        raise ValueError("q_ranges and k_ranges are required")
    _check_range_tensor("q_ranges", q_ranges)
    _check_range_tensor("k_ranges", k_ranges)
    if q_ranges.shape[0] != k_ranges.shape[0]:
        raise ValueError("q_ranges and k_ranges must have the same length")
    if q_ranges.device != k_ranges.device:
        raise ValueError("q_ranges and k_ranges must be on the same device")
    _check_attn_type_map(attn_type_map, q_ranges.shape[0])
    if attn_type_map is not None and attn_type_map.device != q_ranges.device:
        raise ValueError("attn_type_map must be on the same device as q_ranges")

    # Fail fast on the missing optional dependency before touching tensors.
    func = _load_flex_flash_attn_func()

    if tensor_layout == "HND":
        q, k, v = (_swap_token_head_dims(t) for t in (q, k, v))

    out, meta = func(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        **kwargs,
    )

    if tensor_layout == "HND":
        out = _swap_token_head_dims(out)

    if not return_lse:
        return out

    lse = getattr(meta, "lse", None)
    if lse is None:
        raise RuntimeError("MagiAttention did not return lse metadata")
    if tensor_layout == "HND":
        lse = _swap_token_head_dims(lse)
    return out, lse


__all__ = ["flex_flash_attn"]
