# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Sparse-MLA SM120 public Wrapper/config facade and legacy launch adapters.

Wrapper and functional entry points share prepared execution in
:mod:`._sparse_mla_sm120_prepared`. The lazy raw module and format bridges live in
:mod:`._sparse_mla_sm120_execution`; compatibility aliases remain here.
DSv3.2, GLM-NSA, DSV4, GLM53_NOPE, DOTS3_SWA and explicit DSV4.1 storage
are supported. DSV4 NVFP4 has its own format and calibration policy.

Default ordinary routing uses the crossover and CPB policy in
:mod:`._sparse_mla_sm120_policy`: eligible calls up to 64 tokens decode unless
calibration prefers prefill; larger calls prefill when supported. Missing
calibration keeps decode-first and the launcher's CPB heuristic.
Explicit DSV4.1 FP8/BF16 routes remain untuned, with decode CPB=1.

Ordinary FP8 prefill requires whole 64-wide index tiles and family-specific
heads; DOTS3_SWA additionally requires topk >= 513. Full-BF16 DSV4.1 prefill
instead supports runtime H=1..128 and positive ragged topk. DSV4 NVFP4
has a separate compiled envelope. The public config query describes decode
capabilities and loads a compiled host module on its first use; import alone
neither compiles nor initializes CUDA. Eligibility and workspace facts come
from C++, not Python support tables.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from ._sparse_mla_sm120_execution import (
    get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
    normalize_kv_scale_format as _normalize_kv_scale_format,
    resolve_model_type as _resolve_model_type,
    KV_SCALE_FORMATS as _KV_SCALE_FORMATS,  # noqa: F401
)
from ..api_logging import flashinfer_api
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

# The _DECODE_*_DISPATCH pair sets are re-exported here on purpose: vLLM's
# has_flashinfer_sparse_mla_sm120_config probes membership of
# ``flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH`` directly.
from ._sparse_mla_sm120_policy import (
    _BI,
    _DECODE_DSV3_2_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_DSV4_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_DSV4_1_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_MAX_TOKENS,
    _DECODE_DSV3_2_TOPKS,
    _DECODE_DSV4_TOPKS,
    _DECODE_DSV4_1_TOPK,
    _DECODE_DOTS3_SWA_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_DOTS3_SWA_TOPK,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_DSV4_1,
    _MODEL_TYPE_GLM53_NOPE,
    _MODEL_TYPE_GLM_NSA,
    _MODEL_TYPE_DOTS3_SWA,
    _DECODE_GLM53_NOPE_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_GLM53_NOPE_TOPK,
    _D_V_BY_MODEL_TYPE,
    _SUPPORTED_D_V,
    _decode_chunk_width,
    _decode_scratch_heads,
    _MODEL_TYPE_TO_FAMILY,
    _D_V,
    KernelVariant,
    _normalize_prefill_impl,
    _resolve_cpb,
    plan,
)

# Public calibration API, re-exported for the flashinfer.mla lazy export.
from ._sparse_mla_sm120_calibration import (  # noqa: E402
    SparseMLASm120CalibrationReport,  # noqa: F401  (lazy re-export)
    calibrate_sparse_mla_sm120,  # noqa: F401  (lazy re-export)
)

logger = logging.getLogger(__name__)

_KV_CACHE_FORMATS = frozenset({"fp8", "nvfp4"})


@dataclass(frozen=True)
class SparseMLASm120DecodeConfig:
    """Instantiated decode-kernel set for one SM120 sparse-MLA kernel family.

    Decode-form calls (``num_tokens <= max_num_tokens``) prefer a standalone
    decode kernel when their shape matches one of the instantiations
    described here. For FP8, the prefill orchestrator can serve remaining
    decode-form shapes in its own envelope; NVFP4 currently uses the same
    exact head/top-k envelope for both kernels. Crossover calibration may
    route an eligible shape to prefill. This config describes decode only.

    Attributes
    ----------
    d_qk : int
        Query/key head dim served by this family (``512`` for DSv4 /
        GLM53_NOPE, ``576`` for DSv3.2 / GLM-NSA, ``1088`` for the
        DOTS3_SWA sliding-window family, whose ``d_v`` is then 1024).
    page_block_size : int
        Default page size; the required size unless ``page_block_size_is_runtime``.
    page_block_size_is_runtime : bool
        Accept independent positive page sizes instead of a fixed main page size.
    max_num_tokens : int
        Largest ``num_tokens`` routed to the decode kernels (inclusive).
    topks : frozenset[int]
        The calibrated top-k values (the crossover sweep points). When
        ``topk_is_runtime`` is true, this documents measured values rather
        than the eligibility boundary; otherwise it is the exact set.
    min_topk : int
        Smallest legal ``topk`` (the indices-row width). ``513`` for the
        sliding-window family (the window must fit the buffer); ``1``
        elsewhere.
    max_num_heads : int
        Upper bound of the head-count envelope.
    kv_cache_format : str
        Packed cache format described by this entry (``"fp8"`` or
        ``"nvfp4"``).
    bytes_per_token : int
        Logical packed-cache bytes per token. For inline-scale models this is
        the payload minimum: callers may allocate wider padded rows since the
        kernels take the gmem row advance as a runtime stride.
    head_counts : Optional[frozenset[int]]
        Exact instantiated head counts when the kernel has no runtime-head
        fallback. ``None`` means every count in ``[1, max_num_heads]``.
    topk_is_runtime : bool
        Whether every ``topk >= min_topk`` is accepted. When false, only
        values in ``topks`` are instantiated.
    extra_page_block_sizes : frozenset[int]
        Exact page sizes accepted by the optional secondary cache when the
        family has a finite set exposed here. Empty means unspecified.
    """

    d_qk: int
    page_block_size: int
    max_num_tokens: int
    topks: frozenset[int]
    min_topk: int
    max_num_heads: int
    kv_cache_format: str = "fp8"
    bytes_per_token: int = 0
    head_counts: Optional[frozenset[int]] = None
    topk_is_runtime: bool = True
    extra_page_block_sizes: frozenset[int] = frozenset()
    page_block_size_is_runtime: bool = False

    def supported_num_heads(self) -> tuple[int, ...]:
        """Sorted instantiated head counts, including any runtime-H envelope."""
        if self.head_counts is not None:
            return tuple(sorted(self.head_counts))
        return tuple(range(1, self.max_num_heads + 1))

    def supported_topk(self, num_heads: Optional[int] = None) -> tuple[int, ...]:
        """Sorted calibrated top-k values for ``num_heads`` (or any head count).

        For a runtime-top-k family these are the values with measured
        crossover data; otherwise this is the exact instantiated set."""
        head_supported = num_heads is None or (
            num_heads in self.head_counts
            if self.head_counts is not None
            else 1 <= num_heads <= self.max_num_heads
        )
        if head_supported:
            return tuple(sorted(self.topks))
        return ()

    def supports_decode(
        self,
        num_heads: int,
        topk: int,
        *,
        num_tokens: int = 1,
        page_block_size: Optional[int] = None,
    ) -> bool:
        """True iff a decode-form call with this shape is decode-instantiated.

        Decode-instantiated shapes may still route to prefill according to
        calibration. This predicate describes the decode envelope only.
        """
        if page_block_size is None:
            page_block_size = self.page_block_size
        head_supported = (
            num_heads in self.head_counts
            if self.head_counts is not None
            else 1 <= num_heads <= self.max_num_heads
        )
        topk_supported = (
            topk >= self.min_topk if self.topk_is_runtime else topk in self.topks
        )
        return (
            num_tokens <= self.max_num_tokens
            and (
                page_block_size > 0
                if self.page_block_size_is_runtime
                else page_block_size == self.page_block_size
            )
            and head_supported
            and topk_supported
        )


@flashinfer_api
def supported_sparse_mla_sm120_configs(
    *, kv_cache_format: str = "fp8"
) -> dict[str, SparseMLASm120DecodeConfig]:
    """Enumerate the instantiated SM120 sparse-MLA decode kernel configurations.

    Lets callers validate a serving configuration at initialization time
    instead of discovering an uninstantiated ``(num_heads, topk)`` pair on the
    first decode-form request.

    Parameters
    ----------
    kv_cache_format : {"fp8", "nvfp4"}
        Storage format whose independently calibrated kernel envelope is
        requested. Defaults to ``"fp8"`` for backward compatibility.

    Returns
    -------
    dict[str, SparseMLASm120DecodeConfig]
        Mapping from kernel family to its instantiated decode set. FP8 returns
        DSv4, DSv3.2, GLM-NSA, GLM53_NOPE, DOTS3_SWA, and DSV4.1 entries. NVFP4
        currently returns the independently calibrated DSv4 entry.

    Examples
    --------
    >>> import flashinfer
    >>> configs = flashinfer.mla.supported_sparse_mla_sm120_configs()
    >>> configs["dsv4"].supports_decode(num_heads=64, topk=256)
    True
    >>> nvfp4 = flashinfer.mla.supported_sparse_mla_sm120_configs(
    ...     kv_cache_format="nvfp4"
    ... )
    >>> nvfp4["dsv4"].bytes_per_token
    384
    """
    if kv_cache_format not in _KV_CACHE_FORMATS:
        raise ValueError(
            f"kv_cache_format must be either 'fp8' or 'nvfp4', got {kv_cache_format!r}"
        )
    if kv_cache_format == "nvfp4":
        from ._sparse_mla_sm120_execution import dsv4_nvfp4_format_info

        facts = dsv4_nvfp4_format_info()
        return {
            "dsv4": SparseMLASm120DecodeConfig(
                d_qk=facts["query_dim"],
                page_block_size=facts["page_size"],
                max_num_tokens=_DECODE_MAX_TOKENS,
                topks=frozenset(facts["topks"]),
                min_topk=min(facts["topks"]),
                max_num_heads=max(facts["heads"]),
                kv_cache_format="nvfp4",
                bytes_per_token=facts["bytes_per_token"],
                head_counts=frozenset(facts["heads"]),
                topk_is_runtime=False,
                extra_page_block_sizes=frozenset(facts["extra_page_sizes"]),
            )
        }

    from ._sparse_mla_sm120_execution import format_info

    probes = (
        _DECODE_DSV3_2_TOPKS,
        _DECODE_DSV4_TOPKS,
        _DECODE_DSV3_2_TOPKS,
        frozenset({_DECODE_GLM53_NOPE_TOPK}),
        frozenset({_DECODE_DOTS3_SWA_TOPK}),
        frozenset({_DECODE_DSV4_1_TOPK}),
    )
    result = {}
    for model, family in _MODEL_TYPE_TO_FAMILY.items():
        info = format_info(model)
        result[family] = SparseMLASm120DecodeConfig(
            d_qk=info["query_dim"],
            page_block_size=info["page_size"],
            max_num_tokens=_DECODE_MAX_TOKENS,
            topks=probes[model],
            min_topk=info["min_topk"],
            max_num_heads=info["max_heads"],
            bytes_per_token=info["bytes_per_token"],
            page_block_size_is_runtime=bool(info["runtime_page"]),
        )
    result["glm_nsa"] = result["dsv3_2"]
    return result


def _decode_dispatch_error_message(
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_qk: int,
    page_block_size: int,
    model_type: int,
    extra_topk: int,
) -> str:
    """Build the decode dispatch-miss error, naming the mismatched parameter."""
    family = _MODEL_TYPE_TO_FAMILY[model_type]
    config = supported_sparse_mla_sm120_configs()[family]
    reasons = []
    if d_qk != config.d_qk:
        reasons.append(
            f"d_qk={d_qk} does not match the {family} decode family "
            f"(requires d_qk={config.d_qk})"
        )
    if page_block_size <= 0:
        reasons.append("page_block_size must be positive")
    elif (
        not config.page_block_size_is_runtime
        and page_block_size != config.page_block_size
    ):
        reasons.append(
            f"page_block_size={page_block_size} is unsupported; decode kernels "
            f"are instantiated only for page_block_size={config.page_block_size}"
        )
    if topk < config.min_topk:
        reasons.append(
            f"topk={topk} is below the {family} decode minimum "
            f"(topk >= {config.min_topk}"
            + (
                ", the 513-wide sliding window must fit the indices buffer)"
                if config.min_topk > 1
                else ")"
            )
            + f"; calibrated topk values: {list(config.supported_topk())}"
        )
    if not 1 <= num_heads <= config.max_num_heads:
        reasons.append(
            f"num_heads={num_heads} exceeds the decode envelope "
            f"[1, {config.max_num_heads}]"
        )
    # The dispatch branches guarantee at least one reason; the fallback only
    # guards future drift between them and this diagnosis.
    detail = "; ".join(reasons) or "no matching decode instantiation"
    return (
        "SM120 sparse-MLA has no decode kernel for this shape: "
        f"num_tokens={num_tokens}, num_heads={num_heads}, topk={topk}, "
        f"d_qk={d_qk}, page_block_size={page_block_size}, "
        f"model_type={family}, extra_topk={extra_topk}. "
        f"Mismatch: {detail}. "
        f"The decode instantiations (num_tokens <= {_DECODE_MAX_TOKENS}) and "
        "the prefill envelope both reject it. "
        "Query supported shapes at init time with "
        "flashinfer.mla.supported_sparse_mla_sm120_configs()."
    )


def _expected_d_v(model_type: Optional[int] = None) -> int:
    """d_v for a model type; the DeepSeek-family default when unspecified."""
    if model_type is None:
        return _D_V
    try:
        return _D_V_BY_MODEL_TYPE[model_type]
    except KeyError:
        raise ValueError(
            f"Unsupported SM120 sparse-MLA model_type={model_type}"
        ) from None


def _require_d_v(d_v: int, model_type: Optional[int] = None) -> None:
    expected = _expected_d_v(model_type)
    if int(d_v) != expected:
        raise ValueError(f"SM120 sparse-MLA requires d_v == {expected}, got {d_v}")


def _require_supported_d_v(d_v: int) -> None:
    """Check ``d_v`` against the supported set without loading the JIT module.

    The strict per-model check runs again at plan/run time through
    :func:`_require_d_v` once ``d_qk`` has resolved the model type.
    """
    if int(d_v) not in _SUPPORTED_D_V:
        raise ValueError(
            f"SM120 sparse-MLA requires d_v in {sorted(_SUPPORTED_D_V)}, got {d_v}"
        )


def _check_last_dim(
    tensor: torch.Tensor, name: str, model_type: Optional[int] = None
) -> None:
    expected = _expected_d_v(model_type)
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} last dimension must be {expected}, got shape {tuple(tensor.shape)}"
        )


def _bytes_per_token_for_model_type(model_type: int) -> int:
    from ._sparse_mla_sm120_execution import format_info

    return format_info(model_type)["bytes_per_token"]


def _inline_cache_block_contiguous(kv_cache: torch.Tensor) -> bool:
    """Block-contiguity predicate for inline-scale (DSv3.2/GLM) caches.

    Inline-scale kernels address the cache as a flat token array with a
    runtime row stride, so padded *rows* (a wider last dim or a sliced view)
    are fine, but pages must pack rows back-to-back. The FFI binding
    re-checks the same invariant; this only produces an earlier, clearer
    error at the wrapper layer.
    """
    if kv_cache.is_contiguous():
        return True
    if kv_cache.ndim == 2 or kv_cache.stride(-1) != 1:
        return False
    token_axis = 2 if kv_cache.ndim == 4 and kv_cache.shape[1] == 1 else 1
    return kv_cache.stride(0) == kv_cache.shape[token_axis] * kv_cache.stride(
        token_axis
    )


def _packed_kv_page_block_size(
    kv_cache: torch.Tensor,
    *,
    model_type: int,
    name: str,
    extra_fp4: bool = False,
) -> int:
    bytes_per_token = 288 if extra_fp4 else _bytes_per_token_for_model_type(model_type)
    if kv_cache.ndim == 2:
        block_bytes = int(kv_cache.shape[1])
        if block_bytes % bytes_per_token != 0:
            raise ValueError(
                f"{name} 2-D block width {block_bytes} is not divisible by "
                f"{bytes_per_token} bytes/token"
            )
        return block_bytes // bytes_per_token
    if kv_cache.ndim == 3:
        # >=, not ==: callers may pad each token row so layers with different
        # geometries share one KV cache group. The packed payload stays at the
        # row start and the kernel advances by the real row stride.
        if kv_cache.shape[-1] < bytes_per_token:
            raise ValueError(
                f"{name} last dim must be >= {bytes_per_token}, got {kv_cache.shape[-1]}"
            )
        return int(kv_cache.shape[1])
    if kv_cache.ndim == 4:
        # >=, not ==: callers may pad each token row so layers with different
        # geometries share one KV cache group. The packed payload stays at the
        # row start and the kernel advances by the real row stride.
        if kv_cache.shape[-1] < bytes_per_token:
            raise ValueError(
                f"{name} last dim must be >= {bytes_per_token}, got {kv_cache.shape[-1]}"
            )
        if kv_cache.shape[1] == 1:
            # HND: [num_pages, 1, page_block_size, bytes_per_token].
            return int(kv_cache.shape[2])
        if kv_cache.shape[2] == 1:
            # NHD: [num_pages, page_block_size, 1, bytes_per_token].
            return int(kv_cache.shape[1])
        raise ValueError(
            f"{name} must have a singleton KV-head axis in dim 1 (HND) or "
            f"dim 2 (NHD), got shape {tuple(kv_cache.shape)}"
        )
    raise ValueError(f"{name} must have ndim 2, 3, or 4, got {kv_cache.ndim}")


def _decode_scratch_views(
    mid_out: Optional[torch.Tensor],
    mid_lse: Optional[torch.Tensor],
    num_tokens: int,
    num_heads: int,
    num_splits: int,
    d_v: int,
    *,
    scratch_heads: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve caller-supplied scratch buffers for split-K decode kernels.

    By default, the scratch head dim is the true ``num_heads`` for the
    dedicated ``num_heads=8`` instantiation and HPB(16)-aligned otherwise.
    ``scratch_heads`` overrides that policy for kernels with an exact-H ABI.
    """
    if scratch_heads is None:
        scratch_heads = _decode_scratch_heads(num_heads)
    if mid_out is None or mid_lse is None:
        raise ValueError(
            "SM120 sparse-MLA decode requires caller-supplied mid_out and "
            "mid_lse scratch. Allocate shapes "
            f"[{num_tokens}, {scratch_heads}, {num_splits}, {d_v}] bf16 and "
            f"[{num_tokens}, {scratch_heads}, {num_splits}] fp32."
        )
    need_out = (num_tokens, scratch_heads, num_splits, d_v)
    need_lse = (num_tokens, scratch_heads, num_splits)
    # Exact-size scratch needs no slicing; identity views cost ~5us/call.
    if mid_out.shape == need_out and mid_lse.shape == need_lse:
        return mid_out, mid_lse
    if any(mid_out.size(d) < need_out[d] for d in range(4)):
        raise ValueError(
            f"mid_out shape {tuple(mid_out.shape)} too small for required "
            f"[num_tokens={num_tokens}, num_heads={scratch_heads}, "
            f"num_splits={num_splits}, d_v={d_v}]"
        )
    if any(mid_lse.size(d) < need_lse[d] for d in range(3)):
        raise ValueError(
            f"mid_lse shape {tuple(mid_lse.shape)} too small for required "
            f"[num_tokens={num_tokens}, num_heads={scratch_heads}, "
            f"num_splits={num_splits}]"
        )
    if not mid_out.is_contiguous() or not mid_lse.is_contiguous():
        raise ValueError("decode scratch must be contiguous")
    return (
        mid_out.view(-1)[: num_tokens * scratch_heads * num_splits * d_v].view(
            need_out
        ),
        mid_lse.view(-1)[: num_tokens * scratch_heads * num_splits].view(need_lse),
    )


@functools.cache
def get_sparse_mla_sm120_module():
    """Build and cache the sparse-MLA SM120 module + bound custom op."""
    module = _get_sparse_mla_sm120_decode_module()

    @register_custom_op(
        "flashinfer::sparse_mla_sm120_paged_attention",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
    def _paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        sm_scale: float,
        d_v: int,
        model_type: int,
        prefill_impl: int,
        topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        extra_fp4: bool,
    ) -> None:
        num_tokens, num_heads, d_qk = q.shape
        topk = indices.shape[-1]
        _require_d_v(d_v, model_type)
        _check_last_dim(output, "output", model_type)
        if num_tokens == 0:
            # Empty request: outputs are already-sized empty tensors; a kernel
            # launch would hit a grid.x=0 CUDA error.
            return

        kv_pbs = _packed_kv_page_block_size(
            kv_cache, model_type=model_type, name="kv_cache"
        )
        if model_type in (
            _MODEL_TYPE_DSV3_2,
            _MODEL_TYPE_GLM_NSA,
            _MODEL_TYPE_GLM53_NOPE,
        ) and not _inline_cache_block_contiguous(kv_cache):
            # Inline-scale prefill kernels address the cache as a flat token
            # array with a runtime row stride — and crossover can route any
            # decode-form call there, so the restriction cannot wait for a
            # prefill-routed call to fire. Padded rows are honored everywhere;
            # only inter-block gaps are rejected.
            raise ValueError(
                "inline-scale (DSv3.2/GLM) KV caches must pack rows "
                "contiguously within each block through this entry (padded "
                "rows are fine): prefill-routed calls address the cache as a "
                "flat token array, and the calibrated crossover can route "
                "decode-form calls to prefill"
            )
        extra_topk = int(extra_indices.size(-1)) if extra_indices is not None else 0
        planned = plan(
            num_tokens,
            num_heads,
            topk,
            model_type,
            kv_pbs,
            extra_kv_cache is not None,
            prefill_impl,
            q.device,
            extra_topk=extra_topk,
            extra_fp4=extra_fp4,
        )
        if planned is None:
            # Neither the decode instantiations nor the prefill envelope
            # serves this shape.
            raise ValueError(
                _decode_dispatch_error_message(
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    topk=topk,
                    d_qk=d_qk,
                    page_block_size=kv_pbs,
                    model_type=model_type,
                    extra_topk=extra_topk,
                )
            )
        if planned.variant is KernelVariant.DECODE_SPLITK:
            if model_type in (
                _MODEL_TYPE_DSV4,
                _MODEL_TYPE_DOTS3_SWA,
                _MODEL_TYPE_DSV4_1,
            ):
                num_splits = _decode_dsv4_num_splits(topk, extra_topk, model_type)
                mid_out_view, mid_lse_view = _decode_scratch_views(
                    mid_out, mid_lse, num_tokens, num_heads, num_splits, d_v
                )
                # FFI binding extracts the true block stride from
                # kv_cache.stride(0), so paged layouts with padded strides
                # and microbench 2-D layouts both work.
                sparse_mla_sm120_decode_dsv4(
                    q,
                    kv_cache,
                    indices,
                    mid_out_view,
                    mid_lse_view,
                    output,
                    out_lse,
                    sm_scale,
                    topk_length=topk_length,
                    attn_sink=attn_sink,
                    extra_kv_cache=extra_kv_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_topk_length,
                    chunks_per_block=planned.cpb,
                    model_type=model_type,
                    extra_fp4=extra_fp4,
                )
                return

            num_splits = (topk + _BI - 1) // _BI
            mid_out_view, mid_lse_view = _decode_scratch_views(
                mid_out, mid_lse, num_tokens, num_heads, num_splits, d_v
            )
            sparse_mla_sm120_decode_dsv3_2(
                q,
                kv_cache,
                indices,
                mid_out_view,
                mid_lse_view,
                output,
                out_lse,
                sm_scale,
                topk_length=topk_length,
                attn_sink=attn_sink,
                model_type=model_type,
                chunks_per_block=planned.cpb,
            )
            return

        module.sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            sm_scale,
            model_type,
            int(planned.variant),
            topk_length,
            attn_sink,
            extra_kv_cache,
            extra_indices,
            extra_topk_length,
            extra_fp4,
        )

    @register_fake_op("flashinfer::sparse_mla_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(
        paged_attention=_paged_attention,
        sparse_mla_sm120_dsv41_fp4_quantize_pack=module.sparse_mla_sm120_dsv41_fp4_quantize_pack,
        sparse_mla_sm120_dsv41_fp4_quantize_append=module.sparse_mla_sm120_dsv41_fp4_quantize_append,
    )


@supported_compute_capability([120, 121])
def _sparse_mla_sm120_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    d_v: int = _D_V,
    kv_scale_format: str = "auto",
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
    prefill_impl: Optional[str] = None,
    extra_fp4: bool = False,
) -> None:
    r"""Internal Sparse-MLA paged attention on SM120.

    Routes decode-form calls (``num_tokens <= 64``) to decode or prefill per
    the calibrated crossover policy, and larger calls to prefill. Mutates
    ``output`` and ``out_lse`` in place.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``[num_tokens, num_heads, d_qk]``, dtype bf16.
        ``d_qk=576`` uses the V32-family inline-scale cache,
        ``d_qk=512`` uses the DSv4 footer-scale cache, and ``d_qk=1088``
        uses the DOTS3_SWA sliding-window footer-scale cache (d_v=1024).
    kv_cache : torch.Tensor
        Byte-packed paged main KV cache. Accepted forms are 3D
        ``[num_blocks, page_block_size, bytes]``, HND
        ``[num_blocks, 1, page_block_size, bytes]``, or NHD
        ``[num_blocks, page_block_size, 1, bytes]``. The SM120 binding derives
        page size and block stride from the tensor metadata without
        materializing a layout conversion. Padded block strides are honored
        only for footer-scale models (DSv4 / DOTS3_SWA). Inline-scale
        (DSv3.2 / GLM) caches take the row advance as a runtime stride, so
        padded rows (a wider last dim, e.g. a legacy 656B pool serving the
        528B GLM53_NOPE payload) work in both decode and prefill as long as
        blocks pack rows contiguously. Cache origins and page strides must be
        16-byte aligned; inline row strides must also be aligned. Footer rows
        remain packed. Flat GLM53_NOPE pages use 528 bytes per token; expose
        the token axis in a 3D/4D view for a padded 656-byte pool.
    indices : torch.Tensor
        Paged slot IDs per query token, shape ``[num_tokens, topk]`` or
        ``[num_tokens, 1, topk]``, dtype int32. ``-1`` marks invalid /
        out-of-window slots; masked slots gather a dedicated zero row, so
        arbitrary cache contents (including NaNs) cannot contaminate valid
        outputs, and slots past ``topk_length`` are never gathered regardless
        of the padding contents. Prefill-routed calls require
        ``topk % 64 == 0`` (whole 64-wide index tiles) and, for DOTS3_SWA,
        ``topk >= 513`` so the sliding window fits the buffer.
    output : torch.Tensor
        In-place output, shape ``[num_tokens, num_heads, d_v]``, dtype bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, shape ``[num_tokens, num_heads]``, dtype float32.
    sm_scale : float
        Softmax scale (typically ``1 / sqrt(d_qk)``).
    d_v : int
        Value head dim. ``512`` for DSV3_2 / DSV4 / GLM variants, ``1024``
        for DOTS3_SWA.
    kv_scale_format : str
        Scale semantics, disambiguating the families that share a query
        width. ``"auto"`` and ``"pow2_fp32"`` select DSv3.2 power-of-2 FP32
        inline scales at ``d_qk=576``; ``"arbitrary_fp32"`` selects
        GLM-style arbitrary FP32 inline scales (GLM_NSA at ``d_qk=576``,
        GLM53_NOPE at ``d_qk=512``); ``"auto"`` at ``d_qk=512`` selects
        DSV4; ``"ue8m0_g32"`` at ``d_qk=512`` selects DSV4_1 (DeepSeek-V4.1:
        32-wide UE8M0 groups over the all-FP8 512-wide K).
    topk_length : Optional[torch.Tensor]
        Effective top-k length per query token, shape ``[num_tokens]``, dtype
        int32. Required for sliding-window MLA near sequence start; ``None``
        for uniform top-k.
    attn_sink : Optional[torch.Tensor]
        Per-head learnable bias added pre-softmax, shape ``[num_heads]``,
        dtype float32. FlashMLA V4 convention: ``output *= sigmoid(lse -
        sink)`` and ``lse' = log(exp(lse) + exp(sink))``.
    extra_kv_cache : Optional[torch.Tensor]
        Optional secondary KV cache (DSv4 C4A / C128A layers). When provided,
        ``extra_indices`` must also be passed. Supports DSV4 and DSV4_1.
    extra_fp4 : bool
        Select the 288-byte V41_FP4 secondary cache with a DSV4_1 main cache.
        Decode and prefill upconvert it into FP8 shared-memory tiles.
    extra_indices : Optional[torch.Tensor]
        Paged slot IDs for the secondary cache, shape
        ``[num_tokens, extra_topk]`` or ``[num_tokens, 1, extra_topk]``,
        dtype int32.
    extra_topk_length : Optional[torch.Tensor]
        Effective top-k length per query token for the secondary cache,
        shape ``[num_tokens]``, dtype int32.
    mid_out : Optional[torch.Tensor]
        Pre-allocated split-K partial-output scratch, shape
        ``[>=num_tokens, >=num_heads, >=num_splits, >=d_v]``, dtype bf16. Only
        consumed by the decode path; required when the call dispatches to a
        decode kernel. The head dimension must match the kernel's scratch
        stride: exactly 8 for ``num_heads == 8``, otherwise padded up to the
        nearest multiple of 16 (see ``_decode_scratch_heads``).
    mid_lse : Optional[torch.Tensor]
        Pre-allocated split-K LSE scratch, shape
        ``[>=num_tokens, >=num_heads, >=num_splits]``, dtype float32. Pair with
        ``mid_out`` when the call dispatches to a decode kernel; the head
        dimension follows the same rule as ``mid_out``.
    prefill_impl : Optional[str]
        Prefill-kernel override for calls that dispatch to prefill. ``None``
        or ``"auto"`` keeps the default order (swapAB preferred where
        instantiated); ``"swapab"`` forces the warp-specialized swapAB kernel
        and raises ``ValueError`` unless the shape is swapAB-eligible (DSV3_2
        family, single cache, whole-tile ``topk``, ``num_heads`` in
        {64, 128}); ``"mg"`` forces the non-swapAB SG/MG path. For the DSV4
        family ``"mg"`` and ``None`` are no-ops on dispatch, and ``"swapab"``
        always raises.

    Notes
    -----
    Requires SM120a / SM121a (block-scaled MXFP8 MMA + cp.async.bulk TMA).
    """
    model_type = _resolve_model_type(q.shape[-1], kv_scale_format)
    _require_d_v(d_v, model_type)
    _check_last_dim(output, "output", model_type)
    # The secondary cache is an all-or-nothing argument group: without this
    # check, extra_indices without extra_kv_cache would reach the planner as
    # has_extra=False with extra_topk>0, and could be forwarded to a
    # single-cache variant alongside the null cache.
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_kv_cache is None and extra_topk_length is not None:
        raise ValueError("extra_topk_length requires extra_kv_cache and extra_indices")
    if extra_fp4 and model_type != _MODEL_TYPE_DSV4_1:
        raise ValueError(
            "extra_fp4 (V41_FP4 extra cache) requires a DSV4_1 main cache "
            '(kv_scale_format="ue8m0_g32")'
        )

    impl = get_sparse_mla_sm120_module()
    impl.paged_attention(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v,
        model_type,
        _normalize_prefill_impl(prefill_impl),
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        mid_out,
        mid_lse,
        extra_fp4,
    )


class _SparseMLAPagedAttentionRunner:
    """Sparse-MLA paged attention implementation runner for SM120.

    ``max_num_tokens`` and ``max_num_heads`` are optional validation bounds.
    Eager calls prepare exact-shape execution plans and required scratch/LSE.
    Matching hot calls reuse them. Decode scratch may instead be supplied via
    ``run(mid_out=..., mid_lse=...)`` as contiguous buffers with sufficient
    plan-defined byte capacity. Warm every captured metadata shape first.
    Eager tuning may replace a shape's plan and resources when the profile
    changes; recapture graphs after such a change. The wrapper does not own
    graph objects or maintain historical profile resources.

    Parameters
    ----------
    max_num_tokens : Optional[int]
        Optional worst-case ``num_tokens`` the wrapper will accept. Must be
        provided together with ``max_num_heads``; buffers follow resolved shapes.
    max_num_heads : Optional[int]
        Optional worst-case ``num_heads``.
    d_v : int
        Value head dim. ``512`` for DSV3_2 / DSV4 / GLM variants, ``1024``
        for DOTS3_SWA. Must agree with the model type ``d_qk`` selects on each
        ``run``.
    kv_scale_format : str
        Scale semantics, disambiguating the families that share a query
        width. ``"auto"`` and ``"pow2_fp32"`` select DSv3.2 power-of-2 FP32
        inline scales at ``d_qk=576``; ``"arbitrary_fp32"`` selects
        GLM-style arbitrary FP32 inline scales (GLM_NSA at ``d_qk=576``,
        GLM53_NOPE at ``d_qk=512``); ``"auto"`` at ``d_qk=512`` selects
        DSV4; ``"ue8m0_g32"`` at ``d_qk=512`` selects DSV4_1 (DeepSeek-V4.1:
        32-wide UE8M0 groups over the all-FP8 512-wide K).
    kv_cache_format : {"fp8", "nvfp4"}
        Packed cache format. Both formats reuse this wrapper and its ``run``
        signature; each format keeps its own planner and internal kernels.
    extra_kv_fp4 : bool
        When True, the extra (compressed) cache of a dual-cache call holds
        288-byte V41_FP4 rows. The numerical route decodes them on chip.
        Requires ``kv_cache_format="fp8"`` and ``kv_scale_format="ue8m0_g32"``.
    compute_precision : {"default", "fp8", "bf16", "nvfp4"}
        Fixed for this Wrapper. Default retains the legacy family/shape hybrid
        and DSV4 NVFP4 cache routes. Explicit DSV4.1 FP8 uses decode where eligible
        (T<=64, H=1..128), otherwise the existing FP8 prefill (H=8/16/32/64,
        main topk multiple of 64). Without a matching DSV4.1 profile the
        decode-first policy uses CPB=1, isolated from legacy calibration. Exact
        profiles select same-precision phase/CPB by the next larger token bucket.
        FP8 prefill indices must be contiguous;
        independent page sizes and pitched cache/LSE buffers are supported.
        Full BF16 uses decode at T<=64 and direct prefill at larger T, retaining
        runtime H=1..128, positive runtime topk and pitched index rows. Both QK
        and PV use BF16 operands with FP32 accumulation/softmax, never the legacy
        BF16-QK/FP8-PV mode. Warm up each
        captured shape; live graph buffers are retained across shape changes.
        NVFP4 requires DSV4 384-byte storage, not DSV4.1: H=16/32/64/128,
        main topk=128/512 and PBS=64; extra topk>0 with PBS=2/64. It retains
        NVFP4 Q/P quantization and V requantization, BF16 RoPE, and the separately
        calibrated non-monotonic decode/streaming selection. CPB=0 keeps its
        existing heuristic. DSV4 NVFP4 graphs reuse warmed plans and exact-shape
        buffers without tuning or allocating during capture; LSE is contiguous.
    device : Optional[torch.device]
        Allocation target. Defaults to the current CUDA device.

    Example
    -------
    >>> runner = flashinfer.mla.SparseMLASm120Wrapper()
    >>> runner.run(q, kv_cache, indices, output, sm_scale=...)
    """

    @supported_compute_capability([120, 121])
    def __init__(
        self,
        max_num_tokens: Optional[int] = None,
        max_num_heads: Optional[int] = None,
        *,
        d_v: int = _D_V,
        kv_scale_format: str = "auto",
        kv_cache_format: str = "fp8",
        extra_kv_fp4: bool = False,
        compute_precision: str = "default",
        device: Optional[torch.device] = None,
    ) -> None:
        if (max_num_tokens is None) != (max_num_heads is None):
            raise ValueError(
                "max_num_tokens and max_num_heads must be provided together"
            )
        if max_num_tokens is not None and max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be > 0, got {max_num_tokens}")
        if max_num_heads is not None and (max_num_heads <= 0 or max_num_heads > 128):
            raise ValueError(f"max_num_heads must be in (0, 128], got {max_num_heads}")
        _require_supported_d_v(d_v)
        self._kv_scale_format = _normalize_kv_scale_format(kv_scale_format)
        if compute_precision not in ("default", "fp8", "bf16", "nvfp4"):
            raise ValueError("compute_precision must be default, fp8, bf16, or nvfp4")
        if compute_precision in ("fp8", "bf16") and (
            kv_cache_format != "fp8"
            or self._kv_scale_format != "ue8m0_g32"
            or d_v != 512
        ):
            raise ValueError(
                "explicit FP8/BF16 compute_precision requires DSV4.1 FP8 storage (ue8m0_g32, d_v=512)"
            )
        if compute_precision == "nvfp4" and kv_cache_format != "nvfp4":
            raise ValueError("compute_precision='nvfp4' requires DSV4 NVFP4 storage")
        self._compute_precision = compute_precision
        self._prepared_calls: dict = {}
        if kv_cache_format not in _KV_CACHE_FORMATS:
            raise ValueError(
                "kv_cache_format must be either 'fp8' or 'nvfp4', got "
                f"{kv_cache_format!r}"
            )
        if kv_cache_format == "nvfp4":
            if d_v != 512:
                raise ValueError("NVFP4 sparse MLA requires d_v=512")
            if self._kv_scale_format != "auto":
                raise ValueError(
                    "kv_scale_format applies to FP8 caches and must remain 'auto' "
                    "when kv_cache_format='nvfp4'"
                )
        self._kv_cache_format = kv_cache_format
        if extra_kv_fp4 and (
            kv_cache_format != "fp8" or self._kv_scale_format != "ue8m0_g32"
        ):
            raise ValueError(
                "extra_kv_fp4 (V41_FP4 extra cache) requires kv_cache_format='fp8' "
                "and kv_scale_format='ue8m0_g32' (a DSV4_1 main cache)"
            )
        self._extra_kv_fp4 = extra_kv_fp4

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            # Allocated tensors always carry a device index; pin it so
            # caller-passed buffers compare equal on the same device.
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._max_num_tokens = max_num_tokens
        self._max_num_heads = max_num_heads
        self._d_v = d_v

    # The runner owns out_lse internally so no separate template is needed.
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
        *,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        extra_kv_cache: Optional[torch.Tensor] = None,
        extra_indices: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        out_lse: Optional[torch.Tensor] = None,
        mid_out: Optional[torch.Tensor] = None,
        mid_lse: Optional[torch.Tensor] = None,
        prefill_impl: Optional[str] = None,
        return_lse: bool = False,
    ) -> Optional[torch.Tensor]:
        """Run sparse-MLA paged attention.

        Mutates ``output`` and an LSE buffer in place. When ``out_lse`` is
        passed, that buffer is used; otherwise the wrapper uses an internal
        lazily-sized buffer. When ``return_lse=True``, returns a view into the
        LSE buffer sized to the actual ``num_tokens``; otherwise returns
        ``None``.

        Accepts ``q``/``output`` either as 3-D ``[num_tokens, num_heads, head_dim]``
        or as 4-D ``[num_tokens, 1, num_heads, head_dim]`` (some callers carry
        a singleton s_q dim); the 4-D form is squeezed in place. Calls that
        dispatch to a decode kernel consume split-K scratch: caller-supplied
        via ``mid_out``/``mid_lse`` when given, otherwise buffers cached on
        the runner.

        ``prefill_impl`` (``None``/``"auto"``/``"swapab"``/``"mg"``) overrides
        the prefill-kernel selection for calls that dispatch to prefill;
        ``"swapab"`` raises ``ValueError`` on shapes outside its envelope
        (DSV3_2 family, single cache, whole-tile ``topk``, ``num_heads`` in
        {64, 128}) and is a no-op distinction for DSV4, where only the
        non-swapAB path exists. NVFP4 accepts ``None``, ``"auto"``, or
        ``"mg"`` and uses its separately calibrated streaming/split-K planner.
        NVFP4 also accepts ``indices``/``extra_indices`` as either
        ``[num_tokens, topk]`` or ``[num_tokens, 1, topk]``; the singleton
        query axis is normalized before planning and launch.
        """
        from ._sparse_mla_sm120_prepared import wrapper_run

        return wrapper_run(
            self,
            q,
            kv_cache,
            indices,
            output,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            out_lse=out_lse,
            mid_out=mid_out,
            mid_lse=mid_lse,
            prefill_impl=prefill_impl,
            return_lse=return_lse,
        )


# Public alias of the runner, exported as ``flashinfer.mla.SparseMLASm120Wrapper``.
#
# Public contract:
# - Construct once and hold persistently (e.g. per framework attention layer).
#   Warm every metadata shape eagerly before capture, including caller-scratch
#   calls. Matching hot calls reuse the prepared descriptor and owned resources.
#   A changed calibration profile may replace resources during eager warmup;
#   recapture affected graphs after that update.
# - ``run()`` is the single entry point. There is no separate plan stage;
#   dispatch decisions are made internally per call and memoized.
# - ``d_v``, ``kv_scale_format``, and ``kv_cache_format`` are fixed at
#   construction and select the
#   model-type semantics applied to every ``run()`` call (512 for
#   DSV3_2 / DSV4 / GLM variants, 1024 for DOTS3_SWA; scale semantics per
#   ``kv_scale_format``).
SparseMLASm120Wrapper = _SparseMLAPagedAttentionRunner


# Decode-DSv3.2 / DSv4: chunks_per_block (cpb) comes from the calibrated
# analytical model in _sparse_mla_sm120_calibration. Constants are calibrated once per
# (device, family) during autotune() tuning mode and cached on disk; without
# them the launcher's built-in heuristic (cpb_override=-1) is used.


def _decode_dsv4_num_splits(
    topk: int, extra_topk: int = 0, model_type: int = _MODEL_TYPE_DSV4
) -> int:
    """Split-K partitions: one per candidate-tile-wide chunk of each index set.

    The tile width is model-dependent: the DeepSeek family consumes ``_BI``=64
    candidates per iteration, DOTS3_SWA 32 (its 1040-byte KV smem stride does
    not fit BI=64 on SM120). Deriving ``num_splits`` with the wrong width makes
    the launch grid cover only part of each token's candidate list and silently
    drop the tail.
    """
    bi = _decode_chunk_width(model_type)
    return (topk + bi - 1) // bi + (extra_topk + bi - 1) // bi


@supported_compute_capability([120, 121])
def sparse_mla_sm120_decode_dsv3_2(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    model_type: int = _MODEL_TYPE_DSV3_2,
    chunks_per_block: Optional[int] = None,
) -> torch.Tensor:
    """Sparse-MLA paged decode (DSv3.2 / GLM-NSA kernel) on SM120.

    Cache pages must contain 64 tokens. Contiguous scratch uses head capacity
    8 for H=8, otherwise ceil(H/16)*16, and allocated (not active) split stride.

    ``chunks_per_block`` follows the same contract as the DSv4 decode helper:
    an explicit value is used directly; otherwise the calibrated analytical
    model picks one when its constants are available (calibrated once per
    device during ``autotune()`` tuning mode), falling back to the C++
    heuristic. DSv3.2 and GLM-NSA share the same calibrated constants;
    GLM53_NOPE has its own constants entry.
    """
    _check_last_dim(output, "output", int(model_type))
    _check_last_dim(mid_out, "mid_out", int(model_type))
    if q.shape[0] == 0:
        # Empty request: a kernel launch would hit a grid.x=0 CUDA error.
        return output

    module = _get_sparse_mla_sm120_decode_module()
    num_splits = _decode_dsv4_num_splits(indices.shape[-1], model_type=int(model_type))

    if chunks_per_block is not None:
        cpb_override = int(chunks_per_block)
    else:
        cpb_override = _resolve_cpb(
            q.device,
            _MODEL_TYPE_TO_FAMILY[int(model_type)],
            q.shape[0],
            q.shape[1],
            indices.shape[-1],
            0,
        )

    module.sparse_mla_sm120_decode_dsv3_2(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        num_splits,
        sm_scale,
        topk_length,
        attn_sink,
        int(model_type),
        cpb_override,
    )
    return output


@supported_compute_capability([120, 121])
def sparse_mla_sm120_decode_dsv4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    chunks_per_block: Optional[int] = None,
    model_type: Optional[int] = None,
    extra_fp4: bool = False,
) -> torch.Tensor:
    r"""Sparse-MLA paged decode (DSv4 standalone kernel) on SM120.

    The decode-dsv4 path is the split-K decode variant where each block handles
    ``chunks_per_block`` chunks of 64 candidates each. The wall-time-optimal
    value is shape-dependent; this wrapper picks it per call with the
    calibrated analytical model in :mod:`._sparse_mla_sm120_calibration`.

    Behaviour:

    - ``chunks_per_block`` explicitly given → use that value directly.
    - Otherwise, if calibrated model constants are available for this device
      (calibrated once per device in ``autotune()`` tuning mode and cached on
      disk) → use the model's choice.
    - Otherwise → fall back to the C++ closed-form heuristic.

    Parameters
    ----------
    q : torch.Tensor
        ``[T, num_heads, d_qk]`` bf16. ``d_qk == 512`` (DSV4) or
        ``d_qk == 1088`` (DOTS3_SWA; d_v is then 1024).
    kv_cache : torch.Tensor
        Paged FP8 cache, shape ``[num_blocks, page_bytes]`` uint8.
    indices : torch.Tensor
        ``[T, topk]`` int32. Any ``topk >= 1`` for DSV4 (any ``topk >= 513``
        for DOTS3_SWA: the 513-token sliding-window floor; tiled 32-wide, so
        ``num_splits`` uses the 32-candidate chunk width); ``-1`` marks
        invalid slots. Row-strided views into a wider persistent buffer
        are accepted (the last dim must stay contiguous).
    mid_out : torch.Tensor
        Contiguous scratch, ``[T, scratch_heads, num_splits, d_v]`` bf16.
        ``scratch_heads`` is 8 for H=8, otherwise ``ceil(H/16)*16``.
        ``num_splits = ceil(topk / 64) + ceil(extra_topk / 64)`` (DOTS3_SWA
        tiles 32). The stride uses allocated splits, not CPB-reduced active splits.
    mid_lse : torch.Tensor
        Contiguous scratch, ``[T, scratch_heads, num_splits]`` float32.
    output : torch.Tensor
        In-place output, ``[T, num_heads, d_v]`` bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, ``[T, num_heads]`` float32.
    sm_scale : float
        Softmax scale.
    topk_length : Optional[torch.Tensor]
        Per-token effective top-k length, ``[T]`` int32.
    attn_sink : Optional[torch.Tensor]
        Per-head learnable bias added pre-softmax, shape ``[num_heads]``,
        dtype float32. FlashMLA V4 convention: ``output *= sigmoid(lse -
        sink)`` and ``lse' = log(exp(lse) + exp(sink))``.
    extra_kv_cache : Optional[torch.Tensor]
        Optional secondary KV cache (DSv4 C4A / C128A layers). When provided,
        ``extra_indices`` must also be passed.
    extra_indices : Optional[torch.Tensor]
        Paged slot IDs for the secondary cache, shape ``[T, extra_topk]``
        int32.
    extra_topk_length : Optional[torch.Tensor]
        Per-token effective top-k length for the secondary cache, ``[T]``
        int32.
    chunks_per_block : Optional[int]
        Explicit override. If ``None``, the calibrated model picks a value when
        available, else the C++ heuristic is used.
    extra_fp4 : bool
        The extra cache holds V41_FP4 rows (288 B/token: 256 B packed E2M1 +
        32 B E4M3-G16 scales; FlashMLA ``tests/quant.py``) instead of the main
        cache format. Requires a DSV4_1 main cache; the gather path upconverts
        FP4 rows to the canonical FP8 smem row. Reuses the DSV4_1 calibration
        family (the cpb model is chunk-count based).

    Returns
    -------
    output : torch.Tensor
        The mutated output tensor (for chaining).
    """
    # model_type selects the footer-scale model explicitly; None keeps the
    # legacy width inference: 512 -> DSV4 (d_v 512), 1088 -> DOTS3_SWA (d_v
    # 1024). DSV4_1 shares d_qk=512 with DSV4 and is only reachable explicitly
    # (e.g. _MODEL_TYPE_DSV4_1 from the planner, keyed by
    # kv_scale_format="ue8m0_g32" upstream).
    if model_type is None:
        model_type = _MODEL_TYPE_DOTS3_SWA if q.shape[-1] == 1088 else _MODEL_TYPE_DSV4
    model_type = int(model_type)
    _check_last_dim(output, "output", model_type)
    _check_last_dim(mid_out, "mid_out", model_type)
    if q.shape[0] == 0:
        # Empty request: a kernel launch would hit a grid.x=0 CUDA error.
        return output

    module = _get_sparse_mla_sm120_decode_module()
    topk = indices.shape[-1]  # 2D [T, topk] or 3D [T, 1, topk]
    extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
    num_splits = _decode_dsv4_num_splits(topk, extra_topk, model_type)

    if chunks_per_block is not None:
        cpb_override = int(chunks_per_block)
    elif model_type == _MODEL_TYPE_DSV4_1 and extra_kv_cache is not None:
        cpb_override = -1
    else:
        cpb_override = _resolve_cpb(
            q.device,
            _MODEL_TYPE_TO_FAMILY[model_type],
            q.shape[0],
            q.shape[1],
            topk,
            extra_topk,
        )

    module.sparse_mla_sm120_decode_dsv4(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        num_splits,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        model_type,
        cpb_override,
        extra_fp4,
    )
    return output


@supported_compute_capability([120, 121])
@flashinfer_api
def dsv41_fp4_quantize_pack_sparse_mla_cache(
    latent_kv: torch.Tensor,
    *,
    kv_layout: str = "HND",
) -> torch.Tensor:
    r"""Quantize complete DeepSeek-V4.1 latent-KV pages to the V41_FP4 ABI.

    Parameters
    ----------
    latent_kv : torch.Tensor
        Contiguous CUDA BF16/FP16 tensor with shape
        ``[num_pages, page_size, 512]``. A singleton latent-head axis is also
        accepted in HND or NHD position. All 512 values per token (RoPE dims
        included) are quantized in groups of 16 to packed E2M1 with E4M3
        scales (scale = amax/6), mirroring the FlashMLA V41_FP4 trajectory.
    kv_layout : str
        Output layout, either ``"HND"`` or ``"NHD"``.

    Returns
    -------
    torch.Tensor
        Opaque uint8 paged cache with logical shape
        ``[num_pages, 1, page_size, 288]`` for HND or
        ``[num_pages, page_size, 1, 288]`` for NHD. Within each physical page
        it stores ``page_size * 256`` data bytes followed by
        ``page_size * 32`` scale bytes. Consumers must not interpret the last
        dimension as a contiguous per-token record.
    """
    if kv_layout not in ("HND", "NHD"):
        raise ValueError(f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}")
    if latent_kv.ndim == 2:
        raise ValueError(
            "full-page pack requires a page dimension; use shape "
            "[num_pages, page_size, 512]"
        )
    if latent_kv.ndim == 3:
        num_pages, page_size = latent_kv.shape[:2]
    elif latent_kv.ndim == 4 and latent_kv.shape[1] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[2]
    elif latent_kv.ndim == 4 and latent_kv.shape[2] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[1]
    else:
        raise ValueError(
            "latent_kv must be [num_pages, page_size, 512] with an optional "
            "singleton latent-head axis (HND or NHD)"
        )
    from ._sparse_mla_sm120_execution import format_info

    bytes_per_token = format_info(_MODEL_TYPE_DSV4_1)["fp4_bytes_per_token"]
    cache_shape = (
        (num_pages, 1, page_size, bytes_per_token)
        if kv_layout == "HND"
        else (num_pages, page_size, 1, bytes_per_token)
    )
    cache = torch.empty(cache_shape, dtype=torch.uint8, device=latent_kv.device)
    if int(num_pages) == 0 or int(page_size) == 0:
        return cache
    get_sparse_mla_sm120_module().sparse_mla_sm120_dsv41_fp4_quantize_pack(
        latent_kv, cache
    )
    return cache


@supported_compute_capability([120, 121])
@flashinfer_api
def dsv41_fp4_quantize_append_sparse_mla_cache(
    latent_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    r"""Quantize and append DeepSeek-V4.1 latent KV by physical cache slot.

    Parameters
    ----------
    latent_kv : torch.Tensor
        Contiguous CUDA BF16/FP16 tensor with one 512-element latent-KV row
        per entry in ``slot_mapping``.
    slot_mapping : torch.Tensor
        Contiguous 1D CUDA int32 or int64 tensor. ``slot_mapping[i]`` is
        ``page_id * page_size + entry_id``. Negative and out-of-range slots
        are padding and are ignored. If a valid slot occurs more than once,
        the lowest-index input row is written deterministically.
    cache : torch.Tensor
        Destination opaque uint8 paged cache (2D ``[num_pages, page_bytes]``,
        3D ``[num_pages, page_size, 288]``, or the 4D HND/NHD forms).
        Page-strided views may place the cache inside vLLM's packed physical
        block allocation. Only addressed data rows and scale slots are
        written, so prefill history is reused directly by decode.
    """
    if not slot_mapping.is_cuda:
        raise ValueError(
            f"slot_mapping must be a CUDA tensor, got {slot_mapping.device}"
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "slot_mapping must have dtype torch.int32 or torch.int64, got "
            f"{slot_mapping.dtype}"
        )
    if slot_mapping.ndim != 1 or not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be a contiguous 1D tensor")
    if latent_kv.device != cache.device or slot_mapping.device != cache.device:
        raise ValueError(
            "latent_kv, slot_mapping, and cache must be on the same device"
        )
    get_sparse_mla_sm120_module().sparse_mla_sm120_dsv41_fp4_quantize_append(
        latent_kv, slot_mapping, cache
    )
