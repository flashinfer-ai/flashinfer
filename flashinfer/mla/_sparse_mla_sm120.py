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

"""Internal Sparse-MLA paged attention implementation for SM120.

Decode-form calls (``num_tokens <= 64``) route to dedicated warp-spec
standalone decode kernels when the shape is decode-instantiated, and to the
shared prefill orchestrator otherwise (prefill serves any ``num_tokens >=
1``). ``num_tokens > 64`` always routes to prefill. DSv3.2 / GLM-NSA
(d_qk=576), DSv4 (d_qk=512), GLM-5.3 native NoPE (d_qk=512,
arbitrary FP32 scales), and DOTS3_SWA (d_qk=1088, d_v=1024, sliding-window)
are supported; prefill dispatches through the orchestrator.

When crossover constants are calibrated (same ``autotune()`` tuning-mode
pass as the cpb constants; see :mod:`._sparse_mla_sm120_cpb`), a
decode-instantiated decode-form call routes to prefill once
``num_tokens`` exceeds the measured ``decode_max_tokens`` for its
``(model_type, num_heads, topk)`` config; without calibration the historical
decode-first policy is unchanged.

The user-facing sparse MLA entry points are
``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`` for DeepSeek V4 and
``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=...)``
for the DSv3.2 / GLM sparse top-k path. This module keeps only the SM120
implementation hooks used by those dispatchers and focused kernel
tests/benchmarks.

Decode kernels serve, per family, any ``num_heads`` in [1, 128] and any
``topk >= min_topk`` (513 for the sliding-window family, 1 elsewhere);
``flashinfer.mla.supported_sparse_mla_sm120_configs`` describes each
family's envelope so callers can validate a configuration at init time.
Decode-eligible is not required: shapes outside the decode envelope are
served by prefill.

Prefill kernels likewise take ``topk`` (the indices row width) as a runtime
argument: any ``topk >= 1`` with ``topk % 64 == 0`` is served (whole 64-wide
index tiles; the tail tile is not masked), with ``topk >= 513`` for
DOTS3_SWA so its sliding window fits the buffer. The binding rejects ragged
widths loudly.

The decode launch parameter ``chunks_per_block`` is picked per call by the
calibrated analytical model in :mod:`._sparse_mla_sm120_cpb` when constants
are available (calibrated once per device during ``autotune()`` tuning mode,
cached on disk); otherwise the launcher's built-in heuristic is used.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.mla import gen_sparse_mla_sm120_module
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

# The _DECODE_*_DISPATCH pair sets are re-exported here on purpose: vLLM's
# has_flashinfer_sparse_mla_sm120_config probes membership of
# ``flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH`` directly.
from ._sparse_mla_sm120_plan import (
    _BI,
    _BPT_DSV3_2,
    _BPT_DSV4,
    _BPT_DOTS3_SWA,
    _DECODE_DSV3_2_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_DSV4_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_MAX_HEADS,
    _DECODE_MAX_TOKENS,
    _DECODE_DSV3_2_TOPKS,
    _DECODE_DSV4_TOPKS,
    _DECODE_DOTS3_SWA_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_DOTS3_SWA_TOPK,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_GLM53_NOPE,
    _MODEL_TYPE_GLM_NSA,
    _MODEL_TYPE_DOTS3_SWA,
    _DECODE_GLM53_NOPE_DISPATCH,  # noqa: F401  (vLLM probe surface)
    _DECODE_GLM53_NOPE_TOPK,
    _D_V_BY_MODEL_TYPE,
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
from ._sparse_mla_sm120_cpb import (  # noqa: E402
    SparseMLASm120CalibrationReport,  # noqa: F401  (lazy re-export)
    calibrate_sparse_mla_sm120,  # noqa: F401  (lazy re-export)
)

logger = logging.getLogger(__name__)

_KV_SCALE_FORMATS = frozenset({"auto", "pow2_fp32", "arbitrary_fp32"})

# Page block size the decode kernels are instantiated for (same constant for
# both families; every instantiated kernel is pbs=64).
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64
_DECODE_DSV3_2_PAGE_BLOCK_SIZE = 64


@dataclass(frozen=True)
class SparseMLASm120DecodeConfig:
    """Instantiated decode-kernel set for one SM120 sparse-MLA kernel family.

    Decode-form calls (``num_tokens <= max_num_tokens``) prefer a standalone
    decode kernel when their shape matches one of the instantiations
    described here; decode-eligible is not required, since the prefill
    orchestrator serves any remaining decode-form shape at
    ``num_tokens >= 1`` (and crossover calibration may route even eligible
    shapes to prefill past a measured ``num_tokens`` threshold). Larger calls
    go through the prefill orchestrator, which has its own separately
    instantiated shape envelope; this config describes decode only.

    Attributes
    ----------
    d_qk : int
        Query/key head dim served by this family (``512`` for DSv4 /
        GLM53_NOPE, ``576`` for DSv3.2 / GLM-NSA, ``1088`` for the
        DOTS3_SWA sliding-window family, whose ``d_v`` is then 1024).
    page_block_size : int
        The only KV page block size the decode kernels are instantiated for.
    max_num_tokens : int
        Largest ``num_tokens`` routed to the decode kernels (inclusive).
    topks : frozenset[int]
        The calibrated top-k values (the crossover sweep points). Decode
        serves ANY ``topk >= min_topk`` — topk is a runtime kernel argument —
        so this set is documentation of what has measured crossover data,
        not the eligibility boundary.
    min_topk : int
        Smallest legal ``topk`` (the indices-row width). ``513`` for the
        sliding-window family (the window must fit the buffer); ``1``
        elsewhere.
    max_num_heads : int
        Every ``num_heads`` in ``[1, max_num_heads]`` is served: dedicated
        instantiations at ``{8, 16, 32, 64, 128}`` plus one
        runtime-head-count instantiation covering any other count.
    """

    d_qk: int
    page_block_size: int
    max_num_tokens: int
    topks: frozenset[int]
    min_topk: int
    max_num_heads: int

    def supported_num_heads(self) -> tuple[int, ...]:
        """Every head count from 1 through ``max_num_heads`` (runtime-H)."""
        return tuple(range(1, self.max_num_heads + 1))

    def supported_topk(self, num_heads: Optional[int] = None) -> tuple[int, ...]:
        """Sorted calibrated top-k values for ``num_heads`` (or any head count).

        Decode serves any ``topk >= min_topk``; these are the values with
        measured crossover data."""
        if num_heads is None or 1 <= num_heads <= self.max_num_heads:
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

        Decode-instantiated shapes may still route to prefill past the
        calibrated crossover, and non-instantiated decode-form shapes are
        served by prefill; this predicate describes the decode envelope only.
        """
        if page_block_size is None:
            page_block_size = self.page_block_size
        return (
            num_tokens <= self.max_num_tokens
            and page_block_size == self.page_block_size
            and 1 <= num_heads <= self.max_num_heads
            and topk >= self.min_topk
        )


@flashinfer_api
def supported_sparse_mla_sm120_configs() -> dict[str, SparseMLASm120DecodeConfig]:
    """Enumerate the instantiated SM120 sparse-MLA decode kernel configurations.

    Lets callers validate a serving configuration at initialization time
    instead of discovering an uninstantiated ``(num_heads, topk)`` pair on the
    first decode-form request.

    Returns
    -------
    dict[str, SparseMLASm120DecodeConfig]
        Mapping from kernel family to its instantiated decode set, keyed by
        ``"dsv4"`` (``d_qk=512``), ``"dsv3_2"`` (``d_qk=576``, power-of-2
        FP32 scales), ``"glm_nsa"`` (``d_qk=576``, arbitrary FP32 scales;
        shares the DSv3.2 decode instantiations), ``"glm53_nope"``
        (GLM-5.3 native NoPE, ``d_qk=512``, arbitrary FP32 scales), and
        ``"dots3_swa"`` (sliding-window family, ``d_qk=1088``, UE8M0 scales).

    Examples
    --------
    >>> import flashinfer
    >>> configs = flashinfer.mla.supported_sparse_mla_sm120_configs()
    >>> configs["dsv4"].supports_decode(num_heads=64, topk=256)
    True
    """
    dsv3_2 = SparseMLASm120DecodeConfig(
        d_qk=576,
        page_block_size=_DECODE_DSV3_2_PAGE_BLOCK_SIZE,
        max_num_tokens=_DECODE_MAX_TOKENS,
        topks=_DECODE_DSV3_2_TOPKS,
        min_topk=1,
        max_num_heads=_DECODE_MAX_HEADS,
    )
    return {
        "dsv4": SparseMLASm120DecodeConfig(
            d_qk=512,
            page_block_size=_DECODE_DSV4_PAGE_BLOCK_SIZE,
            max_num_tokens=_DECODE_MAX_TOKENS,
            topks=_DECODE_DSV4_TOPKS,
            min_topk=1,
            max_num_heads=_DECODE_MAX_HEADS,
        ),
        "dsv3_2": dsv3_2,
        "glm_nsa": dsv3_2,
        "glm53_nope": SparseMLASm120DecodeConfig(
            d_qk=512,
            page_block_size=_DECODE_DSV3_2_PAGE_BLOCK_SIZE,
            max_num_tokens=_DECODE_MAX_TOKENS,
            topks=frozenset({_DECODE_GLM53_NOPE_TOPK}),
            min_topk=1,
            max_num_heads=_DECODE_MAX_HEADS,
        ),
        "dots3_swa": SparseMLASm120DecodeConfig(
            d_qk=1088,
            page_block_size=_DECODE_DSV4_PAGE_BLOCK_SIZE,
            max_num_tokens=_DECODE_MAX_TOKENS,
            topks=frozenset({_DECODE_DOTS3_SWA_TOPK}),
            min_topk=513,
            max_num_heads=_DECODE_MAX_HEADS,
        ),
    }


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
    if page_block_size != config.page_block_size:
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
    """Check ``d_v`` against every supported model type.

    Used where the model type is not yet known -- the runner is constructed
    before it sees a ``q`` to read ``d_qk`` from. Each call still goes through
    the strict :func:`_require_d_v` once ``d_qk`` has resolved the model type.
    """
    supported = sorted(set(_D_V_BY_MODEL_TYPE.values()))
    if int(d_v) not in supported:
        raise ValueError(f"SM120 sparse-MLA requires d_v in {supported}, got {d_v}")


def _check_last_dim(
    tensor: torch.Tensor, name: str, model_type: Optional[int] = None
) -> None:
    expected = _expected_d_v(model_type)
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} last dimension must be {expected}, got shape {tuple(tensor.shape)}"
        )


def _normalize_kv_scale_format(kv_scale_format: str) -> str:
    fmt = str(kv_scale_format).lower().replace("-", "_")
    if fmt not in _KV_SCALE_FORMATS:
        raise ValueError(
            "kv_scale_format must be one of "
            f"{sorted(_KV_SCALE_FORMATS)}, got {kv_scale_format!r}"
        )
    return fmt


def _resolve_model_type(d_qk: int, kv_scale_format: str) -> int:
    fmt = _normalize_kv_scale_format(kv_scale_format)
    if d_qk == 576:
        if fmt == "arbitrary_fp32":
            return _MODEL_TYPE_GLM_NSA
        return _MODEL_TYPE_DSV3_2
    if d_qk == 512:
        # GLM-5.3 native NoPE (512+0) shares the DSv4 query width; the scale
        # format disambiguates.
        if fmt == "arbitrary_fp32":
            return _MODEL_TYPE_GLM53_NOPE
        if fmt != "auto":
            raise ValueError(
                "kv_scale_format for d_qk=512 must be 'auto' (DSV4) or "
                f"'arbitrary_fp32' (GLM53_NOPE); got {kv_scale_format!r}"
            )
        return _MODEL_TYPE_DSV4
    if d_qk == 1088:
        # Sliding-window MLA family: 1024-wide latent + 64-wide rope.
        if fmt != "auto":
            raise ValueError(
                "kv_scale_format is only configurable for d_qk=576 and "
                f"d_qk=512; got d_qk=1088 with kv_scale_format={kv_scale_format!r}"
            )
        return _MODEL_TYPE_DOTS3_SWA
    raise ValueError(
        f"SM120 sparse-MLA supports d_qk=576, 512 or 1088, got d_qk={d_qk}"
    )


def _bytes_per_token_for_model_type(model_type: int) -> int:
    if model_type in (_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA, _MODEL_TYPE_GLM53_NOPE):
        return _BPT_DSV3_2
    if model_type == _MODEL_TYPE_DSV4:
        return _BPT_DSV4
    if model_type == _MODEL_TYPE_DOTS3_SWA:
        return _BPT_DOTS3_SWA
    raise ValueError(f"Unsupported SM120 sparse-MLA model_type={model_type}")


def _packed_kv_page_block_size(
    kv_cache: torch.Tensor,
    *,
    model_type: int,
    name: str,
) -> int:
    bytes_per_token = _bytes_per_token_for_model_type(model_type)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve caller-supplied scratch buffers for split-K decode kernels.

    The scratch head dim is the true ``num_heads`` for the dedicated
    ``num_heads=8`` instantiation and HPB(16)-aligned otherwise (the runtime-H
    kernel writes both halves of its head tile unconditionally).
    """
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
    return (
        mid_out[:num_tokens, :scratch_heads, :num_splits, :d_v],
        mid_lse[:num_tokens, :scratch_heads, :num_splits],
    )


@functools.cache
def get_sparse_mla_sm120_module():
    """Build and cache the sparse-MLA SM120 module + bound custom op."""
    module = gen_sparse_mla_sm120_module().build_and_load()

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
        if (
            model_type
            in (
                _MODEL_TYPE_DSV3_2,
                _MODEL_TYPE_GLM_NSA,
                _MODEL_TYPE_GLM53_NOPE,
            )
            and not kv_cache.is_contiguous()
        ):
            # Inline-scale prefill kernels address the cache as a flat token
            # array, so a padded block stride would be silently misread — and
            # crossover can route any decode-form call there, so the
            # restriction cannot wait for a prefill-routed call to fire.
            # Contiguous padded-row caches (wider last dim) stay allowed:
            # decode-v32 honors their row stride, and a prefill-routed call
            # rejects them loudly at the binding.
            raise ValueError(
                "inline-scale (DSv3.2/GLM) KV caches must be contiguous "
                "through this entry: prefill-routed calls address the cache "
                "as a flat token array, and the calibrated crossover can "
                "route decode-form calls to prefill. Padded block strides are "
                "supported only by the standalone decode entry "
                "(sparse_mla_sm120_decode_dsv3_2)"
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
            if model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DOTS3_SWA):
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
        )

    @register_fake_op("flashinfer::sparse_mla_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(paged_attention=_paged_attention)


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
        only for footer-scale models (DSv4 / DOTS3_SWA); inline-scale
        (DSv3.2 / GLM) caches must be contiguous through this entry, since
        crossover can route any decode-form call to the flat-addressing
        prefill kernels. Contiguous caches with padded rows (a wider last
        dim) are served by the decode-v32 kernel and rejected loudly if a
        call routes to prefill.
    indices : torch.Tensor
        Paged slot IDs per query token, shape ``[num_tokens, topk]`` or
        ``[num_tokens, 1, topk]``, dtype int32. ``-1`` marks invalid /
        out-of-window slots (kernel skips). Prefill-routed calls require
        ``topk % 64 == 0`` (whole 64-wide index tiles; the tail tile is not
        masked) and, for DOTS3_SWA, ``topk >= 513`` so the sliding window
        fits the buffer.
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
        DSV4.
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
        ``extra_indices`` must also be passed. DSV4-only.
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
    )


class _SparseMLAPagedAttentionRunner:
    """Sparse-MLA paged attention implementation runner for SM120.

    ``max_num_tokens`` and ``max_num_heads`` are optional upper bounds. When
    both are provided, the wrapper pre-allocates its LSE buffer. Otherwise, the
    buffer is allocated lazily and grown as needed. Decode split-K scratch may be
    supplied by the caller via ``run(mid_out=..., mid_lse=...)``; if omitted for
    a call that dispatches to a decode kernel, the wrapper allocates the scratch
    and caches it on the instance, growing it on demand.

    Parameters
    ----------
    max_num_tokens : Optional[int]
        Optional worst-case ``num_tokens`` the wrapper will accept. Used to size
        the pre-allocated ``out_lse`` buffer when paired with
        ``max_num_heads``.
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
        DSV4.
    device : Optional[torch.device]
        Allocation target. Defaults to the current CUDA device.

    Example
    -------
    >>> runner = _SparseMLAPagedAttentionRunner()
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

        # Internally-owned decode split-K scratch, allocated on the first
        # decode-routed call and grown on demand. Held for the runner's
        # lifetime: scratch freed after run() returns its block to the
        # allocator, where a later CUDA graph capture can recycle it while an
        # older captured graph still replays into it.
        self._mid_out: Optional[torch.Tensor] = None
        self._mid_lse: Optional[torch.Tensor] = None

        self._out_lse: Optional[torch.Tensor] = None
        if max_num_tokens is not None and max_num_heads is not None:
            # Pre-allocated LSE buffer; sliced to actual shape on run(). Sized
            # for prefill worst case since prefill writes here too.
            self._out_lse = torch.empty(
                (max_num_tokens, max_num_heads),
                dtype=torch.float32,
                device=self._device,
            )

    def _get_out_lse(
        self,
        num_tokens: int,
        num_heads: int,
        out_lse: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if out_lse is not None:
            need = (num_tokens, num_heads)
            if out_lse.ndim != 2 or any(out_lse.size(d) < need[d] for d in range(2)):
                raise ValueError(
                    f"out_lse shape {tuple(out_lse.shape)} too small for required "
                    f"[num_tokens={num_tokens}, num_heads={num_heads}]"
                )
            if out_lse.dtype != torch.float32:
                raise ValueError(
                    f"out_lse must have dtype float32, got {out_lse.dtype}"
                )
            if out_lse.device != self._device:
                raise ValueError(
                    f"out_lse must be on device {self._device}, got {out_lse.device}"
                )
            return out_lse[:num_tokens, :num_heads]

        if self._out_lse is None or any(
            self._out_lse.size(d) < need
            for d, need in enumerate((num_tokens, num_heads))
        ):
            self._out_lse = torch.empty(
                (num_tokens, num_heads),
                dtype=torch.float32,
                device=self._device,
            )
        return self._out_lse[:num_tokens, :num_heads]

    def _maybe_allocate_decode_scratch(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        prefill_impl: Optional[str],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (mid_out is None) != (mid_lse is None):
            raise ValueError("mid_out and mid_lse must be passed together")
        if mid_out is not None:
            return mid_out, mid_lse

        num_tokens, num_heads, d_qk = q.shape
        if num_tokens == 0 or num_tokens > _DECODE_MAX_TOKENS:
            # The op no-ops on empty requests before planning, and prefill-form
            # calls never route to decode — skip the plan() lookup on both.
            return None, None
        model_type = _resolve_model_type(d_qk, self._kv_scale_format)
        topk = indices.shape[-1]
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        # Route with the same memoized plan() the op makes; only a
        # decode-routed call consumes split-K scratch. A dispatch miss
        # (None) falls through so the op reports it.
        planned = plan(
            num_tokens,
            num_heads,
            topk,
            model_type,
            _packed_kv_page_block_size(
                kv_cache, model_type=model_type, name="kv_cache"
            ),
            extra_kv_cache is not None,
            _normalize_prefill_impl(prefill_impl),
            q.device,
            extra_topk=extra_topk,
        )
        if planned is None or planned.variant is not KernelVariant.DECODE_SPLITK:
            return None, None

        num_splits = _decode_dsv4_num_splits(topk, extra_topk, model_type)
        need_out: tuple[int, ...] = (
            num_tokens,
            _decode_scratch_heads(num_heads),
            num_splits,
            self._d_v,
        )
        if (
            self._mid_out is None
            or self._mid_out.device != q.device
            or any(self._mid_out.size(d) < need_out[d] for d in range(4))
        ):
            if self._mid_out is not None and self._mid_out.device == q.device:
                # Grow-only per dim so smaller later calls keep the buffers.
                need_out = tuple(
                    max(self._mid_out.size(d), need_out[d]) for d in range(4)
                )
            self._mid_out = torch.empty(need_out, dtype=torch.bfloat16, device=q.device)
            self._mid_lse = torch.empty(
                need_out[:3], dtype=torch.float32, device=q.device
            )
        return self._mid_out, self._mid_lse

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
        non-swapAB path exists.
        """
        if q.dim() == 4:
            if q.size(1) != 1:
                raise ValueError(
                    f"4-D q is only supported with s_q=1, got q.shape={tuple(q.shape)}"
                )
            q = q.squeeze(1)
        if output.dim() == 4:
            if output.size(1) != 1:
                raise ValueError(
                    f"4-D output is only supported with s_q=1, got "
                    f"output.shape={tuple(output.shape)}"
                )
            output = output.squeeze(1)
        num_tokens, num_heads, _ = q.shape
        if self._max_num_tokens is not None and num_tokens > self._max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens "
                f"({self._max_num_tokens})"
            )
        if self._max_num_heads is not None and num_heads > self._max_num_heads:
            raise ValueError(
                f"num_heads ({num_heads}) exceeds max_num_heads ({self._max_num_heads})"
            )

        mid_out, mid_lse = self._maybe_allocate_decode_scratch(
            q,
            kv_cache,
            indices,
            extra_kv_cache,
            extra_indices,
            mid_out,
            mid_lse,
            prefill_impl,
        )

        out_lse_view = self._get_out_lse(num_tokens, num_heads, out_lse)
        _sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse_view,
            sm_scale,
            d_v=self._d_v,
            kv_scale_format=self._kv_scale_format,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            mid_out=mid_out,
            mid_lse=mid_lse,
            prefill_impl=prefill_impl,
        )
        return out_lse_view if return_lse else None


# Public alias of the runner, exported as ``flashinfer.mla.SparseMLASm120Wrapper``.
#
# Public contract:
# - Construct once and hold persistently (e.g. per framework attention layer).
#   The constructor pre-allocates the LSE buffer from the ``max_num_tokens`` /
#   ``max_num_heads`` upper bounds, so construction must complete before CUDA
#   graph capture. Internally-cached decode scratch is allocated on the first
#   decode-routed call of each new maximum shape, so warm up every captured
#   shape (or pass caller scratch) before capture; steady-state ``run()``
#   calls then allocate nothing.
# - ``run()`` is the single entry point. There is no separate plan stage;
#   dispatch decisions are made internally per call and memoized.
# - ``d_v`` and ``kv_scale_format`` are fixed at construction and select the
#   model-type semantics applied to every ``run()`` call (512 for
#   DSV3_2 / DSV4 / GLM variants, 1024 for DOTS3_SWA; scale semantics per
#   ``kv_scale_format``).
SparseMLASm120Wrapper = _SparseMLAPagedAttentionRunner


# Decode-DSv3.2 / DSv4: chunks_per_block (cpb) comes from the calibrated
# analytical model in _sparse_mla_sm120_cpb. Constants are calibrated once per
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


@functools.cache
def _get_sparse_mla_sm120_decode_module():
    """Build and cache the SM120 sparse-MLA decode kernel module."""
    return gen_sparse_mla_sm120_module().build_and_load()


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
) -> torch.Tensor:
    r"""Sparse-MLA paged decode (DSv4 standalone kernel) on SM120.

    The decode-dsv4 path is the split-K decode variant where each block handles
    ``chunks_per_block`` chunks of 64 candidates each. The wall-time-optimal
    value is shape-dependent; this wrapper picks it per call with the
    calibrated analytical model in :mod:`._sparse_mla_sm120_cpb`.

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
        Scratch, ``[T, num_heads, num_splits, d_v]`` bf16. ``num_splits =
        ceil(topk / 64) + ceil(extra_topk / 64)`` (64-wide candidate tiles;
        DOTS3_SWA tiles 32).
    mid_lse : torch.Tensor
        Scratch, ``[T, num_heads, num_splits]`` float32.
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

    Returns
    -------
    output : torch.Tensor
        The mutated output tensor (for chaining).
    """
    # d_qk resolves the model type: 512 -> DSV4 (d_v 512), 1088 -> DOTS3_SWA
    # (d_v 1024). The FFI applies the same resolution.
    model_type = _MODEL_TYPE_DOTS3_SWA if q.shape[-1] == 1088 else _MODEL_TYPE_DSV4
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
        cpb_override,
    )
    return output
