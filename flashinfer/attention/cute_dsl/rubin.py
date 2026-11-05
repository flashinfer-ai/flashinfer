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
"""Rubin plans for the public ragged prefill wrapper."""

import math
import numbers
import struct
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...utils import check_shape_dtype_device, get_compute_capability


def _finite_fp32_scale(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite host scalar representable in FP32")
    try:
        value = float(value)
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        raise ValueError(f"{name} must be finite and representable in FP32") from None
    if not math.isfinite(value) or not math.isfinite(rounded):
        raise ValueError(f"{name} must be finite and representable in FP32")
    if positive and rounded <= 0:
        raise ValueError(f"{name} must remain positive after FP32 conversion")
    return value


def _validate_rubin_fmha_scales(sm_scale, scale_q, scale_k, scale_v, scale_o):
    """Validate host arithmetic before empty-row writes or an FMHA launch."""
    sm_scale, scale_q, scale_k, scale_v, scale_o = (
        _finite_fp32_scale(name, value)
        for name, value in (
            ("sm_scale", sm_scale if sm_scale is not None else 1.0 / math.sqrt(192)),
            ("q_scale", scale_q),
            ("k_scale", scale_k),
            ("v_scale", scale_v),
            ("o_scale", scale_o),
        )
    )
    if scale_o == 0:
        raise ValueError("o_scale must be nonzero")
    # Nonpositive kernel scales invalidate online rescaling and masked logits.
    # Check the effective FP32 values; individual factors may be signed or tiny.
    score = _finite_fp32_scale(
        "effective softmax scale", scale_q * scale_k * sm_scale, positive=True
    )
    score_log2 = _finite_fp32_scale(
        "base-2 softmax scale", score * math.log2(math.e), positive=True
    )
    output = _finite_fp32_scale("effective output scale", scale_v / scale_o)
    return score, score_log2, output


def _validate_rubin_outputs(q, k, v, out, lse=None, lse_out=None) -> None:
    """Check contiguous output byte ranges before any output writes."""
    from .rubin_gqa import _overlaps

    outputs = [("out", out)]
    if lse is not None:
        outputs.append(("lse", lse))
    if lse_out is not None:
        outputs.append(("lse_out", lse_out))
    for name, tensor in outputs:
        if not tensor.is_contiguous():
            raise ValueError(f"Rubin prefill requires contiguous {name}")
    # Inputs are already validated as contiguous by both callers. Empty
    # tensors occupy no bytes, even when their pointer lies inside another view.
    for i, (_, output) in enumerate(outputs):
        if not output.numel():
            continue
        if any(t.numel() and _overlaps(output, t) for t in (q, k, v)):
            raise ValueError("Rubin prefill output tensors must not overlap inputs")
        if any(t.numel() and _overlaps(output, t) for _, t in outputs[:i]):
            raise ValueError("Rubin prefill output tensors must not overlap each other")


def _validate_rubin_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    lse: Optional[torch.Tensor] = None,
    is_causal: bool,
    window_left: int,
    window_right: int = -1,
    attention_sinks: Optional[torch.Tensor],
    skip_softmax_threshold_scale_factor: Optional[float],
) -> None:
    """Validate explicit FMHA requests, including calls with no active rows."""
    if q.device.type != "cuda" or get_compute_capability(q.device) != (10, 7):
        raise ValueError("Rubin FMHA requires a CUDA SM107 device")
    if q.dtype != k.dtype or (q.dtype, v.dtype, out.dtype) != (
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
    ):
        raise ValueError("Rubin FMHA requires E4M3 Q/K/V and BF16 output")
    if (q.shape[-1], k.shape[-1], v.shape[-1]) != (192, 192, 128):
        raise ValueError("Rubin FMHA requires Q/K head_dim=192 and V head_dim=128")
    if q.shape[1] <= 0 or q.shape[1] != k.shape[1] or k.shape[:2] != v.shape[:2]:
        raise ValueError(
            "Rubin FMHA requires equal positive Q/K/V head counts and matching KV lengths"
        )
    if any(t.device != q.device for t in (k, v, out)):
        raise ValueError("Rubin FMHA requires Q/K/V and output on the same device")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Rubin FMHA requires contiguous Q/K/V")
    _validate_rubin_outputs(q, k, v, out, lse)
    if (
        window_left != -1
        or (window_right != -1 and not (is_causal and window_right == 0))
        or attention_sinks is not None
        or (
            skip_softmax_threshold_scale_factor is not None
            and skip_softmax_threshold_scale_factor > 0
        )
    ):
        raise NotImplementedError(
            "Rubin FMHA does not support sliding windows, sinks or skip-softmax"
        )


class RubinPrefillPlan:
    """Own the qualified D192/V128 FMHA or fixed-length GQA plan."""

    def __init__(
        self,
        device: torch.device,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> None:
        self.device = device
        self.qo_indptr = qo_indptr
        self.kv_indptr = kv_indptr
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.causal = causal
        self.sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(192)
        self.batch_size = len(qo_indptr_host) - 1
        self.total_q_len = int(qo_indptr_host[-1])
        self.total_kv_len = int(kv_indptr_host[-1])
        self.q_seq_lens_cpu = (qo_indptr_host[1:] - qo_indptr_host[:-1]).to(torch.int32)
        self.kv_seq_lens_cpu = (kv_indptr_host[1:] - kv_indptr_host[:-1]).to(
            torch.int32
        )
        self.q_len = int(self.q_seq_lens_cpu[0])
        self.kv_len = int(self.kv_seq_lens_cpu[0])
        self.max_q_len = int(self.q_seq_lens_cpu.max())
        self.max_kv_len = int(self.kv_seq_lens_cpu.max())
        self._kernels: Dict[bool, Any] = {}

    @property
    def is_grouped(self) -> bool:
        return self.num_qo_heads != self.num_kv_heads

    def _graph_signature(self) -> Tuple[int, int, int, int, int, bool, float]:
        return (
            self.batch_size,
            self.q_len if self.is_grouped else self.total_q_len,
            self.kv_len if self.is_grouped else self.total_kv_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.causal,
            self.sm_scale,
        )

    def check_replan(
        self,
        *,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> None:
        """Compare a validated request with the installed graph configuration."""
        if self._graph_signature() != (
            len(qo_indptr_host) - 1,
            int(qo_indptr_host[1]) if self.is_grouped else int(qo_indptr_host[-1]),
            int(kv_indptr_host[1]) if self.is_grouped else int(kv_indptr_host[-1]),
            num_qo_heads,
            num_kv_heads,
            causal,
            sm_scale if sm_scale is not None else 1.0 / math.sqrt(192),
        ):
            lengths = (
                "fixed sequence lengths"
                if self.is_grouped
                else "fixed total Q length and fixed total KV length"
            )
            raise ValueError(
                f"Rubin CUDA-graph plans require {lengths}, batch size, head counts, "
                "causal mode and softmax scale; create a new wrapper to change "
                "the configuration."
            )

    def check_legacy_options(self, causal: bool, sm_scale: Optional[float]) -> None:
        """Reject legacy overrides that the installed plan cannot honor."""
        normalized_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(192)
        if causal != self.causal or normalized_scale != self.sm_scale:
            raise ValueError(
                "Rubin forward causal and sm_scale must match plan(); "
                "call plan() to change the configuration."
            )

    @staticmethod
    def validate(
        *,
        device: torch.device,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        kv_layout: str,
        causal: bool,
        custom_mask: Optional[torch.Tensor],
        packed_custom_mask: Optional[torch.Tensor],
        pos_encoding_mode: str,
        use_fp16_qk_reduction: bool,
        window_left: int,
        logits_soft_cap: float,
        prefix_len_ptr: Optional[torch.Tensor],
        token_pos_in_items_ptr: Optional[torch.Tensor],
        token_pos_in_items_len: int,
        max_item_len_ptr: Optional[torch.Tensor],
        fixed_split_size: int,
        seq_lens: Optional[torch.Tensor],
        seq_lens_q: Optional[torch.Tensor],
        max_token_per_sequence: Optional[int],
        max_sequence_kv: Optional[int],
        v_indptr: Optional[torch.Tensor],
        o_indptr: Optional[torch.Tensor],
    ) -> None:
        """Validate before the wrapper updates its registered device buffers."""
        if get_compute_capability(device) != (10, 7):
            raise ValueError("CuTe DSL D192/V128 prefill requires SM107")
        if (head_dim_qk, head_dim_vo) != (192, 128):
            raise ValueError(
                "Rubin prefill requires head_dim_qk=192 and head_dim_vo=128"
            )
        if (
            q_dtype != torch.float8_e4m3fn
            or kv_dtype != torch.float8_e4m3fn
            or o_dtype != torch.bfloat16
        ):
            raise ValueError("Rubin prefill requires E4M3 Q/K/V and BF16 output")
        if kv_layout != "NHD":
            raise NotImplementedError("Rubin prefill only supports NHD layout")
        if not isinstance(causal, bool):
            raise ValueError("causal must be bool")
        if (
            num_qo_heads <= 0
            or num_kv_heads <= 0
            or num_qo_heads % num_kv_heads
            or num_qo_heads // num_kv_heads not in (1, 2, 4, 8)
        ):
            raise ValueError("Rubin prefill requires a Q/KV head ratio of 1, 2, 4 or 8")
        unsupported = {
            "custom_mask": custom_mask is not None or packed_custom_mask is not None,
            "pos_encoding_mode": pos_encoding_mode != "NONE",
            "use_fp16_qk_reduction": use_fp16_qk_reduction,
            "window_left": window_left != -1,
            "logits_soft_cap": logits_soft_cap != 0.0,
            "multi-item scoring": any(
                x is not None
                for x in (prefix_len_ptr, token_pos_in_items_ptr, max_item_len_ptr)
            )
            or token_pos_in_items_len != 0,
            "fixed_split_size": fixed_split_size != -1,
            "sequence-length overrides": any(
                x is not None
                for x in (
                    seq_lens,
                    seq_lens_q,
                    max_token_per_sequence,
                    max_sequence_kv,
                )
            ),
            "separate V/O indptr": v_indptr is not None or o_indptr is not None,
        }
        for option, enabled in unsupported.items():
            if enabled:
                raise NotImplementedError(f"Rubin prefill does not support {option}")
        for name, indptr in (
            ("qo_indptr", qo_indptr_host),
            ("kv_indptr", kv_indptr_host),
        ):
            if indptr.ndim != 1 or indptr.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be a one-dimensional integer tensor")
            if indptr.numel() < 2:
                raise ValueError("Rubin prefill requires a positive batch size")
        if len(qo_indptr_host) != len(kv_indptr_host):
            raise ValueError(
                "The kv_indptr length should be equal to mask_indptr length."
            )
        for name, indptr in (
            ("qo_indptr", qo_indptr_host),
            ("kv_indptr", kv_indptr_host),
        ):
            if (
                int(indptr[0]) != 0
                or bool(torch.any(indptr[1:] < indptr[:-1]))
                or int(indptr[-1]) >= 2**31
            ):
                raise ValueError(
                    f"{name} must start at zero and contain nondecreasing signed 32-bit offsets"
                )
        if num_qo_heads == num_kv_heads and len(qo_indptr_host) - 1 > 65535:
            raise ValueError("Rubin FMHA requires batch_size <= 65535")
        if num_qo_heads != num_kv_heads:
            if (len(qo_indptr_host) - 1) * num_kv_heads > 65535:
                raise ValueError(
                    "Rubin GQA requires batch_size * num_kv_heads <= 65535"
                )
            q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
            kv_lens = kv_indptr_host[1:] - kv_indptr_host[:-1]
            q_len, kv_len = int(q_lens[0]), int(kv_lens[0])
            if (
                q_len <= 0
                or kv_len <= 0
                or not bool(torch.all(q_lens == q_len))
                or not bool(torch.all(kv_lens == kv_len))
            ):
                raise ValueError(
                    "Rubin GQA requires positive uniform Q and KV sequence lengths"
                )
            if causal and q_len > kv_len:
                raise ValueError("Causal Rubin GQA requires Q length <= KV length")

    @classmethod
    def plan(
        cls,
        *,
        device: torch.device,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> "RubinPrefillPlan":
        """Build a complete plan from validated metadata and wrapper buffers."""
        plan = cls(
            device,
            qo_indptr,
            kv_indptr,
            qo_indptr_host,
            kv_indptr_host,
            num_qo_heads,
            num_kv_heads,
            causal,
            sm_scale,
        )
        if plan.is_grouped:
            from ...cute_dsl.attention.fmha.compile_rubin_gqa import (
                compile_rubin_gqa_kernel,
            )

            with torch.cuda.device(device):
                plan._kernels = {
                    with_lse: compile_rubin_gqa_kernel(
                        num_qo_heads,
                        num_kv_heads,
                        causal,
                        with_lse,
                        device,
                    )
                    for with_lse in (False, True)
                }
        return plan

    def run_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        lse: Optional[torch.Tensor],
        lse_out: Optional[torch.Tensor],
        enable_pdl: bool,
        q_scale: Optional[float],
        k_scale: Optional[float],
        v_scale: Optional[float],
        o_scale: Optional[float],
        custom_args: Tuple[Any, ...],
        sinks: Optional[torch.Tensor],
        kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        if kv_cache_sf is not None:
            raise NotImplementedError(
                "cute-dsl backend does not support NVFP4 packed KV cache (kv_cache_sf)"
            )
        if sinks is not None:
            raise NotImplementedError("Rubin prefill does not support attention sinks")
        if custom_args:
            raise NotImplementedError(
                "Rubin prefill does not accept custom run arguments"
            )
        for name, tensor, shape in (
            ("q", q, (self.total_q_len, self.num_qo_heads, 192)),
            ("k", k, (self.total_kv_len, self.num_kv_heads, 192)),
            ("v", v, (self.total_kv_len, self.num_kv_heads, 128)),
        ):
            check_shape_dtype_device(
                tensor, shape, torch.float8_e4m3fn, self.device, name
            )
            if not tensor.is_contiguous():
                raise ValueError(f"Rubin prefill requires contiguous {name}")
        # Include the caller's HN destination, which is filled after the kernel
        # runs into an internal NH LSE buffer.
        _validate_rubin_outputs(q, k, v, out, lse, lse_out)
        if self.is_grouped:
            from .rubin_gqa import _run_rubin_gqa_prefill

            _run_rubin_gqa_prefill(
                q.view(self.batch_size, self.q_len, self.num_qo_heads, 192),
                k.view(self.batch_size, self.kv_len, self.num_kv_heads, 192),
                v.view(self.batch_size, self.kv_len, self.num_kv_heads, 128),
                out.view(self.batch_size, self.q_len, self.num_qo_heads, 128),
                kernel=self._kernels[lse is not None],
                causal=self.causal,
                lse=(
                    lse.view(self.batch_size, self.q_len, self.num_qo_heads)
                    if lse is not None
                    else None
                ),
                sm_scale=self.sm_scale,
                scale_q=q_scale if q_scale is not None else 1.0,
                scale_k=k_scale if k_scale is not None else 1.0,
                scale_v=v_scale if v_scale is not None else 1.0,
                scale_o=o_scale if o_scale is not None else 1.0,
            )
        else:
            from .fmha import cute_dsl_fmha_ragged_prefill

            _validate_rubin_fmha_scales(
                self.sm_scale,
                q_scale if q_scale is not None else 1.0,
                k_scale if k_scale is not None else 1.0,
                v_scale if v_scale is not None else 1.0,
                o_scale if o_scale is not None else 1.0,
            )

            # Graph plans fix the total query length, so no replay can give
            # this empty launch work. A zero-sized FMHA grid is invalid.
            if self.total_q_len == 0:
                return

            # FMHA skips empty KV rows. Neutralize their output without host
            # synchronization; graph replay may change which rows are empty.
            active_rows = (self.q_seq_lens_cpu > 0) & (self.kv_seq_lens_cpu > 0)
            is_capturing = torch.cuda.is_current_stream_capturing()
            if not bool(active_rows.all()):
                if not bool(active_rows.any()):
                    out.zero_()
                    if lse is not None:
                        lse.fill_(-float("inf"))
                    if not is_capturing:
                        return
                else:
                    q_lens = self.qo_indptr[1:] - self.qo_indptr[:-1]
                    kv_lens = self.kv_indptr[1:] - self.kv_indptr[:-1]
                    row_active = (q_lens > 0) & (kv_lens > 0)
                    token_inactive = ~torch.repeat_interleave(
                        row_active, q_lens, output_size=q.shape[0]
                    )
                    out.masked_fill_(token_inactive.view(-1, 1, 1), 0)
                    if lse is not None:
                        lse.masked_fill_(token_inactive.view(-1, 1), -float("inf"))
            elif is_capturing:
                out.zero_()
                if lse is not None:
                    lse.fill_(-float("inf"))

            cute_dsl_fmha_ragged_prefill(
                q=q,
                k=k,
                v=v,
                o=out,
                qo_indptr=self.qo_indptr,
                kv_indptr=self.kv_indptr,
                sm_scale=self.sm_scale,
                scale_q=q_scale if q_scale is not None else 1.0,
                scale_k=k_scale if k_scale is not None else 1.0,
                scale_v=v_scale if v_scale is not None else 1.0,
                scale_o=o_scale if o_scale is not None else 1.0,
                # Replay may redistribute queries across sequences within the
                # captured storage. Device lengths skip unused query tiles.
                max_qo_len=self.total_q_len if is_capturing else self.max_q_len,
                max_kv_len=self.max_kv_len,
                enable_pdl=enable_pdl,
                is_causal=self.causal,
                lse=lse,
                backend="cute-dsl-rubin",
            )
