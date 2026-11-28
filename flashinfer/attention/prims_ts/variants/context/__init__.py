# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Task-scheduled contiguous and paged context attention.

The public surface intentionally exposes attention semantics and optional
compile-time metadata contracts, not scheduler choices. Reusable contiguous
and paged plans select resource topology from capacities, while logical Q/KV
lengths and request metadata remain runtime inputs. Both wrappers consume
the current request extents at launch. Paged plans
conservatively treat all request lengths as variable unless the caller opts
into an exact-uniform or zero-causal-offset contract.
Single-instance dense paged plans use a static persistent launch when their
logical CTA grid exceeds one resident wave; smaller grids launch one CTA per
tile. Causal and paired persistent paged plans use CLC. The private policy is
query-paired unless a positive left window requires head-paired GQA.
Causal windows are bottom-right aligned: for row ``q``, the inclusive right
position is ``q + (S_kv - S_q)`` and ``window_left`` is measured from that
position.

PrimTS context entry points are intentionally excluded from ``fi_trace`` for
now; unlike the decode APIs, their ``@flashinfer_experimental_api`` decorators do not
register trace templates.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch

from ...common.scales import (
    GlobalScale,
    normalize_scale,
    resolve_scale,
    scale_types,
)

from ..vc_attention import (
    VCAttentionConfig,
    VCAttentionParams,
    validate_vc_params,
    vc_scale_shapes,
)

from flashinfer.api_logging import flashinfer_experimental_api
from ...common.attention_args import (
    prepare_lse,
    validate_lse_mode,
    validate_attention_sinks,
    validate_layout as validate_kv_layout,
)

from ..sage.config import SageAttentionConfig, SageAttentionParams, validate_sage_params
from ..sage.plan import SagePreparedScales
from .validation import (
    DEFAULT_PAGED_KV_PAGE_SIZE,
    mxfp8_scale_atoms,
    refresh_variable_window_cta_starts,
    validate_alignment,
    validate_compact,
    validate_context_softmax_scale,
    validate_paged_runtime_inputs,
    validate_paged_runtime_metadata,
    validate_runtime_inputs,
    validate_tensor,
    validate_variable_window_bounds,
    validate_variable_window_cta_starts,
    variable_window_q_tile_size,
)
from .geometry import (
    ContextPlanGeometry,
    PagedContextPlanGeometry,
    resolve_context_plan_geometry,
    resolve_geometry,
    resolve_paged_geometry,
    resolve_paged_plan_geometry,
)
from .compile import (
    build_context_sage_state,
    context_compile_spec,
    get_compiled_context,
    get_compiled_paged_context,
    paged_context_compile_spec,
)


if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ContextPlanState:
    """Atomically published state for one compiled contiguous context plan."""

    geometry: ContextPlanGeometry
    softmax_scale: GlobalScale
    output_scale: GlobalScale
    empty_i32: torch.Tensor
    variable_window_padded_starts: Optional[torch.Tensor]
    variable_window_cta_starts: torch.Tensor
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]
    vc_scale_shapes: Optional[dict[str, tuple[int, ...]]] = None
    # Output scale of VC runs: the per-channel V scale times the planned or passed scale.
    vc_output_scale: Optional[torch.Tensor] = None
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    sage: Optional[SagePreparedScales] = None


@dataclass(frozen=True)
class PagedContextPlanState:
    """Atomically published state for one compiled paged-context plan."""

    geometry: PagedContextPlanGeometry
    softmax_scale: GlobalScale
    output_scale: GlobalScale
    compiled: Callable[..., None]
    policy: tuple[tuple[str, object], ...]
    lse_mode: str = "none"
    use_attention_sinks: bool = False


def prepare_out(
    out: Optional[torch.Tensor],
    *,
    q: torch.Tensor,
    out_dtype: torch.dtype,
    head_dim_vo: int | None = None,
) -> torch.Tensor:
    out_shape = (*q.shape[:-1], head_dim_vo or q.shape[-1])
    if out is None:
        return torch.empty(out_shape, dtype=out_dtype, device=q.device)
    validate_tensor(out, "out")
    if tuple(out.shape) != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
    if out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = "[total_q, Hq, D]" if q.ndim == 3 else "[B, Sq, Hq, D]"
    validate_compact(out, "out", layout)
    validate_alignment(out, "out", 16)
    return out


def variable_window_buffers(geometry, empty_i32):
    if geometry.mask_type != "variable_window":
        return None, empty_i32
    tile_size_q = variable_window_q_tile_size(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    cta_starts = torch.empty(
        geometry.batch_size * num_seq_tiles, dtype=torch.int32, device=geometry.device
    )
    padded_starts = (
        torch.empty(
            (geometry.batch_size, padded_rows),
            dtype=torch.int32,
            device=geometry.device,
        )
        if padded_rows != geometry.max_q_len
        else None
    )
    return padded_starts, cta_starts


class BatchPrefillTSWrapper:
    """Reusable contiguous context attention with eager planning.

    run() accepts fixed BSHD or packed THD Q/K/V. Plan owns model defaults and
    variable-window fallback metadata, while every run supplies request tensors.
    Supports equal D128/D192/D256 or contiguous non-absorbed MLA QK192/V128 geometry.
    Q/K/O are compact. V has compact heads/features, with 16-byte-aligned,
    nonoverlapping token and optional batch strides supplied at runtime.
    Compact and row-padded V share the same plan and compiled kernel.
    Dense fixed D128 Sage plans bind prepared per-block/channel scale tensors.
    Replanning invalidates graphs; complete prior launches and replays first.
    VC plans also own a refreshed per-channel output-scale buffer; their runs
    must not overlap."""

    @flashinfer_experimental_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled context-attention wrapper."""
        self.plan_state: Optional[ContextPlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        head_dim_vo: int | None = None,
        q_dtype: torch.dtype,
        k_dtype: torch.dtype,
        v_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        packed_query: bool = False,
        mask_type: Literal["dense", "causal", "variable_window"] = "dense",
        window_left: int = -1,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: GlobalScale = 1.0,
        vc_config: Optional[VCAttentionConfig] = None,
        lse_mode: Literal["none", "base2"] = "none",
        use_attention_sinks: bool = False,
        sage: Optional[SageAttentionConfig] = None,
    ) -> None:
        """Select/cache one execution policy, obtain compiled kernels, and bind resources.

        ``vc_config`` selects a VC-Attention recipe and the planned Q/K dtype
        picks its Q/K side. bf16 Q/K runs VC-Attention-QK16 and E4M3 or INT8
        Q/K runs VC-Attention-QK8, whose Hadamard-rotated, per-block scaled Q
        and K need ``vc.q_scale`` and ``vc.k_scale`` on every :meth:`run`. V is
        E4M3 in both, stored as per-128-token-tile residuals whose bf16 tile
        means the kernel restores. Every :meth:`run` then supplies the permuted
        K, the E4M3 V and ``vc=VCAttentionParams(...)``.

        batch_size is exact. Fixed contiguous inputs use exact Q/KV tensor extents;
        packed/paged request lengths may vary within max_q_len/max_kv_len.
        Complete planning outside CUDA graph capture. Replanning replaces owned
        resources and invalidates previous graphs; finish their launches first.
        lse_mode='base2' enables final LSE stores, and use_attention_sinks enables
        a per-run sink binding. Each global scale independently accepts a positive
        host float or a contiguous CUDA FP32 tensor of shape () or (1,) on this device.
        Tensor values must remain finite and positive; metadata validation never
        reads them back. Types specialize the kernel, while values and addresses
        do not. Softmax scales use natural-exponential units for either type.
        Sage plans use BF16 output by default and require per-run Sage scales.
        Sage V-mean restoration cannot be combined with attention sinks.
        Equivalent plans reuse policy and compiled-kernel caches.
        VC runs sharing a plan must not overlap.
        q_dtype=k_dtype=torch.float8_e8m0fnu selects MXFP8 Q/K: float8_e4m3fn data
        with one E8M0 scale per 32 channels, E4M3 V by default, fixed dense or causal
        plans only; every run then supplies q_scales and k_scales.
        """

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(use_attention_sinks, bool):
            raise TypeError("use_attention_sinks must be a bool")
        if out_dtype is not None:
            resolved_out_dtype = out_dtype
        elif sage is not None or q_dtype in (torch.float8_e8m0fnu, torch.int8):
            # As in block-sparse Sage plans; an 8-bit output is not a recipe, and
            # the MXFP8 token is a scale dtype, not an output dtype.
            resolved_out_dtype = torch.bfloat16
        else:
            resolved_out_dtype = q_dtype
        if v_dtype is not None:
            resolved_v_dtype = v_dtype
        elif k_dtype in (torch.float8_e8m0fnu, torch.int8):
            # MXFP8 and INT8 Q/K keep E4M3 V.
            resolved_v_dtype = torch.float8_e4m3fn
        else:
            resolved_v_dtype = k_dtype
        if k_dtype != q_dtype:
            raise NotImplementedError(
                "attention-ts context requires Q and K to use the same dtype; "
                f"got q_dtype={q_dtype} and k_dtype={k_dtype}"
            )
        if k_dtype == torch.int8 and sage is None and vc_config is None:
            raise NotImplementedError(
                "int8 Q/K require a Sage recipe (sage=SageAttentionConfig(...)) or a "
                "VC-Attention recipe (vc_config=VCAttentionConfig(...)) that "
                "dequantizes them"
            )
        geometry = resolve_context_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_vo=head_dim_vo,
            qk_dtype=k_dtype,
            pv_dtype=resolved_v_dtype,
            packed_query=packed_query,
            mask_type=mask_type,
            window_left=window_left,
            out_dtype=resolved_out_dtype,
            sage=sage,
        )
        if sage is not None and sage.v_mean and use_attention_sinks:
            raise NotImplementedError(
                "Sage V-mean restoration does not support attention sinks"
            )
        if vc_config is not None and not isinstance(vc_config, VCAttentionConfig):
            raise TypeError("vc_config must be a VCAttentionConfig instance")
        if vc_config is not None and (packed_query or mask_type != "dense"):
            raise ValueError(
                "vc_config requires a dense contiguous fixed-length plan; VC-Attention "
                "has no packed, causal or variable-window kernel"
            )
        if vc_config is not None and (lse_mode != "none" or use_attention_sinks):
            # VC uses quantized P sums; ordinary context LSE/sink math does not apply.
            raise NotImplementedError(
                "VC-Attention does not support LSE or attention sinks"
            )
        geometry = replace(geometry, vc=vc_config)
        if softmax_scale is None:
            softmax_scale = geometry.head_dim**-0.5
        softmax_scale = normalize_scale(softmax_scale, "softmax_scale", geometry.device)
        output_scale = normalize_scale(output_scale, "output_scale", geometry.device)
        validate_context_softmax_scale(softmax_scale)
        vc_output_scale_buffer = (
            torch.empty(
                (
                    geometry.batch_size,
                    geometry.num_qo_heads,
                    geometry.head_dim_vo or geometry.head_dim,
                ),
                dtype=torch.float32,
                device=geometry.device,
            )
            if vc_config is not None
            else None
        )
        empty_i32 = torch.empty(1, dtype=torch.int32, device=geometry.device)
        variable_window_padded_starts, variable_window_cta_starts = (
            variable_window_buffers(geometry, empty_i32)
        )
        # Keep all runtime tensor allocation ahead of CUTLASS JIT. Besides
        # making plan publication atomic, this lets compute-sanitizer patch the
        # generated attention kernel without interleaving later PyTorch setup
        # launches with the DSL compiler/runtime callbacks.
        vc_shapes = None
        if vc_config is not None:
            vc_shapes = vc_scale_shapes(
                vc_config,
                batch_size=geometry.batch_size,
                seq_len_kv=geometry.max_seq_len_k,
                num_kv_heads=geometry.num_kv_heads,
                head_dim=geometry.head_dim_vo or geometry.head_dim,
                seq_len_q=geometry.max_q_len,
                num_qo_heads=geometry.num_qo_heads,
                qk_quant=geometry.qk_dtype in (torch.float8_e4m3fn, torch.int8),
            )
        sage_state = (
            build_context_sage_state(geometry) if geometry.sage is not None else None
        )
        compiled, policy = get_compiled_context(
            context_compile_spec(
                geometry,
                lse_mode=lse_mode,
                use_attention_sinks=use_attention_sinks,
                scale_types=scale_types(softmax_scale, output_scale),
            )
        )
        self.plan_state = ContextPlanState(
            geometry=geometry,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            empty_i32=empty_i32,
            variable_window_padded_starts=variable_window_padded_starts,
            variable_window_cta_starts=variable_window_cta_starts,
            compiled=compiled,
            policy=policy,
            vc_scale_shapes=vc_shapes,
            vc_output_scale=vc_output_scale_buffer,
            lse_mode=lse_mode,
            use_attention_sinks=use_attention_sinks,
            sage=sage_state,
        )

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        *,
        q_scales: Optional[torch.Tensor] = None,
        k_scales: Optional[torch.Tensor] = None,
        variable_window_token_starts: Optional[torch.Tensor] = None,
        variable_window_token_ends: Optional[torch.Tensor] = None,
        variable_window_cta_starts: Optional[torch.Tensor] = None,
        max_q_len: int | None = None,
        max_kv_len: int | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: Optional[GlobalScale] = None,
        vc: Optional[VCAttentionParams] = None,
        sage: Optional[SageAttentionParams] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch the prepared contiguous attention with live request tensors.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Q/K are compact; V may have padding between token rows and batches,
        with compact heads/features and 16-byte-aligned strides.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        Ordinary dense plans with LSE enabled write zero output and -inf LSE for
        empty K/V requests. Packed storage must contain at least one physical
        K/V token; empty requests are represented by equal adjacent KV offsets.

        max_q_len/max_kv_len are launch bounds within plan capacities.
        Validation infers omitted values from existing metadata; trusted packed
        calls use plan bounds. Exact runtime lengths do not specialize kernels.

        Packed inputs require qo_indptr and kv_indptr. Variable-window fixed inputs
        require inclusive start/end bounds; ends must be nondecreasing within each
        request, while starts may be unordered. Optional CTA minima avoid refreshing
        wrapper-owned fallback scratch. Scale overrides must match each planned
        float/tensor kind, even with validate=False; None inherits the default.
        softmax_scale is the effective QK scale in natural-exponential units,
        and output_scale scales V/output. CUDA FP32 scales have shape () or (1,).
        Keep tensor buffers alive and update their contents before replay;
        captured floats remain fixed. Scale values and addresses are not JIT keys.

        Enabled attention_sinks supplies compact CUDA FP32 [Hq] natural-log sink
        logits after QK scaling. Values must be finite or -inf; validation checks only
        metadata. Each sink adds denominator mass with zero V contribution.
        Sage plans require matching SageAttentionParams; their prepared scales
        are refreshed on the current stream before attention.
        VC plans require vc=VCAttentionParams(...) with permuted K and residual V;
        its per-channel V scale multiplies output_scale in the plan-owned buffer.
        MXFP8 plans require q_scales and k_scales, one 512-byte 128x4 scale atom per
        128-token tile and head as mxfp8_quantize_bshd writes them; other plans
        reject them."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self.plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        geometry = state.geometry
        if geometry.mask_type != "variable_window" and (
            variable_window_token_starts is not None
            or variable_window_token_ends is not None
            or variable_window_cta_starts is not None
        ):
            raise ValueError(
                "variable-window metadata requires mask_type='variable_window'"
            )
        if not validate and not geometry.packed_query:
            max_q_len = int(q.shape[1])
            max_kv_len = (
                geometry.max_seq_len_k if geometry.vc is not None else int(k.shape[1])
            )
        if geometry.packed_query:
            if qo_indptr is None or kv_indptr is None:
                raise ValueError(
                    "packed plans require qo_indptr and kv_indptr in run()"
                )
            runtime_qo_indptr = qo_indptr
            runtime_kv_indptr = kv_indptr
        else:
            if qo_indptr is not None or kv_indptr is not None:
                raise ValueError(
                    "qo_indptr and kv_indptr require packed_query=True in plan()"
                )
            runtime_qo_indptr = state.empty_i32
            runtime_kv_indptr = state.empty_i32

        if (vc is None) == (geometry.vc is not None):
            raise ValueError(
                "vc=VCAttentionParams(...) is required by a VC plan and rejected "
                "by a plan without vc_config"
            )
        vc_output_scale: Optional[torch.Tensor] = None
        vc_mu: Optional[torch.Tensor] = None
        vc_q_scale: Optional[torch.Tensor] = None
        vc_k_scale: Optional[torch.Tensor] = None
        if vc is not None:
            if validate:
                validate_vc_params(vc, state.vc_scale_shapes, device=geometry.device)
            if geometry.qk_dtype in (torch.float8_e4m3fn, torch.int8):
                if vc.q_scale is None or vc.k_scale is None:
                    raise ValueError(
                        "VC-Attention-QK8 plans require vc.q_scale and vc.k_scale"
                    )
                vc_q_scale = vc.q_scale.view(geometry.num_qo_heads, -1)
                vc_k_scale = vc.k_scale.view(geometry.num_kv_heads, -1)
            elif vc.q_scale is not None or vc.k_scale is not None:
                raise ValueError(
                    "vc.q_scale and vc.k_scale belong to VC-Attention-QK8 plans with "
                    "E4M3 or INT8 Q/K"
                )
            vc_mu = vc.tile_means
            # VC plans have equal Q and K/V head counts, so the per-channel V
            # scale is already in output order.
            vc_output_scale = vc.v_scale
        if validate:
            actual_q, actual_kv = validate_runtime_inputs(
                q,
                k,
                v,
                geometry,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
            )
            from ...common.attention_args import resolve_max_sequence_length

            max_q_len = resolve_max_sequence_length(
                max_q_len, actual_q, "max_q_len", geometry.max_q_len
            )
            max_kv_len = resolve_max_sequence_length(
                max_kv_len, actual_kv, "max_kv_len", geometry.max_seq_len_k
            )
        else:
            if max_q_len is None:
                max_q_len = geometry.max_q_len
            if max_kv_len is None:
                max_kv_len = geometry.max_seq_len_k
        if (sage is None) != (state.sage is None):
            raise ValueError(
                "sage=SageAttentionParams(...) is required by a Sage plan and "
                "rejected by a plan without Sage attention"
            )
        if sage is not None and validate:
            validate_sage_params(sage, state.sage.scale_shapes, device=geometry.device)
        mxfp8 = geometry.qk_dtype == torch.float8_e8m0fnu
        if (q_scales is not None) != mxfp8 or (k_scales is not None) != mxfp8:
            raise ValueError(
                "q_scales and k_scales are required by an MXFP8 plan and rejected "
                "by a plan without MXFP8 Q/K"
            )
        runtime_q_scales = runtime_k_scales = None
        if mxfp8:
            # One 512-byte 128x4 atom per 128-token tile and head.
            runtime_q_scales, runtime_k_scales = (
                mxfp8_scale_atoms(scales, tensor, name, validate)
                for scales, tensor, name in (
                    (q_scales, q, "q_scales"),
                    (k_scales, k, "k_scales"),
                )
            )
        if geometry.mask_type == "variable_window":
            if (
                variable_window_token_starts is None
                or variable_window_token_ends is None
            ):
                raise ValueError(
                    "variable-window plans require start and end bounds in run()"
                )
            if validate:
                (
                    runtime_window_starts,
                    runtime_window_ends,
                    runtime_window_start_values,
                ) = validate_variable_window_bounds(
                    variable_window_token_starts,
                    variable_window_token_ends,
                    geometry=geometry,
                )
            else:
                runtime_window_starts = variable_window_token_starts.flatten()
                runtime_window_ends = variable_window_token_ends.flatten()
            if variable_window_cta_starts is None:
                runtime_window_cta_starts = refresh_variable_window_cta_starts(
                    runtime_window_starts,
                    state=state,
                )
            elif validate:
                runtime_window_cta_starts = validate_variable_window_cta_starts(
                    variable_window_cta_starts,
                    token_start_values=runtime_window_start_values,
                    geometry=geometry,
                )
            else:
                runtime_window_cta_starts = variable_window_cta_starts.flatten()
        else:
            runtime_window_starts = state.empty_i32
            runtime_window_ends = state.empty_i32
            runtime_window_cta_starts = state.empty_i32
        softmax_scale = resolve_scale(
            softmax_scale, state.softmax_scale, "softmax_scale", geometry.device
        )
        validate_context_softmax_scale(softmax_scale)
        output_scale = resolve_scale(
            output_scale, state.output_scale, "output_scale", geometry.device
        )
        if vc_output_scale is not None:
            output_scale = torch.mul(
                vc_output_scale, output_scale, out=state.vc_output_scale
            )

        if out is None:
            out = torch.empty(
                (*q.shape[:-1], geometry.head_dim_vo or geometry.head_dim),
                dtype=geometry.out_dtype,
                device=q.device,
            )
        elif validate:
            out = prepare_out(
                out,
                q=q,
                out_dtype=geometry.out_dtype,
                head_dim_vo=geometry.head_dim_vo,
            )
        if (
            state.lse_mode != "none"
            and geometry.mask_type == "dense"
            and q.numel()
            and not k.numel()
        ):
            raise ValueError(
                "Dense LSE output requires at least one physical K/V token"
            )
        lse = prepare_lse(
            lse,
            q=q,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=q,
            enabled=state.use_attention_sinks,
            validate=validate,
        )
        if sage is not None:
            state.sage.prepare(
                sage.q_scale, sage.k_scale, sage.v_scale, sage.v_mean, state.sage.scales
            )
            runtime_sage_scales = state.sage.scales
        else:
            runtime_sage_scales = None
        state.compiled(
            q,
            k,
            v,
            out,
            lse,
            attention_sinks,
            softmax_scale,
            output_scale,
            runtime_qo_indptr,
            runtime_kv_indptr,
            runtime_q_scales,
            runtime_k_scales,
            runtime_window_starts,
            runtime_window_ends,
            runtime_window_cta_starts,
            vc_mu,
            vc_q_scale,
            vc_k_scale,
            runtime_sage_scales,
            max_q_len,
            max_kv_len,
        )
        return (out, lse) if return_lse else out


class BatchPrefillPagedTSWrapper:
    """Reusable packed-Q context attention over HND or NHD paged K/V.

    plan() retains model semantics and optional exact-uniform/zero-offset/zero-V-tail
    promises. run() supplies Q offsets, a row-strided page table, KV lengths, and
    host execution bounds. Cache normalization changes metadata without copying.
    Replanning invalidates graphs; complete prior launches and replays first."""

    @flashinfer_experimental_api
    def __init__(self, kv_layout: Literal["HND", "NHD"] = "HND") -> None:
        """Initialize an unplanned paged-context wrapper with a K/V layout."""
        validate_kv_layout(kv_layout)
        self.kv_layout = kv_layout
        self.plan_state: Optional[PagedContextPlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        *,
        device: int | str | torch.device,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        k_dtype: torch.dtype,
        v_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        page_size: int = DEFAULT_PAGED_KV_PAGE_SIZE,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: GlobalScale = 1.0,
        lse_mode: Literal["none", "base2"] = "none",
        use_attention_sinks: bool = False,
        uniform_packed_lengths: bool = False,
        has_q_offset: bool = True,
        paged_v_tail_is_zero: bool = False,
    ) -> None:
        """Select/cache one execution policy, obtain compiled kernels, and bind resources.

        Page-table padding exposes ceil(max_kv_len / page_size) columns.
        Uniform-length and zero-V-tail promises remain caller-owned.
        batch_size is exact. Fixed contiguous inputs use exact Q/KV tensor extents;
        packed/paged request lengths may vary within max_q_len/max_kv_len.
        Complete planning outside CUDA graph capture. Replanning replaces owned
        resources and invalidates previous graphs; finish their launches first.
        lse_mode='base2' enables final LSE stores, and use_attention_sinks enables
        a per-run sink binding. Each global scale independently accepts a positive
        host float or a contiguous CUDA FP32 tensor of shape () or (1,) on this device.
        Tensor values must remain finite and positive; metadata validation never
        reads them back. Types specialize the kernel, while values and addresses
        do not. Softmax scales use natural-exponential units for either type.
        Equivalent plans reuse policy and compiled-kernel caches."""

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(use_attention_sinks, bool):
            raise TypeError("use_attention_sinks must be a bool")
        resolved_out_dtype = q_dtype if out_dtype is None else out_dtype
        if v_dtype is not None:
            resolved_v_dtype = v_dtype
        elif k_dtype == torch.float8_e8m0fnu:
            # MXFP8 Q/K keeps E4M3 V.
            resolved_v_dtype = torch.float8_e4m3fn
        else:
            resolved_v_dtype = k_dtype
        if k_dtype != q_dtype:
            raise NotImplementedError(
                "attention-ts paged context requires Q and K to use the same "
                f"dtype; got q_dtype={q_dtype} and k_dtype={k_dtype}"
            )
        geometry = resolve_paged_plan_geometry(
            device=device,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qk_dtype=k_dtype,
            pv_dtype=resolved_v_dtype,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            out_dtype=resolved_out_dtype,
            uniform_packed_lengths=uniform_packed_lengths,
            has_q_offset=has_q_offset,
            paged_v_tail_is_zero=paged_v_tail_is_zero,
        )
        if softmax_scale is None:
            softmax_scale = geometry.head_dim**-0.5
        softmax_scale = normalize_scale(softmax_scale, "softmax_scale", geometry.device)
        output_scale = normalize_scale(output_scale, "output_scale", geometry.device)
        validate_context_softmax_scale(softmax_scale)
        compiled, policy = get_compiled_paged_context(
            paged_context_compile_spec(
                geometry,
                lse_mode=lse_mode,
                use_attention_sinks=use_attention_sinks,
                scale_types=scale_types(softmax_scale, output_scale),
            )
        )
        policy = tuple(
            (key, f"paged_{self.kv_layout.lower()}" if key == "kv_layout" else value)
            for key, value in policy
        )
        # Publish one immutable state object only after every fallible step.
        self.plan_state = PagedContextPlanState(
            geometry=geometry,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            compiled=compiled,
            policy=policy,
            lse_mode=lse_mode,
            use_attention_sinks=use_attention_sinks,
        )

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        max_q_len: int | None = None,
        max_kv_len: int | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: Optional[GlobalScale] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch packed-Q context with live offsets, page tables and KV lengths.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Q/K are compact; V may have padding between token rows and batches,
        with compact heads/features and 16-byte-aligned strides.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        * ``qo_indptr`` starts at zero, is nondecreasing, ends at the packed
          Q extent, and has deltas no greater than ``max_q_len``;
        * ``C`` is at least ``ceil(max_kv_len / page_size)``;
        * ``seq_lens_kv`` entries are nonnegative and do not exceed
          ``max_kv_len``;
        * active causal requests additionally satisfy ``Sq[b] <= Sk[b]``; and
        * every active page-table entry is a valid physical page ID in both
          separate, isomorphic K and V pools. Padding entries beyond the
          logical K/V length are not dereferenced.

        Metadata uses CUDA int32 on Q's device. qo_indptr starts at zero, is
        nondecreasing, and ends at the packed Q extent. Page-table columns are contiguous;
        rows may have non-overlapping padding. Active page IDs must index both caches.
        Active lengths must respect the supplied bounds and selected model promises, with
        Sq[b] <= Sk[b] for causal attention.

        When ``plan`` selected ``uniform_packed_lengths=True``, every Q delta
        must equal ``max_q_len`` and every K/V length must equal
        ``max_kv_len``. For a causal plan with ``has_q_offset=False``, every
        request must additionally satisfy ``Sq[b] == Sk[b]``.

        Scale overrides independently retain each planned float/tensor kind, even
        with validate=False; None inherits the default. CUDA FP32 scale tensors
        have shape () or (1,) and must remain finite and positive. Softmax units match
        the contiguous wrapper. Keep captured scale buffers alive and update
        their contents before replay; captured floats remain fixed.
        Enabled attention_sinks is a compact CUDA FP32 [Hq]
        vector of finite or -inf natural-log sink logits after QK scaling, with zero
        V contribution. Sink values and cache zero-tail promises are caller-owned."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self.plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        geometry = state.geometry
        if self.kv_layout == "NHD":
            k_cache = k_cache.transpose(1, 2)
            v_cache = v_cache.transpose(1, 2)
        if validate:
            validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
            actual_q, actual_kv = validate_paged_runtime_metadata(
                qo_indptr,
                block_tables,
                seq_lens_kv,
                geometry,
                total_q=int(q.shape[0]),
                num_physical_pages=int(k_cache.shape[0]),
            )
            from ...common.attention_args import resolve_max_sequence_length

            max_q_len = resolve_max_sequence_length(
                max_q_len,
                actual_q,
                "max_q_len",
                geometry.max_q_len,
            )
            max_kv_len = resolve_max_sequence_length(
                max_kv_len, actual_kv, "max_kv_len", geometry.max_kv_len
            )
        else:
            if max_q_len is None:
                max_q_len = geometry.max_q_len
            if max_kv_len is None:
                max_kv_len = geometry.max_kv_len
        softmax_scale = resolve_scale(
            softmax_scale, state.softmax_scale, "softmax_scale", geometry.device
        )
        validate_context_softmax_scale(softmax_scale)
        output_scale = resolve_scale(
            output_scale, state.output_scale, "output_scale", geometry.device
        )

        if out is None:
            out = torch.empty(tuple(q.shape), dtype=geometry.out_dtype, device=q.device)
        elif validate:
            out = prepare_out(out, q=q, out_dtype=geometry.out_dtype)
        lse = prepare_lse(
            lse,
            q=q,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=q,
            enabled=state.use_attention_sinks,
            validate=validate,
        )
        state.compiled(
            q,
            k_cache,
            v_cache,
            out,
            lse,
            attention_sinks,
            softmax_scale,
            output_scale,
            qo_indptr,
            block_tables,
            seq_lens_kv,
            max_q_len,
            max_kv_len,
        )
        return (out, lse) if return_lse else out


@flashinfer_experimental_api
def batch_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    mask_type: Literal["dense", "causal", "variable_window"] = "dense",
    window_left: int = -1,
    variable_window_token_starts: Optional[torch.Tensor] = None,
    variable_window_token_ends: Optional[torch.Tensor] = None,
    softmax_scale: Optional[GlobalScale] = None,
    output_scale: GlobalScale = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    vc: Optional[VCAttentionParams] = None,
    vc_config: Optional[VCAttentionConfig] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    attention_sinks: Optional[torch.Tensor] = None,
    lse_mode: Literal["base2"] = "base2",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run one-shot fixed or packed-ragged task-scheduled context attention.

    Fixed tensors use ``[B, S, H, D]`` storage. Providing both cumulative
    int32 offset tensors selects packed ``[total_tokens, H, D]`` storage.
    Equal Q/K/V head dimensions may be 128, 192, or 256. The dtype contract
    is shared by all three dimensions. Separate Q/K=192 and
    V=128 supports non-absorbed MLA; O uses V's head dimension.
    Causal masking is bottom-right aligned.  ``window_left=-1`` disables the
    left window; a positive value selects the private head-paired GQA policy
    and retains at most ``window_left + 1`` keys at each causal row, including
    when ``S_q < S_kv``.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Fixed or packed query, key, and value tensors. Q and K share one
        dtype; V matches it or is ``torch.float8_e4m3fn`` with
        ``torch.bfloat16`` Q/K. Q/K are compact; V may have nonoverlapping,
        16-byte-aligned token and batch strides with compact heads/features.
    qo_indptr, kv_indptr : torch.Tensor, optional
        Cumulative query and K/V offsets for packed-ragged input.
    mask_type : {"dense", "causal", "variable_window"}
        Attention mask mode. ``variable_window`` is supported only for
        fixed-shape inputs.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    variable_window_token_starts, variable_window_token_ends : torch.Tensor, optional
        Inclusive per-query K bounds required for ``variable_window``. Both
        must be CUDA int32 tensors shaped ``[B, Sq]`` and satisfy
        ``0 <= starts[b, q] <= ends[b, q] < Sk``.
        Ends must be nondecreasing along Q within each batch item; starts
        need not be ordered.
    softmax_scale : float or torch.Tensor, optional
        Softmax scale in natural-exponential units; defaults to the inverse
        square root of head size. Tensor scales are contiguous CUDA FP32 () or (1,)
        buffers on Q's device, with finite positive values.
    output_scale : float or torch.Tensor
        Scale applied to the attention output, independently using a host float
        or a CUDA FP32 () or (1,) buffer with finite positive values.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The fixed or packed attention output.

    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.

    ``attention_sinks`` is a compact CUDA float32 vector ``[Hq]`` shared
    across batch/query positions. Values are natural-log sink logits after
    QK scaling, with zero V contribution. Finite values and ``-inf`` are
    accepted. Validation rejects NaN/positive infinity and may synchronize.
    With validation disabled, the caller must preserve these contracts.
    Sink values may change between completed CUDA graph replays.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill cannot derive host plan bounds during CUDA graph "
            "capture; plan BatchPrefillTSWrapper before capture"
        )
    if vc_config is not None and vc is None:
        raise ValueError("vc_config requires the vc operands of this call")
    if (vc_config is not None and vc_config.repair_tiles) or (
        vc_config is None and vc is not None and vc.tile_means is None
    ):
        # The K/V rows of V repair operands include the repair tiles, so the
        # plan bounds cannot be derived from the operands alone. Without a
        # config, operands with no tile means may be repair operands.
        raise NotImplementedError(
            "batch_prefill does not run V repair operands; plan "
            "BatchPrefillTSWrapper with VCAttentionConfig(repair_tiles=...), or "
            "pass vc_config=VCAttentionConfig(demean=False) for plain operands"
        )
    if vc is not None and vc_config is None:
        vc_config = VCAttentionConfig(demean=vc.tile_means is not None)
    geometry = resolve_geometry(
        q,
        k,
        v,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_seq_len_k,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        head_dim_vo=geometry.head_dim_vo,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        out_dtype=geometry.out_dtype,
        packed_query=geometry.packed_query,
        mask_type=mask_type,
        window_left=window_left,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        vc_config=vc_config,
        use_attention_sinks=attention_sinks is not None,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
    )
    return wrapper.run(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        variable_window_token_starts=variable_window_token_starts,
        variable_window_token_ends=variable_window_token_ends,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_seq_len_k,
        out=out,
        vc=vc,
        lse=lse,
        return_lse=return_lse,
        attention_sinks=attention_sinks,
    )


@flashinfer_experimental_api
def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGED_KV_PAGE_SIZE,
    kv_layout: Literal["HND", "NHD"] = "HND",
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    softmax_scale: Optional[GlobalScale] = None,
    output_scale: GlobalScale = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    attention_sinks: Optional[torch.Tensor] = None,
    lse_mode: Literal["base2"] = "base2",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run one-shot packed-Q context attention over HND or NHD page pools.

    Q/output use ``[total_q, Hq, D]`` storage and K/V each use
    ``[num_pages, Hkv, page_size, D]`` for HND or
    ``[num_pages, page_size, Hkv, D]`` for NHD, with page size 4/8/16/32/64/128.
    ``qo_indptr`` describes Q rows, ``block_tables`` supplies one fixed
    row-strided page table, and ``seq_lens_kv`` supplies logical K/V lengths.
    Physical page indices need not be identity ordered. ``D`` may be 128 or
    256. Q and K must share one supported dtype; V matches it or is
    ``torch.float8_e4m3fn`` alongside ``torch.bfloat16`` Q/K.

    This convenience API reads request metadata on the host to derive exact
    plan bounds and is not CUDA-graph-capturable. Capture-sensitive callers
    should plan :class:`BatchPrefillPagedTSWrapper` before capture and bind the
    same metadata tensors directly in ``run``.

    Parameters
    ----------
    q : torch.Tensor
        Packed query tensor.
    k_cache : torch.Tensor
        HND key page pool.
    v_cache : torch.Tensor
        HND value page pool isomorphic to ``k_cache``.
    qo_indptr : torch.Tensor
        Cumulative packed-query offsets.
    block_tables : torch.Tensor
        Physical page IDs with row-strided shape ``[B, C]``.
    seq_lens_kv : torch.Tensor
        Logical K/V lengths with shape ``[B]``.
    page_size : int
        Number of K/V tokens stored in each page.
    kv_layout : {"HND", "NHD"}
        Layout of the separate K and V page pools.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    softmax_scale : float or torch.Tensor, optional
        Softmax scale in natural-exponential units; defaults to the inverse
        square root of head size. Tensor scales are contiguous CUDA FP32 () or (1,)
        buffers on Q's device, with finite positive values.
    output_scale : float or torch.Tensor
        Scale applied to the attention output, independently using a host float
        or a CUDA FP32 () or (1,) buffer with finite positive values.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` when ``out`` is provided,
        otherwise the query dtype.
    out : torch.Tensor, optional
        Caller-owned output tensor.

    Returns
    -------
    torch.Tensor
        The packed attention output.

    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.

    ``attention_sinks`` is a compact CUDA float32 vector ``[Hq]`` shared
    across batch/query positions. Values are natural-log sink logits after
    QK scaling, with zero V contribution. Finite values and ``-inf`` are
    accepted. Validation rejects NaN/positive infinity and may synchronize.
    With validation disabled, the caller must preserve these contracts.
    Sink values may change between completed CUDA graph replays.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    if resolved_out_dtype is None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        resolved_out_dtype = q.dtype
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "batch_prefill_with_paged_kv_cache cannot derive host plan bounds "
            "during CUDA graph capture; plan BatchPrefillPagedTSWrapper before "
            "capture"
        )
    validate_kv_layout(kv_layout)
    logical_k = k_cache.transpose(1, 2) if kv_layout == "NHD" else k_cache
    logical_v = v_cache.transpose(1, 2) if kv_layout == "NHD" else v_cache
    geometry = resolve_paged_geometry(
        q,
        logical_k,
        logical_v,
        qo_indptr=qo_indptr,
        block_tables=block_tables,
        seq_lens_kv=seq_lens_kv,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=resolved_out_dtype,
    )
    wrapper = BatchPrefillPagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        device=geometry.device,
        batch_size=geometry.batch_size,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_kv_len,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        out_dtype=geometry.out_dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        use_attention_sinks=attention_sinks is not None,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
        uniform_packed_lengths=False,
        has_q_offset=geometry.has_q_offset,
        paged_v_tail_is_zero=geometry.paged_v_tail_is_zero,
    )
    return wrapper.run(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        max_q_len=geometry.max_q_len,
        max_kv_len=geometry.max_kv_len,
        out=out,
        lse=lse,
        return_lse=return_lse,
        attention_sinks=attention_sinks,
    )


__all__ = [
    "BatchPrefillPagedTSWrapper",
    "BatchPrefillTSWrapper",
    "batch_prefill",
    "batch_prefill_with_paged_kv_cache",
]
