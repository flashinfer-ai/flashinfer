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

"""Task-scheduled paged MLA decode with a plan/run lifecycle."""

from collections.abc import Callable
from dataclasses import dataclass, replace
import copy
import math
from typing import Literal, Optional

import functools

import torch

from ...common.scales import (
    GlobalScale,
    normalize_scale,
    resolve_scale,
    scale_types,
)


from .kernel.device.kv_formats import cache_record_size

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import (
    prims_ts_decode_mla_one_shot_trace_dispatch,
    prims_ts_decode_mla_wrapper_trace_dispatch,
)


from ...common.attention_args import (
    prepare_lse,
    validate_lse_mode,
    validate_mask,
    validate_page_size,
    validate_positive_int,
    resolve_max_sequence_length,
)
from ...common.workspace import (
    align_up,
    validate_workspace_buffer,
    workspace_section_view,
)
from ...common.tensors import (
    attention_dtype_key,
    resolve_cuda_device,
    validate_runtime_device,
)


from ...compile_cache import is_attention_kernel_cached
from .launch import (
    COMPILE_OPTIONS,
    MLARuntime,
    MLAWorkspaceLayout,
    MLAWorkspaceViews,
    MLA_LATENT_DIM,
    MLA_ROPE_DIM,
    bind_mla_workspace,
    cached_mla_decode_compile_spec,
    ceil_div,
    validate_softmax_rescale_threshold_log2,
    derive_max_seq_len_q,
    compile_mla_decode,
    get_compiled_mla_decode,
    launch_mla_decode,
    make_mla_workspace_layout,
    mla_kernel_compile_signature,
    normalize_mla_kv_cache,
    prepare_mla_runtime,
    resolve_max_seq_len_q_alias,
    resolve_mla_decode_launch_spec,
    validate_mla_dims,
    validate_mla_dtype_pair,
    validate_mla_int32_extent,
    mla_cp_local_length,
    validate_mla_cp,
    validate_mla_cp_rank,
    validate_mla_max_kv_len,
    validate_mla_metadata,
    validate_mla_out,
    validate_mla_qo_indptr,
    validate_mla_query,
    validate_mla_query_head_extent,
    validate_mla_run_metadata,
)
from .sparse_plan import (
    D576_HEAD_DIM,
    D576_V_DIM,
    prepare_sparse_output_quant,
    resolve_sparse_mla_plan,
)
from .launch import has_runtime_query_lengths as has_runtime_query_lengths
from .sparse_plan import (
    resolve_sparse_d576_reload_plan as resolve_sparse_d576_reload_plan,
)


class BatchSparseMLADecodePagedTSWrapper:
    """Plan native BF16/E4M3 attention, then bind one or two source index lists.

    Q is [B,Sq,H,D] or packed [total_q,H,D], where D is 512 or 576.
    D576 adds 64 QK-only RoPE channels; V and output remain D512.
    Each source supplies int32
    [rows, capacity] storage-row indices and int32 [rows] live lengths directly
    to run(). Indices include physical page strides; -1 is masked. Both source
    lists participate in one attention distribution. The caller folds the
    shared KV descale into the effective softmax and output scales.

    ``output_quant_block_size=32`` or ``128`` fuses conjugate output RoPE on
    D[448:512] and E4M3 quantization after the final attention reduction. Q/KV
    remain prepared native tensors. Supply the local ``num_output_groups`` G
    and ``out_scale_format="ue8m0"`` (packed exponent bytes) or ``"fp32"``.
    This option does not change attention dispatch or softmax correction.

    Planning with ``assume_valid_prefix=True`` promises that every index before
    each live source length is valid (no -1 holes). Kernels can then derive
    masks from lengths and omit per-index hole checks. This does not imply
    that the physical KV rows are consecutive.
    ``run(validate=True)`` checks the promise; graph replay with validation
    disabled must preserve it. Tail entries beyond the lengths are ignored.

    A wrapper/workspace permits one in-flight run. Graphs require a completed
    warmup, stable addresses, preallocated outputs, and validate=False.
    Input/output/workspace storage must not overlap; this is an unchecked
    caller precondition. Separate wrappers/workspaces are needed per stream.

    Planning infers each scale's scalar/tensor type independently. Host scales
    are runtime FP32 arguments; tensor scales are contiguous CUDA FP32 buffers
    of shape () or (1,) on the plan device. Values and addresses are not compile
    keys. CUDA graphs capture host values or tensor addresses. Update tensor contents
    at those addresses to change scales on replay, keeping them positive/finite.
    For a fixed dispatch profile over a range of independent sparse query rows,
    plan with ``batch_size=1``, ``packed_query=True`` and
    ``max_q_len=row_capacity``. Run with ``[rows,H,D]`` queries and
    ``qo_indptr=[0,rows]`` for any ``rows <= row_capacity``.

    Q quantization and query RoPE belong to the caller. For a serving hot path,
    establish valid metadata upstream and pass ``run(validate=False)``.
    The default ``validate=True`` reads CUDA metadata values on the host and
    therefore synchronizes. Kernel-builder advisories are controlled separately
    at plan/JIT time by ``FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1``.
    """

    @flashinfer_experimental_api
    def __init__(self, workspace_buffer=None):
        self.workspace_buffer = workspace_buffer
        self.state = None
        self.workspace_size_bytes = 0

    def is_kernel_cached(self) -> bool:
        """Whether this plan's kernel has a matching on-disk export.

        Call after `plan`, including `plan(compile=False)`. This scheduling hint
        neither loads nor builds a kernel. An executable plan still requires
        `plan(compile=True)`. A process-local executable remains usable if its
        disk export is later evicted; this query only describes the disk cache.
        """
        state = self.state
        if state is None:
            raise RuntimeError("plan() must be called before is_kernel_cached()")
        spec = state["compile_spec"]
        with torch.cuda.device(state["device"]):
            return is_attention_kernel_cached(
                "mla_decode",
                "main",
                spec,
                COMPILE_OPTIONS,
                functools.partial(compile_mla_decode, spec),
            )

    def plan(
        self,
        device,
        batch_size,
        num_qo_heads,
        *,
        max_topk,
        max_extra_topk=0,
        max_q_len=1,
        packed_query=False,
        q_dtype=torch.bfloat16,
        qk_rope_head_dim=0,
        kv_cache_format="native",
        extra_kv_cache_format=None,
        kv_layout="NHD",
        use_attention_sinks=False,
        return_lse=False,
        assume_valid_prefix=False,
        output_quant_block_size=None,
        out_scale_format="ue8m0",
        num_output_groups=None,
        tuning=None,
        kv_lora_rank=512,
        softmax_scale: GlobalScale | None = None,
        output_scale: GlobalScale = 1.0,
        kv_cache_scale: GlobalScale = 1.0,
        compile: bool = True,
    ):
        """Plan D512/D576 attention over one primary and an optional extra pool.

        With ``compile=False``, resolve the finalized specialization and
        ``workspace_size_bytes`` without allocating launch buffers or compiling
        or loading a kernel. Call ``plan()`` with compilation enabled before
        ``run()``.

        ``kv_cache_format="nvfp4"`` uses the DS v4.1 D512 format: 256 packed
        E2M1 bytes followed by 32 E4M3 scales in linear block16 order. Every
        QK/V channel, including rotated channels, uses the same storage.
        Keep ``qk_rope_head_dim=0``: there is no additional QK-only tail.

        Set ``qk_rope_head_dim=64`` and ``kv_cache_format="nvfp4.fp8rope"``
        for D576: 256 packed E2M1 NoPE bytes, 64 unscaled E4M3 RoPE bytes,
        then 32 E4M3 scales, with block b stored at 320 + 8*(b%4) + b//4.
        ``kv_cache_format="fp8"`` selects D512 per-tensor E4M3 storage:
        512 bytes per token with no scale trailer. Pass its positive tensor
        descale as ``kv_cache_scale`` when planning, then optionally override it
        with the same scalar/tensor type in run(). Pair it with
        ``extra_kv_cache_format="nvfp4"`` for FP8 SWA and compressed NVFP4 KV.
        The extra format defaults to NVFP4 for an FP8 primary, otherwise to
        the primary format. D512 supports NVFP4 in both pools or FP8 primary
        with NVFP4 extra; native and mixed-RoPE formats require matching pools.
        NVFP4-containing recipes require E4M3 Q. NVFP4 block scales are
        applied in the loader before ordinary FP8 MMA, without a global
        storage scale. Per-tensor FP8 bytes feed MMA directly; its scale is
        applied to QK scores and FP32 output accumulation, avoiding another
        E4M3 rounding. Caches are uint8 records; FP8 accepts E4M3 tensors too.
        In NVFP4 bytes, even-indexed values occupy the low nibble. NVFP4
        expands to E4M3 in the loader; BF16 Q must be converted upstream for
        these recipes.

        ``max_topk`` describes arbitrary selected primary entries, not an SWA
        window. Prepared indices already include the physical page stride, so
        the core uses page size one regardless of the pools' external pages.
        Planning compiles no metadata-preparation kernel.

        Fixed queries use exactly ``batch_size * max_q_len`` rows. To reuse
        one dispatch profile over a range of independent query-row counts,
        use ``batch_size=1``, ``packed_query=True`` and
        ``max_q_len=row_capacity``. Supply ``[rows,H,D]`` Q and CUDA INT32
        ``qo_indptr=[0,rows]`` to each run, with ``rows <= row_capacity``.
        Index/length tensors and outputs use the same live row count. The
        profile is chosen for the planned capacity; each captured graph keeps
        its captured views and addresses. For example::

            wrapper.plan("cuda", 1, heads, max_topk=topk,
                         max_q_len=row_capacity, packed_query=True,
                         q_dtype=torch.float8_e4m3fn,
                         kv_cache_format="nvfp4", return_lse=True)
            # Q is already quantized/rotated; update metadata upstream.
            wrapper.run(q, cache, indices, lengths, qo_indptr=qo_indptr,
                        out=out, lse=lse, validate=False)

        Whole-tile K/V reuse supports nonpersistent split-KV schedules.
        The 1CTA persistent/CLC scheduler requires ``split_kv=1`` because its
        work queue enumerates query/head tiles without a KV-split coordinate.
        Low-level profile overrides must use a nonpersistent scheduler with
        multiple KV splits; this restriction does not apply to 2CTA profiles.

        Routine task-builder overlap and unused-routing advisories are quiet
        by default. Set ``FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1`` before a cold
        plan/JIT build to show them. This controls build diagnostics separately
        from :meth:`run`'s runtime input validation; other warnings/errors are
        retained.

        Output quantization is disabled by default. When enabled, G must divide
        the local query-head count H. Blocks of 32 with UE8M0 scales implement
        MXFP8; FP32 scales use unrounded amax/448. Both use an amax floor of
        1e-4. A block never crosses a head boundary.

        ``tuning`` (an :class:`MLAAttentionTuning`) has the meaning of
        :meth:`BatchMLADecodePagedTSWrapper.plan`; its
        ``softmax_rescale_threshold_log2`` takes 0 to 8 binades for E4M3 and
        0 to 32 for BF16. ``None`` (default) keeps each schedule's own
        behavior: some BF16 schedules keep the row anchor while a K tile
        raises it by at most 6 binades, and all others rescale on every new
        maximum. An explicit value applies to every schedule; ``0.0``
        disables the skip (slower than ``None`` on the deferred BF16
        schedules), and a positive value T also quantizes E4M3 P at
        ``448 * 2**-T`` instead of 448, with the precision cost described
        there.
        """
        (device, profile, spec, sections, byte_end, _, output_quant) = (
            resolve_sparse_mla_plan(
                device,
                batch_size,
                num_qo_heads,
                max_topk=max_topk,
                max_extra_topk=max_extra_topk,
                max_q_len=max_q_len,
                packed_query=packed_query,
                q_dtype=q_dtype,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_cache_format=kv_cache_format,
                extra_kv_cache_format=extra_kv_cache_format,
                kv_layout=kv_layout,
                use_attention_sinks=use_attention_sinks,
                return_lse=return_lse,
                assume_valid_prefix=assume_valid_prefix,
                output_quant_block_size=output_quant_block_size,
                out_scale_format=out_scale_format,
                num_output_groups=num_output_groups,
                tuning=tuning,
                kv_lora_rank=kv_lora_rank,
            )
        )
        softmax_scale = normalize_scale(
            (512 + qk_rope_head_dim) ** -0.5
            if softmax_scale is None
            else softmax_scale,
            "softmax_scale",
            device,
        )
        output_scale = normalize_scale(output_scale, "output_scale", device)
        kv_cache_scale = normalize_scale(kv_cache_scale, "kv_cache_scale", device)
        if kv_cache_format != "fp8" and (
            isinstance(kv_cache_scale, torch.Tensor) or kv_cache_scale != 1.0
        ):
            raise ValueError("kv_cache_scale is only supported for an FP8 primary pool")
        spec = replace(
            spec,
            scale_types=scale_types(softmax_scale, output_scale, kv_cache_scale),
        )
        kernel = spec.kernel
        reload_d576 = (
            profile.family == "keep" and kernel.make_config().sparse_d576_reload
        )
        family = profile.family
        splits = profile.split_kv
        workspace = self.workspace_buffer
        if workspace is not None:
            validate_workspace_buffer(workspace, device=device, required_bytes=byte_end)
        # Finalize the joint distribution in the main kernel or its existing
        # split-KV reducer. No per-source outputs or finishing launch are needed.
        if not reload_d576:
            kernel = copy.copy(kernel)
            kernel.output_quant = output_quant
            kernel.assume_valid_prefix = assume_valid_prefix
            if family == "2cta":
                kernel.fuse_sparse_epilogue = splits == 1
                kernel.fuse_sparse_reduction = splits > 1
            else:
                kernel.finalize_output = True
            kernel.direct_sparse = True
            kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
            spec = replace(
                spec,
                kernel=kernel,
                kernel_signature=mla_kernel_compile_signature(kernel),
            )
        if not compile:
            self.state = dict(device=device, compile_spec=spec, compiled=None)
            self.workspace_size_bytes = byte_end
            return

        max_rows = batch_size * max_q_len
        if workspace is None:
            workspace = torch.empty(byte_end, device=device, dtype=torch.uint8)
        buffers = {
            name: workspace_section_view(workspace, section)
            for name, section in sections.items()
        }
        compiled = get_compiled_mla_decode(spec)
        self.state = dict(
            reload_d576=reload_d576,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            max_rows=max_rows,
            packed_query=packed_query,
            q_dtype=q_dtype,
            query_dim=512 + qk_rope_head_dim,
            cache_dim=cache_record_size(kernel.kv_cache_format, qk_rope_head_dim),
            extra_cache_dim=cache_record_size(
                kernel.extra_kv_cache_format, qk_rope_head_dim
            ),
            cache_dtype=torch.uint8 if kernel.has_nvfp4_kv else q_dtype,
            kv_cache_format=kernel.kv_cache_format,
            extra_kv_cache_format=kernel.extra_kv_cache_format,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            kv_layout=kv_layout,
            use_attention_sinks=use_attention_sinks,
            return_lse=return_lse,
            output_quant=output_quant,
            buffers=buffers,
            workspace=workspace,
            compile_spec=spec,
            compiled=compiled,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            kv_cache_scale=kv_cache_scale,
            assume_valid_prefix=assume_valid_prefix,
            requested_splits=splits,
            default_cl=torch.full(
                (max_rows,), max_extra_topk, device=device, dtype=torch.int32
            ),
            dummy_indices=torch.zeros((max_rows, 1), device=device, dtype=torch.int32),
            dummy_cache=torch.zeros(
                (
                    1,
                    1,
                    cache_record_size(kernel.extra_kv_cache_format, qk_rope_head_dim),
                ),
                device=device,
                dtype=torch.uint8 if kernel.has_nvfp4_kv else q_dtype,
            ),
            default_attention_sinks=torch.full(
                (num_qo_heads,), -torch.inf, device=device, dtype=torch.float32
            ),
        )

        self.workspace_size_bytes = byte_end

    def pool(self, cache, name):
        """Validate one cache pool and expose its physical rows without repacking storage."""
        state = self.state
        dim = state["extra_cache_dim" if name == "extra_kv_cache" else "cache_dim"]
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        fmt = state[
            "extra_kv_cache_format" if name == "extra_kv_cache" else "kv_cache_format"
        ]
        if fmt == "fp8" and cache.dtype == torch.float8_e4m3fn:
            cache = cache.view(torch.uint8)
        if cache.ndim == 2:
            cache = cache.unsqueeze(1)
        if cache.ndim == 4:
            axis = 1 if state["kv_layout"] == "HND" else 2
            if cache.shape[axis] != 1:
                raise ValueError(f"{name} requires one KV head")
            cache = cache.squeeze(axis)
        if (
            cache.ndim != 3
            or cache.shape[1] <= 0
            or cache.shape[2] != dim
            or cache.shape[0] <= 0
        ):
            raise ValueError(
                f"{name} must have shape [pages,page_size,{dim}] with nonempty pages"
            )
        page_size = cache.shape[1]
        if cache.device != state["device"] or cache.dtype != state["cache_dtype"]:
            raise ValueError(f"{name} must match the planned device and dtype")
        if (
            cache.stride(-1) != 1
            or cache.stride(1) != dim
            or cache.stride(0) < page_size * dim
            or cache.stride(0) % dim
            or cache.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} requires aligned compact rows and a page stride divisible by {dim} elements"
            )
        row_stride = cache.stride(0) // dim
        rows = (cache.shape[0] - 1) * row_stride + page_size
        if rows >= 0x7FFFFFFF:
            raise ValueError(f"{name} exceeds signed gather coordinate range")
        return cache.as_strided((rows, 1, dim), (dim, dim, 1))

    @flashinfer_experimental_api
    def run(
        self,
        query,
        kv_cache,
        indices,
        lengths,
        extra_kv_cache=None,
        *,
        extra_indices=None,
        extra_lengths=None,
        qo_indptr=None,
        softmax_scale: GlobalScale | None = None,
        output_scale: GlobalScale | None = None,
        kv_cache_scale: GlobalScale | None = None,
        attention_sinks=None,
        out=None,
        lse=None,
        token_positions=None,
        cos_sin_cache=None,
        out_scale=None,
        validate=True,
    ):
        """Attend to caller-selected storage rows in one or two native KV pools.

        ``indices`` and ``extra_indices`` are CUDA INT32 [T,K] tensors, with
        T equal to the query token count and K equal to the corresponding
        planned max_topk/max_extra_topk. The last dimension is contiguous;
        row strides may include padding. Each index addresses its own pool's
        flattened storage, including physical page strides. The caller performs
        logical-token/page-table conversion and applies causality upstream.
        ``lengths`` and ``extra_lengths`` are contiguous CUDA INT32 [T]. Only
        entries before each length are considered; -1 entries are masked holes.
        Omit both extra tensors when no extra pool was planned. Input widths
        and live lengths need no 128-entry padding: the kernel aligns source
        execution spans internally and bounds all source reads.

        ``validate=True`` (default) checks CUDA metadata contents using host
        reads, which synchronize the device. Serving should establish index,
        length, offset and position bounds upstream and use ``validate=False``
        on the hot path and during CUDA graph capture. Shape, dtype, device,
        stride and alignment checks still run without reading CUDA values.
        Q must already use the planned dtype and include the required query
        RoPE; run() performs neither Q quantization nor query rotation.

        ``softmax_scale`` and ``output_scale`` independently accept positive finite
        host floats or contiguous CUDA FP32 tensors of shape () or (1,). None inherits
        the plan default; overrides must match its scalar/tensor type. They are
        already combined:
        softmax_scale = attention_scale * q_descale * kv_descale;
        output_scale = output_multiplier * kv_descale. Both pools share that
        KV descale. These multiply QK scores and the weighted stored V values,
        respectively; sink logits are not multiplied by softmax_scale. The
        output multiplier applies before fused RoPE/quantization. ``out_scale``
        instead holds the generated per-block quantization scales.
        Combine scales before each launch on CPU or GPU. Their values are not
        compilation keys. CUDA graphs capture float values or tensor addresses;
        updating tensor contents at captured addresses changes replay scales
        without a host readback. Tensor contents must stay positive and finite.

        For ``kv_cache_format="fp8"``, ``kv_cache_scale`` is the independent
        per-tensor descale of the primary pool. It accepts either scale type;
        leave it at the host default 1.0
        for other formats. The extra pool is NVFP4 and has no tensor descale.
        The primary scale is applied internally to both K and V; do not also
        fold it into softmax_scale/output_scale.

        Caller-owned inputs and plan workspace must outlive graph replay,
        including attention_sinks, and must not alias writable output
        or workspace storage. Q and native KV are used directly; flattening
        padded pages only creates a tensor view. Required attention reductions
        remain included. Caller-owned LSE must be contiguous FP32 and 16-byte
        aligned, including with validate=False.

        With output quantization enabled, ``token_positions`` is contiguous
        INT32 with the query's batch/token prefix; ``cos_sin_cache`` is aligned
        contiguous FP32[P,64] containing forward [cos(32), sin(32)] rows. Use
        absolute query positions and the same layer-specific table as Q RoPE.
        Positions must be in [0,P); validate=False leaves this to the caller.

        Return (out, out_scale), or (out, out_scale, lse) with return_lse=True.
        Out is the [T,G,C] view (usual strides (C,T*C,1)) of a 16-byte aligned
        contiguous group-major E4M3[G,T,C] tensor, where T is total query
        tokens and C=(H/G)*512, in natural head/channel order. Equivalent
        singleton-axis strides are accepted. A live prefix may retain a
        larger [G,T_capacity,C] allocation; only its live T rows are written. Pass
        ``torch.empty((G,T,C), dtype=torch.float8_e4m3fn).transpose(0,1)`` as
        ``out`` to preallocate. For K=output_quant_block_size
        and N=C/K, FP32 scales have logical shape [T,G,N]; UE8M0 scales have
        shape [T,G,N/4], INT32 packing four block exponent bytes, first in the
        least significant byte (scale=2**(byte-127)). Allocate scale backing as
        16-byte aligned [G,columns,pad4(T)] and use backing.permute(2,0,1)[:T],
        with strides (1,columns*pad4(T),pad4(T)). Padding is not written. The scalar
        ``output_scale`` multiplier remains applied before quantization.

        Preallocate out/out_scale/lse and warm up before graph capture. With
        fusion disabled, the existing BF16 output and return contract apply.
        """
        state = self.state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if state["compiled"] is None:
            raise RuntimeError("plan(compile=True) must be called before run()")
        validate_mla_query(
            query,
            packed_query=state["packed_query"],
            device=state["device"],
            batch_size=state["batch_size"],
            num_qo_heads=state["num_qo_heads"],
            max_q_len=state["max_q_len"],
            q_dtype=state["q_dtype"],
            query_dim=state["query_dim"],
        )
        prefix = tuple(query.shape[:-2])
        rows = math.prod(prefix)
        if state["packed_query"]:
            validate_mla_qo_indptr(
                qo_indptr, device=state["device"], batch_size=state["batch_size"]
            )
            if validate:
                maximum, total, _ = derive_max_seq_len_q(
                    qo_indptr, batch_size=state["batch_size"]
                )
                if total != rows or maximum > state["max_q_len"]:
                    raise ValueError("invalid packed query offsets")
        elif qo_indptr is not None:
            raise ValueError("fixed queries do not accept qo_indptr")

        primary = self.pool(kv_cache, "kv_cache")
        if extra_kv_cache is None:
            if (
                state["max_extra_topk"]
                or extra_indices is not None
                or extra_lengths is not None
            ):
                raise ValueError("extra pool and indices are required by the plan")
            extra = state["dummy_cache"]
        else:
            extra = self.pool(extra_kv_cache, "extra_kv_cache")

        si, sl = self.source_metadata(
            indices,
            lengths,
            state["max_topk"],
            primary.shape[0],
            "primary",
            rows=rows,
            validate=validate,
        )
        ci, cl = self.source_metadata(
            extra_indices,
            extra_lengths,
            state["max_extra_topk"],
            extra.shape[0],
            "extra",
            rows=rows,
            validate=validate,
        )
        if state["use_attention_sinks"] != (attention_sinks is not None):
            raise ValueError("attention_sinks presence must match the plan")
        if attention_sinks is None:
            attention_sinks = state["default_attention_sinks"]
        if (
            attention_sinks.shape != (state["num_qo_heads"],)
            or attention_sinks.dtype != torch.float32
            or attention_sinks.device != state["device"]
            or not attention_sinks.is_contiguous()
        ):
            raise ValueError("attention_sinks must be contiguous CUDA FP32[H]")
        if validate and torch.isnan(attention_sinks).any().item():
            raise ValueError("attention_sinks must not contain NaN")
        softmax_scale = resolve_scale(
            softmax_scale, state["softmax_scale"], "softmax_scale", state["device"]
        )
        output_scale = resolve_scale(
            output_scale, state["output_scale"], "output_scale", state["device"]
        )
        kv_cache_scale = resolve_scale(
            kv_cache_scale, state["kv_cache_scale"], "kv_cache_scale", state["device"]
        )
        if state["kv_cache_format"] != "fp8" and (
            isinstance(kv_cache_scale, torch.Tensor) or kv_cache_scale != 1.0
        ):
            raise ValueError("kv_cache_scale is only supported for an FP8 primary pool")
        quant = state["output_quant"]
        output_quant_args = None
        if quant is not None:
            out, out_scale, output_quant_args = prepare_sparse_output_quant(
                quant, query, token_positions, cos_sin_cache, out, out_scale, validate
            )
        elif any(v is not None for v in (token_positions, cos_sin_cache, out_scale)):
            raise ValueError(
                "plan output quantization before passing RoPE/scale tensors"
            )
        elif out is None:
            out = torch.empty(
                (*prefix, state["num_qo_heads"], 512),
                device=state["device"],
                dtype=torch.bfloat16,
            )
        if quant is None:
            validate_mla_out(
                out,
                device=state["device"],
                batch_size=state["batch_size"],
                num_qo_heads=state["num_qo_heads"],
                max_q_len=state["max_q_len"],
                packed_query=state["packed_query"],
                total_q=rows if state["packed_query"] else None,
                out_dtype=torch.bfloat16,
            )
        if lse is None:
            # A returned result must survive the next call on this wrapper.
            # Use plan-owned scratch only when the LSE is not exposed.
            lse = (
                torch.empty(
                    (*prefix, state["num_qo_heads"]),
                    device=state["device"],
                    dtype=torch.float32,
                )
                if state["return_lse"]
                else state["buffers"]["public_lse"][:rows].view(
                    *prefix, state["num_qo_heads"]
                )
            )
        if (
            lse.shape != (*prefix, state["num_qo_heads"])
            or lse.dtype != torch.float32
            or lse.device != state["device"]
            or not lse.is_contiguous()
            or lse.data_ptr() % 16
        ):
            raise ValueError(
                "lse must be contiguous, 16-byte-aligned FP32 with one value per query/head"
            )
        sparse_inputs = (
            si,
            ci,
            sl,
            cl,
            attention_sinks,
            kv_cache_scale,
        )
        if rows and state["reload_d576"]:
            core = state["buffers"]["core"]
            # One token per batch row through the 1CTA page-table ABI; 128 heads
            # are two physical query positions of 64 heads.
            phys = (rows, state["num_qo_heads"] // 64, 64)
            with torch.cuda.device(state["device"]):
                launch_mla_decode(
                    MLARuntime(
                        query.view(*phys, D576_HEAD_DIM),
                        primary,
                        out.view(*phys, D576_V_DIM),
                        primary.shape[0],
                        softmax_scale,
                        output_scale,
                    ),
                    block_tables=si,
                    seq_lens_kv=sl,
                    qo_indptr=None,
                    packed_query=False,
                    kv_lora_rank=512,
                    split_kv=state["requested_splits"],
                    workspace=MLAWorkspaceViews(
                        core if core.numel() else None, lse.view(*phys)
                    ),
                    compiled=state["compiled"],
                    max_q_len=phys[1],
                )
        elif rows:
            buffers = state["buffers"]
            # Quantized stores use output_quant_args, including group/bucket
            # strides. The core O view supplies compact launch geometry only.
            core_out = (
                out.view(rows, 1, state["num_qo_heads"], 512)
                if quant is None
                else output_quant_args[0][: rows * state["num_qo_heads"] * 512].view(
                    rows, 1, state["num_qo_heads"], 512
                )
            )
            launch_mla_decode(
                MLARuntime(
                    query.view(rows, 1, state["num_qo_heads"], state["query_dim"]),
                    primary,
                    core_out,
                    primary.shape[0],
                    softmax_scale,
                    output_scale,
                    extra_cache=extra,
                    sparse_inputs=sparse_inputs,
                    output_quant_args=output_quant_args,
                ),
                block_tables=si,
                seq_lens_kv=sl,
                qo_indptr=None,
                packed_query=False,
                kv_lora_rank=512,
                split_kv=state["requested_splits"],
                workspace=MLAWorkspaceViews(
                    buffers["core"] if buffers["core"].numel() else None,
                    lse.view(rows, 1, state["num_qo_heads"]),
                ),
                compiled=state["compiled"],
                max_q_len=1,
            )
        if quant is not None:
            return (out, out_scale, lse) if state["return_lse"] else (out, out_scale)
        return (out, lse) if state["return_lse"] else out

    def source_metadata(
        self, indices, lengths, capacity, tokens, name, *, rows, validate
    ):
        state = self.state
        if capacity == 0:
            if indices is not None or lengths is not None:
                raise ValueError(f"{name} metadata is not expected by the plan")
            return state["dummy_indices"][:rows], state["default_cl"][:rows]
        if not isinstance(indices, torch.Tensor) or (
            indices.shape != (rows, capacity)
            or indices.dtype != torch.int32
            or indices.device != state["device"]
            or indices.stride(-1) != 1
        ):
            raise ValueError(f"invalid {name} indices shape/dtype/device/stride")
        if not isinstance(lengths, torch.Tensor) or (
            lengths.shape != (rows,)
            or lengths.dtype != torch.int32
            or lengths.device != state["device"]
            or not lengths.is_contiguous()
        ):
            raise ValueError(f"invalid {name} lengths")
        if validate:
            if ((lengths < 0) | (lengths > capacity)).any().item():
                raise ValueError(f"{name} length exceeds capacity")
            active = (
                torch.arange(capacity, device=state["device"])[None, :]
                < lengths[:, None]
            )
            if (active & ((indices < -1) | (indices >= tokens))).any().item():
                raise ValueError(f"{name} active index outside pool")
            if state["assume_valid_prefix"] and (active & (indices < 0)).any().item():
                raise ValueError(
                    f"{name} active prefix contains a hole with assume_valid_prefix=True"
                )
        return indices, lengths


@dataclass(frozen=True)
class MLAAttentionTuning:
    """Compile-time tuning of one PrimTS MLA attention plan.

    It applies to every query length the MLA wrappers serve, context
    (multi-token Q) as well as generation. ``softmax_rescale_threshold_log2``
    is in binades. ``None`` keeps the entry point's default: exact rescale for
    dense, and each schedule's existing deferral for sparse. An explicit
    value, including ``0.0``, is compiled into the kernel; see
    :meth:`BatchMLADecodePagedTSWrapper.plan`.
    """

    softmax_rescale_threshold_log2: Optional[float] = None


class BatchMLADecodePagedTSWrapper:
    """Reusable paged MLA with eager planning and automatic runtime splits.

    Plans select cached kernels from explicit problem bounds and retain scratch.
    Each run supplies live tensors and request metadata. Replanning invalidates
    graphs; concurrent execution requires independent wrappers/workspaces."""

    @flashinfer_experimental_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled paged-MLA wrapper."""
        self.plan_state: Optional[MLADecodePlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        device: int | str | torch.device,
        batch_size: int,
        num_qo_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_kv_len: int,
        *,
        max_q_len: int,
        packed_query: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        out_dtype: torch.dtype,
        mask_type: Literal["dense", "causal"] = "causal",
        softmax_scale: GlobalScale = 1.0,
        output_scale: GlobalScale = 1.0,
        lse_mode: Literal["none", "base2"] = "none",
        cp_world: int = 1,
        cp_interleave_size: int = 1,
        zero_empty_rows: bool = True,
        workspace_buffer: Optional[torch.Tensor] = None,
        validate: bool = True,
        tuning: Optional[MLAAttentionTuning] = None,
    ) -> None:
        """Compile one static MLA shape and bind its reusable workspace.

        Planning consumes geometry, capacity, dtype, storage-mode inputs, and
        default scale operands. Each scale independently accepts a positive host
        float or a contiguous CUDA FP32 tensor of shape () or (1,) on the plan
        device. Types select the compiled signature; values and addresses do not.
        Tensor contents must remain positive and finite. No scale data is read
        on CPU.
        Request metadata belongs exclusively to
        :meth:`run` and is never retained by the wrapper. A successful re-plan
        atomically replaces the previous immutable state; a failed re-plan
        leaves that state usable.

        ``packed_query=False`` selects fixed ``[B, SQ, H, 576]`` query storage,
        where ``SQ`` is exactly ``max_q_len``. ``packed_query=True`` selects
        ``[total_q, H, 576]`` storage with per-run cumulative offsets supplied to
        every run. ``max_q_len`` is then the per-request capacity.
        Individual packed requests may be empty; an all-empty run returns its
        empty output without dispatching a GPU kernel.

        The request count B remains fixed in both modes. Fixed mode also fixes
        the total query rows B*SQ; packed mode accepts changing per-request
        lengths within the planned capacity. Sparse independent rows can use
        the capacity-plan recipe on :class:`BatchSparseMLADecodePagedTSWrapper`.
        Q quantization and RoPE are caller-owned preparation; the kernel uses
        Q in ``q_dtype`` directly.

        If ``workspace_buffer`` is omitted, the plan allocates private scratch.
        A workspace is mutable and exclusive to one in-flight launch or graph
        replay. Warm the plan before graph capture, and call ``run`` with
        ``validate=False`` inside compiled or captured regions.

        Equivalent plans/layers reuse CPU policy and compiled-kernel caches.
        The active split count remains a runtime attention/reduction operand.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the plan will execute.
        batch_size : int
            Exact runtime request count.
        num_qo_heads, kv_lora_rank, qk_rope_head_dim, page_size : int
            Static MLA head geometry and K/V page size.
        max_kv_len, max_q_len : int
            Static per-request local K/V and Q capacities. K/V capacity is
            rank-local even when runtime sequence lengths are global.
        packed_query : bool
            Select packed rather than fixed query storage.
        q_dtype, kv_dtype, out_dtype : torch.dtype
            Query, K/V, and output dtypes used to compile the plan.
        mask_type : {"dense", "causal"}
            Attention mask mode.
        cp_world, cp_interleave_size : int
            Compile-time context-parallel rank count and token interleave size,
            both defaulting to one. Global token ``t`` belongs to rank
            ``(t // cp_interleave_size) % cp_world``. Each local cache stores
            its owned tokens in global order; page tables address that cache.
            Every run supplies its rank and global logical request lengths.
        zero_empty_rows : bool
            Initialize rows with no locally visible KV to zero output and
            negative-infinity LSE. Defaults to ``True``; ignored for cp_world=1.
            ``False`` omits the initialization launch but keeps CP masking.
            Empty-row results are then unspecified: sanitize them before
            consumption or exclude them from merging.
            This selects the compiled launch program, not workspace size.
        workspace_buffer : torch.Tensor, optional
            Caller-owned contiguous int8 or uint8 scratch on ``device``. It
            must be 32-byte aligned and large enough for the selected plan.
            Size it with :func:`get_prims_ts_batch_mla_decode_workspace_size`
            using matching geometry, dtypes, ``cp_world``, and
            ``cp_interleave_size``, including when ``validate=False``.
            When omitted, planning allocates the buffer. The retained buffer
            is exclusive to one in-flight launch or graph replay.
        validate : bool
            Validate static geometry and caller scratch. Defaults to ``True``.
            Disable only for previously validated inputs and warmed topology.
        tuning : MLAAttentionTuning, optional
            Compile-time tuning; ``None`` keeps every default.
            ``tuning.softmax_rescale_threshold_log2`` is the opt-in skip of
            the online-softmax O rescale (correction), in binades: 0 to 8 for
            FP8 QKV, 0 to 32 for BF16 QKV. The name and the FP8 P scale
            follow the Prims-TS FMHA decode option of the same name; the
            bounds follow the trtllm-gen FMHA ``skipCorrThreshold``.
            ``None`` or ``0.0`` rescales on every new row maximum. A positive
            value T keeps a row's softmax anchor while a K tile raises the row
            maximum by at most T binades, which skips that tile's rescale of
            the accumulated output. Probabilities then reach ``2**T``, so FP8
            P is quantized at ``448 * 2**-T`` instead of 448 (1.75 at T = 8,
            the trtllm-gen scale). E4M3 P then loses precision from about
            ``10.3 - 0.69 * T`` nats below the anchor and flushes to zero
            below about ``13 - 0.69 * T`` nats (4.7 and 7.5 at T = 8); the
            FP32 row sum keeps the exact values, so the tail's weight is
            biased low. The loss covers the KV range
            that shares the anchor's split, which is the whole row when
            split-KV is not used (typically large batches). Rows whose largest
            scores come first, such as attention sinks, with a long
            low-probability tail are the most exposed, so check model accuracy
            when enabling it for FP8. BF16 P is not scaled. The value is
            compiled into the kernel, so each distinct value builds its own
            kernel. It is range-checked even when ``validate`` is False.
        """

        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        validate_mla_cp(cp_world, cp_interleave_size)
        if not isinstance(zero_empty_rows, bool):
            raise TypeError("zero_empty_rows must be a bool")
        if validate:
            if not isinstance(packed_query, bool):
                raise TypeError("packed_query must be a bool")
            validate_mask(mask_type)
            batch_size = validate_positive_int(batch_size, "batch_size")
            validate_mla_int32_extent(batch_size, "batch_size")
            num_qo_heads = validate_positive_int(num_qo_heads, "num_qo_heads")
            validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
            page_size = validate_page_size(page_size)
            max_kv_len = validate_mla_max_kv_len(max_kv_len, "max_kv_len")
            max_q_len = validate_positive_int(max_q_len, "max_q_len")
            validate_mla_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_q_len=max_q_len,
            )
            validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
        # Range-check the threshold even when validate is False: it selects a
        # compiled kernel, and past the per-dtype bound FP8 P (scaled by
        # 448 * 2**-T) would lose most of the row to E4M3 subnormals.
        threshold = tuning_rescale_threshold(tuning)
        softmax_rescale_threshold_log2 = validate_softmax_rescale_threshold_log2(
            0.0 if threshold is None else threshold, q_dtype=q_dtype
        )

        device, device_index = resolve_cuda_device(device)
        softmax_scale = normalize_scale(softmax_scale, "softmax_scale", device)
        output_scale = normalize_scale(output_scale, "output_scale", device)
        if validate:
            validate_runtime_device(device)

        required_page_columns = ceil_div(max_kv_len, page_size)

        spec_key = (
            device_index,
            batch_size,
            num_qo_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_kv_len,
            attention_dtype_key(q_dtype),
            attention_dtype_key(kv_dtype),
            attention_dtype_key(out_dtype),
            mask_type,
            max_q_len,
        )
        spec = resolve_mla_decode_launch_spec(
            *spec_key,
            softmax_rescale_threshold_log2,
            cp_world,
            cp_interleave_size,
        )
        compile_spec = cached_mla_decode_compile_spec(
            spec,
            device_index=device_index,
            num_qo_heads=num_qo_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype_key=attention_dtype_key(q_dtype),
            out_dtype_key=attention_dtype_key(out_dtype),
            max_q_len=max_q_len,
            packed_query=packed_query,
            lse_mode=lse_mode,
            runtime_kv_lengths=True,
            write_final_lse=lse_mode != "none",
            scale_types=(*scale_types(softmax_scale, output_scale), False),
            cp_world=cp_world,
            cp_interleave_size=cp_interleave_size,
            zero_empty_rows=zero_empty_rows,
        )
        policy = spec.policy
        workspace_layout = make_mla_workspace_layout(
            spec.kernel_workspace_bytes,
            batch_size,
            num_qo_heads,
            max_q_len,
            include_lse=False,
        )
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes, device=device, dtype=torch.int8
            )
        elif validate:
            validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
        workspace = bind_mla_workspace(workspace_buffer, workspace_layout)
        compiled = get_compiled_mla_decode(compile_spec)

        # Publish only after validation, compilation, allocation, and binding
        # succeed, so a failed re-plan leaves the previous plan usable.
        self.plan_state = MLADecodePlanState(
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            out_dtype=out_dtype,
            mask_type=mask_type,
            max_kv_len=max_kv_len,
            required_page_columns=required_page_columns,
            workspace_buffer=workspace_buffer,
            workspace_layout=workspace_layout,
            workspace_views=workspace,
            compiled=compiled,
            policy=policy,
            split_kv=spec.split_kv,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            lse_mode=lse_mode,
            softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
            cp_world=cp_world,
            cp_interleave_size=cp_interleave_size,
            zero_empty_rows=zero_empty_rows,
        )

    @flashinfer_experimental_api(trace=prims_ts_decode_mla_wrapper_trace_dispatch)
    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        qo_indptr: Optional[torch.Tensor] = None,
        max_q_len: int | None = None,
        cp_rank: int = 0,
        softmax_scale: GlobalScale | None = None,
        output_scale: GlobalScale | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch MLA using live query/cache, page tables and KV lengths.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; local K/V lengths stay within its capacity.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.

        seq_lens_kv contains GLOBAL logical request lengths, including the
        current Q tokens, in a contiguous CUDA int32 vector. cp_rank is a host
        integer in [0, cp_world), passed as a runtime scalar (captured by value
        in a CUDA graph), not a compile key. With CP interleave I and world C,
        a global prefix n maps to n//(I*C)*I + clamp(n%(I*C)-cp_rank*I, 0, I)
        local tokens. Dense rows use n=seq_lens_kv[b]; causal row i uses
        n=seq_lens_kv[b]-actual_Q+i+1, where packed actual_Q is the qo_indptr
        delta, not max_q_len. Local cache/pages and max_kv_len stay rank-local.
        For cp_world>1, zero local prefixes write zero O and negative-infinity
        LSE unless zero_empty_rows=False was planned; that opt-out leaves empty
        rows unspecified, not preserved. With cp_world=1, nonpositive request
        lengths retain the inactive-request behavior. validate=False performs
        no metadata readback; the caller must preserve all value constraints.

        Fixed queries use [B, Sq, H, 576]; packed queries use [total_q, H, 576] and
        qo_indptr with nonnegative per-request deltas. All-empty packed Q is a no-op.
        softmax_scale and output_scale independently accept host floats or CUDA
        FP32 tensors of shape () or (1,). None inherits the plan default; overrides
        must match its scalar/tensor type. Tensor contents must remain positive and
        finite, and can change at stable addresses between graph replays. Host
        values are captured by value. The dense MLA interface has no sinks.
        max_q_len covers the live Q lengths within plan capacity. Validation
        infers an omitted bound from existing metadata; trusted packed calls
        default to plan capacity without a readback. This bound is runtime."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self.plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        softmax_scale = resolve_scale(
            softmax_scale, state.softmax_scale, "softmax_scale", state.device
        )
        output_scale = resolve_scale(
            output_scale, state.output_scale, "output_scale", state.device
        )
        validate_mla_cp_rank(cp_rank, state.cp_world)
        runtime_qo_indptr = qo_indptr if state.packed_query else None
        runtime = prepare_mla_runtime(
            query,
            kv_cache,
            device=state.device,
            batch_size=state.batch_size,
            num_qo_heads=state.num_qo_heads,
            max_q_len=state.max_q_len,
            packed_query=state.packed_query,
            qo_indptr=runtime_qo_indptr,
            page_size=state.page_size,
            q_dtype=state.q_dtype,
            kv_dtype=state.kv_dtype,
            out_dtype=state.out_dtype,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            out=out,
            validate=validate,
        )
        if validate:
            actual_q = validate_mla_run_metadata(
                runtime,
                block_tables,
                seq_lens_kv,
                qo_indptr,
                device=state.device,
                batch_size=state.batch_size,
                required_page_columns=state.required_page_columns,
                packed_query=state.packed_query,
                max_q_len=state.max_q_len,
                max_kv_len=state.max_kv_len,
                page_size=state.page_size,
                mask_type=state.mask_type,
                cp_world=state.cp_world,
                cp_interleave_size=state.cp_interleave_size,
                cp_rank=cp_rank,
            )
            max_q_len = resolve_max_sequence_length(
                max_q_len, actual_q, "max_q_len", state.max_q_len
            )
        elif not state.packed_query:
            max_q_len = int(query.shape[1])
        elif max_q_len is None:
            max_q_len = state.max_q_len
        lse = prepare_lse(
            lse,
            q=query,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        result = launch_mla_decode(
            runtime,
            block_tables=block_tables,
            seq_lens_kv=seq_lens_kv,
            qo_indptr=runtime_qo_indptr,
            packed_query=state.packed_query,
            kv_lora_rank=state.kv_lora_rank,
            split_kv=state.split_kv,
            workspace=state.workspace_views,
            compiled=state.compiled,
            lse=lse,
            max_q_len=max_q_len,
            cp_rank=cp_rank,
        )
        return (result, lse) if return_lse else result


@dataclass(frozen=True)
class MLADecodePlanState:
    """Immutable compile and workspace state for one reusable MLA plan."""

    device: torch.device
    batch_size: int
    num_qo_heads: int
    max_q_len: int
    packed_query: bool
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: Literal["dense", "causal"]
    max_kv_len: int
    required_page_columns: int
    workspace_buffer: torch.Tensor
    workspace_layout: MLAWorkspaceLayout
    workspace_views: MLAWorkspaceViews
    compiled: Callable[..., object]
    policy: tuple[tuple[str, object], ...]
    split_kv: int
    softmax_rescale_threshold_log2: float = 0.0
    lse_mode: str = "none"
    softmax_scale: GlobalScale = 1.0
    output_scale: GlobalScale = 1.0
    cp_world: int = 1
    cp_interleave_size: int = 1
    zero_empty_rows: bool = True


@flashinfer_experimental_api(trace=prims_ts_decode_mla_one_shot_trace_dispatch)
def batch_mla_decode_with_paged_kv_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    kv_lora_rank: int = MLA_LATENT_DIM,
    qk_rope_head_dim: int = MLA_ROPE_DIM,
    mask_type: Literal["dense", "causal"] = "causal",
    max_kv_len: Optional[int] = None,
    softmax_scale: GlobalScale = 1.0,
    output_scale: GlobalScale = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_mode: Literal["base2"] = "base2",
    out_dtype: torch.dtype = torch.bfloat16,
    workspace_buffer: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_interleave_size: int = 1,
    cp_rank: int = 0,
    zero_empty_rows: bool = True,
    validate: bool = True,
    tuning: Optional[MLAAttentionTuning] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One-shot convenience wrapper for fixed or packed-query MLA decode.

    Without caller scratch, this helper may read ``seq_lens_kv`` on the host
    to derive the K/V bound and constructs a temporary wrapper outside capture.
    Packed Q requires a caller-provided ``max_q_len`` in either mode.
    See Notes for the explicit capture-safe mode.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    block_tables : torch.Tensor
        Dense physical-page table for each request. Rows must be inner
        contiguous and non-overlapping, but may have padding between them.
    seq_lens_kv : torch.Tensor
        Global logical K/V sequence lengths, including current Q tokens.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_q_len : int, optional
        Per-request packed-query length capacity. Required for packed Q;
        it is never inferred from ``qo_indptr`` and may exceed the maximum
        request length.
        For fixed Q it defaults to the query's sequence extent and, when
        provided, must equal that extent.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    max_kv_len : int, optional
        Static rank-local K/V length bound; defaults to the maximum local
        length derived from global metadata for cp_rank. An all-empty CP rank
        uses capacity one when this bound is omitted.
    cp_world, cp_interleave_size, cp_rank : int
        Context-parallel rank count, token interleave size, and runtime rank.
        Defaults are 1, 1, and 0. See :meth:`BatchMLADecodePagedTSWrapper.run`.
    zero_empty_rows : bool
        Initialize CP empty rows to neutral output/LSE by default.
        ``False`` omits this launch and leaves empty-row results unspecified.
        Ignored for cp_world=1; does not change workspace size.
    softmax_scale, output_scale : float or torch.Tensor
        QK and value/output scaling factors.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype
        Output dtype.
    workspace_buffer : torch.Tensor, optional
        Caller-owned byte scratch, exclusive to one in-flight launch/graph.
        Size it with :func:`get_prims_ts_batch_mla_decode_workspace_size`
        using matching geometry, dtypes, ``cp_world``, and
        ``cp_interleave_size``, including when ``validate=False``.
    validate : bool
        Validate tensors and metadata values (may synchronize), default True.
        False trusts the caller and requires workspace and explicit bounds;
        skips tensor and metadata validation. Invalid inputs have undefined behavior.
    tuning : MLAAttentionTuning, optional
        Compile-time tuning; see :meth:`BatchMLADecodePagedTSWrapper.plan`.
        ``None`` (default) keeps the exact rescale. A positive
        ``softmax_rescale_threshold_log2`` T quantizes FP8 P at
        ``448 * 2**-T`` instead of 448, which costs E4M3 precision for
        probabilities far below the anchor.

    Returns
    -------
    torch.Tensor
        The fixed or packed MLA attention output.
    Notes
    -----
    Both owned and caller-provided scratch use the same wrapper plan/run path.
    For CUDA Graph capture, supply ``workspace_buffer``, ``max_kv_len``,
    ``out``, and ``validate=False``; packed Q also needs an explicit
    ``max_q_len``.
    Warm this exact topology outside capture first. Retain stable tensor storage
    and mutate metadata only between completed launches/replays. Local lengths
    derived for cp_rank must be within the static bound. With cp_world=1,
    nonpositive lengths mark inactive requests whose output is untouched;
    CP global lengths must be nonnegative. Active page IDs must index the
    local cache, and causal Q lengths must not exceed positive global KV lengths.
    Packed offsets must start at zero, end at the query token count, and have
    nonnegative deltas within the Q bound. Scratch/output must not alias
    any inputs or each other. No metadata is copied to the host on the trusted
    explicit path. Kernel policy, necessary control resets, and output layout
    are unchanged. Missing output may be allocated only outside capture.


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
    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be a bool")
    validate_mla_cp(cp_world, cp_interleave_size)
    validate_mla_cp_rank(cp_rank, cp_world)
    if validate or workspace_buffer is None or out is None:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture requires workspace_buffer, max_kv_len, "
                "out, validate=False, and explicit packed-Q bounds; warm up first"
            )
    if not validate:
        if workspace_buffer is None:
            raise ValueError("validate=False requires workspace_buffer")
        if max_kv_len is None:
            raise ValueError("validate=False requires max_kv_len")
        if qo_indptr is not None and max_q_len is None:
            raise ValueError("validate=False requires max_q_len for packed Q")

    packed_query = qo_indptr is not None
    if packed_query and max_q_len is None:
        raise ValueError("packed Q requires max_q_len")
    if validate:
        validate_mla_query(query, packed_query=packed_query)
        metadata_device, batch_size, _ = validate_mla_metadata(
            block_tables, seq_lens_kv
        )
        if metadata_device != query.device:
            raise ValueError(
                f"MLA metadata must be on {query.device}, got {metadata_device}"
            )
        normalized_cache, _, page_size = normalize_mla_kv_cache(
            kv_cache, expected_device=query.device
        )
        if packed_query:
            validate_mla_qo_indptr(
                qo_indptr, device=query.device, batch_size=batch_size
            )
        elif max_q_len is not None and max_q_len != int(query.shape[1]):
            raise ValueError(
                "fixed query length must equal max_q_len: "
                f"got SQ={query.shape[1]} and max_q_len={max_q_len}"
            )
    else:
        batch_size = int(seq_lens_kv.shape[0])
        normalized_cache = kv_cache[:, 0] if kv_cache.ndim == 4 else kv_cache
        page_size = int(normalized_cache.shape[1])
    num_qo_heads = int(query.shape[-2])
    if not packed_query and max_q_len is None:
        max_q_len = int(query.shape[1])
    if max_kv_len is None:
        max_kv_len = mla_cp_local_length(
            int(seq_lens_kv.max().item()), cp_world, cp_interleave_size, cp_rank
        )
        if cp_world > 1:
            max_kv_len = max(1, max_kv_len)
    assert max_q_len is not None
    if validate and out is not None:
        validate_mla_out(
            out,
            device=query.device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            total_q=int(query.shape[0]) if packed_query else None,
            out_dtype=out_dtype,
        )
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        query.device,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        packed_query=packed_query,
        q_dtype=query.dtype,
        kv_dtype=normalized_cache.dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        workspace_buffer=workspace_buffer,
        cp_world=cp_world,
        cp_interleave_size=cp_interleave_size,
        zero_empty_rows=zero_empty_rows,
        validate=validate,
        tuning=tuning,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
    )
    return wrapper.run(
        query,
        normalized_cache,
        block_tables,
        seq_lens_kv,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out,
        lse=lse,
        return_lse=return_lse,
        validate=validate,
        cp_rank=cp_rank,
    )


@flashinfer_experimental_api
def batch_sparse_mla_decode_with_paged_kv_cache(
    query,
    kv_cache,
    indices,
    lengths,
    extra_kv_cache=None,
    *,
    extra_indices=None,
    extra_lengths=None,
    qo_indptr=None,
    max_q_len=None,
    kv_layout="NHD",
    kv_cache_format="native",
    extra_kv_cache_format=None,
    softmax_scale: GlobalScale | None = None,
    output_scale: GlobalScale = 1.0,
    kv_cache_scale: GlobalScale = 1.0,
    attention_sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    workspace_buffer=None,
    assume_valid_prefix=False,
    output_quant_block_size=None,
    out_scale_format="ue8m0",
    num_output_groups=None,
    token_positions=None,
    cos_sin_cache=None,
    out_scale=None,
    tuning=None,
):
    """Eager plan-and-run helper using separate source indices and lengths.

    Use a planned wrapper for CUDA Graph replay. Index conversion and scale
    composition remain external. softmax_scale includes attention/Q/K descales;
    output_scale includes the V descale and optional output multiplier.
    Source contracts, output quantization arguments, ``tuning`` and return
    layouts follow
    :class:`BatchSparseMLADecodePagedTSWrapper`.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "plan BatchSparseMLADecodePagedTSWrapper before CUDA Graph capture"
        )
    packed = query.ndim == 3
    if packed:
        if qo_indptr is None:
            raise ValueError("packed queries require qo_indptr")
        batch = qo_indptr.numel() - 1
        if max_q_len is None:
            max_q_len = max(1, int((qo_indptr[1:] - qo_indptr[:-1]).max().item()))
    else:
        batch = query.shape[0]
        max_q_len = query.shape[1] if max_q_len is None else max_q_len
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace_buffer)
    wrapper.plan(
        query.device,
        batch,
        query.shape[-2],
        max_topk=indices.shape[-1],
        max_extra_topk=0 if extra_indices is None else extra_indices.shape[-1],
        max_q_len=max_q_len,
        packed_query=packed,
        q_dtype=query.dtype,
        qk_rope_head_dim=query.shape[-1] - 512,
        kv_cache_format=kv_cache_format,
        extra_kv_cache_format=extra_kv_cache_format,
        kv_layout=kv_layout,
        use_attention_sinks=attention_sinks is not None,
        return_lse=return_lse,
        assume_valid_prefix=assume_valid_prefix,
        output_quant_block_size=output_quant_block_size,
        out_scale_format=out_scale_format,
        num_output_groups=num_output_groups,
        tuning=tuning,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        kv_cache_scale=kv_cache_scale,
    )
    return wrapper.run(
        query,
        kv_cache,
        indices,
        lengths,
        extra_kv_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        kv_cache_scale=kv_cache_scale,
        attention_sinks=attention_sinks,
        out=out,
        lse=lse,
        token_positions=token_positions,
        cos_sin_cache=cos_sin_cache,
        out_scale=out_scale,
    )


def get_prims_ts_batch_mla_decode_workspace_size(
    batch_size: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    *,
    seq_len_q: Optional[int] = None,
    max_q_len: Optional[int] = None,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    mask_type: Literal["dense", "causal"] = "causal",
    device=None,
    cp_world: int = 1,
    cp_interleave_size: int = 1,
    validate: bool = True,
) -> int:
    """Return caller-workspace bytes for one automatic MLA policy.

    The arguments define the static geometry used to resolve the same automatic
    policy and private scratch layout as
    :func:`batch_mla_decode_with_paged_kv_cache`, without compiling attention kernels.
    ``max_q_len`` is the static per-request Q bound for both fixed and
    packed-query launches;
    ``seq_len_q`` remains a backward-compatible fixed-Q alias. If neither is
    supplied, the bound is one. The returned byte count covers split-KV
    scratch. Optional final LSE uses a separate caller-owned output. Allocate
    with the same ``cp_world`` and ``cp_interleave_size`` as the plan. CP adds
    no scratch. Use a contiguous ``torch.int8`` or ``torch.uint8`` CUDA buffer;
    MLA does not require its contents to be initialized before first use.
    ``max_kv_len`` is rank-local capacity, not the global request length.

    ``validate=False`` skips repeated input/device validation. The caller must
    ensure supported hardware, valid geometry and dtypes, and a static Q bound
    covering every request. Alias and device resolution are retained, as is
    the shared automatic-policy cache; a cold policy lookup may still query
    device properties or compile a hardware occupancy probe. Warm up outside
    CUDA graph capture.
    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be a bool")
    validate_mla_cp(cp_world, cp_interleave_size)
    if validate:
        batch_size = validate_positive_int(batch_size, "batch_size")
        num_qo_heads = validate_positive_int(num_qo_heads, "num_qo_heads")
        validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
        page_size = validate_page_size(page_size)
        max_kv_len = validate_mla_max_kv_len(max_kv_len, "max_kv_len")
    max_q_len = resolve_max_seq_len_q_alias(
        seq_len_q=seq_len_q,
        max_q_len=max_q_len,
        default=1,
        validate=validate,
    )
    assert max_q_len is not None
    if validate:
        validate_mla_query_head_extent(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
        )
        validate_mask(mask_type)
    if kv_dtype is None:
        kv_dtype = q_dtype
    if validate:
        validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
    resolved_device, device_index = resolve_cuda_device(device)
    if validate:
        validate_runtime_device(resolved_device)

    spec = resolve_mla_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_kv_len,
        attention_dtype_key(q_dtype),
        attention_dtype_key(kv_dtype),
        attention_dtype_key(out_dtype),
        mask_type,
        max_q_len,
        # The skip-correction threshold does not change the workspace; pass the default
        # positionally so a default plan() hits the same cached spec.
        0.0,
        cp_world,
        cp_interleave_size,
    )
    # No final LSE section is needed; only the kernel scratch alignment remains.
    return align_up(spec.kernel_workspace_bytes)


def get_prims_ts_sparse_mla_decode_workspace_size(*plan_args, **plan_kwargs):
    """Return workspace bytes using the same arguments as wrapper.plan().

    Resolves the default kernel geometry without allocating GPU scratch or
    compiling a kernel. Bind the resulting byte buffer to the wrapper constructor
    and call plan() before graph capture; run() is the prepared standalone launch.
    """
    _, _, _, _, size_bytes, _, _ = resolve_sparse_mla_plan(*plan_args, **plan_kwargs)
    return size_bytes


def tuning_rescale_threshold(tuning: object) -> Optional[float]:
    if tuning is None:
        return None
    if not isinstance(tuning, MLAAttentionTuning):
        raise TypeError("tuning must be an MLAAttentionTuning or None")
    return tuning.softmax_rescale_threshold_log2


__all__ = [
    "BatchMLADecodePagedTSWrapper",
    "batch_mla_decode_with_paged_kv_cache",
    "get_prims_ts_batch_mla_decode_workspace_size",
]
