# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native D512 sparse MLA over one or two pools using task-scheduled kernels."""

from dataclasses import replace
import copy
import functools
import math
import struct

import torch

from .policy import select_sparse_mla_profile

from flashinfer.attention.prims_ts.mla_decode import (
    _MLADecodeLaunchSpec,
    _MLARuntime,
    _MLAWorkspaceViews,
    _get_compiled_mla_decode,
    _launch_mla_decode,
    _make_mla_decode_compile_spec,
    _mla_kernel_compile_signature,
)
from flashinfer.attention.prims_ts.decode import (
    _append_workspace_section,
    _resolve_cuda_device,
    _validate_workspace_buffer,
    _workspace_section_view,
)
from flashinfer.attention.prims_ts.sparse_mla_decode import SparseMLAPreparedMetadata


@functools.cache
def _compile_finish(device_index, heads, independent):
    import cutlass
    import cutlass.cute as cute
    from .finish import FinishSparseMla

    rows = cute.sym_int()

    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=4,
        )

    with torch.cuda.device(device_index):
        partial = tensor(cutlass.BFloat16, (rows, heads, 512))
        lse = tensor(cutlass.Float32, (rows, heads))
        lens = tensor(cutlass.Int32, (rows,))
        return cute.compile[cute.FrontendNext](
            FinishSparseMla(independent),
            partial,
            partial,
            lse,
            lse,
            lens,
            lens,
            tensor(cutlass.Float32, (heads,)),
            partial,
            lse,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 2",
        )


@functools.cache
def _compile_sparse_reduce(device_index, heads, storage_heads, splits, direct=False):
    import cutlass
    import cutlass.cute as cute
    from .reduce import FinishSparseMlaSplit

    rows = cute.sym_int()

    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=4,
        )

    with torch.cuda.device(device_index):
        return cute.compile[cute.FrontendNext](
            FinishSparseMlaSplit(splits, direct),
            tensor(cutlass.BFloat16, (rows, storage_heads, splits, 512)),
            tensor(cutlass.Float32, (rows, storage_heads, splits)),
            tensor(cutlass.Int32, (rows,)),
            tensor(cutlass.Float32, (heads,)),
            tensor(cutlass.BFloat16, (rows, heads, 512)),
            tensor(cutlass.Float32, (rows, heads)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 2",
        )


class SparseMLADecodePlan:
    """Plan native BF16/E4M3 attention over one or two pools, then bind metadata.

    Q is [B,Sq,H,512] or packed [total_q,H,512]. Caller-prepared metadata
    contains int32 [rows, capacity] storage-row indices and live lengths.
    Indices already include physical page strides; -1 is masked. Both source
    lists participate in one attention distribution. Packed KV is unsupported.

    Planning with ``assume_valid_prefix=True`` promises that every index before
    each live source length is valid (no -1 holes). Direct FP8 2CTA kernels
    then derive masks from lengths without reading indices or issuing ballots.
    Other kernels retain their generic mask path; no gain was established.
    ``run(validate=True)`` checks the promise; graph replay with validation
    disabled must preserve it. Tail entries beyond the lengths are ignored.

    A wrapper/workspace permits one in-flight run. Graphs require a completed
    warmup, stable addresses, preallocated outputs, and validate=False.
    Input/output/workspace storage must not overlap; this is an unchecked
    caller precondition. Separate wrappers/workspaces are needed per stream.
    """

    def __init__(self, workspace_buffer=None):
        self._workspace_buffer = workspace_buffer
        self._tuning = None
        self._state = None
        self._scalar_cache = {}
        self._workspace_size_only = False
        self.workspace_size_bytes = 0

    def plan(
        self,
        device,
        batch_size,
        num_heads,
        *,
        max_topk,
        max_extra_topk=0,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_layout="NHD",
        has_sinks=False,
        return_lse=False,
        assume_valid_prefix=False,
    ):
        """Plan D512 attention over one primary and an optional extra pool.

        ``max_topk`` describes arbitrary selected primary entries, not an SWA
        window. Prepared indices already include the physical page stride, so
        the core uses page size one regardless of the pools' external pages.
        Planning compiles no metadata-preparation kernel.
        """
        import cutlass
        from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
            MlaProfile,
            compute_workspace_size,
        )
        from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
            ThroughputLatencyMlaDecodeTs,
        )
        from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
            compute_workspace_size as workspace_2cta,
        )
        from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
            MlaDecodeTs,
        )

        if not isinstance(assume_valid_prefix, bool):
            raise TypeError("assume_valid_prefix must be bool")
        device, _ = _resolve_cuda_device(device)
        if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
            raise NotImplementedError("sparse TS MLA requires SM100 or SM103")
        for name, value in (
            ("batch_size", batch_size),
            ("num_heads", num_heads),
            ("max_seq_len_q", max_seq_len_q),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_heads > 128:
            raise ValueError("num_heads must be at most 128")
        for value in (max_topk, max_extra_topk):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("source capacities must be nonnegative integers")
        if max_topk == 0:
            raise ValueError("the primary selected-slot capacity must be positive")
        if q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("native BF16 or E4M3 is required")
        if kv_layout not in ("NHD", "HND"):
            raise ValueError("kv_layout must be NHD or HND")
        max_rows = batch_size * max_seq_len_q
        capacity = max(
            256,
            ((max_topk + 127) // 128 + (max_extra_topk + 127) // 128) * 128,
        )
        if max_rows * num_heads >= 2**31 or capacity >= 2**31 - 32768:
            raise ValueError("query/route extent exceeds the int32 kernel domain")
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        selection = select_sparse_mla_profile(
            rows=max_rows,
            heads=num_heads,
            query_length=max_seq_len_q,
            capacity=capacity,
            dtype="bf16" if q_data_type == torch.bfloat16 else "fp8",
            sm_count=sm_count,
            forced=self._tuning,
        )
        tuning = selection.tuning
        family, tile = selection.family, selection.tile
        splits, head_dim_ctas = selection.splits, selection.head_dim_ctas
        dtype_name = "bf16" if q_data_type == torch.bfloat16 else "e4m3"
        if family == "2cta":
            kernel = MlaDecodeTs(
                page_size=1,
                max_active_clusters=sm_count // 2,
                is_persistent=tuning.scheduler != "nonpersistent",
                is_var_seq=False,
                is_var_split_kv=False,
                static_split_kv=splits,
                qkv_dtype=dtype_name,
                out_dtype="bf16",
                rope_dim=0,
                num_heads=num_heads,
                seq_len_q=1,
                batch_size=max_rows,
                mask_type="dense",
                device_scales=True,
            )
            kernel.sparse_gather_warps = tuning.gather_issue_warps
            kernel.sparse_kv_stages = tuning.kv_pipeline_stages
            kernel.sparse_uniform_pages = tuning.uniform_offset_cache
            kernel.sparse_balanced_registers = tuning.balanced_registers
            kernel.sparse_defer_max_update = tuning.defer_max_update
            core_bytes = workspace_2cta(
                tile_size_q=128,
                num_q_tiles=1,
                latent_dim=512,
                batch_size=max_rows,
                split_kv=splits,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
        else:
            profile = MlaProfile(
                name="sparse",
                kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
                tile_size_q=tile,
                num_ctas_per_head_dim=head_dim_ctas,
                num_ctas_per_seq_kv=splits,
                use_multi_ctas_kv=int(splits > 1),
                use_cluster_reduction=int(tuning.reduction == "cluster"),
                use_persistent_scheduler=int(tuning.scheduler != "nonpersistent"),
                use_clc_dynamic_persistent_scheduler=int(tuning.scheduler == "clc"),
            )
            kernel = ThroughputLatencyMlaDecodeTs(
                batch_size=max_rows,
                num_heads=tile,
                seq_len_q=(num_heads + tile - 1) // tile,
                seq_len_k=capacity,
                rope_dim=0,
                page_size=1,
                max_active_clusters=sm_count,
                qkv_dtype=dtype_name,
                out_dtype="bf16",
                profile=profile,
                reduction_mode=tuning.reduction,
                logical_num_heads=num_heads,
                logical_seq_len_q=1,
                tile_size_q=tile,
                gather_issue_warps=tuning.gather_issue_warps,
                sparse_offset_cache=tuning.offset_cache,
                mask_type="dense",
                device_scales=True,
            )
            kernel.compact_sparse_loader = tuning.compact_loader
            kernel.balanced_sparse_registers = tuning.balanced_registers
            kernel.reuse_sparse_kv = tuning.reuse_kv
            kernel.sparse_kv_tile_size = tuning.kv_tile_size
            kernel.page_pipeline_stages = tuning.page_pipeline_stages
            kernel.reuse_sparse_kv_stages = tuning.reuse_kv_stages
            kernel.paired_sparse_correction = tuning.paired_correction
            kernel.single_sparse_stream = tuning.single_kv_stream
            kernel.defer_sparse_max_update = tuning.defer_max_update
            kernel.uniform_sparse_offset_cache = tuning.uniform_offset_cache
            core_bytes = compute_workspace_size(
                cfg=kernel._make_config(),
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
        spec = _make_mla_decode_compile_spec(
            _MLADecodeLaunchSpec(kernel, (), core_bytes, splits),
            device_index=device.index,
            num_heads=num_heads,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            page_size=1,
            q_dtype_key=str(q_data_type).removeprefix("torch."),
            output_dtype_key="bfloat16",
            max_seq_len_q=1,
            packed_query=False,
            device_scales=True,
        )
        use_cluster_epilogue = (
            tuning.fuse_epilogue
            and tuning.direct_inputs
            and family == "swap"
            and tile in (16, 32)
            and splits > 1
            and tuning.reduction == "cluster"
        )
        use_fused_epilogue = (
            tuning.fuse_epilogue
            and (
                (family, tile)
                in (("swap", 8), ("swap", 16), ("swap", 32), ("keep", 64))
                or (
                    family == "2cta"
                    and q_data_type in (torch.float8_e4m3fn, torch.bfloat16)
                )
            )
            and (splits == 1 or use_cluster_epilogue)
        )
        use_fused_reduction = (
            tuning.fuse_epilogue
            and family in ("swap", "keep", "2cta")
            and splits > 1
            and tuning.reduction == "gmem_separate"
        )
        use_direct = (
            tuning.direct_inputs
            and q_data_type in (torch.float8_e4m3fn, torch.bfloat16)
            and (
                tuning.scheduler == "nonpersistent"
                or family in ("keep", "swap")
                or (
                    family == "2cta"
                    and dtype_name == "e4m3"
                    and tuning.scheduler == "static"
                )
            )
            and (use_fused_epilogue or use_fused_reduction)
        )
        sections = {}
        byte_end = 0
        for name, shape, dtype in (
            ("core", (core_bytes,), torch.int8),
            ("partial", (2, max_rows, 1, num_heads, 512), torch.bfloat16),
            # Both source views must satisfy the core's 16-byte base alignment,
            # including H6/H12 and odd maximum query counts.
            ("lse", (2, (max_rows + 3) // 4 * 4, 1, num_heads), torch.float32),
            ("public_lse", (max_rows, num_heads), torch.float32),
        ):
            sections[name], byte_end = _append_workspace_section(byte_end, shape, dtype)
        self.workspace_size_bytes = byte_end
        if self._workspace_size_only:
            return
        workspace = self._workspace_buffer
        if workspace is None:
            workspace = torch.empty(byte_end, device=device, dtype=torch.uint8)
        _validate_workspace_buffer(workspace, device=device, required_bytes=byte_end)
        buffers = {
            name: _workspace_section_view(workspace, section)
            for name, section in sections.items()
        }
        compiled = _get_compiled_mla_decode(spec)
        compiled_fused = None
        compiled_static = None
        compiled_reducer = None
        storage_heads = ((num_heads + tile - 1) // tile) * tile
        if use_fused_epilogue or use_fused_reduction:
            fused_kernel = copy.copy(kernel)
            fused_kernel.fuse_sparse_epilogue = (
                use_fused_epilogue and not use_cluster_epilogue
            )
            if use_cluster_epilogue:
                fused_kernel.fuse_sparse_cluster_epilogue = True
            fused_kernel.external_sparse_reduction = use_fused_reduction
            fused_kernel.direct_sparse = use_direct
            if family == "2cta":
                fused_kernel.assume_valid_prefix = assume_valid_prefix
            fused_kernel.direct_sparse_pages = (1, 1)
            fused_kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
            fused_spec = replace(
                spec,
                kernel=fused_kernel,
                kernel_signature=_mla_kernel_compile_signature(fused_kernel),
            )
            compiled_fused = _get_compiled_mla_decode(fused_spec)
            if use_direct:
                static_kernel = copy.copy(fused_kernel)
                static_kernel.direct_static_scales = True
                compiled_static = _get_compiled_mla_decode(
                    replace(
                        fused_spec,
                        kernel=static_kernel,
                        kernel_signature=_mla_kernel_compile_signature(static_kernel),
                    )
                )
            if use_fused_reduction:
                compiled_reducer = _compile_sparse_reduce(
                    device.index, num_heads, storage_heads, splits, use_direct
                )
        finish = tuple(
            _compile_finish(device.index, num_heads, independent)
            for independent in (False, True)
        )
        self._scalar_cache.clear()
        self._state = dict(
            device=device,
            batch=batch_size,
            heads=num_heads,
            max_q=max_seq_len_q,
            max_rows=max_rows,
            packed=packed_query,
            dtype=q_data_type,
            ks=max_topk,
            kc=max_extra_topk,
            capacity=capacity,
            kv_layout=kv_layout,
            has_sinks=has_sinks,
            return_lse=return_lse,
            buffers=buffers,
            workspace=workspace,
            workspace_bytes=byte_end,
            compiled=compiled,
            compiled_fused=compiled_fused,
            compiled_static=compiled_static,
            compiled_reducer=compiled_reducer,
            storage_heads=storage_heads,
            direct_inputs=use_direct,
            assume_valid_prefix=assume_valid_prefix,
            finish=finish,
            splits=splits,
            head_dim_ctas=head_dim_ctas,
            family=family,
            tile=tile,
            tuning=tuning,
            selection_reason=selection.reason,
            default_sl=torch.full(
                (max_rows,), max_topk, device=device, dtype=torch.int32
            ),
            default_cl=torch.full(
                (max_rows,), max_extra_topk, device=device, dtype=torch.int32
            ),
            dummy_indices=torch.zeros((max_rows, 1), device=device, dtype=torch.int32),
            dummy_scales=torch.ones((1, 2), device=device, dtype=torch.float32),
            dummy_cache=torch.zeros((1, 1, 512), device=device, dtype=q_data_type),
            default_sinks=torch.full(
                (num_heads,), -torch.inf, device=device, dtype=torch.float32
            ),
        )

    def _scalar(self, value, name, validate):
        state = self._state
        if isinstance(value, torch.Tensor):
            if (
                value.dtype != torch.float32
                or value.device != state["device"]
                or value.numel() != 1
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} must be a scalar CUDA FP32 tensor")
            if validate and (
                not torch.isfinite(value).all().item() or (value <= 0).any().item()
            ):
                raise ValueError(f"{name} must be positive and finite")
            return value.reshape(1)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
        try:
            key = struct.unpack("f", struct.pack("f", float(value)))[0]
        except OverflowError as error:
            raise ValueError(f"{name} must be representable in float32") from error
        if not math.isfinite(key) or key <= 0:
            raise ValueError(f"{name} must be positive finite float32")
        if key not in self._scalar_cache:
            self._scalar_cache[key] = torch.full(
                (1,), key, device=state["device"], dtype=torch.float32
            )
        return self._scalar_cache[key]

    def _pool(self, cache, page_size, name):
        state = self._state
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if cache.ndim == 4:
            axis = 1 if state["kv_layout"] == "HND" else 2
            if cache.shape[axis] != 1:
                raise ValueError(f"{name} requires one KV head")
            cache = cache.squeeze(axis)
        if (
            cache.ndim != 3
            or tuple(cache.shape[1:]) != (page_size, 512)
            or cache.shape[0] <= 0
        ):
            raise ValueError(f"{name} must have shape [pages,{page_size},512]")
        if cache.device != state["device"] or cache.dtype != state["dtype"]:
            raise ValueError(f"{name} must match the planned device and dtype")
        if (
            cache.stride(-1) != 1
            or cache.stride(1) != 512
            or cache.stride(0) < page_size * 512
            or cache.stride(0) % 512
            or cache.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} requires aligned compact rows and a page stride divisible by 512 elements"
            )
        row_stride = cache.stride(0) // 512
        rows = (cache.shape[0] - 1) * row_stride + page_size
        if rows >= 0x7FFFFFFF:
            raise ValueError(f"{name} exceeds signed gather coordinate range")
        linear = cache.as_strided((rows, 1, 512), (512, 512, 1))
        return linear, row_stride, cache.shape[0] * page_size

    def run(
        self,
        query,
        kv_cache,
        metadata: SparseMLAPreparedMetadata,
        extra_kv_cache=None,
        *,
        qo_indptr=None,
        softmax_scale=512**-0.5,
        q_scale=1.0,
        kv_scale=1.0,
        extra_kv_scale=None,
        output_scale=1.0,
        sinks=None,
        out=None,
        lse=None,
        validate=True,
    ):
        """Launch with caller-prepared metadata, without index conversion.

        Caller-owned metadata and plan workspace must outlive graph replay. Q and
        native KV are used directly; flattening padded pages only creates a
        tensor view. Required attention reductions/finishing remain included.
        Metadata is an unchecked consistency contract: its packed routes,
        counts and scales must agree with the supplied indices and scalars.
        """
        state = self._state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(metadata, SparseMLAPreparedMetadata):
            raise TypeError("metadata must be SparseMLAPreparedMetadata")
        prefix = tuple(query.shape[:-2])
        expected_rank = 3 if state["packed"] else 4
        if (
            query.ndim != expected_rank
            or tuple(query.shape[-2:]) != (state["heads"], 512)
            or query.dtype != state["dtype"]
            or query.device != state["device"]
            or not query.is_contiguous()
            or query.data_ptr() % 16
        ):
            raise ValueError(
                "query shape, dtype, device or alignment does not match plan"
            )
        rows = math.prod(prefix)
        if rows > state["max_rows"] or (
            not state["packed"] and prefix != (state["batch"], state["max_q"])
        ):
            raise ValueError("query token extents do not match plan")
        if state["packed"]:
            if (
                qo_indptr is None
                or qo_indptr.shape != (state["batch"] + 1,)
                or qo_indptr.dtype != torch.int32
                or qo_indptr.device != state["device"]
                or not qo_indptr.is_contiguous()
            ):
                raise ValueError("packed queries require int32 qo_indptr[B+1]")
            if validate:
                offsets = qo_indptr.cpu()
                delta = offsets[1:] - offsets[:-1]
                if (
                    offsets[0] != 0
                    or offsets[-1] != rows
                    or (delta < 0).any()
                    or (delta > state["max_q"]).any()
                ):
                    raise ValueError("invalid packed query offsets")
        elif qo_indptr is not None:
            raise ValueError("fixed queries do not accept qo_indptr")

        def flatten(cache, name):
            if not isinstance(cache, torch.Tensor) or cache.ndim not in (3, 4):
                raise ValueError(f"{name} must be a native paged KV tensor")
            page_axis = 2 if cache.ndim == 4 and state["kv_layout"] == "HND" else 1
            return self._pool(cache, cache.shape[page_axis], name)[0]

        primary = flatten(kv_cache, "kv_cache")
        if extra_kv_cache is None:
            if (
                state["kc"]
                or metadata.extra_indices is not None
                or metadata.extra_lengths is not None
            ):
                raise ValueError("extra pool and indices are required by the plan")
            extra = state["dummy_cache"]
        else:
            extra = flatten(extra_kv_cache, "extra_kv_cache")

        def source_metadata(indices, lengths, capacity, tokens, name):
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
                if (
                    state["assume_valid_prefix"]
                    and (active & (indices < 0)).any().item()
                ):
                    raise ValueError(
                        f"{name} active prefix contains a hole with assume_valid_prefix=True"
                    )
            return indices, lengths

        si, sl = source_metadata(
            metadata.indices, metadata.lengths, state["ks"], primary.shape[0], "primary"
        )
        ci, cl = source_metadata(
            metadata.extra_indices,
            metadata.extra_lengths,
            state["kc"],
            extra.shape[0],
            "extra",
        )
        if state["has_sinks"] != (sinks is not None):
            raise ValueError("sinks presence must match the plan")
        if sinks is None:
            sinks = state["default_sinks"]
        if (
            sinks.shape != (state["heads"],)
            or sinks.dtype != torch.float32
            or sinks.device != state["device"]
            or not sinks.is_contiguous()
        ):
            raise ValueError("sinks must be contiguous CUDA FP32[H]")
        if validate and torch.isnan(sinks).any().item():
            raise ValueError("sinks must not contain NaN")
        if extra_kv_scale is None:
            extra_kv_scale = kv_scale
        shared_scale = extra_kv_scale is kv_scale or (
            not isinstance(kv_scale, torch.Tensor)
            and not isinstance(extra_kv_scale, torch.Tensor)
            and kv_scale == extra_kv_scale
        )
        independent = state["kc"] > 0 and not shared_scale
        state["last_execution_mode"] = (
            "independent_sources" if independent else "joint_sources"
        )
        scale_tensors = [
            self._scalar(v, n, validate)
            for v, n in (
                (softmax_scale, "softmax_scale"),
                (q_scale, "q_scale"),
                (kv_scale, "kv_scale"),
                (extra_kv_scale, "extra_kv_scale"),
                (output_scale, "output_scale"),
            )
        ]
        if validate and state["dtype"] == torch.bfloat16:
            if any(t.item() != 1 for t in scale_tensors[1:4]):
                raise ValueError("BF16 Q/KV descales must be one")
        if out is None:
            out = torch.empty(
                (*prefix, state["heads"], 512),
                device=state["device"],
                dtype=torch.bfloat16,
            )
        if (
            out.shape != query.shape
            or out.dtype != torch.bfloat16
            or out.device != state["device"]
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise ValueError("out must be contiguous BF16 with the query shape/device")
        if lse is None:
            # A returned result must survive the next call on this wrapper.
            # Use plan-owned scratch only when the LSE is not exposed.
            lse = (
                torch.empty(
                    (*prefix, state["heads"]),
                    device=state["device"],
                    dtype=torch.float32,
                )
                if state["return_lse"]
                else state["buffers"]["public_lse"][:rows].view(*prefix, state["heads"])
            )
        if (
            lse.shape != (*prefix, state["heads"])
            or lse.dtype != torch.float32
            or lse.device != state["device"]
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be contiguous FP32 with one value per query/head"
            )
        fused = (
            state["compiled_fused"] is not None
            and not independent
            and lse.data_ptr() % 16 == 0
        )
        state["last_fused_epilogue"] = fused
        fused_main = fused and state["compiled_reducer"] is None
        direct = fused and state["direct_inputs"]
        state["last_direct_inputs"] = direct
        static_scales = direct and all(
            not isinstance(v, torch.Tensor)
            for v in (softmax_scale, q_scale, kv_scale, output_scale)
        )
        state["last_static_scales"] = static_scales
        bmm1_scale, bmm2_scale = 1.0, 1.0
        if static_scales:

            def f32(value):
                return struct.unpack("f", struct.pack("f", float(value)))[0]

            bmm1_scale = f32(f32(f32(softmax_scale) * f32(q_scale)) * f32(kv_scale))
            bmm2_scale = f32(f32(output_scale) * f32(kv_scale))
        sparse_inputs = (
            (
                si,
                ci,
                sl,
                cl,
                scale_tensors[0],
                scale_tensors[1],
                scale_tensors[2],
                scale_tensors[4],
                sinks,
                1,
                1,
            )
            if direct
            else None
        )
        if rows:
            buffers = state["buffers"]
            passes = 2 if independent else 1
            required = not direct
            prepared_buffers = {}
            for key, value, dtype, shape in (
                (
                    "routes",
                    metadata.routes,
                    torch.int32,
                    (passes, rows, state["capacity"]),
                ),
                (
                    "lengths",
                    metadata.execution_lengths,
                    torch.int32,
                    (passes, rows),
                ),
                (
                    "counts",
                    metadata.valid_counts,
                    torch.int32,
                    (passes, rows),
                ),
                (
                    "scales",
                    metadata.scale_params,
                    torch.float32,
                    (passes, 2 + state["heads"] + rows),
                ),
            ):
                if value is None:
                    if required:
                        raise ValueError(
                            f"prepared {key} are required by this schedule"
                        )
                    continue
                if (
                    value.dtype != dtype
                    or value.device != state["device"]
                    or value.shape != shape
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"prepared {key} must be contiguous {dtype}{shape}"
                    )
                prepared_buffers[key] = value
            buffers = (
                buffers
                | dict(
                    routes=state["dummy_indices"][:rows].unsqueeze(0),
                    lengths=sl.unsqueeze(0),
                    counts=sl.unsqueeze(0),
                    scales=state["dummy_scales"],
                )
                | prepared_buffers
            )
            query_view = query.view(rows, 1, state["heads"], 512)
            for slot in range(2 if independent else 1):
                _launch_mla_decode(
                    _MLARuntime(
                        query_view,
                        primary,
                        (
                            out.view(rows, 1, state["heads"], 512)
                            if fused_main
                            else buffers["partial"][slot, :rows]
                        ),
                        primary.shape[0],
                        bmm1_scale,
                        bmm2_scale,
                        extra_cache=extra,
                        scale_params=buffers["scales"][slot],
                        sparse_inputs=sparse_inputs,
                    ),
                    block_tables=buffers["routes"][slot, :rows],
                    seq_lens=buffers["lengths"][slot, :rows],
                    qo_indptr=None,
                    packed_query=False,
                    kv_lora_rank=512,
                    split_kv=state["splits"],
                    workspace=_MLAWorkspaceViews(
                        buffers["core"] if buffers["core"].numel() else None,
                        (
                            lse.view(rows, 1, state["heads"])
                            if fused_main
                            else buffers["lse"][slot, :rows]
                        ),
                    ),
                    compiled=(
                        state["compiled_static"]
                        if static_scales
                        else state["compiled_fused"]
                        if fused
                        else state["compiled"]
                    ),
                )
            if fused and not fused_main:
                split_elements = rows * state["storage_heads"] * state["splits"]
                o_bytes = split_elements * 512 * 2
                partial = (
                    buffers["core"][:o_bytes]
                    .view(torch.bfloat16)
                    .view(rows, state["storage_heads"], state["splits"], 512)
                )
                partial_lse = (
                    buffers["core"][o_bytes : o_bytes + split_elements * 4]
                    .view(torch.float32)
                    .view(rows, state["storage_heads"], state["splits"])
                )
                state["compiled_reducer"](
                    partial,
                    partial_lse,
                    buffers["counts"][0, :rows],
                    sinks,
                    out.view(rows, state["heads"], 512),
                    lse.view(rows, state["heads"]),
                )
            if not fused:
                state["finish"][int(independent)](
                    buffers["partial"][0, :rows].view(rows, state["heads"], 512),
                    buffers["partial"][1, :rows].view(rows, state["heads"], 512),
                    buffers["lse"][0, :rows].view(rows, state["heads"]),
                    buffers["lse"][1, :rows].view(rows, state["heads"]),
                    buffers["counts"][0, :rows],
                    buffers["counts"][1 if independent else 0, :rows],
                    sinks,
                    out.view(rows, state["heads"], 512),
                    lse.view(rows, state["heads"]),
                )
        return (out, lse) if state["return_lse"] else out


def get_prims_ts_sparse_mla_decode_workspace_size(*plan_args, **plan_kwargs):
    """Return workspace bytes using the same arguments as wrapper.plan().

    Resolves the default kernel geometry without allocating GPU scratch or
    compiling a kernel. Bind the resulting byte buffer to the wrapper constructor
    and call plan() before graph capture; run() is the prepared standalone launch.
    """
    wrapper = SparseMLADecodePlan()
    wrapper._workspace_size_only = True
    wrapper.plan(*plan_args, **plan_kwargs)
    return wrapper.workspace_size_bytes


def batch_sparse_mla_decode_with_paged_kv_cache(
    query,
    kv_cache,
    metadata: SparseMLAPreparedMetadata,
    extra_kv_cache=None,
    *,
    qo_indptr=None,
    max_seq_len_q=None,
    kv_layout="NHD",
    softmax_scale=512**-0.5,
    q_scale=1.0,
    kv_scale=1.0,
    extra_kv_scale=None,
    output_scale=1.0,
    sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    workspace_buffer=None,
    assume_valid_prefix=False,
):
    """Eager plan-and-run helper using caller-prepared metadata.

    Use a planned wrapper for CUDA Graph replay. Preparation remains external.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("plan SparseMLADecodePlan before CUDA Graph capture")
    if not isinstance(metadata, SparseMLAPreparedMetadata):
        raise TypeError("metadata must be SparseMLAPreparedMetadata")
    packed = query.ndim == 3
    if packed:
        if qo_indptr is None:
            raise ValueError("packed queries require qo_indptr")
        batch = qo_indptr.numel() - 1
        if max_seq_len_q is None:
            max_seq_len_q = max(1, int((qo_indptr[1:] - qo_indptr[:-1]).max().item()))
    else:
        batch = query.shape[0]
        max_seq_len_q = query.shape[1] if max_seq_len_q is None else max_seq_len_q
    wrapper = SparseMLADecodePlan(workspace_buffer)
    wrapper.plan(
        query.device,
        batch,
        query.shape[-2],
        max_topk=metadata.indices.shape[-1],
        max_extra_topk=0
        if metadata.extra_indices is None
        else metadata.extra_indices.shape[-1],
        max_seq_len_q=max_seq_len_q,
        packed_query=packed,
        q_data_type=query.dtype,
        kv_layout=kv_layout,
        has_sinks=sinks is not None,
        return_lse=return_lse,
        assume_valid_prefix=assume_valid_prefix,
    )
    return wrapper.run(
        query,
        kv_cache,
        metadata,
        extra_kv_cache,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        extra_kv_scale=extra_kv_scale,
        output_scale=output_scale,
        sinks=sinks,
        out=out,
        lse=lse,
    )
