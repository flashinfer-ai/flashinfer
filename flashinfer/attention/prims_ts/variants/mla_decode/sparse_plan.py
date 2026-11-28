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

"""Sparse MLA planning: launch profile, D576 reload plan and output quantization."""

import math
import torch
from dataclasses import replace
from ...common.attention_args import validate_positive_int
from ...common.scales import GlobalScale
from ...common.tensors import resolve_cuda_device, validate_runtime_device
from ...common.workspace import append_workspace_section
from .kernel.device.kv_formats import validate_nvfp4_kv_formats
from .kernel.sparse_selection import (
    prefer_sparse_d576_reload,
    select_sparse_mla_profile,
)
from .launch import (
    MLADecodeCompileSpec,
    MLADecodeLaunchSpec,
    validate_softmax_rescale_threshold_log2,
    make_mla_decode_compile_spec,
    mla_kernel_compile_signature,
    output_quant_scale_layout,
)


D576_HEAD_DIM = 576


D576_V_DIM = 512


D576_HEADS = (64, 128)


def make_sparse_output_quant(heads, block_size, scale_format, groups):
    if block_size is None:
        if groups is not None or scale_format != "ue8m0":
            raise ValueError(
                "output quantization options require output_quant_block_size"
            )
        return None
    block_size = validate_positive_int(block_size, "output_quant_block_size")
    if block_size not in (32, 128):
        raise ValueError("output_quant_block_size must be 32, 128, or None")
    if scale_format not in ("fp32", "ue8m0"):
        raise ValueError("out_scale_format must be 'fp32' or 'ue8m0'")
    groups = validate_positive_int(groups, "num_output_groups")
    if heads % groups:
        raise ValueError("num_output_groups must divide num_qo_heads")
    from .kernel.device.output_quant import MlaOutputQuant

    return MlaOutputQuant(block_size, scale_format == "ue8m0", heads, groups)


def prepare_sparse_output_quant(
    quant, query, positions, cos_sin, out, scales, validate
):
    """Bind the token-major output and column-major block scales."""
    prefix = tuple(query.shape[:-2])
    rows, device = math.prod(prefix), query.device
    channels = quant.num_heads // quant.num_groups * 512
    padded_rows = (rows + 3) // 4 * 4
    columns, scale_stride = output_quant_scale_layout(quant, padded_rows)
    if (
        positions is None
        or positions.shape != prefix
        or positions.dtype != torch.int32
        or positions.device != device
        or not positions.is_contiguous()
    ):
        raise ValueError(
            "token_positions must be contiguous CUDA INT32 matching the query prefix"
        )
    if (
        cos_sin is None
        or cos_sin.ndim != 2
        or cos_sin.shape[0] == 0
        or cos_sin.shape[1] != 64
        or cos_sin.dtype != torch.float32
        or cos_sin.device != device
        or not cos_sin.is_contiguous()
        or cos_sin.data_ptr() % 16
    ):
        raise ValueError(
            "cos_sin_cache must be 16-byte aligned contiguous CUDA FP32[P,64]"
        )
    if (
        validate
        and rows
        and ((positions < 0) | (positions >= cos_sin.shape[0])).any().item()
    ):
        raise ValueError("token_positions must index cos_sin_cache")
    shape = (rows, quant.num_groups, channels)
    # Physical storage is group-major [G, T, C] (as in vLLM and TensorRT-LLM);
    # the public tensor is its token-major [T, G, C] view.
    if out is None:
        out = torch.empty(
            (quant.num_groups, rows, channels), dtype=torch.float8_e4m3fn, device=device
        ).transpose(0, 1)
    if (
        out.shape != shape
        or out.dtype != torch.float8_e4m3fn
        or out.device != device
        or out.data_ptr() % 16
    ):
        raise ValueError(
            f"out must be the [T, G, C] view of a contiguous group-major CUDA "
            f"E4M3 [G, T, C] tensor ({shape}), aligned to 16 bytes"
        )
    physical_out = out.transpose(0, 1)
    if rows and quant.num_groups > 1:
        # A live prefix can retain a larger graph bucket's group stride.
        capacity, remainder = divmod(physical_out.stride(0), channels)
        if remainder or capacity < rows or capacity * quant.num_heads >= 2**31:
            raise ValueError("out must have valid group-major token capacity")
        try:
            physical_out = physical_out.as_strided(
                (quant.num_groups, capacity, channels), physical_out.stride()
            )
        except RuntimeError as exc:
            raise ValueError(
                "out must view complete group-major backing storage"
            ) from exc
    if not physical_out.is_contiguous():
        raise ValueError("out must have contiguous physical group-major storage")
    scale_dtype = torch.int32 if quant.ue8m0 else torch.float32
    scale_shape = (rows, quant.num_groups, columns)
    if scales is None:
        scales = torch.empty(
            (quant.num_groups, columns, padded_rows), dtype=scale_dtype, device=device
        ).permute(2, 0, 1)[:rows]
        if rows == 0:
            scales = scales.as_strided(scale_shape, scale_stride)
    if (
        scales.shape != scale_shape
        or scales.stride() != scale_stride
        or scales.dtype != scale_dtype
        or scales.device != device
        or scales.data_ptr() % 16
    ):
        raise ValueError(
            f"out_scale must be 16-byte aligned CUDA {scale_dtype}{scale_shape} "
            f"with strides {scale_stride}"
        )
    return (
        out,
        scales,
        (physical_out.view(-1), positions.view(-1), cos_sin, scales),
    )


def resolve_sparse_d576_reload_plan(
    device,
    max_rows,
    num_heads,
    *,
    max_topk,
    max_extra_topk,
    q_dtype,
    use_attention_sinks,
    assume_valid_prefix,
    threshold,
):
    """Plan the native D576 reload schedule and its split-KV scratch.

    One CTA per (token, 64 heads) on the 1CTA keeps-MMA-AB q64 schedule; 128
    heads run as two physical query positions of one token. The kernel is
    compiled for the top-k capacity rounded up to whole 128-key tiles (it uses
    the capacity only through its tile count). Below one wave the plan splits
    each top-k list over ``split_kv`` CTAs (and the 512 V columns over
    ``head_dim_ctas``); the parallel GMEM reducer, launched with PDL, merges
    the partials and publishes natural-log LSE. The split is fixed per plan,
    so a CUDA graph captured from it uses one compiled kernel.
    """
    import cutlass

    from ...common.hardware import get_max_active_clusters
    from .kernel.sparse_selection import SparseMlaProfile, select_sparse_mla_d576_split
    from .kernel.latency_1cta.config import MlaProfile
    from .kernel.latency_1cta.kernel import ThroughputLatencyMlaDecodeTs

    if q_dtype != torch.float8_e4m3fn:
        raise ValueError("D576 sparse MLA (qk_rope_head_dim=64) requires E4M3 Q / KV")
    if num_heads not in D576_HEADS:
        raise ValueError(
            f"D576 sparse MLA supports 64 or 128 query heads (64-row tiles), got {num_heads}"
        )
    if max_extra_topk:
        raise ValueError("D576 sparse MLA attends to one KV pool (max_extra_topk=0)")
    if use_attention_sinks:
        raise ValueError("D576 sparse MLA does not support attention sinks")
    if max_rows > 65535:
        # Each query token is one grid-z slice of the attention and reducer
        # launches; CUDA caps grid z at 65535.
        raise ValueError(
            "D576 sparse MLA supports at most 65535 query tokens per plan "
            f"(batch_size * max_q_len), got {max_rows}"
        )
    split_kv, head_dim_ctas = select_sparse_mla_d576_split(
        max_rows * (num_heads // 64),
        max_topk,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    profile = SparseMlaProfile(
        family="keep",
        tile_size_q=64,
        split_kv=split_kv,
        head_dim_ctas=head_dim_ctas,
        uniform_offset_cache=True,
        kv_pipeline_stages=10,
    )
    capacity = -(-max_topk // 128) * 128
    phys_sq = num_heads // 64
    kernel_profile = "h64_keeps_mma_ab"
    if split_kv > 1:
        hdim = "" if head_dim_ctas == 1 else f"_hdim{D576_V_DIM // head_dim_ctas}"
        kernel_profile = MlaProfile(
            name=f"h64_keeps_mma_ab_splitkv{split_kv}{hdim}_gmem",
            num_ctas_per_seq_kv=split_kv,
            num_ctas_per_head_dim=head_dim_ctas,
            use_persistent_scheduler=0,
            use_clc_dynamic_persistent_scheduler=0,
            use_multi_ctas_kv=1,
            use_cluster_reduction=0,
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
        )
    with torch.cuda.device(device):
        max_active_clusters = get_max_active_clusters(1)
    kernel = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=phys_sq,
        seq_len_k=capacity,
        latent_dim=512,
        rope_dim=64,
        page_size=1,
        max_active_clusters=max_active_clusters,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=kernel_profile,
        reduction_mode=None if split_kv == 1 else "gmem_separate",
        logical_num_heads=num_heads,
        logical_seq_len_q=1,
        fixed_seqlen=True,
        tile_size_q=64,
        explicit_split_kv=None,
        explicit_persistent=None,
        mask_type="dense",
        softmax_rescale_threshold_log2=0.0 if threshold is None else threshold,
    )
    # Preserve the sparse wrapper's natural-log LSE contract.
    kernel.lse_base_on_e = True
    cfg = kernel.make_config()
    if (
        int(cfg.num_ctas_per_seq_kv) != split_kv
        or int(cfg.num_ctas_per_head_dim) != head_dim_ctas
        or int(cfg.use_cluster_reduction) != 0
    ):
        raise NotImplementedError(
            f"D576 sparse MLA expects {split_kv} KV CTAs, {head_dim_ctas} V "
            f"CTAs and GMEM reduction; got {cfg.num_ctas_per_seq_kv}, "
            f"{cfg.num_ctas_per_head_dim}, cluster={cfg.use_cluster_reduction}"
        )
    if split_kv > 1 and not kernel.use_parallel_reduction:
        # Only the parallel GMEM reducer publishes the D576 natural-log LSE.
        raise NotImplementedError("D576 split-KV requires the parallel GMEM reducer")
    spec = MLADecodeCompileSpec(
        device_index=device.index,
        kernel_signature=mla_kernel_compile_signature(kernel),
        num_qo_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=1,
        q_dtype_key="float8_e4m3fn",
        out_dtype_key="bfloat16",
        max_q_len=phys_sq,
        packed_query=False,
        has_kernel_workspace=split_kv > 1,
        split_kv=split_kv,
        kernel=kernel,
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        (
            "core",
            (sparse_d576_workspace_bytes(max_rows, num_heads, split_kv),),
            torch.int8,
        ),
        ("public_lse", (max_rows, num_heads), torch.float32),
    ):
        sections[name], byte_end = append_workspace_section(byte_end, shape, dtype)
    return device, profile, spec, sections, byte_end, capacity, None


def resolve_sparse_mla_plan(
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
):
    """Resolve sparse geometry and scratch sizing without allocation/compilation.

    Scale operands are accepted for plan() argument compatibility; they do not
    affect geometry or workspace sizing.
    """
    # Imported here: the public MLAAttentionTuning type is defined in __init__.
    from . import tuning_rescale_threshold
    import cutlass
    from flashinfer.attention.prims_ts.variants.mla_decode.kernel.latency_1cta.config import (
        MlaProfile,
        compute_workspace_size,
    )
    from flashinfer.attention.prims_ts.variants.mla_decode.kernel.latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from flashinfer.attention.prims_ts.variants.mla_decode.kernel.throughput_2cta.config import (
        compute_workspace_size as workspace_2cta,
    )
    from flashinfer.attention.prims_ts.variants.mla_decode.kernel.throughput_2cta.kernel import (
        MlaDecodeTs,
    )

    if isinstance(kv_lora_rank, bool) or kv_lora_rank != 512:
        raise ValueError("sparse MLA requires kv_lora_rank=512")
    if not isinstance(assume_valid_prefix, bool):
        raise TypeError("assume_valid_prefix must be bool")
    device, _ = resolve_cuda_device(device)
    validate_runtime_device(device)
    for name, value in (
        ("batch_size", batch_size),
        ("num_qo_heads", num_qo_heads),
        ("max_q_len", max_q_len),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_qo_heads > 128:
        raise ValueError("num_qo_heads must be at most 128")
    if isinstance(qk_rope_head_dim, bool) or qk_rope_head_dim not in (0, 64):
        raise ValueError("qk_rope_head_dim must be 0 or 64")
    if qk_rope_head_dim and output_quant_block_size is not None:
        raise ValueError("fused output RoPE requires D512 attention")
    output_quant = make_sparse_output_quant(
        num_qo_heads, output_quant_block_size, out_scale_format, num_output_groups
    )
    for value in (max_topk, max_extra_topk):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("source capacities must be nonnegative integers")
    if max_topk == 0:
        raise ValueError("the primary selected-slot capacity must be positive")
    if q_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("native BF16 or E4M3 is required")
    if extra_kv_cache_format is None:
        extra_kv_cache_format = "nvfp4" if kv_cache_format == "fp8" else kv_cache_format
    validate_nvfp4_kv_formats(
        kv_cache_format,
        extra_kv_cache_format,
        qk_rope_head_dim,
        "e4m3" if q_dtype == torch.float8_e4m3fn else "bf16",
        1,
    )
    if kv_layout not in ("NHD", "HND"):
        raise ValueError("kv_layout must be NHD or HND")
    softmax_rescale_threshold_log2 = tuning_rescale_threshold(tuning)
    if softmax_rescale_threshold_log2 is not None:
        softmax_rescale_threshold_log2 = validate_softmax_rescale_threshold_log2(
            softmax_rescale_threshold_log2, q_dtype=q_dtype
        )
    max_rows = batch_size * max_q_len
    capacity = max(
        256,
        ((max_topk + 127) // 128 + (max_extra_topk + 127) // 128) * 128,
    )
    if max_rows * num_qo_heads >= 2**31 or capacity >= 2**31 - 32768:
        raise ValueError("query/route extent exceeds the int32 kernel domain")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    compute_capability = torch.cuda.get_device_capability(device)
    if compute_capability == (10, 7):
        # Imported here so this module loads without the CuTe DSL.
        from .kernel.device.nvfp4_kv import is_sm107_target

        if not is_sm107_target():
            # SM107 schedules assume the SM107 kernel resources gated on the
            # compile target; an sm_100f build keeps the generic schedule.
            compute_capability = None
    profile = select_sparse_mla_profile(
        rows=max_rows,
        heads=num_qo_heads,
        query_length=max_q_len,
        capacity=capacity,
        dtype="bf16" if q_dtype == torch.bfloat16 else "fp8",
        sm_count=sm_count,
        rope_dim=qk_rope_head_dim,
        kv_cache_format=kv_cache_format,
        compute_capability=compute_capability,
    )
    if (
        output_quant is not None
        and profile.family == "2cta"
        and q_dtype != torch.bfloat16
    ):
        # The staged quantized-output tile takes the SMEM of a third V stage.
        profile = replace(profile, kv_pipeline_stages=0)
    # Reloading V from L2 shortens native FP8 decode's producer lifetime.
    # Sustained grids amortize the retained-KV/CLC pipeline instead; packed
    # caches and two-source/sink plans retain their format-specific schedules.
    # The general path also exploits a hole-free prefix promise; the reload
    # loader keeps arbitrary-hole masks and does not use that optimization.
    if (
        qk_rope_head_dim == 64
        and kv_cache_format == "native"
        and q_dtype == torch.float8_e4m3fn
        and num_qo_heads in D576_HEADS
        and not max_extra_topk
        and not use_attention_sinks
        and not assume_valid_prefix
        and prefer_sparse_d576_reload(
            rows=max_rows,
            heads=num_qo_heads,
            query_length=max_q_len,
            capacity=max_topk,
            sm_count=sm_count,
            general_profile=profile,
        )
    ):
        return resolve_sparse_d576_reload_plan(
            device,
            max_rows,
            num_qo_heads,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            q_dtype=q_dtype,
            use_attention_sinks=use_attention_sinks,
            assume_valid_prefix=assume_valid_prefix,
            threshold=softmax_rescale_threshold_log2,
        )
    kernel_threshold = 0.0
    if softmax_rescale_threshold_log2 is not None:
        # An explicit threshold replaces the schedule's fixed six-binade
        # deferral; zero restores exact maxima for every schedule.
        profile = replace(profile, defer_max_update=False)
        kernel_threshold = softmax_rescale_threshold_log2
    family, tile = profile.family, profile.tile_size_q
    splits, head_dim_ctas = profile.split_kv, profile.head_dim_ctas
    dtype_name = "bf16" if q_dtype == torch.bfloat16 else "e4m3"
    if family == "2cta":
        active_clusters = sm_count // 2
        kernel = MlaDecodeTs(
            page_size=1,
            max_active_clusters=active_clusters,
            is_persistent=profile.scheduler != "nonpersistent",
            is_var_seq=False,
            is_var_split_kv=False,
            static_split_kv=splits,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            rope_dim=qk_rope_head_dim,
            num_heads=num_qo_heads,
            seq_len_q=1,
            batch_size=max_rows,
            mask_type="dense",
            softmax_rescale_threshold_log2=kernel_threshold,
            # Sparse MLA flattens Q rows into virtual batch items, so the
            # physical Q length stays compile-time.
            fixed_seqlen=True,
        )
        kernel.kv_cache_format = kv_cache_format
        kernel.extra_kv_cache_format = extra_kv_cache_format
        kernel.sparse_gather_warps = profile.gather_issue_warps
        kernel.sparse_kv_stages = profile.kv_pipeline_stages
        kernel.sparse_uniform_pages = profile.uniform_offset_cache
        kernel.sparse_balanced_registers = profile.balanced_registers
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
        kernel_profile = MlaProfile(
            name="sparse",
            kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
            tile_size_q=tile,
            num_ctas_per_head_dim=head_dim_ctas,
            num_ctas_per_seq_kv=splits,
            use_multi_ctas_kv=int(splits > 1),
            use_cluster_reduction=int(profile.reduction == "cluster"),
            use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
            use_clc_dynamic_persistent_scheduler=int(profile.scheduler == "clc"),
        )
        kernel = ThroughputLatencyMlaDecodeTs(
            batch_size=max_rows,
            num_heads=tile,
            seq_len_q=(num_qo_heads + tile - 1) // tile,
            seq_len_k=capacity,
            rope_dim=qk_rope_head_dim,
            page_size=1,
            max_active_clusters=sm_count,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            profile=kernel_profile,
            reduction_mode=profile.reduction,
            logical_num_heads=num_qo_heads,
            logical_seq_len_q=1,
            tile_size_q=tile,
            sparse_profile=profile,
            mask_type="dense",
            softmax_rescale_threshold_log2=kernel_threshold,
            kv_cache_format=kv_cache_format,
            extra_kv_cache_format=extra_kv_cache_format,
        )
        core_bytes = compute_workspace_size(
            cfg=kernel.make_config(),
            partial_o_dtype=cutlass.BFloat16,
            lse_dtype=cutlass.Float32,
        )
    spec = make_mla_decode_compile_spec(
        MLADecodeLaunchSpec(kernel, (), core_bytes, splits),
        device_index=device.index,
        num_qo_heads=num_qo_heads,
        kv_lora_rank=512,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=1,
        q_dtype_key=str(q_dtype).removeprefix("torch."),
        out_dtype_key="float8_e4m3fn" if output_quant is not None else "bfloat16",
        max_q_len=1,
        packed_query=False,
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        ("core", (core_bytes,), torch.int8),
        ("public_lse", (max_rows, num_qo_heads), torch.float32),
    ):
        sections[name], byte_end = append_workspace_section(byte_end, shape, dtype)
    return device, profile, spec, sections, byte_end, capacity, output_quant


def sparse_d576_workspace_bytes(max_rows: int, num_heads: int, split_kv: int) -> int:
    """Split-KV partials of a D576 plan: BF16 O and FP32 LSE per split row."""
    if split_kv <= 1:
        return 0
    return max_rows * num_heads * split_kv * (D576_V_DIM * 2 + 4)
