from typing import Optional, Tuple, Dict, Any
from .generator_utils import spec_fields, kernel_spec
from collections import namedtuple
from enum import IntEnum
from ... import env as jit_env
from dataclasses import dataclass, asdict
import math
import torch

from .utils import (
    get_effective_sm_and_name,
    get_hopper_instruction_traits,
    get_reg_count,
    enable_mutex,
    enable_tma_store,
    selected_mask_types,
    pythonBoolean2cpp,
    dtype2bytes,
    dtype2traits,
    hopper_dtype2traits,
    MAX_STGS_PER_LOOP,
    sm2name,
    dtype2OutputType,
    hopper_traits2shape,
    dtype2typename,
    AttentionMaskType,
    InputLayout,
    encode_name,
    copyright,
)

import jinja2


def select_kv_loop_step(head_size: int) -> int:
    """
    Select the KV loop step based on head size.

    For warp-specialized Hopper kernels:
    - Small heads (32-64): 256 step for better occupancy
    - Medium heads (72-128): 128 step
    - Large heads (160-256): 64 step to fit in registers
    """
    if head_size <= 64:
        return 256
    elif head_size <= 128:
        return 128
    else:
        return 64


@dataclass(frozen=True)
class FMHAv2KernelSpec:
    sm: int
    dtype: str
    seq_len: int
    head_size: int
    warps_m: int
    warps_n: int
    version: int
    interleaved: bool
    ldgsts_q: int
    ldgsts_k: int
    ldgsts_v: int
    share_smem_k_v: bool
    loop_step: int
    has_noloop: bool
    noloop_step: int
    unroll_threshold: int
    has_scale_max: bool
    ctas_per_head: int = 1
    sm_mma: int = 1
    head_interleaved: bool = True
    flash_attention: bool = False
    kv_loop_step: int = 64
    limit_qk_fragments: bool = False
    limit_v_fragments: bool = False
    tiled: int = 0
    warp_specialization: bool = False
    q_tile_buffers: int = 1
    kv_tile_buffers: int = 1
    scheduling_mode: int = 0
    input_layout: InputLayout = InputLayout.PACKED_QKV
    cross_mha: int = 0
    alibi: bool = True
    enable_attn_logit_softcapping: bool = False
    return_softmax_stats: bool = False
    disabled_mask_types: Optional[Tuple[int]] = None
    head_size_v: int = 0
    sage_block_sizes: Optional[Tuple[int]] = None
    output_dtype: Optional[str] = None
    is_mtp: bool = False


# BF16-QKV+BF16-out and BF16-Q + FP8-KV + BF16-out (or FP8-QKV+BF16-out)


def select_ldgsts(sm: int, warp_specialization: bool, head_size: int, dtype: str):
    # TODO(jimmyzho): Implement this
    return (False, False, False)
    # if warp_specialization:
    #     return (False, False, False)
    # elif sm == 120:)
    #     if dtype == "fp16":
    #         # tune ldgsts
    #         ldgsts_q = True
    #         ldgsts_k = True
    #         ldgsts_v = True
    #         if head_size >= 256:
    #             ldgsts_k = False
    #             ldgsts_v = False
    #         if head_size > 256:
    #             ldgsts_q = False
    #             ldgsts_k = False
    #             ldgsts_v = False
    #     elif dtype == "e4m3":
    #         pass
    #     return (ldgsts_q, ldgsts_k, ldgsts_v)
    # else:
    #     raise ValueError(f"Unsupported SM version: {sm}")


def generate_kernel_spec(
    sm: int,
    head_size: int,
    dtype: str,
    return_softmax: Optional[bool] = False,
    enable_attn_logit_softcapping: Optional[bool] = False,
    alibi: Optional[bool] = True,
    is_mla: Optional[bool] = False,
    head_size_v: Optional[int] = 0,
    input_layout: Optional[InputLayout] = InputLayout.Q_PAGED_KV,
    output_dtype: Optional[str] = None,
):
    """
    Args:
        sm: int -GPU SM version (90, 120)
        head_size: int - Q/K head dimension
        dtype: str - Data type ("fp16", "bf16", "e4m3", "e4m3_fp32")
        return_softmax: Optional[bool] = False
        is_mla: Optional[bool] = False - MLA mode
        head_size_v: Optional[int] = 0 - V head dimension if is_mla
    """

    """
    Non-default values need to be set:
    # warps_m, warps_n, version, interleaved, ldgsts_q, ldgsts_k, ldgsts_v,
    # share_smem_k_v, loop_step, has_noloop, noloop_step, unroll_threshold,
    # has_scale_max, sm_mma,

    Default values need to be set:
    # flash_attention, kv_loop_step, limit_qk_fragments, limit_v_fragments, tiled, warp_specialization, q_tile_buffers, kv_tile_buffers

    User passed values:
    # sm, dtype, seq_len, head_size, head_size_v,
    # alibi, enable_attn_logit_softcapping, return_softmax_stats, is_mtp
    """
    # Fixed variables for all kernels
    warps_m, warps_n = 4, 1
    version = 2
    interleaved = False
    share_smem_k_v = False
    unroll_threshold = 1
    has_scale_max = False
    warp_specialization = sm == 90 and head_size >= 32
    scheduling_mode = 1
    seq_len = 0
    flash_attention = True
    # input_layout is passed as parameter, default is Q_PAGED_KV

    ldgsts_q, ldgsts_k, ldgsts_v = select_ldgsts(
        sm, warp_specialization, head_size, dtype
    )
    # enumerate_hgmma_flash_warpspec_kernels, enumerate_qgmma_flash_warpspec_kernels
    if warp_specialization:
        sm_mma = 90
        loop_step = 64
        has_noloop = 0
        noloop_step = 64
        limit_qk_fragments = False
        limit_v_fragments = False
        tiled = 0
        q_tile_buffers = 1
        kv_tile_buffers = 2

        if dtype in ["fp16", "bf16"]:
            # enumerate_hgmma_flash_warpspec_kernels
            if head_size <= 64:
                kv_loop_step = 256
            elif head_size <= 128:
                kv_loop_step = 128
            else:
                kv_loop_step = 64
        elif dtype == "e4m3":
            # enumerate_qgmma_flash_warpspec_kernels
            if head_size <= 64:
                kv_tile_buffers = 4
            if head_size <= 128:
                kv_loop_step = 256
            else:
                kv_loop_step = 128
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    elif sm == 120:
        # enumerate_hmma_flash_kernels, enumerate_qmma_flash_kernels
        sm_mma = 80
        loop_step = 128 if head_size <= 64 else 64
        has_noloop = 1
        noloop_step = 64
        limit_qk_fragments = False
        limit_v_fragments = False
        if dtype in ["fp16", "bf16"]:
            if head_size <= 64:
                q_loop_step = 128
                kv_loop_step = 128
            elif head_size <= 256:
                q_loop_step = 64
                kv_loop_step = 128
            elif head_size <= 512:
                q_loop_step = 64
                kv_loop_step = 64
            else:
                raise ValueError(f"Unsupported head size: {head_size}")
            noloop_step = q_loop_step
            loop_step = q_loop_step
            tiled = 1
        elif dtype == "e4m3":
            if is_mla:
                # MLA kernels. (TODO)
                # ((192, 128), (64, 64), 1),
                # ((576, 512), (64, 64), 1),
                pass
            else:
                tiled = 0
                if head_size <= 64:
                    q_loop_step = 128
                    kv_loop_step = 128
                elif head_size <= 256:
                    q_loop_step = 64
                    kv_loop_step = 32
                loop_step = q_loop_step
                has_noloop = 1
                noloop_step = q_loop_step
                tiled = 0
    elif sm == 90:
        raise ValueError(f"(jimmyzho): Only Warp Specialization is supported for SM 90")

    spec = {
        "sm": sm,
        "dtype": dtype,
        "seq_len": seq_len,
        "head_size": head_size,
        "head_size_v": head_size_v,
        "warps_m": warps_m,
        "warps_n": warps_n,
        "version": version,
        "interleaved": interleaved,
        "ldgsts_q": ldgsts_q,
        "ldgsts_k": ldgsts_k,
        "ldgsts_v": ldgsts_v,
        "share_smem_k_v": share_smem_k_v,
        "loop_step": loop_step,
        "has_noloop": has_noloop,
        "noloop_step": noloop_step,
        "unroll_threshold": unroll_threshold,
        "has_scale_max": has_scale_max,
        "sm_mma": sm_mma,
        "flash_attention": flash_attention,
        "kv_loop_step": kv_loop_step,
        "limit_qk_fragments": limit_qk_fragments,
        "limit_v_fragments": limit_v_fragments,
        "tiled": tiled,
        "warp_specialization": warp_specialization,
        "q_tile_buffers": q_tile_buffers,
        "kv_tile_buffers": kv_tile_buffers,
        "scheduling_mode": scheduling_mode,
        "input_layout": input_layout,
        "alibi": alibi,
        "enable_attn_logit_softcapping": enable_attn_logit_softcapping,
        "return_softmax_stats": return_softmax,
        "output_dtype": output_dtype,
        "is_mtp": is_mla,
    }
    return FMHAv2KernelSpec(**spec)


def is_kernel_spec_valid(kspec: FMHAv2KernelSpec) -> bool:
    if kspec.alibi and kspec.enable_attn_logit_softcapping:
        return False

    # Standard flash attention support
    flash_valid = (
        kspec.sm in [80, 86, 89, 90, 120]
        and kspec.dtype in ["fp16", "bf16", "fp16_fp32", "e4m3", "e4m3_fp32"]
        and kspec.head_size <= 256
        and kspec.head_size_v == 0
        and kspec.sage_block_sizes is None
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and kspec.input_layout != InputLayout.SEPARATE_Q_K_V
    )
    # SM90 non-flash ldgsts support (fixed seq len)
    non_flash_valid = (
        kspec.sm == 90
        and kspec.dtype in ["fp16", "bf16", "fp16_fp32"]
        and kspec.head_size <= 256
        and kspec.ldgsts_q
        and kspec.version == 2
        and not kspec.cross_mha
        and not kspec.flash_attention
    )
    # Clip/SigLip support
    clip_valid = (
        kspec.sm == 100
        and kspec.dtype in ["fp16", "bf16", "fp16_fp32", "e4m3", "e4m3_fp32"]
        and kspec.head_size == 80
        and kspec.head_size_v == 0
        and kspec.sage_block_sizes is None
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and kspec.input_layout != InputLayout.SEPARATE_Q_K_V
    )
    # Deepseek MLA (generation 576/512 paged)
    mla_valid_576_512 = (
        kspec.sm in [90, 100, 120]
        and kspec.dtype in ["bf16", "e4m3_fp32"]
        and kspec.head_size == 576
        and kspec.head_size_v == 512
        and kspec.input_layout == InputLayout.Q_PAGED_KV
        and kspec.sage_block_sizes is None
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and not kspec.warp_specialization
        and kspec.tiled
    )
    # Deepseek MLA (context 192/128 separate-q-k-v)
    mla_valid_192_128 = (
        kspec.sm in [90, 100, 120]
        and kspec.dtype in ["bf16", "e4m3", "e4m3_fp32"]
        and kspec.head_size == 192
        and kspec.head_size_v == 128
        and kspec.input_layout == InputLayout.SEPARATE_Q_K_V
        and kspec.sage_block_sizes is None
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and (
            (kspec.warp_specialization and not kspec.alibi)  # sm90
            or (not kspec.warp_specialization and kspec.tiled)
        )  # non-sm90
        and not kspec.enable_attn_logit_softcapping
    )
    # SageAttention (warp_spec, head_size in (80, 128), packed QKV, padding mask)
    sage_valid_sm90 = (
        kspec.sm == 90
        and kspec.head_size in [80, 128]
        and kspec.version == 2
        and kspec.sage_block_sizes in [(64, 64, 256)]
        and not kspec.cross_mha
        and kspec.flash_attention
        and kspec.warp_specialization
        and kspec.input_layout == InputLayout.PACKED_QKV
        and not kspec.alibi
        and not kspec.enable_attn_logit_softcapping
    )
    # SageAttention on Ada (head_size in (80, 128), packed QKV, padding mask)
    sage_valid_sm89 = (
        kspec.sm == 89
        and kspec.head_size in [80, 128]
        and kspec.sage_block_sizes in [(64, 32, 32)]
        and kspec.output_dtype in ["fp16", "bf16"]
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and not kspec.warp_specialization
        and kspec.input_layout == InputLayout.PACKED_QKV
    )

    print(f"flash_valid: {flash_valid}")
    print(f"non_flash_valid: {non_flash_valid}")
    print(f"clip_valid: {clip_valid}")
    print(f"mla_valid_576_512: {mla_valid_576_512}")
    print(f"mla_valid_192_128: {mla_valid_192_128}")
    print(f"sage_valid_sm90: {sage_valid_sm90}")
    print(f"sage_valid_sm89: {sage_valid_sm89}")
    return (
        flash_valid
        or non_flash_valid
        or clip_valid
        or mla_valid_576_512
        or mla_valid_192_128
        or sage_valid_sm90
        or sage_valid_sm89
    )


def get_kernel_code(kspec, kname, lname):
    min_cuda_version = 0  # no restriction

    # The architecture that determines the instruction.
    effective_sm, sm_name = get_effective_sm_and_name(kspec)

    if effective_sm >= 80:
        min_cuda_version = 11000

    launcher_name = lname
    causal_kernel_name = kname.replace("__placeholder__", "_causal")
    custom_mask_kernel_name = kname.replace("__placeholder__", "_custom_mask")
    sliding_or_chunked_causal_kernel_name = kname.replace(
        "__placeholder__", "_sliding_or_chunked_causal"
    )
    kernel_name = kname.replace("__placeholder__", "")

    # FIXME: use separate parameters when generating cubins for trtllm.
    if not kspec.cross_mha:
        params_type = "bert::Fused_multihead_attention_params_v{}".format(kspec.version)
    else:
        params_type = "bert::Fused_multihead_attention_params_mhca"

    if effective_sm < 90:
        instruction_traits = sm_name.capitalize() + "_" + dtype2traits[kspec.dtype]
    elif effective_sm == 90:
        instruction_traits = (
            sm_name.capitalize() + "_" + hopper_dtype2traits[kspec.dtype]
        )
        # for hopper, we differentiate instruction_traits_o and instruction_traits_p
        instruction_traits_p, instruction_traits_o = get_hopper_instruction_traits(
            instruction_traits, kspec
        )
        # print(instruction_traits_p, instruction_traits_o)

    if effective_sm < 90:
        if kspec.flash_attention:
            kernel_variant = "flash_attention"
        else:
            kernel_variant = "1xN" if kspec.warps_m == 1 else "2x2"
    elif effective_sm == 90:
        if kspec.warps_n > 1:
            # for hopper we slice the problem along the M dim.
            kernel_variant = "4xN" + "_hopper"
        else:
            kernel_variant = "4x1" + "_hopper"

    if effective_sm < 90:
        kernel_traits = "Kernel_traits_"
    elif effective_sm == 90:
        kernel_traits = "FMHA_kernel_traits_hopper_"

    if kspec.interleaved:
        kernel_traits += "interleaved_v2"
    elif kspec.cross_mha:
        kernel_traits += "fmhca"
    else:
        kernel_traits += "v{}".format(kspec.version)

    # decide whether to paged_kv kernel traits for ampere-style kernels.
    if effective_sm < 90:
        if kspec.input_layout == InputLayout.Q_PAGED_KV:
            kernel_traits += "_paged_kv_cache"
        elif kspec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            kernel_traits += "_contiguous_kv_cache"
        elif kspec.input_layout == InputLayout.SEPARATE_Q_K_V:
            kernel_traits += "_q_k_v"

    flags = 0
    if kspec.ldgsts_q:
        flags |= 1
    if kspec.ldgsts_k:
        flags |= 2
    if kspec.ldgsts_v:
        flags |= 4
    if kspec.share_smem_k_v and not kspec.limit_qk_fragments:
        flags |= 8
    if kspec.has_scale_max:
        flags |= 16
    if not kspec.head_interleaved:
        flags |= 32
    if kspec.limit_qk_fragments:
        flags |= 128
    if kspec.limit_v_fragments:
        flags |= 256
    if kspec.has_noloop:
        # NOTE do not use flags 512 = 0x200 as it is reserved; do not add to flags because it
        # will be selectively added to no-loop kernel trait upon generating .cu templates
        pass
    if kspec.enable_attn_logit_softcapping:
        flags |= 2048
    if kspec.tiled:
        flags |= 4096
    if kspec.is_mtp:
        flags |= 8192

    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask = (
        selected_mask_types(kspec)
    )

    if any(
        selected_mask_flag == "1" for selected_mask_flag in selected_mask_types(kspec)
    ):
        padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask = (
            selected_mask_types(kspec)
        )
    else:
        return None

    kernel_flags = "0x{:02x}u".format(flags)

    heads_interleaved_flag = pythonBoolean2cpp[kspec.head_interleaved]

    disable_fadd_trick = (
        1 if effective_sm >= 86 else 0
    )  # this will force generating F2IP

    enable_mutex_flag = enable_mutex(kspec)

    has_alibi = pythonBoolean2cpp[kspec.alibi]

    input_layout_flag = str(int(kspec.input_layout))

    run_fct_name = (
        "run_packed_qkv"
        if kspec.input_layout == InputLayout.PACKED_QKV
        else "run_separate_q_and_kv"
    )

    dma_reg_count, compute_reg_count = get_reg_count(kspec)

    use_tma_store_flag = enable_tma_store(kspec)

    enable_attn_logit_softcapping_flag = pythonBoolean2cpp[
        kspec.enable_attn_logit_softcapping
    ]

    return_softmax_stats_flag = pythonBoolean2cpp[kspec.return_softmax_stats]

    # needed by warpspec kernels.
    fp8_kernel = kspec.dtype in ["e4m3", "e4m3_fp32"]
    kernel_traits_header = (
        "fmha::ws::Kernel_traits_Hopper_qgmma_e4m3_fp32<"
        if fp8_kernel
        else f"fmha::ws::Kernel_traits<fmha::{instruction_traits},"
    )

    # output type.
    output_dtype_ = f"fmha::{dtype2OutputType[kspec.output_dtype if kspec.output_dtype is not None else kspec.dtype]}"

    # sage attention block sizes.
    sage_block_size_q = 0
    sage_block_size_k = 0
    sage_block_size_v = 0
    if fp8_kernel and kspec.sage_block_sizes:
        assert kspec.output_dtype is not None, (
            "output_dtype must be specified for fp8 sage attention kernels"
        )
        sage_block_size_q = kspec.sage_block_sizes[0]
        sage_block_size_k = kspec.sage_block_sizes[1]
        sage_block_size_v = kspec.sage_block_sizes[2]

    # Following is taken from generation.py
    TMA_config = r"""
    // TMA configuration
    // Note that this may only need to init once during inference (for different layers)
    // Reuse the same traits for initializing tma descriptors.
    fmha::ws::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, stream);
    """
    params_str = "params"
    attn_mask_type_str = "using Attention_mask_type = fmha::Attention_mask_type;"
    bert_launch_params = (
        "using Launch_params = bert::Fused_multihead_attention_launch_params;"
    )
    include_str = ""
    num_compute_groups_str = "static constexpr int NUM_COMPUTE_GROUPS = 2;"
    fused_multihead_attention_params_v2_str = f"{params_type}"
    const_fused_multihead_attention_params_v2_str = f"const {params_type}"
    setmaxnreg_dma_str = r"""
        const int DMA_REG_COUNT = {dma_reg_count};
        asm volatile("{{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}}" ::"n"(DMA_REG_COUNT));""".format(
        dma_reg_count=dma_reg_count
    )
    setmaxnreg_compute_str = r"""
        const int COMPUTE_REG_COUNT = {compute_reg_count};
        asm volatile("{{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}}" ::"n"(COMPUTE_REG_COUNT));""".format(
        compute_reg_count=compute_reg_count
    )
    local_ns_open = ""
    local_ns_close = ""

    tmp = dict(locals(), **asdict(kspec))

    template_dir = jit_env.FLASHINFER_CSRC_DIR / "fmha_v2" / "templates"
    if effective_sm < 90:
        if kspec.flash_attention:
            with open(template_dir / "fa_kernel.jinja", "r") as f:
                template = jinja2.Template(f.read())

            tmp["MAX_STGS_PER_LOOP"] = MAX_STGS_PER_LOOP
            tmp["use_multi_cta"] = False
            code = template.render(tmp)
        else:
            with open(template_dir / "kernel.jinja", "r") as f:
                template = jinja2.Template(f.read())
            tmp["MAX_STGS_PER_LOOP"] = MAX_STGS_PER_LOOP
            use_multi_cta = 1 if kspec.ctas_per_head > 1 else 0
            tmp["use_multi_cta"] = use_multi_cta
            code = template.render(tmp)
    elif effective_sm == 90:
        use_tma = 1
        if kspec.ldgsts_q:
            use_tma = 0
        if kspec.warp_specialization:
            with open(template_dir / "kernel_hopper_ws.jinja", "r") as f:
                template = jinja2.Template(f.read())
            tmp["use_tma"] = use_tma
            tmp["bytes_per_elt"] = dtype2bytes[kspec.dtype]
            code = template.render(tmp)
        else:
            with open(template_dir / "kernel_hopper.jinja", "r") as f:
                template = jinja2.Template(f.read())
            tmp["use_tma"] = use_tma
            code = template.render(tmp)
    else:
        raise RuntimeError("No template found for this configuration.")
    return code


def generate_jit_sources(
    sm: int,
    head_size: int,
    dtype: str,
    input_layout: InputLayout = InputLayout.Q_PAGED_KV,
    return_softmax: Optional[bool] = False,
    enable_attn_logit_softcapping: Optional[bool] = False,
    alibi: Optional[bool] = True,
    is_mla: Optional[bool] = False,
    head_size_v: Optional[int] = 0,
    output_dtype: Optional[str] = None,
) -> dict:
    """
    Generate JIT source code for FMHAv2 kernel.

    Parameters
    ----------
    sm : int
        GPU SM version (90, 120)
    head_size : int
        Q/K head dimension
    dtype : str
        Data type string: "fp16", "bf16", "e4m3", "e4m3_fp32"
    input_layout : InputLayout
        Input layout enum
    return_softmax : bool
        Return softmax statistics
    enable_attn_logit_softcapping : bool
        Enable logit softcapping
    alibi : bool
        Enable ALiBi positional encoding
    is_mla : bool
        MLA mode (different head sizes for Q/K and V)
    head_size_v : int
        V head dimension (0 = same as head_size)
    output_dtype : Optional[str]
        Output dtype string (None = same as input dtype)

    Returns
    -------
    dict
        Dictionary with kernel_code, dispatcher_code, kernel_filename, launcher_name, kernel_name
    """
    kspec = generate_kernel_spec(
        sm=sm,
        head_size=head_size,
        dtype=dtype,
        return_softmax=return_softmax,
        enable_attn_logit_softcapping=enable_attn_logit_softcapping,
        alibi=alibi,
        input_layout=input_layout,
        head_size_v=head_size_v,
        output_dtype=output_dtype,
    )
    print(f"kspec: {kspec}")
    if not is_kernel_spec_valid(kspec):
        raise ValueError(f"Invalid kernel spec: {kspec}")
    fname, lname, kname = encode_name(kspec)
    kernel_code = get_kernel_code(kspec, kname, lname)
    if kernel_code is None:
        raise ValueError(f"Failed to generate kernel code for spec: {kspec}")

    with open(jit_env.FLASHINFER_CSRC_DIR / "fmha_v2_dispatcher.jinja", "r") as f:
        dispatcher_template = jinja2.Template(f.read())
    dispatcher_code = dispatcher_template.render(launcher_name=lname)

    return {
        "kernel_code": kernel_code,
        "dispatcher_code": dispatcher_code,
        "kernel_filename": fname,
        "launcher_name": lname,
        "kernel_name": kname,
    }
