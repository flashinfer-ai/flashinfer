import itertools
from typing import Optional, Tuple
from ... import env as jit_env
from dataclasses import dataclass, asdict
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
    dtype2OutputType,
    InputLayout,
    encode_name,
    dtype2typename,
    copyright,
)

from ...utils import write_if_different

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
) -> FMHAv2KernelSpec:
    """
    Generate a kernel spec for FMHAv2.

    Args:
        sm: GPU SM version (90, 120)
        head_size: Q/K head dimension
        dtype: Data type ("fp16", "bf16", "e4m3", "e4m3_fp32")
        return_softmax: Return softmax statistics
        enable_attn_logit_softcapping: Enable logit softcapping
        alibi: Enable ALiBi positional encoding
        is_mla: MLA mode (different head sizes for Q/K and V)
        head_size_v: V head dimension (0 = same as head_size)
        input_layout: Input layout enum
        output_dtype: Output dtype string
    """
    # Initialize spec with required fields (no class defaults)
    # and user-provided optional fields
    spec = {
        # Required fields
        "sm": sm,
        "dtype": dtype,
        "seq_len": 0,
        "head_size": head_size,
        "warps_m": 4,
        "warps_n": 1,
        "version": 2,
        "interleaved": False,
        "share_smem_k_v": False,
        "unroll_threshold": 1,
        "has_scale_max": False,
        # head_interleaved=False means input layout [tokens, 3, H, D] (not [tokens, H, 3, D])
        # This matches the Python API's expected format and TRT-LLM convention
        "head_interleaved": False,
        # User-provided values (override class defaults if different)
        "input_layout": input_layout,
        "alibi": alibi,
        "enable_attn_logit_softcapping": enable_attn_logit_softcapping,
        "return_softmax_stats": return_softmax,
        "head_size_v": head_size_v,
        "output_dtype": output_dtype,
        "is_mtp": is_mla,
    }

    # Compute ldgsts flags
    warp_specialization = sm == 90 and head_size >= 32
    ldgsts_q, ldgsts_k, ldgsts_v = select_ldgsts(
        sm, warp_specialization, head_size, dtype
    )
    spec["ldgsts_q"] = ldgsts_q
    spec["ldgsts_k"] = ldgsts_k
    spec["ldgsts_v"] = ldgsts_v

    # Override class defaults that always differ
    spec["flash_attention"] = True  # Class default is False
    spec["scheduling_mode"] = 1  # Class default is 0

    # SM-specific configuration
    if warp_specialization:
        spec["warp_specialization"] = True
        spec["sm_mma"] = 90
        spec["loop_step"] = 64
        spec["has_noloop"] = 0
        spec["noloop_step"] = 64
        spec["kv_tile_buffers"] = 2  # Class default is 1

        if dtype in ["fp16", "bf16"]:
            if head_size <= 64:
                spec["kv_loop_step"] = 256
            elif head_size <= 128:
                spec["kv_loop_step"] = 128
            # else: use class default 64
        elif dtype == "e4m3":
            if head_size <= 64:
                spec["kv_tile_buffers"] = 4
            if head_size <= 128:
                spec["kv_loop_step"] = 256
            else:
                spec["kv_loop_step"] = 128
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    elif sm == 120:
        spec["sm_mma"] = 80
        spec["has_noloop"] = 1
        spec["noloop_step"] = 64
        spec["loop_step"] = 128 if head_size <= 64 else 64

        if dtype in ["fp16", "bf16"]:
            if head_size <= 64:
                q_loop_step = 128
                spec["kv_loop_step"] = 128
            elif head_size <= 256:
                q_loop_step = 64
                spec["kv_loop_step"] = 128
            elif head_size <= 512:
                q_loop_step = 64
                # kv_loop_step uses class default 64
            else:
                raise ValueError(f"Unsupported head size: {head_size}")
            spec["noloop_step"] = q_loop_step
            spec["loop_step"] = q_loop_step
            spec["tiled"] = 1  # Class default is 0
        elif dtype == "e4m3":
            if is_mla:
                # MLA kernels (TODO)
                pass
            else:
                if head_size <= 64:
                    q_loop_step = 128
                    spec["kv_loop_step"] = 128
                elif head_size <= 256:
                    q_loop_step = 64
                    spec["kv_loop_step"] = 32
                else:
                    q_loop_step = 64
                spec["loop_step"] = q_loop_step
                spec["noloop_step"] = q_loop_step

    elif sm == 90:
        raise ValueError("(jimmyzho): Only Warp Specialization is supported for SM 90")

    return FMHAv2KernelSpec(**spec)


def is_kernel_spec_valid(kspec: FMHAv2KernelSpec) -> bool:
    if kspec.alibi and kspec.enable_attn_logit_softcapping:
        return False

    # Standard flash attention support
    flash_valid: bool = (
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
    non_flash_valid: bool = (
        kspec.sm == 90
        and kspec.dtype in ["fp16", "bf16", "fp16_fp32"]
        and kspec.head_size <= 256
        and bool(kspec.ldgsts_q)
        and kspec.version == 2
        and not kspec.cross_mha
        and not kspec.flash_attention
    )
    # Clip/SigLip support
    clip_valid: bool = (
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
    mla_valid_576_512: bool = (
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
        and bool(kspec.tiled)
    )
    # Deepseek MLA (context 192/128 separate-q-k-v)
    mla_valid_192_128: bool = (
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
            or (not kspec.warp_specialization and bool(kspec.tiled))
        )  # non-sm90
        and not kspec.enable_attn_logit_softcapping
    )
    # SageAttention (warp_spec, head_size in (80, 128), packed QKV, padding mask)
    sage_valid_sm90: bool = (
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
    sage_valid_sm89: bool = (
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
    # SM90 warp-specialized flash attention with SEPARATE_Q_K_V layout
    # Supports standard attention (head_size_v == 0 means same as head_size)
    flash_separate_qkv_valid: bool = (
        kspec.sm == 90
        and kspec.dtype in ["fp16", "bf16"]
        and kspec.head_size <= 256
        and kspec.head_size_v == 0
        and kspec.sage_block_sizes is None
        and kspec.version == 2
        and not kspec.cross_mha
        and kspec.flash_attention
        and kspec.warp_specialization
        and kspec.input_layout == InputLayout.SEPARATE_Q_K_V
        and not kspec.enable_attn_logit_softcapping
    )

    # print(f"flash_valid: {flash_valid}")
    # print(f"non_flash_valid: {non_flash_valid}")
    # print(f"clip_valid: {clip_valid}")
    # print(f"mla_valid_576_512: {mla_valid_576_512}")
    # print(f"mla_valid_192_128: {mla_valid_192_128}")
    # print(f"sage_valid_sm90: {sage_valid_sm90}")
    # print(f"sage_valid_sm89: {sage_valid_sm89}")
    return (
        flash_valid
        or non_flash_valid
        or clip_valid
        or mla_valid_576_512
        or mla_valid_192_128
        or sage_valid_sm90
        or sage_valid_sm89
        or flash_separate_qkv_valid
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


def get_api_code(specs_names):
    def get_signature(lname, version, cross_mha, use_tma):
        # The architecture that determines the instruction.
        effective_sm, sm_name = get_effective_sm_and_name(kspec)
        if cross_mha:
            return "void {}(const Params_mhca &params, cudaStream_t stream);".format(
                lname
            )
        elif effective_sm >= 90:
            # need to set tma desc in params
            return "void {}(Params_v{} &params, const Launch_params &launch_params, cudaStream_t stream);".format(
                lname, version
            )
        else:
            return "void {}(const Params_v{} &params, const Launch_params &launch_params, cudaStream_t stream);".format(
                lname, version
            )

    signatures = []
    for kspec, _fname, lname, _kname in specs_names:
        effective_sm, _ = get_effective_sm_and_name(kspec)
        use_tma = effective_sm == 90 and not kspec.ldgsts_q
        signatures.append(get_signature(lname, kspec.version, kspec.cross_mha, use_tma))
        if kspec.has_noloop and not kspec.tiled:
            signatures.append(
                get_signature(lname + "_nl", kspec.version, kspec.cross_mha, use_tma)
            )
        elif kspec.tiled:
            signatures.append(
                get_signature(
                    lname + "_nl_tiled", kspec.version, kspec.cross_mha, use_tma
                )
            )
        if not kspec.warp_specialization:
            signatures.append("void {}_get_max_heads_per_wave(int*);".format(lname))
    signatures = "\n".join(signatures)

    # v1
    # - normal
    # - no loop
    # v2
    # - normal
    # - no loop
    # - normal interleaved
    # - no loop interleaved
    # - flash attention no loop
    # - flash attention no loop tiled
    # - flash attention warp_specialized (on Hopper)

    def gen_unroll_check(kspec):
        code = "if (!{has_noloop} || (!force_unroll && (ignore_b1opt || b > {unroll_threshold})))".format(
            **asdict(kspec)
        )
        if kspec.flash_attention:
            code = "if (!{has_noloop} || (!force_unroll && (ignore_b1opt || b * h > {unroll_threshold})))".format(
                **asdict(kspec)
            )
        return code

    def gen_call(kspec, lname):
        effective_sm, _ = get_effective_sm_and_name(kspec)
        data_type = dtype2typename[kspec.dtype]
        output_data_type = data_type
        if kspec.output_dtype:
            output_data_type = dtype2typename[kspec.output_dtype]
        il_check = ""
        if kspec.version == 2 and kspec.dtype in ["fp16", "bf16"]:
            il_check += (
                "&& use_flash_attention "
                if kspec.flash_attention
                else "&& !use_flash_attention "
            )
        if kspec.version == 2:
            # attention input layout.
            il_check += f"&& attention_input_layout == {kspec.input_layout.value} "
            # interleaved layout or not.
            il_check += "&& interleaved " if kspec.interleaved else "&& !interleaved "
            if effective_sm == 90:
                il_check += "&& !use_tma " if kspec.ldgsts_q else "&& use_tma "
                il_check += (
                    "&& warp_specialization "
                    if kspec.warp_specialization
                    else "&& !warp_specialization "
                )
            else:
                il_check += "&& !warp_specialization && !use_tma "
            # Different accumulation types.
            if "_fp32" in kspec.dtype or "bf16" in kspec.dtype or kspec.dtype == "e4m3":
                il_check += "&& force_fp32_acc "
            else:
                il_check += "&& !force_fp32_acc "
            # whether support alibi or not.
            if kspec.warp_specialization:
                il_check += (
                    "&& params.has_alibi " if kspec.alibi else "&& !params.has_alibi "
                )
                il_check += (
                    "&& params.softmax_stats_ptr != nullptr "
                    if kspec.return_softmax_stats
                    else "&& params.softmax_stats_ptr == nullptr "
                )
            # use enable_attn_logit_softcapping or not.
            il_check += (
                "&& enable_attn_logit_softcapping "
                if kspec.enable_attn_logit_softcapping
                else "&& !enable_attn_logit_softcapping "
            )
            # check sage block sizes
            sage_block_size_q = 0
            sage_block_size_k = 0
            sage_block_size_v = 0
            if kspec.sage_block_sizes:
                # override the data_type to output type, otherwise it is always E4M3
                data_type = output_data_type
                sage_block_size_q = kspec.sage_block_sizes[0]
                sage_block_size_k = kspec.sage_block_sizes[1]
                sage_block_size_v = kspec.sage_block_sizes[2]
            il_check += (
                f"&& sage_block_size_q == {sage_block_size_q} "
                f"&& sage_block_size_k == {sage_block_size_k} "
                f"&& sage_block_size_v == {sage_block_size_v} "
            )

        il_check += (
            "&& params.use_int8_scale_max "
            if kspec.has_scale_max
            else "&& !params.use_int8_scale_max "
        )

        slen = kspec.seq_len * kspec.ctas_per_head if not kspec.flash_attention else 0

        ## NOTE: need to tune here
        if kspec.has_noloop and not kspec.flash_attention:
            call_stmt = """\
if( data_type == {data_type} && output_data_type == {output_data_type} && s == {slen} && d == {head_size} && sm == {sm}
    {il_check}) {{

    {unroll_check} {{
        {lname}(params, launch_params, stream);
    }} else {{
        {lname}_nl(params, launch_params, stream);
    }}

}} """.format(
                **asdict(kspec),
                data_type=data_type,
                output_data_type=output_data_type,
                slen=slen,
                lname=lname,
                il_check=il_check,
                unroll_check=gen_unroll_check(kspec),
            )

        elif kspec.flash_attention:  # NOTE: flash attention uses no_loop as default
            # TypeError: got multiple values for keyword argument if using key 'head_size_v', so 'dv' instead
            dv = kspec.head_size_v or kspec.head_size
            if kspec.tiled:  # higher precedence; does not require bh_upper_thres
                call_stmt = """\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    {il_check} && use_tiled) {{

    {lname}_nl_tiled(params, launch_params, stream);
}} """.format(  # type: ignore[str-format]
                    **asdict(kspec),
                    data_type=data_type,
                    output_data_type=output_data_type,
                    slen=slen,
                    lname=lname,
                    il_check=il_check,
                    dv=dv,
                )
            # warp specialization kernels need launch_params
            elif kspec.warp_specialization:
                call_stmt = """\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    {il_check}) {{

    {lname}(params, launch_params, stream);
}} """.format(  # type: ignore[str-format]
                    **asdict(kspec),
                    data_type=data_type,
                    output_data_type=output_data_type,
                    slen=slen,
                    lname=lname,
                    il_check=il_check,
                    dv=dv,
                )
            else:
                call_stmt = """\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    && !use_tiled {il_check}) {{

    {lname}_nl(params, launch_params, stream);
}} """.format(  # type: ignore[str-format]
                    **asdict(kspec),
                    data_type=data_type,
                    output_data_type=output_data_type,
                    slen=slen,
                    lname=lname,
                    il_check=il_check,
                    dv=dv,
                )
        else:
            call_stmt = """\
if( data_type == {data_type} && output_data_type == {output_data_type} && s == {slen} && d == {head_size} && sm == {sm}
    {il_check}) {{

    {lname}(params, launch_params, stream);
}} """.format(
                **asdict(kspec),
                data_type=data_type,
                output_data_type=output_data_type,
                slen=slen,
                lname=lname,
                il_check=il_check,
            )
        return call_stmt

    def gen_call_fmhca(kspec, lname):
        effective_sm, _ = get_effective_sm_and_name(kspec)
        data_type = dtype2typename[kspec.dtype]
        il_check = ""
        if kspec.version == 2:
            il_check = "&& interleaved " if kspec.interleaved else "&& !interleaved "
        if effective_sm == 90:
            il_check += "&& !use_tma " if kspec.ldgsts_q else "&& use_tma "
        il_check += (
            "&& params.use_int8_scale_max "
            if kspec.has_scale_max
            else "&& !params.use_int8_scale_max "
        )

        s_kv_len = kspec.seq_len
        if kspec.has_noloop:
            call_stmt = """\
if( data_type == {data_type} && s_kv == {s_kv_len} && d == {head_size} && sm == {sm} {il_check}) {{

    {unroll_check} {{
        {lname}(params, stream);
    }} else {{
        {lname}_nl(params, stream);
    }}

}} """.format(
                **asdict(kspec),
                data_type=data_type,
                s_kv_len=s_kv_len,
                lname=lname,
                il_check=il_check,
                unroll_check=gen_unroll_check(kspec),
            )

        else:
            call_stmt = """\
if( data_type == {data_type} && s_kv == {s_kv_len} && d == {head_size} && sm == {sm} {il_check}) {{
        {lname}(params, stream);
    }} """.format(
                **asdict(kspec),
                data_type=data_type,
                s_kv_len=s_kv_len,
                lname=lname,
                il_check=il_check,
            )
        return call_stmt

    calls_v2 = [
        gen_call(kspec, lname)
        for kspec, fname, lname, kname in specs_names
        if kspec.version == 2 and kspec.cross_mha == 0
    ]

    calls_v2 = "else ".join(calls_v2) if len(calls_v2) > 0 else "if( false ) {}"

    calls_v1 = [
        gen_call(kspec, lname)
        for kspec, fname, lname, kname in specs_names
        if kspec.version == 1 and kspec.cross_mha == 0
    ]

    calls_v1 = "else ".join(calls_v1) if len(calls_v1) > 0 else "if( false ) {}"

    calls_mhca = [
        gen_call_fmhca(kspec, lname)
        for kspec, fname, lname, kname in specs_names
        if kspec.cross_mha == 1
    ]

    calls_mhca = "else ".join(calls_mhca) if len(calls_mhca) > 0 else "if( false ) {}"

    def gen_warp_spec(kspec):
        data_type = dtype2typename[kspec.dtype]
        if kspec.sage_block_sizes is not None:
            assert kspec.output_dtype is not None
            # override the data_type to output type, otherwise it is always E4M3
            data_type = dtype2typename[kspec.output_dtype]
        slen = kspec.seq_len * kspec.ctas_per_head
        effective_sm, _ = get_effective_sm_and_name(kspec)
        warp_spec_check = ""
        nl_warps_m = kspec.warps_m if effective_sm == 90 else 1
        nl_warps_n = (
            kspec.warps_n if effective_sm == 90 else kspec.warps_m * kspec.warps_n
        )
        if kspec.version == 2 and kspec.dtype in ["fp16", "bf16"]:
            warp_spec_check += (
                "&& use_flash_attention "
                if kspec.flash_attention
                else "&& !use_flash_attention "
            )
        if kspec.version == 2:
            if effective_sm == 90:
                warp_spec_check += "&& !use_tma " if kspec.ldgsts_q else "&& use_tma "
                warp_spec_check += (
                    "&& warp_specialization "
                    if kspec.warp_specialization
                    else "&& !warp_specialization "
                )
            else:
                warp_spec_check += "&& !use_tma && !warp_specialization "

        if kspec.flash_attention:  # NOTE support any sequence
            return """\
if( data_type == {data_type} && d == {head_size} && sm == {sm} {warp_spec_check}
    && version == {version} ) {{
    warps_m = {warps_m};
    warps_n = {warps_n};
}} """.format(  # type: ignore[str-format]
                **locals(), **asdict(kspec), unroll_check=gen_unroll_check(kspec)
            )
        return """\
if( data_type == {data_type} && s == {slen} && d == {head_size} && sm == {sm} {warp_spec_check}
    && version == {version} ) {{
    {unroll_check} {{
      warps_m = {warps_m};
      warps_n = {warps_n};
    }} else {{
      warps_m = {nl_warps_m};
      warps_n = {nl_warps_n};
    }}
}} """.format(**locals(), **asdict(kspec), unroll_check=gen_unroll_check(kspec))

    warp_specs = "else ".join([gen_warp_spec(spec[0]) for spec in specs_names])
    if len(warp_specs) > 0:
        warp_specs += 'else {\n\tassert(false && "Unsupported config");\n}'

    # Generate the cta spec.
    def gen_cta_spec(spec):
        kspec, _, lname, _ = spec
        slen = kspec.seq_len * kspec.ctas_per_head
        return """\
if( data_type == {data_type} && s == {slen} && d == {head_size} && use_multi_ctas
    && version == {version} ) {{

    ctas_per_head = {ctas_per_head};
    {lname}_get_max_heads_per_wave(&max_heads_per_wave);

}} """.format(**locals(), **asdict(kspec), data_type=dtype2typename[kspec.dtype])

    cta_specs = "else ".join(
        [gen_cta_spec(spec) for spec in specs_names if spec[0].ctas_per_head > 1]
    )
    # pragma once
    api_code = """\
{copyright}


#include <cuda.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>
#include <tuple>

using Params_v1         = bert::Fused_multihead_attention_params_v1;
using Params_v2         = bert::Fused_multihead_attention_params_v2;
using Params_mhca       = bert::Fused_multihead_attention_params_mhca;
using Launch_params     = bert::Fused_multihead_attention_launch_params;

{signatures}

inline void run_fmha_v1(Params_v1 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0){{
const size_t s                 = params.s;
const size_t b                 = params.b;
const size_t d                 = params.d;
const bool force_unroll        = launch_params.force_unroll;
const bool ignore_b1opt        = launch_params.ignore_b1opt;

const bool use_flash_attention = false;

{calls_v1}
else {{
    assert(false && "Unsupported config.");
}}

}}

// Note: transitioning to moving kernel launch parameters into launch_params to reduce the
// occurrences the interface needs to be modified
inline void run_fmha_v2(Params_v2 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0) {{

const size_t s = params.s;
const size_t b = params.b;
const size_t h = params.h;
const size_t d = params.d;
const size_t dv = params.dv;
const size_t sage_block_size_q = params.sage.q.block_size;
const size_t sage_block_size_k = params.sage.k.block_size;
const size_t sage_block_size_v = params.sage.v.block_size;

const bool interleaved                       = launch_params.interleaved;
const bool force_unroll                      = launch_params.force_unroll;
const bool ignore_b1opt                      = launch_params.ignore_b1opt;
const bool force_fp32_acc                    = launch_params.force_fp32_acc;
const bool warp_specialization               = launch_params.warp_specialization;
const bool use_tma                           = launch_params.use_tma;
const bool use_flash_attention               = launch_params.flash_attention;
const bool enable_attn_logit_softcapping     = launch_params.enable_attn_logit_softcapping;
const int  attention_input_layout            = static_cast<int>(launch_params.attention_input_layout);
// tiled variant uses ldgsts
const bool  use_tiled            = launch_params.use_granular_tiling;

{calls_v2}
else {{
    assert(false && "Unsupported config.");
}}

}}

#if __guard_fmhca_placeholder__ // fmhca api header

inline void run_fmhca(Params_mhca &params,
                      const Launch_params &launch_params,
                      Data_type data_type,
                      int sm,
                      cudaStream_t stream=0) {{

const size_t s_kv   = params.s;
const size_t b      = params.b;
const size_t d      = params.d_padded;

const bool interleaved  = launch_params.interleaved;
const bool force_unroll = launch_params.force_unroll;
const bool ignore_b1opt = launch_params.ignore_b1opt;

{calls_mhca}
else {{
    assert(false && "Unsupported config");
}}

}}

#endif // fmhca api header

inline std::tuple<size_t, size_t, size_t> get_warps(Launch_params& launch_params,
                                                    int sm,
                                                    Data_type data_type,
                                                    size_t s,
                                                    size_t b,
                                                    size_t d,
                                                    int version) {{
    size_t warps_m, warps_n, warps_k = 1;
    const bool interleaved           = launch_params.interleaved;
    const bool use_tma               = launch_params.use_tma;
    const bool force_unroll          = launch_params.force_unroll;
    const bool ignore_b1opt          = launch_params.ignore_b1opt;
    const bool use_flash_attention   = launch_params.flash_attention;
    // tiled variant uses ldgsts
    const bool use_tiled             = launch_params.use_granular_tiling;
    const bool warp_specialization   = launch_params.warp_specialization;

{warp_specs}

    return std::make_tuple(warps_m, warps_n, warps_k);
}}

// The constant is defined in "setup.py".
constexpr int MAX_STGS_PER_LOOP = {MAX_STGS_PER_LOOP};

// The number of CTAs and threads per CTA to launch the kernel.
inline void get_grid_size(int &heads_per_wave,
                          int &ctas_per_head,
                          int sm,
                          Data_type data_type,
                          size_t b,
                          size_t s,
                          size_t h,
                          size_t d,
                          bool use_multi_ctas,
                          int version) {{

    // Determine the number of CTAs per head (kernel constant).
    int max_heads_per_wave = 0;
    ctas_per_head = 1;
    heads_per_wave = b*h;
{cta_specs}

    // Adjust the number of heads per wave.
    if( heads_per_wave > max_heads_per_wave ) {{
        heads_per_wave = max_heads_per_wave;
    }}
}}

""".format(**locals(), copyright=copyright, MAX_STGS_PER_LOOP=MAX_STGS_PER_LOOP)
    return api_code


def generate_jit_sources(
    uri: str, input_layout: str, input_dtype: str, output_dtype: str
) -> list:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    source_paths = []
    specs_names = []
    head_size_qk_values = [16, 32, 64, 128, 256, 512]
    head_size_qk_warpspec_values = [32, 40, 48, 64, 80, 96, 104, 128, 160, 192, 256]

    # 0 means head_size_v = head_size_qk (required for flash_valid)
    head_size_v_values = [0]
    map_input_layout = {
        "q_paged_kv": InputLayout.Q_PAGED_KV,
        "packed_qkv": InputLayout.PACKED_QKV,
        "separate_q_k_v": InputLayout.SEPARATE_Q_K_V,
        "contiguous_q_kv": InputLayout.CONTIGUOUS_Q_KV,
    }

    input_layout_values = [map_input_layout[input_layout.lower()]]
    dtype_values = [input_dtype]
    output_dtype_values = [output_dtype] if output_dtype is not None else [None]

    is_mla_values = [False]

    enable_attn_logit_softcapping_values = [True, False]
    return_softmax_values = [True, False]
    alibi_values = [True, False]
    warp_spec_configs: itertools.product = itertools.product(
        [90],
        dtype_values,
        head_size_qk_warpspec_values,
        head_size_v_values,
        enable_attn_logit_softcapping_values,
        return_softmax_values,
        alibi_values,
        is_mla_values,
        input_layout_values,
        output_dtype_values,
    )

    other_configs: itertools.product = itertools.product(
        [],
        dtype_values,
        head_size_qk_values,
        head_size_v_values,
        enable_attn_logit_softcapping_values,
        return_softmax_values,
        alibi_values,
        is_mla_values,
        input_layout_values,
        output_dtype_values,
    )

    for config_list in [warp_spec_configs, other_configs]:
        for (
            sm_iter,
            dtype_iter,
            head_size_qk_iter,
            head_size_v_iter,
            enable_attn_logit_softcapping_iter,
            return_softmax_iter,
            alibi_iter,
            is_mla_iter,
            input_layout_iter,
            output_dtype_iter,
        ) in config_list:
            kspec = generate_kernel_spec(
                sm=sm_iter,
                head_size=head_size_qk_iter,
                dtype=dtype_iter,
                return_softmax=return_softmax_iter,
                enable_attn_logit_softcapping=enable_attn_logit_softcapping_iter,
                alibi=alibi_iter,
                is_mla=is_mla_iter,
                input_layout=input_layout_iter,
                head_size_v=head_size_v_iter,
                output_dtype=output_dtype_iter,
            )
            if not is_kernel_spec_valid(kspec):
                continue

            fname, lname, kname = encode_name(kspec)
            kernel_code = get_kernel_code(kspec, kname, lname)
            if kernel_code is None:
                continue

            # Write kernel source file
            kernel_path = gen_directory / fname
            write_if_different(kernel_path, kernel_code)
            source_paths.append(kernel_path)
            specs_names.append((kspec, fname, lname, kname))

    api_code = get_api_code(specs_names)
    api_path = gen_directory / "fmha_v2_api.h"
    write_if_different(api_path, api_code)
    return source_paths
