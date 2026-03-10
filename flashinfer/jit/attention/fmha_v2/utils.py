import os
from collections import namedtuple
from enum import IntEnum
from dataclasses import asdict

copyright = r"""/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
"""


sm2name = {
    70: "volta",
    72: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    87: "ampere",
    89: "ada",
    90: "hopper",
    120: "blackwell",
}

dtype2traits = {
    "int8": "imma_int8_int32_traits",
    "fp16": "hmma_fp16_traits",
    "fp16_fp32": "hmma_fp32_traits",
    "bf16": "hmma_bf16_traits",
    "e4m3": "qmma_e4m3_fp32_traits",
    "e4m3_fp32": "qmma_e4m3_fp32_traits",
    "e4m3_fp16": "qmma_e4m3_fp16_traits",
}

dtype2OutputType = {
    "int8": "int8_t",
    "fp16": "fp16_t",
    "fp16_fp32": "fp16_t",
    "bf16": "bf16_t",
    "e4m3": "e4m3_t",
    "e4m3_fp32": "e4m3_t",
    "e4m3_fp16": "e4m3_t",
}

dtype2bytes = {
    "int8": 1,
    "fp16": 2,
    "fp16_fp32": 2,
    "bf16": 2,
    "e4m3": 1,
    "e4m3_fp32": 1,
    "e4m3_fp16": 1,
}

# TODO merge with above?
hopper_dtype2traits = {
    "int8": "igmma_int8_int32_traits",
    "fp16": "hgmma_fp16_traits",
    "fp16_fp32": "hgmma_fp32_traits",
    "bf16": "hgmma_bf16_traits",
    "e4m3": "qgmma_e4m3_fp32_traits",
    "e4m3_fp32": "qgmma_e4m3_fp32_traits",
}

# The minimal instruction shapes per warp group.
# TODO should this not be known to the trait itself?
hopper_traits2shape = {
    "Hopper_igmma_int8_int32_traits": (64, 8, 32),
    "Hopper_hgmma_fp16_traits": (64, 8, 16),
    "Hopper_hgmma_fp32_traits": (64, 8, 16),
    "Hopper_hgmma_bf16_traits": (64, 8, 16),
    "Hopper_qgmma_e4m3_fp32_traits": (64, 8, 32),
}

dtype2typename = {
    "int8": "DATA_TYPE_INT8",
    "fp16": "DATA_TYPE_FP16",
    "fp16_fp32": "DATA_TYPE_FP16",
    "bf16": "DATA_TYPE_BF16",
    "e4m3": "DATA_TYPE_E4M3",
    "e4m3_fp16": "DATA_TYPE_E4M3",
    "e4m3_fp32": "DATA_TYPE_E4M3",
}

pythonBoolean2cpp = {True: "true", False: "false"}


# same definition as fused_multihead_attention.h.
class AttentionMaskType(IntEnum):
    PADDING = 0
    CAUSAL = 1
    SLIDING_OR_CHUNKED_CAUSAL = 2
    CUSTOM_MASK = 3


class InputLayout(IntEnum):
    PACKED_QKV = 0
    CONTIGUOUS_Q_KV = 1
    Q_PAGED_KV = 2
    SEPARATE_Q_K_V = 3


spec_fields = (
    "sm",
    "dtype",
    "seq_len",
    "head_size",
    "warps_m",
    "warps_n",
    "version",
    "interleaved",
    "ldgsts_q",
    "ldgsts_k",
    "ldgsts_v",
    "share_smem_k_v",
    "loop_step",
    "has_noloop",
    "noloop_step",
    "unroll_threshold",
    "has_scale_max",
    "ctas_per_head",
    "sm_mma",
    "head_interleaved",
    # new added fields (only used by flash attention implementation)
    "flash_attention",
    "kv_loop_step",
    "flash_attention_bh_upper_threshold",  # to deprecate; not actively used
    "limit_qk_fragments",
    "limit_v_fragments",
    "tiled",
    # fields for warp specialized kernel
    "warp_specialization",
    "q_tile_buffers",
    "kv_tile_buffers",
    "scheduling_mode",
    # attention qkv input layout.
    "input_layout",
    # fused MHCA.
    "cross_mha",
    # other features
    "alibi",
    "enable_attn_logit_softcapping",
    "return_softmax_stats",
    "enable_skip_softmax",
    "disabled_mask_types",
    "head_size_v",
    "sage_block_sizes",
    "output_dtype",
    "is_mtp",
)

kernel_spec = namedtuple("kernel_spec", spec_fields)  # type: ignore[misc]
kernel_spec.__new__.__defaults__ = (
    1,  # ctas_per_head
    1,  # sm_mma
    True,  # head_interleaved
    False,  # flash_attention
    64,  # kv_loop_step
    -1,  # flash_attention_bh_upper_threshold
    False,  # limit_qk_fragments
    False,  # limit_v_fragments
    0,  # tiled
    False,  # warp_specialization
    1,  # q_tile_buffers
    1,  # kv_tile_buffers
    0,  # scheduling_mode
    InputLayout.PACKED_QKV,
    0,  # cross_mha
    True,  # alibi
    False,  # enable_attn_logit_softcapping
    False,  # return_softmax_stats
    False,  # enable_skip_softmax
    None,  # disabled_mask_types
    0,  # head size of V
    None,  # sage_block_sizes
    None,  # output_dtype, same as dtype by default.
    False,
)  # use MTP or not

generate_cu_trtllm = os.environ.get("GENERATE_CU_TRTLLM", "False").lower() == "true"

ns_open = (
    r"""
namespace tensorrt_llm
{
namespace kernels
{
// clang-format off
"""
    if generate_cu_trtllm
    else ""
)

ns_close = (
    r"""
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
"""
    if generate_cu_trtllm
    else ""
)

copyright = (
    """\
/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
"""
    if not generate_cu_trtllm
    else r"""/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
"""
)


MAX_STGS_PER_LOOP = 4


def encode_name(kernel_spec):
    effective_sm, sm_name = get_effective_sm_and_name(kernel_spec)
    # Is it a kernel for the interleaved NC/32HW32 INT8 layout?
    il_tag = "_il" if kernel_spec.interleaved else ""
    # Is it using the quantization scaling factor as an approximation of the max in softmax?
    scale_max_tag = "_scale_max" if kernel_spec.has_scale_max else ""
    # Deal with multi-CTA kernels for which the sequence length is seq_len per CTA * # of CTAs.
    seqlen = kernel_spec.seq_len * kernel_spec.ctas_per_head
    # The qkv layout.
    qkv_layout_tag = ""
    if kernel_spec.input_layout == InputLayout.PACKED_QKV:
        qkv_layout_tag = "_qkv"
    elif kernel_spec.input_layout == InputLayout.Q_PAGED_KV:
        qkv_layout_tag = "_q_paged_kv"
    elif kernel_spec.input_layout == InputLayout.SEPARATE_Q_K_V:
        qkv_layout_tag = "_q_k_v"
    else:
        qkv_layout_tag = "_q_kv"
    # for SM90 kernels, let's also differentiate ldgsts and tma kernels
    feature_tags = ""
    if effective_sm == 90:
        # let's think about where to insert tma/ldgsts in the string before MR. [Timmy]
        if kernel_spec.ldgsts_q:
            tma_or_ldgsts = "_ldgsts"
        else:
            tma_or_ldgsts = "_tma"
        if kernel_spec.warp_specialization:
            warp_specialization_tag = "_ws"
        else:
            warp_specialization_tag = ""
    else:
        tma_or_ldgsts = ""
        warp_specialization_tag = ""

    # Add alibi and return_softmax_stats to feature_tags for all kernels
    # to ensure unique filenames across all kernel configurations
    if kernel_spec.alibi:
        feature_tags += "_alibi"
    if kernel_spec.return_softmax_stats:
        feature_tags += "_softmax"
    if kernel_spec.enable_attn_logit_softcapping:
        feature_tags += "_softcapping"
    if kernel_spec.enable_skip_softmax:
        feature_tags += "_skipSoftmax"
    if kernel_spec.sage_block_sizes:
        feature_tags += f"_sage_{'_'.join(map(str, kernel_spec.sage_block_sizes))}"
    if kernel_spec.output_dtype:
        feature_tags += f"_output_{kernel_spec.output_dtype}"
    if kernel_spec.is_mtp:
        feature_tags += "_mtp"
    if kernel_spec.ctas_per_head > 1:
        fmt = (
            "fmha_v{version}{il_tag}_{dtype}_"
            + str(seqlen)
            + "_{head_size}{attrib}{scale_max_tag}{tma_or_ldgsts}_sm{sm}"
        )
    elif kernel_spec.flash_attention:
        fmt = "fmha_v{version}{il_tag}_flash_attention_{dtype}_{loop_step}_{kv_loop_step}_S{qkv_layout_tag}_{head_size}{head_size_v_str}{attrib}{feature_tags}{scale_max_tag}{tma_or_ldgsts}{warp_specialization_tag}_sm{sm}"
    elif kernel_spec.cross_mha:
        fmt = "fmha_mhca_{dtype}_{seq_len}_{head_size}{scale_max_tag}{tma_or_ldgsts}_sm{sm}"
    else:
        fmt = "fmha_v{version}{il_tag}_{dtype}_{seq_len}_{head_size}{attrib}{scale_max_tag}{tma_or_ldgsts}_sm{sm}"
    head_size_v_str = (
        "" if kernel_spec.head_size_v == 0 else f"x{kernel_spec.head_size_v}"
    )
    # Assemble the name of the kernel.
    name_base = fmt.format(
        **asdict(kernel_spec),
        head_size_v_str=head_size_v_str,
        il_tag=il_tag,
        qkv_layout_tag=qkv_layout_tag,
        scale_max_tag=scale_max_tag,
        tma_or_ldgsts=tma_or_ldgsts,
        warp_specialization_tag=warp_specialization_tag,
        feature_tags=feature_tags,
        attrib="__placeholder__",
    )

    # Produce file, launch function and kernel names.
    fname = name_base.replace("__placeholder__", "")
    if seqlen >= 1024 and not kernel_spec.flash_attention:
        fname += ".no_i2f_f2i"
    fname += ".cu"
    lname = ("run_" + name_base).replace("__placeholder__", "")
    kname = name_base + "_kernel"

    # remove causal
    fname = fname.replace("causal_", "")
    return fname, lname, kname


def get_GMMA_shape(instruction_traits, m, n, k, warps_n):
    gmma_k = hopper_traits2shape[instruction_traits][-1]

    # gmma shape is 64xgmma_nx16, gmma_n should be as big as possible, but not bigger than n
    # gmma_n should also be smaller than 256
    gmma_m = 64
    gmma_n = 0
    # find the largest supported n
    n_supported = [(i + 1) * 8 for i in range(32)][::-1]
    n_target = n // warps_n
    assert n_target * warps_n == n
    assert n_supported[0] == 256 and n_supported[-1] == 8
    for cand_n in n_supported:
        if n_target % cand_n == 0:
            gmma_n = cand_n
            break
    assert gmma_n > 0, "No supported GMMA_N found!"

    return gmma_m, gmma_n, gmma_k


def enable_mutex(kspec):
    # Mutex is needed for head_size > 64 to synchronize HGMMA operations between warp groups.
    # This applies to all 2-byte element types (fp16, bf16) regardless of accumulation precision.
    # enable_mutex = "false" if kspec.head_size <= 64 else "true"
    fp32_accu_dtype = kspec.dtype in ["fp16_fp32", "bf16"]
    enable_mutex = "false" if (fp32_accu_dtype or kspec.head_size <= 64) else "true"
    return enable_mutex


def enable_tma_store(kspec):
    output_dtype = kspec.output_dtype if kspec.output_dtype is not None else kspec.dtype
    # TMA copies data in the 16B granularity.
    return (
        "true"
        if (output_dtype in ["e4m3", "e4m3_fp32"] and kspec.head_size % 16 == 0)
        else "false"
    )


def get_reg_count(kspec):
    # if kspec.paged_kv_input and kspec.dtype in ['fp16', 'fp16_fp32', 'bf16']:
    #     dma_reg_count = 72
    #     compute_reg_count = 216
    if kspec.input_layout == InputLayout.Q_PAGED_KV:
        dma_reg_count = 56
        compute_reg_count = 224
    else:
        dma_reg_count = 40
        compute_reg_count = 232
    return dma_reg_count, compute_reg_count


def get_hopper_instruction_traits(instruction_traits, kernel_spec):
    gmma_shape_p = get_GMMA_shape(
        instruction_traits,
        kernel_spec.loop_step,
        kernel_spec.seq_len,
        kernel_spec.head_size,
        kernel_spec.warps_n,
    )

    instruction_traits_p = f"{instruction_traits}<{', '.join([str(x) for x in gmma_shape_p])}, false, false>"

    gmma_shape_o = get_GMMA_shape(
        instruction_traits,
        kernel_spec.loop_step,
        kernel_spec.head_size,
        kernel_spec.seq_len,
        1,
    )
    instruction_traits_o = f"{instruction_traits}<{', '.join([str(x) for x in gmma_shape_o])}, true, false>"

    return instruction_traits_p, instruction_traits_o


def get_effective_sm_and_name(kspec):
    sm = kspec.sm
    # Override the mma instruction with an older one.
    if kspec.sm_mma in sm2name:
        assert kspec.sm_mma <= kspec.sm, (
            "Instruction version should be at most target arch"
        )
        sm = kspec.sm_mma
    sm_name = sm2name[sm]
    return sm, sm_name


def selected_mask_types(kspec):
    # by default, we generate all combinations.
    # '1' means true, '0' means false.
    padding_mask = "1"
    causal_mask = "1"
    sliding_or_chunked_causal_mask = "1"
    custom_mask = "1"
    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    if "GENERATE_CUBIN" in os.environ:
        if kspec.sage_block_sizes:
            # SageAttention only needs padding mask now
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        elif (kspec.head_size, kspec.head_size_v) == (192, 128):
            # MLA context phase only needs causal mask and padding mask (for chunked prefill) now
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        elif (kspec.head_size, kspec.head_size_v) == (576, 512):
            # MLA generation phase only needs padding mask (MtpMask) now
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        # encoder models (head_size = 32 / 64 / 128) need packed_qkv input layout + padding mask.
        elif kspec.input_layout == InputLayout.PACKED_QKV:
            # NOTE: 72/80 are added for vision transformer
            if kspec.head_size not in [32, 64, 72, 80, 128]:
                padding_mask = "0"
        # only cross attention (head_size = 32/64/128) needs contiguous_q_kv input layout + padding mask / custom_mask.
        elif kspec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            if kspec.head_size not in [32, 64, 72, 128]:
                padding_mask = "0"
                custom_mask = "0"
        # paged kv cache is always needed in gpt variants.
        # cross-attention also needs paged kv cache.
        elif kspec.input_layout == InputLayout.Q_PAGED_KV:
            if kspec.head_size not in [32, 64, 128]:
                padding_mask = "0"

        # alibi specialized kernels only need causal mask.
        if kspec.alibi and kspec.warp_specialization:
            padding_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"

        # enable_attn_logit_softcapping kernels only need causal mask or sliding_or_chunked_causal_mask.
        if kspec.enable_attn_logit_softcapping:
            padding_mask = "0"
            custom_mask = "0"

    return padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask


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
            **kspec._asdict()
        )
        if kspec.flash_attention:
            code = "if (!{has_noloop} || (!force_unroll && (ignore_b1opt || b * h > {unroll_threshold})))".format(
                **kspec._asdict()
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
                "&& enable_skip_softmax "
                if kspec.enable_skip_softmax
                else "&& !enable_skip_softmax "
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
                **kspec._asdict(),
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
                    **kspec._asdict(),
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
                    **kspec._asdict(),
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
                    **kspec._asdict(),
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
                **kspec._asdict(),
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
                **kspec._asdict(),
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
                **kspec._asdict(),
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
                **locals(), **kspec._asdict(), unroll_check=gen_unroll_check(kspec)
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
}} """.format(**locals(), **kspec._asdict(), unroll_check=gen_unroll_check(kspec))

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

}} """.format(**locals(), **kspec._asdict(), data_type=dtype2typename[kspec.dtype])

    cta_specs = "else ".join(
        [gen_cta_spec(spec) for spec in specs_names if spec[0].ctas_per_head > 1]
    )

    api_code = """\
{copyright}
#pragma once

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
