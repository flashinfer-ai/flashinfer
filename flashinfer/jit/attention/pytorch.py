"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from typing import List

import jinja2
import torch

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, logger, sm90a_nvcc_flags, sm100a_nvcc_flags
from ..utils import (
    dtype_map,
    filename_safe_dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)
from .utils import generate_additional_params


def get_single_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"single_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_batch_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"batch_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_batch_mla_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
    use_profiler: bool,
) -> str:
    return (
        f"batch_mla_attention_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_ckv_{head_dim_ckv}_"
        f"head_dim_kpe_{head_dim_kpe}_"
        f"profiler_{use_profiler}"
    ) + ("_sm90" if backend == "fa3" else "")


def gen_batch_mla_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
    use_profiler: bool,
) -> JitSpec:
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    uri = get_batch_mla_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_ckv,
        head_dim_kpe,
        use_profiler,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    if backend == "fa2":
        with open(jit_env.FLASHINFER_CSRC_DIR / "batch_mla_config.jinja") as f:
            config_templ = jinja2.Template(f.read())
        generated_config_path = gen_directory / "batch_mla_config.inc"
        write_if_different(
            generated_config_path,
            config_templ.render(
                dtype_q=dtype_map[dtype_q],
                dtype_kv=dtype_map[dtype_kv],
                dtype_o=dtype_map[dtype_o],
                dtype_idx=dtype_map[dtype_idx],
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
            ),
        )

        source_paths = []
        for filename in [
            "batch_mla_plan.cu",
            "batch_mla_run.cu",
            "batch_mla_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)
    elif backend == "fa3":
        with open(jit_env.FLASHINFER_CSRC_DIR / "batch_mla_config.jinja") as f:
            config_templ = jinja2.Template(f.read())
        generated_config_path = gen_directory / "batch_mla_sm90_config.inc"
        write_if_different(
            generated_config_path,
            config_templ.render(
                dtype_q=dtype_map[dtype_q],
                dtype_kv=dtype_map[dtype_kv],
                dtype_o=dtype_map[dtype_o],
                dtype_idx=dtype_map[dtype_idx],
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
            ),
        )
        source_paths = []
        for filename in [
            "batch_mla_sm90_plan.cu",
            "batch_mla_sm90_run.cu",
            "batch_mla_sm90_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    extra_cuda_cflags = []
    if backend == "fa3":
        extra_cuda_cflags += sm90a_nvcc_flags
    if use_profiler:
        extra_cuda_cflags += ["-DFLASHINFER_ENABLE_PROFILER"]

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=extra_cuda_cflags,
    )


def get_batch_decode_mla_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_ckv: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    arc: str,
) -> str:
    return (
        f"batch_decode_mla_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_ckv{head_dim_ckv}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"arc_{arc}"
    )


def gen_batch_decode_mla_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    num_qo_heads: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_tensor_cores: bool,
) -> JitSpec:
    cuda_arch_major = torch.cuda.get_device_properties(0).major

    if cuda_arch_major >= 9:  # smem size of SM90 can accommodate all 128 qo-heads data
        qo_tile_len = 128
    else:
        qo_tile_len = 64

    if (
        use_tensor_cores
        and cuda_arch_major >= 8
        and num_qo_heads % qo_tile_len == 0
        and dtype_q == torch.float16
        and dtype_kv == torch.float16
        and dtype_o == torch.float16
    ):
        logger.info("Use tensor-core SM80 version of MLA decode kernel.")
        arc = "sm80"
    else:
        logger.info("Fall back to cuda-core version of MLA decode kernel.")
        arc = "cuda_core"

    uri = get_batch_decode_mla_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        use_sliding_window,
        use_logits_soft_cap,
        arc,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_decode_mla_config.jinja") as f:
        config_templ = jinja2.Template(f.read())
    generated_config_path = gen_directory / "mla_config.inc"
    write_if_different(
        generated_config_path,
        config_templ.render(
            dtype_q=dtype_map[dtype_q],
            dtype_kv=dtype_map[dtype_kv],
            dtype_o=dtype_map[dtype_o],
            dtype_idx=dtype_map[dtype_idx],
            head_dim_ckv=head_dim,
            head_dim_kpe=head_dim // 8,
            qo_tile_len=qo_tile_len,
            use_sliding_window=str(use_sliding_window).lower(),
            use_logits_soft_cap=str(use_logits_soft_cap).lower(),
        ),
    )

    filenames = []
    if arc == "sm80":
        filenames = [
            "batch_decode_mla_cute_sm80.cu",
            "batch_decode_mla_pybind.cu",
        ]
    else:
        filenames = [
            "batch_decode_mla_plan.cu",
            "batch_decode_mla_run.cu",
            "batch_decode_mla_pybind.cu",
        ]

    source_paths = []
    for filename in filenames:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return gen_jit_spec(uri, source_paths)


def get_single_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    return (
        f"single_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )


def get_pod_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> str:
    return (
        f"pod_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_"
        f"posenc_p_{pos_encoding_mode_p}_"
        f"use_swa_p_{use_sliding_window_p}_"
        f"use_logits_cap_p_{use_logits_soft_cap_p}_"
        f"posenc_d_{pos_encoding_mode_d}_"
        f"use_swa_d_{use_sliding_window_d}_"
        f"use_logits_cap_d_{use_logits_soft_cap_d}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def get_batch_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    return (
        f"batch_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )


def get_batch_attention_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_profiler: bool,
) -> str:
    return (
        f"batch_attention_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_profiler_{str(use_profiler).lower()}"
    )


def gen_single_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> JitSpec:
    uri = get_single_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    return gen_customize_single_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["double", "double", "double", "double"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        "#include<flashinfer/attention/variants.cuh>",  # variant_decl
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_single_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    uri = get_single_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )

    # use `fp8_enabled` flag to use separate kernel template
    # this is used for fp8 tensor core computation
    # KV-only quant is not influenced by this flag
    fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]

    if backend == "fa2":
        assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
        additional_tensor_names = ["maybe_custom_mask", "maybe_alibi_slopes"]
        additional_tensor_dtypes = ["uint8_t", "float"]
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ]
        additional_scalar_dtypes = ["double", "double", "double", "double"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    else:
        if not fp8_enabled:
            additional_tensor_names = []
            additional_tensor_dtypes = []
            additional_scalar_names = ["logits_soft_cap", "sm_scale"]
            additional_scalar_dtypes = ["double", "double"]
            variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"
        else:
            additional_tensor_names = ["scale_q", "scale_k", "scale_v"]
            additional_tensor_dtypes = ["float", "float", "float"]
            additional_scalar_names = ["sm_scale"]
            additional_scalar_dtypes = ["double"]
            variant_name = "DefaultFP8Attention"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_single_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        fp8_enabled=fp8_enabled,
    )


def gen_pod_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> JitSpec:
    uri = get_pod_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        use_fp16_qk_reduction,
        dtype_idx,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
    )
    additional_tensor_names = ["maybe_custom_mask", "maybe_alibi_slopes"]
    additional_tensor_dtypes = ["uint8_t", "float"]
    additional_scalar_names = [
        "logits_soft_cap",
        "sm_scale",
        "rope_rcp_scale",
        "rope_rcp_theta",
    ]
    additional_scalar_dtypes = ["float", "float", "float", "float"]
    variant_name_p = f"DefaultAttention<use_custom_mask_p, {str(use_sliding_window_p).lower()}, {str(use_logits_soft_cap_p).lower()}, {str(pos_encoding_mode_p == 2).lower()}>"
    variant_name_d = f"DefaultAttention<use_custom_mask_d, {str(use_sliding_window_d).lower()}, {str(use_logits_soft_cap_d).lower()}, {str(pos_encoding_mode_d == 2).lower()}>"
    variant_decl = "#include<flashinfer/attention/variants.cuh>"

    return gen_customize_pod_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name_p,
        variant_name_d,
        variant_decl,
        pos_encoding_mode_p=pos_encoding_mode_p,
        use_sliding_window_p=use_sliding_window_p,
        use_logits_soft_cap_p=use_logits_soft_cap_p,
        pos_encoding_mode_d=pos_encoding_mode_d,
        use_sliding_window_d=use_sliding_window_d,
        use_logits_soft_cap_d=use_logits_soft_cap_d,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def gen_customize_pod_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name_p: str,
    variant_name_d: str,
    variant_decl: str,
    pos_encoding_mode_p: int = 0,
    use_sliding_window_p: bool = False,
    use_logits_soft_cap_p: bool = False,
    pos_encoding_mode_d: int = 0,
    use_sliding_window_d: bool = False,
    use_logits_soft_cap_d: bool = False,
    use_fp16_qk_reduction: bool = False,
) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    (
        additional_params_decl,
        additional_func_params,
        additional_params_setter,
    ) = generate_additional_params(
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
    )

    with open(jit_env.FLASHINFER_CSRC_DIR / "pod_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "pod_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    kwargs = {
        "additional_func_params": additional_func_params,
        "additional_params_decl": additional_params_decl,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name_p": variant_name_p,
        "variant_name_d": variant_name_d,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[dtype_idx],
        "head_dim_qk": head_dim,
        "head_dim_vo": head_dim,
        "pos_encoding_mode_p": pos_encoding_mode_literal[pos_encoding_mode_p],
        "pos_encoding_mode_d": pos_encoding_mode_literal[pos_encoding_mode_d],
        "use_sliding_window_p": str(use_sliding_window_p).lower(),
        "use_logits_soft_cap_p": str(use_logits_soft_cap_p).lower(),
        "use_sliding_window_d": str(use_sliding_window_d).lower(),
        "use_logits_soft_cap_d": str(use_logits_soft_cap_d).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []

    for mask_mode_p in [0, 1, 2, 3]:
        for mask_mode_d in [0, 1, 2, 3]:
            kwargs["mask_mode_p"] = mask_mode_literal[mask_mode_p]
            kwargs["mask_mode_d"] = mask_mode_literal[mask_mode_d]

            filename = f"pod_kernel_mask_{mask_mode_p}p_{mask_mode_d}d.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                **kwargs,
            )
            write_if_different(dest_path, source)

    for filename in [
        "pod.cu",
        "pod_jit_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "pod_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return gen_jit_spec(uri, source_paths)


def gen_batch_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> JitSpec:
    uri = get_batch_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    return gen_customize_batch_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["double", "double", "double", "double"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        "#include<flashinfer/attention/variants.cuh>",  # variant_decl
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_batch_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    uri = get_batch_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )

    # use `fp8_enabled` flag to use separate kernel template
    # this is used for fp8 tensor core computation
    # KV-only quant is not influenced by this flag
    fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]

    if backend == "fa2":
        assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
        additional_tensor_names = [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
        ]
        additional_tensor_dtypes = [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
        ]  # NOTE(Zihao): int32_t should follow dtype_idx
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ]
        additional_scalar_dtypes = ["double", "double", "double", "double", "int64_t"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    else:
        if not fp8_enabled:
            additional_tensor_names = [
                "maybe_prefix_len_ptr",
                "maybe_token_pos_in_items_ptr",
                "maybe_max_item_len_ptr",
            ]
            additional_tensor_dtypes = ["uint32_t", "uint16_t", "uint16_t"]
            additional_scalar_names = [
                "logits_soft_cap",
                "sm_scale",
                "token_pos_in_items_len",
            ]
            additional_scalar_dtypes = ["double", "double", "int64_t"]
            variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"
        else:
            additional_tensor_names = ["scale_q", "scale_k", "scale_v"]
            additional_tensor_dtypes = ["float", "float", "float"]
            additional_scalar_names = ["sm_scale"]
            additional_scalar_dtypes = ["double"]
            variant_name = "DefaultFP8Attention"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_batch_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        fp8_enabled=fp8_enabled,
    )


def gen_batch_attention_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_profiler: bool,
):
    uri = get_batch_attention_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_profiler,
    )

    additional_tensor_names = []
    additional_tensor_dtypes = []
    additional_scalar_names = []
    additional_scalar_dtypes = []
    variant_name = f"StandardAttention"
    variant_decl = f"#include<flashinfer/attention/variants.cuh>"

    return gen_customize_batch_attention_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_profiler=use_profiler,
    )


def gen_customize_single_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    (
        additional_params_decl,
        additional_func_params,
        additional_params_setter,
    ) = generate_additional_params(
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
    )

    with open(
        jit_env.FLASHINFER_CSRC_DIR / "single_decode_customize_config.jinja"
    ) as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "single_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    kwargs = {
        "additional_func_params": additional_func_params,
        "additional_params_decl": additional_params_decl,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []

    dest_path = gen_directory / "single_decode_kernel.cu"
    source_paths.append(dest_path)
    source = kernel_inst_templ.render(
        **kwargs,
    )
    write_if_different(dest_path, source)

    for filename in [
        "single_decode.cu",
        "single_decode_jit_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "single_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return gen_jit_spec(uri, source_paths)


def gen_customize_single_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
) -> JitSpec:
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
        additional_params_decl, additional_func_params, additional_params_setter = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "single_prefill_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "single_prefill_kernel_inst.jinja"
        ) as f:
            kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_func_params": additional_func_params,
            "additional_params_decl": additional_params_decl,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            filename = f"single_prefill_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "single_prefill.cu",
            "single_prefill_jit_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)

        return gen_jit_spec(uri, source_paths)
    elif backend == "fa3":
        gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
                is_sm90_template=True,
            )
        )

        _file_config = "single_prefill_sm90_customize_config.jinja"
        if fp8_enabled:
            _file_kernel_inst = "single_prefill_fp8_sm90_kernel_inst.jinja"
            _file_csrc = "single_prefill_fp8_sm90.cu"
        else:
            _file_kernel_inst = "single_prefill_sm90_kernel_inst.jinja"
            _file_csrc = "single_prefill_sm90.cu"

        with open(jit_env.FLASHINFER_CSRC_DIR / _file_config) as f:
            config_templ = jinja2.Template(f.read())

        with open(jit_env.FLASHINFER_CSRC_DIR / _file_kernel_inst) as f:
            kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_func_params": additional_func_params,
            "additional_params_decl": additional_params_decl,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            filename = f"single_prefill_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            _file_csrc,
            "single_prefill_sm90_jit_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(
            uri,
            source_paths,
            extra_cuda_cflags=sm90a_nvcc_flags,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


def gen_customize_batch_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(
            additional_tensor_names,
            additional_tensor_dtypes,
            additional_scalar_names,
            additional_scalar_dtypes,
        )
    )

    kwargs = {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    source_paths = []

    dest_path = gen_directory / "batch_decode_kernel.cu"
    source_paths.append(dest_path)
    source = kernel_inst_templ.render(
        **kwargs,
    )
    write_if_different(dest_path, source)

    for filename in [
        "batch_decode.cu",
        "batch_decode_jit_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return gen_jit_spec(uri, source_paths)


def gen_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
) -> JitSpec:
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_paged_kernel_inst.jinja"
        ) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_ragged_kernel_inst.jinja"
        ) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            dest_path = (
                gen_directory / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            dest_path = (
                gen_directory / f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "batch_prefill.cu",
            "batch_prefill_jit_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(uri, source_paths)
    elif backend == "fa3":
        gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
                is_sm90_template=True,
            )
        )

        _file_config = "batch_prefill_sm90_customize_config.jinja"
        if fp8_enabled:
            _file_paged_kernel_inst = "batch_prefill_fp8_paged_sm90_kernel_inst.jinja"
            _file_ragged_kernel_inst = "batch_prefill_fp8_ragged_sm90_kernel_inst.jinja"
            _file_csrc = "batch_prefill_fp8_sm90.cu"
        else:
            _file_paged_kernel_inst = "batch_prefill_paged_sm90_kernel_inst.jinja"
            _file_ragged_kernel_inst = "batch_prefill_ragged_sm90_kernel_inst.jinja"
            _file_csrc = "batch_prefill_sm90.cu"

        with open(jit_env.FLASHINFER_CSRC_DIR / _file_config) as f:
            config_templ = jinja2.Template(f.read())

        with open(jit_env.FLASHINFER_CSRC_DIR / _file_paged_kernel_inst) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(jit_env.FLASHINFER_CSRC_DIR / _file_ragged_kernel_inst) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }
        generated_inc_str = config_templ.render(**kwargs)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            filename = f"batch_prefill_paged_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            filename = f"batch_prefill_ragged_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            _file_csrc,
            "batch_prefill_sm90_jit_pybind.cu",
        ]:
            src_path = jit_env.FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(
            uri,
            source_paths,
            extra_cuda_cflags=sm90a_nvcc_flags,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


def get_fmha_cutlass_sm100a_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    # NOTE(Zihao): use different uri after when support customize attention
    return "fmha_cutlass_sm100a"
    # return (
    #     f"fmha_cutlass_sm100a_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
    #     f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
    #     f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
    #     f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
    #     f"head_dim_qk_{head_dim_qk}_"
    #     f"head_dim_vo_{head_dim_vo}_"
    #     f"posenc_{pos_encoding_mode}_"
    #     f"use_swa_{use_sliding_window}_"
    #     f"use_logits_cap_{use_logits_soft_cap}"
    # )


def gen_fmha_cutlass_sm100a_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> JitSpec:
    uri = get_fmha_cutlass_sm100a_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )

    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fmha_cutlass_sm100.cu",
        jit_env.FLASHINFER_CSRC_DIR / "fmha_cutlass_sm100_pybind.cu",
        jit_env.FLASHINFER_CSRC_DIR / "blackwell_fmha_plan.cu",
    ]
    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


def trtllm_fmha_gen_module():
    return gen_jit_spec(
        "fmha_gen",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fmha_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fmha_kernel_launcher.cu",
        ],
        extra_ldflags=["-lcuda"],
    )


def gen_customize_batch_attention_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_profiler: bool = False,
):
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
    }
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(
            additional_tensor_names,
            additional_tensor_dtypes,
            additional_scalar_names,
            additional_scalar_dtypes,
        )
    )

    with open(
        jit_env.FLASHINFER_CSRC_DIR / "batch_attention_customize_config.jinja"
    ) as f:
        config_templ = jinja2.Template(f.read())

    with open(
        jit_env.FLASHINFER_CSRC_DIR / "batch_attention_paged_kernel_inst.jinja"
    ) as f:
        paged_kernel_inst_templ = jinja2.Template(f.read())

    kwargs |= {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
    }

    generated_inc_str = config_templ.render(
        **kwargs,
    )
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []
    for mask_mode in [0, 1, 2, 3]:
        dest_path = gen_directory / f"batch_attention_paged_kernel_mask_{mask_mode}.cu"
        source_paths.append(dest_path)
        source = paged_kernel_inst_templ.render(
            mask_mode=mask_mode_literal[mask_mode],
            **kwargs,
        )
        write_if_different(dest_path, source)

    for filename in [
        "batch_attention.cu",
        "batch_attention_jit_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_attention_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=["-DFLASHINFER_ENABLE_PROFILER"] if use_profiler else [],
    )
