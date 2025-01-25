"""
Copyright (c) 2024 by FlashInfer team.

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
import pathlib
from collections import namedtuple
from typing import List, Tuple

import jinja2
import torch

from .core import load_cuda_ops, sm90a_nvcc_flags
from .env import FLASHINFER_CSRC_DIR, FLASHINFER_GEN_SRC_DIR
from .utils import (
    dtype_map,
    filename_safe_dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)


def generate_additional_params(
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    is_sm90_template: bool = False,
):
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_tensor_dtypes,
                additional_tensor_names,
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(additional_scalar_dtypes, additional_scalar_names)
        ]
    )
    additional_func_params = "".join(
        [
            (
                f", std::optional<at::Tensor> {var}"
                if var.startswith("maybe")
                else f", at::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(additional_scalar_dtypes, additional_scalar_names)
        ]
    )
    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
            ]
            + [
                f"params.additional_params.{var} = {var};"
                for var in additional_scalar_names
            ]
        )
    else:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)


def get_single_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"single_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_batch_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"batch_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_batch_decode_mla_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"batch_decode_mla_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_{head_dim}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def gen_batch_decode_mla_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    uri = get_batch_decode_mla_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(FLASHINFER_CSRC_DIR / "batch_decode_mla_config.jinja") as f:
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
            use_sliding_window=str(use_sliding_window).lower(),
            use_logits_soft_cap=str(use_logits_soft_cap).lower(),
        ),
    )

    source_paths = []
    for filename in [
        "batch_decode_mla_plan.cu",
        "batch_decode_mla_run.cu",
        "batch_decode_mla_pybind.cu",
    ]:
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return load_cuda_ops(uri, source_paths)


def get_single_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    return (
        f"single_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )


def get_batch_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
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
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )


def gen_single_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    uri = get_single_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    return gen_customize_single_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["float", "float", "float", "float"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        f"#include<flashinfer/attention/variants.cuh>",  # variant_decl
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_single_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
):
    uri = get_single_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )
    if backend == "fa2":
        additional_tensor_names = ["maybe_custom_mask", "maybe_alibi_slopes"]
        additional_tensor_dtypes = ["uint8_t", "float"]
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ]
        additional_scalar_dtypes = ["float", "float", "float", "float"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = f"#include<flashinfer/attention/variants.cuh>"
    else:
        additional_tensor_names = []
        additional_tensor_dtypes = []
        additional_scalar_names = ["logits_soft_cap", "sm_scale"]
        additional_scalar_dtypes = ["float", "float"]
        variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
        variant_decl = f"#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_single_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
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
    )


def gen_batch_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    uri = get_batch_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
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
        head_dim,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["float", "float", "float", "float"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        f"#include<flashinfer/attention/variants.cuh>",  # variant_decl
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
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
):
    uri = get_batch_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )

    if backend == "fa2":
        additional_tensor_names = [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
        ]
        additional_tensor_dtypes = [
            "uint8_t",
            "int32_t",
            "float",
        ]  # NOTE(Zihao): int32_t should follow dtype_idx
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ]
        additional_scalar_dtypes = ["float", "float", "float", "float"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = f"#include<flashinfer/attention/variants.cuh>"
    else:
        additional_tensor_names = []
        additional_tensor_dtypes = []
        additional_scalar_names = ["logits_soft_cap", "sm_scale"]
        additional_scalar_dtypes = ["float", "float"]
        variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
        variant_decl = f"#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_batch_prefill_module(
        backend,
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
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def gen_customize_single_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
):
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri

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

    with open(FLASHINFER_CSRC_DIR / "single_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(FLASHINFER_CSRC_DIR / "single_decode_kernel_inst.jinja") as f:
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
        "head_dim": head_dim,
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
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "single_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return load_cuda_ops(uri, source_paths)


def gen_customize_single_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
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
):
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "head_dim": head_dim,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        additional_params_decl, additional_func_params, additional_params_setter = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(FLASHINFER_CSRC_DIR / "single_prefill_customize_config.jinja") as f:
            config_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "single_prefill_kernel_inst.jinja") as f:
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
        for mask_mode in [0, 1, 2]:
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
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)

        return load_cuda_ops(uri, source_paths)
    elif backend == "fa3":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri

        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
                is_sm90_template=True,
            )
        )

        with open(
            FLASHINFER_CSRC_DIR / "single_prefill_sm90_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "single_prefill_sm90_kernel_inst.jinja") as f:
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
        for mask_mode in [0, 1, 2]:
            filename = f"single_prefill_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "single_prefill_sm90.cu",
            "single_prefill_sm90_jit_pybind.cu",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return load_cuda_ops(
            uri,
            source_paths,
            extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"],
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


def gen_customize_batch_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
):
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri
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
        "head_dim": head_dim,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    with open(FLASHINFER_CSRC_DIR / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(FLASHINFER_CSRC_DIR / "batch_decode_kernel_inst.jinja") as f:
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
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return load_cuda_ops(
        uri,
        source_paths,
    )


def gen_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim: int,
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
):
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim": head_dim,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_customize_config.jinja") as f:
            config_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_paged_kernel_inst.jinja") as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_ragged_kernel_inst.jinja") as f:
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
        for mask_mode in [0, 1, 2]:
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
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return load_cuda_ops(
            uri,
            source_paths,
        )
    elif backend == "fa3":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
                is_sm90_template=True,
            )
        )

        with open(
            FLASHINFER_CSRC_DIR / "batch_prefill_sm90_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(
            FLASHINFER_CSRC_DIR / "batch_prefill_paged_sm90_kernel_inst.jinja"
        ) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(
            FLASHINFER_CSRC_DIR / "batch_prefill_ragged_sm90_kernel_inst.jinja"
        ) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }
        generated_inc_str = config_templ.render(**kwargs)

        source_paths = []
        for mask_mode in [0, 1, 2]:
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
            "batch_prefill_sm90.cu",
            "batch_prefill_sm90_jit_pybind.cu",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return load_cuda_ops(
            uri,
            source_paths,
            extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"],
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
