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
from typing import List, Tuple

import jinja2
import torch

from .batch_decode_mla_templ import batch_decode_mla_suffix, batch_decode_mla_templ
from .batch_decode_templ import batch_decode_suffix, batch_decode_templ
from .batch_prefill_sm90_templ import (
    batch_prefill_sm90_suffix,
    batch_prefill_sm90_templ,
)
from .batch_prefill_templ import batch_prefill_suffix, batch_prefill_templ
from .core import load_cuda_ops, sm90a_nvcc_flags
from .env import FLASHINFER_GEN_SRC_DIR
from .single_decode_templ import (
    customizable_single_decode_templ,
    single_decode_suffix,
    single_decode_templ,
)
from .single_prefill_sm90_templ import (
    single_prefill_sm90_suffix,
    single_prefill_sm90_templ,
)
from .single_prefill_templ import (
    customizable_single_prefill_templ,
    single_prefill_suffix,
    single_prefill_templ,
)
from .utils import (
    dtype_map,
    filename_safe_dtype_map,
    pos_encoding_mode_literal,
    write_if_different,
)


def render_templates(template_strs: List[str], context: dict) -> List[str]:
    return [
        template.render(**context) for template in map(jinja2.Template, template_strs)
    ]


def get_single_decode_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> List[str]:
    return render_templates(
        single_decode_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
        },
    )


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


def gen_single_decode_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    os.makedirs(gen_directory, exist_ok=True)
    uri = get_single_decode_uri(*args)
    sources = get_single_decode_sources(*args)
    source_paths = []
    for suffix, source in zip(single_decode_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)
    return load_cuda_ops(uri, source_paths)


def get_batch_decode_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> List[str]:
    return render_templates(
        batch_decode_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "dtype_idx": dtype_map[dtype_idx],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
        },
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


def gen_batch_decode_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_batch_decode_uri(*args)
    sources = get_batch_decode_sources(*args)
    source_paths = []
    for suffix, source in zip(batch_decode_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)
    return load_cuda_ops(uri, source_paths)


def get_batch_decode_mla_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> List[str]:
    return render_templates(
        batch_decode_mla_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "dtype_idx": dtype_map[dtype_idx],
            "head_dim_ckv": head_dim,
            "head_dim_kpe": head_dim
            // 8,  # fixme: head_dim_ckv(kv_lora_rank) is 8 times the size of head_dim_kpe(qk_rope_head_dim) for all MLA model (DeepSeek-V2-Lite, DeepSeek-V2.5, MiniCPM3) at the time Oct.2024
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
        },
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


def gen_batch_decode_mla_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_batch_decode_mla_uri(*args)
    sources = get_batch_decode_mla_sources(*args)
    source_paths = []
    for suffix, source in zip(batch_decode_mla_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)
    return load_cuda_ops(uri, source_paths)


def get_single_prefill_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> List[str]:
    return render_templates(
        single_prefill_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
            "use_fp16_qk_reduction": "true" if use_fp16_qk_reduction else "false",
        },
    )


def get_single_prefill_sm90_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> List[str]:
    assert not use_fp16_qk_reduction, "fp16 qk reduction is not supported on sm90"
    assert (
        pos_encoding_mode == 0
    ), "Currently we only support pos_encoding_mode=0 on sm90"
    return render_templates(
        single_prefill_sm90_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
            "use_fp16_qk_reduction": "true" if use_fp16_qk_reduction else "false",
        },
    )


def get_single_prefill_uri(
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
        f"f16qk_{use_fp16_qk_reduction}"
    )


def get_single_prefill_sm90_uri(*args):
    return get_single_prefill_uri(*args) + "_sm90"


def gen_single_prefill_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_single_prefill_uri(*args)
    sources = get_single_prefill_sources(*args)
    source_paths = []
    for suffix, source in zip(single_prefill_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(uri, source_paths)


def gen_single_prefill_sm90_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_single_prefill_sm90_uri(*args)
    sources = get_single_prefill_sm90_sources(*args)
    source_paths = []
    for suffix, source in zip(single_prefill_sm90_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(uri, source_paths, extra_cuda_cflags=sm90a_nvcc_flags)


def get_batch_prefill_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> List[str]:
    return render_templates(
        batch_prefill_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "dtype_idx": dtype_map[dtype_idx],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
            "use_fp16_qk_reduction": "true" if use_fp16_qk_reduction else "false",
        },
    )


def get_batch_prefill_sm90_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> List[str]:
    assert not use_fp16_qk_reduction, "fp16 qk reduction is not supported on sm90"
    assert (
        pos_encoding_mode == 0
    ), "Currently we only support pos_encoding_mode=0 on sm90"
    return render_templates(
        batch_prefill_sm90_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "dtype_idx": dtype_map[dtype_idx],
            "head_dim": head_dim,
            "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
            "use_sliding_window": "true" if use_sliding_window else "false",
            "use_logits_soft_cap": "true" if use_logits_soft_cap else "false",
            "use_fp16_qk_reduction": "true" if use_fp16_qk_reduction else "false",
        },
    )


def get_batch_prefill_uri(
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
        f"f16qk_{use_fp16_qk_reduction}"
    )


def get_batch_prefill_sm90_uri(*args):
    return get_batch_prefill_uri(*args) + "_sm90"


def gen_batch_prefill_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_batch_prefill_uri(*args)
    sources = get_batch_prefill_sources(*args)
    source_paths = []
    for suffix, source in zip(batch_prefill_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(uri, source_paths)


def gen_batch_prefill_sm90_module(*args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    uri = get_batch_prefill_sm90_uri(*args)
    sources = get_batch_prefill_sm90_sources(*args)
    source_paths = []
    for suffix, source in zip(batch_prefill_sm90_suffix, sources):
        path = gen_directory / f"{uri}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(uri, source_paths, extra_cuda_cflags=sm90a_nvcc_flags)


def get_customize_single_decode_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    additional_input_tensor_var_names: List[str],
    additional_input_tensor_var_types: List[str],
    additional_input_scalar_var_names: List[str],
    additional_input_scalar_var_types: List[str],
    variant_name: str,
    variant_decl: str,
) -> List[str]:
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params = "".join(
        [
            f", {dtype}* {var}"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params_init = "".join(
        [f", {var}({var})" for var in additional_input_tensor_var_names]
        + [f", {var}({var})" for var in additional_input_scalar_var_names]
    )
    additional_func_params = "".join(
        [f", at::Tensor {var}" for var in additional_input_tensor_var_names]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params_data = "".join(
        [
            f", static_cast<{dtype}*>({var}.data_ptr())"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [f", {var}" for var in additional_input_scalar_var_names]
    )

    return render_templates(
        customizable_single_decode_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "head_dim": head_dim,
            "additional_params_decl": additional_params_decl,
            "additional_params": additional_params,
            "additional_params_init": additional_params_init,
            "variant_decl": variant_decl,
            "variant_name": variant_name,
            "additional_func_params": additional_func_params,
            "additional_params_data": additional_params_data,
        },
    )


def get_customize_single_prefill_sources(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    additional_input_tensor_var_names: List[str],
    additional_input_tensor_var_types: List[str],
    additional_input_scalar_var_names: List[str],
    additional_input_scalar_var_types: List[str],
    variant_name: str,
    variant_decl: str,
) -> List[str]:
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params = "".join(
        [
            f", {dtype}* {var}"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params_init = "".join(
        [f", {var}({var})" for var in additional_input_tensor_var_names]
        + [f", {var}({var})" for var in additional_input_scalar_var_names]
    )
    additional_func_params = "".join(
        [f", at::Tensor {var}" for var in additional_input_tensor_var_names]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_input_scalar_var_types, additional_input_scalar_var_names
            )
        ]
    )
    additional_params_data = "".join(
        [
            f", static_cast<{dtype}*>({var}.data_ptr())"
            for dtype, var in zip(
                additional_input_tensor_var_types, additional_input_tensor_var_names
            )
        ]
        + [f", {var}" for var in additional_input_scalar_var_names]
    )

    return render_templates(
        customizable_single_prefill_templ,
        {
            "dtype_q": dtype_map[dtype_q],
            "dtype_kv": dtype_map[dtype_kv],
            "dtype_o": dtype_map[dtype_o],
            "head_dim": head_dim,
            "additional_params_decl": additional_params_decl,
            "additional_params": additional_params,
            "additional_params_init": additional_params_init,
            "variant_decl": variant_decl,
            "variant_name": variant_name,
            "additional_func_params": additional_func_params,
            "additional_params_data": additional_params_data,
        },
    )


def gen_customize_single_decode_module(module_name, *args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    sources = get_customize_single_decode_sources(*args)
    source_paths = []
    for suffix, source in zip(single_decode_suffix, sources):
        path = gen_directory / f"{module_name}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(module_name, source_paths)


def gen_customize_single_prefill_module(module_name, *args):
    gen_directory = FLASHINFER_GEN_SRC_DIR
    sources = get_customize_single_prefill_sources(*args)
    source_paths = []
    for suffix, source in zip(single_prefill_suffix, sources):
        path = gen_directory / f"{module_name}{suffix}"
        source_paths.append(path)
        write_if_different(path, source)

    return load_cuda_ops(module_name, source_paths)
