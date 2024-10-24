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

import torch
import jinja2
import os
import pathlib
from .env import FLASHINFER_GEN_SRC_DIR
from .utils import (
    write_if_different,
    dtype_map,
    filename_safe_dtype_map,
    pos_encoding_mode_literal,
    mask_mode_literal,
)
from .single_decode_templ import single_decode_templ, customizable_single_decode_templ
from .batch_decode_templ import batch_decode_templ
from .batch_decode_mla_templ import batch_decode_mla_templ
from .single_prefill_templ import single_prefill_templ, customizable_single_prefill_templ
from .batch_prefill_templ import batch_prefill_templ
from typing import List, Tuple


def get_single_decode_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    template = jinja2.Template(single_decode_templ)
    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
        use_sliding_window="true" if use_sliding_window else "false",
        use_logits_soft_cap="true" if use_logits_soft_cap else "false",
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


def gen_single_decode_cu(*args) -> Tuple[str, pathlib.Path]:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    uri = get_single_decode_uri(*args)
    file_name = f"{uri}.cu"
    path = gen_directory / file_name
    write_if_different(
        path,
        get_single_decode_cu_str(*args),
    )
    return uri, path


def get_batch_decode_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    template = jinja2.Template(batch_decode_templ)
    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        dtype_idx=dtype_map[dtype_idx],
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
        use_sliding_window="true" if use_sliding_window else "false",
        use_logits_soft_cap="true" if use_logits_soft_cap else "false",
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


def gen_batch_decode_cu(*args) -> Tuple[str, pathlib.Path]:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    uri = get_batch_decode_uri(*args)
    file_name = f"{uri}.cu"
    path = gen_directory / file_name
    write_if_different(
        path,
        get_batch_decode_cu_str(*args),
    )
    return uri, path


def get_batch_decode_mla_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    template = jinja2.Template(batch_decode_mla_templ)
    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        dtype_idx=dtype_map[dtype_idx],
        head_dim_ckv=head_dim,
        head_dim_kpe=head_dim//8, # fixme: head_dim_ckv(kv_lora_rank) is 8 times the size of head_dim_kpe(qk_rope_head_dim) for all MLA model (DeepSeek-V2-Lite, DeepSeek-V, MiniCPM3) at the time Oct.2024
        use_sliding_window="true" if use_sliding_window else "false",
        use_logits_soft_cap="true" if use_logits_soft_cap else "false",
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


def gen_batch_decode_mla_cu(*args) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    file_name = f"{get_batch_decode_mla_uri(*args)}.cu"
    write_if_different(
        gen_directory / file_name,
        get_batch_decode_mla_cu_str(*args),
    )

def get_single_prefill_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    mask_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    template = jinja2.Template(single_prefill_templ)
    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
        mask_mode=mask_mode_literal[mask_mode],
        use_sliding_window="true" if use_sliding_window else "false",
        use_logits_soft_cap="true" if use_logits_soft_cap else "false",
        use_fp16_qk_reduction="true" if use_fp16_qk_reduction else "false",
    )


def get_single_prefill_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    mask_mode: int,
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
        f"mask_{mask_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def gen_single_prefill_cu(*args) -> Tuple[str, pathlib.Path]:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    uri = get_single_prefill_uri(*args)
    file_name = f"{uri}.cu"
    path = gen_directory / file_name
    write_if_different(
        path,
        get_single_prefill_cu_str(*args),
    )
    return uri, path


def get_batch_prefill_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    mask_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    template = jinja2.Template(batch_prefill_templ)
    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        dtype_idx=dtype_map[dtype_idx],
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
        mask_mode=mask_mode_literal[mask_mode],
        use_sliding_window="true" if use_sliding_window else "false",
        use_logits_soft_cap="true" if use_logits_soft_cap else "false",
        use_fp16_qk_reduction="true" if use_fp16_qk_reduction else "false",
    )


def get_batch_prefill_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    mask_mode: int,
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
        f"mask_{mask_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def gen_batch_prefill_cu(*args) -> Tuple[str, pathlib.Path]:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    uri = get_batch_prefill_uri(*args)
    file_name = f"{uri}.cu"
    path = gen_directory / file_name
    write_if_different(
        path,
        get_batch_prefill_cu_str(*args),
    )
    return uri, path


def get_customize_single_decode_cu_str(
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
) -> str:
    template = jinja2.Template(customizable_single_decode_templ)
    additional_params_decl = "".join(
        [f"{dtype}* {var};\n"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f"{dtype} {var};\n"
        for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params = "".join(
        [f", {dtype}* {var}"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f", {dtype} {var}"
        for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params_init = "".join(
        [f", {var}({var})" for var in additional_input_tensor_var_names] +
        [f", {var}({var})" for var in additional_input_scalar_var_names]
    )
    additional_func_params = "".join(
        [f", torch::Tensor {var}" for var in additional_input_tensor_var_names] +
        [f", {dtype} {var}" for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params_data = "".join(
        [f", static_cast<{dtype}*>({var}.data_ptr())"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f", {var}"
        for var in additional_input_scalar_var_names]
    )

    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        head_dim=head_dim,
        additional_params_decl=additional_params_decl,
        additional_params=additional_params,
        additional_params_init=additional_params_init,
        variant_decl=variant_decl,
        variant_name=variant_name,
        additional_func_params=additional_func_params,
        additional_params_data=additional_params_data
    )


def get_customize_single_prefill_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    mask_mode: int,
    additional_input_tensor_var_names: List[str],
    additional_input_tensor_var_types: List[str],
    additional_input_scalar_var_names: List[str],
    additional_input_scalar_var_types: List[str],
    variant_name: str,
    variant_decl: str,
) -> str:
    template = jinja2.Template(customizable_single_prefill_templ)
    additional_params_decl = "".join(
        [f"{dtype}* {var};\n"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f"{dtype} {var};\n"
        for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params = "".join(
        [f", {dtype}* {var}"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f", {dtype} {var}"
        for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params_init = "".join(
        [f", {var}({var})" for var in additional_input_tensor_var_names] +
        [f", {var}({var})" for var in additional_input_scalar_var_names]
    )
    additional_func_params = "".join(
        [f", torch::Tensor {var}" for var in additional_input_tensor_var_names] +
        [f", {dtype} {var}" for dtype, var in zip(
            additional_input_scalar_var_types, additional_input_scalar_var_names
        )]
    )
    additional_params_data = "".join(
        [f", static_cast<{dtype}*>({var}.data_ptr())"
        for dtype, var in zip(
            additional_input_tensor_var_types, additional_input_tensor_var_names
        )] + [f", {var}"
        for var in additional_input_scalar_var_names]
    )

    return template.render(
        dtype_q=dtype_map[dtype_q],
        dtype_kv=dtype_map[dtype_kv],
        dtype_o=dtype_map[dtype_o],
        head_dim=head_dim,
        mask_mode=mask_mode_literal[mask_mode],
        additional_params_decl=additional_params_decl,
        additional_params=additional_params,
        additional_params_init=additional_params_init,
        variant_decl=variant_decl,
        variant_name=variant_name,
        additional_func_params=additional_func_params,
        additional_params_data=additional_params_data,
    )
