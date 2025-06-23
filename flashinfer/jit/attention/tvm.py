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

import itertools
import os
from typing import List

import jinja2
import torch

from .. import env as jit_env
from ..utils import (
    dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)
from .utils import generate_additional_params


def gen_sampling_tvm_binding(uri: str):
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []
    for filename in ["sampling.cu", "sampling_jit_tvm_binding.cu"]:
        src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return uri, source_paths


def gen_customize_batch_prefill_tvm_binding(
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
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    enable_inline_rope: bool = True,
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
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "fa3":
        # NOTE: fa3 backend is not supported for now, which will be resolved in the near future.
        raise ValueError("TVM binding does not support fa3 backend for now.")

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
            jit_env.FLASHINFER_TVM_BINDING_DIR / "batch_prefill_customize_config.jinja"
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

        generated_inc_str = config_templ.render(**kwargs)
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        pos_encoding_modes = [0]
        if enable_inline_rope:
            pos_encoding_modes.append(1)
        for mask_mode, pos_encoding_mode in itertools.product(
            [0, 1], pos_encoding_modes
        ):
            dest_path = (
                gen_directory / f"batch_prefill_paged_kernel_mask_{mask_mode}_"
                f"pos_encoding_{pos_encoding_mode}.cu"
            )
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            dest_path = (
                gen_directory / f"batch_prefill_ragged_kernel_mask_{mask_mode}_"
                f"pos_encoding_{pos_encoding_mode}.cu"
            )
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "batch_prefill.cu",
            "batch_prefill_jit_tvm_binding.cu",
        ]:
            src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return uri, source_paths
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

        with open(
            jit_env.FLASHINFER_TVM_BINDING_DIR
            / "batch_prefill_sm90_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_paged_sm90_kernel_inst.jinja"
        ) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(
            jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_ragged_sm90_kernel_inst.jinja"
        ) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }
        generated_inc_str = config_templ.render(**kwargs)

        source_paths = []
        for mask_mode, pos_encoding_mode in itertools.product([0, 1], [0, 1]):
            filename = (
                f"batch_prefill_paged_sm90_kernel_mask_{mask_mode}_"
                f"pos_encoding_{pos_encoding_mode}.cu"
            )
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            filename = (
                f"batch_prefill_ragged_sm90_kernel_mask_{mask_mode}_"
                f"pos_encoding_{pos_encoding_mode}.cu"
            )
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "batch_prefill_sm90.cu",
            "batch_prefill_sm90_jit_tvm_binding.cu",
        ]:
            src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return uri, source_paths
    else:
        raise ValueError(f"Invalid backend: {backend}")


def gen_customize_batch_decode_tvm_binding(
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
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
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
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
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
        jit_env.FLASHINFER_TVM_BINDING_DIR / "batch_decode_customize_config.jinja"
    ) as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    kwargs |= {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
    }
    generated_inc_str = config_templ.render(**kwargs)
    source_paths = []
    for pos_encoding_mode in [0, 1]:
        dest_path = (
            gen_directory / f"batch_decode_kernel_pos_encoding_{pos_encoding_mode}.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            pos_encoding_mode=pos_encoding_mode_literal[pos_encoding_mode],
            **kwargs,
        )
        write_if_different(dest_path, source)

    for filename in [
        "batch_decode.cu",
        "batch_decode_jit_tvm_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return uri, source_paths


def gen_batch_mla_tvm_binding(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
):
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(jit_env.FLASHINFER_TVM_BINDING_DIR / "batch_mla_config.jinja") as f:
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
        "batch_mla_jit_tvm_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return uri, source_paths
