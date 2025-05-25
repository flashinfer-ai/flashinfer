This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
jit/
  attention/
    __init__.py
    pytorch.py
    tvm.py
    utils.py
  __init__.py
  activation.py
  core.py
  cpp_ext.py
  env.py
  utils.py
profiler/
  __init__.py
triton/
  kernels/
    activation.py
    cascade.py
    norm.py
    quant.py
    sm_constraint_gemm.py
  __init__.py
  activation.py
  cascade.py
  gemm.py
  norm.py
  page.py
  sm_constraint_gemm.py
  utils.py
__init__.py
activation.py
aot.py
cascade.py
comm.py
decode.py
gemm.py
mla.py
norm.py
page.py
pod.py
prefill.py
quantization.py
rope.py
sampling.py
sparse.py
utils.py
```

# Files

## File: jit/attention/__init__.py
````python
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

from . import pytorch, tvm
from .pytorch import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .pytorch import gen_batch_decode_module as gen_batch_decode_module
from .pytorch import gen_batch_mla_module as gen_batch_mla_module
from .pytorch import gen_batch_prefill_module as gen_batch_prefill_module
from .pytorch import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .pytorch import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .pytorch import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .pytorch import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .pytorch import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .pytorch import gen_pod_module as gen_pod_module
from .pytorch import gen_single_decode_module as gen_single_decode_module
from .pytorch import gen_single_prefill_module as gen_single_prefill_module
from .pytorch import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .pytorch import get_batch_decode_uri as get_batch_decode_uri
from .pytorch import get_batch_mla_uri as get_batch_mla_uri
from .pytorch import get_batch_prefill_uri as get_batch_prefill_uri
from .pytorch import get_pod_uri as get_pod_uri
from .pytorch import get_single_decode_uri as get_single_decode_uri
from .pytorch import get_single_prefill_uri as get_single_prefill_uri
from .tvm import gen_batch_mla_tvm_binding as gen_batch_mla_tvm_binding
from .tvm import (
    gen_customize_batch_decode_tvm_binding as gen_customize_batch_decode_tvm_binding,
)
from .tvm import (
    gen_customize_batch_prefill_tvm_binding as gen_customize_batch_prefill_tvm_binding,
)
from .tvm import gen_sampling_tvm_binding as gen_sampling_tvm_binding
````

## File: jit/attention/pytorch.py
````python
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
    ]
    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=sm100a_nvcc_flags,
    )
````

## File: jit/attention/tvm.py
````python
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
````

## File: jit/attention/utils.py
````python
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

from typing import List


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
````

## File: jit/__init__.py
````python
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

import ctypes
import os

# Re-export
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_mla_tvm_binding as gen_batch_mla_tvm_binding
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_decode_tvm_binding as gen_customize_batch_decode_tvm_binding,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_batch_prefill_tvm_binding as gen_customize_batch_prefill_tvm_binding,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_sampling_tvm_binding as gen_sampling_tvm_binding
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .core import JitSpec as JitSpec
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import gen_jit_spec as gen_jit_spec
from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
from .core import sm100a_nvcc_flags as sm100a_nvcc_flags

cuda_lib_path = os.environ.get(
    "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
)
if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
    ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
````

## File: jit/activation.py
````python
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

import jinja2

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import write_if_different

activation_templ = r"""
#include <flashinfer/activation.cuh>
#include "pytorch_extension_utils.h"
#include <cuda_runtime.h>

{% set func_name = act_func_name ~ '_and_mul' %}

using namespace flashinfer;

{{ act_func_def }}

void {{ func_name }}(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const c10::cuda::OptionalCUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, {{ act_func_name }}>;

    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),
                       static_cast<c_type*>(input.data_ptr()), d);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Failed to launch kernel: ", cudaGetErrorString(err));

    return true;
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("{{ func_name }}", {{ func_name }});
}
"""


def get_act_and_mul_cu_str(act_func_name: str, act_func_def: str) -> str:
    template = jinja2.Template(activation_templ)
    return template.render(act_func_name=act_func_name, act_func_def=act_func_def)


def gen_act_and_mul_module(act_func_name: str, act_func_def: str) -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR
    os.makedirs(gen_directory, exist_ok=True)
    sources = [gen_directory / f"{act_func_name}_and_mul.cu"]
    write_if_different(
        sources[0],
        get_act_and_mul_cu_str(act_func_name, act_func_def),
    )
    return gen_jit_spec(
        f"{act_func_name}_and_mul",
        sources,
    )
````

## File: jit/core.py
````python
import dataclasses
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("flashinfer.jit: " + msg)


logger = FlashInferJITLogger("flashinfer.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


def clear_cache_dir():
    if os.path.exists(jit_env.FLASHINFER_JIT_DIR):
        import shutil

        shutil.rmtree(jit_env.FLASHINFER_JIT_DIR)


sm90a_nvcc_flags = ["-gencode=arch=compute_90a,code=sm_90a"]
sm100a_nvcc_flags = ["-gencode=arch=compute_100a,code=sm_100a"]


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]

    @property
    def ninja_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

    @property
    def library_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

    @property
    def aot_path(self) -> Path:
        return jit_env.FLASHINFER_AOT_DIR / self.name / f"{self.name}.so"

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        ninja_path.parent.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool) -> None:
        tmpdir = get_tmpdir()
        with FileLock(tmpdir / f"{self.name}.lock", thread_local=False):
            run_ninja(jit_env.FLASHINFER_JIT_DIR, self.ninja_path, verbose)

    def build_and_load(self):
        if self.aot_path.exists():
            so_path = self.aot_path
        else:
            so_path = self.library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose)
        torch.ops.load_library(so_path)
        return getattr(torch.ops, self.name)


def gen_jit_spec(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
) -> JitSpec:
    check_cuda_arch()
    verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads=4",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    if extra_include_paths is not None:
        extra_include_paths = [Path(x) for x in extra_include_paths]
    sources = [Path(x) for x in sources]

    spec = JitSpec(
        name=name,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=extra_include_paths,
    )
    spec.write_ninja()
    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.FLASHINFER_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def build_jit_specs(
    specs: List[JitSpec],
    verbose: bool = False,
    skip_prebuilt: bool = True,
) -> None:
    lines: List[str] = []
    for spec in specs:
        if skip_prebuilt and spec.aot_path.exists():
            continue
        lines.append(f"subninja {spec.ninja_path}")
    if not lines:
        return

    lines = ["ninja_required_version = 1.3"] + lines + [""]

    tmpdir = get_tmpdir()
    with FileLock(tmpdir / "flashinfer_jit.lock", thread_local=False):
        ninja_path = tmpdir / "flashinfer_jit.ninja"
        write_if_different(ninja_path, "\n".join(lines))
        run_ninja(jit_env.FLASHINFER_JIT_DIR, ninja_path, verbose)


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags=None,
    extra_include_paths=None,
):
    # TODO(lequn): Remove this function and use JitSpec directly.
    warnings.warn(
        "load_cuda_ops is deprecated. Use JitSpec directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    spec = gen_jit_spec(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
    )
    return spec.build_and_load()
````

## File: jit/cpp_ext.py
````python
# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    CUDA_HOME,
    _get_cuda_arch_flags,
    _get_num_workers,
    _get_pybind11_abi_build_flags,
)

from . import env as jit_env


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
) -> str:
    system_includes = [
        sysconfig.get_path("include"),
        "$torch_home/include",
        "$torch_home/include/torch/csrc/api/include",
        "$cuda_home/include",
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]

    common_cflags = [
        "-DTORCH_EXTENSION_NAME=$name",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DPy_LIMITED_API=0x03090000",
    ]
    common_cflags += _get_pybind11_abi_build_flags()
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for dir in extra_include_dirs:
            common_cflags.append(f"-I{dir.resolve()}")
    for dir in system_includes:
        common_cflags.append(f"-isystem {dir}")

    cflags = [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
    ]
    cuda_cflags += _get_cuda_arch_flags(extra_cuda_cflags)
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    ldflags = [
        "-shared",
        "-L$torch_home/lib",
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-L$cuda_home/lib64",
        "-lcudart",
    ]
    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    cuda_home = CUDA_HOME or "/usr/local/cuda"
    nvcc = os.environ.get("PYTORCH_NVCC", "$cuda_home/bin/nvcc")

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"torch_home = {_TORCH_PATH}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule cuda_compile",
        "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")

    lines.append("")
    lines.append("build $name/$name.so: link " + " ".join(objects))
    lines.append("default $name/$name.so")
    lines.append("")

    return "\n".join(lines)


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
````

## File: jit/env.py
````python
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

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags


def _get_workspace_dir_name() -> pathlib.Path:
    try:
        with warnings.catch_warnings():
            # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
            warnings.filterwarnings(
                "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
            )
            flags = _get_cuda_arch_flags()
        arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    except Exception:
        arch = "noarch"
    flashinfer_base = os.getenv(
        "FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix()
    )
    # e.g.: $HOME/.cache/flashinfer/75_80_89_90/
    return pathlib.Path(flashinfer_base) / ".cache" / "flashinfer" / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[1]
FLASHINFER_DATA = _package_root / "data"
FLASHINFER_INCLUDE_DIR = _package_root / "data" / "include"
FLASHINFER_CSRC_DIR = _package_root / "data" / "csrc"
# FLASHINFER_SRC_DIR = _package_root / "data" / "src"
FLASHINFER_TVM_BINDING_DIR = _package_root / "data" / "tvm_binding"
FLASHINFER_AOT_DIR = _package_root / "data" / "aot"
CUTLASS_INCLUDE_DIRS = [
    _package_root / "data" / "cutlass" / "include",
    _package_root / "data" / "cutlass" / "tools" / "util" / "include",
]
````

## File: jit/utils.py
````python
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

import pathlib

import torch


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.int32: "int32_t",
    torch.uint32: "uint32_t",
    torch.int64: "int64_t",
    torch.uint64: "uint64_t",
}

filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int32: "i32",
    torch.uint32: "u32",
    torch.int64: "i64",
    torch.uint64: "u64",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

mask_mode_literal = {
    0: "MaskMode::kNone",
    1: "MaskMode::kCausal",
    2: "MaskMode::kCustom",
    3: "MaskMode::kMultiItemScoring",
}
````

## File: profiler/__init__.py
````python
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

import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    block_group_tag = tag >> 12
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        event = event_names[event_idx]
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()
````

## File: triton/kernels/activation.py
````python
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from flashinfer.triton.kernels.quant import scale_and_clamp


@triton.jit
def silu_and_mul_kernel(
    o_ptr,
    o_stride,
    o_scale_ptr,
    x_ptr,
    x_stride,
    x_scale_ptr,
    d,
    BLOCK_SIZE: tl.constexpr,
    HAS_X_SCALE: tl.constexpr,
    HAS_O_SCALE: tl.constexpr,
) -> None:
    """Sigmoid Linear Unit and Multiplication Kernel

    Args:
        o_ptr:       Pointer to the 2D output tensor.
        o_stride:    Output tensor stride.
        o_scale_ptr: The optional, known scale of the output activations.
        x_ptr:       Pointer to the 2D input tensor.
        x_stride:    Input tensor stride.
        x_scale_ptr: The optional, known scale of the input tensor.
        d:           The number of elements along the second dimension.
        BLOCK_SIZE:  Tunable block size to process in each kernel.

    Operating on a 2D grid, computes the following:

    ```
    out[i, j] = sigmoid(x[i, j]) * x[i, j] * x[i, j + d]
    ```

    If scales are provided, the input and output tensors are scaled.
    """

    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    if HAS_X_SCALE:
        x_scale = tl.load(x_scale_ptr)
        a *= x_scale
        b *= x_scale

    result = tl.sigmoid(a) * a * b

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_row_ptr + offsets, result, mask=mask)
````

## File: triton/kernels/cascade.py
````python
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def state_merge(o, m, d, other_o, other_m, other_d):
    m_max = tl.maximum(m, other_m)
    d = d * tl.exp2(m - m_max) + other_d * tl.exp2(other_m - m_max)
    o = o * tl.exp2(m - m_max) + other_o * tl.exp2(other_m - m_max)
    return o, m_max, d


@triton.jit
def state_normalize(o, m, d):
    o = o / d
    return o, m, d


@triton.jit
def state_get_lse(o, m, d):
    return m + tl.log2(d)


@triton.jit
def merge_state_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            s_a_val = tl.load(s_a_ptr + pos * num_heads + head_idx)
            s_b_val = tl.load(s_b_ptr + pos * num_heads + head_idx)

            offsets = (pos * num_heads + head_idx) * head_dim + tx
            v_a = tl.load(v_a_ptr + offsets)
            v_b = tl.load(v_b_ptr + offsets)

            v_merged, s_max, d = state_merge(
                o=v_a, m=s_a_val, d=1, other_o=v_b, other_m=s_b_val, other_d=1
            )
            v_merged, s_max, d = state_normalize(v_merged, s_max, d)
            v_merged_offset = (pos * num_heads + head_idx) * head_dim + tx
            tl.store(v_merged_ptr + v_merged_offset, v_merged)

            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx,
                    tl.log2(d) + s_max,
                )


@triton.jit
def merge_state_in_place_kernel(
    v_ptr,
    s_ptr,
    v_other_ptr,
    s_other_ptr,
    num_heads,
    head_dim,
    mask_ptr,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    if mask_ptr:
        if tl.load(mask_ptr + pos) == 0:
            return

    for head_idx in tl.range(bdy):
        s_val = tl.load(s_ptr + pos * num_heads + head_idx)
        s_other_val = tl.load(s_other_ptr + pos * num_heads + head_idx)
        s_max = tl.maximum(s_val, s_other_val)
        s_val = tl.exp2(s_val - s_max)
        s_other_val = tl.exp2(s_other_val - s_max)
        scale = s_val / (s_val + s_other_val)
        other_scale = s_other_val / (s_val + s_other_val)
        for tx in tl.range(bdx):
            offset = (pos * num_heads + head_idx) * head_dim + tx
            v_vec = tl.load(v_ptr + offset)
            v_other_vec = tl.load(v_other_ptr + offset)
            v_vec = scale * v_vec + other_scale * v_other_vec
            tl.store(v_ptr + offset, v_vec)
        if s_ptr:
            tl.store(
                s_ptr + pos * num_heads + head_idx,
                tl.log2(s_val + s_other_val) + s_max,
            )


@triton.jit
def merge_states_kernel(
    v_ptr,
    s_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_index_sets,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)

    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            o, m, d = 0.0, -5e4, 1.0
            for iter in tl.range(num_index_sets):
                s = tl.load(
                    s_ptr + (pos * num_index_sets + iter) * num_heads + head_idx
                )
                v = tl.load(
                    v_ptr
                    + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim
                    + tx
                )
                o, m, d = state_merge(o, m, d, v, s, 1)
            o, m, d = state_normalize(o, m, d)
            tl.store(v_merged_ptr + (pos * num_heads + head_idx) * head_dim + tx, o)
            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx, state_get_lse(o, m, d)
                )


@triton.jit
def variable_length_merge_states_kernel(
    v_ptr,
    s_ptr,
    indptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            o, m, d = 0.0, -5e4, 1.0
            for iter in tl.range(tl.load(indptr + pos), tl.load(indptr + pos + 1)):
                s = tl.load(s_ptr + iter * num_heads + head_idx)
                v = tl.load(v_ptr + (iter * num_heads + head_idx) * head_dim + tx)
                o, m, d = state_merge(o, m, d, v, s, 1)
            o, m, d = state_normalize(o, m, d)
            tl.store(v_merged_ptr + (pos * num_heads + head_idx) * head_dim + tx, o)
            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx, state_get_lse(o, m, d)
                )
````

## File: triton/kernels/norm.py
````python
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from flashinfer.triton.kernels.quant import scale_and_clamp


@triton.jit
def rms_norm_kernel(
    n,
    b,
    x_ptr,
    x_stride,
    x_scale_ptr,
    r_ptr,
    r_stride,
    w_ptr,
    o_ptr,
    o_stride,
    o_scale_ptr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_IN_SCALE: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
    HAS_OUTPUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)

    # If r_ptr is present, the input to norm is x + r.
    x_row = x_ptr + i * x_stride
    o_row = o_ptr + i * o_stride if HAS_OUTPUT else x_row
    r_row = r_ptr + i * r_stride if HAS_RESIDUAL else None

    x_scale = tl.load(x_scale_ptr) if HAS_IN_SCALE else None
    o_scale = tl.load(o_scale_ptr) if HAS_OUT_SCALE else None

    # Find the root mean square for the given row.
    square_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_IN_SCALE:
            x *= x_scale

        if HAS_RESIDUAL:
            r = tl.load(r_row + offsets, mask=mask, other=0.0).to(tl.float32)
            x += r
            tl.store(r_row + offsets, x, mask=mask)

        square_sum += x * x

    # Compute the norm.
    rms = tl.rsqrt(tl.sum(square_sum) / n + EPS)

    # x[i] = r[i] + x[i] / rms * weight[i]
    output_dtype = o_row.dtype.element_ty
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        if HAS_RESIDUAL:
            x = tl.load(r_row + offsets, mask=mask).to(tl.float32)
        else:
            x = tl.load(x_row + offsets, mask=mask).to(tl.float32)
            if HAS_IN_SCALE:
                x *= x_scale

        w = tl.load(w_ptr + offsets, mask=mask).to(tl.float32)

        # Multiply x with RMS on float32, but cast to the narrower type before
        # multiplying with the weights to replicate the HF behaviour precisely.
        result = w * (x * rms)
        if HAS_OUT_SCALE:
            result = scale_and_clamp(result, o_scale, output_dtype)
        tl.store(o_row + offsets, result, mask=mask)
````

## File: triton/kernels/quant.py
````python
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def scale_and_clamp(x, scale, dtype):
    """Scales a value and clamps it to the range of the target dtype.

    This function hard-wires the upper/lower bounds in order to be
    compatible with both `torch.compile` and `triton.jit`.
    """
    if dtype == tl.float8e4nv:
        clamp_min = -448.0
        clamp_max = 448.0
    elif dtype == tl.float8e5:
        clamp_min = -57344.0
        clamp_max = 57344.0
    elif dtype == tl.float16:
        clamp_min = -65504.0
        clamp_max = 65504.0
    elif dtype == tl.bfloat16:
        clamp_min = -3.3895313892515355e38
        clamp_max = 3.3895313892515355e38
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")

    return tl.clamp(x.to(tl.float32) * scale, clamp_min, clamp_max).to(dtype)
````

## File: triton/kernels/sm_constraint_gemm.py
````python
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [64]
        for s in ([3])
        for w in [4]
    ]


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)

        c = tl.fma(c, alpha, beta * tl.load(c_ptrs, mask=c_mask))
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            acc0 = tl.fma(acc0, alpha, beta * c_desc.load([offs_cm, offs_cn]))
            acc1 = tl.fma(
                acc1, alpha, beta * c_desc.load([offs_cm, offs_cn + BLOCK_SIZE_N // 2])
            )
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = tl.fma(
                accumulator, alpha, beta * c_desc.load([offs_cm, offs_cn])
            )
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


# only for testing
@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = tl.fma(c, alpha, beta * tl.load(c_ptrs, mask=c_mask))
    tl.store(c_ptrs, c, mask=c_mask)
````

## File: triton/__init__.py
````python
from . import cascade  # noqa: F401
from . import sm_constraint_gemm  # noqa: F401
````

## File: triton/activation.py
````python
from collections.abc import Mapping
from typing import Optional

import torch
import triton  # type: ignore[import]

from flashinfer.triton.kernels.activation import silu_and_mul_kernel


def silu_and_mul(
    x: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sigmoid Linear Unit and Multiplication

    Computes `silu(x[:,:d]) * x[:, d:]`, where `d = x.shape[-1] // 2.

    If the scale of `x` is `x_scale`, the scale applied to the output
    is the square of that, as the sigmoid function ranges in (0, 1).

    Args:
        x: The input tensor, of shape `(b, 2 * d)`.
        x_scale: An optional scale which was applied to `x`.
        o_scale: The scale to apply to the output.
        dtype: The desired output dtype.

    Returns:
        The output activation, of shape `(b, d)`.
    """

    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    o = torch.empty((b, d), dtype=o_dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    silu_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        o_scale_ptr=o_scale,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=x_scale,
        d=d,
        BLOCK_SIZE=1024,
        HAS_X_SCALE=x_scale is not None,
        HAS_O_SCALE=o_scale is not None,
    )

    return o
````

## File: triton/cascade.py
````python
from typing import Optional

import torch

from .kernels.cascade import (
    merge_state_in_place_kernel,
    merge_state_kernel,
    merge_states_kernel,
    variable_length_merge_states_kernel,
)
from .utils import check_device, check_dim, check_input, check_shape


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    check_input(v_a)
    check_input(s_a)
    check_input(v_b)
    check_input(s_b)
    check_device([v_a, s_a, v_b, s_b])
    check_dim(3, v_a)
    check_dim(2, s_a)
    check_dim(3, v_b)
    check_dim(2, s_b)
    check_shape(v_a, v_b)
    check_shape(s_a, s_b)
    assert v_a.size(0) == s_a.size(0)
    assert v_a.size(1) == s_b.size(1)
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    seq_len = v_a.size(0)
    num_heads = v_a.size(1)
    head_dim = v_a.size(2)
    v_merged = torch.empty_like(v_a).to(s_a.device)
    s_merged = torch.empty((seq_len, num_heads)).to(s_a.device)
    bdx = head_dim
    bdy = num_heads

    merge_state_kernel[lambda meta: (seq_len,)](
        v_a, s_a, v_b, s_b, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy
    )

    return v_merged, s_merged


def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    check_input(v)
    check_input(s)
    check_input(v_other)
    check_input(s_other)
    check_device([v, s, v_other, s_other])
    check_dim(3, v)
    check_dim(2, s)
    check_dim(3, v_other)
    check_dim(2, s_other)
    check_shape(v, v_other)
    check_shape(s, s_other)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    assert s.dtype == torch.float32
    assert s_other.dtype == torch.float32
    if mask is not None:
        check_dim(1, mask)
        assert v.size(0) == mask.size(0)
        assert mask.device == v.device
    seq_len = v.size(0)
    num_heads = v.size(1)
    head_dim = v.size(2)

    bdx = head_dim
    bdy = num_heads
    merge_state_in_place_kernel[(seq_len,)](
        v, s, v_other, s_other, num_heads, head_dim, mask, bdx=bdx, bdy=bdy
    )


def merge_states(v: torch.Tensor, s: torch.Tensor):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(4, v)
    check_dim(3, s)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    assert v.size(2) == s.size(2)
    seq_len = v.size(0)
    num_index_sets = v.size(1)
    num_heads = v.size(2)
    head_dim = v.size(3)
    s = s.to(torch.float32)
    v_merged = torch.empty(
        (seq_len, num_heads, head_dim), dtype=v.dtype, device=v.device
    )
    s_merged = torch.empty((seq_len, num_heads), dtype=s.dtype, device=s.device)

    bdx = head_dim
    bdy = num_heads
    merge_states_kernel[(seq_len,)](
        v,
        s,
        v_merged,
        s_merged,
        num_index_sets,
        num_heads,
        head_dim,
        bdx=bdx,
        bdy=bdy,
    )
    return v_merged, s_merged


def variable_length_merge_states(
    v: torch.Tensor, s: torch.Tensor, indptr: torch.Tensor
):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(3, v)
    check_dim(2, s)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    seq_len = indptr.size(0) - 1
    num_heads = v.size(1)
    head_dim = v.size(2)
    s = s.to(torch.float32)
    indptr = indptr.to(torch.int32)
    v_merged = torch.empty(
        (seq_len, num_heads, head_dim), dtype=v.dtype, device=v.device
    )
    s_merged = torch.empty((seq_len, num_heads), dtype=s.dtype, device=s.device)

    bdx = head_dim
    bdy = num_heads
    variable_length_merge_states_kernel[(seq_len,)](
        v,
        s,
        indptr,
        v_merged,
        s_merged,
        num_heads,
        head_dim,
        bdx=bdx,
        bdy=bdy,
    )
    return v_merged, s_merged
````

## File: triton/gemm.py
````python
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

import triton
import triton.language as tl


@triton.jit
def compute_sm80_group_gemm_args(
    all_problems_ptr,
    x_ptr,
    w_ptr,
    y_ptr,
    x_ld_ptr,
    w_ld_ptr,
    y_ld_ptr,
    x,
    w,
    y,
    xy_indptr,
    w_indices,
    d_in,
    d_out,
    w_column_major,
):

    pid = tl.program_id(0)

    m = tl.load(xy_indptr + pid + 1) - tl.load(xy_indptr + pid)
    k, n = d_in, d_out

    tl.store(all_problems_ptr + pid * 3, m)
    tl.store(all_problems_ptr + pid * 3 + 1, n)
    tl.store(all_problems_ptr + pid * 3 + 2, k)

    w_i = tl.load(w_indices + pid) if w_indices else tl.cast(pid, tl.int64)
    w_curr_ptr = w + w_i * k * n
    tl.store(w_ptr + pid, w_curr_ptr)

    x_curr_ptr = x + tl.load(xy_indptr + pid) * k
    tl.store(x_ptr + pid, x_curr_ptr)

    y_curr_ptr = y + tl.load(xy_indptr + pid) * n
    tl.store(y_ptr + pid, y_curr_ptr)

    tl.store(x_ld_ptr + pid, k)
    tl.store(w_ld_ptr + pid, k if w_column_major else n)
    tl.store(y_ld_ptr + pid, n)


@triton.jit
def compute_sm90_group_gemm_args(
    all_problems_ptr,
    x_ptr,
    w_ptr,
    y_ptr,
    x_stride_ptr,
    w_stride_ptr,
    y_stride_ptr,
    x,
    w,
    y,
    xy_indptr,
    w_indices,
    d_in,
    d_out,
    w_column_major,
):

    pid = tl.program_id(0)

    m = tl.load(xy_indptr + pid + 1) - tl.load(xy_indptr + pid)
    k, n = d_in, d_out

    tl.store(all_problems_ptr + pid * 3, m)
    tl.store(all_problems_ptr + pid * 3 + 1, n)
    tl.store(all_problems_ptr + pid * 3 + 2, k)

    w_i = tl.load(w_indices + pid) if w_indices else tl.cast(pid, tl.int64)
    w_curr_ptr = w + w_i * k * n
    tl.store(w_ptr + pid, w_curr_ptr)

    x_curr_ptr = x + tl.load(xy_indptr + pid) * k
    tl.store(x_ptr + pid, x_curr_ptr)

    y_curr_ptr = y + tl.load(xy_indptr + pid) * n
    tl.store(y_ptr + pid, y_curr_ptr)

    tl.store(x_stride_ptr + pid, k)
    tl.store(w_stride_ptr + pid, k if w_column_major else n)
    tl.store(y_stride_ptr + pid, n)


@triton.jit
def compute_padding_mapping(
    m_indptr,
    padded_m_indptr,
    m_rank,
    padded_m_rank,
):
    pid = tl.program_id(0)
    m_start = tl.load(m_indptr + pid)
    m_end = tl.load(m_indptr + pid + 1)
    padded_m_start = tl.load(padded_m_indptr + pid)
    for i in range(m_end - m_start):
        tl.store(m_rank + m_start + i, m_start + i)
        tl.store(padded_m_rank + m_start + i, padded_m_start + i)
````

## File: triton/norm.py
````python
from typing import Optional

import torch
import triton  # type: ignore[import]

from flashinfer.triton.kernels.norm import rms_norm_kernel


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    in_scale: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> None:
    """RMS norm.

    Computes `out[i,j] = x[i,j] * weight[j] / sqrt(eps + sum(x[i]^2) / n)`.
    """

    b, n = x.shape

    block_size = triton.next_power_of_2(n)
    num_warps = max(8, min(32, block_size // 256))

    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=in_scale,
        r_ptr=None,
        r_stride=0,
        w_ptr=weight,
        o_ptr=out,
        o_stride=out.stride(0),
        o_scale_ptr=out_scale,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=in_scale is not None,
        HAS_OUT_SCALE=out_scale is not None,
        HAS_OUTPUT=True,
        HAS_RESIDUAL=False,
        num_warps=num_warps,
    )


def rms_norm_add_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    x_out: Optional[torch.Tensor] = None,
    x_in_scale: Optional[torch.Tensor] = None,
    x_out_scale: Optional[torch.Tensor] = None,
) -> None:
    """In-place RMS norm with fused residual addition.

    Computes `r = r + x`, followed by `x = rmsnorm(r)`.
    """

    b, n = x.shape

    assert x.shape == residual.shape
    assert x.stride(0) == residual.stride(0)

    block_size = triton.next_power_of_2(n)
    num_warps = min(32, triton.cdiv(block_size, 32))

    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=x_in_scale,
        r_ptr=residual,
        r_stride=residual.stride(0),
        w_ptr=weight,
        o_ptr=x_out,
        o_stride=x_out.stride(0) if x_out is not None else 0,
        o_scale_ptr=x_out_scale,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=x_in_scale is not None,
        HAS_OUT_SCALE=x_out_scale is not None,
        HAS_OUTPUT=x_out is not None,
        HAS_RESIDUAL=True,
        num_warps=num_warps,
    )
````

## File: triton/page.py
````python
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


import triton
import triton.language as tl


@triton.jit
def get_batch_indices_positions_kernel(
    append_indptr,
    seq_lens_ptr,
    batch_indices_ptr,
    positions_ptr,
    num_stages: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    batch_start = tl.load(append_indptr + batch_idx)
    batch_end = tl.load(append_indptr + batch_idx + 1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)

    for i in tl.range(batch_start, batch_end, 128, num_stages=num_stages):
        offsets = tl.arange(0, 128) + i
        mask = offsets < batch_end
        tl.store(batch_indices_ptr + offsets, batch_idx, mask)
        tl.store(positions_ptr + offsets, offsets + seq_len - batch_end, mask)
````

## File: triton/sm_constraint_gemm.py
````python
from typing import Optional

import torch
import triton

from .kernels.sm_constraint_gemm import (
    gemm_kernel,
    gemm_kernel_descriptor_persistent,
    gemm_kernel_persistent,
)
from .utils import check_device, check_dim, check_input


def gemm_persistent(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None, num_sms=None):
    """
    GEMM operation with SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """

    # Check inputs.
    check_input(a)
    # b can be non-contiguous
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[1] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    # Set num_sms to be 100% of the available SMs
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    gemm_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
        NUM_SMS=num_sms,
    )
    return c


def gemm(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None):
    """
    GEMM operation without SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """
    # Check inputs.
    check_input(a)
    # b can be non-contiguous
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[1] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
    )
    return c


def gemm_descriptor_persistent(
    a,
    b,
    c=None,
    alpha=1.0,
    beta=0.0,
    out_dtype=None,
    num_sms=None,
    EPILOGUE_SUBTILE=False,
):
    """
    GEMM operation with SM constraint by Triton.
    Requires TMA support and descriptor creation.
    C = alpha * (a @ b.T) + beta * C

    Note:
        - K and N must be greater than 16B.
        - Support float16, float8_e4m3fn, bfloat16.
        - float32 is not supported due to performance issues.

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (N, K)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
        EPILOGUE_SUBTILE: Whether to use the epilogue subtile optimization.
    """
    # Check inputs.
    check_input(a)
    check_input(b)
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[1], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[0] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    # check on TMA tensor map swizzling granularity
    # Swizzle 16B chunks within at least 32B span
    if dtype == torch.float8_e4m3fn:
        assert K >= 16, "Least chunk size must be 16B"
        assert N >= 16, "Least chunk size must be 16B"
    else:
        assert K >= 8, "Least chunk size must be 16B"
        assert N >= 8, "Least chunk size must be 16B"

    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    gemm_kernel_descriptor_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        alpha,
        beta,
        NUM_SMS=num_sms,  #
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128 if dtype != torch.float32 else 64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=8,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
    )
    return c
````

## File: triton/utils.py
````python
from typing import List

import torch


def check_input(x: torch.Tensor):
    assert x.is_cuda, f"{str(x)} must be a CUDA Tensor"
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: torch.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == b.dim(), "tensors should have same dim"
    for i in range(a.dim()):
        assert a.size(i) == b.size(
            i
        ), f"tensors shape mismatch, {a.size()} and {b.size()}"


def check_device(tensors: List[torch.Tensor]):
    device = tensors[0].device
    for t in tensors:
        assert (
            t.device == device
        ), f"All tensors should be on the same device, but got {device} and {t.device}"
````

## File: __init__.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

try:
    from ._build_meta import __version__ as __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"


from . import jit as jit
from .activation import gelu_and_mul as gelu_and_mul
from .activation import gelu_tanh_and_mul as gelu_tanh_and_mul
from .activation import silu_and_mul as silu_and_mul
from .cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
)
from .cascade import merge_state as merge_state
from .cascade import merge_state_in_place as merge_state_in_place
from .cascade import merge_states as merge_states
from .decode import (
    BatchDecodeMlaWithPagedKVCacheWrapper as BatchDecodeMlaWithPagedKVCacheWrapper,
)
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
)
from .decode import (
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
)
from .decode import single_decode_with_kv_cache as single_decode_with_kv_cache
from .gemm import SegmentGEMMWrapper as SegmentGEMMWrapper
from .gemm import bmm_fp8 as bmm_fp8
from .mla import BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper
from .norm import fused_add_rmsnorm as fused_add_rmsnorm
from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from .norm import gemma_rmsnorm as gemma_rmsnorm
from .norm import rmsnorm as rmsnorm
from .page import append_paged_kv_cache as append_paged_kv_cache
from .page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from .page import get_batch_indices_positions as get_batch_indices_positions
from .page import get_seq_lens as get_seq_lens
from .pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
)
from .prefill import (
    BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
)
from .prefill import single_prefill_with_kv_cache as single_prefill_with_kv_cache
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .quantization import packbits as packbits
from .quantization import segment_packbits as segment_packbits
from .rope import apply_llama31_rope as apply_llama31_rope
from .rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
from .rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
from .rope import (
    apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
)
from .rope import apply_rope as apply_rope
from .rope import apply_rope_inplace as apply_rope_inplace
from .rope import apply_rope_pos_ids as apply_rope_pos_ids
from .rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
from .rope import apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache
from .rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
)
from .sampling import chain_speculative_sampling as chain_speculative_sampling
from .sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
from .sampling import sampling_from_logits as sampling_from_logits
from .sampling import sampling_from_probs as sampling_from_probs
from .sampling import top_k_mask_logits as top_k_mask_logits
from .sampling import top_k_renorm_probs as top_k_renorm_probs
from .sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
from .sampling import (
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
)
from .sampling import top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs
from .sampling import top_p_renorm_probs as top_p_renorm_probs
from .sampling import top_p_sampling_from_probs as top_p_sampling_from_probs
from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper
````

## File: activation.py
````python
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

from types import SimpleNamespace

import torch

from .jit import JitSpec
from .jit import gen_act_and_mul_module as gen_act_and_mul_module_impl
from .utils import register_custom_op, register_fake_op

silu_def_cu_str = r"""
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}
"""

gelu_def_cu_str = r"""
__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}
"""

gelu_def_tanh_cu_str = r"""
__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}
"""

act_func_def_str = {
    "silu": silu_def_cu_str,
    "gelu": gelu_def_cu_str,
    "gelu_tanh": gelu_def_tanh_cu_str,
}


_jit_modules = {}


def gen_act_and_mul_module(act_func_name: str) -> JitSpec:
    return gen_act_and_mul_module_impl(act_func_name, act_func_def_str[act_func_name])


def get_act_and_mul_module(act_func_name: str):
    global _jit_modules
    if act_func_name not in _jit_modules:
        module = gen_act_and_mul_module(act_func_name).build_and_load()

        # torch library for act_and_mul
        fname = f"{act_func_name}_and_mul"
        fn = getattr(module, fname).default

        @register_custom_op(f"flashinfer::{fname}", mutates_args=("out",))
        def _act_and_mul(
            out: torch.Tensor, input: torch.Tensor, enable_pdl: bool = False
        ) -> None:
            fn(out, input, enable_pdl)

        @register_fake_op(f"flashinfer::{fname}")
        def _fake_act_and_mul(
            out: torch.Tensor, input: torch.Tensor, enable_pdl: bool = False
        ) -> None:
            pass

        # Register the module
        _jit_modules[act_func_name] = SimpleNamespace(**{fname: _act_and_mul})

    return _jit_modules[act_func_name]


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def silu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: bool = False
) -> torch.Tensor:
    r"""Fused SiLU and Mul operation.

    ``silu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("silu").silu_and_mul(
        out,
        input,
        enable_pdl,
    )
    return out


def gelu_tanh_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: bool = False
) -> torch.Tensor:
    r"""Fused GeLU Tanh and Mul operation.

    ``gelu(tanh(input[..., :hidden_size])) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu_tanh").gelu_tanh_and_mul(out, input, enable_pdl)
    return out


def gelu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: bool = False
) -> torch.Tensor:
    r"""Fused GeLU and Mul operation.

    ``gelu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu").gelu_and_mul(out, input, enable_pdl)
    return out
````

## File: aot.py
````python
import argparse
import os
import shutil
from itertools import product
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.cpp_extension import _get_cuda_arch_flags

from .activation import act_func_def_str, gen_act_and_mul_module
from .cascade import gen_cascade_module
from .comm import gen_comm_module
from .gemm import gen_gemm_module, gen_gemm_sm90_module, gen_gemm_sm100_module
from .jit import JitSpec, build_jit_specs
from .jit import env as jit_env
from .jit import (
    gen_batch_decode_module,
    gen_batch_mla_module,
    gen_batch_prefill_module,
    gen_fmha_cutlass_sm100a_module,
    gen_single_decode_module,
    gen_single_prefill_module,
)
from .mla import gen_mla_module
from .norm import gen_norm_module
from .page import gen_page_module
from .quantization import gen_quantization_module
from .rope import gen_rope_module
from .sampling import gen_sampling_module


def gen_fa2(
    dtype_qo: torch.dtype,
    dtype_kv: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> List[JitSpec]:
    if dtype_qo.itemsize == dtype_kv.itemsize and dtype_qo != dtype_kv:
        return []
    if dtype_qo.itemsize == 1:
        return []  # fp8 tensor cores not supported in fa2
    return [
        gen_single_prefill_module(
            backend="fa2",
            dtype_q=dtype_qo,
            dtype_kv=dtype_kv,
            dtype_o=dtype_qo,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_fp16_qk_reduction=False,
        ),
        gen_batch_prefill_module(
            backend="fa2",
            dtype_q=dtype_qo,
            dtype_kv=dtype_kv,
            dtype_o=dtype_qo,
            dtype_idx=torch.int32,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_fp16_qk_reduction=False,
        ),
        gen_single_decode_module(
            dtype_q=dtype_qo,
            dtype_kv=dtype_kv,
            dtype_o=dtype_qo,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
        ),
        gen_batch_decode_module(
            dtype_q=dtype_qo,
            dtype_kv=dtype_kv,
            dtype_o=dtype_qo,
            dtype_idx=torch.int32,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
        ),
    ]


def gen_fa3(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> List[JitSpec]:
    if dtype_q != dtype_kv:
        return []  # fa3 template do not support mixed precision
    if dtype_q.itemsize == 2:
        if dtype_q != dtype_o:
            return []  # for fp16, dtype_o must be the same as dtype_q/dtype_kv

    if dtype_kv.itemsize == 1:
        if head_dim_qk == 192 or head_dim_qk == 64:
            return []  # (192, 128) & (64, 64) not supported for fp8 yet.

    return [
        gen_single_prefill_module(
            backend="fa3",
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            dtype_o=dtype_o,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_fp16_qk_reduction=False,
        ),
        gen_batch_prefill_module(
            backend="fa3",
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            dtype_o=dtype_o,
            dtype_idx=torch.int32,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_fp16_qk_reduction=False,
        ),
    ]


def gen_attention(
    f16_dtype_: List[torch.dtype],
    f8_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    fa3_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
    has_sm90: bool,
    has_sm100: bool,
    add_gemma: bool,
) -> List[JitSpec]:
    head_dim_ckv = 512
    head_dim_kpe = 64
    jit_specs: List[JitSpec] = []

    # FA2 MHA / MQA / GQA
    for (
        (head_dim_qk, head_dim_vo),
        dtype_qo,
        dtype_kv,
        use_sliding_window,
        use_logits_soft_cap,
    ) in product(
        fa2_head_dim_,
        f16_dtype_,
        f16_dtype_ + f8_dtype_,
        use_sliding_window_,
        use_logits_soft_cap_,
    ):
        jit_specs += gen_fa2(
            dtype_qo=dtype_qo,
            dtype_kv=dtype_kv,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
        )

    # FA3 MHA / MQA / GQA
    if has_sm90:
        for (
            (head_dim_qk, head_dim_vo),
            dtype_qkv,
            dtype_o,
            use_sliding_window,
            use_logits_soft_cap,
        ) in product(
            fa3_head_dim_,
            f16_dtype_ + f8_dtype_,
            f16_dtype_,
            use_sliding_window_,
            use_logits_soft_cap_,
        ):
            jit_specs += gen_fa3(
                dtype_q=dtype_qkv,
                dtype_kv=dtype_qkv,
                dtype_o=dtype_o,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_soft_cap,
            )

    # Gemma
    if add_gemma:
        for (
            dtype_qo,
            dtype_kv,
            (use_sliding_window, use_logits_soft_cap),
        ) in product(
            f16_dtype_,
            f16_dtype_ + f8_dtype_,
            [(True, True)],
        ):
            jit_specs += gen_fa2(
                dtype_qo=dtype_qo,
                dtype_kv=dtype_kv,
                head_dim_qk=256,
                head_dim_vo=256,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_soft_cap,
            )
        if has_sm90:
            for (
                dtype_qkv,
                dtype_o,
                (use_sliding_window, use_logits_soft_cap),
            ) in product(
                f16_dtype_ + f8_dtype_,
                f16_dtype_,
                [(True, True)],
            ):
                jit_specs += gen_fa3(
                    dtype_q=dtype_qkv,
                    dtype_kv=dtype_qkv,
                    dtype_o=dtype_o,
                    head_dim_qk=256,
                    head_dim_vo=256,
                    use_sliding_window=use_sliding_window,
                    use_logits_soft_cap=use_logits_soft_cap,
                )

    # fmha_cutlass_sm100a
    # NOTE: currently there's only one uri.
    if has_sm100:
        jit_specs.append(
            gen_fmha_cutlass_sm100a_module(
                dtype_q=torch.bfloat16,
                dtype_kv=torch.bfloat16,
                dtype_o=torch.bfloat16,
                dtype_idx=torch.int32,
                head_dim_qk=128,
                head_dim_vo=128,
                pos_encoding_mode=0,
                use_sliding_window=False,
                use_logits_soft_cap=False,
            )
        )

    # MLA
    # NOTE: fp8 kv not supported in MLA
    mla_backend_ = ["fa2"]
    if has_sm90:
        mla_backend_.append("fa3")
    for dtype_qo in f16_dtype_:
        dtype_kv = dtype_qo
        for backend in mla_backend_:
            jit_specs.append(
                gen_batch_mla_module(
                    backend=backend,
                    dtype_q=dtype_qo,
                    dtype_kv=dtype_kv,
                    dtype_o=dtype_qo,
                    dtype_idx=torch.int32,
                    head_dim_ckv=head_dim_ckv,
                    head_dim_kpe=head_dim_kpe,
                    use_profiler=False,
                )
            )

    # MLA SM100
    if has_sm100:
        jit_specs.append(gen_mla_module())

    return jit_specs


def gen_all_modules(
    f16_dtype_: List[torch.dtype],
    f8_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    fa3_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
    has_sm90: bool,
    has_sm100: bool,
    add_gemma: bool,
) -> List[JitSpec]:
    jit_specs: List[JitSpec] = []

    jit_specs += gen_attention(
        f16_dtype_,
        f8_dtype_,
        fa2_head_dim_,
        fa3_head_dim_,
        use_sliding_window_,
        use_logits_soft_cap_,
        has_sm90,
        has_sm100,
        add_gemma,
    )
    for act_name in act_func_def_str:
        jit_specs.append(gen_act_and_mul_module(act_name))
    jit_specs.append(gen_gemm_module())
    if has_sm90:
        jit_specs.append(gen_gemm_sm90_module())
    if has_sm100:
        jit_specs.append(gen_gemm_sm100_module())
    jit_specs += [
        gen_cascade_module(),
        gen_comm_module(),
        gen_norm_module(),
        gen_page_module(),
        gen_quantization_module(),
        gen_rope_module(),
        gen_sampling_module(),
    ]

    # dedup
    names = set()
    ret: List[JitSpec] = []
    for jit_spec in jit_specs:
        if jit_spec.name not in names:
            names.add(jit_spec.name)
            ret.append(jit_spec)
    return ret


def copy_built_kernels(
    jit_specs: List[JitSpec],
    out_dir: Path,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for jit_spec in jit_specs:
        src = jit_env.FLASHINFER_JIT_DIR / jit_spec.name / f"{jit_spec.name}.so"
        dst = out_dir / jit_spec.name / f"{jit_spec.name}.so"
        dst.parent.mkdir(exist_ok=False, parents=False)
        shutil.copy2(src, dst)


def parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1"):
        return True
    elif s.lower() in ("false", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")


def parse_head_dim(head_dim: str) -> Tuple[int, int]:
    qo, kv = map(int, head_dim.split(","))
    return qo, kv


def main():
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build all modules"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory",
    )
    parser.add_argument(
        "--fa2-head-dim",
        nargs="*",
        help="FA2 head dim pair of qk and vo, separated by comma",
    )
    parser.add_argument(
        "--fa3-head-dim",
        nargs="*",
        help="FA3 head dim pair of qk and vo, separated by comma",
    )
    parser.add_argument(
        "--f16-dtype",
        nargs="*",
        choices=["float16", "bfloat16"],
        help="16-bit data type",
    )
    parser.add_argument(
        "--f8-dtype",
        nargs="*",
        choices=["float8_e4m3fn", "float8_e5m2"],
        help="8-bit data type",
    )
    parser.add_argument(
        "--use-sliding-window",
        nargs="*",
        help="Use sliding window attention",
    )
    parser.add_argument(
        "--use-logits-soft-cap",
        nargs="*",
        help="Use logits soft cap",
    )
    parser.add_argument(
        "--add-gemma",
        type=parse_bool,
        help="Add kernels for Gemma Model (head_dim=256, use_sliding_window, use_logits_soft_cap)",
    )
    args = parser.parse_args()

    # Default values
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "aot-ops"
    build_dir = project_root / "build" / "aot"
    fa2_head_dim_ = [
        (64, 64),
        (128, 128),
        # (256, 256),
    ]
    fa3_head_dim_ = [
        (192, 128),
        (128, 128),
        # (64, 64),
        # (256, 256),
    ]
    f16_dtype_ = [
        torch.float16,
        torch.bfloat16,
    ]
    f8_dtype_ = [
        torch.float8_e4m3fn,
        # torch.float8_e5m2,
    ]
    use_sliding_window_ = [
        False,
        # True,
    ]
    use_logits_soft_cap_ = [
        False,
        # True,
    ]
    add_gemma = True

    # Override
    if args.out_dir:
        out_dir = Path(args.out_dir)
    if args.build_dir:
        build_dir = Path(args.build_dir)
    if args.fa2_head_dim:
        fa2_head_dim_ = [parse_head_dim(dim) for dim in args.fa2_head_dim]
    if args.fa3_head_dim:
        fa3_head_dim_ = [parse_head_dim(dim) for dim in args.fa3_head_dim]
    if args.f16_dtype:
        f16_dtype_ = [getattr(torch, dtype) for dtype in args.f16_dtype]
    if args.f8_dtype:
        f8_dtype_ = [getattr(torch, dtype) for dtype in args.f8_dtype]
    if args.use_sliding_window:
        use_sliding_window_ = [parse_bool(s) for s in args.use_sliding_window]
    if args.use_logits_soft_cap:
        use_logits_soft_cap_ = [parse_bool(s) for s in args.use_logits_soft_cap]
    if args.add_gemma is not None:
        add_gemma = bool(args.add_gemma)

    # Cuda Arch
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        raise RuntimeError("Please explicitly set env var TORCH_CUDA_ARCH_LIST.")
    gencode_flags = _get_cuda_arch_flags()
    has_sm90 = any("compute_90" in flag for flag in gencode_flags)
    has_sm100 = any("compute_100" in flag for flag in gencode_flags)

    # Update data dir
    jit_env.FLASHINFER_CSRC_DIR = project_root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = project_root / "include"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        project_root / "3rdparty" / "cutlass" / "include",
        project_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]

    # Update workdir
    jit_env.FLASHINFER_WORKSPACE_DIR = build_dir
    jit_env.FLASHINFER_JIT_DIR = build_dir / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = build_dir / "generated"
    jit_env.FLASHINFER_JIT_DIR.mkdir(parents=True, exist_ok=True)
    jit_env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("AOT build summary:")
    print("  out_dir:", out_dir)
    print("  build_dir:", build_dir)
    print("  fa2_head_dim:", fa2_head_dim_)
    print("  fa3_head_dim:", fa3_head_dim_)
    print("  f16_dtype:", f16_dtype_)
    print("  f8_dtype:", f8_dtype_)
    print("  use_sliding_window:", use_sliding_window_)
    print("  use_logits_soft_cap:", use_logits_soft_cap_)
    print("  TORCH_CUDA_ARCH_LIST:", os.environ["TORCH_CUDA_ARCH_LIST"])
    print("  has_sm90:", has_sm90)
    print("  has_sm100:", has_sm100)
    print("  add_gemma:", add_gemma)

    # Generate JIT specs
    print("Generating JIT specs...")
    jit_specs = gen_all_modules(
        f16_dtype_,
        f8_dtype_,
        fa2_head_dim_,
        fa3_head_dim_,
        use_sliding_window_,
        use_logits_soft_cap_,
        has_sm90,
        has_sm100,
        add_gemma,
    )
    print("Total ops:", len(jit_specs))

    # Build
    build_jit_specs(jit_specs, verbose=True, skip_prebuilt=False)

    # Copy built kernels
    copy_built_kernels(jit_specs, out_dir)
    print("AOT kernels saved to:", out_dir)


if __name__ == "__main__":
    main()
````

## File: cascade.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

from functools import cache
from typing import Any, List, Optional, Tuple

import torch

from .decode import BatchDecodeWithPagedKVCacheWrapper
from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .prefill import BatchPrefillWithPagedKVCacheWrapper, single_prefill_with_kv_cache
from .utils import register_custom_op, register_fake_op

_cascade_module = None


def gen_cascade_module() -> JitSpec:
    return gen_jit_spec(
        "cascade",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cascade.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_cascade_ops.cu",
        ],
    )


def get_cascade_module():
    global _cascade_module
    if _cascade_module is None:
        _cascade_module = gen_cascade_module().build_and_load()
    return _cascade_module


@cache
def get_module_attr(attr: str) -> Any:
    global _cascade_module
    if _cascade_module is None:
        get_cascade_module()
    return getattr(_cascade_module, attr).default


@register_custom_op("flashinfer::merge_state", mutates_args=())
def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Merge the attention output ``V`` and the logsumexp value ``S`` from the two
    KV-segments.
    Check :ref:`our tutorial <recursive-attention>` on the mathematical details.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment ``A``, shape:
        ``[seq_len, num_heads, head_dim]``.
    s_a : torch.Tensor
        The logsumexp value from the KV segment ``A``. expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``.
    v_b : torch.Tensor
        The attention output from the KV segment ``B``,
        shape: ``[seq_len, num_heads, head_dim]``.
    s_b : torch.Tensor
        The logsumexp value from the KV segment ``B``, expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``

    Returns
    -------
    V : torch.Tensor
        The merged attention output (equivalent to attention with merged KV-segment
        ``[A: B]``), shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segment ``[A: B]``, shape:
        ``[seq_len, num_heads]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_state(va, sa, vb, sb)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    device = v_a.device
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    v_merged = torch.empty_like(v_a)
    s_merged = torch.empty_like(s_a)
    get_module_attr("merge_state")(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


@register_fake_op("flashinfer::merge_state")
def _fake_merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.empty_like(v_a)
    s = torch.empty_like(s_a)
    return v, s


@register_custom_op("flashinfer::merge_state_in_place", mutates_args=("v", "s"))
def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    r"""Merge the self-attention state ``(v, s)`` with another state
    ``(v_other, s_other)`` in-place.

    Parameters
    ----------
    v : torch.Tensor
        The partial attention output to be updated in-place, shape:
        ``(seq_len, num_heads, head_dim)``.
    s : torch.Tensor
        The partial logsumexpr value to be updated in-place, expected to be a float32
        tensor, shape: ``(seq_len, num_heads)``.
    v_other : torch.Tensor
        The other attention output to be merged, shape:
        ``(seq_len, num_heads, head_dim)``.
    s_other : torch.Tensor
        The other logsumexp value to be merged, expected to be a float32 tensor,
        shape: ``(seq_len, num_heads)``.
    mask : Optional[torch.Tensor]
        The boolean mask tensor for whether to merge the state for a corresponding sequence
        or not. Useful for CUDA graphs. If not specified (default), will merge states for
        all sequences.
        shape: ``[seq_len]``

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> v = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_other = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s_other = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> flashinfer.merge_state_in_place(v, s, v_other, s_other)
    """
    s = s.to(torch.float32)
    s_other = s_other.to(torch.float32)
    get_module_attr("merge_state_in_place")(v, s, v_other, s_other, mask)


@register_fake_op("flashinfer::merge_state_in_place")
def _fake_merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    pass


@register_custom_op("flashinfer::merge_states", mutates_args=())
def merge_states(v: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Merge multiple attention states (v, s).

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments, shape:
        ``[seq_len, num_states, num_heads, head_dim]``.
    s : torch.Tensor
        The logsumexp value from the KV segments, shape:
        ``[seq_len, num_states, num_heads]``, expected
        to be a float32 tensor.

    Returns
    -------
    V : torch.Tensor
        The merged attention output, shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segments, shape:
        ``[seq_len, num_heads]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> num_states = 100
    >>> v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_states(v, s)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    device = v.device
    s = s.to(torch.float32)
    seq_len, _, num_heads, head_dim = v.size()
    v_merged = torch.empty(seq_len, num_heads, head_dim, dtype=v.dtype, device=device)
    s_merged = torch.empty(seq_len, num_heads, dtype=torch.float32, device=device)
    get_module_attr("merge_states")(v, s, v_merged, s_merged)
    return v_merged, s_merged


@register_fake_op("flashinfer::merge_states")
def _fake_merge_states(
    v: torch.Tensor, s: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len, _, num_heads, head_dim = v.size()
    v_merged = torch.empty(seq_len, num_heads, head_dim, dtype=v.dtype)
    s_merged = torch.empty(seq_len, num_heads, dtype=torch.float32)
    return v_merged, s_merged


class MultiLevelCascadeAttentionWrapper:
    r"""Attention wrapper for memory efficient multi-level cascade inference, this API assumes all
    levels KV-Cache are stored in a unified paged table.

    Please check :ref:`cascade-inference-data-layout` for data layout in cascade inference.
    Note that it's not always beneficial to increase the number of levels because of the overhead
    of merging attention results.

    The idea of cascade inference is introduced in our `blog post <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    ...     2, workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_kv_num_pages = 512
    >>> unique_kv_num_pages = 128
    >>> total_num_pages = shared_kv_num_pages + unique_kv_num_pages
    >>> shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to("cuda:0")
    >>> shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0")
    >>> unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda:0")
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> qo_indptr_arr = [
    ...     torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),  # top-level for shared KV-Cache
    ...     torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")    # bottom-level for unique KV-Cache
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.plan(
    ...     qo_indptr_arr,
    ...     [shared_kv_page_indptr, unique_kv_page_indptr],
    ...     [shared_kv_page_indices, unique_kv_page_indices],
    ...     [shared_kv_last_page_len, unique_kv_last_page_len],
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.run(q, kv_cache_at_layer[i])
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    See Also
    --------
    BatchPrefillWithPagedKVCacheWrapper
    """

    def __init__(
        self,
        num_levels,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indptr_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_indices_buf_arr: Optional[List[torch.Tensor]] = None,
        paged_kv_last_page_len_buf_arr: Optional[List[torch.Tensor]] = None,
    ) -> None:
        r"""Constructor of :class:`MultiLevelCascadeAttentionWrapper`.

        Parameters
        ----------
        num_levels : int
            The number of levels in the cascade attention.
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        use_cuda_graph : bool
            Whether to use CUDA graph to capture the kernels, if enabled, the auxiliary data structures
            will be stored in provided buffers.
        qo_indptr_buf_arr : Optional[List[torch.Tensor]]
            An array of qo indptr buffers for each level, the array length should be equal to
            the number of levels.
            The last element of each tensor should be the total number of queries/outputs.
        paged_kv_indptr_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache indptr buffers for each level, the array length should be
            equal to the number of levels.
        paged_kv_indices_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache indices buffers for each level, the array length should be
            equal to the number of levels.
        paged_kv_last_page_len_buf_arr : Optional[List[torch.Tensor]]
            An array of paged kv-cache last page length buffers for each level, the array length
            should be equal to the number of levels.
        """
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            self._batch_prefill_wrappers = [
                BatchPrefillWithPagedKVCacheWrapper(
                    float_workspace_buffer,
                    kv_layout,
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr_buf,
                    paged_kv_indptr_buf=paged_kv_indptr_buf,
                    paged_kv_indices_buf=paged_kv_indices_buf,
                    paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
                )
                for (
                    qo_indptr_buf,
                    paged_kv_indptr_buf,
                    paged_kv_indices_buf,
                    paged_kv_last_page_len_buf,
                ) in zip(
                    qo_indptr_buf_arr,
                    paged_kv_indptr_buf_arr,
                    paged_kv_indices_buf_arr,
                    paged_kv_last_page_len_buf_arr,
                )
            ]
        else:
            self._batch_prefill_wrappers = [
                BatchPrefillWithPagedKVCacheWrapper(float_workspace_buffer, kv_layout)
                for _ in range(num_levels)
            ]
        self._num_levels = num_levels
        self._kv_layout = kv_layout

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffers: List[torch.Tensor],
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffers : List[torch.Tensor]
            The array of new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        for wrapper, int_workspace_buffer in zip(
            self._batch_prefill_wrappers, int_workspace_buffers
        ):
            wrapper.reset_workspace_buffer(float_workspace_buffer, int_workspace_buffer)

    def plan(
        self,
        qo_indptr_arr: List[torch.Tensor],
        paged_kv_indptr_arr: List[torch.Tensor],
        paged_kv_indices_arr: List[torch.Tensor],
        paged_kv_last_page_len: List[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: str = "float16",
    ):
        r"""Create auxiliary data structures for multi-level cascade attention for multiple
        forward calls within the same decode step. Please check
        :ref:`cascade-inference-data-layout` for data layout in cascade inference.

        Parameters
        ----------
        qo_indptr_arr : List[torch.Tensor]
            An array of qo indptr tensors for each level, the array length should be equal to
            the number of levels.
            The last element of each tensor should be the total number of queries/outputs.
        paged_kv_indptr_arr : List[torch.Tensor]
            An array of paged kv-cache indptr tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_indices_arr : List[torch.Tensor]
            An array of paged kv-cache indices tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_last_page_len : List[torch.Tensor]
            An array of paged kv-cache last page length tensors for each level, the array length
            should be equal to the number of levels.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor. If None, will be set to torch.float16.
        """
        for i, (
            wrapper,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) in enumerate(
            zip(
                self._batch_prefill_wrappers,
                qo_indptr_arr,
                paged_kv_indptr_arr,
                paged_kv_indices_arr,
                paged_kv_last_page_len,
            )
        ):
            wrapper.plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=causal if i == self._num_levels - 1 else False,
                pos_encoding_mode=pos_encoding_mode,
                use_fp16_qk_reduction=use_fp16_qk_reduction,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                rope_scale=rope_scale,
                rope_theta=rope_theta,
                q_data_type=q_data_type,
            )

    begin_forward = plan

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
    ):
        r"""Compute multi-level cascade attention.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        """
        out, lse = self._batch_prefill_wrappers[-1].run(
            q,
            paged_kv_cache,
            return_lse=True,
        )
        for wrapper in self._batch_prefill_wrappers[:-1]:
            out_i, lse_i = wrapper.run(q, paged_kv_cache, return_lse=True)
            merge_state_in_place(out, lse, out_i, lse_i)

        return out

    forward = run


class BatchDecodeWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with shared-prefix paged kv-cache for batch
    of requests. The shared-prefix KV-Cache was stored in a standalone tensors, and the
    unique KV-Cache of each request was stored in a paged KV-Cache data structure.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Warning
    -------
    This API will be deprecated in the future, please use
    :class:`MultiLevelCascadeAttentionWrapper` instead.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> unique_kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> unique_kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.begin_forward(
    ...     unique_kv_page_indptr,
    ...     unique_kv_page_indices,
    ...     unique_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     unique_kv_cache = unique_kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.forward(q, k_shared, v_shared, unique_kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared prefix batch decode attention creates
    some auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        self._batch_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_decode_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

    def begin_forward(
        self,
        unique_kv_indptr: torch.Tensor,
        unique_kv_indices: torch.Tensor,
        unique_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        data_type: str = "float16",
    ) -> None:
        r"""Plan shared-prefix batch decode attention for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls,
        auxiliary data structures will be created during this call and cached for
        multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.


        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        self._batch_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=data_type,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute batch decode attention between queries and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared : torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[batch_size, num_heads, head_dim]``
        """
        V_shared, S_shared = single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            sm_scale=self._batch_decode_wrapper._sm_scale,
            rope_scale=self._batch_decode_wrapper._rope_scale,
            rope_theta=self._batch_decode_wrapper._rope_theta,
            return_lse=True,
        )
        V_unique, S_unique = self._batch_decode_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            pos_encoding_mode="NONE",
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect"""
        pass


class BatchPrefillWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with shared-prefix paged kv-cache for
    batch of requests.

    Check :ref:`our tutorial<kv-layout>` for paged kv-cache layout.

    Warning
    -------
    This API will be deprecated in the future, please use
    :class:`MultiLevelCascadeAttentionWrapper` instead.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len= torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.forward(
    ...         q, k_shared, v_shared, kv_cache, causal=True
    ...     )
    ...     outputs.append(o)
    ...
    s[0].shape>>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([100, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared-prefix batch prefill/append attention
    operators creates some auxiliary data structures, these data structures can be
    reused across multiple prefill/append attention calls (e.g. different Transformer
    layers). This wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithSharedPrefixPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        self._batch_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_prefill_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
    ) -> None:
        r"""Create auxiliary data structures for shared-prefix batch prefill/append
        attention for multiple forward calls within the same prefill/append step.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward`
        or :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        self._batch_prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
        causal: bool = False,
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Compute batch prefill/append attention between query and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared ; torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        causal : bool
            Whether to apply causal mask on the attention matrix.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_heads, head_dim]``.

        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        V_shared, S_shared = single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            return_lse=True,
        )
        V_unique, S_unique = self._batch_prefill_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            causal=causal,
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect"""
        pass
````

## File: comm.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

import ctypes
import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import register_custom_op

# NOTE(Zihao): we should use cuda-python instead of ctypes cuda runtime bindings.
# However, cuda-python's API is not stable yet, so we use ctypes bindings instead.
# which is copied from vllm codebase.


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(
        lib_name
    ), f"Unexpected filename: {filename} for library {lib_name}"
    return path


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t   cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char*   cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t    cudaMalloc ( void** devPtr, size_t size )
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​cudaError_t    cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            assert so_file is not None, "libcudart is not loaded in the current process"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr


cudart = CudaRTLibrary()


def gen_comm_module() -> JitSpec:
    return gen_jit_spec(
        "comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR / "custom_all_reduce.cu",
        ],
    )


@functools.cache
def get_comm_module():
    module = gen_comm_module().build_and_load()

    # torch library for all
    @register_custom_op(
        "flashinfer::init_custom_ar",
        mutates_args=["ipc_ptrs", "rank_data", "rank", "full_nvlink"],
    )
    def init_custom_ar(
        ipc_ptrs: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        return module.init_custom_ar(ipc_ptrs, rank_data, rank, full_nvlink)

    @register_custom_op("flashinfer::dispose", mutates_args=["fa"])
    def dispose(fa: int) -> None:
        module.dispose(fa)

    @register_custom_op("flashinfer::get_graph_buffer_ipc_meta", mutates_args=["fa"])
    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return module.get_graph_buffer_ipc_meta(fa)

    @register_custom_op(
        "flashinfer::register_buffer", mutates_args=["fa", "fake_ipc_ptrs"]
    )
    def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
        return module.register_buffer(fa, fake_ipc_ptrs)

    @register_custom_op(
        "flashinfer::register_graph_buffers",
        mutates_args=["fa", "handles", "offsets"],
    )
    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        module.register_graph_buffers(fa, handles, offsets)

    @register_custom_op("flashinfer::meta_size", mutates_args=[])
    def meta_size() -> int:
        return module.meta_size()

    @register_custom_op(
        "flashinfer::all_reduce",
        mutates_args=["out", "reg_buffer", "reg_buffer_sz_bytes"],
    )
    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
        num_ctas: int,
    ) -> None:
        module.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas)

    return SimpleNamespace(
        init_custom_ar=init_custom_ar,
        dispose=dispose,
        get_graph_buffer_ipc_meta=get_graph_buffer_ipc_meta,
        register_buffer=register_buffer,
        register_graph_buffers=register_graph_buffers,
        meta_size=meta_size,
        all_reduce=all_reduce,
    )


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    return get_comm_module().init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)


def dispose(fa: int) -> None:
    get_comm_module().dispose(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
    num_ctas: int,
) -> None:
    """Performs an out-of-place all reduce.

    Args:
        fa: The handle to the custom all reduce.
        inp: The input tensor to all reduce.
        out: The output tensor to all reduce.
        reg_buffer: The register buffer to all reduce.
        reg_buffer_sz_bytes: The size of the register buffer.
        num_ctas: The number of CTAs to use for the all reduce.
        CTA upper bounds: 36. Generally, we can saturate the bandwidth even with small amount the SMs.
    """
    get_comm_module().all_reduce(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas
    )


def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
    return get_comm_module().get_graph_buffer_ipc_meta(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    return get_comm_module().register_buffer(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    get_comm_module().register_graph_buffers(fa, handles, offsets)


def meta_size() -> int:
    return get_comm_module().meta_size()


def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    pointer = cudart.cudaMalloc(size_in_bytes)
    handle = cudart.cudaIpcGetMemHandle(pointer)
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    input_tensor = torch.tensor(bytearray(handle_bytes), dtype=torch.uint8).to(
        f"cuda:{rank}"
    )
    gathered_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input_tensor, group=group)

    handles = []
    handle_type = type(handle)
    for tensor in gathered_tensors:
        bytes_data = tensor.cpu().numpy().tobytes()
        handle_obj = handle_type()
        ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
        handles.append(handle_obj)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            try:
                opened_ptr = cudart.cudaIpcOpenMemHandle(h)
                pointers.append(opened_ptr.value)
            except Exception as e:
                print(f"Rank {rank}: Failed to open IPC handle from rank {i}: {e}")
                raise

    dist.barrier(group=group)
    return pointers


def free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    if pointers and len(pointers) > rank and pointers[rank] is not None:
        cudart.cudaFree(ctypes.c_void_p(pointers[rank]))
    dist.barrier(group=group)
````

## File: decode.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

import functools
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import (
    gen_batch_decode_mla_module,
    gen_batch_decode_module,
    gen_customize_batch_decode_module,
    gen_customize_batch_prefill_module,
    gen_single_decode_module,
    get_batch_decode_uri,
    get_single_decode_uri,
)
from .page import get_seq_lens
from .prefill import (
    get_batch_prefill_jit_module,
    get_batch_prefill_module,
    get_single_prefill_module,
)
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    register_custom_op,
    register_fake_op,
)

_single_decode_modules = {}
_batch_decode_modules = {}
_batch_decode_mla_modules = {}
_batch_decode_jit_modules = {}


def get_single_decode_module(*args):
    global _single_decode_modules
    if args not in _single_decode_modules:
        uri = get_single_decode_uri(*args)
        module = gen_single_decode_module(*args).build_and_load()
        run_func = module.run.default

        # torch library for single_decode_with_kv_cache

        @register_custom_op(f"flashinfer::{uri}_run", mutates_args=("tmp", "o"))
        def run_single_decode(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            tmp: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            alibi_slopes: Optional[torch.Tensor],
            kv_layout_code: int,
            window_left: int,
            logits_soft_cap: float,
            sm_scale: float,
            rope_scale: float,
            rope_theta: float,
        ) -> None:
            run_func(
                q,
                k,
                v,
                tmp,
                o,
                maybe_lse,
                kv_layout_code,
                window_left,
                alibi_slopes,
                logits_soft_cap,
                sm_scale,
                1.0 / rope_scale,  # rope_rcp_scale
                1.0 / rope_theta,  # rope_rcp_theta
            )

        @register_fake_op(f"flashinfer::{uri}_run")
        def _fake_run_single_decode(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            tmp: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            alibi_slopes: Optional[torch.Tensor],
            kv_layout_code: int,
            window_left: int,
            logits_soft_cap: float,
            sm_scale: float,
            rope_scale: float,
            rope_theta: float,
        ) -> None:
            pass

        # Register the module.
        _single_decode_modules[args] = SimpleNamespace(run=run_single_decode)
    return _single_decode_modules[args]


def get_batch_decode_jit_module(module_name: str, jit_module: Any):
    global _batch_decode_jit_modules
    if module_name in _batch_decode_jit_modules:
        return _batch_decode_jit_modules[module_name]

    plan_func = jit_module.plan.default
    run_func = jit_module.run.default

    @register_custom_op(
        f"flashinfer::{module_name}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        *args,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    _batch_decode_jit_modules[module_name] = SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )
    return _batch_decode_jit_modules[module_name]


def get_batch_decode_module(*args):
    global _batch_decode_modules
    if args not in _batch_decode_modules:
        uri = get_batch_decode_uri(*args)
        mod = gen_batch_decode_module(*args).build_and_load()
        plan_func = mod.plan.default
        run_func = mod.run.default

        # torch library for batch_decode_with_paged_kv_cache_run

        @register_custom_op(
            f"flashinfer::{uri}_run",
            mutates_args=(
                "float_workspace_buffer",
                "int_workspace_buffer",
                "paged_k_cache",
                "paged_v_cache",
                "o",
                "maybe_lse",
            ),
        )
        def run_batch_decode(
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            plan_info_vec: List[int],
            q: torch.Tensor,
            paged_k_cache: Optional[torch.Tensor],
            paged_v_cache: Optional[torch.Tensor],
            paged_kv_indptr: torch.Tensor,
            paged_kv_indices: torch.Tensor,
            paged_kv_last_page_len: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            kv_layout_code: int,
            window_left: int,
            alibi_slopes: Optional[torch.Tensor],
            logits_soft_cap: float,
            sm_scale: float,
            rope_scale: float,
            rope_theta: float,
        ) -> None:
            run_func(
                float_workspace_buffer,
                int_workspace_buffer,
                plan_info_vec,
                q,
                paged_k_cache,
                paged_v_cache,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                o,
                maybe_lse,
                kv_layout_code,
                window_left,
                alibi_slopes,
                logits_soft_cap,
                sm_scale,
                1.0 / rope_scale,  # rope_rcp_scale
                1.0 / rope_theta,  # rope_rcp_theta
            )

        @register_fake_op(f"flashinfer::{uri}_run")
        def _fake_run_batch_decode(
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            plan_info_vec: List[int],
            q: torch.Tensor,
            paged_k_cache: Optional[torch.Tensor],
            paged_v_cache: Optional[torch.Tensor],
            paged_kv_indptr: torch.Tensor,
            paged_kv_indices: torch.Tensor,
            paged_kv_last_page_len: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            kv_layout_code: int,
            window_left: int,
            alibi_slopes: Optional[torch.Tensor],
            logits_soft_cap: float,
            sm_scale: float,
            rope_scale: float,
            rope_theta: float,
        ) -> None:
            pass

        # Register the module.
        #
        # Note that plan is not part of model logic. It should not be included in
        # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
        _batch_decode_modules[args] = SimpleNamespace(
            plan=plan_func,
            run=run_batch_decode,
        )
    return _batch_decode_modules[args]


def single_decode_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    window_left: int = -1,
    return_lse: bool = False,
):
    device = q.device
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, device)
    o = torch.empty_like(q)
    if return_lse:
        lse = torch.empty((q.size(0)), dtype=torch.float32, device=device)
    else:
        lse = None
    jit_module.run.default(
        q,
        k,
        v,
        tmp,
        o,
        lse,
        TensorLayout[kv_layout].value,
        window_left,
        *args,
    )
    return o


def get_batch_decode_mla_module(*args):
    global _batch_decode_mla_modules
    if args not in _batch_decode_mla_modules:
        _batch_decode_mla_modules[args] = gen_batch_decode_mla_module(
            *args
        ).build_and_load()
    return _batch_decode_mla_modules[args]


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode attention with KV Cache for single request, return attention output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The value tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if
        :attr:`kv_layout` is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    use_tensor_cores: bool
        Whether to use tensor cores for the computation. Will be faster for large group
        size in grouped query attention. Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    window_left : int
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    logits_soft_cap : Optional[float]
        The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
        provided, will be set to ``0``. If greater than 0, the logits will be capped according to
        formula:
        :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
        where :math:`x` is the input logits.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_decode_with_kv_cache(q, k, v)
    >>> o.shape
    torch.Size([32, 128])

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    num_qo_heads = q.shape[0]

    lse = None
    if return_lse:
        lse = torch.empty((num_qo_heads,), dtype=torch.float32, device=q.device)

    if use_tensor_cores:
        out = torch.empty_like(q.unsqueeze(0))
        get_single_prefill_module("fa2")(
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            False,  # use_fp16_qk_reduction
        ).run(
            q.unsqueeze(0),
            k,
            v,
            tmp,
            out,
            lse.unsqueeze(0) if lse is not None else None,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[kv_layout].value,
            window_left,
            None,  # packed_custom_mask
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            logits_soft_cap,
            sm_scale,
            None,  # scale_q, not supported yet
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
        )
        out = out.squeeze(0)
        if return_lse:
            lse = lse.squeeze(0)
    else:
        out = torch.empty_like(q)
        get_single_decode_module(
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
        ).run(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            TensorLayout[kv_layout].value,
            window_left,
            logits_soft_cap,
            sm_scale,
            rope_scale,
            rope_theta,
        )

    if v_scale is not None:
        out *= v_scale
    if return_lse:
        return out, lse
    else:
        return out


class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with paged kv-cache (first proposed in
    `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.plan(
    ...     kv_page_indptr,
    ...     kv_page_indices,
    ...     kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     pos_encoding_mode="NONE",
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = decode_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's batch decode attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0],
                    gen_customize_batch_prefill_module(
                        "fa2", *jit_args
                    ).build_and_load(),
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0],
                    gen_customize_batch_decode_module(*jit_args).build_and_load(),
                )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = use_cuda_graph

        if use_tensor_cores:
            if use_cuda_graph:
                # NOTE(Zihao): if once created, no need to update it in plan/run
                self._qo_indptr_buf = torch.arange(
                    self._fixed_batch_size + 1,
                    dtype=torch.int32,
                    device=float_workspace_buffer.device,
                )

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        if self.use_tensor_cores:
            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_prefill_module("fa2")(
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                indptr_host,
                kv_lens_arr_host,
                batch_size,  # total_num_rows
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim,
                head_dim,
                False,  # causal
            )
        else:
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_decode_module(
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                indptr_host,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                window_left,
                logits_soft_cap,
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: this function is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q, paged_kv_cache, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
        )

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        *args
            Additional arguments for the custom kernel.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )

        pos_encoding_mode = self._pos_encoding_mode
        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q)
        else:
            _check_shape_dtype_device(out, q.shape, q.dtype, q.device, "out")

        if self.use_tensor_cores:
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                MaskMode.NON_CAUSAL.value,
                TensorLayout[self._kv_layout].value,
                window_left,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    None,  # packed_custom_mask
                    None,  # mask_indptr_buf
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    None,  # maybe_prefix_len_ptr
                    None,  # maybe_token_pos_in_items_ptr
                    None,  # maybe_max_item_len_ptr
                    logits_soft_cap,
                    sm_scale,
                    None,  # scale_q, not supported yet
                    None,  # scale_k
                    None,  # scale_v
                    rope_scale,
                    rope_theta,
                    0,  # token_pos_in_items_len
                ]

            self._cached_module.paged_run(*run_args)
        else:
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                window_left,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    logits_soft_cap,
                    sm_scale,
                    rope_scale,
                    rope_theta,
                ]

            self._cached_module.run(*run_args)
        if v_scale is not None:
            out *= v_scale

        return (out, lse) if return_lse else out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: this function is deprecated, please use :meth:`run_return_lse` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q,
            paged_kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    r"""CUDAGraph-compatible Wrapper class for decode attention with paged kv-cache (first
    proposed in `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Note that this wrapper may not be as efficient as :class:`BatchDecodeWithPagedKVCacheWrapper`
    because we won't dispatch to different kernels for different batch sizes/sequence lengths/etc
    to accommodate the CUDAGraph requirement.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Note
    ----
    The :meth:`plan` method could not be captured by CUDAGraph.

    See Also
    --------
    :class:`BatchDecodeWithPagedKVCacheWrapper`
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_tensor_cores: bool = False,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer on GPU used to store auxiliary data structures,
            recommended size is 128MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        indptr_buffer : torch.Tensor
            The user reserved buffer on GPU to store the indptr of the paged kv cache, should
            be large enough to store the indptr of maximum batch size (``[max_batch_size + 1]``)
            during the lifecycle of this wrapper.

        indices_buffer : torch.Tensor
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.

        last_page_len_buffer : torch.Tensor
            The user reserved buffer on GPU to store the number of entries in the last page,
            should be large enough to store the maximum batch size (``[max_batch_size]``)
            during the lifecycle of this wrapper.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        super().__init__(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=indptr_buffer,
            paged_kv_indices_buffer=indices_buffer,
            paged_kv_last_page_len_buffer=last_page_len_buffer,
        )


class BatchDecodeMlaWithPagedKVCacheWrapper:
    r"""Warning: this class is deprecated and will be removed in a future release.
    Please use :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper` instead, which provides
    a more efficient and general MLA implementation that supports decode and incremental prefill.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._use_tensor_cores = use_tensor_cores
        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_cuda_graph = use_cuda_graph

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        head_dim_compressed_kv: int,
        page_size: int,
        sm_scale: float,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        data_type: Union[str, torch.dtype] = "float16",
        q_data_type: Optional[Union[str, torch.dtype]] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        head_dim_compressed_kv : int
            The dimension of the compressed kv, is also kv_lora_rank
        page_size : int
            The page size of the paged kv cache
        sm_scale : float
            The scale of softmax, should be ``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)``
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache. Defaults to ``float16``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor. If None, will be set to
            ``data_type``. Defaults to ``None``.

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.
        """
        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr)
            self._paged_kv_indices_buf[: len(indices)] = indices
            self._paged_kv_last_page_len_buf.copy_(last_page_len)
        else:
            self._paged_kv_indptr_buf = indptr.to(self.device)
            self._paged_kv_indices_buf = indices.to(self.device)
            self._paged_kv_last_page_len_buf = last_page_len.to(self.device)

        data_type = canonicalize_torch_dtype(data_type)
        if not q_data_type:
            q_data_type = data_type
        q_data_type = canonicalize_torch_dtype(q_data_type)

        indptr_host = indptr.to("cpu")

        self._cached_module = get_batch_decode_mla_module(
            q_data_type,
            data_type,
            q_data_type,
            indptr.dtype,
            head_dim_compressed_kv,
            num_qo_heads,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            self._use_tensor_cores,
        )
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            indptr_host,
            batch_size,
            num_qo_heads,
            page_size,
            self.is_cuda_graph_enabled,
        )

        self._sm_scale = sm_scale
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        paged_ckv_cache: torch.Tensor,
        paged_kpe_cache: torch.Tensor,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor not related to ROPE, shape: ``[batch_size, num_qo_heads, head_dim_ckv]``
        q_pe : torch.Tensor
            The query tensor related to ROPE, shape: ``[batch_size, num_qo_heads, head_dim_kpe]``
        paged_ckv_cache : torch.Tensor
            The paged compressed-KV-Cache stored as a single tensor:
            * 3-D tensors, each with shape: ``[max_num_pages, page_size, head_dim_ckv]``.
        paged_kpe_cache : torch.Tensor
            The paged k-pe-Cache stored as a single tensor:
            * 3-D tensors, each with shape: ``[max_num_pages, page_size, head_dim_kpe]``.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """
        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        device = self.device
        if out is None:
            out = torch.empty_like(q_nope, device=device)
        else:
            _check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q_nope.size(0), q_nope.size(1)),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                _check_shape_dtype_device(
                    lse,
                    (q_nope.size(0), q_nope.size(1)),
                    q_nope.dtype,
                    q_nope.device,
                    "lse",
                )
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            paged_ckv_cache,
            paged_kpe_cache,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            out,
            sm_scale,
            window_left,
            logits_soft_cap,
            rope_scale,
            rope_theta,
            lse,
        )
        out = [out, lse] if return_lse else [out]
        if v_scale is not None:
            out[0] *= v_scale

        return tuple(out) if return_lse else out[0]

    run_return_lse = functools.partialmethod(run, return_lse=True)
````

## File: gemm.py
````python
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

import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm90a_nvcc_flags, sm100a_nvcc_flags
from .utils import (
    _get_cache_buf,
    determine_gemm_backend,
    get_indptr,
    is_float8,
    register_custom_op,
    register_fake_op,
)

_gemm_module = None
_gemm_module_sm90 = None


def gen_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_ops.cu",
        ],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )


def get_gemm_module():
    global _gemm_module
    if _gemm_module is None:
        module = gen_gemm_module().build_and_load()

        # torch library for bmm_fp8

        @register_custom_op(
            "flashinfer::bmm_fp8", mutates_args=("workspace_buffer", "D")
        )
        def bmm_fp8(
            workspace_buffer: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            D: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
        ) -> None:
            cublas_handle = torch.cuda.current_blas_handle()
            module.bmm_fp8.default(
                A,
                B,
                D,
                A_scale,
                B_scale,
                workspace_buffer,
                cublas_handle,
            )

        @register_fake_op("flashinfer::bmm_fp8")
        def _fake_bmm_fp8(
            workspace_buffer: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            D: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
        ) -> None:
            pass

        # torch library for cutlass_segment_gemm

        @register_custom_op("flashinfer::cutlass_segment_gemm", mutates_args=("y"))
        def cutlass_segment_gemm(
            workspace_buffer: torch.Tensor,
            all_problems: torch.Tensor,
            x_data: torch.Tensor,
            w_data: torch.Tensor,
            y_data: torch.Tensor,
            x_ld: torch.Tensor,
            w_ld: torch.Tensor,
            y_ld: torch.Tensor,
            y: torch.Tensor,
            empty_x_data: torch.Tensor,
            weight_column_major: bool,
        ) -> None:
            module.cutlass_segment_gemm.default(
                workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld,
                w_ld,
                y_ld,
                empty_x_data,
                weight_column_major,
            )

        @register_fake_op("flashinfer::cutlass_segment_gemm")
        def _fake_cutlass_segment_gemm(
            workspace_buffer: torch.Tensor,
            all_problems: torch.Tensor,
            x_data: torch.Tensor,
            w_data: torch.Tensor,
            y_data: torch.Tensor,
            x_ld: torch.Tensor,
            w_ld: torch.Tensor,
            y_ld: torch.Tensor,
            y: torch.Tensor,
            empty_x_data: torch.Tensor,
            weight_column_major: bool,
        ) -> None:
            pass

        # Register the module
        _gemm_module = SimpleNamespace(
            bmm_fp8=bmm_fp8,
            cutlass_segment_gemm=cutlass_segment_gemm,
        )

    return _gemm_module


def gen_gemm_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "gemm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "gemm_groupwise_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_groupwise_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "gemm_sm100_pybind.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm100_pybind.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


@functools.cache
def get_gemm_sm100_module():
    module = gen_gemm_sm100_module().build_and_load()

    return module


def gen_gemm_sm90_module() -> JitSpec:
    return gen_jit_spec(
        "gemm_sm90",
        [
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_sm90_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_f16_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_bf16_bf16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e4m3_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e5m2_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e4m3_bf16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e5m2_bf16_sm90.cu",
        ],
        extra_cuda_cflags=sm90a_nvcc_flags,
    )


def get_gemm_sm90_module():
    global _gemm_module_sm90
    if _gemm_module_sm90 is None:
        module = gen_gemm_sm90_module().build_and_load()

        # torch library for cutlass_segment_gemm_sm90

        @register_custom_op(
            "flashinfer::cutlass_segment_gemm_sm90",
            mutates_args=("workspace_buffer", "y"),
        )
        def cutlass_segment_gemm_sm90(
            workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            all_problems: torch.Tensor,
            x_data: torch.Tensor,
            w_data: torch.Tensor,
            y_data: torch.Tensor,
            x_stride: torch.Tensor,
            w_stride: torch.Tensor,
            y_stride: torch.Tensor,
            y: torch.Tensor,
            empty_x_data: torch.Tensor,
            empty_y_data: torch.Tensor,
            weight_column_major: bool,
        ) -> None:
            module.cutlass_segment_gemm_sm90.default(
                workspace_buffer,
                int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride,
                w_stride,
                y_stride,
                empty_x_data,
                empty_y_data,
                weight_column_major,
            )

        @register_fake_op("flashinfer::cutlass_segment_gemm_sm90")
        def _fake_cutlass_segment_gemm_sm90(
            workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            all_problems: torch.Tensor,
            x_data: torch.Tensor,
            w_data: torch.Tensor,
            y_data: torch.Tensor,
            x_stride: torch.Tensor,
            w_stride: torch.Tensor,
            y_stride: torch.Tensor,
            y: torch.Tensor,
            empty_x_data: torch.Tensor,
            empty_y_data: torch.Tensor,
            weight_column_major: bool,
        ) -> None:
            pass

        # Register the module
        _gemm_module_sm90 = SimpleNamespace(
            cutlass_segment_gemm_sm90=cutlass_segment_gemm_sm90,
        )

    return _gemm_module_sm90


def launch_compute_sm80_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    ld_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)

    from .triton.gemm import compute_sm80_group_gemm_args

    compute_sm80_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


def launch_compute_sm90_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    stride_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)

    from .triton.gemm import compute_sm90_group_gemm_args

    compute_sm90_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


class SegmentGEMMWrapper:
    r"""Wrapper for segment GEMM kernels.

    Example
    -------
    >>> import torch
    >>> from flashinfer import SegmentGEMMWrapper
    >>> # create a 1MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    >>> segment_gemm = SegmentGEMMWrapper(workspace_buffer)
    >>> seq_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    >>> # create packed input tensor (10 = 1 + 2 + 3 + 4)
    >>> x = torch.randn(10, 128, device="cuda", dtype=torch.float16)
    >>> # create weight tensor with 4 weights, each with 128 input and 256 output channels, column major
    >>> weights = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
    >>> # compute the segment GEMM
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[2].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[3].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    >>>
    >>> # another example with weight indices
    >>> weight_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens, weight_indices=weight_indices)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[0].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[1].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, backend: str = "auto"
    ) -> None:
        r"""Initialize the wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The workspace buffer for the kernels, we use it for storing intermediate results in cutlass
            segment GEMM kernels. Encouraged size is 128MB.
        """
        self._int_workspace_buffer = torch.empty(
            (1024 * 1024,), dtype=torch.int8, device=float_workspace_buffer.device
        )
        self._float_workspace_buffer = float_workspace_buffer
        self.backend = backend

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer for the kernels.
        int_workspace_buffer : torch.Tensor
            The new int workspace buffer for the kernels.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer

    def run(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        out: Optional[torch.Tensor] = None,
        seg_lens: Optional[torch.Tensor] = None,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the segment GEMM kernel.

        Compute the matrix multiplication between a batch of input tensor (with variable number of rows, but fixed
        number of columns) and a batch of weight tensor with fixed number of rows and columns:

        .. math::

            y[i] = x[i] \times W[i]

        if :attr:`weight_indices` is provided, we will select the weight tensor based on the indices in the
        :attr:`weight_indices` tensor:

        .. math::

            y[i] = x[i] \times W[\text{weight_indices}[i]]

        We use Ragged Tensor to represent the input tensor :attr:`x` and the output tensor :attr:`y`, and each x[i]
        is a segment of the concatenated tensor. Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details.
        We use a ``seg_len`` or ``seg_indptr`` tensor (either would work) to indicate the start and end of each segment,
        where the ``seg_indptr`` is the cumulative sum of the ``seg_lens`` tensor (with an additional 0 at the beginning):

        .. math::

            \text{seg_indptr}[i] = \sum_{j=0}^{i-1} \text{seg_lens}[j], \quad \text{seg_indptr}[0] = 0

        - If ``seg_lens`` is provided, then :attr:`x` has shape ``(sum(seg_lens), d_in)`` and :attr:`y` has shape
            ``(sum(seg_lens), d_out)``, where ``d_in`` is the number of columns of the input tensor and ``d_out`` is the
            number of columns of the output tensor.
        - If ``seg_indptr`` is provided, then :attr:`x` has shape ``(seg_indptr[-1], d_in)`` and :attr:`y` has shape
            ``(seg_indptr[-1], d_out)``.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape ``(sum(seg_lens), d_in)``.
        weights : torch.Tensor
            The 3D weight tensor with shape ``(num_weights, d_in, d_out)`` if :attr:`weight_column_major` is ``False``,
            or ``(num_weights, d_out, d_in)`` if :attr:`weight_column_major` is ``True``.
        batch_size : int
            The number of segments.
        weight_column_major : bool
            Whether the weight tensor is column major.
        out : Optional[torch.Tensor]
            The output tensor, with shape ``(sum(seg_lens), d_out)``.
            If not provided, a new tensor will be created internally.
        seg_lens : Optional[torch.Tensor]
            The length of each segment, with shape ``(batch_size,)``, expects a 1D tensor of dtype ``torch.int64``.
        seg_indptr : Optional[torch.Tensor]
            The indptr of the segments, with shape ``(batch_size + 1,)``, expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then :attr:`seg_lens` will be ignored, otherwise ``seg_indptr`` will be computed
            internally from :attr:`seg_lens`.
        weight_indices : Optional[torch.Tensor]
            The indices of the weight tensor to be selected for each segment, with shape ``(batch_size,)``.
            Expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then the weight tensor will be selected based on the indices in this tensor.

        Returns
        -------
        torch.Tensor
            The output tensor with shape ``(sum(seg_lens), d_out)``.
        """
        if seg_lens is None and seg_indptr is None:
            raise ValueError("Either seg_lens or seg_indptr should be provided.")
        if seg_indptr is None:
            seg_indptr = get_indptr(seg_lens.to(x))
        if weight_indices is None:
            # create an empty CPU tensor as placeholder
            weight_indices = torch.empty(0, dtype=torch.int64)
        cumulative_batch_size = x.size(0)
        d_out = weights.size(1) if weight_column_major else weights.size(2)
        if out is None:
            if is_float8(x):
                out_dtype = torch.bfloat16
            else:
                out_dtype = x.dtype
            out = torch.zeros(
                (cumulative_batch_size, d_out), dtype=out_dtype, device=x.device
            )
        else:
            if out.shape != (cumulative_batch_size, d_out):
                raise ValueError(
                    f"Output tensor shape mismatch, expected {cumulative_batch_size, d_out}, got {out.shape}"
                )
        empty_x_data = torch.empty(0, dtype=x.dtype, device=x.device)
        empty_y_data = torch.empty(0, dtype=out.dtype, device=out.device)

        if self.backend == "auto":
            backend = determine_gemm_backend(x.device)
        else:
            backend = self.backend

        if backend == "sm90":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
            ) = launch_compute_sm90_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_sm90_module().cutlass_segment_gemm_sm90(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
                out,  # for torch compile mutates_args
                empty_x_data,  # for kernel type dispatch
                empty_y_data,
                weight_column_major,
            )
        elif backend == "sm80":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
            ) = launch_compute_sm80_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_module().cutlass_segment_gemm(
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
                out,
                empty_x_data,
                weight_column_major,
            )
        else:
            raise ValueError(f"Unsupported gemm backend: {backend}")
        return out

    forward = run


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""BMM FP8

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 or fp8 e5m2.

    B: torch.Tensor
        Mat2 tensor, shape (b, k, n), should be column major, fp8 e4m3 or fp8 e5m2.

    A_scale: torch.Tensor
        Scale tensor for A, float.

    B_scale: torch.Tensor
        Scale tensor for B, float.

    dtype: torch.dtype
        out dtype, bf16 or fp16.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16 or fp16, defaults to ``None``.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (b, m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import flashinfer
    >>> def to_float8(x, dtype=torch.float8_e4m3fn):
    ...     finfo = torch.finfo(dtype)
    ...     min_val, max_val = x.aminmax()
    ...     amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    ...     scale = finfo.max / amax
    ...     x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    ...     return x_scl_sat.to(dtype), scale.float().reciprocal()
    >>>
    >>> input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    >>> # column major weight
    >>> weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> weight_fp8, weight_inv_s = to_float8(weight, dtype=torch.float8_e4m3fn)
    >>> out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, torch.bfloat16)
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.bfloat16
    """
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    get_gemm_module().bmm_fp8(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using groupwise scaling.

    This function implements a GEMM operation that allows for fine-grained control over
    scale granularity across different dimensions. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape (m, k), fp8 e4m3 or fp8 e5m2.

    b: torch.Tensor
        Column-major input tensor shape (n, k), fp8 e4m3 or fp8 e5m2.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape (ceil_div(k, k_granularity), ceil_div(m, m_granularity)).

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape (ceil_div(k, k_granularity), ceil_div(n, n_granularity)).

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    out: Optional[torch.Tensor]
        Output tensor, shape (m, n). If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        If out is not specified, we will create an output tensor with this dtype.
        Defaults to ``torch.bfloat16``.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape (m, n).

    Notes
    -----
    If ``m`` is not a multiple of 4, we will pad ``m`` to the next multiple of 4 to accommodate the kernel's requirement.
    """
    workspace_buffer = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", 32 * 1024 * 1024, a.device
    )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Shape mismatch. a.shape = {a.shape}, b.shape = {b.shape}")
    m = a.shape[0]
    m_padded = ((m + 3) // 4) * 4
    need_padding = m_padded != m
    if need_padding:
        a_padded = F.pad(a, (0, 0, 0, m_padded - m))
        # a_scale is (k // 128, m)
        a_scale_padded = F.pad(a_scale, (0, m_padded - m))
    else:
        a_padded = a
        a_scale_padded = a_scale

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Shape mismatch. a.shape[1] = {a.shape[1]}, b.shape[1] = {b.shape[1]}"
        )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
    else:
        out_dtype = out.dtype

    # NOTE(Zihao): (out_specified, need_padding)
    # (False, False) -> create out_padded tensor explicitly
    # (False, True) -> create out_padded tensor explicitly
    # (True, False) -> use out tensor as out_padded
    # (True, True) -> create out_padded tensor explicitly

    if need_padding or out is None:
        out_padded = torch.empty(
            m_padded,
            b.shape[0],
            device=a.device,
            dtype=out_dtype,
        )
    else:
        out_padded = out

    get_gemm_sm100_module().gemm_fp8_nt_groupwise.default(
        workspace_buffer,
        a_padded,
        b,
        a_scale_padded,
        b_scale,
        out_padded,
        *scale_granularity_mnk,
    )

    # NOTE(Zihao): if out is specified and need_padding is True, we need to copy
    # the result back to out

    if out is not None and need_padding:
        out.copy_(out_padded[:m])
    else:
        out = out_padded[:m]

    return out


def gemm_fp8_nt_blockscaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using block-scaled scaling.

    Block-scaled scaling is a special case of groupwise scaling where the scale granularity
    is (128, 128, 128).
    """
    return gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_granularity_mnk=(128, 128, 128),
        out=out,
        out_dtype=out_dtype,
    )


def group_gemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (k // block_size, cum_m)
    b_scale: torch.Tensor,  # (batch_size, k // block_size, n // block_size)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with FP8 data types using groupwise scaling. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor shape ``(batch_size, n, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(k // block_size, cum_m)``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, k // block_size, n // block_size)``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``.
        Element element in ``m_indptr`` must be a multiple of 4.

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.
    """
    int_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_int_workspace", 32 * 1024 * 1024, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_float_workspace", 32 * 1024 * 1024, a.device
    )

    batch_size = m_indptr.shape[0] - 1
    assert b.shape[0] == batch_size
    assert b_scale.shape[0] == batch_size
    n = b.shape[1]
    k = b.shape[2]

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], n, dtype=out_dtype, device=a.device)

    get_gemm_sm100_module().group_gemm_fp8_nt_groupwise.default(
        int_workspace_buffer,
        float_workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        m_indptr,
        n,
        k,
        *scale_granularity_mnk,
        mma_sm,
    )
    return out


def pad_indptr_to_multiple_of_4(
    m_indptr: torch.Tensor,
):
    from .triton.gemm import compute_padding_mapping

    batch_size = m_indptr.shape[0] - 1
    m = m_indptr[1:] - m_indptr[:-1]
    m = m + 3 - (m + 3) % 4
    padded_m_indptr = torch.cat((torch.zeros((1,), device=m.device, dtype=m.dtype), m))
    padded_m_indptr = padded_m_indptr.cumsum(dim=0, dtype=padded_m_indptr.dtype)

    m_rank = torch.zeros((m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device)
    padded_m_rank = torch.zeros(
        (m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device
    )

    compute_padding_mapping[(batch_size,)](
        m_indptr, padded_m_indptr, m_rank, padded_m_rank
    )

    return padded_m_indptr, padded_m_rank
````

## File: mla.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

from typing import Literal, Optional, Tuple, Union, overload

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_batch_mla_module, gen_jit_spec, sm100a_nvcc_flags
from .utils import (
    MaskMode,
    _check_shape_dtype_device,
    determine_mla_backend,
)


def _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table):
    if q_nope_pe.ndim != 3:
        raise ValueError(f"Expected q_nope_pe.ndim == 3, got {q_nope_pe.ndim}")
    if ckv_kpe_cache.ndim != 3:
        raise ValueError(f"Expected ckv_kpe_cache.ndim == 3, got {ckv_kpe_cache.ndim}")
    if kv_len.ndim != 1:
        raise ValueError(f"Expected kv_len.ndim == 1, got {kv_len.ndim}")
    if page_table.ndim != 2:
        raise ValueError(f"Expected page_table.ndim == 2, got {page_table.ndim}")
    B_q, H, D_q = q_nope_pe.shape
    D_ckv = ckv_kpe_cache.shape[2]
    if H != 128:
        raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for q_nope_pe and ckv_kpe_cache, got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = page_table.shape
    block_size = ckv_kpe_cache.shape[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, got {B_q} and {B_block_table}"
        )
    if block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
        )


_mla_module = None


def gen_mla_module() -> JitSpec:
    return gen_jit_spec(
        "mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_mla.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mla_ops.cu",
        ],
        extra_include_paths=[
            jit_env.CUTLASS_INCLUDE_DIRS[0] / ".." / "examples" / "77_blackwell_fmha",
            jit_env.CUTLASS_INCLUDE_DIRS[0] / ".." / "examples" / "common",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


def get_mla_module():
    global _mla_module
    if _mla_module is None:
        _mla_module = gen_mla_module().build_and_load()
    return _mla_module


_batch_mla_modules = {}
_batch_mla_sm90_modules = {}


def get_batch_mla_module(backend):
    def backend_module(*args):
        global _batch_mla_modules, _batch_mla_sm90_modules
        modules_dict = (
            _batch_mla_modules if backend == "fa2" else _batch_mla_sm90_modules
        )
        if args not in modules_dict:
            modules_dict[args] = gen_batch_mla_module(backend, *args).build_and_load()
        return modules_dict[args]

    return backend_module


class BatchMLAPagedAttentionWrapper:
    r"""Wrapper class for MLA (`Multi-head Latent Attention <https://arxiv.org/abs/2405.04434>`_)
    PagedAttention on DeepSeek models. This kernel can be used in decode, and incremental prefill
    and should be used together with `Matrix Absorption trick
    <https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md>`_:
    where :math:`W_{UQ}` is absorbed with :math:`W_{UK}`, and :math:`W_{UV}` is
    absorbed with :math:`W_{O}`.
    For MLA attention without Matrix Absorption (``head_dim_qk=192`` and ``head_dim_vo=128``, which is
    used in prefilling self-attention stage), please use
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`.

    More information about The Paged KV-Cache layout in MLA is explained in our tutorial
    :ref:`MLA Page Layout <mla-page-layout>`.

    For more details about the MLA computation, Matrix Absorption and FlashInfer's MLA implementation,
    please refer to our `blog post <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_local_heads = 128
    >>> batch_size = 114
    >>> head_dim_ckv = 512
    >>> head_dim_kpe = 64
    >>> page_size = 1
    >>> mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    ...     torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    ...     backend="fa2"
    ... )
    >>> q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    >>> kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    >>> kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    >>> kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    >>> q_nope = torch.randn(
    ...     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> q_pe = torch.zeros(
    ...     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> ckv = torch.randn(
    ...     batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> kpe = torch.zeros(
    ...     batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    >>> mla_wrapper.plan(
    ...     q_indptr,
    ...     kv_indptr,
    ...     kv_indices,
    ...     kv_lens,
    ...     num_local_heads,
    ...     head_dim_ckv,
    ...     head_dim_kpe,
    ...     page_size,
    ...     False,  # causal
    ...     sm_scale,
    ...     q_nope.dtype,
    ...     ckv.dtype,
    ... )
    >>> o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    >>> o.shape
    torch.Size([114, 128, 512])
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        r"""Constructor for BatchMLAPagedAttentionWrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indices`` array.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_len_arr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_len_arr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability. If ``cutlass`` is provided, the MLA
            kernels will be generated by CUTLASS and only float_workspace_buffer is required and
            other arguments are ignored.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        if backend == "cutlass":
            self._backend = backend
            return

        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        if backend == "auto":
            self._backend = determine_mla_backend(self.device)
        else:
            self._backend = backend

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan the MLA attention computation.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
            For decoding attention, the length of each query is 1, and the content
            of the tensor should be ``[0, 1, 2, ..., batch_size]``.
        kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.Tensor
            The query length of each request, shape: ``[batch_size]``.
        num_heads : int
            The number of heads in query/output tensor.
        head_dim_ckv : int
            The head dimension of compressed-kv.
        head_dim_kpe : int
            The head dimension for rope k-cache.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to use causal attention.
        sm_scale : float
            The scale factor for softmax operation.
        q_data_type : torch.dtype
            The data type of the query tensor.
        kv_data_type : torch.dtype
            The data type of the kv-cache tensor.
        use_profiler : bool, optional
            Whether to enable intra-kernel profiler, default is False.
        """
        self._cached_module = get_batch_mla_module(self._backend)(
            q_data_type,
            kv_data_type,
            q_data_type,
            qo_indptr.dtype,
            head_dim_ckv,
            head_dim_kpe,
            use_profiler,
        )
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")

        if self._use_cuda_graph:
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=True)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=True)
            self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
            self._kv_len_arr_buf.copy_(kv_len_arr, non_blocking=True)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler

        self._plan_info = self._cached_module.plan.default(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,  # head_dim_o
            causal,
        )

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor without rope, shape: ``[batch_size, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            The rope part of the query tensor, shape: ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models.
        kpe_cache : torch.Tensor
            The rope part of the kv-cache tensor, shape: ``[num_pages, page_size, head_dim_kpe]``.
            ``head_dim_kpe`` is 64 in DeepSeek v2/v3 models.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The query length of each request, shape: ``[batch_size]``. Required when ``backend`` is ``cutlass``.
        page_table : Optional[torch.Tensor]
            The page table of the paged kv-cache, shape: ``[batch_size, num_pages]``. Required when ``backend`` is ``cutlass``.
        """
        if self._backend == "cutlass":
            if return_lse:
                raise ValueError("return_lse does not support cutlass backend for now.")
            if profiler_buffer is not None:
                raise ValueError(
                    "profiler_buffer does not support cutlass backend for now."
                )
            self._cached_module = get_mla_module()
            if out is None:
                out = torch.empty_like(q_nope)
            else:
                _check_shape_dtype_device(
                    out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
                )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            ckv_kpe_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
            _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
            lse = torch.empty(0, dtype=torch.float32, device=self.device)
            self._cached_module.cutlass_mla_paged_attention.default(
                self._float_workspace_buffer,
                out,
                lse,
                q_nope_pe,
                ckv_kpe_cache,
                kv_len,
                page_table,
            )
            return out

        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        num_heads = q_nope.shape[1]
        page_size = self._page_size
        sm_scale = self._sm_scale
        causal = self._causal
        mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        device = self.device
        if out is None:
            out = torch.empty_like(q_nope)
        else:
            _check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=device)
            else:
                _check_shape_dtype_device(
                    lse, q_nope.shape[:2], torch.float32, q_nope.device, "lse"
                )
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run.default(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            num_heads,
            page_size,
            sm_scale,
            *profiler_args,
        )

        return (out, lse) if return_lse else out
````

## File: norm.py
````python
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

from functools import cache
from typing import Any, Optional

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import register_custom_op, register_fake_op

_norm_module = None


def gen_norm_module() -> JitSpec:
    return gen_jit_spec(
        "norm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "norm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_norm_ops.cu",
        ],
    )


def get_norm_module():
    global _norm_module
    if _norm_module is None:
        _norm_module = gen_norm_module().build_and_load()
    return _norm_module


@cache
def get_module_attr(attr: str) -> Any:
    global _norm_module
    if _norm_module is None:
        get_norm_module()
    return getattr(_norm_module, attr).default


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    get_module_attr("rmsnorm")(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    pass


@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    get_module_attr("fused_add_rmsnorm")(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    pass


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    get_module_attr("gemma_rmsnorm")(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    get_module_attr("gemma_fused_add_rmsnorm")(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    pass
````

## File: page.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

from functools import cache
from typing import Any, Optional, Tuple, Union

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import (
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    register_custom_op,
    register_fake_op,
)

_page_module = None


def gen_page_module() -> JitSpec:
    return gen_jit_spec(
        "page",
        [
            jit_env.FLASHINFER_CSRC_DIR / "page.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_page_ops.cu",
        ],
    )


def get_page_module():
    global _page_module
    if _page_module is None:
        _page_module = gen_page_module().build_and_load()
    return _page_module


@cache
def get_module_attr(attr: str) -> Any:
    global _page_module
    if _page_module is None:
        get_page_module()
    return getattr(_page_module, attr).default


def block_sparse_indices_to_vector_sparse_offsets(
    block_sparse_indices: torch.Tensor,
    block_sparse_indptr: torch.Tensor,
    vector_sparse_offsets: torch.Tensor,
    vector_sparse_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    stride_block: int,
    stride_n: int,
    block_size: int,
) -> torch.Tensor:
    if block_size == 1:
        if stride_block == 1:
            return block_sparse_indices
        else:
            return block_sparse_indices * stride_block

    assert block_sparse_indices.dtype == torch.int32
    assert block_sparse_indptr.dtype == torch.int32
    assert vector_sparse_offsets.dtype == torch.int32
    assert vector_sparse_indptr.dtype == torch.int32
    assert kv_lens.dtype == torch.int32
    batch_size = block_sparse_indptr.size(0) - 1
    get_module_attr("block_sparse_indices_to_vector_sparse_offsets")(
        block_sparse_indices,
        block_sparse_indptr,
        vector_sparse_offsets,
        vector_sparse_indptr,
        kv_lens,
        stride_block,
        stride_n,
        batch_size,
        block_size,
    )
    return vector_sparse_offsets


@register_custom_op(
    "flashinfer::append_paged_mla_kv_cache",
    mutates_args=("ckv_cache", "kpe_cache"),
)
def _append_paged_mla_kv_cache_kernel(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_module_attr("append_paged_mla_kv_cache")(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


@register_custom_op(
    "flashinfer::append_paged_kv_cache",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: Optional[torch.Tensor],
    paged_v_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    layout: int,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_module_attr("append_paged_kv_cache")(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout,
    )


@register_fake_op("flashinfer::append_paged_kv_cache")
def _fake_append_paged_kv_cache_kernel(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_k_cache: Optional[torch.Tensor],
    paged_v_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    layout: int,
) -> None:
    pass


def get_batch_indices_positions(
    append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert append indptr and sequence lengths to batch indices and positions.

    Parameters
    ----------
    append_indptr : torch.Tensor
        The indptr of the ragged tensor, shape: ``[batch_size + 1]``.
    seq_lens: torch.Tensor
        The sequence lengths of each request in the KV-Cache, shape: ``[batch_size]``.
    nnz : int
        The number of entries in the ragged tensor.

    Returns
    -------
    batch_indices: torch.Tensor
        The batch indices of each entry in the ragged tensor, shape: ``[nnz]``.
    positions: torch.Tensor
        The positions of each entry in the ragged tensor, shape: ``[nnz]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> nnz_kv = 10
    >>> append_indptr = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device="cuda:0")
    >>> seq_lens = torch.tensor([5, 5, 5, 5])
    >>> batch_indices, positions = flashinfer.get_batch_indices_positions(append_indptr, seq_lens, nnz_kv)
    >>> batch_indices
    tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
    >>> positions  # the rightmost column index of each row
    tensor([4, 3, 4, 2, 3, 4, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function is similar to `CSR2COO <https://docs.nvidia.com/cuda/cusparse/#csr2coo>`_
    conversion in cuSPARSE library, with the difference that we are converting from a ragged
    tensor (which doesn't require a column indices array) to a COO format.

    See Also
    --------
    append_paged_kv_cache
    """
    batch_size = append_indptr.size(0) - 1
    batch_indices = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
    positions = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
    from .triton.page import get_batch_indices_positions_kernel

    get_batch_indices_positions_kernel[(batch_size,)](
        append_indptr, seq_lens, batch_indices, positions, num_stages=2
    )
    return batch_indices, positions


def get_seq_lens(
    kv_indptr: torch.Tensor, kv_last_page_len: torch.Tensor, page_size: int
) -> torch.Tensor:
    r"""Convert KV indptr and last page length to sequence lengths.

    Parameters
    ----------
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    page_size : int
        The size of a page in the paged kv-cache.

    Returns
    -------
    seq_lens: torch.Tensor
        The sequence lengths of each request in the paged kv-cache, shape: ``[batch_size]``.
    """
    return (
        torch.clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size
        + kv_last_page_len
    )


def append_paged_mla_kv_cache(
    append_ckv: torch.Tensor,
    append_kpe: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    ckv_cache: Optional[torch.Tensor],
    kpe_cache: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    r"""Append a batch of key-value pairs to a paged key-value cache,
    Note: current only support ckv=512 and kpe=64

    Parameters
    ----------
    append_ckv : torch.Tensor
        The compressed kv tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], ckv_dim]``.
    append_kpe : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], kpe_dim]``.
    batch_indices : torch.Tensor
        The batch indices of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    positions : torch.Tensor
        The positions of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    ckv_cache : cache for compressed kv, torch.Tensor, shape: [page_num, page_size, ckv_dim]
    kpe_cache : cache for key position embedding, torch.Tensor, shape: [page_num, page_size, kpe_dim]
    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    """
    _append_paged_mla_kv_cache_kernel(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
) -> None:
    r"""Append a batch of key-value pairs to a paged key-value cache.

    Parameters
    ----------
    append_key : torch.Tensor
        The key tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    append_value : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    batch_indices : torch.Tensor
        The batch indices of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    positions : torch.Tensor
        The positions of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        The paged KV-Cache stored as a tuple of tensors or a single tensor:

        * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
          ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
          and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

        * a single 5-D tensor with shape:
          ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
          :attr:`kv_layout` is ``NHD``, and
          ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
          :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
          ``paged_kv_cache[:, 1]`` is the value-cache.

    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    kv_layout : str
        The layout of the paged kv-cache, either ``NHD`` or ``HND``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> nnz_kv = 100
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    >>> v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    >>> # 45 + 8 + 25 + 22 = nnz_kv
    >>> kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    >>> kv_append_indptr = torch.cat(
    ...     [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ... ).int()  # [0, 45, 53, 78, 100]
    >>> max_num_pages = 1000
    >>> page_size = 16
    >>> paged_kv_cache = torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    >>> num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    >>> kv_page_indptr = torch.cat(
    ...     [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ... ).int()
    >>> # use first 8 pages in the paged-kv
    >>> kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    >>> # 45 = (3 - 1) * 16 + 13
    >>> # 8 = (1 - 1) * 16 + 8
    >>> # 25 = (2 - 1) * 16 + 9
    >>> # 22 = (2 - 1) * 16 + 6
    >>> kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    >>> batch_indices, positions = flashinfer.get_batch_indices_positions(
    ...     kv_append_indptr, flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size), nnz_kv
    ... )
    >>> batch_indices
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
    >>> positions
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44,  0,  1,  2,  3,  4,  5,  6,  7,  0,
            1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21], device='cuda:0',
        dtype=torch.int32)
    >>> flashinfer.append_paged_kv_cache(
    ...     k_append,
    ...     v_append,
    ...     batch_indices,
    ...     positions,
    ...     paged_kv_cache,
    ...     kv_page_indices,
    ...     kv_page_indptr,
    ...     kv_last_page_len
    ... )

    Note
    ----
    The function assumes that the space for appended k/v has already been allocated,
    which means :attr:`kv_indices`, :attr:`kv_indptr`, :attr:`kv_last_page_len` has
    incorporated appended k/v.

    See Also
    --------
    get_batch_indices_positions
    """
    _check_kv_layout(kv_layout)
    _append_paged_kv_cache_kernel(
        append_key,
        append_value,
        batch_indices,
        positions,
        *_unpack_paged_kv_cache(paged_kv_cache, kv_layout),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[kv_layout].value,
    )
````

## File: pod.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

import math
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union

import torch

from .jit import (
    gen_pod_module,
)
from .page import get_seq_lens
from .prefill import get_batch_prefill_module
from .quantization import packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
)

_pod_modules = {}


def get_pod_module(*args):
    global _pod_modules
    if args not in _pod_modules:
        module = gen_pod_module(*args).build_and_load()
        run_tensor = module.pod_with_kv_cache_tensor.default
        # Register the module
        _pod_modules[args] = SimpleNamespace(run_tensor=run_tensor)
    return _pod_modules[args]


class PODWithPagedKVCacheWrapper:
    r"""Wrapper class for POD-Attention with paged kv-cache (first proposed in
    `<https://arxiv.org/abs/2410.18038>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.PODWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.plan(
    ...     kv_page_indptr,
    ...     kv_page_indices,
    ...     kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     pos_encoding_mode="NONE",
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     # TODO_AK: DEMONSTRATE USAGE OF POD
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's POD-Attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`PODWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)
        """
        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0], gen_customize_batch_prefill_module("fa2", *jit_args)
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0], gen_customize_batch_decode_module(*jit_args)
                )
        else:
        """
        # Override options. Only tensor core version is performant.
        use_tensor_cores = True
        self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = use_cuda_graph

        if use_cuda_graph:
            # NOTE(Zihao): if once created, no need to update it in plan/run
            self._qo_indptr_buf = torch.arange(
                self._fixed_batch_size + 1,
                dtype=torch.int32,
                device=float_workspace_buffer.device,
            )

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
    ) -> None:
        r"""Plan POD's batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        # Logits soft cap is not supported currently
        logits_soft_cap = False
        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            self._cached_module = get_batch_prefill_module("fa2")(
                q_data_type,
                kv_data_type,
                q_data_type,
                indptr.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                window_left != -1,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            indptr_host,
            kv_lens_arr_host,
            batch_size,  # total_num_rows
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,  # causal
        )

        self._indptr_type = indptr.dtype
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def run(
        self,
        # Main params (prefill and decode)
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        v_p: torch.Tensor,
        q_d: torch.Tensor,
        paged_kv_cache_d: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        # Prefill options
        custom_mask_p: Optional[torch.Tensor] = None,
        packed_custom_mask_p: Optional[torch.Tensor] = None,
        causal_p: bool = False,
        kv_layout_p: str = "NHD",
        pos_encoding_mode_p: str = "NONE",
        sm_scale_p: Optional[float] = None,
        window_left_p: int = -1,
        rope_scale_p: Optional[float] = None,
        rope_theta_p: Optional[float] = None,
        return_lse_p: bool = False,
        # Decode options
        custom_mask_d: Optional[torch.Tensor] = None,
        packed_custom_mask_d: Optional[torch.Tensor] = None,
        causal_d: bool = False,
        kv_layout_d: str = "NHD",
        pos_encoding_mode_d: str = "NONE",
        sm_scale_d: Optional[float] = None,
        window_left_d: int = -1,
        rope_scale_d: Optional[float] = None,
        rope_theta_d: Optional[float] = None,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse_d: bool = False,
        use_fp16_qk_reduction: bool = False,
        *args,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute POD-attention for a batch of requests."""
        # Currently unsupported
        logits_soft_cap_p = None
        logits_soft_cap_d = None
        # Prefill setup
        _check_pos_encoding_mode(pos_encoding_mode_p)
        _check_kv_layout(kv_layout_p)
        tmp_p = _get_cache_buf("pod_with_kv_cache_tmp", 32 * 1024 * 1024, q_p.device)
        if logits_soft_cap_p is None:
            logits_soft_cap_p = 0.0
        if sm_scale_p is None:
            sm_scale_p = 1.0 / math.sqrt(q_p.size(-1))
        if rope_scale_p is None:
            rope_scale_p = 1.0
        if rope_theta_p is None:
            rope_theta_p = 1e4
        if custom_mask_p is not None and packed_custom_mask_p is None:
            # create packed custom mask from custom mask
            packed_custom_mask_p = packbits(
                custom_mask_p.contiguous().view(-1), bitorder="little"
            )

        if packed_custom_mask_p is not None:
            mask_mode_p = MaskMode.CUSTOM.value
        else:
            if causal_p:
                mask_mode_p = MaskMode.CAUSAL.value
            else:
                mask_mode_p = MaskMode.NON_CAUSAL.value

        lse_p = None
        if return_lse_p:
            lse_p = torch.empty(
                (q_p.size(0), q_p.size(1)), dtype=torch.float32, device=q_p.device
            )

        out_p = torch.empty_like(q_p)

        # Decode setup
        k_cache_d, v_cache_d = _unpack_paged_kv_cache(paged_kv_cache_d, self._kv_layout)
        _check_cached_qkv_data_type(
            q_d, k_cache_d, self._cached_q_data_type, self._cached_kv_data_type
        )
        # TODO_AK: Where are these coming from?
        pos_encoding_mode_d = self._pos_encoding_mode
        window_left_d = self._window_left
        logits_soft_cap_d = self._logits_soft_cap
        sm_scale_d = self._sm_scale
        rope_scale_d = self._rope_scale
        rope_theta_d = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode_d)
        # What are the above for and what are the below?
        if logits_soft_cap_d is None:
            logits_soft_cap_d = 0.0
        if sm_scale_d is None:
            head_dim = q_d.shape[-1]
            sm_scale_d = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale_d *= q_scale
        if k_scale is not None:
            sm_scale_d *= k_scale
        if rope_scale_d is None:
            rope_scale_d = 1.0
        if rope_theta_d is None:
            rope_theta_d = 1e4

        lse_d = None
        if return_lse_d:
            lse_d = torch.empty(
                (q_d.size(0), q_d.size(1)), dtype=torch.float32, device=q_d.device
            )
        out_d = torch.empty_like(q_d)

        module_getter = get_pod_module(
            # Prefill params
            q_p.dtype,
            k_p.dtype,
            q_p.dtype,
            q_p.shape[-1],
            PosEncodingMode[pos_encoding_mode_p].value,
            window_left_p >= 0,  # use_sliding_window
            logits_soft_cap_p > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
            # Decode params
            # q_d.dtype,
            # self._cached_kv_data_type,
            # self._cached_q_data_type,
            self._indptr_type,
            # head_dim,  # head_dim_qk
            # head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode_d].value,
            window_left_d != -1,  # use_sliding_window
            logits_soft_cap_d > 0,  # use_logits_soft_cap
        )
        module_getter.run_tensor(
            # Prefill params
            q_p,
            k_p,
            v_p,
            tmp_p,
            out_p,
            lse_p,
            mask_mode_p,
            TensorLayout[kv_layout_p].value,
            window_left_p,
            packed_custom_mask_p,
            _get_cache_alibi_slopes_buf(q_p.shape[1], q_p.device),
            logits_soft_cap_p,
            sm_scale_p,
            1.0 / rope_scale_p,
            1.0 / rope_theta_p,
            # Decode params
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_d,
            k_cache_d,
            v_cache_d,
            self._qo_indptr_buf,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            out_d,
            lse_d,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[self._kv_layout].value,
            window_left_d,
            None,  # packed_custom_mask
            None,  # mask_indptr_buf
            _get_cache_alibi_slopes_buf(q_d.shape[1], q_d.device),
            logits_soft_cap_d,
            sm_scale_d,
            1.0 / rope_scale_d,
            1.0 / rope_theta_d,
        )

        if v_scale is not None:
            out_d *= v_scale

        return (out_p, out_d)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass
````

## File: prefill.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

import functools
import logging
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import (
    gen_batch_prefill_module,
    gen_customize_batch_prefill_module,
    gen_fmha_cutlass_sm100a_module,
    gen_single_prefill_module,
    get_batch_prefill_uri,
    get_single_prefill_uri,
)
from .page import block_sparse_indices_to_vector_sparse_offsets, get_seq_lens
from .quantization import packbits, segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    determine_attention_backend,
    is_float8,
    is_sm100a_supported,
    register_custom_op,
    register_fake_op,
)

_single_prefill_modules = {}
_single_prefill_sm90_modules = {}
_batch_prefill_modules = {}
_batch_prefill_sm90_modules = {}
_batch_prefill_jit_modules = {}


@functools.cache
def get_fmha_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: PosEncodingMode,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool = False,
):
    if is_sm100a_supported(torch.device("cuda")):
        return gen_fmha_cutlass_sm100a_module(
            dtype_q,
            dtype_kv,
            dtype_o,
            dtype_idx,
            head_dim_qk,
            head_dim_vo,
            pos_encoding_mode,
            use_sliding_window,
            use_logits_soft_cap,
        ).build_and_load()
    else:
        raise ValueError("SM100A is not supported on this device")


def get_single_prefill_module(backend):
    def backend_module(*args):
        global _single_prefill_modules, _single_prefill_sm90_modules
        modules_dict = (
            _single_prefill_modules
            if backend == "fa2"
            else _single_prefill_sm90_modules
        )
        if args not in modules_dict:
            uri = get_single_prefill_uri(backend, *args)
            module = gen_single_prefill_module(backend, *args).build_and_load()
            run_func = module.run.default

            # torch library for single_prefill_with_kv_cache

            @register_custom_op(
                f"flashinfer::{uri}_run", mutates_args=("tmp", "o", "maybe_lse")
            )
            def run_single_prefill(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                tmp: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_packed_custom_mask: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                scale_q: Optional[torch.Tensor],
                scale_k: Optional[torch.Tensor],
                scale_v: Optional[torch.Tensor],
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                if backend == "fa3":
                    if not is_float8(q):
                        run_func(
                            q,
                            k,
                            v,
                            tmp,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            logits_soft_cap,
                            sm_scale,
                        )
                    else:
                        # FP8 enabled
                        run_func(
                            q,
                            k,
                            v,
                            tmp,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            scale_q,
                            scale_k,
                            scale_v,
                            sm_scale,
                        )
                else:
                    run_func(
                        q,
                        k,
                        v,
                        tmp,
                        o,
                        maybe_lse,
                        mask_mode,
                        layout,
                        window_left,
                        maybe_packed_custom_mask,
                        maybe_alibi_slopes,
                        logits_soft_cap,
                        sm_scale,
                        1.0 / rope_scale,  # rope_rcp_scale
                        1.0 / rope_theta,  # rope_rcp_theta
                    )
                return o

            @register_fake_op(f"flashinfer::{uri}_run")
            def _fake_run_single_prefill(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                tmp: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_packed_custom_mask: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                pass

            # Register the module
            modules_dict[args] = SimpleNamespace(run=run_single_prefill)

        return modules_dict[args]

    return backend_module


def get_batch_prefill_module(backend):
    def backend_module(*args):
        global _batch_prefill_modules, _batch_prefill_sm90_modules
        modules_dict = (
            _batch_prefill_modules if backend == "fa2" else _batch_prefill_sm90_modules
        )
        if args not in modules_dict:
            uri = get_batch_prefill_uri(backend, *args)
            module = gen_batch_prefill_module(backend, *args).build_and_load()
            plan_func = module.plan.default
            ragged_run_func = module.ragged_run.default
            paged_run_func = module.paged_run.default

            # torch library for ragged_run

            @register_custom_op(
                f"flashinfer::{uri}_ragged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "o",
                    "maybe_lse",
                ),
            )
            def ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                if backend == "fa2":
                    ragged_run_func(
                        float_workspace_buffer,
                        int_workspace_buffer,
                        plan_info_vec,
                        q,
                        k,
                        v,
                        qo_indptr,
                        kv_indptr,
                        o,
                        maybe_lse,
                        mask_mode,
                        layout,
                        window_left,
                        maybe_custom_mask,
                        maybe_mask_indptr,
                        maybe_alibi_slopes,
                        maybe_prefix_len_ptr,
                        maybe_token_pos_in_items_ptr,
                        maybe_max_item_len_ptr,
                        logits_soft_cap,
                        sm_scale,
                        1.0 / rope_scale,  # rope_rcp_scale
                        1.0 / rope_theta,  # rope_rcp_theta
                        token_pos_in_items_len,
                    )
                else:
                    ragged_run_func(
                        float_workspace_buffer,
                        int_workspace_buffer,
                        plan_info_vec,
                        q,
                        k,
                        v,
                        qo_indptr,
                        kv_indptr,
                        o,
                        maybe_lse,
                        mask_mode,
                        layout,
                        window_left,
                        maybe_prefix_len_ptr,
                        maybe_token_pos_in_items_ptr,
                        maybe_max_item_len_ptr,
                        logits_soft_cap,
                        sm_scale,
                        token_pos_in_items_len,
                    )

                return o

            @register_fake_op(f"flashinfer::{uri}_ragged_run")
            def _fake_ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                pass

            # torch library for paged_run

            @register_custom_op(
                f"flashinfer::{uri}_paged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "paged_k_cache",
                    "paged_v_cache",
                    "o",
                    "maybe_lse",
                ),
            )
            def paged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                paged_k_cache: torch.Tensor,
                paged_v_cache: torch.Tensor,
                qo_indptr: torch.Tensor,
                paged_kv_indptr: torch.Tensor,
                paged_kv_indices: torch.Tensor,
                paged_kv_last_page_len: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                scale_q: Optional[torch.Tensor],
                scale_k: Optional[torch.Tensor],
                scale_v: Optional[torch.Tensor],
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                if backend == "fa2":
                    assert not is_float8(q)
                    paged_run_func(
                        float_workspace_buffer,
                        int_workspace_buffer,
                        plan_info_vec,
                        q,
                        paged_k_cache,
                        paged_v_cache,
                        qo_indptr,
                        paged_kv_indptr,
                        paged_kv_indices,
                        paged_kv_last_page_len,
                        o,
                        maybe_lse,
                        mask_mode,
                        layout,
                        window_left,
                        maybe_custom_mask,
                        maybe_mask_indptr,
                        maybe_alibi_slopes,
                        maybe_prefix_len_ptr,
                        maybe_token_pos_in_items_ptr,
                        maybe_max_item_len_ptr,
                        logits_soft_cap,
                        sm_scale,
                        1.0 / rope_scale,  # rope_rcp_scale
                        1.0 / rope_theta,  # rope_rcp_theta
                        token_pos_in_items_len,
                    )
                else:
                    if not is_float8(q):
                        paged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            paged_k_cache,
                            paged_v_cache,
                            qo_indptr,
                            paged_kv_indptr,
                            paged_kv_indices,
                            paged_kv_last_page_len,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            maybe_prefix_len_ptr,
                            maybe_token_pos_in_items_ptr,
                            maybe_max_item_len_ptr,
                            logits_soft_cap,
                            sm_scale,
                            token_pos_in_items_len,
                        )
                    else:
                        paged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            paged_k_cache,
                            paged_v_cache,
                            qo_indptr,
                            paged_kv_indptr,
                            paged_kv_indices,
                            paged_kv_last_page_len,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            scale_q,
                            scale_k,
                            scale_v,
                            sm_scale,
                        )
                return o

            @register_fake_op(f"flashinfer::{uri}_paged_run")
            def _fake_paged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                paged_k_cache: torch.Tensor,
                paged_v_cache: torch.Tensor,
                qo_indptr: torch.Tensor,
                paged_kv_indptr: torch.Tensor,
                paged_kv_indices: torch.Tensor,
                paged_kv_last_page_len: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                pass

            # Register the module.
            #
            # Note that plan is not part of model logic. It should not be included in
            # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
            modules_dict[args] = SimpleNamespace(
                plan=plan_func,
                ragged_run=ragged_run,
                paged_run=paged_run,
            )
        return modules_dict[args]

    return backend_module


def get_batch_prefill_jit_module(module_name: str, jit_module: Any):
    global _batch_prefill_jit_modules
    if module_name in _batch_prefill_jit_modules:
        return _batch_prefill_jit_modules[module_name]

    plan_func = jit_module.plan.default
    ragged_run_func = jit_module.ragged_run.default
    paged_run_func = jit_module.paged_run.default

    # torch library for ragged_run
    @register_custom_op(
        f"flashinfer::{module_name}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        ragged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_ragged_run")
    def _fake_ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # torch library for paged_run
    @register_custom_op(
        f"flashinfer::{module_name}_paged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        paged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_paged_run")
    def _fake_paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    _batch_prefill_jit_modules[module_name] = SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )

    return _batch_prefill_jit_modules[module_name]


def single_prefill_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    mask_mode: int = MaskMode.NON_CAUSAL.value,
    window_left: int = -1,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    device = q.device
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, device=device
    )
    o = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=device)
    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=device)
    jit_module.run.default(
        q,
        k,
        v,
        tmp,
        o,
        lse,
        mask_mode,
        TensorLayout[kv_layout].value,
        window_left,
        *args,
    )
    return (o, lse) if return_lse else o


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim_qk]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_qk]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim_qk]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_vo]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim_vo]`` if :attr:`kv_layout` is
        ``HND``.
    scale_q : Optional[torch.Tensor]
        The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_k : Optional[torch.Tensor]
        The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_v : Optional[torch.Tensor]
        The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    o_dtype : Optional[torch.dtype]
        The output tensor data type, if not provided, will be set to the same as the q.
        This is necessary as output dtype cannot be automatically inferred in quant.
    custom_mask : Optional[torch.Tensor]
        The custom boolean mask tensor, shape: ``[qo_len, kv_len]``.
        The elements in the mask tensor should be either ``True`` or ``False``,
        where ``False`` means the corresponding element in the attention matrix will be
        masked out.

        When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
        function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
        additional overhead.
    packed_custom_mask : Optional[torch.Tensor]
        The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
        The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    use_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    window_left : int
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    logits_soft_cap : Optional[float]
        The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
        provided, will be set to ``0``. If greater than 0, the logits will be capped according to
        formula:
        :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
        where :math:`x` is the input logits.
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim_qk)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    backend : str
        The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
        If set to ``auto``, the function will automatically choose the backend based on the
        device architecture and kernel availability.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[qo_len, num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>> q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,
            use_fp16_qk_reduction=True)
    >>> o.shape
    torch.Size([128, 32, 128])
    >>> mask = torch.tril(
    >>>     torch.full((qo_len, kv_len), True, device="cuda:0"),
    >>>     diagonal=(kv_len - qo_len),
    >>> )
    >>> mask
    tensor([[ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            ...,
            [ True,  True,  True,  ...,  True, False, False],
            [ True,  True,  True,  ...,  True,  True, False],
            [ True,  True,  True,  ...,  True,  True,  True]], device='cuda:0')
    >>> o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_custom, rtol=1e-3, atol=1e-3)
    True

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if custom_mask is not None and packed_custom_mask is None:
        # create packed custom mask from custom mask
        packed_custom_mask = packbits(
            custom_mask.contiguous().view(-1), bitorder="little"
        )

    if packed_custom_mask is not None:
        mask_mode = MaskMode.CUSTOM.value
    else:
        if causal:
            mask_mode = MaskMode.CAUSAL.value
        else:
            mask_mode = MaskMode.NON_CAUSAL.value

    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)

    if is_float8(q):
        # FP8 quant enabled, do sanity check:
        #   1. unsupported feature
        #   2. dtype check
        assert window_left == -1
        assert q.dtype == k.dtype == v.dtype
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        if scale_q is None:
            scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
        if scale_k is None:
            scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
        if scale_v is None:
            scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

    if backend == "auto":
        backend = determine_attention_backend(
            q.device,
            PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            packed_custom_mask is not None,  # use_custom_mask
            q.dtype,
            k.dtype,
        )
    module_getter = get_single_prefill_module(backend)

    # o_dtype should be provided for FP8 attention
    if o_dtype is None:
        o_dtype = q.dtype
    out = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=o_dtype, device=q.device)

    module_getter(
        q.dtype,
        k.dtype,
        out.dtype,
        q.shape[-1],  # head_dim_qk
        v.shape[-1],  # head_dim_vo
        PosEncodingMode[pos_encoding_mode].value,
        window_left >= 0,  # use_sliding_window
        logits_soft_cap > 0,  # use_logits_soft_cap
        use_fp16_qk_reduction,
    ).run(
        q,
        k,
        v,
        tmp,
        out,
        lse,
        mask_mode,
        TensorLayout[kv_layout].value,
        window_left,
        packed_custom_mask,
        _get_cache_alibi_slopes_buf(q.shape[1], q.device),
        logits_soft_cap,
        sm_scale,
        scale_q,
        scale_k,
        scale_v,
        rope_scale,
        rope_theta,
    )

    return (out, lse) if return_lse else out


single_prefill_with_kv_cache_return_lse = functools.partial(
    single_prefill_with_kv_cache, return_lse=True
)


def _compute_page_mask_indptr(
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if len(qo_indptr) != len(paged_kv_indptr):
        raise ValueError(
            "The length of qo_indptr and paged_kv_indptr should be the same."
        )
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1])
        * (
            (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) * page_size
            + paged_kv_last_page_len
        ),
        0,
    )
    return mask_indptr


class BatchPrefillWithPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with paged kv-cache for batch of
    requests.

    Check :ref:`our tutorial <kv-layout>` for page table layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> kv_cache_at_layer = torch.randn(
    ...     num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ... )
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     custom_mask=mask,
    ... )
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, kv_cache)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...



    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    create some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indptr`` array, the size of this
            buffer should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indices`` array, should be large
            enough to store the maximum possible size of the ``paged_kv_indices`` array during
            the lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True``.

        paged_kv_last_page_len_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_last_page_len`` array, the size of
            the buffer should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved buffer to store the custom mask tensor, should be large enough to
            store the maximum possible size of the packed custom mask tensor during the lifetime of
            the wrapper. This argument is only effective when ``use_cuda_graph`` is set to ``True``
            and the custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and the custom
            mask will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                gen_customize_batch_prefill_module(backend, *jit_args).build_and_load(),
            )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        if backend in ["fa3", "auto"]:
            # NOTE(Zihao): assume maximum accumulate kv length is 16M
            self._vector_sparse_indices_buffer = torch.empty(
                (16 * 1024 * 1024,), dtype=torch.int32, device=self.device
            )
            # NOTE(Zihao): assume maximum batch size is 32768
            self._vector_sparse_indptr_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indptr_buf):
                raise ValueError(
                    "paged_kv_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buf):
                raise ValueError(
                    "paged_kv_indices_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buf):
                raise ValueError(
                    "paged_kv_last_page_len_buf should be a torch.Tensor in CUDA graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                )
            if len(paged_kv_last_page_len_buf) != self._fixed_batch_size:
                raise ValueError(
                    "The length of paged_kv_last_page_len_buf should be batch_size."
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here, as they are optional
        else:
            self._fixed_batch_size = 0

        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Plan batch prefill/append attention on Paged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim_qk : int
            The dimension of the query/key heads.
        page_size : int
            The size of each page in the paged kv-cache.
        head_dim_vo : Optional[int]
            The dimension of the value/output heads, if not provided, will be set to
            ``head_dim_qk``.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        batch_size = len(qo_indptr) - 1
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                page_size,
            )
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        qo_indptr_host = qo_indptr.to("cpu")
        paged_kv_indptr_host = paged_kv_indptr.to("cpu")
        paged_kv_last_page_len_host = paged_kv_last_page_len.to("cpu")
        kv_lens_arr_host = get_seq_lens(
            paged_kv_indptr_host, paged_kv_last_page_len_host, page_size
        )
        self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
            kv_lens_arr_host, non_blocking=non_blocking
        )

        total_num_rows = qo_indptr_host[-1]

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    "cuda graph mode, the runtime batch size {} mismatches the batch size {} "
                    " set during initialization.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(paged_kv_indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The length of paged_kv_indices exceeds the allocated buffer size."
                )

            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
                paged_kv_indices,
                non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
            )

            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)].copy_(
                    packed_custom_mask,
                    non_blocking=(packed_custom_mask.device == self.device)
                    and non_blocking,
                )
                # NOTE(Zihao): mask_indptr has the same length as qo_indptr
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._paged_kv_indptr_buf = paged_kv_indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = paged_kv_indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = paged_kv_last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )
            else:
                self._custom_mask_buf = None
                self._mask_indptr_buf = None

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type

        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    self._custom_mask_buf is not None,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                q_data_type,
                paged_kv_indptr.dtype,
                head_dim_qk,
                head_dim_vo,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )

            self._cached_module = get_batch_prefill_module(self._backend)(
                *get_module_args
            )

        if self._backend == "fa3":
            if page_size != 1:
                vector_sparse_indptr_host = torch.cat(
                    [
                        torch.tensor(
                            [0], dtype=torch.int32, device=kv_lens_arr_host.device
                        ),
                        torch.cumsum(kv_lens_arr_host, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                self._vector_sparse_indptr_buffer[
                    : len(vector_sparse_indptr_host)
                ].copy_(vector_sparse_indptr_host, non_blocking=non_blocking)
                paged_kv_indptr_host = vector_sparse_indptr_host

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            paged_kv_indptr_host,
            kv_lens_arr_host,
            self._max_total_num_rows or total_num_rows,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim_qk,
            head_dim_vo,
            causal,
        )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and paged kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        *args
            Additional arguments for custom kernels.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )
        stride_block = k_cache.stride(0)
        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
            stride_n = k_cache.stride(1)
        else:
            page_size = k_cache.shape[2]
            stride_n = k_cache.stride(2)
        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty(
                q.shape[:-1] + v_cache.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            _check_shape_dtype_device(
                out, q.shape[:-1] + v_cache.shape[-1:], q.dtype, q.device, "out"
            )

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        if self._prefix_len_ptr is not None:
            mask_mode = MaskMode.MULTIITEMSCORING.value

        if self._backend == "fa3":
            # NOTE(Zihao): we divide both stride_block and stride_n by stride_n
            # because we will multiply stride_n back in the kernel
            sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                self._paged_kv_indices_buf,
                self._paged_kv_indptr_buf,
                self._vector_sparse_indices_buffer,  # output
                self._vector_sparse_indptr_buffer,
                self._kv_lens_buffer,
                stride_block // stride_n,
                1,  # stride_n // stride_n
                page_size,
            )
            sparse_indptr = self._vector_sparse_indptr_buffer
        else:
            sparse_indices = self._paged_kv_indices_buf
            sparse_indptr = self._paged_kv_indptr_buf

        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._qo_indptr_buf,
            sparse_indptr,
            sparse_indices,
            self._paged_kv_last_page_len_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
        ]

        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                self._prefix_len_ptr,
                self._token_pos_in_items_ptr,
                self._max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                None,  # scale_q, not supported yet
                None,  # scale_k
                None,  # scale_v
                rope_scale,
                rope_theta,
                self._token_pos_in_items_len,
            ]

        self._cached_module.paged_run(*run_args)
        if v_scale is not None:
            out *= v_scale

        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: This function is deprecated, please use :meth:`run_return_lse` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run_return_lse(q, paged_kv_cache, k_scale, v_scale)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


def _compute_mask_indptr(
    qo_indptr: torch.Tensor, kv_indptr: torch.Tensor
) -> torch.Tensor:
    if len(qo_indptr) != len(kv_indptr):
        raise ValueError("The length of qo_indptr and kv_indptr should be the same.")
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1]) * (kv_indptr[1:] - kv_indptr[:-1]),
        0,
    )
    return mask_indptr


class BatchPrefillWithRaggedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with ragged (tensor) kv-cache for
    batch of requests.

    Check :ref:`our tutorial <kv-layout>` for ragged kv-cache layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_kv = 100
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_indptr = qo_indptr.clone()
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, k, v)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     custom_mask=mask
    ... )
    >>> outputs_custom_mask = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, k, v)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...
    >>> outputs_custom_mask[0].shape
    torch.Size([100, 64, 128])


    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    create some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithRaggedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored as the provided buffers.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the custom mask tensor, should be large
            enough to store the maximum possible size of the packed custom mask tensor during the
            lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True`` and custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and custom mask
            will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)
        if jit_args is not None:
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                gen_customize_batch_prefill_module(backend, *jit_args).build_and_load(),
            )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            if not torch.is_tensor(kv_indptr_buf):
                raise ValueError(
                    "kv_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of kv_indptr_buf ({}) should be the same as qo_indptr_buf ({}).".format(
                        len(kv_indptr_buf), self._fixed_batch_size
                    )
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here,
            # as they may not be used.

        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Plan batch prefill/append attention on Ragged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr : torch.Tensor
            The indptr of the key/value tensor, shape: ``[batch_size + 1]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim_qk : int
            The dimension of the heads on query/key tensor.
        head_dim_vo : Optional[int]
            The dimension of the heads on value/output tensor.
            If not provided, will be set to ``head_dim_vo``.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.

            If provided, the custom mask will be added to the attention matrix before softmax
            and after scaling. The mask tensor should be in the same device as the input tensors.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This argument is ignored if ``mask`` is provided in :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim_qk)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults to torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this plan call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        batch_size = len(qo_indptr) - 1
        if len(kv_indptr) != batch_size + 1:
            raise ValueError(
                "The kv_indptr length should be equal to mask_indptr length."
            )
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_mask_indptr(qo_indptr, kv_indptr)
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")

        total_num_rows = qo_indptr_host[-1]

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=non_blocking)
            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in the attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)] = packed_custom_mask
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_len_arr = kv_indptr_host[1:] - kv_indptr_host[:-1]

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    self._custom_mask_buf is not None,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                q_data_type,
                kv_indptr.dtype,
                head_dim_qk,
                head_dim_vo,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            if self._backend == "cutlass":
                self._cached_module = get_cutlass_mha_module()(*get_module_args)
            else:
                self._cached_module = get_batch_prefill_module(self._backend)(
                    *get_module_args
                )

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr,
            self._max_total_num_rows or total_num_rows,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            1,  # page_size
            self.is_cuda_graph_enabled,
            head_dim_qk,
            head_dim_vo,
            causal,
        )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and kv-cache stored as
        ragged tensor.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_qk]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_qk]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_vo]``
        *args
            Additional arguments for the custom kernel.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        _check_cached_qkv_data_type(
            q, k, self._cached_q_data_type, self._cached_kv_data_type
        )

        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )
        if out is None:
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            _check_shape_dtype_device(
                out, q.shape[:-1] + v.shape[-1:], q.dtype, q.device, "out"
            )

        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr_buf,
            self._kv_indptr_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
        ]
        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                self._prefix_len_ptr,
                self._token_pos_in_items_ptr,
                self._max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                self._token_pos_in_items_len,
            ]

        self._cached_module.ragged_run(*run_args)
        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def forward_return_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: This function is deprecated, please use :meth:`run_return_lse` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run_return_lse(q, k, v)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


def fmha_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    workspace_buffer = _get_cache_buf(
        "fmha_varlen_cutlass_workspace", 32 * 1024 * 1024, q.device
    )
    module = get_fmha_module(
        q.dtype,
        k.dtype,
        v.dtype,
        torch.int32,
        q.shape[2],
        v.shape[2],
        PosEncodingMode.NONE.value,
        False,  # use_sliding_window
        False,  # use_logits_soft_cap
    )
    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    nnz_kv, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)

    qo_lens = qo_segment_offsets[1:] - qo_segment_offsets[:-1]
    kv_lens = kv_segment_offsets[1:] - kv_segment_offsets[:-1]
    batch_size = qo_lens.shape[0]
    max_qo_len = qo_lens.max()
    max_kv_len = kv_lens.max()
    qo_total_len = nnz_qo

    if out is None:
        out = torch.empty(
            qo_total_len + max(max_qo_len, 128),
            num_qo_heads,
            head_dim_vo,
            device=q.device,
            dtype=q.dtype,
        )[max(max_qo_len, 128) :]

    if lse is None:
        lse = torch.empty(
            qo_total_len, num_qo_heads, device=q.device, dtype=torch.float32
        )

    module.run(
        workspace_buffer,
        q,
        k,
        v,
        qo_lens,
        kv_lens,
        qo_segment_offsets,
        kv_segment_offsets,
        out,
        lse,
        mask_mode_code,
        sm_scale,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        batch_size,
        nnz_qo,
        nnz_kv,
        max_qo_len,
        max_kv_len,
    )

    return out, lse


@functools.cache
def get_cutlass_mha_module():
    def backend_module(*args):
        modules_dict = _batch_prefill_modules

        if args not in modules_dict:
            uri = get_batch_prefill_uri("cutlass", *args)
            module = get_fmha_module(*args)

            @register_custom_op(
                f"flashinfer::{uri}_ragged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "o",
                    "maybe_lse",
                ),
            )
            def ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                return fmha_varlen(
                    q,
                    k,
                    v,
                    qo_indptr,
                    kv_indptr,
                    o,
                    maybe_lse,
                    mask_mode == MaskMode.CAUSAL.value,
                    sm_scale,
                )

            @register_custom_op(
                f"flashinfer::{uri}_paged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "paged_k_cache",
                    "paged_v_cache",
                    "o",
                    "maybe_lse",
                ),
            )
            def paged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                paged_k_cache: torch.Tensor,
                paged_v_cache: torch.Tensor,
                qo_indptr: torch.Tensor,
                paged_kv_indptr: torch.Tensor,
                paged_kv_indices: torch.Tensor,
                paged_kv_last_page_len: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                scale_q: Optional[torch.Tensor],
                scale_k: Optional[torch.Tensor],
                scale_v: Optional[torch.Tensor],
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                pass

            @register_fake_op(f"flashinfer::{uri}_ragged_run")
            def _fake_ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                pass

            @register_fake_op(f"flashinfer::{uri}_paged_run")
            def _fake_paged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                paged_k_cache: torch.Tensor,
                paged_v_cache: torch.Tensor,
                qo_indptr: torch.Tensor,
                paged_kv_indptr: torch.Tensor,
                paged_kv_indices: torch.Tensor,
                paged_kv_last_page_len: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                maybe_prefix_len_ptr: Optional[torch.Tensor],
                maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
                maybe_max_item_len_ptr: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                scale_q: Optional[torch.Tensor],
                scale_k: Optional[torch.Tensor],
                scale_v: Optional[torch.Tensor],
                rope_scale: float,
                rope_theta: float,
                token_pos_in_items_len: int,
            ) -> None:
                pass

            def plan(*args):
                return None

            # Register the module.
            #
            # Note that plan is not part of model logic. It should not be included in
            # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
            modules_dict[args] = SimpleNamespace(
                plan=plan,
                ragged_run=ragged_run,
                paged_run=paged_run,
            )

        return modules_dict[args]

    return backend_module
````

## File: quantization.py
````python
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

from functools import cache
from typing import Any, Tuple

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import register_custom_op, register_fake_op

_quantization_module = None


def gen_quantization_module() -> JitSpec:
    return gen_jit_spec(
        "quantization",
        [
            jit_env.FLASHINFER_CSRC_DIR / "quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_quantization_ops.cu",
        ],
    )


def get_quantization_module():
    global _quantization_module
    if _quantization_module is None:
        _quantization_module = gen_quantization_module().build_and_load()
    return _quantization_module


@cache
def get_module_attr(attr: str) -> Any:
    global _quantization_module
    if _quantization_module is None:
        get_quantization_module()
    return getattr(_quantization_module, attr).default


@register_custom_op("flashinfer::packbits", mutates_args=())
def _packbits(x: torch.Tensor, bitorder: str) -> torch.Tensor:
    device = x.device
    x = x.to(torch.bool)
    y = torch.empty((x.size(0) + 7) // 8, dtype=torch.uint8, device=device)
    get_module_attr("packbits")(x, bitorder, y)
    return y


@register_fake_op("flashinfer::packbits")
def _fake_packbits(x: torch.Tensor, bitorder: str) -> torch.Tensor:
    return torch.empty((x.size(0) + 7) // 8, dtype=torch.uint8, device=x.device)


def packbits(x: torch.Tensor, bitorder: str = "big") -> torch.Tensor:
    r"""Pack the elements of a binary-valued array into bits in a uint8 array.

    The semantics of this function is the same as `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_.

    Parameters
    ----------
    x: torch.Tensor
        The 1D binary-valued array to pack.
    bitorder: str
        The bit-order ("bit"/"little") of the output. Default is "big".

    Returns
    -------
    y: torch.Tensor
        An uint8 packed array, shape ``((x.size(0) + 7) / 8),)``.

    Examples
    --------

    >>> import torch
    >>> from flashinfer import packbits
    >>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool, device="cuda")
    >>> x_packed = packbits(x)
    >>> list(map(bin, x_packed.tolist()))
    ['0b10110011']

    See Also
    --------
    segment_packbits
    """
    return _packbits(x, bitorder)


def segment_packbits(
    x: torch.Tensor, indptr: torch.Tensor, bitorder: str = "big"
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Pack a batch of binary-valued segments into bits in a uint8 array.

    For each segment, the semantics of this function is the same as `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_.

    Parameters
    ----------
    x: torch.Tensor
        The 1D binary-valued array to pack, shape ``(indptr[-1],)``.
    indptr: torch.Tensor
        The index pointer of each segment in :attr:`x`, shape ``(batch_size + 1,)``.
        The i-th segment in :attr:`x` is ``x[indptr[i]:indptr[i+1]]``.
    bitorder: str
        The bit-order ("bit"/"little") of the output. Default is "big".

    Returns
    -------
    y: torch.Tensor
        An uint8 packed array, shape: ``(new_indptr[-1],)``.
        The ``y[new_indptr[i]:new_indptr[i+1]]`` contains the packed bits ``x[indptr[i]:indptr[i+1]]``.
    new_indptr: torch.Tensor
        The new index pointer of each packed segment in :attr:`y`, shape ``(batch_size + 1,)``.
        It's guaranteed that ``new_indptr[i+1] - new_indptr[i] == (indptr[i+1] - indptr[i] + 7) // 8``.

    Examples
    --------

    >>> import torch
    >>> from flashinfer import segment_packbits
    >>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=torch.bool, device="cuda")
    >>> x_packed, new_indptr = segment_packbits(x, torch.tensor([0, 4, 7, 11], device="cuda"), bitorder="big")
    >>> list(map(bin, x_packed.tolist()))
    ['0b10110000', '0b100000', '0b11010000']
    >>> new_indptr
    tensor([0, 1, 2, 3], device='cuda:0')

    Note
    ----
    ``torch.compile`` is not supported for this function because it's data dependent.

    See Also
    --------
    packbits
    """
    seglen = indptr[1:] - indptr[:-1]
    packed_len = (seglen + 7) // 8
    indptr_new = torch.zeros(len(indptr), dtype=indptr.dtype, device=indptr.device)
    indptr_new[1:] = torch.cumsum(packed_len, 0)
    output_nnzs = indptr_new[-1].item()

    device = x.device
    indptr = indptr.to(torch.int32)
    indptr_new = indptr_new.to(torch.int32)
    y = torch.empty(output_nnzs, dtype=torch.uint8, device=device)
    get_module_attr("segment_packbits")(x, indptr, indptr_new, bitorder, y)
    return y, indptr_new
````

## File: rope.py
````python
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

from functools import cache
from typing import Any, Optional, Tuple

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import register_custom_op, register_fake_op

_rope_module = None


def gen_rope_module() -> JitSpec:
    return gen_jit_spec(
        "rope",
        [
            jit_env.FLASHINFER_CSRC_DIR / "rope.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_rope_ops.cu",
        ],
    )


def get_rope_module():
    global _rope_module
    if _rope_module is None:
        _rope_module = gen_rope_module().build_and_load()
    return _rope_module


@cache
def get_module_attr(attr: str) -> Any:
    global _rope_module
    if _rope_module is None:
        get_rope_module()
    return getattr(_rope_module, attr).default


@register_custom_op("flashinfer::apply_rope", mutates_args=("q_rope", "k_rope"))
def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    indptr = indptr.int()
    offsets = offsets.int()
    get_module_attr("apply_rope")(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope")
def _fake_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


@register_custom_op("flashinfer::apply_llama31_rope", mutates_args=("q_rope", "k_rope"))
def _apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    indptr = indptr.int()
    offsets = offsets.int()
    get_module_attr("apply_llama31_rope")(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope")
def _fake_apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


@register_custom_op("flashinfer::apply_rope_pos_ids", mutates_args=("q_rope", "k_rope"))
def _apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pos_ids = pos_ids.int()
    get_module_attr("apply_rope_pos_ids")(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids")
def _fake_apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::apply_rope_pos_ids_cos_sin_cache", mutates_args=("q_rope", "k_rope")
)
def _apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    pos_ids = pos_ids.int()
    get_module_attr("apply_rope_pos_ids_cos_sin_cache")(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids_cos_sin_cache")
def _fake_apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::apply_llama31_rope_pos_ids", mutates_args=("q_rope", "k_rope")
)
def _apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pos_ids = pos_ids.int()
    get_module_attr("apply_llama31_rope_pos_ids")(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope_pos_ids")
def _fake_apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope(
        q, k, q, k, indptr, offsets, rotary_dim, interleave, rope_scale, rope_theta
    )


def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> None:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor) inplace.
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    See Also
    --------
    apply_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope_pos_ids(
        q, k, q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )


def apply_llama31_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor) inplace. cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> flashinfer.apply_llama31_rope_inplace(q, k, indptr, offsets)

    See Also
    --------
    apply_llama31_rope
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope(
        q,
        k,
        q,
        k,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> None:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor) inplace. cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    See Also
    --------
    apply_llama31_rope_pos_ids
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q,
        k,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )
    return q_rope, k_rope


def apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1,
    rope_theta: float = 1e4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).
    cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)`, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(batch_size + 1)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``1``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``1e4``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_rope_pos_ids(
        q, k, q_rope, k_rope, pos_ids, rotary_dim, interleave, rope_scale, rope_theta
    )
    return q_rope, k_rope


def apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor). cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    indptr : torch.Tensor
        Indptr tensor, shape: ``(batch_size + 1)``.
    offsets : torch.Tensor
        The relative position offsets of each query in the batch, shape: ``(batch_size)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> batch_size = 128
    >>> qkv_len = 1024
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> nnz = batch_size * qkv_len
    >>> qkv_packed = torch.randn(
    >>>    nnz,
    >>>    (num_qo_heads + 2 * num_kv_heads) * head_dim,
    >>>    dtype=torch.float16,
    >>>    device="cuda:0",
    >>> )
    >>> q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    >>> k = qkv_packed[
    ...    :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ... ].reshape(nnz, num_kv_heads, head_dim)
    >>> indptr = torch.tensor(
    ...    [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    >>> )
    >>> offsets = torch.full((batch_size,), 10, dtype=torch.int32, device="cuda:0")
    >>> q_rope, k_rope = flashinfer.apply_llama31_rope(q, k, indptr, offsets)
    >>> q_rope.shape
    torch.Size([131072, 32, 128])
    >>> k_rope.shape
    torch.Size([131072, 32, 128])

    See Also
    --------
    apply_llama31_rope_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


def apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
    old_context_len: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Apply Llama 3.1 style rotary embedding to a batch of queries/keys (stored as
    RaggedTensor). cos/sin values are computed on the fly inside the kernel.

    We use :attr:`indptr` to denote the start pointer of each segment in the batch, the i-th
    segment the query of the i-th segment is ``q[indptr[i]:indptr[i+1]]`` and the key of the
    i-th segment is ``k[indptr[i]:indptr[i+1]]``, the first element of :attr:`indptr` is always
    0 and the last element of :attr:`indptr` is the total number of queries/keys in the batch.
    Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details about the
    ragged tensor.

    Parameters
    ----------
    q : torch.Tensor
        Query ragged tensor, shape: ``(nnz, num_q_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    k : torch.Tensor
        Key ragged tensor, shape: ``(nnz, num_k_heads, head_dim)``, where ``nnz`` is the last
        element of ``indptr``.
    pos_ids : torch.Tensor
        Position indices, shape: ``(nnz)``.
    rotary_dim : Optional[int]
        The dimensions to apply RoPE, if ``None``, we apply RoPE to the entire head dimension,
        otherwise, we apply RoPE to the first ``rotary_dim`` dimensions, default: ``None``.
    interleave : bool
        Whether to use interleaved layout in the last dimension, default: ``False``.

        * If ``True``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

        * If ``False``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``
    rope_scale : float
        The scaling factor used in the rope embedding, default: ``8``.
    rope_theta : float
        The theta value used in the rope embedding, default: ``5e5``.
    low_freq_factor : float
        The low frequency factor used in Llama 3.1 RoPE, default: ``1``.
    high_freq_factor : float
        The high frequency factor used in Llama 3.1 RoPE, default: ``4``.
    old_context_len : int
        The old context length used in Llama 3.1 RoPE, default: ``8192``.

    Returns
    -------
    q_rope : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads, head_dim)``.
    k_rope : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads, head_dim)``.

    See Also
    --------
    apply_llama31_rope_pos_ids_inplace
    """
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    _apply_llama31_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        float(old_context_len),
    )
    return q_rope, k_rope


def apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.

    Returns
    -------
    query_out : torch.Tensor
        The rotated query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key_out : torch.Tensor
        The rotated key tensor, shape: ``(nnz, num_k_heads * head_size)``.

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query_out.view(query_out.shape[0], -1, head_size),
        k_rope=key_out.view(key_out.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )

    return query_out, key_out


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.
    The result is inplace applied to the input tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.
    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    # pass q_rope and k_rope as q and k to perform inplace operation
    _apply_rope_pos_ids_cos_sin_cache(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query.view(query.shape[0], -1, head_size),
        k_rope=key.view(key.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=(not is_neox),
    )
````

## File: sampling.py
````python
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

from types import SimpleNamespace
from typing import Optional, Union

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec
from .utils import register_custom_op, register_fake_op

_sampling_module = None


def gen_sampling_module() -> JitSpec:
    return gen_jit_spec(
        "sampling",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sampling.cu",
            jit_env.FLASHINFER_CSRC_DIR / "renorm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_sampling_ops.cu",
        ],
    )


def get_sampling_module():
    global _sampling_module
    if _sampling_module is None:
        module = gen_sampling_module().build_and_load()

        # torch library for sampling_from_logits
        @register_custom_op("flashinfer::sampling_from_logits", mutates_args=())
        def sampling_from_logits(
            logits: torch.Tensor,
            indices: Optional[torch.Tensor],
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = logits.device
            # TODO: support more data types in logits to avoid conversion
            # to float32
            logits = logits.float()
            batch_size = indices.size(0) if indices is not None else logits.size(0)
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.sampling_from_logits.default(
                logits,
                samples,
                indices,
                deterministic,
                generator,
            )
            return samples

        @register_fake_op("flashinfer::sampling_from_logits")
        def _fake_sampling_from_logits(
            logits: torch.Tensor,
            indices: Optional[torch.Tensor],
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            batch_size = indices.size(0) if indices is not None else logits.size(0)
            return torch.empty(batch_size, dtype=torch.int32, device=logits.device)

        # torch library for sampling_from_probs

        @register_custom_op("flashinfer::sampling_from_probs", mutates_args=())
        def sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.sampling_from_probs.default(
                probs,
                samples,
                indices,
                deterministic,
                generator,
            )
            return samples

        # torch library for sampling_from_probs

        @register_fake_op("flashinfer::sampling_from_probs")
        def _fake_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            return torch.empty(batch_size, dtype=torch.int32, device=probs.device)

        # torch library for top_p_sampling_from_probs

        @register_custom_op("flashinfer::top_p_sampling_from_probs", mutates_args=())
        def top_p_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            maybe_top_p_arr = (
                maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
            )
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.top_p_sampling_from_probs.default(
                probs,
                samples,
                indices,
                maybe_top_p_arr,
                top_p_val,
                deterministic,
                generator,
            )
            return samples

        @register_fake_op("flashinfer::top_p_sampling_from_probs")
        def _fake_top_p_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            sample = torch.empty(probs.size(0), dtype=torch.int32, device=probs.device)
            return sample

        # torch library for top_k_sampling_from_probs

        @register_custom_op("flashinfer::top_k_sampling_from_probs", mutates_args=())
        def top_k_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            maybe_top_k_arr = (
                maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
            )
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.top_k_sampling_from_probs.default(
                probs,
                samples,
                indices,
                maybe_top_k_arr,
                top_k_val,
                deterministic,
                generator,
            )
            return samples

        @register_fake_op("flashinfer::top_k_sampling_from_probs")
        def _fake_top_k_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            sample = torch.empty(batch_size, dtype=torch.int32, device=probs.device)
            return sample

        # torch library for min_p_sampling_from_probs

        @register_custom_op("flashinfer::min_p_sampling_from_probs", mutates_args=())
        def min_p_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_min_p_arr: Optional[torch.Tensor],
            min_p_val: float,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            maybe_min_p_arr = (
                maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
            )
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.min_p_sampling_from_probs.default(
                probs,
                samples,
                indices,
                maybe_min_p_arr,
                min_p_val,
                deterministic,
                generator,
            )
            return samples

        # torch library for top_k_top_p_sampling_from_probs

        @register_custom_op(
            "flashinfer::top_k_top_p_sampling_from_probs", mutates_args=()
        )
        def top_k_top_p_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            maybe_top_k_arr = (
                maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
            )
            maybe_top_p_arr = (
                maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
            )
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            samples = torch.empty(batch_size, dtype=torch.int32, device=device)
            module.top_k_top_p_sampling_from_probs.default(
                probs,
                samples,
                indices,
                maybe_top_k_arr,
                top_k_val,
                maybe_top_p_arr,
                top_p_val,
                deterministic,
                generator,
            )
            return samples

        @register_fake_op("flashinfer::top_k_top_p_sampling_from_probs")
        def _fake_top_k_top_p_sampling_from_probs(
            probs: torch.Tensor,
            indices: Optional[torch.Tensor],
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            batch_size = indices.size(0) if indices is not None else probs.size(0)
            sample = torch.empty(batch_size, dtype=torch.int32, device=probs.device)
            return sample

        # torch library for top_p_renorm_probs

        @register_custom_op("flashinfer::top_p_renorm_probs", mutates_args=())
        def top_p_renorm_probs(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            maybe_top_p_arr = (
                maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
            )
            renorm_probs = torch.empty_like(probs)
            module.top_p_renorm_probs.default(
                probs,
                renorm_probs,
                maybe_top_p_arr,
                top_p_val,
            )
            return renorm_probs

        @register_fake_op("flashinfer::top_p_renorm_probs")
        def _fake_top_p_renorm_probs(
            probs: torch.Tensor,
            maybe_top_p_arr: Optional[torch.Tensor],
            top_p_val: float,
        ) -> torch.Tensor:
            return torch.empty_like(probs)

        # torch library for top_k_renorm_probs

        @register_custom_op("flashinfer::top_k_renorm_probs", mutates_args=())
        def top_k_renorm_probs(
            probs: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            device = probs.device
            probs = probs.float()
            maybe_top_k_arr = (
                maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
            )
            renorm_probs = torch.empty_like(probs)
            module.top_k_renorm_probs.default(
                probs,
                renorm_probs,
                maybe_top_k_arr,
                top_k_val,
            )
            return renorm_probs

        @register_fake_op("flashinfer::top_k_renorm_probs")
        def _fake_top_k_renorm_probs(
            probs: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            return torch.empty_like(probs)

        # torch library for top_k_mask_logits

        @register_custom_op("flashinfer::top_k_mask_logits", mutates_args=())
        def top_k_mask_logits(
            logits: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            device = logits.device
            logits = logits.float()
            maybe_top_k_arr = (
                maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
            )
            mask_logits = torch.empty_like(logits)
            module.top_k_mask_logits.default(
                logits,
                mask_logits,
                maybe_top_k_arr,
                top_k_val,
            )
            return mask_logits

        @register_fake_op("flashinfer::top_k_mask_logits")
        def _fake_top_k_mask_logits(
            logits: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
        ) -> torch.Tensor:
            return torch.empty_like(logits)

        # torch library for chain_speculative_sampling

        @register_custom_op(
            "flashinfer::chain_speculative_sampling",
            mutates_args=(
                "output_accepted_token_num",
                "output_emitted_draft_token_num",
            ),
        )
        def chain_speculative_sampling(
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            target_probs: torch.Tensor,
            output_accepted_token_num: torch.Tensor,
            output_emitted_draft_token_num: torch.Tensor,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            device = draft_probs.device
            draft_probs = draft_probs.float()
            draft_token_ids = draft_token_ids.int()
            target_probs = target_probs.float()
            output_accepted_token_num = output_accepted_token_num.int()
            output_emitted_draft_token_num = output_emitted_draft_token_num.int()
            b, n = draft_token_ids.shape
            output_token_ids = torch.empty((b, n + 1), dtype=torch.int32, device=device)
            module.chain_speculative_sampling.default(
                draft_probs,
                draft_token_ids,
                target_probs,
                output_token_ids,
                output_accepted_token_num,
                output_emitted_draft_token_num,
                deterministic,
                generator,
            )
            return output_token_ids

        @register_fake_op("flashinfer::chain_speculative_sampling")
        def _fake_chain_speculative_sampling(
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            target_probs: torch.Tensor,
            output_accepted_token_num: torch.Tensor,
            output_emitted_draft_token_num: torch.Tensor,
            deterministic: bool,
            generator: Optional[torch.Generator],
        ) -> torch.Tensor:
            b, n = draft_token_ids.shape
            device = draft_token_ids.device
            return torch.empty((b, n + 1), dtype=torch.int32, device=device)

        # Register the module
        _sampling_module = SimpleNamespace(
            sampling_from_probs=sampling_from_probs,
            sampling_from_logits=sampling_from_logits,
            top_p_sampling_from_probs=top_p_sampling_from_probs,
            top_k_sampling_from_probs=top_k_sampling_from_probs,
            min_p_sampling_from_probs=min_p_sampling_from_probs,
            top_k_top_p_sampling_from_probs=top_k_top_p_sampling_from_probs,
            top_p_renorm_probs=top_p_renorm_probs,
            top_k_renorm_probs=top_k_renorm_probs,
            top_k_mask_logits=top_k_mask_logits,
            chain_speculative_sampling=chain_speculative_sampling,
        )

    return _sampling_module


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def sampling_from_logits(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for category sampling from logits. It's equivalent to sampling
    from :attr:`logits` after applying softmax.
    Parameters
    ----------
    logits: torch.Tensor
        Logits for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of logits. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in logits.
        For example, if indices[i] = j, then the i-th output will be sampled from logits[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of logits.
    deterministic: bool
        Since the sampling doesn't use cub's BlockScan, the sampling is deterministic. We keep this
        argument for compatibility with other sampling functions.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`logits`, default is ``False``.
    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape (batch_size,). It's equivalent to sampling from
        :attr:`logits` after applying softmax.
    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904],
            [0.6009, 0.2566, 0.7936, 0.9408, 0.1332],
            [0.9346, 0.5936, 0.8694, 0.5677, 0.7411],
            [0.4294, 0.8854, 0.5739, 0.2666, 0.6274]], device='cuda:0')
    >>> samples = flashinfer.sampling.sampling_from_logits(logits)
    >>> samples
    tensor([0, 1, 1, 1], device='cuda:0', dtype=torch.int32)
    """
    if check_nan:
        if torch.any(torch.isnan(logits)):
            raise ValueError("Input logits contains NaN.")
    return get_sampling_module().sampling_from_logits(
        logits, indices, deterministic, generator
    )


def sampling_from_probs(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for category sampling from probabilities.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape (batch_size,).

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.sampling_from_probs(norm_prob)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().sampling_from_probs(
        probs, indices, deterministic, generator
    )


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-p sampling (nucleus sampling) from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_p_sampling_from_probs(norm_prob, top_p)
    >>> samples
    tensor([1, 2, 0, 4], device='cuda:0', dtype=torch.int32)


    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_sampling_from_probs
    top_p_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_p_sampling_from_probs(
        probs, indices, *_to_tensor_scalar_tuple(top_p), deterministic, generator
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-k sampling from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 1
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_sampling_from_probs(norm_prob, top_k)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)


    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_k_sampling_from_probs(
        probs, indices, *_to_tensor_scalar_tuple(top_k), deterministic, generator
    )


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for `min_p sampling <https://arxiv.org/abs/2407.01082>`_ from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    min_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for min-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    <torch._C.Generator object at 0x7f8b3db06df0>
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> min_p = torch.full((batch_size,), 0.05).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.min_p_sampling_from_probs(norm_prob, min_p)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().min_p_sampling_from_probs(
        probs, indices, *_to_tensor_scalar_tuple(min_p), deterministic, generator
    )


def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-k and top-p sampling from pre-softmax logits,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    logits: torch.Tensor
        Pre-softmax logits for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of logits. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.5
    >>> top_k = 3
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32
    >>> probs = torch.softmax(logits, dim=-1)
    >>> probs
    tensor([[0.4788, 0.3085, 0.1716, 0.0085, 0.0327],
        [0.2358, 0.1787, 0.4307, 0.1145, 0.0404],
        [0.1358, 0.2831, 0.1764, 0.2318, 0.1729],
        [0.3613, 0.1122, 0.1029, 0.3415, 0.0821]], device='cuda:0')
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_mask_logits
    top_p_sampling_from_probs
    """
    if filter_apply_order == "top_k_first":
        masked_logits = top_k_mask_logits(logits, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(
            probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
        )
    elif filter_apply_order == "joint":
        probs = torch.softmax(logits, dim=-1)
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-k and top-p sampling from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = torch.full((batch_size,), 0.2).to(0)
    >>> top_k = torch.full((batch_size,), 2).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(norm_prob, top_k, top_p)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    top_p_renorm_probs
    top_k_mask_logits
    """
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-p thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-p threshold for for
        re-normalizing probabilities, should be in ``(0, 1)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We mask out the probabilities less than `threshold` where the cumulative sum
        of ``probs[probs >= threshold]`` is `top_p`, and renormalize the probabilities.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_p_renorm_probs(prob, top_p)
    >>> renormed_probs
    tensor([[0.0000, 0.4882, 0.0000, 0.5118, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.5181, 0.0000, 0.4819, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')

    Note
    ----
    This combination of ``top_p_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_p_sampling_from_probs``.

    See Also
    --------
    top_p_sampling_from_probs
    sampling_from_probs
    top_k_renorm_probs
    """
    return get_sampling_module().top_p_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_p)
    )


top_p_renorm_prob = top_p_renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-k thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for re-normalizing probabilities, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k probabilities, set the rest to zero, and renormalize the probabilities.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_k_renorm_probs(prob, top_k)
    >>> renormed_probs
    tensor([[0.3201, 0.3319, 0.0000, 0.3480, 0.0000],
            [0.2573, 0.0000, 0.3398, 0.4028, 0.0000],
            [0.3672, 0.0000, 0.3416, 0.0000, 0.2912],
            [0.0000, 0.4243, 0.2750, 0.0000, 0.3007]], device='cuda:0')

    Note
    ----
    This combination of ``top_k_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_k_sampling_from_probs``.

    See Also
    --------
    top_k_sampling_from_probs
    sampling_from_probs
    top_p_renorm_probs
    """
    return get_sampling_module().top_k_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_k)
    )


top_k_renorm_prob = top_k_renorm_probs


def top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    r"""Fused GPU kernel for masking logits by top-k thresholding.

    Parameters
    ----------
    logits: torch.Tensor
        Logits before softmax, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for masking logits, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k logits, set the rest to negative infinity.

    Returns
    -------
    masked_logits: torch.Tensor
        Masked logits, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> logits = torch.randn(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> masked_logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    >>> masked_logits
    tensor([[ 1.9269,  1.4873,  0.9007,    -inf,    -inf],
            [ 1.0783,  0.8008,  1.6806,    -inf,    -inf],
            [   -inf,  0.2415, -0.2316,  0.0418,    -inf],
            [ 0.8599, -0.3097,    -inf,  0.8034,    -inf]], device='cuda:0')

    Note
    ----
    The combination of ``top_k_mask_logits`` and ``softmax`` should be equivalent to ``top_k_renorm_probs``.

    See Also
    --------
    top_k_renorm_probs
    """
    return get_sampling_module().top_k_mask_logits(
        logits, *_to_tensor_scalar_tuple(top_k)
    )


def chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    target_probs,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Fused-GPU kernel for speculative sampling for sequence generation (proposed in
    paper `Accelerating Large Language Model Decoding with Speculative Sampling <https://arxiv.org/pdf/2302.01318>`_),
    where the draft model generates a sequence(chain) of tokens for each request.

    Parameters
    ----------
    draft_probs: torch.Tensor
        The probability over vocabulary generated by draft model.
        Shape: ``(batch_size, num_speculate_tokens, vocab_size)``
    draft_token_ids: torch.Tensor
        The draft model's generated token indices.
        Shape: ``(batch_size, num_speculate_tokens)``
    target_probs: torch.Tensor
        The probability over vocabulary generated by target model.
        Compared to input :attr:`draft_probs`, the target model's probability has an additional
        slot at the end because the target model will generate one more token than the draft model.
        Shape: ``(batch_size, num_speculate_tokens + 1, vocab_size)``
    maybe_output_accepted_token_num: Optional[torch.Tensor]
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
        If specified, the number of accepted token number will be added to this tensor inplace. Default is ``None``.
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor]
        The number of draft tokens that are finally emitted for each request. Does not include
        the bonus token. (Thus the total number of tokens sampled for a given request is
        output_emitted_draft_token_num + 1).
        Shape: ``(batch_size)``
        If specified, the number of emitted token number will be added to this tensor inplace. Default is ``None``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.

    Returns
    -------
    output_token_ids: torch.Tensor
        The output token indices verified by the target model, rejected samples are
        padded with ``-1``.
        Compared to input :attr:`draft_token_ids`, the output tensor has an additional
        token index at the end for the final token, if all previous tokens are accepted,
        another "bonus" token will be sampled from the target model's probability.
        Shape: (batch_size, num_speculate_tokens + 1)
    output_accepted_token_num: torch.Tensor
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
    output_emitted_draft_token_num: torch.Tensor
        The number of draft tokens that are finally emitted for each request. Does not include
        the bonus token. (Thus the total number of tokens sampled for a given request is
        output_emitted_draft_token_num + 1).
        Shape: ``(batch_size)``

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 1
    >>> num_speculate_tokens = 2
    >>> vocab_size = 4
    >>> draft_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1]]]).to(0)
    >>> # token 2 was sampled from draft model for the first token, and
    >>> # token 1 was sampled from draft model for the second token
    >>> draft_token_ids = torch.tensor([[2, 1]], dtype=torch.int32).to(0)
    >>> target_probs = torch.tensor([[[0.0, 0.1, 0.6, 0.3], [1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1]]]).to(0)
    >>> output_token_ids, output_accepted_token_num, output_emitted_draft_token_num =\
    ...     flashinfer.sampling.chain_speculative_sampling(
    ...         draft_probs, draft_token_ids, target_probs)
    >>> # the first token is accepted, the second token is rejected and sampled from the difference
    >>> # between the target model and the draft model, the third token is padded with -1
    >>> output_token_ids
    tensor([[ 2,  0, -1]], device='cuda:0', dtype=torch.int32)
    >>> output_accepted_token_num
    tensor([1], device='cuda:0')
    >>> output_emitted_draft_token_num
    tensor([1], device='cuda:0')
    """
    b = draft_probs.size(0)
    dev = draft_probs.device
    if maybe_output_accepted_token_num is None:
        output_accepted_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_accepted_token_num = maybe_output_accepted_token_num
    if maybe_output_emitted_draft_token_num is None:
        output_emitted_draft_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_emitted_draft_token_num = maybe_output_emitted_draft_token_num
    output_token_ids = get_sampling_module().chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        output_accepted_token_num,
        output_emitted_draft_token_num,
        deterministic,
        generator,
    )
    return output_token_ids, output_accepted_token_num, output_emitted_draft_token_num
````

## File: sparse.py
````python
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

import math
from typing import Optional, Tuple, Union

import torch

from .decode import get_batch_decode_module
from .page import block_sparse_indices_to_vector_sparse_offsets
from .prefill import _compute_page_mask_indptr, get_batch_prefill_module
from .quantization import segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    canonicalize_torch_dtype,
    determine_attention_backend,
    is_float8,
)


def convert_bsr_mask_layout(mask: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    r"""Convert mask from BSR data layout to flashinfer's flattened mask layout.

    Parameters
    ----------
    mask : torch.Tensor
        A boolean mask tensor with shape ``(nnz, R, C)``.
    indptr : torch.Tensor
        The indptr tensor in BSR format.

    Returns
    -------
    flattened_mask : torch.Tensor
        A flattenedd mask tensor with shape ``(nnz * R * C,)``.
    """
    nnz, R, C = mask.shape
    MB = len(indptr) - 1
    mask_flashinfer = torch.empty((nnz * R * C,), dtype=mask.dtype, device=mask.device)
    for i in range(MB):
        mask_flashinfer[indptr[i] * R * C : indptr[i + 1] * R * C] = (
            mask[indptr[i] : indptr[i + 1]].transpose(0, 1).reshape(-1)
        )
    return mask_flashinfer


class BlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    The definition of block sparse matrix can be found at
    `bsr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_
    in SciPy.

    This API supports any block size ``(R, C)``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 32
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
    >>> # sparse mask: [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    >>> M = 3
    >>> N = 3
    >>> indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")
    >>> indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")
    >>> bsr_wrapper.plan(
    ...     indptr,
    ...     indices,
    ...     M,
    ...     N,
    ...     1, # R(block_rows)=1
    ...     1, # C(block_columns)=1
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = bsr_wrapper.run(q, k, v)
    >>> # use dense implementation with attention mask for comparison
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")
    >>> o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_ref)
    True
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`BlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        if backend in ["fa3", "auto"]:
            # NOTE(Zihao): assume maximum accumulate kv length is 4M
            self._vector_sparse_indices_buffer = torch.empty(
                (4 * 1024 * 1024,), dtype=torch.int32, device=self.device
            )
            # NOTE(Zihao): assume maximum batch size is 32768
            self._vector_sparse_indptr_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._packed_mask_buf: Optional[torch.Tensor] = None
        self._mask_indptr_buf: Optional[torch.Tensor] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self._backend = backend

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        M: int,
        N: int,
        R: int,
        C: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mask: Optional[torch.Tensor] = None,
        packed_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Union[str, torch.dtype] = "float16",
        non_blocking: bool = True,
    ) -> None:
        r"""Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        indptr : torch.Tensor
            The block index pointer of the block-sparse matrix on row dimension, shape ``(MB + 1,)``,
            where ``MB`` is the number of blocks in the row dimension.
        indices: torch.Tensor
            The block indices of the block-sparse matrix on column dimension, shape ``(nnz,)``, where
            ``nnz`` is the number of non-zero blocks. The elements in ``indices`` array should be less then ``NB``:
            the number of blocks in the column dimension.
        M : int
            The number of rows of the block-sparse matrix, ``MB = ceil_div(M, R)``.
        N : int
            The number of columns of the block-sparse matrix, ``NB = N // C``, ``N`` should be divisible by ``C``.
        R : int
            The number of rows in each block.
        C : int
            The number of columns in each block.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor.
        head_dim : int
            The dimension of each head.
        mask : torch.Tensor, optional
            The mask tensor with shape ``(nnz, R, C,)``, where nnz is the number of non-zero blocks.
            If every block is full, then we don't need to provide the mask tensor.
        packed_mask : torch.Tensor, optional
            The 1D packed mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : str, optional
            The data type of the query tensor.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        o_data_type : str, optional
            The data type of the output tensor. Default is ``half``. As output dtype cannot
            be inferred by input dtype in quantization
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = canonicalize_torch_dtype(o_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        num_blocks_row = len(indptr) - 1
        qo_indptr_host = R * torch.arange(num_blocks_row + 1, dtype=torch.int32)
        qo_indptr_host[-1] = M
        qo_indptr = qo_indptr_host.to(indptr.device, non_blocking=non_blocking)
        if indices.max().item() * C > N:
            raise ValueError("indices out of bound")
        last_block_len = torch.full(
            (num_blocks_row,), C, dtype=torch.int32, device=indptr.device
        )

        if mask is not None or packed_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                indptr,  # paged_kv_indptr
                last_block_len,  # paged_kv_last_page_len
                C,  # page_size
            )
        if packed_mask is None and mask is not None:
            # first convert BSR mask to flashinfer layout
            mask = convert_bsr_mask_layout(mask, indptr)
            # create packed mask from mask
            packed_mask, mask_indptr = segment_packbits(
                mask.contiguous().view(-1), mask_indptr, bitorder="little"
            )

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking)
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, non_blocking=non_blocking
        )
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask.to(
                self.device, non_blocking=non_blocking
            )
            self._mask_indptr_buf = mask_indptr.to(
                self.device, non_blocking=non_blocking
            )
            mask_mode = MaskMode.CUSTOM.value
        else:
            self._packed_mask_buf = None
            self._mask_indptr_buf = None
            mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._mask_mode = mask_mode

        self.M = M
        self.N = N
        self.R = R
        self.C = C

        kv_indptr_host = indptr.to("cpu")

        # NOTE(Zihao): we haven't supported mask in cuda-core implementations but it should
        # be easy to add support for it if needed, leave it as a future work.
        # at this moment, when mask is provided, we use the tensor-core implementation
        if (
            R * (num_qo_heads // num_kv_heads) < 4
            and mask_mode != MaskMode.CUSTOM.value
            and q_data_type not in [torch.float8_e4m3fn, torch.float8_e5m2]
        ):
            # If the operation is not compute-bound, we use the cuda-core implementation
            self._use_tensor_cores = False
            self._cached_module = get_batch_decode_module(
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
            )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                kv_indptr_host,
                num_blocks_row,
                num_qo_heads,
                num_kv_heads,
                C,
                False,  # is_cuda_graph_enabled
                -1,  # window_left
                logits_soft_cap,  # logits_soft_cap
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )
        else:
            # if the operation is compute-bound, we use the tensor-core implementation
            self._use_tensor_cores = True

            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    mask_mode == MaskMode.CUSTOM.value,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(self._backend)(
                *get_module_args
            )

            kv_lens_arr_host = (kv_indptr_host[1:] - kv_indptr_host[:-1]) * self.C
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host,
            )

            if self._backend == "fa3":
                if self.C != 1:
                    vector_sparse_indptr_host = torch.cat(
                        [
                            torch.tensor([0], dtype=torch.int32),
                            torch.cumsum(kv_lens_arr_host, dim=0, dtype=torch.int32),
                        ],
                        dim=0,
                    )
                    self._vector_sparse_indptr_buffer[
                        : len(vector_sparse_indptr_host)
                    ].copy_(vector_sparse_indptr_host, non_blocking=non_blocking)
                    kv_indptr_host = vector_sparse_indptr_host

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_lens_arr_host,
                M,  # total_num_rows
                num_blocks_row,  # batch_size
                num_qo_heads,
                num_kv_heads,
                self.C,  # page_size
                False,  # is_cuda_graph_enabled,
                head_dim,
                head_dim,
                causal,
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v, scale_q, scale_k, scale_v)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(M, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(N, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(N, num_kv_heads, head_dim)``.
        scale_q : Optional[torch.Tensor]
            The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_k : Optional[torch.Tensor]
            The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_v : Optional[torch.Tensor]
            The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        k = k.reshape(-1, self.C, *k.shape[-2:])
        v = v.reshape(-1, self.C, *v.shape[-2:])

        stride_block = k.stride(0)
        stride_n = k.stride(1)

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            _check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        if is_float8(q):
            assert q.dtype == k.dtype == v.dtype
            assert q.shape[-1] == k.shape[-1] == v.shape[-1]
            assert self._backend == "fa3" and self._use_tensor_cores

            if scale_q is None:
                scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
            if scale_k is None:
                scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
            if scale_v is None:
                scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

        if self._use_tensor_cores:
            if self._backend == "fa3":
                sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                    self._paged_kv_indices_buf,
                    self._paged_kv_indptr_buf,
                    self._vector_sparse_indices_buffer,  # output
                    self._vector_sparse_indptr_buffer,
                    self._kv_lens_buffer,
                    stride_block // stride_n,
                    1,  # stride_n // stride_n
                    self.C,  # block_size
                )
                sparse_indptr = self._vector_sparse_indptr_buffer
            else:
                sparse_indices = self._paged_kv_indices_buf
                sparse_indptr = self._paged_kv_indptr_buf

            self._cached_module.paged_run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._qo_indptr,
                sparse_indptr,
                sparse_indices,
                self._paged_kv_last_page_len,
                out,
                lse,
                self._mask_mode,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                self._packed_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                None,  # maybe_prefix_len_ptr
                None,  # maybe_token_pos_in_items_ptr
                None,  # maybe_max_item_len_ptr
                logits_soft_cap,
                sm_scale,
                scale_q,
                scale_k,
                scale_v,
                rope_scale,
                rope_theta,
                0,  # token_pos_in_items_len
            )
        else:
            self._cached_module.run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            )

        return (out, lse) if return_lse else out

    def end_forward(self) -> None:
        r"""Warning: This method is deprecated and has no effect."""
        pass
````

## File: utils.py
````python
"""
Copyright (c) 2023 by FlashInfer team.

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

import math
import os
from enum import Enum
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.version
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

IS_BUILDING_DOCS = os.environ.get("FLASHINFER_BUILDING_DOCS") == "1"


class PosEncodingMode(Enum):
    NONE = 0
    ROPE_LLAMA = 1
    ALIBI = 2


class MaskMode(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1
    CUSTOM = 2
    MULTIITEMSCORING = 3


class TensorLayout(Enum):
    NHD = 0
    HND = 1


log2e = 1.44269504088896340736


def _expand_5d(x: torch.Tensor, kv_layout: str) -> torch.Tensor:
    if x.ndim not in [4, 5]:
        raise ValueError("x must be 4D or 5D")
    if x.ndim == 4:
        # page_size == 1
        if kv_layout == "NHD":
            # (num_pages, 2, num_heads, head_dim) -> (num_pages, 2, page_size=1, num_heads, head_dim)
            # expand to 5D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # (num_pages, 2, num_heads, head_dim) -> (num_pages, 2, num_heads, page_size=1, head_dim)
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def _expand_4d(x: torch.Tensor, kv_layout: str) -> torch.Tensor:
    if x.ndim not in [3, 4]:
        raise ValueError("x must be 3D or 4D")
    if x.ndim == 3:
        # page_size == 1
        if kv_layout == "NHD":
            # (num_pages, num_heads, head_dim) -> (num_pages, page_size=1, num_heads, head_dim)
            # expand to 4D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # (num_pages, num_heads, head_dim) -> (num_pages, num_heads, page_size=1, head_dim)
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def _check_pos_encoding_mode(pos_encoding_mode: str) -> None:
    if not hasattr(PosEncodingMode, pos_encoding_mode):
        raise KeyError("Invalid pos_encoding_mode {}".format(pos_encoding_mode))


def _check_kv_layout(kv_layout: str) -> None:
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalid kv_layout {}".format(kv_layout))


def is_float8(x: torch.Tensor) -> bool:
    return x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]


def get_indptr(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int64)
    ret = torch.zeros(x.shape[0] + 1, dtype=x.dtype, device=x.device)
    ret[1:] = x.cumsum(0)
    return ret


def _unpack_paged_kv_cache(
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(paged_kv_cache, tuple):
        paged_k_cache, paged_v_cache = paged_kv_cache
        return (
            _expand_4d(paged_k_cache, kv_layout),
            _expand_4d(paged_v_cache, kv_layout),
        )
    elif torch.is_tensor(paged_kv_cache):
        # NOTE(Zihao): split on the second dimension
        paged_kv_cache = _expand_5d(paged_kv_cache, kv_layout)
        paged_k_cache, paged_v_cache = paged_kv_cache.unbind(dim=1)
        return paged_k_cache, paged_v_cache
    else:
        raise KeyError(
            "Unrecognized paged_kv_cache type {}, expect a single tensor or a tuple of tensor.".format(
                type(paged_kv_cache)
            )
        )


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m.float()


_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


# find the least power of 2 that is greater than or equal to x
def _ceil_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _get_range_buf(seq_len: int, device: torch.device) -> torch.Tensor:
    seq_len_pow2 = _ceil_pow2(seq_len)
    key = (f"range_{seq_len_pow2}", device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.arange(seq_len_pow2, device=device, dtype=torch.int32)
        _cache_buf[key] = buf
    return buf[:seq_len]


def _get_cache_alibi_slopes_buf(
    num_qo_heads: int, device: torch.device
) -> torch.Tensor:
    key = (f"alibi_slopes_{num_qo_heads}", device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = get_alibi_slopes(num_qo_heads).to(device)
        _cache_buf[key] = buf
    return buf


def canonicalize_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(
            "dtype must be a string or torch.dtype, got {}".format(type(dtype))
        )


def get_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    return torch.cuda.get_device_capability(device.index)


def _check_cached_qkv_data_type(
    q: torch.Tensor, k: torch.Tensor, dtype_q: torch.dtype, dtype_kv: torch.dtype
) -> None:
    if q.dtype != dtype_q:
        raise ValueError(
            f"The dtype of q {q.dtype} does not match the q_data_type {dtype_q} specified in plan function."
        )
    if k.dtype != dtype_kv:
        raise ValueError(
            f"The dtype of k {k.dtype} does not match the kv_data_type {dtype_kv} specified in plan function."
        )


if IS_BUILDING_DOCS or TorchVersion(torch_version) < TorchVersion("2.4"):

    def register_custom_op(
        name: str,
        fn: Optional[Callable] = None,
        /,
        *,
        mutates_args: Union[str, Iterable[str]],
        device_types: Optional[Union[str, Sequence[str]]] = None,
        schema: Optional[str] = None,
    ) -> Callable:
        return lambda x: x

    def register_fake_op(
        name: str,
        fn: Optional[Callable] = None,
    ) -> Callable:
        return lambda x: x

else:

    def register_custom_op(
        name: str,
        fn: Optional[Callable] = None,
        /,
        *,
        mutates_args: Union[str, Iterable[str]],
        device_types: Optional[Union[str, Sequence[str]]] = None,
        schema: Optional[str] = None,
    ) -> Callable:
        # NOTE(Zihao): torch.library.custom_op has significant overhead as mentioned in the following link
        # https://github.com/vllm-project/vllm/blob/36e76700453924c8d421db99af70a88a1df835cd/vllm/utils.py#L1660-L1674

        # return torch.library.custom_op(
        #     name,
        #     fn,
        #     mutates_args=mutates_args,
        #     device_types=device_types,
        #     schema=schema,
        # )
        return lambda x: x

    def register_fake_op(
        name: str,
        fn: Optional[Callable] = None,
    ) -> Callable:
        # return torch.library.register_fake(name, fn)
        return lambda x: x


def determine_gemm_backend(device: torch.device) -> str:
    major, _ = get_compute_capability(device)
    if major == 9 and torch.version.cuda >= "12.3":
        return "sm90"
    else:
        return "sm80"


def is_fa3_backend_supported(
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> bool:
    """
    Check if the FA3 backend is supported based on the given parameters.
    NOTE(Zihao): this function is a workaround for the lack of support for certain features in
    our FA3 backend, and will be removed once the backend is fully supported.

    Parameters
    ----------
    pos_encoding_mode : int
        The positional encoding mode.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reductions are allowed.
    use_custom_mask : bool
        Whether a custom mask is used.
    dtype_q : torch.dtype
        The data type of the query tensor.
    dtype_kv : torch.dtype
        The data type of the key-value tensor.

    Returns
    -------
    bool
        True if the FA3 backend is supported, False otherwise.
    """
    if use_custom_mask:
        return False
    if pos_encoding_mode != PosEncodingMode.NONE.value:
        return False
    if use_fp16_qk_reductions:
        return False
    # NOTE: currently fp8 is not supported in our FA3 backend
    # will add support soon
    if dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return False
    if dtype_kv in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return False
    return True


def determine_attention_backend(
    device: torch.device,
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> str:
    """
    Determine the appropriate attention backend based on the device and parameters.

    Parameters
    ----------
    device : torch.device
        The device to be used.
    mask_mode : int
        The mask mode.
    pos_encoding_mode : int
        The positional encoding mode.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reductions are allowed.
    use_custom_mask : bool
        Whether a custom mask is used.
    dtype_q : torch.dtype
        The data type of the query tensor.
    dtype_kv : torch.dtype
        The data type of the key-value tensor.

    Returns
    -------
    str
        The name of the attention backend to be used.
    """
    if is_sm90a_supported(device) and is_fa3_backend_supported(
        pos_encoding_mode,
        use_fp16_qk_reductions,
        use_custom_mask,
        dtype_q,
        dtype_kv,
    ):
        return "fa3"
    else:
        return "fa2"


def is_sm90a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 9 and torch.version.cuda >= "12.3"


def is_sm100a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 10 and torch.version.cuda >= "12.8"


def determine_mla_backend(device: torch.device) -> str:
    return "fa3" if is_sm90a_supported(device) else "fa2"


def _check_shape_dtype_device(
    x: torch.Tensor,
    expected_shape: Sequence[int],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
    name: str,
) -> None:
    if x.shape != torch.Size(expected_shape):
        raise ValueError(
            f"Invalid shape of {name}: expected {expected_shape}, got {x.shape}"
        )
    if x.dtype != expected_dtype:
        raise ValueError(
            f"Invalid dtype of {name}: expected {expected_dtype}, got {x.dtype}"
        )
    if x.device != expected_device:
        raise ValueError(
            f"Invalid device of {name}: expected {expected_device}, got {x.device}"
        )
````
