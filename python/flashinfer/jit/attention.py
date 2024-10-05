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
from .env import FLASHINFER_GEN_SRC_DIR
from .utils import (
    write_if_different,
    dtype_map,
    filename_safe_dtype_map,
    pos_encoding_mode_literal,
    mask_mode_literal,
)
from .single_decode_templ import single_decode_templ
from .batch_decode_templ import batch_decode_templ
from .single_prefill_templ import single_prefill_templ
from .batch_prefill_templ import batch_prefill_templ


def get_single_decode_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_alibi: bool,
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
        use_alibi="true" if use_alibi else "false",
    )


def get_single_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_alibi: bool,
) -> str:
    return (
        f"single_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"use_alibi_{use_alibi}"
    )


def gen_single_decode_cu(*args) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    file_name = f"{get_single_decode_uri(*args)}.cu"
    write_if_different(
        gen_directory / file_name,
        get_single_decode_cu_str(*args),
    )


def get_batch_decode_cu_str(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_alibi: bool,
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
        use_alibi="true" if use_alibi else "false",
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
    use_alibi: bool,
) -> str:
    return (
        f"batch_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_{head_dim}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"use_alibi_{use_alibi}"
    )


def gen_batch_decode_cu(*args) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    file_name = f"{get_batch_decode_uri(*args)}.cu"
    write_if_different(
        gen_directory / file_name,
        get_batch_decode_cu_str(*args),
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
    use_alibi: bool,
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
        use_alibi="true" if use_alibi else "false",
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
    use_alibi: bool,
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
        f"use_alibi_{use_alibi}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def gen_single_prefill_cu(*args) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    file_name = f"{get_single_prefill_uri(*args)}.cu"
    write_if_different(
        gen_directory / file_name,
        get_single_prefill_cu_str(*args),
    )


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
    use_alibi: bool,
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
        use_alibi="true" if use_alibi else "false",
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
    use_alibi: bool,
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
        f"use_alibi_{use_alibi}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def gen_batch_prefill_cu(*args) -> None:
    gen_directory = FLASHINFER_GEN_SRC_DIR
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)
    file_name = f"{get_batch_prefill_uri(*args)}.cu"
    write_if_different(
        gen_directory / file_name,
        get_batch_prefill_cu_str(*args),
    )
