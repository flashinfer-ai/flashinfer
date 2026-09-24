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

import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from filelock import FileLock

import flashinfer
from flashinfer.jit import JitSpec
from flashinfer.jit.core import JitSpecNvcc
from flashinfer.jit.cpp_ext import run_ninja
from flashinfer.utils import (
    is_fa3_backend_supported,
    is_fa3_prefill_head_dim_supported,
    is_sm90a_supported,
)


def gen_decode_attention_modules(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
) -> list[JitSpec]:
    jit_specs: list[JitSpec] = []

    for (
        q_dtype,
        kv_dtype,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    ) in itertools.product(
        q_dtypes,
        kv_dtypes,
        head_dims,
        pos_encoding_modes,
        use_sliding_window_options,
        use_logits_soft_cap_options,
    ):
        if q_dtype != kv_dtype:
            if kv_dtype.itemsize > 1:
                continue  # skip fp16/bf16 mixed precision

        jit_specs.append(
            flashinfer.decode.gen_single_decode_module(
                q_dtype,
                kv_dtype,
                q_dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
            )
        )
        jit_specs.append(
            flashinfer.decode.gen_batch_decode_module(
                q_dtype,
                kv_dtype,
                q_dtype,
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
            )
        )

    return jit_specs


def gen_persistent_batch_attention_modules(
    q_dtypes,
    kv_dtypes,
    head_dims,
    use_logits_soft_cap_options,
) -> list[JitSpec]:
    jit_specs: list[JitSpec] = []

    for (
        q_dtype,
        kv_dtype,
        head_dim,
        use_logits_soft_cap,
    ) in itertools.product(
        q_dtypes,
        kv_dtypes,
        head_dims,
        use_logits_soft_cap_options,
    ):
        if q_dtype != kv_dtype:
            if kv_dtype.itemsize > 1:
                continue  # skip fp16/bf16 mixed precision

        jit_specs.append(
            flashinfer.attention.gen_batch_attention_module(
                q_dtype,
                kv_dtype,
                q_dtype,
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                0,  # pos_encoding_mode
                use_logits_soft_cap,
                False,  # use_profiler
            )
        )

    return jit_specs


def gen_prefill_attention_modules(
    q_dtypes,
    kv_dtypes,
    head_dims,
    pos_encoding_modes,
    use_sliding_window_options,
    use_logits_soft_cap_options,
    use_fp16_qk_reduction_options,
) -> list[JitSpec]:
    jit_specs: list[JitSpec] = []

    for (
        q_dtype,
        kv_dtype,
        head_dim,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    ) in itertools.product(
        q_dtypes,
        kv_dtypes,
        head_dims,
        pos_encoding_modes,
        use_sliding_window_options,
        use_logits_soft_cap_options,
        use_fp16_qk_reduction_options,
    ):
        if q_dtype != kv_dtype:
            if kv_dtype.itemsize > 1:
                continue  # skip fp16/bf16 mixed precision

        if (
            is_sm90a_supported(torch.device("cuda"))
            and is_fa3_backend_supported(
                pos_encoding_mode,
                use_fp16_qk_reduction,
                use_custom_mask=False,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
            )
            and is_fa3_prefill_head_dim_supported(head_dim, head_dim)
        ):
            if q_dtype != kv_dtype:
                continue  # fa3 template do not support mixed precision

            jit_specs.append(
                flashinfer.prefill.gen_single_prefill_module(
                    "fa3",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                )
            )

            jit_specs.append(
                flashinfer.prefill.gen_batch_prefill_module(
                    "fa3",
                    q_dtype,
                    kv_dtype,
                    q_dtype,
                    torch.int32,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    pos_encoding_mode,
                    use_sliding_window,
                    use_logits_soft_cap,
                    use_fp16_qk_reduction,
                )
            )
        jit_specs.append(
            flashinfer.prefill.gen_single_prefill_module(
                "fa2",
                q_dtype,
                kv_dtype,
                q_dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
                use_fp16_qk_reduction,
            )
        )
        jit_specs.append(
            flashinfer.prefill.gen_batch_prefill_module(
                "fa2",
                q_dtype,
                kv_dtype,
                q_dtype,
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                pos_encoding_mode,
                use_sliding_window,
                use_logits_soft_cap,
                use_fp16_qk_reduction,
            )
        )

    # required for attention with custom mask
    jit_specs.append(flashinfer.quantization.gen_quantization_module())

    jit_specs.append(flashinfer.page.gen_page_module())

    return jit_specs


def gen_fp4_quantization_module_for_device(device=None):
    """JitSpec of the FP4 quantization module the runtime selects on ``device``.

    Mirrors ``get_fp4_quantization_module``: the key is ``f"{major}{minor}"``, and
    SM12x prefers the ``120f`` family variant on CUDA >= 12.9. Returns ``None``
    when the architecture has no FP4 quantization module.
    """
    from flashinfer.quantization import fp4_quantization as fp4q
    from flashinfer.utils import get_compute_capability, version_at_least

    major, minor = get_compute_capability(device or torch.device("cuda:0"))
    key = f"{major}{minor}"
    if key in ("120", "121") and version_at_least(torch.version.cuda, "12.9"):
        key = "120f"
    gen = getattr(fp4q, f"gen_fp4_quantization_sm{key}_module", None)
    return gen() if gen is not None else None


def _prebuild_job_budget() -> int:
    """Compiler jobs a prebuild may run at once: FLASHINFER_JIT_PREBUILD_MAX_JOBS,
    else MAX_JOBS, else the CPU count.
    """
    for var in ("FLASHINFER_JIT_PREBUILD_MAX_JOBS", "MAX_JOBS"):
        val = os.environ.get(var)
        if val is not None and val.isdigit() and int(val) > 0:
            return int(val)
    return os.cpu_count() or 1


def prebuild_jit_specs(specs, verbose: bool = False) -> None:
    """Compile ``specs`` up front, one ninja per spec, within one job budget.

    Each ninja runs from ``spec.build_dir`` like ``JitSpecNvcc.build()`` does
    (per-module isolation, #2339), so the module's own ``.ninja_log`` and
    ``.ninja_deps`` record the outputs and the later ``build_and_load()`` is a
    no-op. ``flashinfer.jit.build_jit_specs`` runs one combined ninja from the
    ``cached_ops`` root, whose log the per-module ninja never sees, so it
    recompiles everything on first use.

    At most ``budget`` compiler jobs run at once (see ``_prebuild_job_budget``):
    ``workers`` ninjas, each with ``-j budget // workers``. AOT-cached specs are
    skipped; duplicates are built once.
    """
    todo = {}
    for spec in specs:
        if isinstance(spec, JitSpecNvcc) and not spec.aot_path.exists():
            todo.setdefault(spec.name, spec)
    if not todo:
        return
    budget = _prebuild_job_budget()
    workers = max(1, min(len(todo), budget))
    jobs = max(1, budget // workers)

    def build(spec):
        with FileLock(spec.lock_path, thread_local=False):
            spec.write_ninja()
            run_ninja(spec.build_dir, spec.ninja_path, verbose, max_jobs=jobs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(build, todo.values()))
