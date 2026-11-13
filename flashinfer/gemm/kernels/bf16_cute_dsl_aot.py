"""Prebuild the CuTe-DSL forms of the BF16 GEMM backends for a list of (N, K) shapes."""

from __future__ import annotations

import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Sequence

import torch

from ...jit import env as jit_env
from ...jit.cute_dsl_core import make_cute_dsl_spec
from . import dense_bf16_gemm_direct as direct
from . import dense_bf16_gemm_sm100_splitk as splitk
from . import dense_bf16_gemm_warp_splitk as warp
from . import tgv_gemm_cute_ext as tgv

_SPLITK_LEADING_DIMS = (2, 1, 1)
_TGV_LEADING = (2, 1, 1)


def enumerate_forms(
    shapes: Sequence[tuple[int, int]],
    m_buckets: Sequence[int],
    device_index: int = 0,
):
    dtype = torch.bfloat16
    forms = {}
    torch.cuda.set_device(device_index)
    torch.zeros(1, device="cuda")

    def add(source_file, form):
        module_name, name, compile_fn = form
        forms.setdefault((module_name, name), (compile_fn, source_file))

    for n, k in shapes:
        for m in m_buckets:
            if direct.is_direct_bf16_gemm_candidate_sm100(m, n):
                for tactic in direct.autotune_tactics(m, n, k):
                    add(
                        direct.__file__,
                        direct._direct_form(device_index, dtype, n, k, tactic, True),
                    )
            for tactic in warp.autotune_tactics(m, n, k):
                add(
                    warp.__file__,
                    warp._warp_form(device_index, dtype, n, k, tactic, True, False),
                )
            for tactic in splitk.autotune_tactics(m, n, k):
                add(
                    splitk.__file__,
                    splitk._splitk_form(
                        device_index, dtype, tactic, True, False, _SPLITK_LEADING_DIMS
                    ),
                )
    device = torch.device("cuda", device_index)
    for tactic in tgv.get_tgv_cute_ext_valid_tactics(device):
        cta_m, cta_n, num_ab_stage, use_2cta = tgv._resolve_tactic(tactic)
        arch = "sm_107" if tactic >= tgv._TGV_CUTE_EXT_FIRST_SM107_TACTIC else "sm_100"
        add(
            tgv.__file__,
            tgv._tgv_cute_ext_form(
                dtype,
                cta_m,
                cta_n,
                tgv._TGV_CUTE_EXT_CTA_K,
                num_ab_stage,
                use_2cta,
                True,
                False,
                *_TGV_LEADING,
                arch,
            ),
        )
    return forms


def _build_slice(job):
    jit_dir, shapes, m_buckets, index, workers, device_index, arch = job
    jit_env.FLASHINFER_JIT_DIR = Path(jit_dir)
    os.environ.pop("CUTE_DSL_ARCH", None)
    forms = sorted(enumerate_forms(shapes, m_buckets, device_index).items())
    if arch:
        os.environ["CUTE_DSL_ARCH"] = arch
    built = 0
    for i, ((module_name, name), (compile_fn, source_file)) in enumerate(forms):
        if i % workers != index:
            continue
        spec = make_cute_dsl_spec(
            module_name, name, compile_fn, (source_file,), enable_tvm_ffi=False
        )
        if not spec.is_compiled:
            spec.compile_and_persist()
            built += 1
            torch.cuda.empty_cache()
    return built


def build_bf16_cute_dsl_forms(
    jit_dir: Path,
    shapes: Sequence[tuple[int, int]],
    m_buckets: Sequence[int],
    workers: int,
    arch: str | None = None,
    device_index: int = 0,
) -> int:
    jit_dir = Path(jit_dir)
    jit_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(jit_dir),
            tuple(shapes),
            tuple(m_buckets),
            index,
            workers,
            device_index,
            arch,
        )
        for index in range(workers)
    ]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return sum(pool.map(_build_slice, jobs))


def copy_built_forms(jit_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for module_dir in sorted(Path(jit_dir).glob("*_cute_dsl")):
        destination = out_dir / module_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for artifact in module_dir.iterdir():
            if artifact.suffix == ".o" or artifact.name == "meta.json":
                shutil.copy2(artifact, destination / artifact.name)
        copied.append(module_dir.name)
    return copied
