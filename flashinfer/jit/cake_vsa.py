"""
Copyright (c) 2026 by FlashInfer team.

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

# JIT loader for the Cake SM100/SM103 block-sparse attention (VSA) profiles.
#
# ``csrc/cake_vsa`` holds one device source per kernel profile
# (``cake_vsa_<profile>.cu``, compiled for both ``sm_100a`` and ``sm_103a``) and
# one shared tvm-ffi launcher (``cake_vsa_binding.cu``).  Each ``(profile, arch)``
# pair is one ``JitSpec`` named ``cake_vsa_<profile>_<arch>``: the launcher is
# compiled with the profile's route defines and the profile cubin is embedded
# through FlashInfer's ``embedded_cubin_factory`` path.

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from filelock import FileLock

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .cpp_ext import get_cuda_path

_GENERATED_DIR = "cake_vsa"
_BINDING_SOURCE = "cake_vsa_binding.cu"
_CUBIN_EMBED_NAME = "cake_vsa_kernel"
_CUBIN_OPTIONS = ("-cubin", "--std=c++17", "--use_fast_math")

ARCH_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
CAPABILITY_TO_ARCH: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}


@dataclass(frozen=True)
class Profile:
    """Launch record of one kernel profile (one device source)."""

    source: str
    kernel: str
    threads: int
    smem_bytes: int
    defines: tuple[str, ...]


_BLOCK_MASK_D128 = (
    "CAKE_VSA_ABI=1",
    "CAKE_VSA_Q_LAYOUT=0",
    "CAKE_VSA_Q_BOX_ROWS=64",
    "CAKE_VSA_Q_BOX_SPLIT=2",
    "CAKE_VSA_KV_LAYOUT=0",
    "CAKE_VSA_OUT_ROW_ELEMS=16384",
)

PROFILES: dict[str, Profile] = {
    "blk128_compact": Profile(
        source="cake_vsa_blk128_compact.cu",
        kernel="kernel_flashinfer_blackwell_vsa_blk128_seed_sm100",
        threads=512,
        smem_bytes=134784,
        defines=_BLOCK_MASK_D128,
    ),
    "blk128_fp16_compact": Profile(
        source="cake_vsa_blk128_fp16_compact.cu",
        kernel="kernel_flashinfer_blackwell_vsa_blk128_fp16_compact_sm100",
        threads=512,
        smem_bytes=134784,
        defines=(*_BLOCK_MASK_D128, "CAKE_VSA_FP16=1"),
    ),
    "longseq": Profile(
        source="cake_vsa_longseq.cu",
        kernel="kernel_flashinfer_blackwell_vsa_longseq_warp_specialized_sm100",
        threads=512,
        smem_bytes=201344,
        defines=(*_BLOCK_MASK_D128, "CAKE_VSA_SELECTED_BLOCKS=1"),
    ),
    "ultrasparse_bsr": Profile(
        source="cake_vsa_ultrasparse_bsr.cu",
        kernel="kernel_flashinfer_blackwell_vsa_ultrasparse_bsr_sm100",
        threads=512,
        smem_bytes=135424,
        defines=(
            "CAKE_VSA_ABI=1",
            "CAKE_VSA_Q_LAYOUT=0",
            "CAKE_VSA_Q_BOX_ROWS=128",
            "CAKE_VSA_Q_BOX_SPLIT=2",
            "CAKE_VSA_KV_LAYOUT=0",
            "CAKE_VSA_OUT_ROW_ELEMS=16384",
            "CAKE_VSA_SELECTED_BLOCKS=1",
            "CAKE_VSA_BSR_INDICES=1",
        ),
    ),
    "gqa_mask": Profile(
        source="cake_vsa_gqa_mask.cu",
        kernel="kernel_flashinfer_blackwell_vsa_gqa_mask_seed_sm100",
        threads=512,
        smem_bytes=134784,
        defines=(
            "CAKE_VSA_ABI=1",
            "CAKE_VSA_Q_LAYOUT=2",
            "CAKE_VSA_KV_LAYOUT=0",
            "CAKE_VSA_OUT_ROW_ELEMS=16384",
        ),
    ),
    "head64_native": Profile(
        source="cake_vsa_head64_native.cu",
        kernel="kernel_flashinfer_blackwell_vsa_head64_native_sm100",
        threads=384,
        smem_bytes=108928,
        defines=(
            "CAKE_VSA_ABI=1",
            "CAKE_VSA_Q_LAYOUT=0",
            "CAKE_VSA_Q_BOX_ROWS=64",
            "CAKE_VSA_Q_BOX_SPLIT=1",
            "CAKE_VSA_KV_LAYOUT=0",
            "CAKE_VSA_HEAD_DIM=64",
            "CAKE_VSA_OUT_ROW_ELEMS=8192",
        ),
    ),
    "head96_native": Profile(
        source="cake_vsa_head96_native.cu",
        kernel="kernel_flashinfer_blackwell_vsa_head96_native_sm100",
        threads=384,
        smem_bytes=182656,
        defines=(
            "CAKE_VSA_ABI=1",
            "CAKE_VSA_Q_LAYOUT=1",
            "CAKE_VSA_KV_LAYOUT=1",
            "CAKE_VSA_HEAD_DIM=96",
            "CAKE_VSA_OUT_ROW_ELEMS=12288",
        ),
    ),
    "blk64_persistent": Profile(
        source="cake_vsa_blk64_persistent.cu",
        kernel="kernel_flashinfer_vsa_blk64_persistent_m64_sm100",
        threads=512,
        smem_bytes=183296,
        defines=("CAKE_VSA_ABI=2", "CAKE_VSA_OUT_BOX_COLS=64"),
    ),
    "blk64_persistent_ws_m64n256": Profile(
        source="cake_vsa_blk64_persistent_ws_m64n256.cu",
        kernel="kernel_flashinfer_vsa_blk64_persistent_per_head_m64n256_ws_sm100",
        threads=512,
        smem_bytes=232448,
        defines=("CAKE_VSA_ABI=2", "CAKE_VSA_OUT_BOX_COLS=32"),
    ),
    # Regenerated with the device source by the Cake exporter; do not edit by hand.
    "blk64_balanced": Profile(
        source="cake_vsa_blk64_balanced.cu",
        kernel="kernel_flashinfer_vsa_blk64_balanced_m64n256_ws_sm100",
        threads=384,
        smem_bytes=232448,
        defines=("CAKE_VSA_ABI=4",),
    ),
    "fp16_direct": Profile(
        source="cake_vsa_fp16_direct.cu",
        kernel="kernel_minimax_sparse_prefill_union_sm100",
        threads=512,
        smem_bytes=201728,
        defines=("CAKE_VSA_ABI=3",),
    ),
}


def get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / _GENERATED_DIR
    if (installed / _BINDING_SOURCE).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / _GENERATED_DIR
    if (checkout / _BINDING_SOURCE).is_file():
        return checkout
    raise FileNotFoundError(
        f"Cake VSA sources were not found. Checked:\n  - {installed}\n  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _check_profile(profile: str) -> Profile:
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown Cake VSA profile {profile!r}; known: {sorted(PROFILES)}"
        ) from None


def _check_arch(arch: str) -> list[str]:
    try:
        return ARCH_FLAGS[arch]
    except KeyError:
        raise ValueError(
            f"unsupported Cake VSA arch {arch!r}; known: {sorted(ARCH_FLAGS)}"
        ) from None


def device_source(profile: str) -> Path:
    """Device source of ``profile`` (one file for every supported arch)."""
    return get_csrc_dir() / _check_profile(profile).source


def arch_for_capability(capability: tuple[int, int]) -> str:
    try:
        return CAPABILITY_TO_ARCH[capability]
    except KeyError:
        raise RuntimeError(
            "Cake VSA requires an SM100 or SM103 GPU, "
            f"got compute capability {capability[0]}.{capability[1]}"
        ) from None


@functools.cache
def _nvcc() -> Path:
    candidate = Path(get_cuda_path()) / "bin" / "nvcc"
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which("nvcc")
    if found is None:
        raise RuntimeError("nvcc is required to build the Cake VSA kernel cubins")
    return Path(found).resolve()


def prepare_cake_vsa_cubin(
    build_dir: Path, *, profile: str, arch: str
) -> Mapping[str, Path]:
    """Compile the profile cubin for ``arch`` into ``build_dir`` for Ninja embedding.

    The cubin is rebuilt when it is missing or older than its device source;
    the module's Ninja graph depends on the cubin path, so a rebuilt cubin is
    re-embedded on the next build.
    """
    _check_arch(arch)
    source = device_source(profile)
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / f"cake_vsa_{profile}_{arch}.cubin"
    with FileLock(build_dir / f"{cubin_path.name}.lock", thread_local=False):
        if (
            not cubin_path.is_file()
            or cubin_path.stat().st_mtime < source.stat().st_mtime
        ):
            temporary = build_dir / f".{cubin_path.name}.{os.getpid()}.tmp"
            command = [
                str(_nvcc()),
                *_CUBIN_OPTIONS,
                f"-arch={arch}",
                str(source),
                "-o",
                str(temporary),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Cake VSA cubin compilation failed for {profile}/{arch}:\n{result.stderr}"
                )
            os.replace(temporary, cubin_path)
    return {_CUBIN_EMBED_NAME: cubin_path}


@functools.cache
def gen_cake_vsa_module(profile: str, arch: str) -> JitSpec:
    """JIT spec of one ``(profile, arch)`` module, named ``cake_vsa_<profile>_<arch>``."""
    record = _check_profile(profile)
    arch_flags = _check_arch(arch)
    csrc_dir = get_csrc_dir()
    defines = [
        f"-DCAKE_VSA_KERNEL={record.kernel}",
        f"-DCAKE_VSA_THREADS={record.threads}",
        f"-DCAKE_VSA_SMEM_BYTES={record.smem_bytes}",
        *(f"-D{define}" for define in record.defines),
    ]
    spec = gen_jit_spec(
        name=f"cake_vsa_{profile}_{arch}",
        sources=[csrc_dir / _BINDING_SOURCE],
        extra_cuda_cflags=[*arch_flags, *defines],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[csrc_dir.parent, _get_include_dir()],
        embedded_cubin_factory=functools.partial(
            prepare_cake_vsa_cubin, profile=profile, arch=arch
        ),
    )
    logger.info("Generated Cake VSA JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_vsa_module(profile: str, arch: str) -> Any:
    """Build and load the ``(profile, arch)`` module; the returned module exposes ``run``."""
    return gen_cake_vsa_module(profile, arch).build_and_load()


__all__ = [
    "ARCH_FLAGS",
    "CAPABILITY_TO_ARCH",
    "PROFILES",
    "Profile",
    "arch_for_capability",
    "device_source",
    "gen_cake_vsa_module",
    "get_csrc_dir",
    "load_cake_vsa_module",
    "prepare_cake_vsa_cubin",
]
