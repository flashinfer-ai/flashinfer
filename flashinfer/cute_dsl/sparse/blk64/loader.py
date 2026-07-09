# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent


def _get_cutlass_root() -> Path:
    # Walk up from this file to find the flashinfer repo root, then 3rdparty/cutlass
    root = _THIS_DIR
    for _ in range(10):
        candidate = root / "3rdparty" / "cutlass"
        if candidate.is_dir():
            return candidate
        root = root.parent
    raise RuntimeError("Could not locate 3rdparty/cutlass from blk64 loader")


@lru_cache(maxsize=1)
def load_blk64_ext():
    """JIT-compile and load the blk64 BSA forward kernel extension."""
    cutlass_root = _get_cutlass_root()

    build_dir = (
        Path(
            os.environ.get(
                "FLASHINFER_WORKSPACE_BASE", Path.home() / ".cache" / "flashinfer"
            )
        )
        / "blk64_ext"
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    instantiation_sources = [
        str(_THIS_DIR / "instantiations" / f"flash_fwd_varblk{v}_bs{b}.cu")
        for v in [0, 1]
        for b in [0, 1]
    ]

    nvcc_flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--use_fast_math",
        "-Wno-deprecated-declarations",
        "--threads=4",
        "-DCUDA_CTA_RECONFIG_ACTIVATED=1",
        "-gencode=arch=compute_100a,code=sm_100a",
        f"-I{_THIS_DIR}",
        f"-I{cutlass_root / 'include'}",
        f"-I{cutlass_root / 'tools' / 'util' / 'include'}",
    ]

    ext = load(
        name="bsa_fwd_blk64_ext",
        sources=[
            str(_THIS_DIR / "bindings.cpp"),
            str(_THIS_DIR / "flash_fwd_launch_template.cu"),
        ]
        + instantiation_sources,
        extra_cflags=["-O3", "-std=c++20", "-Wno-deprecated-declarations"],
        extra_cuda_cflags=nvcc_flags,
        build_directory=str(build_dir),
        verbose=bool(int(os.environ.get("FLASHINFER_JIT_VERBOSE", "0"))),
        with_cuda=True,
    )
    return ext
