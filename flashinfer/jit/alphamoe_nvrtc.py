# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Build-time NVRTC packaging for the generated AlphaMoE router entries."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from filelock import FileLock
from tvm_ffi.cpp.nvrtc import nvrtc_compile

from . import env as jit_env
from .cpp_ext import get_cuda_path
from .utils import write_if_different


_ENTRIES = {
    "large": "kernel_alpha_moe_fused_router",
    "large_routed": "kernel_alpha_moe_fused_router",
    "medium": "kernel_alpha_moe_fused_router_medium",
    "medium_routed": "kernel_alpha_moe_fused_router_medium",
    "small": "kernel_alpha_moe_fused_router_small",
    "small_routed": "kernel_alpha_moe_fused_router_small",
    "tiny": "kernel_alpha_moe_fused_router_small",
    "tiny_routed": "kernel_alpha_moe_fused_router_small",
    "large_tail": "kernel_alpha_moe_fused_router_large_tail",
}
# The cake_ routed entries. The host source loads their cubins from the
# byte arrays of _CAKE_IMAGES instead of objects embedded like those of _ENTRIES:
# each embedded object is named after the one before it, and ten more would take
# the object file name past the 255-byte limit.
_CAKE_ENTRIES = {
    "cake_small_routed_e512_c4": "kernel_alpha_moe_fused_router_cluster_c4_w8_threshold",
    "cake_small_routed_e512_c16": "kernel_alpha_moe_fused_router_cluster_c16_w8_threshold",
    "cake_medium_routed_e512": "kernel_alpha_moe_fused_router_medium_e512_threshold",
    "cake_large_routed_e512": "kernel_alpha_moe_fused_router_e512_threshold",
    "cake_large_routed_e512_rb": "kernel_alpha_moe_fused_router_e512_threshold_rebase_batch",
    "cake_medium_routed_split_e256": "kernel_alpha_moe_fused_router_medium_e512_threshold_split_start_rs8",
    "cake_medium_routed_split_e288": "kernel_alpha_moe_fused_router_medium_e512_threshold_split_start_rs9",
    "cake_medium_routed_split_e384": "kernel_alpha_moe_fused_router_medium_e512_threshold_split_start_rs12",
    "cake_cluster_routed_c16_w16": "kernel_alpha_moe_fused_router_cluster_c16_w16_threshold_acq",
    "cake_large_routed_e512_ot": "kernel_alpha_moe_fused_router_e512_threshold_one_token",
    "cake_medium_routed_e512_bm4": "kernel_alpha_moe_fused_router_medium_e512_threshold_bm4",
    "cake_medium_routed_e512_bm5": "kernel_alpha_moe_fused_router_medium_e512_threshold_bm5",
    "cake_large_routed_e512_cps": "kernel_alpha_moe_fused_router_e512_threshold_cta_per_sm",
    "cake_large_routed_e512_rb_cps": "kernel_alpha_moe_fused_router_e512_threshold_rebase_batch_cta_per_sm",
    "cake_large_tail_e512": "kernel_alpha_moe_fused_router_large_tail_e512_idle_padding",
}
_CAKE_IMAGES = "cake_alphamoe_router_images.h"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepare_cubins(
    build_dir: Path, *, source_dir: Path, arches: tuple[str, ...], images_dir: Path
):
    include = (Path(get_cuda_path()) / "include").resolve()
    includes = [include]
    if (include / "cccl" / "cuda" / "std").is_dir():
        includes.append(include / "cccl")
    extra_opts = ["-std=c++17", *(f"-I{path}" for path in includes), "--use_fast_math"]
    cache = build_dir / "alphamoe_nvrtc_cubins"
    cache.mkdir(parents=True, exist_ok=True)
    embedded = {}
    images = []
    records = []
    for arch in arches:
        for key, symbol in {**_ENTRIES, **_CAKE_ENTRIES}.items():
            source_path = source_dir / f"{key}.cu"
            source = source_path.read_text()
            module_ident = f"alphamoe_{key}_{arch}"
            options = [f"--gpu-architecture={arch}", "-default-device", *extra_opts]
            inputs = {
                "source_sha256": _digest(source.encode()),
                "source_name": "kernel.cu",
                "module_ident": module_ident,
                "module_key": key,
                "entry": key,
                "kernel_symbol": symbol,
                "arch": arch,
                "compile_options": options,
            }
            digest = _digest(json.dumps(inputs, sort_keys=True).encode())
            cubin = cache / f"{digest}.cubin"
            receipt = cache / f"{digest}.json"
            with FileLock(cache / f"{digest}.lock", thread_local=False):
                reusable = False
                if cubin.is_file() and receipt.is_file():
                    prior = json.loads(receipt.read_text())
                    reusable = prior["inputs"] == inputs and prior["sha256"] == _digest(
                        cubin.read_bytes()
                    )
                if not reusable:
                    payload = nvrtc_compile(
                        source, name="kernel.cu", arch=arch, extra_opts=extra_opts
                    )
                    cubin.write_bytes(payload)
                    receipt.write_text(
                        json.dumps(
                            {"inputs": inputs, "sha256": _digest(payload)}, indent=2
                        )
                        + "\n"
                    )
            if key in _CAKE_ENTRIES:
                images.append((module_ident, cubin))
            else:
                embedded[module_ident] = cubin
            records.append(
                {
                    **inputs,
                    "path": str(cubin),
                    "sha256": _digest(cubin.read_bytes()),
                    "source_path": str(source_path),
                    "receipt_path": str(receipt),
                }
            )
    lines = [f"// Generated by flashinfer/jit/{Path(__file__).name}.", "#pragma once"]
    for module_ident, cubin in images:
        payload = cubin.read_bytes()
        lines.append(
            f"alignas(64) static const unsigned char {module_ident}_image[] = {{"
        )
        lines += [
            "    " + ", ".join(f"0x{byte:02x}" for byte in payload[i : i + 16]) + ","
            for i in range(0, len(payload), 16)
        ]
        lines.append("};")
    header = images_dir / _CAKE_IMAGES
    write_if_different(header, "\n".join(lines) + "\n")
    build_receipt = {
        "complete": True,
        "embedded_cubins": records,
        "image_header": {"path": str(header), "images": [m for m, _ in images]},
    }
    (build_dir / "alphamoe_nvrtc_receipt.json").write_text(
        json.dumps(build_receipt, indent=2) + "\n"
    )
    return embedded


def get_alphamoe_nvrtc_spec(source_dir: Path, selected_archs):
    """Return the source closure key, target defines and existing JIT factory."""
    arches = tuple(f"sm_{major}{minor}" for major, minor in selected_archs)
    if not arches or any(arch not in {"sm_100a", "sm_103a"} for arch in arches):
        raise ValueError(f"Unsupported AlphaMoE router targets: {arches}")
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update((source_dir.parent / "alphamoe_fused_router.cu").read_bytes())
    for key in {**_ENTRIES, **_CAKE_ENTRIES}:
        digest.update((source_dir / f"{key}.cu").read_bytes())
    closure_key = digest.hexdigest()[:20]
    # The factory writes _CAKE_IMAGES, which the host source includes, to images_dir.
    images_dir = (
        jit_env.FLASHINFER_GEN_SRC_DIR / f"alphamoe_fused_router_nvrtc_{closure_key}"
    ).resolve()
    # TVM-FFI uses the CUDA Runtime API by default on CUDA 12.8 and newer.
    flags = [f"-DFLASHINFER_ALPHAMOE_{arch.upper()}=1" for arch in arches]
    flags.append(f"-I{images_dir}")
    factory = functools.partial(
        _prepare_cubins, source_dir=source_dir, arches=arches, images_dir=images_dir
    )
    return closure_key, flags, factory
