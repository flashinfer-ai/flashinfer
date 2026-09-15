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

from .cpp_ext import get_cuda_path


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


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepare_cubins(build_dir: Path, *, source_dir: Path, arches: tuple[str, ...]):
    include = (Path(get_cuda_path()) / "include").resolve()
    includes = [include]
    if (include / "cccl" / "cuda" / "std").is_dir():
        includes.append(include / "cccl")
    extra_opts = ["-std=c++17", *(f"-I{path}" for path in includes), "--use_fast_math"]
    cache = build_dir / "alphamoe_nvrtc_cubins"
    cache.mkdir(parents=True, exist_ok=True)
    embedded = {}
    records = []
    for arch in arches:
        for key, symbol in _ENTRIES.items():
            source_path = source_dir / f"{key}.cu"
            source = source_path.read_text()
            module_ident = f"alphamoe_{key}_{arch}"
            options = [f"--gpu-architecture={arch}", "-default-device", *extra_opts]
            inputs = {"source_sha256": _digest(source.encode()), "source_name": "kernel.cu",
                      "module_ident": module_ident, "module_key": key, "entry": key,
                      "kernel_symbol": symbol, "arch": arch, "compile_options": options}
            digest = _digest(json.dumps(inputs, sort_keys=True).encode())
            cubin = cache / f"{digest}.cubin"
            receipt = cache / f"{digest}.json"
            with FileLock(cache / f"{digest}.lock", thread_local=False):
                reusable = False
                if cubin.is_file() and receipt.is_file():
                    prior = json.loads(receipt.read_text())
                    reusable = prior["inputs"] == inputs and prior["sha256"] == _digest(cubin.read_bytes())
                if not reusable:
                    payload = nvrtc_compile(source, name="kernel.cu", arch=arch, extra_opts=extra_opts)
                    cubin.write_bytes(payload)
                    receipt.write_text(json.dumps({"inputs": inputs, "sha256": _digest(payload)}, indent=2) + "\n")
            embedded[module_ident] = cubin
            records.append({**inputs, "path": str(cubin), "sha256": _digest(cubin.read_bytes()),
                            "source_path": str(source_path), "receipt_path": str(receipt)})
    (build_dir / "alphamoe_nvrtc_receipt.json").write_text(
        json.dumps({"complete": True, "embedded_cubins": records}, indent=2) + "\n")
    return embedded


def get_alphamoe_nvrtc_spec(source_dir: Path, selected_archs):
    """Return the source closure key, target defines and existing JIT factory."""
    arches = tuple(f"sm_{major}{minor}" for major, minor in selected_archs)
    if not arches or any(arch not in {"sm_100a", "sm_103a"} for arch in arches):
        raise ValueError(f"Unsupported AlphaMoE router targets: {arches}")
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update((source_dir.parent / "alphamoe_fused_router.cu").read_bytes())
    for key in _ENTRIES:
        digest.update((source_dir / f"{key}.cu").read_bytes())
    flags = ["-DTVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API=1"]
    flags += [f"-DFLASHINFER_ALPHAMOE_{arch.upper()}=1" for arch in arches]
    factory = functools.partial(_prepare_cubins, source_dir=source_dir, arches=arches)
    return digest.hexdigest()[:20], flags, factory
