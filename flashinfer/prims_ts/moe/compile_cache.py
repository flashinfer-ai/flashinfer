# Copyright (c) 2026 by FlashInfer team.
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

"""
Compile cache helpers for Prims-TS batched GEMM launches.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flashinfer.jit.cute_dsl_core import _get_compile_arch
from flashinfer.prims_ts.utils import get_prims_ts_compile_options


def stable_config_hash(cfg: Any, **constexpr_args: Any) -> str:
    """Return a stable cache key for compile-time TS kernel arguments."""
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = dict(cfg.__dict__)
    else:
        payload = repr(cfg)
    return json.dumps(
        {
            "cfg": payload,
            "constexpr_args": constexpr_args,
        },
        sort_keys=True,
        default=str,
    )


@functools.cache
def get_compile_options() -> str:
    return get_prims_ts_compile_options()


_CompiledGemmKey = tuple[str, str, str, int, str]


_COMPILED_GEMM_CACHE: dict[_CompiledGemmKey, Any] = {}


def _compile_target_key(io: dict) -> tuple[int, str]:
    """Return the CUDA device and CuTe-DSL architecture for this compilation."""
    import torch

    device_index = None
    for value in io.get("_keepalive", ()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            device_index = value.device.index
            break
    if device_index is None:
        device_index = torch.cuda.current_device()

    return int(device_index), _get_compile_arch()


@functools.cache
def _batched_gemm_source_files() -> tuple[str, ...]:
    """Return every local source that can change PrimsTS BatchedGemm codegen."""
    source_root = Path(__file__).resolve().parents[1] / "batched_gemm"
    return tuple(str(path) for path in sorted(source_root.rglob("*.py")))


def _persistent_kernel_name(key: _CompiledGemmKey) -> str:
    """Return a symbol-safe persistent identity for one native-ABI GEMM."""
    digest = hashlib.sha256(repr(key).encode()).hexdigest()
    return f"{key[1]}_{digest}"


def get_compiled_gemm(cfg_hash: str, fc1_or_fc2: str, io: dict, stream: Any) -> Any:
    """Compile and cache a TS GEMM for the supplied launch IO.

    ``cfg_hash`` includes only the normalized config and other constexpr
    operands. Runtime problem shape and IO objects are passed to
    ``cute.compile`` to establish argument types, but the generated kernel body
    does not depend on their values and they must not be part of the cache key.
    """

    device_index, compile_arch = _compile_target_key(io)
    key = (
        cfg_hash,
        fc1_or_fc2,
        get_compile_options(),
        device_index,
        compile_arch,
    )
    if key in _COMPILED_GEMM_CACHE:
        return _COMPILED_GEMM_CACHE[key]
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _compile_for_launch
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    # PrimsTS launchers pass native CuTe device-pointer arguments and an explicit stream, so
    # persist and reload that ABI rather than converting the body to TVM-FFI. This moves CuTe
    # compilation out of every fresh autotuning process while preserving the launch contract.
    persistent_name = _persistent_kernel_name(key)
    compiled = build_and_load_cute_dsl_kernel(
        "prims_ts_moe_batched_gemm_native",
        persistent_name,
        lambda: _compile_for_launch(io, stream),
        extra_key_files=_batched_gemm_source_files(),
        enable_tvm_ffi=False,
        load_symbol=persistent_name,
        native_function_prefix=persistent_name,
    )
    _COMPILED_GEMM_CACHE[key] = compiled
    return compiled
