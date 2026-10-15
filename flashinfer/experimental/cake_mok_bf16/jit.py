# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT-only loader for the standalone MoK CUDA sources.

``sources.json`` maps every kernel role to one or more source variants. A
variant lists the exact targets it was generated for: kernels whose generated
code does not depend on the target share one variant, while kernels with
target-specific schedules (for example the MXFP8 tile width, which follows
the tensor-memory capacity) carry one variant per target group. The variant
is selected by the device compute capability and compiled for that target.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import torch

from ...jit import env as jit_env
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

_PACKAGE = Path(__file__).resolve().parent
_TARGETS = {
    (10, 0): ("sm_100a", sm100a_nvcc_flags),
    (10, 3): ("sm_103a", sm103a_nvcc_flags),
    (10, 7): ("sm_107a", sm107a_nvcc_flags),
}


def _headers():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file():
        return installed
    checkout = _PACKAGE.parents[2]
    paths = [checkout / "csrc", checkout / "include"]
    if not (paths[0] / "tvm_ffi_utils.h").is_file():
        raise FileNotFoundError("FlashInfer JIT headers were not found")
    return paths


class _Kernel:
    def __init__(self, module, arguments):
        self._run = module.run
        self._arguments = tuple(name for _, name in arguments)

    def launch(self, *, grid, **arguments):
        arguments.update(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
        if set(arguments) != set(self._arguments):
            raise ValueError("Kernel arguments do not match the exported ABI")
        return self._run(*(arguments[name] for name in self._arguments))


def _target(capability):
    if capability not in _TARGETS:
        raise RuntimeError("MoK requires an SM100a, SM103a or SM107a device")
    return _TARGETS[capability]


def target_arch(device=None):
    """Exact generated target (``sm_100a``, ``sm_103a`` or ``sm_107a``) for a device."""
    return _target(torch.cuda.get_device_capability(device))[0]


@functools.cache
def _registry():
    return json.loads((_PACKAGE / "sources.json").read_text())


def kernel_spec(role, capability):
    """Return one content-addressed CUDA build specification."""
    arch, arch_flags = _target(capability)
    record = next(
        (v for v in _registry()[role]["variants"] if arch in v["arches"]), None
    )
    if record is None:
        raise RuntimeError(f"MoK kernel {role!r} has no generated {arch} variant")
    sources = [_PACKAGE / "csrc" / name for name in record["sources"]]
    for source, digest in zip(sources, record["sha256"], strict=True):
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"MoK source checksum mismatch: {source.name}")
    flags = [*arch_flags, *record["flags"]]
    identity = hashlib.sha256(
        json.dumps([role, record, flags], sort_keys=True).encode()
    ).hexdigest()
    spec = gen_jit_spec(
        f"cake_mok_{role}_{identity}",
        sources,
        extra_cuda_cflags=flags,
        extra_ldflags=["-lcuda"],
        extra_include_paths=_headers(),
        use_fast_math=False,
    )
    return spec, record["arguments"]


@functools.cache
def _load(role, device):
    with torch.cuda.device(device):
        spec, arguments = kernel_spec(role, torch.cuda.get_device_capability(device))
        return _Kernel(spec.build_and_load(), arguments)


def load_kernel(role):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare the MoK backend before CUDA Graph capture")
    return _load(role, torch.cuda.current_device())
